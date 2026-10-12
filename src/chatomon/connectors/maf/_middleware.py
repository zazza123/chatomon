"""Run chatomon orbitals inside MAF agent and chat middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from agent_framework import (
    Agent,
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    ChatContext,
    ChatMiddleware,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    HistoryProvider,
    Message,
    ResponseStream,
)

from chatomon.connectors.base import render_guidance
from chatomon.connectors.maf._shared import (
    _NO_FINAL_REPLY,
    _STREAMING_REFUSED,
    _TURN_KEY,
    _reject_remote,
)
from chatomon.errors import ChatomonConfigError
from chatomon.schemas import InputDecision
from chatomon.schemas import Message as OrbitalMessage

if TYPE_CHECKING:
    from chatomon.engine import Engine, TurnState

_REMOTE_CONVERSATION = (
    "MAF returned a remote conversation_id despite store=False; "
    "local session history cannot be guaranteed"
)


def _history(messages: Sequence[Message]) -> list[OrbitalMessage]:
    history: list[OrbitalMessage] = []
    for message in messages:
        if message.role == "user":
            history.append(OrbitalMessage(role="user", content=message.text))
        elif (
            message.role == "assistant"
            and not any(content.type == "function_call" for content in message.contents)
            and message.text
        ):
            history.append(OrbitalMessage(role="assistant", content=message.text))
    return history


async def _stored_history(context: AgentContext) -> list[OrbitalMessage]:
    session = context.session
    if session is None:
        return []
    messages: list[Message] = []
    agent = cast("Agent[Any]", context.agent)
    for provider in agent.context_providers:
        if isinstance(provider, HistoryProvider) and provider.load_messages:
            state = session.state.get(provider.source_id, {})
            messages.extend(await provider.get_messages(session.session_id, state=state))
    return _history(messages)


def _with_guidance(options: Mapping[str, Any], guidance: Sequence[str]) -> dict[str, Any]:
    result = dict(options)
    if guidance:
        base = str(result.get("instructions") or "")
        section = render_guidance(guidance)
        result["instructions"] = f"{base}\n\n{section}" if base else section
    result["store"] = False
    return result


def _synthetic_response(text: str, source: ChatResponse | None = None) -> ChatResponse:
    return ChatResponse(
        messages=[Message("assistant", [text], message_id=uuid4().hex)],
        model=source.model if source is not None else None,
        usage_details=source.usage_details if source is not None else None,
        finish_reason="stop",
    )


def _synthetic_stream(text: str) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
    async def updates():
        yield ChatResponseUpdate(
            role="assistant", contents=[Content.from_text(text)], message_id=uuid4().hex
        )

    return ResponseStream(updates(), finalizer=ChatResponse.from_updates)


def _refuse_remote(update: ChatResponseUpdate) -> ChatResponseUpdate:
    if update.conversation_id is not None:
        raise ChatomonConfigError(_REMOTE_CONVERSATION)
    return update


def _with_message_id(message: Message) -> Message:
    if message.message_id is not None:
        return message
    identified = copy(message)
    identified.message_id = uuid4().hex
    return identified


def _identify_stream(
    stream: ResponseStream[AgentResponseUpdate, AgentResponse],
) -> ResponseStream[AgentResponseUpdate, AgentResponse]:
    role = "assistant"
    message_id: str | None = None
    provider_id: str | None = None
    chunks: list[AgentResponseUpdate] = []

    def identify(update: AgentResponseUpdate) -> AgentResponseUpdate:
        nonlocal role, message_id, provider_id
        new_message = (update.role and update.role != role) or (
            update.message_id and provider_id is not None and update.message_id != provider_id
        )
        if new_message:
            chunks.clear()
            message_id = None
            provider_id = None
        role = update.role or role
        if update.message_id:
            provider_id = update.message_id
            message_id = provider_id
            # A provider can name a message after its first chunk. Update the
            # recorded agent chunks too, so finalization keeps that message whole.
            for chunk in chunks:
                chunk.message_id = message_id
            chunks.clear()
        elif message_id is None:
            message_id = uuid4().hex
        update.message_id = message_id
        if provider_id is None:
            chunks.append(update)
        return update

    # Agent updates are what MAF finalizes and persists. Naming them here keeps
    # the provider's own stream assembly unchanged, including late message IDs.
    return stream.with_transform_hook(identify)


@dataclass
class _CurrentTurn:
    turn: TurnState
    decision: InputDecision
    history_out: tuple[OrbitalMessage, ...]
    output_guidance: tuple[str, ...] = ()


class _AuditedStream(ResponseStream[AgentResponseUpdate, AgentResponse]):
    """Close a turn when MAF finalizes a stream or its pull fails."""

    def __init__(
        self,
        inner: ResponseStream[AgentResponseUpdate, AgentResponse],
        engine: Engine,
        current: _CurrentTurn,
        session: AgentSession | None,
    ) -> None:
        self._inner = inner
        self._engine = engine
        self._current = current
        self._session = session
        super().__init__(inner, finalizer=lambda _updates: inner.get_final_response())

    async def __anext__(self) -> AgentResponseUpdate:
        try:
            update = await super().__anext__()
            self._check_local_session()
            return update
        except StopAsyncIteration:
            raise
        except BaseException as exc:
            if not self._current.turn.finished:
                await self._engine.abort_turn(self._current.turn, exc)
            raise

    def _check_local_session(self) -> None:
        if self._session is not None and self._session.service_session_id is not None:
            self._session.service_session_id = None
            raise ChatomonConfigError(
                "MAF returned a service_session_id despite store=False; "
                "local session history cannot be guaranteed"
            )

    async def get_final_response(self) -> AgentResponse:
        """Finalize the native stream and write one audit report.

        Returns:
            The native agent response.
        """
        try:
            response = await super().get_final_response()
            self._check_local_session()
            turn = self._current.turn
            if not turn.finished:
                if turn.final_action is None and response.text:
                    await self._engine.run_output(
                        turn, reply=response.text, history=self._current.history_out
                    )
                if turn.final_action is None:
                    await self._engine.abort_turn(turn, RuntimeError(_NO_FINAL_REPLY))
                else:
                    await self._engine.finish_turn(turn)
            return response
        except BaseException as exc:
            if not self._current.turn.finished:
                await self._engine.abort_turn(self._current.turn, exc)
            raise


class _ChatomonAgentMiddleware(AgentMiddleware):
    """Check input before MAF's context providers and downstream middleware run."""

    def __init__(self, engine: Engine, placeholder: str) -> None:
        self._engine = engine
        self._placeholder = placeholder

    async def process(
        self, context: AgentContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        _reject_remote(cast("Agent[Any]", context.agent).default_options)
        _reject_remote(context.options)
        _reject_remote(context.client_kwargs)
        if context.session is not None and context.session.service_session_id is not None:
            raise ChatomonConfigError(
                "MAF does not accept session.service_session_id; use agent.create_session()"
            )
        if context.stream and self._engine.output_orbitals:
            raise ChatomonConfigError(_STREAMING_REFUSED)

        index = next(
            (
                i
                for i in range(len(context.messages) - 1, -1, -1)
                if context.messages[i].role == "user"
            ),
            None,
        )
        if index is not None:
            context.messages[index] = _with_message_id(context.messages[index])
        message = context.messages[index].text if index is not None else ""
        history = [*await _stored_history(context), *_history(context.messages[:index])]
        turn = self._engine.start_turn(
            conversation_id=context.session.session_id if context.session is not None else None
        )
        try:
            decision = await self._engine.run_input(turn, message=message, history=history)
            user_text = message
            if decision.redact_user_message and index is not None:
                original = context.messages[index]
                context.messages[index] = Message(
                    "user", [self._placeholder], message_id=original.message_id
                )
                user_text = self._placeholder
            current = _CurrentTurn(
                turn=turn,
                decision=decision,
                history_out=(*history, OrbitalMessage(role="user", content=user_text)),
            )
            context.options = {**dict(context.options or {}), "store": False}
            context.client_kwargs[_TURN_KEY] = current
            await call_next()
            if context.stream:
                if not isinstance(context.result, ResponseStream):
                    raise RuntimeError("MAF did not return a response stream")
                context.result = _AuditedStream(
                    _identify_stream(context.result), self._engine, current, context.session
                )
                return
            if context.session is not None and context.session.service_session_id is not None:
                context.session.service_session_id = None
                raise ChatomonConfigError(
                    "MAF returned a service_session_id despite store=False; "
                    "local session history cannot be guaranteed"
                )
            if turn.final_action is None:
                await self._engine.abort_turn(turn, RuntimeError(_NO_FINAL_REPLY))
            else:
                await self._engine.finish_turn(turn)
        except BaseException as exc:
            if not turn.finished:
                await self._engine.abort_turn(turn, exc)
            raise


class _ChatomonChatMiddleware(ChatMiddleware):
    """Check final chat replies before MAF persists the local session history."""

    def __init__(self, engine: Engine, *, foundry_instructions: str | None = None) -> None:
        self._engine = engine
        self._foundry_instructions = foundry_instructions

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        current = context.kwargs.pop(_TURN_KEY, None)
        if not isinstance(current, _CurrentTurn):
            raise RuntimeError("chatomon's MAF turn state did not reach the chat middleware")
        _reject_remote(context.options)
        _reject_remote(context.kwargs)
        if current.decision.kind == "reply":
            reply = current.decision.reply or ""
            context.result = (
                _synthetic_stream(reply) if context.stream else _synthetic_response(reply)
            )
            return

        base_options = dict(context.options or {})
        base_messages = tuple(context.messages)
        if self._foundry_instructions is not None:
            runtime_instructions = base_options.pop("instructions", None)
            if runtime_instructions not in (None, self._foundry_instructions):
                raise ChatomonConfigError(
                    "FoundryAgent instructions are set by the atom; "
                    "do not override them in run options"
                )
        while True:
            guidance = (*current.decision.guidance, *current.output_guidance)
            if self._foundry_instructions is None:
                context.options = _with_guidance(base_options, guidance)
            else:
                instruction_text = self._foundry_instructions
                if guidance:
                    instruction_text += f"\n\n{render_guidance(guidance)}"
                context.options = _with_guidance(base_options, ())
                context.messages = [Message("developer", [instruction_text]), *base_messages]
            await call_next()
            if context.stream:
                # Refused here, before the agent reads the update: it writes the id it
                # finds on the session, and writes it again when it closes a failed stream.
                if isinstance(context.result, ResponseStream):
                    context.result = context.result.with_transform_hook(_refuse_remote)
                return
            response = context.result
            if not isinstance(response, ChatResponse):
                raise RuntimeError("MAF did not return a chat response")
            if response.conversation_id is not None:
                raise ChatomonConfigError(_REMOTE_CONVERSATION)
            response.messages = [_with_message_id(message) for message in response.messages]
            if any(
                content.type == "function_call"
                for message in response.messages
                for content in message.contents
            ):
                return
            if not response.text:
                return
            if current.turn.final_action is not None:
                self._engine.reopen_output(current.turn)
            outcome = await self._engine.run_output(
                current.turn, reply=response.text, history=current.history_out
            )
            if outcome.kind == "accept":
                return
            if outcome.kind == "replace":
                context.result = _synthetic_response(outcome.text or "", response)
                return
            current.output_guidance = outcome.guidance
