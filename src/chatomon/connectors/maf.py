"""Build an atom into a native Microsoft Agent Framework agent.

Install with ``pip install 'chatomon[maf]'`` or ``chatomon[maf-foundry]``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from chatomon.connectors.base import Connector, ModelT, render_guidance
from chatomon.errors import ChatomonConfigError, MissingExtraError
from chatomon.orbitals.judge import JudgeOrbital
from chatomon.schemas import InputDecision
from chatomon.schemas import Message as OrbitalMessage

try:
    from agent_framework import (
        Agent,
        AgentContext,
        AgentMiddleware,
        AgentResponse,
        AgentResponseUpdate,
        AgentSession,
        ChatContext,
        ChatMiddleware,
        ChatMiddlewareLayer,
        ChatResponse,
        ChatResponseUpdate,
        Content,
        HistoryProvider,
        InMemoryHistoryProvider,
        Message,
        ResponseStream,
    )
except ImportError as exc:
    if importlib.util.find_spec("agent_framework") is None:
        raise MissingExtraError("maf", "agent-framework-core") from exc
    raise

if TYPE_CHECKING:
    from agent_framework.foundry import FoundryAgent

    from chatomon.atom import Atom
    from chatomon.engine import Engine, TurnState

__all__ = ["MAFConnector", "MAFFoundryConnector"]

_RESERVED = frozenset({"client", "name", "instructions"})
_REMOTE_IDS = ("conversation_id", "previous_response_id", "conversation")
_TURN_KEY = "_chatomon_maf_turn"
_NO_FINAL_REPLY = "the MAF run ended without a final text reply"
_STREAMING_REFUSED = (
    "MAF streams model output before chatomon's output orbitals can check it; "
    "use run(..., stream=False) when the atom has output orbitals"
)


def _reject_remote(options: Mapping[str, Any] | None) -> None:
    if options is None:
        return
    store = options.get("store")
    if store is not None and store is not False:
        raise ChatomonConfigError("MAF requires store=False for local session history")
    for key in _REMOTE_IDS:
        if options.get(key) is not None:
            raise ChatomonConfigError(
                f"MAF does not accept {key}; use a local AgentSession instead"
            )
    extra_body = options.get("extra_body")
    if isinstance(extra_body, Mapping) and extra_body.get("agent_session_id") is not None:
        raise ChatomonConfigError(
            "MAF does not accept an explicit extra_body.agent_session_id; "
            "use a local AgentSession instead"
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
        messages=[Message("assistant", [text])],
        model=source.model if source is not None else None,
        usage_details=source.usage_details if source is not None else None,
        finish_reason="stop",
    )


def _synthetic_stream(text: str) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
    async def updates():
        yield ChatResponseUpdate(role="assistant", contents=[Content.from_text(text)])

    return ResponseStream(updates(), finalizer=ChatResponse.from_updates)


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
                    context.result, self._engine, current, context.session
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

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

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
        while True:
            context.options = _with_guidance(
                base_options, (*current.decision.guidance, *current.output_guidance)
            )
            await call_next()
            if context.stream:
                return
            response = context.result
            if not isinstance(response, ChatResponse):
                raise RuntimeError("MAF did not return a chat response")
            if response.conversation_id is not None:
                raise ChatomonConfigError(
                    "MAF returned a remote conversation_id despite store=False; "
                    "local session history cannot be guaranteed"
                )
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


def _prepare_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    reserved = sorted(_RESERVED & kwargs.keys())
    if reserved:
        raise ChatomonConfigError(
            f"{', '.join(reserved)} is set by chatomon from the atom; remove it from the connector"
        )
    if kwargs.get("require_per_service_call_history_persistence"):
        raise ChatomonConfigError(
            "require_per_service_call_history_persistence is incompatible with "
            "chatomon's per-turn output checks"
        )
    default_options = dict(kwargs.get("default_options") or {})
    if "instructions" in default_options:
        raise ChatomonConfigError("default_options.instructions is set by chatomon from the atom")
    _reject_remote(default_options)
    default_options["store"] = False
    result = dict(kwargs)
    result["default_options"] = default_options
    providers: list[Any] = list(result.pop("context_providers", ()) or ())
    if not any(
        isinstance(provider, HistoryProvider) and provider.load_messages for provider in providers
    ):
        providers.insert(0, InMemoryHistoryProvider())
    result["context_providers"] = providers
    return result


def _require_resolver(atom: Atom[Any], connector: Connector[Any], resolve_client: Any) -> None:
    if resolve_client is None and any(
        isinstance(orb, JudgeOrbital) and orb.connector is connector
        for orb in (*atom.input_orbitals, *atom.output_orbitals)
    ):
        raise ChatomonConfigError(
            "JudgeOrbital with MAF requires resolve_client(model_id) on the connector"
        )


async def _generate_structured(
    resolve_client: Callable[[str], Any] | None,
    model: str,
    *,
    system: str,
    prompt: str,
    schema: type[ModelT],
) -> ModelT:
    if resolve_client is None:
        raise ChatomonConfigError(
            "JudgeOrbital with MAF requires resolve_client(model_id) on the connector"
        )
    client = resolve_client(model)
    if not callable(getattr(client, "get_response", None)):
        raise ChatomonConfigError("resolve_client must return a MAF chat client")
    agent = Agent(client=client, instructions=system, default_options={"store": False})
    response = await agent.run(prompt, options={"response_format": schema, "store": False})
    value = response.value
    if value is None:
        raise ValueError("the MAF judge returned no structured response")
    return schema.model_validate(value)


class MAFConnector(Connector[Agent[Any]]):
    """Build an atom into a native `agent_framework.Agent`.

    The client can be any MAF chat client that implements `ChatMiddlewareLayer`,
    including `FoundryChatClient`. Provider clients such as `OpenAIChatClient`
    need their own MAF distribution in addition to `chatomon[maf]`. The
    connector places chatomon middleware
    outside agent middleware supplied by the caller. Local `AgentSession` history
    is used with `store=False`; pass a session from `agent.create_session()` to
    keep turns together. MAF's native run API is asynchronous.

    Args:
        client: The nucleus MAF chat client.
        resolve_client: Maps a `JudgeOrbital` model ID to another MAF chat client.
        **agent_kwargs: Forwarded to `agent_framework.Agent`.

    Raises:
        ChatomonConfigError: If an argument conflicts with atom identity, enables
            remote conversation continuity, or bypasses the chat middleware.

    Examples:
        ```{.python test="skip"}
        from agent_framework.openai import OpenAIChatClient
        from chatomon import Atom
        from chatomon.connectors.maf import MAFConnector

        atom = Atom(name="assistant", instructions="Be concise.",
                    connector=MAFConnector(OpenAIChatClient(model="gpt-4.1-mini")))
        agent = atom.build()
        session = agent.create_session()
        reply = await agent.run("Hello", session=session)
        ```
    """

    def __init__(
        self,
        client: Any,
        *,
        resolve_client: Callable[[str], Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self.client = client
        self.resolve_client = resolve_client
        self.agent_kwargs = _prepare_kwargs(agent_kwargs)

    def build(self, atom: Atom[Agent[Any]], engine: Engine) -> Agent[Any]:
        """Build the native MAF agent with chatomon's middleware installed.

        Args:
            atom: The validated atom.
            engine: The engine that runs the atom's orbitals.

        Returns:
            A native MAF agent.

        Raises:
            ChatomonConfigError: If the client cannot run chat middleware or
                preconfigured client middleware could rewrite checked output.
        """
        _require_resolver(atom, self, self.resolve_client)
        if not isinstance(self.client, ChatMiddlewareLayer):
            raise ChatomonConfigError(
                "MAFConnector requires a client with ChatMiddlewareLayer so chatomon "
                "can check replies before local history is saved"
            )
        if engine.output_orbitals and self.client.chat_middleware:
            raise ChatomonConfigError(
                "preconfigured client chat middleware can rewrite checked output; "
                "move it to MAFConnector(middleware=...)"
            )
        kwargs = dict(self.agent_kwargs)
        middleware = [
            _ChatomonAgentMiddleware(engine, atom.blocked_placeholder),
            _ChatomonChatMiddleware(engine),
            *(kwargs.pop("middleware", ()) or ()),
        ]
        return Agent(
            client=cast(Any, self.client),
            name=atom.name,
            instructions=atom.system_prompt,
            middleware=middleware,
            **kwargs,
        )

    async def generate_structured(
        self, model: str, *, system: str, prompt: str, schema: type[ModelT]
    ) -> ModelT:
        """Run a structured judge call with a resolved MAF chat client.

        Args:
            model: The judge model identifier.
            system: Judge instructions.
            prompt: Judge input.
            schema: Required Pydantic result type.

        Returns:
            The validated judge result.

        Raises:
            ChatomonConfigError: If `resolve_client` is missing or invalid.
            ValueError: If the judge returned no structured answer.
        """  # noqa: DOC502 - delegated to _generate_structured
        return await _generate_structured(
            self.resolve_client, model, system=system, prompt=prompt, schema=schema
        )


class MAFFoundryConnector(Connector["FoundryAgent"]):
    """Build an atom into a native `agent_framework.foundry.FoundryAgent`.

    This connector attaches to an agent already configured in Foundry. Input
    orbitals are supported. Output orbitals are blocked until a real Foundry
    test proves rejected drafts are not reused in later turns.

    Args:
        resolve_client: Maps a `JudgeOrbital` model ID to a MAF chat client.
        **foundry_kwargs: Forwarded to `FoundryAgent`; include the project
            endpoint, agent name and credential or project client.

    Raises:
        ChatomonConfigError: If an argument conflicts with atom identity or
            enables remote conversation continuity.
        MissingExtraError: If `chatomon[maf-foundry]` is not installed.

    Examples:
        ```{.python test="skip"}
        from azure.identity import DefaultAzureCredential
        from chatomon import Atom
        from chatomon.connectors.maf import MAFFoundryConnector

        atom = Atom(name="assistant", instructions="Be concise.",
                    connector=MAFFoundryConnector(
                        project_endpoint="https://example.services.ai.azure.com",
                        agent_name="my-agent", credential=DefaultAzureCredential()))
        agent = atom.build()
        reply = await agent.run("Hello", session=agent.create_session())
        ```
    """

    def __init__(
        self,
        *,
        resolve_client: Callable[[str], Any] | None = None,
        **foundry_kwargs: Any,
    ) -> None:
        self.resolve_client = resolve_client
        self.foundry_kwargs = _prepare_kwargs(foundry_kwargs)

    def build(self, atom: Atom[FoundryAgent], engine: Engine) -> FoundryAgent:
        """Build a native Foundry agent with input checks and local history.

        Args:
            atom: The validated atom.
            engine: The engine that runs the atom's orbitals.

        Returns:
            A native Foundry agent.

        Raises:
            ChatomonConfigError: If output orbitals or a raw client are supplied.
            MissingExtraError: If the Foundry extra is not installed.
        """
        if engine.output_orbitals:
            raise ChatomonConfigError(
                "FoundryAgent output orbitals need a real-service test proving rejected "
                "drafts are not reused in later turns; use MAFConnector(FoundryChatClient(...)) "
                "for checked output"
            )
        _require_resolver(atom, self, self.resolve_client)
        try:
            from agent_framework.foundry import FoundryAgent as NativeFoundryAgent
        except ImportError as exc:
            raise MissingExtraError("maf-foundry", "agent-framework-foundry") from exc
        kwargs = dict(self.foundry_kwargs)
        middleware = [
            _ChatomonAgentMiddleware(engine, atom.blocked_placeholder),
            _ChatomonChatMiddleware(engine),
            *(kwargs.pop("middleware", ()) or ()),
        ]
        agent = NativeFoundryAgent(
            name=atom.name,
            instructions=atom.system_prompt,
            middleware=middleware,
            **kwargs,
        )
        if not isinstance(agent.client, ChatMiddlewareLayer):
            raise ChatomonConfigError(
                "MAFFoundryConnector requires a Foundry client with ChatMiddlewareLayer"
            )
        return agent

    async def generate_structured(
        self, model: str, *, system: str, prompt: str, schema: type[ModelT]
    ) -> ModelT:
        """Run a structured judge call with a resolved MAF chat client.

        Args:
            model: The judge model identifier.
            system: Judge instructions.
            prompt: Judge input.
            schema: Required Pydantic result type.

        Returns:
            The validated judge result.

        Raises:
            ChatomonConfigError: If `resolve_client` is missing or invalid.
            ValueError: If the judge returned no structured answer.
        """  # noqa: DOC502 - delegated to _generate_structured
        return await _generate_structured(
            self.resolve_client, model, system=system, prompt=prompt, schema=schema
        )
