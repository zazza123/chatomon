"""Build an atom into a native PydanticAI agent.

Install with `pip install 'chatomon[pydantic-ai]'`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from chatomon.connectors.base import Connector, ModelT, render_guidance
from chatomon.errors import ChatomonConfigError, MissingExtraError
from chatomon.schemas import InputDecision, Message

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.exceptions import SkipModelRequest
    from pydantic_ai.messages import (
        InstructionPart,
        ModelMessage,
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        UserPromptPart,
    )
    from pydantic_ai.models import (
        Model,
        ModelRequestContext,
        ModelRequestParameters,
        StreamedResponse,
    )
    from pydantic_ai.models.wrapper import WrapperModel
    from pydantic_ai.settings import ModelSettings
except ImportError as exc:
    if importlib.util.find_spec("pydantic_ai") is None:
        raise MissingExtraError("pydantic-ai", "pydantic-ai-slim") from exc
    raise

if TYPE_CHECKING:
    from chatomon.atom import Atom
    from chatomon.engine import Engine, TurnState

__all__ = ["PydanticAIConnector"]

_RESERVED = frozenset({"instructions", "system_prompt", "output_type", "name"})
_NO_FINAL_REPLY = "the run ended without a final text reply"
_STREAMING_REFUSED = (
    "PydanticAI streams model output before chatomon's output orbitals can check it; "
    "use run() or run_sync() when the atom has output orbitals"
)
_NOT_THE_NUCLEUS = (
    "a reply reached the run without passing chatomon's checks: a capability answered in "
    "place of the model, replaced the model or changed its reply; chatomon checks what the "
    "atom's model writes, so such a capability cannot be used with this atom"
)


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return " ".join(item for item in content if isinstance(item, str))
    return ""


def _has_tool_calls(response: ModelResponse) -> bool:
    return any(isinstance(part, ToolCallPart) for part in response.parts)


def _response_text(response: ModelResponse) -> str:
    return "".join(part.content for part in response.parts if isinstance(part, TextPart))


def _history(messages: Sequence[ModelMessage]) -> list[Message]:
    history: list[Message] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            history.extend(
                Message(role="user", content=_text(part.content))
                for part in message.parts
                if isinstance(part, UserPromptPart)
            )
        elif isinstance(message, ModelResponse) and not _has_tool_calls(message):
            text = _response_text(message)
            if text:
                history.append(Message(role="assistant", content=text))
    return history


@dataclass
class _Turn:
    turn: TurnState
    decision: InputDecision
    history_out: tuple[Message, ...]
    output_guidance: tuple[str, ...] = ()
    passed: str | None = None
    """The text of the reply the nucleus let through in the model request under way."""


class _Nucleus(WrapperModel):
    """The model of one request, with chatomon's guidance and output chain around it.

    The capability puts it in place of the model in every request. Every reply
    is checked here, where the model can be asked again: PydanticAI lets a
    capability call the model once for each request.
    """

    def __init__(self, wrapped: Model, engine: Engine, current: _Turn) -> None:
        super().__init__(wrapped)
        self._engine = engine
        self._current = current

    def _guided(self, parameters: ModelRequestParameters) -> ModelRequestParameters:
        guidance = (*self._current.decision.guidance, *self._current.output_guidance)
        if not guidance:
            return parameters
        parts = [
            *(parameters.instruction_parts or []),
            InstructionPart(content=render_guidance(guidance)),
        ]
        return replace(parameters, instruction_parts=parts)

    def _reopen(self) -> None:
        if self._current.turn.final_action is not None:
            # The framework asks again after chatomon accepted a reply (for example
            # an output validator raising ModelRetry): check the new reply too.
            self._engine.reopen_output(self._current.turn)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        current = self._current
        self._reopen()
        while True:
            response = await self.wrapped.request(
                messages, model_settings, self._guided(model_request_parameters)
            )
            text = _response_text(response)
            if _has_tool_calls(response) or not text:
                # Nothing to check: PydanticAI retries empty responses itself.
                return response
            outcome = await self._engine.run_output(
                current.turn, reply=text, history=current.history_out
            )
            if outcome.kind == "accept":
                current.passed = text
                return response
            if outcome.kind == "replace":
                current.passed = outcome.text or ""
                return ModelResponse(parts=[TextPart(content=current.passed)])
            current.output_guidance = outcome.guidance

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        if self._engine.output_orbitals:
            raise ChatomonConfigError(_STREAMING_REFUSED)
        current = self._current
        self._reopen()
        async with self.wrapped.request_stream(
            messages, model_settings, self._guided(model_request_parameters), run_context
        ) as stream:
            yield stream
        # No orbital checks a streamed reply: it is recorded as the turn's reply.
        response = stream.get()
        text = _response_text(response)
        if not _has_tool_calls(response) and text:
            await self._engine.run_output(current.turn, reply=text, history=current.history_out)
            current.passed = text


class _ChatomonCapability(AbstractCapability[Any]):
    """Runs the engine inside one PydanticAI run; `for_run` gives each run its own copy."""

    def __init__(self, engine: Engine, placeholder: str) -> None:
        self._engine = engine
        self._placeholder = placeholder
        self._turn: TurnState | None = None
        self._current: _Turn | None = None

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return _ChatomonCapability(self._engine, self._placeholder)

    async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> Any:
        try:
            result = await handler()
        except BaseException as exc:
            if self._turn is not None and not self._turn.finished:
                await self._engine.abort_turn(self._turn, exc)
            raise
        if self._turn is not None and not self._turn.finished:
            if self._turn.final_action is None:
                await self._engine.abort_turn(self._turn, RuntimeError(_NO_FINAL_REPLY))
            else:
                await self._engine.finish_turn(self._turn)
        return result

    async def _open(self, ctx: RunContext[Any], request_context: ModelRequestContext) -> _Turn:
        """Open the turn of the run and run the input chain on its message.

        Returns:
            The turn, with the user message redacted if the chain asked for it.
        """
        messages = request_context.messages
        last = messages[-1] if messages else None
        prompts = (
            [part for part in last.parts if isinstance(part, UserPromptPart)]
            if isinstance(last, ModelRequest)
            else []
        )
        message = "\n".join(_text(part.content) for part in prompts)
        history = _history(messages[:-1])
        # Kept before the chain runs: a run that fails while an orbital works is closed.
        self._turn = turn = self._engine.start_turn(conversation_id=ctx.conversation_id)
        decision = await self._engine.run_input(turn, message=message, history=history)
        user_text = message
        if decision.redact_user_message and isinstance(last, ModelRequest) and prompts:
            # Changed in place: the request and the run's history hold the same message,
            # and PydanticAI writes back to the history nothing a capability replaces.
            last.parts = [
                replace(part, content=self._placeholder)
                if isinstance(part, UserPromptPart)
                else part
                for part in last.parts
            ]
            user_text = self._placeholder
        self._current = _Turn(
            turn=turn,
            decision=decision,
            history_out=(*history, Message(role="user", content=user_text)),
        )
        return self._current

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        # Where PydanticAI runs this hook before the one that wraps the request, the turn
        # opens here: the capabilities that follow see the message already checked.
        if self._current is None:
            await self._open(ctx, request_context)
        return request_context

    def _check(self, current: _Turn, response: ModelResponse) -> None:
        """Fail closed on a final reply that the nucleus did not let through.

        Raises:
            ChatomonConfigError: If the text is not the one that passed the checks.
        """
        text = _response_text(response)
        if text and not _has_tool_calls(response) and text != current.passed:
            raise ChatomonConfigError(_NOT_THE_NUCLEUS)

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any
    ) -> ModelResponse:
        current = self._current or await self._open(ctx, request_context)
        if current.decision.kind == "reply":
            # The input chain wrote the reply: the model is not called.
            current.passed = current.decision.reply or ""
            return ModelResponse(parts=[TextPart(content=current.passed)])
        current.passed = None
        nucleus = _Nucleus(request_context.model, self._engine, current)
        try:
            response = await handler(replace(request_context, model=nucleus))
        except SkipModelRequest as skipped:
            self._check(current, skipped.response)
            raise
        self._check(current, response)
        return response

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        # The last of these hooks to run: where PydanticAI runs them after the hook that
        # wraps the request, a capability can still change the reply here.
        if self._current is not None:
            self._check(self._current, response)
        return response


class PydanticAIConnector(Connector["Agent[Any, str]"]):
    """Builds an atom into a native `pydantic_ai.Agent`.

    chatomon's capability is the outermost layer; your own `capabilities` run
    inside it. Extra keyword arguments go to `pydantic_ai.Agent` unchanged
    (`tools`, `deps_type`, `model_settings`, ...). With output orbitals, use
    `run()` / `run_sync()`: streaming is refused because PydanticAI streams
    text before it can be checked. The reply must come from the model: a run
    in which a capability answers in its place or changes its reply fails with
    `ChatomonConfigError`.

    Args:
        model: The nucleus model: a PydanticAI model object or a
            `"provider:model"` string.
        resolve_model: Maps the model id of a
            [JudgeOrbital][chatomon.orbitals.judge.JudgeOrbital] to a PydanticAI
            model; by default the id is used as a model string.
        **agent_kwargs: Passed to `pydantic_ai.Agent`.

    Raises:
        ChatomonConfigError: If `agent_kwargs` sets `instructions`,
            `system_prompt`, `output_type` or `name` (chatomon sets them from the atom).

    Examples:
        ```python
        from pydantic_ai.models.test import TestModel

        from chatomon import Atom
        from chatomon.connectors.pydantic_ai import PydanticAIConnector

        atom = Atom(
            name="speedypack",
            instructions="You are the customer assistant of SpeedyPack.",
            connector=PydanticAIConnector(TestModel(custom_output_text="Hi!")),
        )
        agent = atom.build()
        assert agent.run_sync("hello").output == "Hi!"
        ```
    """

    def __init__(
        self,
        model: Model | str,
        *,
        resolve_model: Callable[[str], Model | str] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        reserved = sorted(_RESERVED & set(agent_kwargs))
        if reserved:
            raise ChatomonConfigError(
                f"{', '.join(reserved)} is set by chatomon from the atom; "
                "remove it from PydanticAIConnector(...)"
            )
        self.model = model
        self.resolve_model = resolve_model
        self.agent_kwargs = agent_kwargs

    def build(self, atom: Atom[Agent[Any, str]], engine: Engine) -> Agent[Any, str]:
        """Build the native agent with chatomon's capability installed.

        Args:
            atom: The validated atom.
            engine: The engine that runs the atom's orbitals.

        Returns:
            A `pydantic_ai.Agent` with text output.
        """
        kwargs = dict(self.agent_kwargs)
        capabilities = [
            _ChatomonCapability(engine, atom.blocked_placeholder),
            *kwargs.pop("capabilities", ()),
        ]
        return Agent(
            self.model,
            name=atom.name,
            instructions=atom.system_prompt,
            capabilities=capabilities,
            **kwargs,
        )

    async def generate_structured(
        self, model: str, *, system: str, prompt: str, schema: type[ModelT]
    ) -> ModelT:
        """Make one structured call with a throwaway PydanticAI agent.

        Args:
            model: Model id; passed through `resolve_model` when set.
            system: System instructions.
            prompt: User prompt.
            schema: Output type.

        Returns:
            The validated output.
        """
        resolved = self.resolve_model(model) if self.resolve_model is not None else model
        agent = Agent(resolved, instructions=system, output_type=schema)
        result = await agent.run(prompt)
        return result.output
