"""Build an atom into a native PydanticAI agent.

Install with `pip install 'chatomon[pydantic-ai]'`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from chatomon.connectors.base import Connector, ModelT, render_guidance
from chatomon.errors import ChatomonConfigError, MissingExtraError
from chatomon.schemas import InputDecision, Message

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.messages import (
        InstructionPart,
        ModelMessage,
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        UserPromptPart,
    )
    from pydantic_ai.models import Model, ModelRequestContext
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


def _with_guidance(
    request_context: ModelRequestContext, guidance: Sequence[str]
) -> ModelRequestContext:
    if not guidance:
        return request_context
    params = request_context.model_request_parameters
    parts = [*(params.instruction_parts or []), InstructionPart(content=render_guidance(guidance))]
    return replace(
        request_context, model_request_parameters=replace(params, instruction_parts=parts)
    )


class _ChatomonCapability(AbstractCapability[Any]):
    """Runs the engine inside one PydanticAI run; `for_run` gives each run its own copy."""

    def __init__(self, engine: Engine, placeholder: str) -> None:
        self._engine = engine
        self._placeholder = placeholder
        self._turn: TurnState | None = None
        self._decision: InputDecision | None = None
        self._history_out: tuple[Message, ...] = ()
        self._output_guidance: tuple[str, ...] = ()

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

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if self._turn is not None:
            return request_context
        messages = request_context.messages
        last = messages[-1] if messages else None
        prompts = (
            [part for part in last.parts if isinstance(part, UserPromptPart)]
            if isinstance(last, ModelRequest)
            else []
        )
        message = "\n".join(_text(part.content) for part in prompts)
        history = _history(messages[:-1])
        self._turn = self._engine.start_turn(conversation_id=ctx.conversation_id)
        self._decision = await self._engine.run_input(self._turn, message=message, history=history)
        user_text = message
        if self._decision.redact_user_message and isinstance(last, ModelRequest) and prompts:
            messages[-1] = replace(
                last,
                parts=[
                    replace(part, content=self._placeholder)
                    if isinstance(part, UserPromptPart)
                    else part
                    for part in last.parts
                ],
            )
            user_text = self._placeholder
        self._history_out = (*history, Message(role="user", content=user_text))
        return request_context

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any
    ) -> ModelResponse:
        turn, decision = self._turn, self._decision
        if turn is None or decision is None:
            return await handler(request_context)
        if decision.kind == "reply":
            return ModelResponse(parts=[TextPart(content=decision.reply or "")])
        if request_context.streaming and self._engine.output_orbitals:
            raise ChatomonConfigError(_STREAMING_REFUSED)
        if turn.final_action is not None:
            # The framework asks again after chatomon accepted a reply (for example
            # an output validator raising ModelRetry): check the new reply too.
            self._engine.reopen_output(turn)
        response = await handler(
            _with_guidance(request_context, (*decision.guidance, *self._output_guidance))
        )
        while not _has_tool_calls(response):
            if not _response_text(response):
                # Nothing to check: PydanticAI retries empty responses itself.
                return response
            outcome = await self._engine.run_output(
                turn, reply=_response_text(response), history=self._history_out
            )
            if outcome.kind == "accept":
                return response
            if outcome.kind == "replace":
                return ModelResponse(parts=[TextPart(content=outcome.text or "")])
            self._output_guidance = outcome.guidance
            response = await handler(
                _with_guidance(request_context, (*decision.guidance, *self._output_guidance))
            )
        return response


class PydanticAIConnector(Connector["Agent[Any, str]"]):
    """Builds an atom into a native `pydantic_ai.Agent`.

    chatomon's capability is the outermost layer; your own `capabilities` run
    inside it. Extra keyword arguments go to `pydantic_ai.Agent` unchanged
    (`tools`, `deps_type`, `model_settings`, ...). With output orbitals, use
    `run()` / `run_sync()`: streaming is refused because PydanticAI streams
    text before it can be checked.

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
