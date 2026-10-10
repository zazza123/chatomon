"""Build an atom into a native LangChain agent (a LangGraph compiled graph).

Install with `pip install 'chatomon[langchain]'`.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from opentelemetry import trace

from chatomon._sync import run_sync
from chatomon.connectors.base import Connector, ModelT, render_guidance
from chatomon.errors import ChatomonConfigError, MissingExtraError
from chatomon.schemas import Message

try:
    from typing import NotRequired

    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
    from langchain.agents.middleware.types import ModelRequest, ModelResponse, PrivateStateAttr
    from langchain.chat_models import init_chat_model
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
    from langgraph.config import get_config
    from langgraph.errors import GraphBubbleUp
    from langgraph.graph.state import CompiledStateGraph
except ImportError as exc:
    if importlib.util.find_spec("langchain") is None:
        raise MissingExtraError("langchain", "langchain") from exc
    raise

if TYPE_CHECKING:
    from chatomon.atom import Atom
    from chatomon.engine import Engine, TurnState

__all__ = ["LangChainConnector"]

_RESERVED = frozenset({"system_prompt", "response_format", "name"})
_NO_FINAL_REPLY = "the agent ended without a final text reply"
_STALE_TURN = "the previous turn ended before chatomon could close it"
_NO_AI_MESSAGE = "the model call returned no AI message, so there is no reply to check"

ModelCall = Callable[[ModelRequest], Awaitable[ModelResponse]]


class _ChatomonState(AgentState):
    chatomon_turn: NotRequired[Annotated[str | None, PrivateStateAttr]]


@dataclass
class _Turn:
    turn: TurnState
    input_guidance: tuple[str, ...]
    history_out: tuple[Message, ...]
    output_guidance: tuple[str, ...] = ()
    reply: str | None = None
    """Canned reply decided when a turn is rebuilt inside the model call."""
    finish_in_wrap: bool = False
    """Close the turn in the model call: no private state points at it."""


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, (str, dict))
        ]
        return " ".join(part for part in parts if part)
    return ""


def _history(messages: Sequence[AnyMessage]) -> list[Message]:
    history: list[Message] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            history.append(Message(role="user", content=_text(message.content)))
        elif isinstance(message, AIMessage) and not message.tool_calls:
            text = _text(message.content)
            if text:
                history.append(Message(role="assistant", content=text))
    return history


def _thread_id() -> str | None:
    try:
        thread_id = get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        return None
    return None if thread_id is None else str(thread_id)


def _with_guidance(request: ModelRequest, guidance: Sequence[str]) -> ModelRequest:
    if not guidance:
        return request
    base = _text(request.system_message.content) if request.system_message is not None else ""
    section = render_guidance(guidance)
    text = f"{base}\n\n{section}" if base else section
    return request.override(system_message=SystemMessage(content=text))


def _with_nostream(model: Any) -> Any:
    # A tag on the model object itself survives create_agent's bind_tools();
    # a RunnableBinding from with_config() loses it as soon as tools are bound.
    if isinstance(model, BaseChatModel):
        return model.model_copy(update={"tags": [*(model.tags or []), "nostream"]})
    return model.with_config(tags=["nostream"])


def _last_ai(response: ModelResponse) -> AIMessage | None:
    return next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)


class _ChatomonMiddleware(AgentMiddleware[_ChatomonState]):
    """Runs the engine around a LangChain agent; turns are keyed by id in private state."""

    state_schema = _ChatomonState

    def __init__(
        self,
        engine: Engine,
        placeholder: str,
        parent_span: Callable[[], trace.Span | None] | None,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._placeholder = placeholder
        self._parent_span = parent_span
        self._turns: dict[str, _Turn] = {}

    @property
    def name(self) -> str:
        return "chatomon"

    @contextmanager
    def _parent(self) -> Iterator[None]:
        span = self._parent_span() if self._parent_span is not None else None
        if span is None:
            yield
            return
        with trace.use_span(span, end_on_exit=False):
            yield

    async def _start(self, state: dict[str, Any]) -> dict[str, Any]:
        stale_id = state.get("chatomon_turn")
        stale = self._turns.pop(stale_id, None) if isinstance(stale_id, str) else None
        if stale is not None and not stale.turn.finished:
            with self._parent():
                await self._engine.abort_turn(stale.turn, RuntimeError(_STALE_TURN))
        messages = state["messages"]
        last = messages[-1] if messages else None
        message = _text(last.content) if isinstance(last, HumanMessage) else ""
        history = _history(messages[:-1])
        with self._parent():
            turn = self._engine.start_turn(conversation_id=_thread_id())
            decision = await self._engine.run_input(turn, message=message, history=history)
            updates: dict[str, Any] = {}
            user_text = message
            if decision.redact_user_message and isinstance(last, HumanMessage):
                updates["messages"] = [HumanMessage(content=self._placeholder, id=last.id)]
                user_text = self._placeholder
            if decision.kind == "reply":
                await self._engine.finish_turn(turn)
                updates.setdefault("messages", []).append(AIMessage(content=decision.reply or ""))
                updates["jump_to"] = "end"
                return updates
        self._turns[turn.turn_id] = _Turn(
            turn=turn,
            input_guidance=decision.guidance,
            history_out=(*history, Message(role="user", content=user_text)),
        )
        updates["chatomon_turn"] = turn.turn_id
        return updates

    async def _recover(self, request: ModelRequest) -> tuple[_Turn, ModelRequest]:
        """Rebuild a turn the middleware does not know (resumed elsewhere, restarted).

        The input chain runs again on the last human message, so the model is never
        called without chatomon's checks.

        Returns:
            The rebuilt turn and the request, with the user message redacted if needed.
        """
        messages = list(request.messages)
        index = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
            None,
        )
        message = _text(messages[index].content) if index is not None else ""
        history = _history(messages[:index] if index is not None else messages)
        turn = self._engine.start_turn(conversation_id=_thread_id())
        decision = await self._engine.run_input(turn, message=message, history=history)
        user_text = message
        if decision.redact_user_message and index is not None:
            messages[index] = HumanMessage(content=self._placeholder, id=messages[index].id)
            request = request.override(messages=messages)
            user_text = self._placeholder
        current = _Turn(
            turn=turn,
            input_guidance=decision.guidance,
            history_out=(*history, Message(role="user", content=user_text)),
            reply=decision.reply if decision.kind == "reply" else None,
        )
        return current, request

    async def _wrap(self, request: ModelRequest, call: ModelCall) -> ModelResponse:
        turn_id = request.state.get("chatomon_turn")
        current = self._turns.get(turn_id) if isinstance(turn_id, str) else None
        if current is None:
            with self._parent():
                current, request = await self._recover(request)
            if current.reply is not None:
                with self._parent():
                    await self._engine.finish_turn(current.turn)
                return ModelResponse(result=[AIMessage(content=current.reply)])
            current.finish_in_wrap = not isinstance(turn_id, str)
            self._turns[turn_id if isinstance(turn_id, str) else current.turn.turn_id] = current
        key = turn_id if isinstance(turn_id, str) else current.turn.turn_id
        try:
            if current.turn.final_action is not None:
                # The graph asks the model again after chatomon accepted a reply.
                self._engine.reopen_output(current.turn)
            response = await call(
                _with_guidance(request, (*current.input_guidance, *current.output_guidance))
            )
            while True:
                reply = _last_ai(response)
                if reply is None:
                    raise RuntimeError(_NO_AI_MESSAGE)
                if reply.tool_calls:
                    return response
                text = _text(reply.content)
                if not text:
                    # Nothing to check: a turn that gets no other reply is closed as aborted.
                    if current.finish_in_wrap:
                        self._turns.pop(key, None)
                        with self._parent():
                            await self._engine.abort_turn(
                                current.turn, RuntimeError(_NO_FINAL_REPLY)
                            )
                    return response
                with self._parent():
                    outcome = await self._engine.run_output(
                        current.turn, reply=text, history=current.history_out
                    )
                if outcome.kind == "regenerate":
                    current.output_guidance = outcome.guidance
                    response = await call(
                        _with_guidance(request, (*current.input_guidance, *current.output_guidance))
                    )
                    continue
                if current.finish_in_wrap:
                    self._turns.pop(key, None)
                    with self._parent():
                        await self._engine.finish_turn(current.turn)
                if outcome.kind == "accept":
                    return response
                return ModelResponse(result=[AIMessage(content=outcome.text or "", id=reply.id)])
        except GraphBubbleUp:
            # An interrupt: the turn stays open and resumes with the graph.
            raise
        except BaseException as exc:
            self._turns.pop(key, None)
            if not current.turn.finished:
                with self._parent():
                    await self._engine.abort_turn(current.turn, exc)
            raise

    async def _abort_after_tool_error(self, state: Any, error: BaseException) -> None:
        turn_id = state.get("chatomon_turn") if isinstance(state, dict) else None
        current = self._turns.pop(turn_id, None) if isinstance(turn_id, str) else None
        if current is not None and not current.turn.finished:
            with self._parent():
                await self._engine.abort_turn(current.turn, error)

    async def _finish(self, state: dict[str, Any]) -> dict[str, Any] | None:
        turn_id = state.get("chatomon_turn")
        if not turn_id:
            return None
        current = self._turns.pop(turn_id, None)
        if current is not None and not current.turn.finished:
            with self._parent():
                if current.turn.final_action is None:
                    await self._engine.abort_turn(current.turn, RuntimeError(_NO_FINAL_REPLY))
                else:
                    await self._engine.finish_turn(current.turn)
        return {"chatomon_turn": None}

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return await self._start(state)

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return run_sync(lambda: self._start(state))

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> ModelResponse:
        return await self._wrap(request, handler)

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> ModelResponse:
        async def call(model_request: ModelRequest) -> ModelResponse:
            # Blocking model calls run off the shared event loop.
            return await asyncio.to_thread(handler, model_request)

        return run_sync(lambda: self._wrap(request, call))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except BaseException as exc:
            await self._abort_after_tool_error(request.state, exc)
            raise

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except BaseException as exc:
            error = exc
            run_sync(lambda: self._abort_after_tool_error(request.state, error))
            raise

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return await self._finish(state)

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return run_sync(lambda: self._finish(state))


class LangChainConnector(Connector["CompiledStateGraph[Any, Any, Any, Any]"]):
    """Builds an atom into a native LangChain agent (a LangGraph `CompiledStateGraph`).

    chatomon's middleware is the outermost layer; your own `middleware` run
    inside it. Extra keyword arguments go to `langchain.agents.create_agent`
    unchanged (`tools`, `checkpointer`, `store`, ...). With a checkpointer, the
    `thread_id` of the run config becomes the conversation id. When the atom has
    output orbitals the model is tagged `nostream`, so streaming emits only
    checked replies.

    Args:
        model: The nucleus model: a chat model or a `"provider:model"` string.
        resolve_model: Maps the model id of a
            [JudgeOrbital][chatomon.orbitals.judge.JudgeOrbital] to a chat model;
            by default `init_chat_model(model_id)`.
        parent_span: Returns the span chatomon's spans should nest under;
            LangChain does not propagate an OpenTelemetry context itself.
        **agent_kwargs: Passed to `create_agent`.

    Raises:
        ChatomonConfigError: If `agent_kwargs` sets `system_prompt`,
            `response_format` or `name` (chatomon sets them from the atom).

    Examples:
        ```python
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage, HumanMessage

        from chatomon import Atom
        from chatomon.connectors.langchain import LangChainConnector

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hi!")]))
        agent = Atom(
            name="speedypack",
            instructions="You are the customer assistant of SpeedyPack.",
            connector=LangChainConnector(model),
        ).build()
        state = agent.invoke({"messages": [HumanMessage(content="hello")]})
        assert state["messages"][-1].content == "Hi!"
        ```
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        resolve_model: Callable[[str], BaseChatModel] | None = None,
        parent_span: Callable[[], trace.Span | None] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        reserved = sorted(_RESERVED & set(agent_kwargs))
        if reserved:
            raise ChatomonConfigError(
                f"{', '.join(reserved)} is set by chatomon from the atom; "
                "remove it from LangChainConnector(...)"
            )
        self.model = model
        self.resolve_model = resolve_model
        self.parent_span = parent_span
        self.agent_kwargs = agent_kwargs

    def build(self, atom: Atom[Any], engine: Engine) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Build the native agent with chatomon's middleware installed.

        Args:
            atom: The validated atom.
            engine: The engine that runs the atom's orbitals.

        Returns:
            The compiled LangGraph agent.
        """
        model: Any = self.model
        if engine.output_orbitals:
            if isinstance(model, str):
                model = init_chat_model(model)
            model = _with_nostream(model)
        kwargs = dict(self.agent_kwargs)
        middleware = [
            _ChatomonMiddleware(engine, atom.blocked_placeholder, self.parent_span),
            *kwargs.pop("middleware", ()),
        ]
        return create_agent(
            model,
            system_prompt=atom.system_prompt,
            middleware=middleware,
            name=atom.name,
            **kwargs,
        )

    async def generate_structured(
        self, model: str, *, system: str, prompt: str, schema: type[ModelT]
    ) -> ModelT:
        """Make one structured call with `with_structured_output`.

        Args:
            model: Model id; resolved with `resolve_model` or `init_chat_model`.
            system: System prompt.
            prompt: User prompt.
            schema: Output type.

        Returns:
            The validated output.
        """
        llm = (
            self.resolve_model(model) if self.resolve_model is not None else init_chat_model(model)
        )
        result = await llm.with_structured_output(schema).ainvoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        return result if isinstance(result, schema) else schema.model_validate(result)
