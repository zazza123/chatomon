"""Build an atom into a native Google ADK application.

Install with `pip install 'chatomon[adk]'`.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import Field

from chatomon.connectors.base import Connector, ModelT, render_guidance
from chatomon.errors import ChatomonConfigError, MissingExtraError
from chatomon.schemas import InputDecision, Message


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # the parent package is missing too
        return False


try:
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.events import Event
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.models.registry import LLMRegistry
    from google.adk.plugins.base_plugin import BasePlugin
    from google.genai import types
except ImportError as exc:
    if not _installed("google.adk"):
        raise MissingExtraError("adk", "google-adk") from exc
    raise

if TYPE_CHECKING:
    from chatomon.atom import Atom
    from chatomon.engine import Engine, TurnState

__all__ = ["ADKConnector"]

_FROM_THE_ATOM = frozenset({"name", "instruction"})
_APP_FROM_THE_ATOM = frozenset({"name", "root_agent"})
_REFUSED = {
    "static_instruction": (
        "the instructions come from the atom, and static_instruction would move them out "
        "of the system instruction; put the text in the atom's instructions or knowledge"
    ),
    "global_instruction": (
        "the instructions come from the atom; put the text in the atom's instructions"
    ),
    "output_schema": "the nucleus replies with text, which is what the orbitals check",
    "sub_agents": (
        "a sub-agent would answer with a model of its own, which the output orbitals do "
        "not see; give the other agents to the nucleus as tools (AgentTool)"
    ),
    "mode": (
        "an atom is a chat agent; in the other modes the result leaves through a tool "
        "call, which the output orbitals do not see"
    ),
}
_NO_SUMMARIZER = (
    "events_compaction_config needs a summarizer with a model of its own: without one "
    "ADK would summarise with the nucleus, outside a turn; pass "
    "EventsCompactionConfig(summarizer=LlmEventSummarizer(llm=...), ...)"
)
_BAD_NAME = (
    "ADK needs the atom's name to be a Python identifier other than 'user' (letters, "
    "digits and underscores, not starting with a digit), got {name!r}"
)
_UNKNOWN_MODEL = (
    "ADK knows no model named {name!r}; pass a name its registry resolves or a BaseLlm object"
)
_NOT_A_MODEL = "an ADK model is a BaseLlm object or a model name, got {kind}"
_OUTSIDE_THE_APP = (
    "this agent runs only inside the App that Atom.build() returned; "
    "run it with Runner(app=app, ...) or InMemoryRunner(app=app)"
)
_LIVE_REFUSED = (
    "ADK live runs carry no text turn chatomon's orbitals can check; use run_async() or run()"
)
_NO_FINAL_REPLY = "the run ended without a final text reply"
_NO_ANSWER = "the model returned no answer"

_current: ContextVar[_Turn | None] = ContextVar("chatomon_adk_turn", default=None)
"""The turn of the model call that is about to start, set by the plugin for the nucleus."""


@dataclass
class _Turn:
    turn: TurnState
    decision: InputDecision
    history_out: tuple[Message, ...]
    output_guidance: tuple[str, ...] = ()


def _text(content: types.Content | None, between: str = "") -> str:
    """Read the text of a content, without thoughts.

    Returns:
        The text parts joined: as they are for a reply, which a model writes in
        pieces, and `between` a line break for a message, whose parts are apart.
    """
    parts = content.parts if content is not None and content.parts else []
    return between.join(part.text for part in parts if part.text and not part.thought)


def _calls_a_tool(content: types.Content | None) -> bool:
    parts = content.parts if content is not None and content.parts else []
    return any(part.function_call or part.function_response for part in parts)


def _said(text: str, source: LlmResponse | None = None) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        usage_metadata=source.usage_metadata if source is not None else None,
        model_version=source.model_version if source is not None else None,
    )


def _history(events: Sequence[Event]) -> list[Message]:
    history: list[Message] = []
    for event in events:
        role = "user" if event.author == "user" else "assistant"
        text = _text(event.content, "\n" if role == "user" else "")
        if event.partial or not text or _calls_a_tool(event.content):
            continue
        history.append(Message(role=role, content=text))
    return history


def _last_user_text(events: Sequence[Event]) -> tuple[str, list[Message]]:
    """Find the message a resumed run answers to.

    Returns:
        The last text the user wrote and the history before it.
    """
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.author == "user" and _text(event.content):
            return _text(event.content, "\n"), _history(events[:index])
    return "", _history(events)


def _cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _resolve(model: object) -> BaseLlm:
    if isinstance(model, BaseLlm):
        return model
    if not isinstance(model, str):
        raise ChatomonConfigError(_NOT_A_MODEL.format(kind=type(model).__name__))
    try:
        return LLMRegistry.new_llm(model)
    except ValueError as exc:
        raise ChatomonConfigError(_UNKNOWN_MODEL.format(name=model)) from exc


class _Nucleus(BaseLlm):
    """The developer's model, with chatomon's guidance and output chain around every call."""

    inner: BaseLlm
    engine: Any = Field(exclude=True, repr=False)

    @property
    def capabilities(self) -> Any:
        return self.inner.capabilities

    def connect(self, llm_request: LlmRequest) -> Any:
        raise ChatomonConfigError(_LIVE_REFUSED)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        current = _current.get()
        if current is None or current.turn.finished:
            raise ChatomonConfigError(_OUTSIDE_THE_APP)
        engine: Engine = self.engine
        turn, decision = current.turn, current.decision
        if decision.kind == "reply":
            # The plugin ends such a run before the agent starts; a path that reaches
            # the model all the same gets the reply, never a model call.
            yield _said(decision.reply or "")
            return
        # With output orbitals no text leaves before it is checked.
        streaming = stream and not engine.output_orbitals
        base = llm_request.config.model_copy(deep=True)
        final: LlmResponse
        try:
            if turn.final_action is not None:
                # The flow asks the model again after chatomon decided on a reply.
                engine.reopen_output(turn)
            while True:
                llm_request.config = base.model_copy(deep=True)
                guidance = (*decision.guidance, *current.output_guidance)
                if guidance:
                    llm_request.append_instructions([render_guidance(guidance)])
                response: LlmResponse | None = None
                calling = self.inner.generate_content_async(llm_request, stream=streaming)
                async with aclosing(calling) as responses:
                    async for following in responses:
                        if response is not None:
                            yield response
                        response = following
                if response is None:
                    raise RuntimeError(_NO_ANSWER)
                text = _text(response.content)
                if _calls_a_tool(response.content) or not text:
                    final = response
                    break
                outcome = await engine.run_output(turn, reply=text, history=current.history_out)
                if outcome.kind == "accept":
                    final = response
                    break
                if outcome.kind == "replace":
                    final = _said(outcome.text or "", response)
                    break
                current.output_guidance = outcome.guidance
        finally:
            # ADK records the request once the reply is back: it gets it without the
            # guidance, also when an error or a cancellation ends the call.
            llm_request.config = base
        yield final


class _ChatomonPlugin(BasePlugin):
    """Runs the input chain and follows the turn of every run of the atom's agent."""

    def __init__(self, engine: Engine, placeholder: str, nucleus: _Nucleus) -> None:
        super().__init__(name="chatomon")
        self._engine = engine
        self._placeholder = placeholder
        self._nucleus = nucleus
        self._turns: dict[str, _Turn] = {}

    async def on_user_message_callback(
        self, *, invocation_context: Any, user_message: types.Content
    ) -> types.Content | None:
        if getattr(invocation_context.agent, "model", None) is not self._nucleus:
            # The run of another agent with the same plugins, as an AgentTool starts: a tool.
            return None
        session = invocation_context.session
        message, history = _text(user_message, "\n"), _history(session.events)
        # An answer to a paused run has no text: the message it answers to is checked again.
        resumed = not message and _calls_a_tool(user_message)
        if resumed:
            message, history = _last_user_text(session.events)
        turn = self._engine.start_turn(conversation_id=session.id)
        decision = await self._engine.run_input(turn, message=message, history=history)
        user_text = self._placeholder if decision.redact_user_message else message
        self._turns[invocation_context.invocation_id] = _Turn(
            turn=turn,
            decision=decision,
            history_out=(*history, Message(role="user", content=user_text)),
        )
        if decision.redact_user_message and not resumed:
            return types.Content(role="user", parts=[types.Part(text=self._placeholder)])
        return None

    async def before_run_callback(self, *, invocation_context: Any) -> types.Content | None:
        current = self._turns.get(invocation_context.invocation_id)
        if current is None or current.decision.kind != "reply":
            return None
        reply = current.decision.reply or ""
        return types.Content(role="model", parts=[types.Part(text=reply)])

    async def on_event_callback(self, *, invocation_context: Any, event: Event) -> Event | None:
        current = self._turns.get(invocation_context.invocation_id)
        if current is not None and current.decision.kind == "reply" and event.author == "model":
            # ADK signs the reply that ended the run as "model", and in the turns that
            # follow shows it to the nucleus as the words of another agent. Signed by
            # the agent it is the agent's own turn. Changed in place: the plugins that
            # follow still see the event.
            event.author = invocation_context.agent.name
        return None

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: LlmRequest
    ) -> LlmResponse | None:
        _current.set(self._turns.get(callback_context.invocation_id))
        return None

    async def after_agent_callback(
        self, *, agent: Any, callback_context: Any
    ) -> types.Content | None:
        if _cancelling():
            # Some ADK versions end a cancelled run here and never call after_run.
            await self._close(callback_context.invocation_id)
        return None

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        await self._close(invocation_context.invocation_id)

    async def on_run_error_callback(self, *, invocation_context: Any, error: Exception) -> None:
        current = self._turns.pop(invocation_context.invocation_id, None)
        if current is not None and not current.turn.finished:
            await self._engine.abort_turn(current.turn, error)

    async def _close(self, invocation_id: str) -> None:
        current = self._turns.pop(invocation_id, None)
        if current is None or current.turn.finished:
            return
        if _cancelling():
            await self._engine.abort_turn(current.turn, asyncio.CancelledError())
        elif current.turn.final_action is None:
            await self._engine.abort_turn(current.turn, RuntimeError(_NO_FINAL_REPLY))
        else:
            await self._engine.finish_turn(current.turn)


class ADKConnector(Connector["App"]):
    """Builds an atom into a native `google.adk.apps.App`.

    The app's root agent is the nucleus, an `LlmAgent`, and its first plugin is
    chatomon's; your own `plugins` come after it. Run the app with ADK's own
    `Runner(app=app, ...)`, `adk web` or `adk run`: the agent refuses to run
    outside it. The id of the ADK session is the conversation id.

    When the atom has output orbitals the nucleus is called without streaming,
    so a caller that asks for streaming receives each reply whole, once it is
    checked. Live runs are refused.

    Args:
        model: The nucleus model: an ADK `BaseLlm` or a name ADK's registry resolves.
        resolve_model: Maps the model id of a
            [JudgeOrbital][chatomon.orbitals.judge.JudgeOrbital] to an ADK model
            or to a model name; by default the id is a name for the registry.
        app_kwargs: Passed to `google.adk.apps.App` (`plugins`,
            `context_cache_config`, `resumability_config`, ...).
        **agent_kwargs: Passed to `google.adk.agents.LlmAgent` (`tools`,
            `generate_content_config`, callbacks, ...).

    Raises:
        ChatomonConfigError: If an argument is set by chatomon from the atom
            (`name`, `instruction`; `name` and `root_agent` of the app), cannot
            be checked by the orbitals (`sub_agents`, `mode`, `output_schema`,
            `static_instruction`, `global_instruction`), or is an events
            compaction without a summarizer.

    Examples:
        ```{.python test="skip"}
        import asyncio

        from google.adk.runners import InMemoryRunner

        from chatomon import Atom
        from chatomon.connectors.adk import ADKConnector


        async def main() -> None:
            app = Atom(
                name="speedypack",
                instructions="You are the customer assistant of SpeedyPack.",
                connector=ADKConnector("gemini-3.5-flash"),
            ).build()
            runner = InMemoryRunner(app=app)
            events = await runner.run_debug("Hello", quiet=True)
            print(events[-1].content.parts[0].text)


        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        model: str | BaseLlm,
        *,
        resolve_model: Callable[[str], BaseLlm | str] | None = None,
        app_kwargs: Mapping[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        app_kwargs = dict(app_kwargs or {})
        taken = sorted(_FROM_THE_ATOM & agent_kwargs.keys())
        taken += [f"app_kwargs[{key!r}]" for key in sorted(_APP_FROM_THE_ATOM & app_kwargs.keys())]
        if taken:
            raise ChatomonConfigError(
                f"{', '.join(taken)} is set by chatomon from the atom; "
                "remove it from ADKConnector(...)"
            )
        for key, reason in _REFUSED.items():
            if agent_kwargs.get(key):
                raise ChatomonConfigError(f"ADKConnector does not take {key}: {reason}")
        compaction = app_kwargs.get("events_compaction_config")
        if compaction is not None and getattr(compaction, "summarizer", None) is None:
            raise ChatomonConfigError(_NO_SUMMARIZER)
        self.model = model
        self.resolve_model = resolve_model
        self.app_kwargs = app_kwargs
        self.agent_kwargs = agent_kwargs

    def build(self, atom: Atom[App], engine: Engine) -> App:
        """Build the native app with chatomon's plugin and the nucleus as its root agent.

        Args:
            atom: The validated atom.
            engine: The engine that runs the atom's orbitals.

        Returns:
            A `google.adk.apps.App` named like the atom.

        Raises:
            ChatomonConfigError: If the atom's name is not a valid ADK agent
                name, or ADK's registry does not know the model's name.
        """
        if not atom.name.isidentifier() or atom.name == "user":
            raise ChatomonConfigError(_BAD_NAME.format(name=atom.name))
        inner = _resolve(self.model)
        nucleus = _Nucleus(model=inner.model, inner=inner, engine=engine)
        prompt = atom.system_prompt
        app_kwargs = dict(self.app_kwargs)
        plugins = [
            _ChatomonPlugin(engine, atom.blocked_placeholder, nucleus),
            *(app_kwargs.pop("plugins", None) or ()),
        ]
        agent = LlmAgent(
            name=atom.name,
            model=nucleus,
            # A function, so that ADK does not read braces in the text as session variables.
            instruction=lambda _context: prompt,
            **self.agent_kwargs,
        )
        return App(name=atom.name, root_agent=agent, plugins=plugins, **app_kwargs)

    async def generate_structured(
        self, model: str, *, system: str, prompt: str, schema: type[ModelT]
    ) -> ModelT:
        """Make one structured call straight to an ADK model, with no agent or session.

        Args:
            model: Model id; passed through `resolve_model` when set.
            system: System instruction.
            prompt: User prompt.
            schema: Output type.

        Returns:
            The validated output.

        Raises:
            ChatomonConfigError: If the model id resolves to no ADK model.
            ValueError: If the model returned no text, or text that does not
                match `schema`.
        """  # noqa: DOC502 - the first is raised by _resolve(), the second also by pydantic
        resolved = self.resolve_model(model) if self.resolve_model is not None else model
        llm = _resolve(resolved)
        request = LlmRequest(
            model=llm.model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(system_instruction=system),
        )
        request.set_output_schema(schema)
        async with aclosing(llm.generate_content_async(request, stream=False)) as responses:
            answers = [response async for response in responses]
        text = _text(answers[-1].content) if answers else ""
        if not text:
            raise ValueError(_NO_ANSWER)
        return schema.model_validate_json(text)
