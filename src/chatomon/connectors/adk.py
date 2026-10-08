"""Build an atom into a native Google ADK application.

Install with `pip install 'chatomon[adk]'`.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import weakref
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
        "an atom is a chat agent, the mode a root agent has by default; in the others "
        "the result leaves through a tool call, which the output orbitals do not see"
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
    "this agent runs only inside the App that Atom.build() returned: run it with "
    "Runner(app=app, ...) or InMemoryRunner(app=app), and do not call the agent's model "
    "from your own code"
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
    nucleus: _Nucleus
    """The nucleus of the app the turn belongs to: no other may run on it."""
    output_guidance: tuple[str, ...] = ()
    orphaned: weakref.finalize[Any, Any] | None = None
    """Closes the turn when ADK drops the run without telling its plugins."""


def _text(content: types.Content | None, *, message: bool = False) -> str:
    """Read the text of a reply, or of a message of the user's.

    Returns:
        For a reply, which a model writes in pieces, the text parts joined as
        they are, without thoughts. For a message every text part, each on a
        line of its own: its parts are apart, and a thought is a notion of
        replies, not something the writer of a message can hide text in.
    """
    parts = content.parts if content is not None and content.parts else []
    if message:
        return "\n".join(part.text for part in parts if part.text)
    return "".join(part.text for part in parts if part.text and not part.thought)


def _calls_a_tool(content: types.Content | None) -> bool:
    parts = content.parts if content is not None and content.parts else []
    return any(part.function_call or part.function_response for part in parts)


def _said(text: str, source: LlmResponse | None = None) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        usage_metadata=source.usage_metadata if source is not None else None,
        model_version=source.model_version if source is not None else None,
    )


def _live(events: Sequence[Event]) -> list[Event]:
    """Leave out the turns a rewind took back, as ADK does for the model.

    ADK keeps those events in the session. An event that carries a rewind hides
    itself and every event since the first one of the run it names.

    Returns:
        The events the model is still shown, in the order they were stored.
    """
    first: dict[str, int] = {}
    for index, event in enumerate(events):
        if event.invocation_id:
            first.setdefault(event.invocation_id, index)
    live: list[Event] = []
    index = len(events) - 1
    while index >= 0:
        event = events[index]
        rewound = event.actions.rewind_before_invocation_id if event.actions else None
        if rewound:
            start = first.get(rewound)
            if start is not None and start < index:
                index = start
        else:
            live.append(event)
        index -= 1
    live.reverse()
    return live


def _history(events: Sequence[Event]) -> list[Message]:
    history: list[Message] = []
    for event in events:
        role = "user" if event.author == "user" else "assistant"
        text = _text(event.content, message=role == "user")
        if event.partial or not text or _calls_a_tool(event.content):
            continue
        history.append(Message(role=role, content=text))
    return history


def _message_of(events: Sequence[Event], invocation_id: str) -> tuple[str, list[Message]]:
    """Find the message of a run that ADK started with no new text.

    Returns:
        The text the user wrote in that run, or the last they wrote in the
        session when the run has none, and the history before it.
    """
    written = [
        index
        for index, event in enumerate(events)
        if event.author == "user" and _text(event.content, message=True)
    ]
    own = [index for index in written if events[index].invocation_id == invocation_id]
    if not written:
        return "", _history(events)
    index = (own or written)[-1]
    return _text(events[index].content, message=True), _history(events[:index])


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
        if current is None or current.nucleus is not self or current.turn.finished:
            raise ChatomonConfigError(_OUTSIDE_THE_APP)
        # One hand-off, one model call: whoever calls again without ADK finds no turn.
        _current.set(None)
        engine: Engine = self.engine
        turn, decision = current.turn, current.decision
        if decision.kind == "reply":
            # The plugin ends such a run before the agent starts; a path that reaches
            # the model all the same gets the reply, never a model call.
            yield _said(decision.reply or "")
            return
        # With output orbitals no text leaves before it is checked.
        streaming = stream and not engine.output_orbitals
        # Only the system instruction is touched: the rest of the configuration can
        # hold a live client, which no copy survives.
        base = llm_request.config.system_instruction
        final: LlmResponse
        try:
            if turn.final_action is not None:
                # The flow asks the model again after chatomon decided on a reply.
                engine.reopen_output(turn)
            while True:
                llm_request.config.system_instruction = copy.deepcopy(base)
                guidance = (*decision.guidance, *current.output_guidance)
                if guidance:
                    llm_request.append_instructions([render_guidance(guidance)])
                guided = llm_request.config.system_instruction
                response: LlmResponse | None = None
                calling = self.inner.generate_content_async(llm_request, stream=streaming)
                async with aclosing(calling) as responses:
                    async for following in responses:
                        if response is not None and streaming:
                            # ADK records the request whenever a response is in its
                            # hands: it holds the guidance only while the model works.
                            llm_request.config.system_instruction = base
                            yield response
                            llm_request.config.system_instruction = guided
                        # Without streaming a model gives one response; of one that
                        # gives more, only the last is the reply.
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
            # Also when an error or a cancellation ends the call.
            llm_request.config.system_instruction = base
        yield final


class _ChatomonPlugin(BasePlugin):
    """Runs the input chain and follows the turn of every run of the atom's agent."""

    def __init__(self, engine: Engine, placeholder: str, nucleus: _Nucleus) -> None:
        super().__init__(name="chatomon")
        self._engine = engine
        self._placeholder = placeholder
        self._nucleus = nucleus
        self._turns: dict[str, _Turn] = {}
        self._reporting: set[asyncio.Task[None]] = set()

    def _mine(self, invocation_context: Any) -> bool:
        # Another agent can run with the same plugins, as an AgentTool's does: a tool.
        return getattr(invocation_context.agent, "model", None) is self._nucleus

    async def _open(self, invocation_context: Any, message: str | None) -> _Turn:
        """Open the turn of a run and run the input chain on its message.

        Args:
            invocation_context: The run.
            message: The text of the message that arrived, or `None` when the
                run has none of its own and its message is read from the session.

        Returns:
            The turn, kept until the run ends.
        """
        session = invocation_context.session
        events = _live(session.events)
        if message is None:
            message, history = _message_of(events, invocation_context.invocation_id)
        else:
            history = _history(events)
        turn = self._engine.start_turn(conversation_id=session.id)
        try:
            decision = await self._engine.run_input(turn, message=message, history=history)
        except BaseException as error:
            # Cancelled while an orbital ran: no hook of ADK's will close this turn.
            if not turn.finished:
                await self._engine.abort_turn(turn, error)
            raise
        user_text = self._placeholder if decision.redact_user_message else message
        current = _Turn(
            turn=turn,
            decision=decision,
            history_out=(*history, Message(role="user", content=user_text)),
            nucleus=self._nucleus,
        )
        # ADK calls no plugin when a run is cancelled while the runner itself works
        # (storing an event, another plugin's hook). The run's context dies with the
        # run, whatever ended it: that is what closes a turn nothing else closed.
        current.orphaned = weakref.finalize(
            invocation_context,
            self._orphaned,
            invocation_context.invocation_id,
            current,
            asyncio.get_running_loop(),
        )
        current.orphaned.atexit = False
        self._turns[invocation_context.invocation_id] = current
        return current

    def _take(self, invocation_id: str) -> _Turn | None:
        current = self._turns.pop(invocation_id, None)
        if current is not None and current.orphaned is not None:
            current.orphaned.detach()
        return current

    def _orphaned(
        self, invocation_id: str, current: _Turn, loop: asyncio.AbstractEventLoop
    ) -> None:
        if self._turns.get(invocation_id) is current:
            del self._turns[invocation_id]
        if not current.turn.finished and not loop.is_closed():
            # The context can be collected from any thread; the report belongs to the loop.
            loop.call_soon_threadsafe(self._report, current)

    def _report(self, current: _Turn) -> None:
        async def abort() -> None:
            if not current.turn.finished:
                await self._engine.abort_turn(current.turn, asyncio.CancelledError())

        task = asyncio.ensure_future(abort())
        self._reporting.add(task)
        task.add_done_callback(self._reporting.discard)

    async def on_user_message_callback(
        self, *, invocation_context: Any, user_message: types.Content
    ) -> types.Content | None:
        if not self._mine(invocation_context):
            return None
        message = _text(user_message, message=True)
        if not message and _calls_a_tool(user_message):
            # An answer to a paused run has no text: the message of that run is
            # checked again. It is already stored, so it cannot be replaced.
            await self._open(invocation_context, None)
            return None
        current = await self._open(invocation_context, message)
        if current.decision.redact_user_message:
            return types.Content(role="user", parts=[types.Part(text=self._placeholder)])
        return None

    async def before_run_callback(self, *, invocation_context: Any) -> types.Content | None:
        current = self._turns.get(invocation_context.invocation_id)
        if current is None and self._mine(invocation_context):
            # ADK started the run with no new message, as a run started again by its
            # id is: nothing reached the plugin, and the run's message is checked again.
            current = await self._open(invocation_context, None)
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
        current = self._take(invocation_context.invocation_id)
        if current is not None and not current.turn.finished:
            await self._engine.abort_turn(current.turn, error)

    async def _close(self, invocation_id: str) -> None:
        current = self._take(invocation_id)
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
            be checked by the orbitals (`sub_agents`, a `mode` other than
            `"chat"`, `output_schema`, `static_instruction`,
            `global_instruction`), or is an events compaction without a
            summarizer.

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
            if agent_kwargs.get(key) and (key, agent_kwargs[key]) != ("mode", "chat"):
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
            ValueError: If the model returned an error or no text, or text that
                does not match `schema`.
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
        failed = not answers or answers[-1].error_code is not None
        text = "" if failed else _text(answers[-1].content)
        if not text:
            raise ValueError(_NO_ANSWER)
        return schema.model_validate_json(text)
