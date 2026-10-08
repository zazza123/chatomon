"""Studio turns on atoms built with `ADKConnector`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from chatomon.connectors.base import Connector
from chatomon.studio._runners import InterruptedTurnError, RunOutcome
from chatomon.studio._testgen import Call, Framework, StandIn

_USER = "studio"
_INTERRUPTED = (
    "the agent stopped to wait for an answer, which the studio cannot give: "
    "the turn is left unfinished"
)
_PLUGINS = (
    "this atom has plugins of its own ({names}); they run on every turn of the studio "
    "too, and write where they usually write"
)
_ASYNC = "Google ADK is asynchronous: the test needs `pytest-asyncio` 0.24 or later."
_OWN_CLASS = (
    "`{name}` is written on Google ADK's model classes: if they change, save the test again."
)
_PLUGINS_RUN = (
    "This atom has plugins of its own: they run in this test too, and write where they "
    "usually write."
)
# A callback reaches the model through the agent as a turn does, or answers in its place.
_CALLBACKS = (
    "This atom has callbacks of its own: one that calls a model can still reach a real "
    "model in this test, and one that writes the reply changes what the test sees."
)
_MODEL_CLASSES = (
    "from google.adk.models.base_llm import BaseLlm",
    "from google.adk.models.llm_response import LlmResponse",
    "from google.genai import types",
)
_OFFLINE_NUCLEUS = (
    "class OfflineNucleus(BaseLlm):",
    '    """Stands in for the nucleus: it calls no model and no tool."""',
    "",
    "    async def generate_content_async(self, llm_request, stream=False):",
    '        text = types.Content(role="model", parts=[types.Part(text="offline")])',
    "        yield LlmResponse(content=text)",
)
_SCRIPTED_NUCLEUS = (
    "def scripted_nucleus() -> BaseLlm:",
    '    """Stand in for the nucleus: repeat its replies, call no model and no tool."""',
    "    replies = nucleus_replies()",
    "",
    "    class ScriptedNucleus(BaseLlm):",
    "        async def generate_content_async(self, llm_request, stream=False):",
    "            reply = types.Part(text=next(replies))",
    '            yield LlmResponse(content=types.Content(role="model", parts=[reply]))',
    "",
    '    return ScriptedNucleus(model="scripted")',
)


def _reply(events: Iterable[Event]) -> str:
    """Read the reply of a turn from its events.

    Returns:
        The text of the last whole event that has one, without thoughts.
    """
    for event in reversed(list(events)):
        parts = event.content.parts if event.content is not None and event.content.parts else []
        text = "".join(part.text for part in parts if part.text and not part.thought)
        if text and not event.partial:
            return text
    return ""


async def run_turn(agent: Any, message: str, conversation: Any) -> RunOutcome:
    """Run one turn on a copy of the conversation's session, kept in memory.

    ADK keeps the message of a turn that fails in the session it ran on. The
    copy leaves the session of the turn before as it was, whatever this turn
    does. Sessions, artifacts and memory are ADK's in-memory services: the
    developer's own stores belong to their runner and are never reached.

    Args:
        agent: The `google.adk.apps.App` built from the studio's copy of the atom.
        message: The user message.
        conversation: The ADK `Session` of the conversation, or `None` in the first turn.

    Returns:
        The reply and the session after the turn, which holds blocked messages
        already redacted and keeps the id of the conversation.

    Raises:
        InterruptedTurnError: If the run stopped to wait for an answer, as a
            long-running tool or a request for a confirmation makes it do.
            The conversation that was given is as it was.
    """
    runner = InMemoryRunner(app=agent)
    service = runner.session_service
    session = await service.create_session(
        app_name=runner.app_name,
        user_id=_USER,
        session_id=conversation.id if conversation is not None else uuid4().hex,
    )
    for event in conversation.events if conversation is not None else ():
        await service.append_event(session, event.model_copy(deep=True))
    content = types.Content(role="user", parts=[types.Part(text=message)])
    turn = runner.run_async(user_id=_USER, session_id=session.id, new_message=content)
    events = [event async for event in turn]
    # A request for a confirmation is not the last event of its run.
    if any(event.long_running_tool_ids for event in events):
        raise InterruptedTurnError(_INTERRUPTED)
    after = await service.get_session(
        app_name=runner.app_name, user_id=_USER, session_id=session.id
    )
    return RunOutcome(reply=_reply(events), conversation=after)


def describe(connector: Connector[Any], agent: Any) -> dict[str, str]:
    """Name the connector and its model for the atom card.

    Args:
        connector: An `ADKConnector`.
        agent: The native app; not read.

    Returns:
        `{"connector": ..., "model": ...}`.
    """
    model = getattr(connector, "model", None)
    name = model if isinstance(model, str) else getattr(model, "model", None)
    if not isinstance(name, str) or not name:
        name = type(model).__name__
    return {"connector": type(connector).__name__, "model": name}


def notes(connector: Connector[Any]) -> list[str]:
    """Say what the developer should know before trying this atom in the studio.

    Args:
        connector: An `ADKConnector`.

    Returns:
        One sentence when the atom has plugins of the developer's.
    """
    plugins = getattr(connector, "app_kwargs", {}).get("plugins") or ()
    if not plugins:
        return []
    names = ", ".join(str(getattr(plugin, "name", type(plugin).__name__)) for plugin in plugins)
    return [_PLUGINS.format(names=names)]


def saved_test(connector: Connector[Any]) -> Framework:
    """Return what a saved test needs from Google ADK.

    Args:
        connector: An `ADKConnector`; with callbacks of the developer's the
            stand-ins take a note, and with plugins every test does.

    Returns:
        The awaited call that runs a turn on ADK's in-memory runner, which keeps
        the conversation, the reply, and the models the test defines to stand in
        for the nucleus, put on the connector before the atom is built: one
        offline, one that repeats its replies.
    """
    kwargs = getattr(connector, "agent_kwargs", {})
    callbacks = (_CALLBACKS,) if any(key.endswith("_callback") for key in kwargs) else ()
    plugins = getattr(connector, "app_kwargs", {}).get("plugins")
    turn = Call(arguments=("quiet=True",))
    return Framework(
        imports=("from google.adk.runners import InMemoryRunner",),
        run="await runner.run_debug",
        first=turn,
        later=turn,
        reply="result[-1].content.parts[0].text",
        setup=("runner = InMemoryRunner(app=agent)",),
        marker='@pytest.mark.asyncio(loop_scope="session")',
        notes=(_ASYNC, *((_PLUGINS_RUN,) if plugins else ())),
        offline=StandIn(
            label="a model of its own, `OfflineNucleus`,",
            imports=_MODEL_CLASSES,
            definitions=_OFFLINE_NUCLEUS,
            comment="# No model is called in this test: OfflineNucleus keeps the nucleus offline.",
            before_build=(
                'monkeypatch.setattr(atom.connector, "model", OfflineNucleus(model="offline"))',
            ),
            notes=(_OWN_CLASS.format(name="OfflineNucleus"), *callbacks),
        ),
        scripted=StandIn(
            imports=_MODEL_CLASSES,
            definitions=_SCRIPTED_NUCLEUS,
            before_build=('monkeypatch.setattr(atom.connector, "model", scripted_nucleus())',),
            notes=(_OWN_CLASS.format(name="ScriptedNucleus"), *callbacks),
        ),
    )
