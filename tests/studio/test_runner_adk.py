"""What the Google ADK runner adds to the contract every runner keeps."""

from typing import Any

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types

from chatomon import Atom
from chatomon.connectors.adk import ADKConnector
from chatomon.studio._runners import InterruptedTurnError
from chatomon.studio._runners.adk import describe, notes, run_from_history, run_turn, saved_test
from tests.contract.harness_adk import ScriptedLlm, said
from tests.fakes import ScriptedOrbital

PLUGINS = (
    "this atom has plugins of its own (audit, billing); they run on every turn of the "
    "studio too, and write where they usually write"
)
PLUGINS_RUN = (
    "This atom has plugins of its own: they run in this test too, and write where they "
    "usually write."
)
CALLBACKS = (
    "This atom has callbacks of its own: one that calls a model can still reach a real "
    "model in this test, and one that writes the reply changes what the test sees."
)


def built(connector: Any, **kwargs: Any) -> Any:
    return Atom(name="t", instructions="Be brief.", connector=connector, **kwargs).build()


def approve() -> None:
    """Ask a human; the answer comes later."""
    return None


def asking() -> LlmResponse:
    """A model step that calls the long-running tool."""
    call = types.FunctionCall(name="approve", args={}, id="call-approve")
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))


def test_describe_names_the_model_from_a_name_or_from_the_object():
    named = ADKConnector("gemini-3.5-flash")
    assert describe(named, None) == {"connector": "ADKConnector", "model": "gemini-3.5-flash"}
    scripted = ADKConnector(ScriptedLlm())
    assert describe(scripted, built(scripted))["model"] == "scripted"


def test_the_notes_name_the_plugins_of_the_developers():
    class Mine(BasePlugin):
        pass

    assert notes(ADKConnector(ScriptedLlm())) == []
    assert notes(ADKConnector(ScriptedLlm(), app_kwargs={"plugins": []})) == []
    plugins = [Mine(name="audit"), Mine(name="billing")]
    assert notes(ADKConnector(ScriptedLlm(), app_kwargs={"plugins": plugins})) == [PLUGINS]


async def test_a_run_that_waits_for_an_answer_is_a_failed_turn_and_the_conversation_goes_on():
    guard = ScriptedOrbital(name="guard")
    model = ScriptedLlm(script=["one", asking(), "three"])
    connector = ADKConnector(model, tools=[LongRunningFunctionTool(func=approve)])
    agent = built(connector, input_orbitals=[guard])
    first = await run_turn(agent, "hello", None)
    with pytest.raises(InterruptedTurnError, match="the turn is left unfinished"):
        await run_turn(agent, "may I?", first.conversation)
    assert (await run_turn(agent, "again", first.conversation)).reply == "three"
    # The request without an answer is not in the conversation the next turn runs on.
    assert [(m.role, m.content) for m in guard.contexts[2].history] == [
        ("user", "hello"),
        ("assistant", "one"),
    ]


async def test_a_run_that_waits_for_an_answer_stops_a_turn_run_from_a_history():
    model = ScriptedLlm(script=[asking(), "three"])
    agent = built(ADKConnector(model, tools=[LongRunningFunctionTool(func=approve)]))
    with pytest.raises(InterruptedTurnError, match="the turn is left unfinished"):
        await run_from_history(agent, [("user", "hello"), ("assistant", "one")], "may I?")


async def test_a_nucleus_that_says_nothing_from_a_history_gives_an_empty_text():
    agent = built(ADKConnector(ScriptedLlm(script=[""])))
    assert await run_from_history(agent, [("user", "hello"), ("assistant", "one")], "again") == ""


def pay(amount: int) -> str:
    """Pay an amount."""
    return f"paid {amount}"


async def test_a_request_for_a_confirmation_is_a_turn_that_waits_too():
    """ADK asks for a confirmation in an event that is not the last of the run."""
    call = types.FunctionCall(name="pay", args={"amount": 5}, id="call-pay")
    paying = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(function_call=call)])
    )
    model = ScriptedLlm(script=["one", paying, "three"])
    connector = ADKConnector(model, tools=[FunctionTool(pay, require_confirmation=True)])
    agent = built(connector)
    first = await run_turn(agent, "hello", None)
    with pytest.raises(InterruptedTurnError, match="the turn is left unfinished"):
        await run_turn(agent, "pay five", first.conversation)
    assert (await run_turn(agent, "again", first.conversation)).reply == "three"


async def test_the_copy_a_turn_runs_on_is_still_the_same_conversation():
    seen: list[str | None] = []
    connector = ADKConnector(ScriptedLlm(script=["one", "two"]))
    agent = built(connector, on_turn=lambda report: seen.append(report.conversation_id))
    first = await run_turn(agent, "hello", None)
    second = await run_turn(agent, "again", first.conversation)
    assert second.conversation is not first.conversation
    assert len(first.conversation.events) == 2  # the turn after it left it as it was
    assert seen[0] is not None
    assert seen[0] == seen[1] == second.conversation.id


async def test_the_reply_of_a_turn_has_no_thoughts():
    connector = ADKConnector(ScriptedLlm(script=[said("Hello!", thought="The user greets me.")]))
    assert (await run_turn(built(connector), "hi", None)).reply == "Hello!"


def test_a_saved_test_awaits_its_turns_on_adks_runner_and_swaps_the_connectors_model():
    lines = saved_test(ADKConnector(ScriptedLlm()))
    assert lines.run == "await runner.run_debug"
    assert lines.awaited
    assert not lines.carries  # the runner keeps the conversation
    assert lines.first == lines.later
    assert (lines.first.before, lines.first.after) == ("", "")
    assert lines.first.arguments == ("quiet=True",)
    assert lines.imports == ("from google.adk.runners import InMemoryRunner",)
    assert lines.setup == ("runner = InMemoryRunner(app=agent)",)
    assert lines.reply == "result[-1].content.parts[0].text"
    assert lines.marker == '@pytest.mark.asyncio(loop_scope="session")'
    # On the connector, before the build: the built agent's model is chatomon's own.
    for stand_in in (lines.offline, lines.scripted):
        assert stand_in.after_build == ()
        assert "atom.connector" in stand_in.before_build[-1]
        compile("\n".join([*stand_in.imports, "", *stand_in.definitions]), "x.py", "exec")
    assert lines.offline.definitions[0] == "class OfflineNucleus(BaseLlm):"
    assert lines.scripted.definitions[0] == "def scripted_nucleus() -> BaseLlm:"


def test_a_saved_test_of_an_atom_with_plugins_says_that_they_run():
    class Mine(BasePlugin):
        pass

    assert PLUGINS_RUN not in saved_test(ADKConnector(ScriptedLlm())).notes
    lines = saved_test(ADKConnector(ScriptedLlm(), app_kwargs={"plugins": [Mine(name="audit")]}))
    # In every test, with a stand-in or with the real nucleus: the plugins are the app's.
    assert lines.notes[-1] == PLUGINS_RUN


def test_a_saved_test_of_an_atom_with_callbacks_says_what_they_can_do():
    def before(callback_context: Any, llm_request: LlmRequest) -> None:
        return None

    plain = saved_test(ADKConnector(ScriptedLlm()))
    assert CALLBACKS not in (*plain.offline.notes, *plain.scripted.notes)
    lines = saved_test(ADKConnector(ScriptedLlm(), before_model_callback=before))
    assert lines.offline.notes[-1] == CALLBACKS
    assert lines.scripted.notes[-1] == CALLBACKS
