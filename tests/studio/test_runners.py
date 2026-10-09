"""What every studio runner does, and how the studio finds the runner of a connector."""

import asyncio
import importlib
import pkgutil
from typing import Any

import pytest

import chatomon.connectors
from chatomon import Atom, ChatomonConfigError, Verdict
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER
from chatomon.studio._runners import History, runner_for
from chatomon.studio._testgen import Framework
from tests.contract.harness import ToolCall, lookup
from tests.fakes import FakeConnector, ScriptedOrbital
from tests.studio.frameworks import Reply, StudioHarness


def make_atom(framework: StudioHarness, *replies: Reply, **kwargs: Any) -> Atom:
    return Atom(
        name="t", instructions="Be brief.", connector=framework.connector(*replies), **kwargs
    )


def seen(guard: ScriptedOrbital, turn: int) -> list[tuple[str, str]]:
    """The conversation an input orbital was given in a turn, counted from 0."""
    return [(message.role, message.content) for message in guard.contexts[turn].history]


async def test_the_second_turn_sees_the_first(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, "one", "two", input_orbitals=[guard]).build()
    first = await framework.runner.run_turn(agent, "hello", None)
    second = await framework.runner.run_turn(agent, "again", first.conversation)
    assert (first.reply, second.reply) == ("one", "two")
    assert seen(guard, 1) == [("user", "hello"), ("assistant", "one")]


async def test_the_same_reply_twice_is_kept_twice(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, "same", "same", "other", input_orbitals=[guard]).build()
    conversation = None
    for message in ("one", "two", "three"):
        conversation = (await framework.runner.run_turn(agent, message, conversation)).conversation
    assert seen(guard, 2) == [
        ("user", "one"),
        ("assistant", "same"),
        ("user", "two"),
        ("assistant", "same"),
    ]


async def test_a_blocked_message_stays_the_placeholder(framework: StudioHarness):
    guard = ScriptedOrbital(Verdict.block("No."), Verdict.allow(), name="guard")
    agent = make_atom(framework, "two", input_orbitals=[guard]).build()
    first = await framework.runner.run_turn(agent, "secret", None)
    assert first.reply == "No."
    await framework.runner.run_turn(agent, "again", first.conversation)
    assert seen(guard, 1) == [("user", DEFAULT_BLOCKED_PLACEHOLDER), ("assistant", "No.")]


async def test_a_turn_whose_model_fails_leaves_the_conversation_as_it_was(
    framework: StudioHarness,
):
    guard = ScriptedOrbital(name="guard")
    atom = make_atom(framework, "one", RuntimeError("model down"), "three", input_orbitals=[guard])
    agent = atom.build()
    first = await framework.runner.run_turn(agent, "hello", None)
    with pytest.raises(RuntimeError, match="model down"):
        await framework.runner.run_turn(agent, "boom", first.conversation)
    third = await framework.runner.run_turn(agent, "again", first.conversation)
    assert third.reply == "three"
    assert seen(guard, 2) == [("user", "hello"), ("assistant", "one")]


async def test_a_first_turn_that_fails_leaves_no_conversation(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, RuntimeError("model down"), "two", input_orbitals=[guard]).build()
    with pytest.raises(RuntimeError, match="model down"):
        await framework.runner.run_turn(agent, "boom", None)
    assert (await framework.runner.run_turn(agent, "hello", None)).reply == "two"
    assert seen(guard, 1) == []


async def test_a_turn_never_changes_the_conversation_it_was_given(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, "one", "two", "three", input_orbitals=[guard]).build()
    first = await framework.runner.run_turn(agent, "hello", None)
    await framework.runner.run_turn(agent, "second", first.conversation)
    # Once more from the conversation of the first turn: the second turn is not in it.
    await framework.runner.run_turn(agent, "third", first.conversation)
    assert seen(guard, 2) == [("user", "hello"), ("assistant", "one")]


async def test_a_turn_in_which_the_nucleus_calls_a_tool_stays_in_the_conversation(
    framework: StudioHarness,
):
    guard = ScriptedOrbital(name="guard")
    connector = framework.connector(ToolCall(), "We open at 9.", "two", tools=[lookup])
    atom = Atom(name="t", instructions="Be brief.", connector=connector, input_orbitals=[guard])
    agent = atom.build()
    first = await framework.runner.run_turn(agent, "when do you open?", None)
    assert first.reply == "We open at 9."
    second = await framework.runner.run_turn(agent, "again", first.conversation)
    assert second.reply == "two"
    # An orbital is shown what was said, not the exchange with the tool.
    assert seen(guard, 1) == [("user", "when do you open?"), ("assistant", "We open at 9.")]


async def test_the_conversation_survives_an_agent_built_again(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    atom = make_atom(framework, "one", "two", input_orbitals=[guard])
    first = await framework.runner.run_turn(atom.build(), "hello", None)
    await framework.runner.run_turn(atom.build(), "again", first.conversation)
    assert seen(guard, 1) == [("user", "hello"), ("assistant", "one")]


async def test_the_conversation_goes_on_with_an_agent_of_another_instance_of_the_connector(
    framework: StudioHarness,
):
    """What a reload from code does: new objects, the same framework."""
    before = make_atom(framework, "one", input_orbitals=[ScriptedOrbital(name="guard")])
    first = await framework.runner.run_turn(before.build(), "hello", None)
    guard = ScriptedOrbital(name="guard")
    after = make_atom(framework, "two", input_orbitals=[guard])
    second = await framework.runner.run_turn(after.build(), "again", first.conversation)
    assert (first.reply, second.reply) == ("one", "two")
    assert seen(guard, 0) == [("user", "hello"), ("assistant", "one")]


PAST: History = (("user", "hello"), ("assistant", "one"))


async def test_a_turn_runs_on_a_history_given_as_texts(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, "two", input_orbitals=[guard]).build()
    assert await framework.runner.run_from_history(agent, PAST, "again") == "two"
    assert guard.contexts[0].message == "again"
    assert seen(guard, 0) == [("user", "hello"), ("assistant", "one")]


async def test_a_turn_without_a_history_is_a_first_turn(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, "one", input_orbitals=[guard]).build()
    assert await framework.runner.run_from_history(agent, (), "hello") == "one"
    assert seen(guard, 0) == []


async def test_the_history_reaches_the_output_orbitals_too(framework: StudioHarness):
    judge = ScriptedOrbital(name="judge")
    agent = make_atom(framework, "two", output_orbitals=[judge]).build()
    await framework.runner.run_from_history(agent, PAST, "again")
    assert seen(judge, 0)[:2] == [("user", "hello"), ("assistant", "one")]


async def test_a_message_an_orbital_blocks_gets_the_orbitals_reply(framework: StudioHarness):
    guard = ScriptedOrbital(Verdict.block("No."), name="guard")
    agent = make_atom(framework, "unused", input_orbitals=[guard]).build()
    assert await framework.runner.run_from_history(agent, PAST, "secret") == "No."


async def test_a_history_that_repeats_itself_is_kept_whole(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, "three", input_orbitals=[guard]).build()
    history: History = [
        ("user", "hi"),
        ("assistant", "same"),
        ("user", "hi"),
        ("assistant", "same"),
    ]
    assert await framework.runner.run_from_history(agent, history, "hi") == "three"
    assert seen(guard, 0) == history


@pytest.mark.parametrize(
    "history",
    [
        [("user", "a"), ("user", "b")],
        [("assistant", "welcome"), ("user", "a"), ("assistant", "b")],
        [("user", "a"), ("assistant", "b"), ("user", "c")],
        [("user", "ciao 👻 «è»"), ("assistant", "línea\nnueva")],
    ],
    ids=["two user messages", "the assistant first", "a user message last", "not ASCII"],
)
async def test_a_history_is_given_in_its_order_whoever_speaks(
    framework: StudioHarness, history: History
):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, "ok", input_orbitals=[guard]).build()
    assert await framework.runner.run_from_history(agent, history, "last") == "ok"
    assert seen(guard, 0) == history


async def test_a_history_is_left_as_it_was_given(framework: StudioHarness):
    history: History = [("user", "hello"), ("assistant", "one")]
    agent = make_atom(framework, "two").build()
    await framework.runner.run_from_history(agent, history, "again")
    assert history == [("user", "hello"), ("assistant", "one")]


async def test_a_model_that_fails_raises_its_error_and_the_next_turn_runs(
    framework: StudioHarness,
):
    agent = make_atom(framework, RuntimeError("model down"), "two").build()
    with pytest.raises(RuntimeError, match="model down"):
        await framework.runner.run_from_history(agent, PAST, "boom")
    assert await framework.runner.run_from_history(agent, PAST, "again") == "two"


async def test_a_turn_from_a_history_can_call_a_tool(framework: StudioHarness):
    connector = framework.connector(ToolCall(), "We open at 9.", tools=[lookup])
    agent = Atom(name="t", instructions="Be brief.", connector=connector).build()
    assert await framework.runner.run_from_history(agent, PAST, "when?") == "We open at 9."


async def test_turns_at_the_same_time_keep_their_own_histories(framework: StudioHarness):
    guard = ScriptedOrbital(name="guard")
    agent = make_atom(framework, "a", "b", "c", input_orbitals=[guard]).build()
    histories: list[History] = [[("user", f"q{n}"), ("assistant", f"r{n}")] for n in range(3)]
    replies = await asyncio.gather(
        *(
            framework.runner.run_from_history(agent, past, f"m{n}")
            for n, past in enumerate(histories)
        )
    )
    assert sorted(replies) == ["a", "b", "c"]
    by_message = {ctx.message: [(m.role, m.content) for m in ctx.history] for ctx in guard.contexts}
    assert by_message == {f"m{n}": histories[n] for n in range(3)}


def test_a_runner_names_the_connector_and_supplies_a_saved_test(framework: StudioHarness):
    atom = make_atom(framework, "one")
    described = framework.runner.describe(atom.connector, atom.build())
    assert set(described) == {"connector", "model"}
    assert described["connector"] == type(atom.connector).__name__
    assert described["model"]
    assert isinstance(framework.runner.saved_test(atom.connector), Framework)
    assert all(isinstance(note, str) for note in framework.runner.notes(atom.connector))


def test_each_connector_gets_the_runner_of_its_framework(framework: StudioHarness):
    assert runner_for(framework.connector("one")) is framework.runner


def test_a_subclass_of_the_developers_is_run_like_the_connector_it_extends(
    framework: StudioHarness,
):
    connector = framework.connector("one")
    connector.__class__ = type("Mine", (type(connector),), {"__module__": "shop.connectors"})
    assert type(connector).__module__ == "shop.connectors"
    assert runner_for(connector) is framework.runner


def test_a_class_without_a_module_among_the_bases_is_no_trouble():
    class Nowhere(FakeConnector):
        pass

    setattr(Nowhere, "__module__", None)  # as a class made at run time can have  # noqa: B010
    with pytest.raises(ChatomonConfigError, match="the studio has no runner for Nowhere"):
        runner_for(Nowhere())


def test_a_connector_of_the_developers_own_is_refused_by_name():
    message = (
        "the studio has no runner for FakeConnector: "
        "it opens atoms built with chatomon's own connectors"
    )
    with pytest.raises(ChatomonConfigError, match=message):
        runner_for(FakeConnector())


def test_every_connector_module_has_a_runner_with_the_five_functions():
    frameworks = [m.name for m in pkgutil.iter_modules(chatomon.connectors.__path__)]
    frameworks.remove("base")
    assert frameworks  # the search itself must find chatomon's connectors
    for name in frameworks:
        runner = importlib.import_module(f"chatomon.studio._runners.{name}")
        for function in ("run_turn", "run_from_history", "describe", "notes", "saved_test"):
            assert callable(getattr(runner, function, None)), f"{name} has no {function}()"
