"""What every studio runner does, and how the studio finds the runner of a connector."""

import importlib
import pkgutil
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

import chatomon.connectors
from chatomon import Atom, ChatomonConfigError, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER
from chatomon.studio._runners import pydantic_ai as pydantic_ai_runner
from chatomon.studio._runners import runner_for
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


def test_a_subclass_of_the_developers_is_run_like_the_connector_it_extends():
    class Mine(PydanticAIConnector):
        pass

    assert runner_for(Mine(TestModel())) is pydantic_ai_runner


def test_a_connector_of_the_developers_own_is_refused_by_name():
    message = (
        "the studio has no runner for FakeConnector: "
        "it opens atoms built with chatomon's own connectors"
    )
    with pytest.raises(ChatomonConfigError, match=message):
        runner_for(FakeConnector())


def test_every_connector_module_has_a_runner_with_the_four_functions():
    frameworks = [m.name for m in pkgutil.iter_modules(chatomon.connectors.__path__)]
    frameworks.remove("base")
    assert frameworks  # the search itself must find chatomon's connectors
    for name in frameworks:
        runner = importlib.import_module(f"chatomon.studio._runners.{name}")
        for function in ("run_turn", "describe", "notes", "saved_test"):
            assert callable(getattr(runner, function, None)), f"{name} has no {function}()"
