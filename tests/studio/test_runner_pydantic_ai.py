"""What the PydanticAI runner adds to the contract every runner keeps."""

from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._runners.pydantic_ai import describe, notes, saved_test


@pytest.mark.parametrize(
    ("model", "expected"),
    [("openai:gpt-5", "openai:gpt-5"), (TestModel(model_name="tm"), "tm")],
)
def test_describe_names_the_connector_and_the_model(model: Any, expected: str):
    assert describe(PydanticAIConnector(model), None) == {
        "connector": "PydanticAIConnector",
        "model": expected,
    }


def test_an_atom_that_declares_deps_gets_a_note():
    assert notes(PydanticAIConnector(TestModel())) == []
    assert notes(PydanticAIConnector(TestModel(), deps_type=str)) == [
        "this atom declares deps_type; the studio runs turns without deps, "
        "so tools and instructions that use them fail"
    ]


def test_the_runner_supplies_pydantic_ais_part_of_a_saved_test():
    lines = saved_test(PydanticAIConnector(TestModel()))
    assert lines.run == "agent.run_sync"
    assert not lines.awaited
    assert (lines.first.before, lines.first.after, lines.first.arguments) == ("", "", ())
    assert lines.later.arguments == ("message_history=result.all_messages()",)
    assert lines.reply == "result.output"
    assert lines.imports == lines.setup == ()
    assert lines.marker == ""
    assert lines.offline.label == "PydanticAI's `TestModel`"
    assert lines.offline.imports == ("from pydantic_ai.models.test import TestModel",)
    assert lines.offline.before_build == (
        'monkeypatch.setattr(atom.connector, "model", TestModel(call_tools=[]))',
    )
    assert lines.offline.after_build == lines.offline.definitions == ()
