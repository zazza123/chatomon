from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from chatomon import Atom, ChatomonConfigError, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER
from chatomon.studio._runners import check_supported
from chatomon.studio._runners.pydantic_ai import describe, run_turn
from tests.fakes import FakeConnector, ScriptedOrbital


def user_texts(history: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in history
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def test_pydantic_ai_connectors_are_supported():
    check_supported(PydanticAIConnector(TestModel()))


def test_other_connectors_are_refused_by_name():
    with pytest.raises(
        ChatomonConfigError, match="supports PydanticAI atoms for now; this atom uses FakeConnector"
    ):
        check_supported(FakeConnector())


async def test_a_turn_returns_the_reply_and_the_history():
    seen: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(len(messages))
        return ModelResponse(parts=[TextPart(content="Hi")])

    agent = Atom(
        name="t", instructions="Be brief.", connector=PydanticAIConnector(FunctionModel(respond))
    ).build()
    first = await run_turn(agent, "hello", [])
    second = await run_turn(agent, "again", first.history)
    assert first.reply == "Hi"
    assert user_texts(second.history) == ["hello", "again"]
    assert seen[1] > seen[0]


async def test_a_blocked_message_stays_redacted_in_the_history():
    agent = Atom(
        name="t",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(custom_output_text="ok")),
        input_orbitals=[ScriptedOrbital(Verdict.block("No."), name="guard")],
    ).build()
    outcome = await run_turn(agent, "secret", [])
    assert outcome.reply == "No."
    assert user_texts(outcome.history) == [DEFAULT_BLOCKED_PLACEHOLDER]


@pytest.mark.parametrize(
    ("model", "expected"),
    [("openai:gpt-5", "openai:gpt-5"), (TestModel(model_name="tm"), "tm")],
)
def test_describe_names_the_connector_and_the_model(model: Any, expected: str):
    assert describe(PydanticAIConnector(model)) == {
        "connector": "PydanticAIConnector",
        "model": expected,
    }
