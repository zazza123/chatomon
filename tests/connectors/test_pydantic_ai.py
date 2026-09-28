import importlib
import importlib.util
import sys
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from chatomon import Atom, ChatomonConfigError, MissingExtraError, TurnReport, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import JudgeDecision
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER
from tests.fakes import ScriptedOrbital


@pytest.mark.parametrize("key", ["instructions", "system_prompt", "output_type", "name"])
def test_reserved_arguments_are_rejected(key):
    with pytest.raises(ChatomonConfigError, match=key):
        kwargs: dict[str, Any] = {key: "x"}
        PydanticAIConnector("test", **kwargs)


def test_build_returns_a_native_agent():
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(custom_output_text="Hi!")),
    ).build()
    assert isinstance(agent, Agent)
    assert agent.name == "speedypack"
    assert agent.run_sync("hello").output == "Hi!"


def test_user_capabilities_run_inside_chatomon():
    seen: list[str] = []

    class Recorder(AbstractCapability[Any]):
        async def before_model_request(self, ctx: Any, request_context: Any) -> Any:
            last = request_context.messages[-1]
            seen.extend(part.content for part in last.parts if hasattr(part, "content"))
            return request_context

    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(
            TestModel(custom_output_text="ok"), capabilities=[Recorder()]
        ),
        input_orbitals=[ScriptedOrbital(Verdict.block(guidance="Decline."), name="guard")],
    ).build()
    agent.run_sync("secret request")
    assert seen == [DEFAULT_BLOCKED_PLACEHOLDER]


def _streaming_model(calls: list[str]) -> FunctionModel:
    def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        calls.append("request")
        return ModelResponse(parts=[TextPart(content="Hi there")])

    async def stream(messages: list[Any], info: AgentInfo):
        calls.append("stream")
        for token in ["Hi ", "there"]:
            yield token

    return FunctionModel(respond, stream_function=stream)


async def test_streaming_with_output_orbitals_is_refused_before_the_model():
    calls: list[str] = []
    reports: list[TurnReport] = []
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(_streaming_model(calls)),
        output_orbitals=[ScriptedOrbital(name="out")],
        on_turn=reports.append,
    ).build()
    with pytest.raises(ChatomonConfigError, match="stream"):
        async with agent.run_stream("hello") as result:
            await result.get_output()
    assert calls == []
    assert reports[0].final_action == "aborted"


async def test_streaming_with_only_input_orbitals_works():
    calls: list[str] = []
    reports: list[TurnReport] = []
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(_streaming_model(calls)),
        input_orbitals=[ScriptedOrbital(name="in")],
        on_turn=reports.append,
    ).build()
    async with agent.run_stream("hello") as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True)]
    assert "".join(chunks) == "Hi there"
    assert reports[0].final_action == "answered"


async def test_generate_structured_uses_resolve_model():
    connector = PydanticAIConnector(
        "unused",
        resolve_model=lambda model_id: TestModel(
            custom_output_args={"action": "allow", "reason": model_id}
        ),
    )
    decision = await connector.generate_structured(
        "judge-model", system="Judge.", prompt="Text.", schema=JudgeDecision
    )
    assert decision.action == "allow"
    assert decision.reason == "judge-model"


def test_missing_pydantic_ai_is_reported_as_missing_extra(monkeypatch):
    monkeypatch.delitem(sys.modules, "chatomon.connectors.pydantic_ai", raising=False)
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "pydantic_ai" else real_find_spec(name, *args),
    )
    with pytest.raises(MissingExtraError, match="chatomon\\[pydantic-ai\\]"):
        importlib.import_module("chatomon.connectors.pydantic_ai")
