import importlib
import importlib.util
import sys
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

import chatomon.connectors.langchain as langchain_connector
from chatomon import Atom, ChatomonConfigError, MissingExtraError, TurnReport, Verdict
from chatomon.connectors.langchain import LangChainConnector
from chatomon.orbitals import JudgeDecision
from tests.contract.harness_langchain import ScriptedChatModel
from tests.fakes import ScriptedOrbital


@pytest.mark.parametrize("key", ["system_prompt", "response_format", "name"])
def test_reserved_arguments_are_rejected(key):
    with pytest.raises(ChatomonConfigError, match=key):
        kwargs: dict[str, Any] = {key: "x"}
        LangChainConnector("openai:gpt-5", **kwargs)


def test_build_returns_a_compiled_graph():
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hi!")]))
    agent = Atom(
        name="speedypack", instructions="Be brief.", connector=LangChainConnector(model)
    ).build()
    assert isinstance(agent, CompiledStateGraph)
    state = agent.invoke({"messages": [HumanMessage(content="hello")]})
    assert state["messages"][-1].content == "Hi!"


async def _stream_text(agent: Any, message: str) -> list[str]:
    chunks: list[str] = []
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(content=message)]}, stream_mode="messages"
    ):
        if isinstance(chunk.content, str) and chunk.content:
            chunks.append(chunk.content)
    return chunks


async def test_streaming_with_output_orbitals_emits_only_checked_text():
    judge = ScriptedOrbital(Verdict.block(guidance="Fix the hours."), Verdict.allow(), name="judge")
    model = ScriptedChatModel(script=["Open 24/7 always", "Open 9-18 weekdays"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model),
        output_orbitals=[judge],
    ).build()
    streamed = "".join(await _stream_text(agent, "hours?"))
    assert "24/7" not in streamed
    assert "9-18" in streamed


async def test_streaming_without_output_orbitals_keeps_token_streaming():
    model = ScriptedChatModel(script=["Open 9-18 weekdays"])
    agent = Atom(
        name="speedypack", instructions="Be brief.", connector=LangChainConnector(model)
    ).build()
    chunks = await _stream_text(agent, "hours?")
    assert len(chunks) > 1


async def test_sync_invoke_inside_a_running_loop():
    reports: list[TurnReport] = []
    model = ScriptedChatModel(script=["Hi!"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model),
        input_orbitals=[ScriptedOrbital(name="in")],
        output_orbitals=[ScriptedOrbital(name="out")],
        on_turn=reports.append,
    ).build()
    state = agent.invoke({"messages": [HumanMessage(content="hello")]})
    assert state["messages"][-1].content == "Hi!"
    assert reports[0].final_action == "answered"


async def test_parent_span_resolver_nests_chatomon_spans(spans):
    parent = spans.tracer.start_span("app.request")
    model = ScriptedChatModel(script=["Hi!"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model, parent_span=lambda: parent),
        input_orbitals=[ScriptedOrbital(name="in")],
        tracer_provider=spans.provider,
    ).build()
    await agent.ainvoke({"messages": [HumanMessage(content="hello")]})
    parent.end()
    chain = next(s for s in spans.exporter.get_finished_spans() if s.name == "chatomon.input")
    assert chain.parent is not None
    assert chain.parent.span_id == parent.get_span_context().span_id


def test_model_string_is_initialised_to_tag_it_nostream(monkeypatch):
    created: list[str] = []

    def fake_init(model_id: str) -> ScriptedChatModel:
        created.append(model_id)
        return ScriptedChatModel(script=["Hi!"])

    monkeypatch.setattr(langchain_connector, "init_chat_model", fake_init)
    Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector("openai:gpt-5"),
        output_orbitals=[ScriptedOrbital(name="out")],
    ).build()
    assert created == ["openai:gpt-5"]


async def test_generate_structured_uses_resolve_model():
    judge = ScriptedChatModel(structured=[{"action": "allow", "reason": "ok"}])
    connector = LangChainConnector("unused", resolve_model=lambda _model_id: judge)
    decision = await connector.generate_structured(
        "judge-model", system="Judge.", prompt="Text.", schema=JudgeDecision
    )
    assert decision.action == "allow"


def test_missing_langchain_is_reported_as_missing_extra(monkeypatch):
    monkeypatch.delitem(sys.modules, "chatomon.connectors.langchain", raising=False)
    monkeypatch.setitem(sys.modules, "langchain.agents", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "langchain" else real_find_spec(name, *args),
    )
    with pytest.raises(MissingExtraError, match="chatomon\\[langchain\\]"):
        importlib.import_module("chatomon.connectors.langchain")
