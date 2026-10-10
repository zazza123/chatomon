import importlib
import importlib.util
import sys
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

import chatomon.connectors.langchain as langchain_connector
from chatomon import Atom, ChatomonConfigError, MissingExtraError, TurnReport, Verdict
from chatomon.connectors.langchain import LangChainConnector
from chatomon.orbitals import JudgeDecision
from chatomon.schemas import Message
from tests.contract.harness import lookup
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


def closed(reports: list[TurnReport]) -> list[tuple[str, str | None]]:
    return [(report.final_action, report.error_type) for report in reports]


def test_an_empty_reply_closes_the_turn_as_aborted_with_the_sync_api():
    reports: list[TurnReport] = []
    out = ScriptedOrbital(name="out")
    model = GenericFakeChatModel(messages=iter([AIMessage(content="")]))
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model),
        output_orbitals=[out],
        on_turn=reports.append,
    ).build()
    state = agent.invoke({"messages": [HumanMessage(content="hello")]})
    assert state["messages"][-1].content == ""
    assert out.contexts == []
    assert closed(reports) == [("aborted", "RuntimeError")]


async def test_a_reply_of_content_blocks_with_no_text_is_an_empty_reply():
    reports: list[TurnReport] = []
    out = ScriptedOrbital(name="out")
    thought = AIMessage(content=[{"type": "reasoning", "reasoning": "Nothing to say."}])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(ScriptedChatModel(script=[thought])),
        output_orbitals=[out],
        on_turn=reports.append,
    ).build()
    await agent.ainvoke({"messages": [HumanMessage(content="hello")]})
    assert out.contexts == []
    assert closed(reports) == [("aborted", "RuntimeError")]


async def test_the_turn_after_an_empty_reply_is_answered():
    reports: list[TurnReport] = []
    out = ScriptedOrbital(name="out")
    model = ScriptedChatModel(script=["", "Back."])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model, checkpointer=InMemorySaver()),
        output_orbitals=[out],
        on_turn=reports.append,
    ).build()
    config: RunnableConfig = {"configurable": {"thread_id": "t-empty"}}
    await agent.ainvoke({"messages": [HumanMessage(content="hello")]}, config)
    state = await agent.ainvoke({"messages": [HumanMessage(content="again")]}, config)
    assert state["messages"][-1].content == "Back."
    assert [report.final_action for report in reports] == ["aborted", "answered"]
    # The empty reply is not a message of the conversation the orbitals are shown.
    assert out.contexts[0].history == (
        Message(role="user", content="hello"),
        Message(role="user", content="again"),
    )


class _AskAgainWhenEmpty(AgentMiddleware):
    """A middleware of the developer's that sends an empty reply back to the model."""

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return None if state["messages"][-1].content else {"jump_to": "model"}


async def test_a_reply_written_after_an_empty_one_in_the_same_turn_is_checked():
    reports: list[TurnReport] = []
    out = ScriptedOrbital(name="out")
    model = ScriptedChatModel(script=["", "Hi!"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model, middleware=[_AskAgainWhenEmpty()]),
        output_orbitals=[out],
        on_turn=reports.append,
        capture_content="full",
    ).build()
    state = await agent.ainvoke({"messages": [HumanMessage(content="hello")]})
    assert state["messages"][-1].content == "Hi!"
    # The empty reply left the turn open: it is one turn, with one attempt.
    assert [report.final_action for report in reports] == ["answered"]
    assert [(a.nucleus_reply, a.decision) for a in reports[0].output] == [("Hi!", "accept")]


@tool
def approve_hours() -> str:
    """Ask a human to approve before looking up the opening hours."""
    answer = interrupt("approve?")
    return f"approved: {answer}"


@tool
def broken_lookup() -> str:
    """A tool that always fails."""
    raise ValueError("tool down")


def calling(name: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": f"call-{name}"}])


TOOLS = [approve_hours, tool(lookup), broken_lookup]
PAUSED: RunnableConfig = {"configurable": {"thread_id": "t-plain"}}


async def paused_without_chatomon() -> InMemorySaver:
    """Pause a thread in a graph built without chatomon: its state carries no turn."""
    checkpointer = InMemorySaver()
    model = ScriptedChatModel(script=[calling("approve_hours")])
    plain = create_agent(model, tools=TOOLS, checkpointer=checkpointer)
    await plain.ainvoke({"messages": [HumanMessage(content="hours?")]}, PAUSED)
    return checkpointer


async def test_a_turn_rebuilt_in_the_model_call_lasts_until_the_graph_ends():
    reports: list[TurnReport] = []
    seen = ScriptedOrbital(name="in")
    out = ScriptedOrbital(name="out")
    model = ScriptedChatModel(script=[calling("lookup"), "Open 9-18"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(
            model, tools=TOOLS, checkpointer=await paused_without_chatomon()
        ),
        input_orbitals=[seen],
        output_orbitals=[out],
        on_turn=reports.append,
    ).build()
    state = await agent.ainvoke(Command(resume="yes"), PAUSED)
    assert state["messages"][-1].content == "Open 9-18"
    # One turn for the two model calls: the input chain ran once and there is one report.
    assert [ctx.message for ctx in seen.contexts] == ["hours?"]
    assert [ctx.message for ctx in out.contexts] == ["Open 9-18"]
    assert closed(reports) == [("answered", None)]
    assert agent.get_state(PAUSED).values.get("chatomon_turn") is None


async def test_a_turn_rebuilt_in_the_model_call_lasts_with_the_sync_api_too():
    reports: list[TurnReport] = []
    seen = ScriptedOrbital(name="in")
    model = ScriptedChatModel(script=[calling("lookup"), "Open 9-18"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(
            model, tools=TOOLS, checkpointer=await paused_without_chatomon()
        ),
        input_orbitals=[seen],
        on_turn=reports.append,
    ).build()
    state = agent.invoke(Command(resume="yes"), PAUSED)
    assert state["messages"][-1].content == "Open 9-18"
    assert len(seen.contexts) == 1
    assert closed(reports) == [("answered", None)]


async def test_a_tool_that_fails_aborts_a_turn_rebuilt_in_the_model_call():
    reports: list[TurnReport] = []
    model = ScriptedChatModel(script=[calling("broken_lookup")])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(
            model, tools=TOOLS, checkpointer=await paused_without_chatomon()
        ),
        on_turn=reports.append,
    ).build()
    with pytest.raises(ValueError, match="tool down"):
        await agent.ainvoke(Command(resume="yes"), PAUSED)
    assert closed(reports) == [("aborted", "ValueError")]


async def test_an_interrupt_in_the_model_call_closes_a_turn_rebuilt_there():
    """The state cannot hold the turn of a model call that did not end: it is rebuilt again."""
    reports: list[TurnReport] = []
    seen = ScriptedOrbital(name="in")
    model = ScriptedChatModel(script=[GraphInterrupt(()), "Open 9-18"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(
            model, tools=TOOLS, checkpointer=await paused_without_chatomon()
        ),
        input_orbitals=[seen],
        on_turn=reports.append,
    ).build()
    await agent.ainvoke(Command(resume="yes"), PAUSED)
    assert closed(reports) == [("aborted", "GraphInterrupt")]
    state = await agent.ainvoke(Command(resume="go"), PAUSED)
    assert state["messages"][-1].content == "Open 9-18"
    assert closed(reports) == [("aborted", "GraphInterrupt"), ("answered", None)]
    assert [ctx.message for ctx in seen.contexts] == ["hours?", "hours?"]


async def test_an_empty_reply_closes_a_turn_rebuilt_in_the_model_call():
    reports: list[TurnReport] = []
    out = ScriptedOrbital(name="out")
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(
            ScriptedChatModel(script=[""]),
            tools=TOOLS,
            checkpointer=await paused_without_chatomon(),
        ),
        output_orbitals=[out],
        on_turn=reports.append,
    ).build()
    state = await agent.ainvoke(Command(resume="yes"), PAUSED)
    assert state["messages"][-1].content == ""
    assert out.contexts == []
    assert closed(reports) == [("aborted", "RuntimeError")]


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
