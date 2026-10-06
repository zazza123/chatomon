"""Regression tests for the Plan 2 whole-branch review findings."""

import asyncio
from typing import Any

import pytest
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom, TurnReport, Verdict
from chatomon.connectors.langchain import LangChainConnector
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals.base import Orbital
from chatomon.schemas import OrbitalContext
from tests.contract.harness import lookup
from tests.contract.harness_langchain import ScriptedChatModel
from tests.fakes import ScriptedOrbital


async def _stream_text(agent: Any, message: str, config: RunnableConfig | None = None) -> str:
    chunks: list[str] = []
    async for chunk, _meta in agent.astream(
        {"messages": [HumanMessage(content=message)]}, config, stream_mode="messages"
    ):
        if isinstance(chunk.content, str) and chunk.content:
            chunks.append(chunk.content)
    return "".join(chunks)


async def test_streaming_with_tools_emits_only_checked_text():
    judge = ScriptedOrbital(Verdict.block(guidance="Fix the hours."), Verdict.allow(), name="judge")
    model = ScriptedChatModel(script=["SECRET draft 24/7", "Open 9-18 weekdays"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model, tools=[tool(lookup)]),
        output_orbitals=[judge],
    ).build()
    streamed = await _stream_text(agent, "hours?")
    assert "SECRET" not in streamed
    assert "9-18" in streamed


@tool
def approve_hours() -> str:
    """Ask a human to approve before looking up the opening hours."""
    answer = interrupt("approve?")
    return f"approved: {answer}"


async def test_resuming_on_a_fresh_agent_still_runs_the_output_chain():
    checkpointer = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "t-resume"}}
    first_model = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[{"name": "approve_hours", "args": {}, "id": "c1"}])
        ]
    )
    first = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(first_model, tools=[approve_hours], checkpointer=checkpointer),
        output_orbitals=[ScriptedOrbital(name="out")],
    ).build()
    await first.ainvoke({"messages": [HumanMessage(content="hours?")]}, config)

    reports: list[TurnReport] = []
    blocker = ScriptedOrbital(Verdict.block("Blocked."), name="out")
    second_model = ScriptedChatModel(script=["UNCHECKED MODEL TEXT"])
    second = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(
            second_model, tools=[approve_hours], checkpointer=checkpointer
        ),
        output_orbitals=[blocker],
        on_turn=reports.append,
    ).build()
    state = await second.ainvoke(Command(resume="yes"), config)
    assert state["messages"][-1].content == "Blocked."
    assert len(blocker.contexts) == 1
    assert [report.final_action for report in reports] == ["blocked"]


async def test_interrupt_raised_during_the_model_call_keeps_the_turn():
    reports: list[TurnReport] = []
    model = ScriptedChatModel(script=[GraphInterrupt(()), "Resumed reply"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model, checkpointer=InMemorySaver()),
        output_orbitals=[ScriptedOrbital(name="out")],
        on_turn=reports.append,
    ).build()
    config: RunnableConfig = {"configurable": {"thread_id": "t-bubble"}}
    await agent.ainvoke({"messages": [HumanMessage(content="hello")]}, config)
    assert reports == []
    state = await agent.ainvoke(Command(resume="go"), config)
    assert state["messages"][-1].content == "Resumed reply"
    assert [report.final_action for report in reports] == ["answered"]


@tool
def broken_lookup() -> str:
    """A tool that always fails."""
    raise ValueError("tool down")


async def test_tool_failure_aborts_the_turn():
    reports: list[TurnReport] = []
    model = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[{"name": "broken_lookup", "args": {}, "id": "c1"}])
        ]
    )
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model, tools=[broken_lookup]),
        on_turn=reports.append,
    ).build()
    with pytest.raises(ValueError, match="tool down"):
        await agent.ainvoke({"messages": [HumanMessage(content="hours?")]})
    assert [report.final_action for report in reports] == ["aborted"]
    assert reports[0].error_type == "ValueError"


class _JumpBackOnce(AgentMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.jumped = False

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self.jumped:
            return None
        self.jumped = True
        return {"jump_to": "model"}


async def test_langchain_jump_back_to_the_model_is_checked_again():
    reports: list[TurnReport] = []
    out = ScriptedOrbital(name="out")
    model = ScriptedChatModel(script=["first", "second"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model, middleware=[_JumpBackOnce()]),
        output_orbitals=[out],
        on_turn=reports.append,
    ).build()
    state = await agent.ainvoke({"messages": [HumanMessage(content="hello")]})
    assert state["messages"][-1].content == "second"
    assert [ctx.message for ctx in out.contexts] == ["first", "second"]
    assert reports[0].final_action == "answered"


async def test_pydantic_ai_output_validator_retry_is_checked_again():
    reports: list[TurnReport] = []
    out = ScriptedOrbital(name="out")
    replies = iter(["too short", "a longer reply"])

    def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=next(replies))])

    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
        output_orbitals=[out],
        on_turn=reports.append,
    ).build()

    @agent.output_validator
    def long_enough(output: str) -> str:
        if len(output) < 10:
            raise ModelRetry("Write a longer reply.")
        return output

    result = await agent.run("hello")
    assert result.output == "a longer reply"
    assert [ctx.message for ctx in out.contexts] == ["too short", "a longer reply"]
    assert reports[0].final_action == "answered"


async def test_pydantic_ai_empty_response_is_not_checked():
    reports: list[TurnReport] = []
    out = ScriptedOrbital(name="out")
    replies = iter([ModelResponse(parts=[]), ModelResponse(parts=[TextPart(content="Hi!")])])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(lambda messages, info: next(replies))),
        output_orbitals=[out],
        on_turn=reports.append,
    ).build()
    result = await agent.run("hello")
    assert result.output == "Hi!"
    assert [ctx.message for ctx in out.contexts] == ["Hi!"]
    assert reports[0].final_action == "answered"


class _LoopBoundOrbital(Orbital):
    """Binds to the first event loop it runs on, like a cached async HTTP client."""

    calls_model = False

    def __init__(self) -> None:
        super().__init__(name="loop_bound")
        self.loop: asyncio.AbstractEventLoop | None = None

    async def check(self, ctx: OrbitalContext) -> Verdict:
        loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = loop
        elif self.loop is not loop:
            raise RuntimeError("bound to a different event loop")
        return Verdict.allow()


def test_sync_hooks_share_one_event_loop_across_turns():
    reports: list[TurnReport] = []
    model = ScriptedChatModel(script=["one", "two", "three"])
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model),
        input_orbitals=[_LoopBoundOrbital()],
        on_turn=reports.append,
    ).build()
    for message in ["a", "b", "c"]:
        agent.invoke({"messages": [HumanMessage(content=message)]})
    assert [report.final_action for report in reports] == ["answered"] * 3


def test_tool_failure_aborts_the_turn_with_the_sync_api():
    reports: list[TurnReport] = []
    model = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[{"name": "broken_lookup", "args": {}, "id": "c1"}])
        ]
    )
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(model, tools=[broken_lookup]),
        on_turn=reports.append,
    ).build()
    with pytest.raises(ValueError, match="tool down"):
        agent.invoke({"messages": [HumanMessage(content="hours?")]})
    assert [report.final_action for report in reports] == ["aborted"]


class _EmptyResponse(AgentMiddleware):
    """An inner middleware that answers without any AI message."""

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        from langchain.agents.middleware.types import ModelResponse

        return ModelResponse(result=[])


async def test_langchain_response_without_an_ai_message_fails_closed():
    reports: list[TurnReport] = []
    agent = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=LangChainConnector(ScriptedChatModel(script=[]), middleware=[_EmptyResponse()]),
        output_orbitals=[ScriptedOrbital(name="out")],
        on_turn=reports.append,
    ).build()
    with pytest.raises(RuntimeError, match="no AI message"):
        await agent.ainvoke({"messages": [HumanMessage(content="hello")]})
    assert [report.final_action for report in reports] == ["aborted"]
