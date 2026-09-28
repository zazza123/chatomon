"""Contract harness for LangChain, with a scripted chat model nucleus."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from chatomon.connectors.langchain import LangChainConnector
from tests.contract.harness import Call, Harness, Step, ToolCall, lookup

lookup_tool = tool(lookup)


def visible(messages: Sequence[BaseMessage]) -> list[tuple[str, str]]:
    """Human messages and assistant texts (no tool calls), in order."""
    seen: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            seen.append(("user", message.content))
        elif isinstance(message, AIMessage) and not message.tool_calls and message.content:
            seen.append(("assistant", str(message.content)))
    return seen


class ScriptedChatModel(BaseChatModel):
    """Replies with the scripted steps and records what it received."""

    script: list[Any] = Field(default_factory=list)
    recorder: list[Any] = Field(default_factory=list)
    structured: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages: Sequence[BaseMessage]) -> AIMessage:
        system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
        self.recorder.append(Call(system=system, messages=visible(messages)))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ToolCall):
            tool_call = {"name": "lookup", "args": {}, "id": f"call-{len(self.recorder)}"}
            return AIMessage(content="", tool_calls=[tool_call])
        return AIMessage(content=step)

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        words = str(message.content).split(" ")
        for index, word in enumerate(words):
            text = word if index == len(words) - 1 else f"{word} "
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return RunnableLambda(lambda _messages: schema.model_validate(self.structured.pop(0)))


class LangChainHarness(Harness):
    name = "langchain"

    def __init__(self) -> None:
        self._checkpointer = InMemorySaver()
        self._models: list[ScriptedChatModel] = []

    @property
    def calls(self) -> list[Call]:
        return [call for model in self._models for call in model.recorder]

    def connector(
        self,
        script: Sequence[Step],
        *,
        tools: bool = False,
        judge_decisions: Sequence[dict[str, Any]] = (),
    ) -> LangChainConnector:
        model = ScriptedChatModel(script=list(script))
        self._models.append(model)
        judge = ScriptedChatModel(structured=list(judge_decisions))
        kwargs: dict[str, Any] = {"tools": [lookup_tool]} if tools else {}
        return LangChainConnector(
            model,
            resolve_model=lambda _model_id: judge,
            checkpointer=self._checkpointer,
            **kwargs,
        )

    @staticmethod
    def _config(conversation: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": conversation}}

    async def ask(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        state = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]}, self._config(conversation)
        )
        return str(state["messages"][-1].content)

    def ask_sync(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        state = agent.invoke(
            {"messages": [HumanMessage(content=message)]}, self._config(conversation)
        )
        return str(state["messages"][-1].content)

    def history(self, agent: Any, *, conversation: str = "c1") -> list[tuple[str, str]]:
        return visible(agent.get_state(self._config(conversation)).values["messages"])
