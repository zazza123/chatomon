"""Contract harness for PydanticAI, with a scripted FunctionModel nucleus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from chatomon.connectors.pydantic_ai import PydanticAIConnector
from tests.contract.harness import Call, Harness, Step, ToolCall, lookup


def visible(messages: Sequence[ModelMessage]) -> list[tuple[str, str]]:
    """User prompts and assistant texts, in order."""
    seen: list[tuple[str, str]] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                seen.append(("user", part.content))
            elif isinstance(part, TextPart) and part.content:
                seen.append(("assistant", part.content))
    return seen


class PydanticAIHarness(Harness):
    name = "pydantic_ai"
    empty_reply_error = UnexpectedModelBehavior

    def __init__(self) -> None:
        self._calls: list[Call] = []
        self._histories: dict[str, list[ModelMessage]] = {}

    @property
    def calls(self) -> list[Call]:
        return self._calls

    def _model(self, script: Sequence[Step]) -> FunctionModel:
        steps = list(script)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self._calls.append(Call(system=info.instructions or "", messages=visible(messages)))
            step = steps.pop(0)
            if isinstance(step, Exception):
                raise step
            if isinstance(step, ToolCall):
                return ModelResponse(parts=[ToolCallPart(tool_name="lookup", args={})])
            return ModelResponse(parts=[TextPart(content=step)])

        return FunctionModel(respond)

    def connector(
        self,
        script: Sequence[Step],
        *,
        tools: bool = False,
        judge_decisions: Sequence[dict[str, Any]] = (),
    ) -> PydanticAIConnector:
        decisions = list(judge_decisions)
        kwargs: dict[str, Any] = {"tools": [lookup]} if tools else {}
        return PydanticAIConnector(
            self._model(script),
            resolve_model=lambda _model_id: TestModel(custom_output_args=decisions.pop(0)),
            **kwargs,
        )

    async def ask(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        result = await agent.run(message, message_history=self._histories.get(conversation))
        self._histories[conversation] = result.all_messages()
        return result.output

    def ask_sync(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        result = agent.run_sync(message, message_history=self._histories.get(conversation))
        self._histories[conversation] = result.all_messages()
        return result.output

    def history(self, agent: Any, *, conversation: str = "c1") -> list[tuple[str, str]]:
        return visible(self._histories.get(conversation, []))
