"""The connectors the studio runs, each with a nucleus that follows a script."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon.connectors import Connector
from chatomon.connectors.langchain import LangChainConnector
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._runners import Runner
from chatomon.studio._runners import langchain as langchain_runner
from chatomon.studio._runners import pydantic_ai as pydantic_ai_runner
from tests.contract.harness import ToolCall
from tests.contract.harness_langchain import ScriptedChatModel

Reply = str | Exception | ToolCall
"""One step of a nucleus's script: a text to reply with, an error to raise, a call to `lookup`."""

Maker = Callable[[Sequence[Reply], dict[str, Any], pytest.MonkeyPatch], Connector[Any]]
"""Builds a connector from a script, the connector's own arguments and the test's monkeypatch."""


@dataclass(frozen=True)
class StudioHarness:
    """One framework as the studio's tests drive it."""

    name: str
    runner: Runner
    _make: Maker
    _monkeypatch: pytest.MonkeyPatch

    def connector(self, *replies: Reply, **kwargs: Any) -> Connector[Any]:
        """Return a connector whose nucleus gives `replies` in order.

        Args:
            *replies: What the nucleus does at each call.
            **kwargs: Passed to the connector.
        """
        return self._make(replies, kwargs, self._monkeypatch)


def scripted(*replies: Reply) -> FunctionModel:
    """Return a PydanticAI model that gives `replies` in order."""
    steps = list(replies)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ToolCall):
            return ModelResponse(parts=[ToolCallPart(tool_name="lookup", args={})])
        return ModelResponse(parts=[TextPart(content=step)])

    return FunctionModel(respond)


def _pydantic_ai(
    replies: Sequence[Reply], kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> Connector[Any]:
    return PydanticAIConnector(scripted(*replies), **kwargs)


def _langchain(
    replies: Sequence[Reply], kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> Connector[Any]:
    return LangChainConnector(ScriptedChatModel(script=list(replies)), **kwargs)


# Typed, so pyright checks every runner module against the protocol.
_FRAMEWORKS: dict[str, tuple[Runner, Maker]] = {
    "pydantic_ai": (pydantic_ai_runner, _pydantic_ai),
    "langchain": (langchain_runner, _langchain),
}

NAMES = tuple(_FRAMEWORKS)
"""The frameworks the studio opens."""


def harness(name: str, monkeypatch: pytest.MonkeyPatch) -> StudioHarness:
    """Return the harness of one of `NAMES`.

    Args:
        name: The framework.
        monkeypatch: The test's, for what a harness replaces for the test's duration.
    """
    runner, make = _FRAMEWORKS[name]
    return StudioHarness(name, runner, make, monkeypatch)
