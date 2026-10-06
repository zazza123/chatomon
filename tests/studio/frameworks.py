"""The connectors the studio runs, each with a nucleus that follows a script."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon.connectors import Connector
from chatomon.connectors.langchain import LangChainConnector
from chatomon.connectors.maf import MAFConnector, MAFFoundryConnector
from chatomon.connectors.maf._foundry import _ChatomonFoundryChatClient
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._runners import Runner
from chatomon.studio._runners import langchain as langchain_runner
from chatomon.studio._runners import maf as maf_runner
from chatomon.studio._runners import pydantic_ai as pydantic_ai_runner
from tests.contract.harness import ToolCall
from tests.contract.harness_langchain import ScriptedChatModel
from tests.contract.harness_maf import FakeProjectClient, ScriptedMAFClient

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


class ScriptedClient(ScriptedMAFClient):
    """The contract suite's client, with an id on every reply as real providers give.

    Without ids Microsoft Agent Framework's in-memory history drops a reply that
    repeats an earlier one.
    """

    def _inner_get_response(self, *, messages: Any, stream: bool, options: Any, **kwargs: Any):
        inner = super()._inner_get_response(
            messages=messages, stream=stream, options=options, **kwargs
        )

        async def respond() -> Any:
            response = await inner
            for message in response.messages:
                message.message_id = uuid4().hex
            return response

        return respond()


def _maf(
    replies: Sequence[Reply], kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> Connector[Any]:
    return MAFConnector(ScriptedClient(list(replies), []), **kwargs)


def _maf_foundry(
    replies: Sequence[Reply], kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> Connector[Any]:
    # The session builds its agents itself, so the transport is replaced on the class.
    transport = ScriptedClient(list(replies), [])._inner_get_response
    monkeypatch.setattr(_ChatomonFoundryChatClient, "_inner_get_response", transport)
    return MAFFoundryConnector(
        project_client=FakeProjectClient(), agent_name="remote-agent", agent_version="3", **kwargs
    )


# Typed, so pyright checks every runner module against the protocol.
_FRAMEWORKS: dict[str, tuple[Runner, Maker]] = {
    "pydantic_ai": (pydantic_ai_runner, _pydantic_ai),
    "langchain": (langchain_runner, _langchain),
    "maf": (maf_runner, _maf),
    "maf_foundry": (maf_runner, _maf_foundry),
}

NAMES = tuple(_FRAMEWORKS)
"""The frameworks the studio opens; `maf_foundry` is the second connector of `maf`."""


def harness(name: str, monkeypatch: pytest.MonkeyPatch) -> StudioHarness:
    """Return the harness of one of `NAMES`.

    Args:
        name: The framework.
        monkeypatch: The test's, for what a harness replaces for the test's duration.
    """
    runner, make = _FRAMEWORKS[name]
    return StudioHarness(name, runner, make, monkeypatch)
