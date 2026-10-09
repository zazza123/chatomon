"""The connectors the studio runs, each with a nucleus that follows a script."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon.connectors import Connector
from chatomon.connectors.adk import ADKConnector
from chatomon.connectors.langchain import LangChainConnector
from chatomon.connectors.maf import MAFConnector, MAFFoundryConnector
from chatomon.connectors.maf._foundry import _ChatomonFoundryChatClient
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._runners import Runner
from chatomon.studio._runners import adk as adk_runner
from chatomon.studio._runners import langchain as langchain_runner
from chatomon.studio._runners import maf as maf_runner
from chatomon.studio._runners import pydantic_ai as pydantic_ai_runner
from tests.contract.harness import Call, ToolCall
from tests.contract.harness_adk import ScriptedLlm
from tests.contract.harness_langchain import ScriptedChatModel
from tests.contract.harness_maf import FakeProjectClient, ScriptedMAFClient
from tests.contract.harness_pydantic_ai import visible

Reply = str | Exception | ToolCall
"""One step of a nucleus's script: a text to reply with, an error to raise, a call to `lookup`."""

Nuclei = list[Callable[[], list[Call]]]
"""One reader for every nucleus a harness made: what that nucleus was sent, call by call."""

Maker = Callable[[Sequence[Reply], dict[str, Any], pytest.MonkeyPatch, Nuclei], Connector[Any]]
"""Builds a connector from a script, the connector's own arguments and the test's monkeypatch,
and adds the reader of its nucleus to the list it is given."""


@dataclass(frozen=True)
class StudioHarness:
    """One framework as the studio's tests drive it."""

    name: str
    runner: Runner
    _make: Maker
    _monkeypatch: pytest.MonkeyPatch
    _nuclei: Nuclei = field(default_factory=list)

    def connector(self, *replies: Reply, **kwargs: Any) -> Connector[Any]:
        """Return a connector whose nucleus gives `replies` in order.

        Args:
            *replies: What the nucleus does at each call.
            **kwargs: Passed to the connector.
        """
        return self._make(replies, kwargs, self._monkeypatch, self._nuclei)

    def heard(self) -> list[Call]:
        """What the nuclei of this harness's connectors were sent, call by call.

        An orbital is shown the conversation by chatomon; the nucleus is sent it by
        the framework. The two come from different code, so a test looks at both.
        """
        return [call for read in self._nuclei for call in read()]


def scripted(*replies: Reply, heard: list[Call] | None = None) -> FunctionModel:
    """Return a PydanticAI model that gives `replies` in order.

    Args:
        *replies: What the model does at each call.
        heard: Where to record what the model is sent at each call.
    """
    steps = list(replies)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if heard is not None:
            heard.append(Call(system=info.instructions or "", messages=visible(messages)))
        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ToolCall):
            return ModelResponse(parts=[ToolCallPart(tool_name="lookup", args={})])
        return ModelResponse(parts=[TextPart(content=step)])

    return FunctionModel(respond)


def _pydantic_ai(
    replies: Sequence[Reply],
    kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    nuclei: Nuclei,
) -> Connector[Any]:
    calls: list[Call] = []
    nuclei.append(lambda: calls)
    return PydanticAIConnector(scripted(*replies, heard=calls), **kwargs)


def _langchain(
    replies: Sequence[Reply],
    kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    nuclei: Nuclei,
) -> Connector[Any]:
    model = ScriptedChatModel(script=list(replies))
    nuclei.append(lambda: model.recorder)
    return LangChainConnector(model, **kwargs)


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
    replies: Sequence[Reply],
    kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    nuclei: Nuclei,
) -> Connector[Any]:
    calls: list[Call] = []
    nuclei.append(lambda: calls)
    return MAFConnector(ScriptedClient(list(replies), calls), **kwargs)


def _maf_foundry(
    replies: Sequence[Reply],
    kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    nuclei: Nuclei,
) -> Connector[Any]:
    calls: list[Call] = []
    nuclei.append(lambda: calls)
    # The session builds its agents itself, so the transport is replaced on the class.
    transport = ScriptedClient(list(replies), calls)._inner_get_response
    monkeypatch.setattr(_ChatomonFoundryChatClient, "_inner_get_response", transport)
    return MAFFoundryConnector(
        project_client=FakeProjectClient(), agent_name="remote-agent", agent_version="3", **kwargs
    )


def _adk(
    replies: Sequence[Reply],
    kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    nuclei: Nuclei,
) -> Connector[Any]:
    model = ScriptedLlm(script=list(replies))
    nuclei.append(lambda: model.recorder)
    return ADKConnector(model, **kwargs)


# Typed, so pyright checks every runner module against the protocol.
_FRAMEWORKS: dict[str, tuple[Runner, Maker]] = {
    "pydantic_ai": (pydantic_ai_runner, _pydantic_ai),
    "langchain": (langchain_runner, _langchain),
    "maf": (maf_runner, _maf),
    "maf_foundry": (maf_runner, _maf_foundry),
    "adk": (adk_runner, _adk),
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
