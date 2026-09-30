"""MAF-specific behavior that the shared connector contract does not cover."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from agent_framework import (
    Agent,
    AgentContext,
    AgentMiddleware,
    AgentSession,
    ChatContext,
    ChatMiddleware,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
)
from agent_framework.foundry import FoundryAgent, FoundryChatClient, RawFoundryAgentChatClient
from openai import AsyncOpenAI

from chatomon import Atom, ChatomonConfigError, MissingExtraError, Verdict
from chatomon.connectors.maf import MAFConnector, MAFFoundryConnector
from chatomon.orbitals import JudgeOrbital
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER
from tests.contract.harness import Call
from tests.contract.harness_maf import ScriptedMAFClient, visible
from tests.fakes import ScriptedOrbital


def make_agent(
    script: list[str | Exception], *, input_orbitals=(), output_orbitals=(), **kwargs: Any
):
    calls: list[Call] = []
    client = ScriptedMAFClient(script, calls)
    reports = []
    atom = Atom(
        name="test",
        instructions="Be useful.",
        connector=MAFConnector(client, **kwargs),
        input_orbitals=input_orbitals,
        output_orbitals=output_orbitals,
        on_turn=reports.append,
    )
    return atom.build(), client, reports


async def test_native_agent_uses_local_history_and_store_false():
    agent, client, reports = make_agent(["one", "two"])
    assert isinstance(agent, Agent)
    session = agent.create_session(session_id="local")
    assert (await agent.run("first", session=session)).text == "one"
    assert (await agent.run("second", session=session)).text == "two"
    assert session.service_session_id is None
    assert visible(session.state["in_memory"]["messages"]) == [
        ("user", "first"),
        ("assistant", "one"),
        ("user", "second"),
        ("assistant", "two"),
    ]
    assert all(options["store"] is False for options in client.options_seen)
    assert all("_chatomon_maf_turn" not in kwargs for kwargs in client.kwargs_seen)
    assert [report.conversation_id for report in reports] == ["local", "local"]


async def test_input_only_streams_and_reports_after_completion():
    agent, client, reports = make_agent(["Hello world"], input_orbitals=[ScriptedOrbital()])
    session = agent.create_session()
    stream = agent.run("hello", session=session, stream=True)
    assert reports == []
    updates = [update async for update in stream]
    assert updates
    assert (await stream.get_final_response()).text == "Hello world"
    assert [report.final_action for report in reports] == ["answered"]
    assert visible(session.state["in_memory"]["messages"]) == [
        ("user", "hello"),
        ("assistant", "Hello world"),
    ]
    assert client.options_seen[0]["store"] is False


async def test_blocked_input_streams_a_canned_reply_without_model_call():
    guard = ScriptedOrbital(Verdict.block("No."))
    agent, client, reports = make_agent([], input_orbitals=[guard])
    session = agent.create_session()
    stream = agent.run("secret", session=session, stream=True)
    assert (await stream.get_final_response()).text == "No."
    assert client.recorder == []
    assert visible(session.state["in_memory"]["messages"]) == [
        ("user", DEFAULT_BLOCKED_PLACEHOLDER),
        ("assistant", "No."),
    ]
    assert reports[0].final_action == "blocked"


async def test_output_orbitals_reject_streaming_before_model_call():
    agent, client, reports = make_agent(["draft"], output_orbitals=[ScriptedOrbital()])
    with pytest.raises(ChatomonConfigError, match="stream=False"):
        await agent.run("hello", stream=True).get_final_response()
    assert client.recorder == []
    assert reports == []


@pytest.mark.parametrize(
    "options",
    [
        {"store": True},
        {"conversation_id": "remote"},
        {"previous_response_id": "resp_1"},
        {"conversation": "conv_1"},
        {"extra_body": {"agent_session_id": "hosted_1"}},
    ],
)
async def test_remote_continuity_options_are_rejected_before_model_call(options):
    agent, client, _ = make_agent(["never"])
    with pytest.raises(ChatomonConfigError):
        await agent.run("hello", options=options)
    assert client.recorder == []


async def test_service_session_id_is_rejected_before_model_call():
    agent, client, _ = make_agent(["never"])
    session = AgentSession(service_session_id="remote")
    with pytest.raises(ChatomonConfigError, match="service_session_id"):
        await agent.run("hello", session=session)
    assert client.recorder == []


async def test_remote_id_in_a_provider_response_aborts_before_local_persistence():
    agent, client, reports = make_agent([])

    async def remote_response(**kwargs: Any) -> ChatResponse:
        return ChatResponse(
            messages=[Message("assistant", ["draft"])], conversation_id="conv_remote"
        )

    client._inner_get_response = remote_response
    session = agent.create_session()
    with pytest.raises(ChatomonConfigError, match="remote conversation_id"):
        await agent.run("hello", session=session)
    assert session.service_session_id is None
    assert session.state.get("in_memory", {}).get("messages", []) == []
    assert reports[0].final_action == "aborted"


async def test_cancelled_stream_emits_an_aborted_report():
    agent, client, reports = make_agent([])
    pending = asyncio.Event()

    def slow_stream(**kwargs: Any):
        async def updates():
            yield ChatResponseUpdate(role="assistant", contents=[Content.from_text("first")])
            await pending.wait()

        return client._build_response_stream(updates())

    client._inner_get_response = slow_stream
    stream = agent.run("hello", stream=True)
    assert (await asyncio.wait_for(anext(stream), 2)).text == "first"
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(reports) == 1
    assert reports[0].final_action == "aborted"
    assert reports[0].error_type == "CancelledError"


async def test_stream_rejects_a_provider_remote_session_id():
    agent, client, reports = make_agent([])

    def remote_stream(**kwargs: Any):
        async def updates():
            yield ChatResponseUpdate(
                role="assistant",
                contents=[Content.from_text("draft")],
                conversation_id="conv_remote",
            )

        return client._build_response_stream(updates())

    client._inner_get_response = remote_stream
    session = agent.create_session()
    with pytest.raises(ChatomonConfigError, match="service_session_id"):
        await agent.run("hello", session=session, stream=True).get_final_response()
    assert session.service_session_id is None
    assert reports[0].final_action == "aborted"


async def test_blocked_multimodal_message_is_replaced_whole():
    agent, client, _ = make_agent([], input_orbitals=[ScriptedOrbital(Verdict.block("No."))])
    session = agent.create_session()
    message = Message(
        "user",
        [
            Content.from_text("secret"),
            Content.from_uri("https://example.invalid/image.png", media_type="image/png"),
        ],
    )
    assert (await agent.run(message, session=session)).text == "No."
    stored = session.state["in_memory"]["messages"][0]
    assert stored.text == DEFAULT_BLOCKED_PLACEHOLDER
    assert len(stored.contents) == 1
    assert stored.contents[0].type == "text"
    assert client.recorder == []


async def test_input_guidance_is_limited_to_its_turn():
    hint = ScriptedOrbital(Verdict.allow("Only once."), Verdict.allow(), name="hint")
    agent, client, _ = make_agent(["one", "two"], input_orbitals=[hint])
    session = agent.create_session()
    await agent.run("first", session=session)
    await agent.run("second", session=session)
    assert "Only once." in client.recorder[0].system
    assert "Only once." not in client.recorder[1].system


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "wrong"},
        {"instructions": "wrong"},
        {"default_options": {"instructions": "wrong"}},
        {"default_options": {"store": True}},
        {"default_options": {"conversation_id": "remote"}},
        {"require_per_service_call_history_persistence": True},
    ],
)
def test_build_rejects_conflicting_or_remote_options(kwargs):
    with pytest.raises(ChatomonConfigError):
        MAFConnector(ScriptedMAFClient([], []), **kwargs)


def test_missing_judge_resolver_has_an_actionable_error():
    connector = MAFConnector(ScriptedMAFClient(["draft"], []))
    judge = JudgeOrbital(model="judge", instructions="Check replies")
    atom = Atom(
        name="test", instructions="Be useful.", connector=connector, output_orbitals=[judge]
    )
    with pytest.raises(ChatomonConfigError, match="resolve_client"):
        atom.build()


class _ObserveInput(AgentMiddleware):
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def process(
        self, context: AgentContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        self.seen.extend(message.text for message in context.messages)
        await call_next()


class _RewriteReply(ChatMiddleware):
    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()
        context.result = ChatResponse(messages=[Message("assistant", ["rewritten"])])


async def test_user_middleware_runs_inside_chatomon_checks():
    observer = _ObserveInput()
    output = ScriptedOrbital(name="output")
    agent, client, _ = make_agent(
        ["draft"],
        input_orbitals=[ScriptedOrbital(Verdict.block(guidance="Decline"))],
        output_orbitals=[output],
        middleware=[observer, _RewriteReply()],
    )
    assert (await agent.run("secret")).text == "rewritten"
    assert observer.seen == [DEFAULT_BLOCKED_PLACEHOLDER]
    assert output.contexts[0].message == "rewritten"
    assert client.recorder[0].messages[-1] == ("user", DEFAULT_BLOCKED_PLACEHOLDER)


def test_foundry_output_orbitals_are_blocked_before_agent_construction():
    atom = Atom(
        name="test",
        instructions="Be useful.",
        connector=MAFFoundryConnector(
            project_endpoint="https://example.invalid", agent_name="agent"
        ),
        output_orbitals=[ScriptedOrbital()],
    )
    with pytest.raises(ChatomonConfigError, match="real-service test"):
        atom.build()


class FakeProjectClient:
    def get_openai_client(self, **kwargs: Any) -> AsyncOpenAI:
        return AsyncOpenAI(api_key="fake", base_url="https://example.invalid")


def replace_transport(client: Any, scripted: ScriptedMAFClient) -> None:
    client._inner_get_response = scripted._inner_get_response


async def test_foundry_chat_client_uses_the_core_agent_path_without_network():
    client = FoundryChatClient(project_client=cast(Any, FakeProjectClient()), model="fake")
    scripted = ScriptedMAFClient(["approved"], [])
    replace_transport(client, scripted)
    reports = []
    agent = Atom(
        name="test",
        instructions="Be useful.",
        connector=MAFConnector(client),
        output_orbitals=[ScriptedOrbital(name="out")],
        on_turn=reports.append,
    ).build()
    session = agent.create_session()
    assert (await agent.run("hello", session=session)).text == "approved"
    assert scripted.options_seen[0]["store"] is False
    assert session.service_session_id is None
    assert reports[0].final_action == "answered"


async def test_foundry_agent_input_checks_and_local_history_without_network():
    reports = []
    guard = ScriptedOrbital(Verdict.block("No."), Verdict.allow(), name="guard")
    agent = Atom(
        name="test",
        instructions="Be useful.",
        connector=MAFFoundryConnector(
            project_client=FakeProjectClient(), agent_name="remote-agent", agent_version="1"
        ),
        input_orbitals=[guard],
        on_turn=reports.append,
    ).build()
    assert isinstance(agent, FoundryAgent)
    scripted = ScriptedMAFClient(["okay"], [])
    replace_transport(agent.client, scripted)
    session = agent.create_session()
    assert (await agent.run("secret", session=session)).text == "No."
    assert (await agent.run("hello", session=session)).text == "okay"
    assert scripted.options_seen[0]["store"] is False
    assert session.service_session_id is None
    assert visible(session.state["in_memory"]["messages"]) == [
        ("user", DEFAULT_BLOCKED_PLACEHOLDER),
        ("assistant", "No."),
        ("user", "hello"),
        ("assistant", "okay"),
    ]
    assert [report.final_action for report in reports] == ["blocked", "answered"]


def test_foundry_raw_client_type_is_rejected():
    atom = Atom(
        name="test",
        instructions="Be useful.",
        connector=MAFFoundryConnector(
            project_client=FakeProjectClient(),
            agent_name="remote-agent",
            agent_version="1",
            client_type=RawFoundryAgentChatClient,
        ),
    )
    with pytest.raises(ChatomonConfigError, match="ChatMiddlewareLayer"):
        atom.build()


def test_preconfigured_client_middleware_cannot_rewrite_checked_output():
    client = ScriptedMAFClient([], [])
    client.chat_middleware.append(_RewriteReply())
    atom = Atom(
        name="test",
        instructions="Be useful.",
        connector=MAFConnector(client),
        output_orbitals=[ScriptedOrbital()],
    )
    with pytest.raises(ChatomonConfigError, match="preconfigured client chat middleware"):
        atom.build()


def test_missing_core_is_reported_as_maf_extra(monkeypatch):
    monkeypatch.delitem(sys.modules, "chatomon.connectors.maf", raising=False)
    monkeypatch.delitem(sys.modules, "chatomon.connectors.maf._core", raising=False)
    monkeypatch.delitem(sys.modules, "chatomon.connectors.maf._shared", raising=False)
    monkeypatch.setitem(sys.modules, "agent_framework", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "agent_framework" else real_find_spec(name, *args),
    )
    with pytest.raises(MissingExtraError, match="chatomon\\[maf\\]"):
        importlib.import_module("chatomon.connectors.maf")


def test_missing_foundry_is_reported_as_foundry_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent_framework.foundry", None)
    atom = Atom(
        name="test",
        instructions="Be useful.",
        connector=MAFFoundryConnector(
            project_endpoint="https://example.invalid", agent_name="agent"
        ),
    )
    with pytest.raises(MissingExtraError, match="chatomon\\[maf-foundry\\]"):
        atom.build()
