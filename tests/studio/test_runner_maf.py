"""What the Microsoft Agent Framework runner adds to the contract every runner keeps."""

from collections.abc import Sequence
from typing import Any

import pytest
from agent_framework import HistoryProvider, InMemoryHistoryProvider, Message, tool

from chatomon import Atom
from chatomon.connectors.maf import MAFConnector, MAFFoundryConnector
from chatomon.studio._runners import InterruptedTurnError
from chatomon.studio._runners.maf import describe, notes, run_turn, saved_test
from tests.contract.harness import ToolCall
from tests.contract.harness_maf import FakeProjectClient, ScriptedMAFClient
from tests.fakes import ScriptedOrbital
from tests.studio.frameworks import ScriptedClient

FOUNDRY = (
    "the nucleus of this atom is a Foundry agent; every turn that passes the input chain "
    "calls it with your credentials, and the tools hosted there run"
)
HISTORY = "this atom keeps its history in RedisLike; the studio's conversations are written there"


class RedisLike(HistoryProvider):
    """A history provider of the developer's, as one that writes to a database would be."""

    def __init__(self, *, load_messages: bool = True) -> None:
        super().__init__("redis_like", load_messages=load_messages)
        self.stored: list[Message] = []

    async def get_messages(
        self, session_id: str | None, *, state: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Message]:
        return list(self.stored)

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.stored.extend(messages)


class Named(ScriptedMAFClient):
    model = "gpt-5"


def built(connector: Any, **kwargs: Any) -> Any:
    return Atom(name="t", instructions="Be brief.", connector=connector, **kwargs).build()


def test_describe_names_the_clients_model_or_its_class():
    named = MAFConnector(Named([], []))
    assert describe(named, built(named)) == {"connector": "MAFConnector", "model": "gpt-5"}
    plain = MAFConnector(ScriptedMAFClient([], []))
    assert describe(plain, built(plain))["model"] == "ScriptedMAFClient"


@pytest.mark.parametrize(
    ("providers", "expected"),
    [
        (None, []),
        ([InMemoryHistoryProvider()], []),
        ([RedisLike()], [HISTORY]),
        ([RedisLike(load_messages=False)], []),
    ],
)
def test_the_notes_name_a_history_provider_of_the_developers(providers: Any, expected: list[str]):
    kwargs = {} if providers is None else {"context_providers": providers}
    assert notes(MAFConnector(ScriptedMAFClient([], []), **kwargs)) == expected


@tool(approval_mode="always_require")
def lookup() -> str:
    """Look up the opening hours."""
    return "Monday to Friday, 9-18"


async def test_a_request_for_an_approval_is_a_failed_turn_and_the_conversation_goes_on():
    guard = ScriptedOrbital(name="guard")
    client = ScriptedClient(["one", ToolCall(), "three"], [])
    agent = built(MAFConnector(client, tools=[lookup]), input_orbitals=[guard])
    first = await run_turn(agent, "hello", None)
    with pytest.raises(InterruptedTurnError, match="the turn is left unfinished"):
        await run_turn(agent, "what are your hours?", first.conversation)
    assert (await run_turn(agent, "again", first.conversation)).reply == "three"
    # The request without an answer is not in the conversation the next turn runs on.
    assert [(m.role, m.content) for m in guard.contexts[2].history] == [
        ("user", "hello"),
        ("assistant", "one"),
    ]


async def test_the_copy_a_turn_runs_on_is_still_the_same_conversation():
    seen: list[str | None] = []
    client = ScriptedClient(["one", "two"], [])
    agent = built(MAFConnector(client), on_turn=lambda report: seen.append(report.conversation_id))
    first = await run_turn(agent, "hello", None)
    second = await run_turn(agent, "again", first.conversation)
    assert second.conversation is not first.conversation
    assert seen[0] is not None
    assert seen[0] == seen[1] == second.conversation.session_id


def test_a_saved_test_awaits_its_turns_on_a_session_and_swaps_the_built_agents_client():
    lines = saved_test(MAFConnector(ScriptedMAFClient([], [])))
    assert lines.run == "await agent.run"
    assert lines.awaited
    assert lines.first == lines.later
    assert (lines.first.before, lines.first.after) == ("", "")
    assert lines.first.arguments == ("session=session",)
    assert lines.setup == ("session = agent.create_session()",)
    assert lines.reply == "result.text"
    assert lines.marker == '@pytest.mark.asyncio(loop_scope="session")'
    assert lines.offline.before_build == ()
    assert lines.offline.after_build == ("agent.client = OfflineNucleus()",)
    assert lines.offline.definitions[0] == (
        "class OfflineNucleus(ChatMiddlewareLayer, BaseChatClient):"
    )
    compile("\n".join([*lines.offline.imports, "", *lines.offline.definitions]), "x.py", "exec")


# --- The second connector: an agent hosted in Foundry ---------------------------------


def foundry(**kwargs: Any) -> MAFFoundryConnector:
    return MAFFoundryConnector(project_client=FakeProjectClient(), **kwargs)


def test_describe_names_the_foundry_agent_and_its_version(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FOUNDRY_AGENT_VERSION", raising=False)
    versioned = foundry(agent_name="remote-agent", agent_version="3")
    assert describe(versioned, built(versioned)) == {
        "connector": "MAFFoundryConnector",
        "model": "remote-agent (version 3)",
    }
    latest = foundry(agent_name="remote-agent")
    assert describe(latest, built(latest))["model"] == "remote-agent"


def test_the_foundry_agent_named_in_the_environment_is_shown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FOUNDRY_AGENT_NAME", "from-the-environment")
    monkeypatch.delenv("FOUNDRY_AGENT_VERSION", raising=False)
    connector = foundry()
    assert describe(connector, built(connector))["model"] == "from-the-environment"


@pytest.mark.parametrize(("providers", "expected"), [(None, []), ([RedisLike()], [HISTORY])])
def test_a_foundry_atom_gets_its_note_after_any_other(providers: Any, expected: list[str]):
    kwargs = {} if providers is None else {"context_providers": providers}
    assert notes(foundry(agent_name="remote-agent", **kwargs)) == [*expected, FOUNDRY]


def test_a_foundry_atom_saves_the_same_test_as_any_other():
    plain = saved_test(MAFConnector(ScriptedMAFClient([], [])))
    assert saved_test(foundry(agent_name="remote-agent")) == plain
