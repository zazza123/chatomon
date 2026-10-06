"""What the Microsoft Agent Framework runner adds to the contract every runner keeps."""

from collections.abc import Sequence
from typing import Any

import pytest
from agent_framework import HistoryProvider, InMemoryHistoryProvider, Message, tool

from chatomon import Atom
from chatomon.connectors.maf import MAFConnector
from chatomon.studio._runners import InterruptedTurnError
from chatomon.studio._runners.maf import describe, notes, run_turn, saved_test
from tests.contract.harness import ToolCall
from tests.contract.harness_maf import ScriptedMAFClient
from tests.studio.frameworks import ScriptedClient

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


def built(connector: Any) -> Any:
    return Atom(name="t", instructions="Be brief.", connector=connector).build()


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


async def test_a_request_for_an_approval_is_a_failed_turn():
    client = ScriptedClient(["one", ToolCall(), "three"], [])
    agent = built(MAFConnector(client, tools=[lookup]))
    first = await run_turn(agent, "hello", None)
    with pytest.raises(InterruptedTurnError, match="start a new one"):
        await run_turn(agent, "what are your hours?", first.conversation)


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
