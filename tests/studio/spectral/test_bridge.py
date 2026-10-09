"""The bridge end to end: a fake relay, the real client and adapter, and a scripted atom."""

import asyncio
import json
import threading
import time
from typing import Any

import pytest

from chatomon import Atom, ChatomonConfigError, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.schemas import OrbitalContext
from chatomon.studio import _loading as loading
from chatomon.studio._chains import RefusedError
from chatomon.studio._loading import load, namespace
from chatomon.studio._session import Session, StudioBusyError
from chatomon.studio._spectral import Bridge, _bridge
from chatomon.studio._spectral._adapter import Unanswered
from chatomon.studio._values import InvalidSettingError
from tests.fakes import ScriptedOrbital
from tests.studio import frameworks
from tests.studio.frameworks import Reply, StudioHarness, scripted
from tests.studio.spectral.helpers import eventually, gated

KEY = "good-key"
HELLO = {"input": "hello"}


def atom_with(*replies: Reply, **kwargs: Any) -> Atom:
    connector = PydanticAIConnector(scripted(*replies))
    return Atom(name="shop", instructions="Be brief.", connector=connector, **kwargs)


@pytest.fixture
async def bridges():
    """Make bridges that are closed when the test ends."""
    made: list[Bridge] = []

    def make(atom: Atom) -> Bridge:
        bridge = Bridge(Session(atom, source="app:atom"))
        made.append(bridge)
        return bridge

    yield make
    for bridge in made:
        await bridge.close()


async def connected(bridge: Bridge) -> None:
    await bridge.connect("SPECTRAL_BRIDGE_API_KEY")
    await eventually(lambda: bridge.state()["status"] == "connected")


def test_a_bridge_that_was_never_connected_is_off(bridges, monkeypatch):
    monkeypatch.setenv("SPECTRAL_BRIDGE_API_KEY", KEY)
    monkeypatch.delenv("SPECTRAL_BRIDGE_RELAY_URL", raising=False)
    assert bridges(atom_with("Hi")).state() == {
        "status": "off",
        "active": False,
        "reason": None,
        "message": "Spectral is not connected",
        "local": True,
        "notes": [],
        "api_key_env": "SPECTRAL_BRIDGE_API_KEY",
        "key_set": True,
        "relay": "bridge.spectral.principled.app",
        "adapter": None,
        "since": None,
        "test": None,
        "counts": None,
    }


def test_the_state_carries_what_the_runner_says_of_the_atom(bridges):
    """What a developer reads before connecting: a turn that can fail, a place that is written."""
    connector = PydanticAIConnector(scripted("Hi"), deps_type=str)
    bridge = bridges(Atom(name="shop", instructions="Be brief.", connector=connector))
    [note] = bridge.state()["notes"]
    assert note.startswith("this atom declares deps_type")


async def test_connecting_opens_the_adapter_and_reaches_the_relay(spectral, bridges, capsys):
    bridge = bridges(atom_with("Hi"))
    await bridge.connect("SPECTRAL_BRIDGE_API_KEY")
    assert bridge.state()["status"] in {"connecting", "connected"}
    await eventually(lambda: bridge.state()["status"] == "connected")
    state = bridge.state()
    assert state["message"] == "Connected to Spectral"
    assert state["relay"] == "127.0.0.1"
    assert state["adapter"]["routes"] == ["/v1/responses", "/v1/chat/completions"]
    assert 0 < state["adapter"]["port"] < 65536
    assert state["since"].endswith("Z")
    assert state["counts"]["requests"] == 0
    assert "connecting this atom to Spectral through 127.0.0.1" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/responses", HELLO),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}),
    ],
)
async def test_a_request_from_the_relay_is_answered_by_the_atom(spectral, bridges, path, body):
    guard = ScriptedOrbital(name="guard")
    bridge = bridges(atom_with("We open at 9.", input_orbitals=[guard]))
    await connected(bridge)
    answer = await spectral.request(path, body)
    assert answer["status"] == 200
    assert "We open at 9." in json.dumps(answer["body"])
    assert guard.contexts[0].message == "hello"
    counts = bridge.state()["counts"]
    assert (counts["requests"], counts["running"], counts["failed"]) == (1, 0, 0)
    assert counts["final"]["answered"] == 1
    assert [(row["name"], row["allow"]) for row in counts["orbitals"]] == [("guard", 1)]


async def test_the_history_of_a_request_reaches_the_orbitals(spectral, bridges):
    guard = ScriptedOrbital(name="guard")
    bridge = bridges(atom_with("two", input_orbitals=[guard]))
    await connected(bridge)
    items = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "again"},
    ]
    await spectral.request("/v1/responses", {"input": items})
    seen = [(message.role, message.content) for message in guard.contexts[0].history]
    assert seen == [("user", "hello"), ("assistant", "one")]


async def test_a_message_an_orbital_blocks_is_answered_with_its_reply(spectral, bridges):
    guard = ScriptedOrbital(Verdict.block("No."), name="guard")
    bridge = bridges(atom_with("unused", input_orbitals=[guard]))
    await connected(bridge)
    answer = await spectral.request("/v1/responses", HELLO)
    assert answer["status"] == 200
    assert answer["body"]["output"][0]["content"][0]["text"] == "No."
    counts = bridge.state()["counts"]
    assert (counts["final"]["blocked"], counts["orbitals"][0]["block"]) == (1, 1)


async def test_a_turn_that_fails_is_an_error_in_the_studios_words(spectral, bridges):
    bridge = bridges(atom_with(RuntimeError("the key is sk-123")))
    await connected(bridge)
    answer = await spectral.request("/v1/responses", HELLO)
    assert answer["status"] == 502
    assert answer["body"] == {
        "error": {"type": "AtomFailed", "message": "The atom failed with RuntimeError."}
    }
    counts = bridge.state()["counts"]
    assert (counts["requests"], counts["failed"], sum(counts["final"].values())) == (1, 1, 0)
    assert "sk-123" not in json.dumps(bridge.state())


async def test_a_request_the_adapter_does_not_read_is_counted_as_rejected(spectral, bridges):
    bridge = bridges(atom_with("Hi"))
    await connected(bridge)
    answer = await spectral.request("/v1/responses", {"input": "hello", "stream": True})
    assert answer["status"] == 400
    counts = bridge.state()["counts"]
    assert (counts["rejected"], counts["requests"]) == (1, 0)


async def test_a_change_of_the_orbitals_is_seen_by_the_next_request(spectral, bridges):
    guard = ScriptedOrbital(Verdict.block("No."), name="guard")
    bridge = bridges(atom_with("Hi", "Hi", input_orbitals=[guard]))
    await connected(bridge)
    first = await spectral.request("/v1/responses", HELLO)
    bridge._session.switch("input", "guard", False)
    second = await spectral.request("/v1/responses", HELLO)
    texts = [answer["body"]["output"][0]["content"][0]["text"] for answer in (first, second)]
    assert texts == ["No.", "Hi"]
    counts = bridge.state()["counts"]
    assert counts["configurations"] == 2
    assert counts["orbitals"][0]["not_run"] == 0


SHOP = """
    from pydantic_ai.models.test import TestModel

    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector
    from settings import REPLY

    connector = PydanticAIConnector(TestModel(custom_output_text=REPLY))
    atom = Atom(name="shop", instructions="Be brief.", connector=connector)
"""


async def test_a_request_that_arrives_during_a_reload_is_answered_by_what_was_loaded(
    sources, monkeypatch
):
    sources("shop.py", SHOP)
    sources("settings.py", 'REPLY = "Before."\n')
    session = Session(load("shop:atom"), source="shop:atom", namespace=namespace("shop:atom"))
    bridge = Bridge(session)
    assert (await bridge._run([], "hello", counts=None)).reply == "Before."
    release = threading.Event()
    real = loading.load

    def slow(spec: str) -> Atom:
        release.wait(timeout=5)
        return real(spec)

    monkeypatch.setattr(loading, "load", slow)
    sources("settings.py", 'REPLY = "After."\n')
    reload = asyncio.ensure_future(session.reload())
    await asyncio.sleep(0.05)
    asked = asyncio.ensure_future(bridge._run([], "hello", counts=None))
    await asyncio.sleep(0.05)
    assert not asked.done()
    release.set()
    turn = await asyncio.wait_for(asked, timeout=5)
    assert turn.reply == "After."
    await reload


async def test_a_copy_that_cannot_be_built_is_a_failed_request(spectral, bridges, monkeypatch):
    bridge = bridges(atom_with("Hi"))
    await connected(bridge)

    def broken(**kwargs: Any) -> Any:
        raise ChatomonConfigError("the environment variable SECRET_NAME is not set")

    monkeypatch.setattr(bridge._session, "build_copy", broken)
    answer = await spectral.request("/v1/responses", HELLO)
    assert answer["status"] == 502
    assert answer["body"]["error"]["message"] == "The atom failed with ChatomonConfigError."
    counts = bridge.state()["counts"]
    assert (counts["requests"], counts["running"], counts["failed"]) == (1, 0, 1)
    assert counts["orbitals"] == []


class Meeting(ScriptedOrbital):
    """Holds every turn until all the turns that are expected have arrived."""

    def __init__(self, turns: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.turns = turns
        self.all_here = asyncio.Event()

    async def check(self, ctx: OrbitalContext) -> Verdict:
        verdict = await super().check(ctx)
        if len(self.contexts) == self.turns:
            self.all_here.set()
        await asyncio.wait_for(self.all_here.wait(), timeout=5)
        return verdict


async def test_every_request_is_counted_with_its_own_report(framework: StudioHarness, bridges):
    """Turns that overlap on every framework: each report finds the request it belongs to."""
    guard = Meeting(3, name="guard")
    blocker = ScriptedOrbital(Verdict.allow(), Verdict.block("No."), Verdict.allow(), name="b")
    atom = Atom(
        name="shop",
        instructions="Be brief.",
        connector=framework.connector("a", "b", "c"),
        input_orbitals=[guard, blocker],
    )
    bridge = bridges(atom)
    bridge._counts = _bridge.Counts()
    turns = await asyncio.gather(
        *(bridge._run([], f"m{n}", counts=bridge._counts) for n in range(3))
    )
    assert all(turn.report is not None for turn in turns)
    assert sorted(turn.reply for turn in turns if turn.report.final_action == "blocked") == ["No."]
    counts = bridge.state()["counts"]
    assert (counts["requests"], counts["running"], counts["failed"]) == (3, 0, 0)
    assert (counts["final"]["answered"], counts["final"]["blocked"]) == (2, 1)
    rows = {row["name"]: row for row in counts["orbitals"]}
    assert (rows["guard"]["allow"], rows["b"]["allow"], rows["b"]["block"]) == (3, 2, 1)


# PydanticAI asks its model again for an empty reply, and then raises: it has none to give.
@pytest.mark.parametrize("name", [name for name in frameworks.NAMES if name != "pydantic_ai"])
async def test_a_reply_that_is_empty_is_a_failed_request(
    name: str, bridges, monkeypatch: pytest.MonkeyPatch
):
    framework = frameworks.harness(name, monkeypatch)
    atom = Atom(name="shop", instructions="Be brief.", connector=framework.connector(""))
    bridge = bridges(atom)
    bridge._counts = _bridge.Counts()
    turn = await bridge._run([], "hello", counts=bridge._counts)
    assert turn.failure is not None
    assert (turn.failure.status, turn.failure.message) == (500, "The atom gave no reply.")
    counts = bridge.state()["counts"]
    assert (counts["failed"], sum(counts["final"].values())) == (1, 0)


async def test_the_test_request_runs_a_turn_and_is_not_counted(spectral, bridges):
    guard = ScriptedOrbital(name="guard")
    bridge = bridges(atom_with("Hi", input_orbitals=[guard]))
    await bridge.test()
    test = bridge.state()["test"]
    assert (test["outcome"], test["final_action"], test["message"]) == (
        "answered",
        "answered",
        "The atom answered.",
    )
    assert isinstance(test["duration_ms"], int)
    assert guard.contexts[0].message == "Hello"
    assert bridge.state()["counts"] is None
    assert bridge._session.turns == []


async def test_a_test_request_that_fails_says_so_in_the_studios_words(bridges):
    bridge = bridges(atom_with(RuntimeError("boom sk-123")))
    await bridge.test()
    test = bridge.state()["test"]
    assert (test["outcome"], test["message"]) == ("failed", "The atom failed with RuntimeError.")
    assert "sk-123" not in json.dumps(bridge.state())


async def test_one_test_request_at_a_time(bridges):
    atom, gate = gated()
    bridge = bridges(atom)
    first = asyncio.ensure_future(bridge.test())
    await asyncio.sleep(0.05)
    assert bridge.state()["test"] == {"outcome": "running"}
    with pytest.raises(StudioBusyError, match="a test is running"):
        await bridge.test()
    gate.set()
    await first
    assert bridge.state()["test"]["outcome"] == "answered"


async def test_disconnecting_lets_a_request_that_runs_finish(spectral, bridges):
    atom, gate = gated()
    bridge = bridges(atom)
    await connected(bridge)
    asked = asyncio.ensure_future(spectral.request("/v1/responses", HELLO))
    await eventually(lambda: bridge.state()["counts"]["running"] == 1)
    bridge.disconnect()
    assert bridge.state()["status"] == "closing"
    # What arrives now finds the studio leaving.
    with pytest.raises(Unanswered) as refused:
        await bridge._answer([], "hello")
    assert (refused.value.status, refused.value.kind) == (503, "Closing")
    assert refused.value.message == "The studio is disconnecting."
    gate.set()
    answer = await asked
    assert answer["status"] == 200
    await eventually(lambda: bridge.state()["status"] == "off")
    counts = bridge.state()["counts"]
    assert (counts["requests"], counts["failed"]) == (1, 0)


async def test_disconnecting_closes_the_connection_and_the_adapter(spectral, bridges, capsys):
    bridge = bridges(atom_with("Hi"))
    await connected(bridge)
    await spectral.request("/v1/responses", HELLO)
    bridge.disconnect()
    assert bridge.state()["status"] in {"closing", "off"}
    await eventually(lambda: bridge.state()["status"] == "off")
    state = bridge.state()
    assert (state["adapter"], state["since"]) == (None, None)
    assert state["message"] == "Spectral is not connected"
    # The counts stay to be read.
    assert state["counts"]["requests"] == 1
    assert "Spectral can no longer reach this atom." in capsys.readouterr().err


async def test_a_key_the_relay_refuses_closes_the_adapter(spectral, bridges):
    spectral.refuse = "close-4001"
    bridge = bridges(atom_with("Hi"))
    await bridge.connect("SPECTRAL_BRIDGE_API_KEY")
    await eventually(lambda: bridge.state()["status"] == "refused")
    state = bridge.state()
    assert (state["reason"], state["adapter"]) == ("key", None)
    assert state["message"] == "Spectral refused the key. Check the variable and connect again."
    # After a refusal a new connection can be asked for.
    spectral.refuse = None
    await connected(bridge)


async def test_a_second_connection_is_refused_while_one_is_open(spectral, bridges):
    bridge = bridges(atom_with("Hi"))
    await connected(bridge)
    with pytest.raises(RefusedError, match="already connected"):
        await bridge.connect("SPECTRAL_BRIDGE_API_KEY")


async def test_two_connections_asked_at_once_open_one(spectral, bridges):
    bridge = bridges(atom_with("Hi"))
    asked = [bridge.connect("SPECTRAL_BRIDGE_API_KEY") for _ in range(2)]
    first, second = await asyncio.gather(*asked, return_exceptions=True)
    assert first is None
    assert isinstance(second, RefusedError)
    await eventually(lambda: bridge.state()["status"] == "connected")
    assert spectral.connections == 1


@pytest.mark.parametrize("given", ["sk-live-0123456789", "MY KEY", "", "1KEY", "KEY=value"])
async def test_what_is_not_the_name_of_a_variable_is_refused_and_not_kept(bridges, given):
    bridge = bridges(atom_with("Hi"))
    with pytest.raises(
        InvalidSettingError, match="takes the name of an environment variable"
    ) as error:
        await bridge.connect(given)
    assert error.value.setting == "api_key_env"
    assert given not in str(error.value) or given == ""
    assert bridge.state()["api_key_env"] == "SPECTRAL_BRIDGE_API_KEY"


async def test_a_variable_that_is_not_set_is_named_and_kept_as_the_last_tried(bridges, monkeypatch):
    monkeypatch.delenv("MY_SPECTRAL_KEY", raising=False)
    bridge = bridges(atom_with("Hi"))
    with pytest.raises(
        InvalidSettingError, match="the environment variable MY_SPECTRAL_KEY is not set"
    ):
        await bridge.connect(" MY_SPECTRAL_KEY ")
    state = bridge.state()
    assert (state["api_key_env"], state["key_set"], state["status"]) == (
        "MY_SPECTRAL_KEY",
        False,
        "off",
    )


async def test_a_relay_that_is_not_wss_is_refused(bridges, monkeypatch):
    monkeypatch.setenv("SPECTRAL_BRIDGE_API_KEY", KEY)
    monkeypatch.setenv("SPECTRAL_BRIDGE_RELAY_URL", "ws://relay.example/connect")
    bridge = bridges(atom_with("Hi"))
    with pytest.raises(
        RefusedError, match="must be a wss:// address: check SPECTRAL_BRIDGE_RELAY_URL"
    ):
        await bridge.connect("SPECTRAL_BRIDGE_API_KEY")
    assert bridge.state()["adapter"] is None


async def test_a_studio_on_a_network_address_does_not_connect(spectral, bridges):
    bridge = bridges(atom_with("Hi"))
    bridge._session.local = False
    with pytest.raises(RefusedError, match="only when the studio listens on this machine"):
        await bridge.connect("SPECTRAL_BRIDGE_API_KEY")
    assert bridge.state()["local"] is False


async def test_the_key_and_the_key_of_the_adapter_are_never_shown(
    spectral, bridges, capsys, caplog
):
    caplog.set_level("DEBUG", logger="chatomon")
    caplog.set_level("INFO", logger="httpx")
    bridge = bridges(atom_with("Hi"))
    await connected(bridge)
    await spectral.request("/v1/responses", HELLO)
    client = bridge._relay._client
    secret = client.adapter_url.rsplit("/", 1)[1]
    assert len(secret) >= 32
    shown = json.dumps(bridge.state()) + capsys.readouterr().err + caplog.text
    assert KEY not in shown
    assert secret not in shown
    # httpx logs the address of every call the client makes to the adapter.
    assert f"POST http://127.0.0.1:{bridge.state()['adapter']['port']}/***/v1/responses" in shown
    bridge.disconnect()
    await eventually(lambda: bridge.state()["status"] == "off")
    # Nothing keeps the key once the connection is over.
    assert bridge._relay._client is None


async def test_the_counts_start_again_on_request(spectral, bridges):
    bridge = bridges(atom_with("Hi", "Hi"))
    await connected(bridge)
    await spectral.request("/v1/responses", HELLO)
    bridge.reset_counts()
    assert bridge.state()["counts"]["requests"] == 0
    await spectral.request("/v1/responses", {"input": "hello", "stream": True})
    assert bridge.state()["counts"]["rejected"] == 1


async def test_a_connection_is_active_while_requests_arrive(spectral, bridges, monkeypatch):
    monkeypatch.setattr(_bridge, "_ACTIVE_FOR", 0.3)
    bridge = bridges(atom_with("Hi"))
    await connected(bridge)
    assert bridge.state()["active"] is False
    await spectral.request("/v1/responses", HELLO)
    state = bridge.state()
    assert (state["active"], state["message"]) == (True, "Spectral is sending requests")
    await eventually(lambda: bridge.state()["active"] is False)
    assert bridge.state()["message"] == "Connected to Spectral"


async def test_closing_the_studio_closes_everything_at_once(spectral, bridges):
    bridge = bridges(atom_with("Hi"))
    await connected(bridge)
    started = time.monotonic()
    await bridge.close()
    assert time.monotonic() - started < 3
    state = bridge.state()
    assert (state["status"], state["adapter"]) == ("off", None)
    assert bridge._serving is not None and bridge._serving.done()
