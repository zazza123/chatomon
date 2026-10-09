"""The add-on's routes and its place in the state, through the studio's own app."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._app import create_app
from chatomon.studio._session import Session
from tests.studio.frameworks import scripted
from tests.studio.spectral.fake_relay import FakeRelay
from tests.studio.spectral.helpers import (
    BASE,
    NAMED,
    ORIGIN,
    TOKEN,
    eventually,
    gated,
    page,
    started,
)


def make_session() -> Session:
    connector = PydanticAIConnector(scripted("Hi", "Hi", "Hi"))
    atom = Atom(name="shop", instructions="Be brief.", connector=connector)
    return Session(atom, source="app:atom")


@pytest.fixture
async def client(spectral: FakeRelay) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(make_session(), token=TOKEN, host="127.0.0.1", port=8016)
    async with started(app), page(app) as browser:
        yield browser


async def status(client: httpx.AsyncClient) -> str:
    return (await client.get("/api/spectral")).json()["status"]


async def connect(client: httpx.AsyncClient) -> None:
    answer = await client.post("/api/spectral/connect", json=NAMED, headers=ORIGIN)
    assert answer.status_code == 200, answer.text

    async def is_connected() -> bool:
        return await status(client) == "connected"

    await eventually(is_connected)


async def test_the_state_of_the_studio_carries_the_add_on(client):
    spectral_state = (await client.get("/api/state")).json()["studio"]["spectral"]
    assert spectral_state == (await client.get("/api/spectral")).json()
    assert (spectral_state["status"], spectral_state["local"]) == ("off", True)
    assert spectral_state["api_key_env"] == "SPECTRAL_BRIDGE_API_KEY"


async def test_the_state_is_not_stored_by_the_browser(client):
    assert (await client.get("/api/spectral")).headers["cache-control"] == "no-store"


async def test_the_card_connects_tests_reads_the_counts_and_disconnects(client, spectral):
    await connect(client)
    state = (await client.get("/api/spectral")).json()
    assert state["adapter"]["port"] > 0
    assert state["counts"]["requests"] == 0

    tested = await client.post("/api/spectral/test", json={}, headers=ORIGIN)
    assert tested.status_code == 200
    assert tested.json()["test"]["outcome"] == "answered"
    assert tested.json()["counts"]["requests"] == 0

    answer = await spectral.request("/v1/responses", {"input": "hello"})
    assert answer["status"] == 200
    assert (await client.get("/api/spectral")).json()["counts"]["requests"] == 1

    reset = await client.post("/api/spectral/counts/reset", json={}, headers=ORIGIN)
    assert reset.json()["counts"]["requests"] == 0

    closed = await client.post("/api/spectral/disconnect", json={}, headers=ORIGIN)
    assert closed.json()["status"] in {"closing", "off"}

    async def is_off() -> bool:
        return await status(client) == "off"

    await eventually(is_off)
    assert (await client.get("/api/spectral")).json()["adapter"] is None


@pytest.mark.parametrize("body", [None, {}, {"api_key_env": 5}, {"api_key_env": None}, ["x"]])
async def test_a_connection_asked_without_a_name_is_refused_with_what_to_send(client, body):
    answer = await client.post("/api/spectral/connect", json=body, headers=ORIGIN)
    assert answer.status_code == 422
    assert answer.json()["error"]["type"] == "InvalidBody"
    assert "api_key_env" in answer.json()["error"]["message"]


async def test_a_value_in_place_of_a_name_is_refused_and_does_not_come_back(client):
    pasted = "sk-live-0123456789abcdef"
    asked = {"api_key_env": pasted}
    answer = await client.post("/api/spectral/connect", json=asked, headers=ORIGIN)
    assert answer.status_code == 422
    assert answer.json()["error"] == {
        "type": "InvalidSetting",
        "message": "api_key_env takes the name of an environment variable, not its value",
        "setting": "api_key_env",
    }
    assert pasted not in (await client.get("/api/state")).text


async def test_a_variable_that_is_not_set_is_refused_without_its_name(client, monkeypatch):
    monkeypatch.delenv("OTHER_KEY", raising=False)
    asked = {"api_key_env": "OTHER_KEY"}
    answer = await client.post("/api/spectral/connect", json=asked, headers=ORIGIN)
    assert answer.status_code == 422
    assert answer.json()["error"] == {
        "type": "InvalidSetting",
        "message": "the environment variable that api_key_env names is not set",
        "setting": "api_key_env",
    }
    # What was sent may be a key that reads as a name: the state does not take it.
    assert "OTHER_KEY" not in (await client.get("/api/state")).text


async def test_a_relay_address_that_cannot_be_read_leaves_the_studio_working(client, monkeypatch):
    monkeypatch.setenv("SPECTRAL_BRIDGE_RELAY_URL", "wss://[bad")
    state = await client.get("/api/state")
    assert state.status_code == 200
    assert state.json()["studio"]["spectral"]["relay"] == ""
    refused = await client.post("/api/spectral/connect", json=NAMED, headers=ORIGIN)
    assert (refused.status_code, refused.json()["error"]["type"]) == (409, "Refused")


async def test_a_second_connection_is_refused(client):
    await connect(client)
    again = await client.post("/api/spectral/connect", json=NAMED, headers=ORIGIN)
    assert again.status_code == 409
    assert again.json()["error"]["type"] == "Refused"


@pytest.mark.parametrize("path", ["connect", "disconnect", "test", "counts/reset"])
async def test_the_add_ons_changes_keep_the_studios_defences(client, path):
    # Not from the studio's own page.
    assert (await client.post(f"/api/spectral/{path}", json={})).status_code == 403
    other = {"origin": "https://evil.example"}
    assert (await client.post(f"/api/spectral/{path}", json={}, headers=other)).status_code == 403


async def test_without_the_cookie_nothing_of_the_add_on_answers(spectral):
    app = create_app(make_session(), token=TOKEN, host="127.0.0.1", port=8016)
    transport = httpx.ASGITransport(app=app)
    async with started(app), httpx.AsyncClient(transport=transport, base_url=BASE) as stranger:
        assert (await stranger.get("/api/spectral")).status_code == 401
        refused = await stranger.post("/api/spectral/connect", json=NAMED, headers=ORIGIN)
        assert refused.status_code == 401


async def test_a_studio_on_a_network_address_shows_the_add_on_and_does_not_connect(spectral):
    base = "http://192.168.1.5:8016"
    app = create_app(make_session(), token=TOKEN, host="192.168.1.5", port=8016)
    async with started(app), page(app, base) as remote:
        assert (await remote.get("/api/spectral")).json()["local"] is False
        answer = await remote.post("/api/spectral/connect", json=NAMED, headers={"origin": base})
        assert answer.status_code == 409
        assert "only when the studio listens on this machine" in answer.json()["error"]["message"]
        assert (await remote.get("/api/spectral")).json()["status"] == "off"


async def test_stopping_the_studio_closes_the_connection(spectral):
    app = create_app(make_session(), token=TOKEN, host="127.0.0.1", port=8016)
    async with started(app), page(app) as client:
        await connect(client)
        assert spectral.has_client
    await eventually(lambda: not spectral.has_client)


async def test_a_test_request_is_refused_while_another_runs(spectral):
    atom, gate = gated()
    app = create_app(Session(atom, source="app:atom"), token=TOKEN, host="127.0.0.1", port=8016)
    async with started(app), page(app) as client:
        first = asyncio.ensure_future(client.post("/api/spectral/test", json={}, headers=ORIGIN))
        await asyncio.sleep(0.1)
        busy = await client.post("/api/spectral/test", json={}, headers=ORIGIN)
        assert (busy.status_code, busy.json()["error"]["type"]) == (409, "Busy")
        gate.set()
        assert (await first).json()["test"]["outcome"] == "answered"
