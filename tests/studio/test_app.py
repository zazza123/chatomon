import json
import logging

import pytest
from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

from chatomon import Atom, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._app import create_app
from chatomon.studio._session import Session, StudioBusyError
from tests.fakes import ScriptedOrbital

TOKEN = "t0ken"
BASE = "http://127.0.0.1:8016"
ORIGIN = {"origin": BASE}


def make_session() -> Session:
    first = ScriptedOrbital(Verdict.allow(annotations={"x": 1}), name="first")
    reader = ScriptedOrbital(name="reader", reads=["first"])
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(custom_output_text="Perché no? 📦")),
        input_orbitals=[first, reader],
    )
    return Session(atom, source="app:atom")


@pytest.fixture
def session() -> Session:
    return make_session()


@pytest.fixture
def client(session):
    with TestClient(
        create_app(session, token=TOKEN, host="127.0.0.1", port=8016), base_url=BASE
    ) as client:
        client.get(f"/?token={TOKEN}")
        yield client


def events(response) -> list[dict]:
    blocks = [block for block in response.text.split("\n\n") if block.strip()]
    parsed = []
    for block in blocks:
        lines = block.split("\n")
        assert lines[0].startswith("event: ")
        assert len(lines) == 2 and lines[1].startswith("data: ")
        event = json.loads(lines[1][len("data: ") :])
        assert event["type"] == lines[0][len("event: ") :]
        parsed.append(event)
    return parsed


def test_the_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "chatomon studio" in response.text
    assert client.get("/static/index.html").status_code == 200


def test_the_state(client):
    state = client.get("/api/state").json()
    assert state["atom"]["name"] == "speedypack"
    assert [orbital["name"] for orbital in state["orbitals"]["input"]] == ["first", "reader"]


def test_a_turn_streams_events_and_keeps_text_intact(client):
    message = "Perché?\nUna seconda riga 📦"
    response = client.post("/api/turns", json={"message": message}, headers=ORIGIN)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response)
    assert stream[0]["type"] == "turn_started"
    assert stream[-1]["type"] == "turn_finished"
    turn = stream[-1]["turn"]
    assert turn["message"] == message
    assert turn["reply"] == "Perché no? 📦"
    assert client.get("/api/state").json()["turns"][0]["n"] == 1


@pytest.mark.parametrize("body", [{"message": "   "}, {"text": "hi"}, ["hi"]])
def test_an_empty_or_malformed_message_is_refused(client, body):
    response = client.post("/api/turns", json=body, headers=ORIGIN)
    assert response.status_code == 422


def test_a_body_that_is_not_json_is_refused(client):
    response = client.post(
        "/api/turns", content=b"hi", headers={**ORIGIN, "content-type": "application/json"}
    )
    assert response.status_code == 422


def test_a_turn_while_busy_is_refused(client, session, monkeypatch):
    def busy(message):
        raise StudioBusyError("a turn is running; wait for it to finish")

    monkeypatch.setattr(session, "start_turn", busy)
    response = client.post("/api/turns", json={"message": "hi"}, headers=ORIGIN)
    assert response.status_code == 409
    assert response.json()["error"]["type"] == "Busy"


def test_switching_an_orbital(client):
    response = client.patch("/api/orbitals/input/reader", json={"enabled": False}, headers=ORIGIN)
    assert response.status_code == 200
    assert response.json()["orbitals"]["input"][1]["enabled"] is False


def test_switching_errors(client, session, monkeypatch):
    assert (
        client.patch(
            "/api/orbitals/input/ghost", json={"enabled": False}, headers=ORIGIN
        ).status_code
        == 404
    )
    refused = client.patch("/api/orbitals/input/first", json={"enabled": False}, headers=ORIGIN)
    assert refused.status_code == 409
    assert "reads 'first'" in refused.json()["error"]["message"]
    assert (
        client.patch(
            "/api/orbitals/input/first", json={"enabled": "no"}, headers=ORIGIN
        ).status_code
        == 422
    )

    def busy(side, name, enabled):
        raise StudioBusyError("a turn is running; wait for it to finish")

    monkeypatch.setattr(session, "set_enabled", busy)
    assert (
        client.patch(
            "/api/orbitals/input/first", json={"enabled": False}, headers=ORIGIN
        ).status_code
        == 409
    )


def test_reset(client):
    client.post("/api/turns", json={"message": "hi"}, headers=ORIGIN)
    response = client.post("/api/conversation/reset", json={}, headers=ORIGIN)
    assert response.status_code == 200
    assert response.json()["turns"] == []


def test_the_token_never_reaches_the_logs(session, caplog):
    caplog.set_level(logging.DEBUG)
    with TestClient(
        create_app(session, token=TOKEN, host="127.0.0.1", port=8016), base_url=BASE
    ) as client:
        client.get(f"/?token={TOKEN}")
        client.get("/api/state")
    # The test client logs its own requests; only what the studio logs matters here.
    studio_logs = [
        record.getMessage() for record in caplog.records if not record.name.startswith("httpx")
    ]
    assert all(TOKEN not in message for message in studio_logs)


def test_the_real_app_is_guarded(session):
    app = create_app(session, token=TOKEN, host="127.0.0.1", port=8016)
    with TestClient(app, base_url=BASE) as anonymous:
        assert anonymous.get("/api/state").status_code == 401
        assert anonymous.get("/static/js/main.js").status_code == 401
        assert anonymous.get("/api/state", headers={"host": "evil.example:8016"}).status_code == 400
        anonymous.get(f"/?token={TOKEN}")
        assert anonymous.post("/api/turns", json={"message": "hi"}).status_code == 403
        assert anonymous.get("/api/state").status_code == 200


def test_the_state_says_whether_the_studio_is_local(session, client):
    assert client.get("/api/state").json()["studio"]["local"] is True
    lan = "http://192.168.1.10:8016"
    app = create_app(session, token=TOKEN, host="192.168.1.10", port=8016)
    with TestClient(app, base_url=lan) as remote:
        remote.get(f"/?token={TOKEN}")
        assert remote.get("/api/state").json()["studio"]["local"] is False
        switched = remote.patch(
            "/api/orbitals/input/reader", json={"enabled": False}, headers={"origin": lan}
        )
        assert switched.json()["studio"]["local"] is False
