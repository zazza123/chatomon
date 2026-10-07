import json
import logging
import sys

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

from chatomon import Atom, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital
from chatomon.studio._app import create_app
from chatomon.studio._loading import load, namespace
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


def switch(client, name, enabled, *, side="input", origin=BASE):
    body = {"side": side, "name": name, "enabled": enabled}
    return client.patch("/api/orbitals", json=body, headers={"origin": origin})


def test_switching_an_orbital(client):
    response = switch(client, "reader", False)
    assert response.status_code == 200
    assert response.json()["orbitals"]["input"][1]["enabled"] is False


def test_switching_errors(client, session, monkeypatch):
    assert switch(client, "ghost", False).status_code == 404
    refused = switch(client, "first", False)
    assert refused.status_code == 409
    assert "reads 'first'" in refused.json()["error"]["message"]
    assert switch(client, "first", "no").status_code == 422
    assert client.patch("/api/orbitals", json={"enabled": False}, headers=ORIGIN).status_code == 422

    def busy(side, name, enabled):
        raise StudioBusyError("a turn is running; wait for it to finish")

    monkeypatch.setattr(session, "switch", busy)
    assert switch(client, "first", False).status_code == 409


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
        switched = switch(remote, "reader", False, origin=lan)
        assert switched.json()["studio"]["local"] is False


def test_an_orbital_with_dots_for_a_name_can_be_switched():
    atom = Atom(
        name="t",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel()),
        input_orbitals=[ScriptedOrbital(name="..")],
    )
    app = create_app(Session(atom, source="app:atom"), token=TOKEN, host="127.0.0.1", port=8016)
    with TestClient(app, base_url=BASE) as client:
        client.get(f"/?token={TOKEN}")
        response = switch(client, "..", False)
        assert response.json()["orbitals"]["input"][0]["enabled"] is False


def test_a_message_that_cannot_be_encoded_is_refused(client):
    response = client.post(
        "/api/turns",
        content=b'{"message": "broken \\ud800 text"}',
        headers={**ORIGIN, "content-type": "application/json"},
    )
    assert response.status_code == 422
    assert client.get("/api/state").status_code == 200


def test_every_error_is_json(session, monkeypatch):
    app = create_app(session, token=TOKEN, host="127.0.0.1", port=8016)
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as client:
        client.get(f"/?token={TOKEN}")
        assert client.get("/api/nope").json()["error"]["type"] == "NotFound"
        assert client.get("/api/turns").json()["error"]["type"] == "MethodNotAllowed"

        def broken():
            raise RuntimeError("boom")

        monkeypatch.setattr(session, "state", broken)
        failed = client.get("/api/state")
        assert failed.status_code == 500
        assert failed.json()["error"]["type"] == "InternalError"
        assert "boom" not in failed.text


def test_static_files_are_revalidated(client):
    assert client.get("/static/studio.css").headers["cache-control"] == "no-cache"


def make_tunable() -> Session:
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
        input_orbitals=[MessageLengthOrbital(max_words=60), ScriptedOrbital(name="scope")],
    )
    return Session(atom, source="app:atom")


@pytest.fixture
def tunable():
    app = create_app(make_tunable(), token=TOKEN, host="127.0.0.1", port=8016)
    with TestClient(app, base_url=BASE) as client:
        client.get(f"/?token={TOKEN}")
        yield client


def patch(client, name, *, side="input", **change):
    body = {"side": side, "name": name, **change}
    return client.patch("/api/orbitals", json=body, headers=ORIGIN)


def add(client, kind, settings, *, side="input"):
    body = {"side": side, "kind": kind, "settings": settings}
    return client.post("/api/orbitals", json=body, headers=ORIGIN)


def delete(client, name, *, side="input", headers=ORIGIN):
    return client.request(
        "DELETE", "/api/orbitals", json={"side": side, "name": name}, headers=headers
    )


def input_names(response) -> list[str]:
    return [orbital["name"] for orbital in response.json()["orbitals"]["input"]]


def test_editing_a_setting(tunable):
    response = patch(tunable, "message_length", settings={"max_words": 80})
    assert response.status_code == 200
    orbital = response.json()["orbitals"]["input"][0]
    assert orbital["settings"][0]["value"] == 80
    assert orbital["modified"] is True
    assert response.json()["changes"][0]["text"] == "message_length.max_words changed"


def test_a_refused_value_names_its_setting_and_changes_nothing(tunable):
    refused = patch(tunable, "message_length", settings={"max_words": None})
    assert refused.status_code == 422
    assert refused.json()["error"] == {
        "type": "InvalidSetting",
        "message": "set max_words, max_chars or both",
        "setting": "max_words",
    }
    shape = patch(tunable, "message_length", settings={"max_words": True})
    assert shape.status_code == 422
    assert shape.json()["error"]["message"] == "`max_words` must be a whole number"
    assert tunable.get("/api/state").json()["changes"] == []


def test_moving_adding_renaming_deleting_and_undoing(tunable):
    assert input_names(patch(tunable, "scope", position=0)) == ["scope", "message_length"]
    added = add(tunable, "TurnLimitOrbital", {"max_turns": 3, "name": "a.b-c"})
    assert added.status_code == 200
    assert added.json()["orbitals"]["input"][-1]["origin"] == "studio"
    renamed = patch(tunable, "a.b-c", settings={"name": "limit"})
    assert input_names(renamed) == ["scope", "message_length", "limit"]
    deleted = delete(tunable, "limit")
    assert deleted.json()["undo"] == "limit deleted"
    assert input_names(deleted) == ["scope", "message_length"]
    undone = tunable.post("/api/orbitals/undo", json={}, headers=ORIGIN)
    assert input_names(undone) == ["scope", "message_length", "limit"]
    again = tunable.post("/api/orbitals/undo", json={}, headers=ORIGIN)
    assert again.status_code == 409
    assert again.json()["error"] == {"type": "Refused", "message": "nothing to undo"}


def test_resetting_one_orbital_and_all(tunable):
    patch(tunable, "message_length", settings={"max_words": 80})
    patch(tunable, "scope", enabled=False)
    one = tunable.post(
        "/api/orbitals/reset", json={"side": "input", "name": "message_length"}, headers=ORIGIN
    )
    assert [change["text"] for change in one.json()["changes"]] == ["scope turned off"]
    assert one.json()["undo"] == "message_length reset to code"
    everything = tunable.post("/api/orbitals/reset", json={}, headers=ORIGIN)
    assert everything.json()["changes"] == []
    assert everything.json()["undo"] == "Reset to code"


def test_what_the_orbital_routes_refuse(tunable):
    reset_half = tunable.post("/api/orbitals/reset", json={"side": "input"}, headers=ORIGIN)
    cases = [
        (patch(tunable, "ghost", enabled=False), 404, "UnknownOrbital"),
        (patch(tunable, "scope", settings={"timeout": 5}), 409, "Refused"),
        (patch(tunable, "message_length", settings={"name": "len"}), 409, "Refused"),
        (patch(tunable, "scope", position=9), 422, "InvalidBody"),
        (patch(tunable, "scope", position=True), 422, "InvalidBody"),
        (patch(tunable, "scope", settings={}), 422, "InvalidBody"),
        (patch(tunable, "scope", enabled="no"), 422, "InvalidBody"),
        (patch(tunable, "scope", enabled=False, position=0), 422, "InvalidBody"),
        (patch(tunable, "scope"), 422, "InvalidBody"),
        (delete(tunable, "scope"), 409, "Refused"),
        (add(tunable, "FunctionOrbital", {}), 409, "Refused"),
        (add(tunable, "NoSuchOrbital", {}), 409, "Refused"),
        (add(tunable, "TurnLimitOrbital", {"max_turns": 3}, side="output"), 409, "Refused"),
        (add(tunable, "TurnLimitOrbital", {}), 422, "InvalidSetting"),
        (add(tunable, "TurnLimitOrbital", {"max_turns": 3}, side="sideways"), 422, "InvalidBody"),
        (add(tunable, 3, {}), 422, "InvalidBody"),
        (reset_half, 422, "InvalidBody"),
    ]
    for response, status, kind in cases:
        assert (response.status_code, response.json()["error"]["type"]) == (status, kind)
    assert tunable.get("/api/state").json()["changes"] == []


def test_acting_on_an_orbital_that_is_gone_is_a_404(tunable):
    add(tunable, "TurnLimitOrbital", {"max_turns": 3})
    delete(tunable, "turn_limit")
    stale = patch(tunable, "turn_limit", settings={"max_turns": 5})
    assert stale.status_code == 404
    assert stale.json()["error"]["message"] == "the atom has no input orbital named 'turn_limit'"


def test_the_new_routes_need_the_studios_origin(tunable):
    foreign = {"origin": "http://evil.example"}
    assert delete(tunable, "scope", headers=foreign).status_code == 403
    assert delete(tunable, "scope", headers={}).status_code == 403
    assert tunable.post("/api/orbitals", json={}, headers=foreign).status_code == 403
    assert tunable.post("/api/orbitals/reset", json={}, headers=foreign).status_code == 403
    assert tunable.post("/api/orbitals/undo", json={}, headers=foreign).status_code == 403


def test_every_change_is_refused_while_a_turn_runs(tunable, monkeypatch):
    monkeypatch.setattr(Session, "busy", property(lambda self: True))
    responses = [
        patch(tunable, "scope", enabled=False),
        patch(tunable, "message_length", settings={"max_words": 80}),
        patch(tunable, "scope", position=0),
        add(tunable, "TurnLimitOrbital", {"max_turns": 3}),
        delete(tunable, "scope"),
        tunable.post("/api/orbitals/reset", json={}, headers=ORIGIN),
        tunable.post("/api/orbitals/undo", json={}, headers=ORIGIN),
    ]
    assert [(r.status_code, r.json()["error"]["type"]) for r in responses] == [(409, "Busy")] * 7


def test_copy_as_code(tunable):
    patch(tunable, "message_length", settings={"max_words": 80})
    response = tunable.get("/api/code")
    assert response.status_code == 200
    body = response.json()
    assert body["changes"] == ["message_length.max_words changed"]
    assert "MessageLengthOrbital(max_words=80)," in body["code"]
    assert body["placeholders"] == ["scope: a ScriptedOrbital from your code"]
    assert response.headers["cache-control"] == "no-store"


def test_exporting_a_preset(tunable):
    patch(tunable, "message_length", settings={"max_words": 80})
    response = tunable.get("/api/preset")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "content-disposition" not in response.headers
    assert response.text.startswith('{\n  "chatomon_studio_preset": 1,')
    assert response.json()["input_orbitals"][0]["settings"] == {"max_words": 80}
    download = tunable.get("/api/preset?download=1")
    assert download.headers["content-disposition"] == (
        'attachment; filename="speedypack.preset.json"'
    )
    assert download.json() == response.json()


def test_the_downloaded_file_has_a_safe_name():
    atom = Atom(
        name='my atom/"1"',
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel()),
    )
    app = create_app(Session(atom, source="app:atom"), token=TOKEN, host="127.0.0.1", port=8016)
    with TestClient(app, base_url=BASE) as client:
        client.get(f"/?token={TOKEN}")
        header = client.get("/api/preset?download=1").headers["content-disposition"]
    assert header == 'attachment; filename="my_atom__1_.preset.json"'


def test_importing_a_preset(tunable):
    patch(tunable, "scope", enabled=False)
    text = tunable.get("/api/preset").text
    tunable.post("/api/orbitals/reset", json={}, headers=ORIGIN)
    response = tunable.post("/api/preset", json={"text": text}, headers=ORIGIN)
    assert response.status_code == 200
    body = response.json()
    assert body["warnings"] == []
    assert [change["text"] for change in body["state"]["changes"]] == ["scope turned off"]
    assert body["state"]["undo"] == "Preset imported"


def test_an_invalid_preset_lists_its_problems(tunable):
    response = tunable.post("/api/preset", json={"text": "{oops"}, headers=ORIGIN)
    assert response.status_code == 422
    error = response.json()["error"]
    assert (error["type"], error["message"]) == ("InvalidPreset", "the preset was not applied")
    assert len(error["problems"]) == 1
    assert error["problems"][0].startswith("this is not valid JSON: ")
    missing = tunable.post("/api/preset", json={"preset": {}}, headers=ORIGIN)
    assert (missing.status_code, missing.json()["error"]["type"]) == (422, "InvalidBody")
    assert tunable.post("/api/preset", json={"text": "{}"}).status_code == 403


def raw(client, method, path, body: bytes):
    headers = {**ORIGIN, "content-type": "application/json"}
    return client.request(method, path, content=body, headers=headers)


def test_text_that_cannot_be_sent_back_is_refused_at_the_door(tunable):
    bad = "\\ud800"  # a lone surrogate, as a JSON escape
    requests = [
        ("POST", "/api/orbitals", f'{{"side": "input", "kind": "{bad}", "settings": {{}}}}'),
        ("PATCH", "/api/orbitals", f'{{"side": "{bad}", "name": "scope", "enabled": false}}'),
        (
            "PATCH",
            "/api/orbitals",
            f'{{"side": "input", "name": "message_length", "settings": {{"{bad}": 1}}}}',
        ),
        ("DELETE", "/api/orbitals", f'{{"side": "input", "name": "{bad}"}}'),
        ("POST", "/api/orbitals/reset", f'{{"side": "input", "name": "{bad}"}}'),
    ]
    for method, path, body in requests:
        response = raw(tunable, method, path, body.encode())
        assert response.status_code == 422, body
        assert response.json()["error"]["type"] == "InvalidBody", body
    assert tunable.get("/api/state").status_code == 200


def test_a_side_that_does_not_exist_is_a_malformed_request(tunable):
    for response in (
        patch(tunable, "scope", side="sideways", enabled=False),
        delete(tunable, "scope", side="sideways"),
    ):
        assert response.status_code == 422
        assert response.json()["error"] == {
            "type": "InvalidBody",
            "message": "side must be 'input' or 'output'",
        }


def test_a_reset_with_a_body_that_is_not_an_object_resets_nothing(tunable):
    patch(tunable, "scope", enabled=False)
    for body in (b"[1]", b"nonsense", b'"all"'):
        assert raw(tunable, "POST", "/api/orbitals/reset", body).status_code == 422
    assert len(tunable.get("/api/state").json()["changes"]) == 1


LONG = "one two three four"
SAVED = "tests/studio/test_one_two_three_four.py"
KEYS = {"code", "path", "root", "writable", "integration", "earlier", "notes", "warnings", "turn"}


def serving(atom: Atom, **kwargs):
    session = Session(atom, source="app:atom", **kwargs)
    return TestClient(create_app(session, token=TOKEN, host="127.0.0.1", port=8016), base_url=BASE)


@pytest.fixture
def saving(tmp_path, monkeypatch):
    """A studio started from a module, in a directory of its own."""
    monkeypatch.chdir(tmp_path)
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
        input_orbitals=[MessageLengthOrbital(max_words=3, reply="Too long.")],
    )
    with serving(atom, namespace={"atom": atom}) as client:
        client.get(f"/?token={TOKEN}")
        yield client


def turn(client, message: str) -> None:
    assert client.post("/api/turns", json={"message": message}, headers=ORIGIN).status_code == 200


def write(client, n: int = 1, **body):
    """Ask to write turn `n`, naming the turn the card was made from unless the test does."""
    if "turn" not in body:
        shown = client.get(f"/api/turns/{n}/test")
        body["turn"] = shown.json()["turn"] if shown.status_code == 200 else "none"
    return client.post(f"/api/turns/{n}/test", json=body, headers=ORIGIN)


def refused(response) -> tuple[int, str]:
    return response.status_code, response.json()["error"]["type"]


def test_the_test_of_a_turn(saving, tmp_path):
    turn(saving, LONG)
    response = saving.get("/api/turns/1/test")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert set(body) == KEYS
    assert body["code"].startswith('"""Saved from chatomon studio: app:atom, turn 1."""\n')
    assert (body["path"], body["root"]) == (SAVED, str(tmp_path.resolve()))
    assert (body["writable"], body["integration"], body["earlier"]) == (True, False, 0)
    assert list(tmp_path.iterdir()) == []  # reading the test writes nothing


def test_the_turn_alone_is_asked_for_in_the_query(saving):
    turn(saving, "hello")
    turn(saving, LONG)
    whole = saving.get("/api/turns/2/test").json()
    alone = saving.get("/api/turns/2/test?alone=1").json()
    assert "hello" in whole["code"] and "hello" not in alone["code"]
    assert whole["earlier"] == alone["earlier"] == 1


def test_a_turn_that_does_not_exist(saving):
    assert refused(saving.get("/api/turns/1/test")) == (404, "UnknownTurn")
    assert refused(write(saving, 7, path=SAVED)) == (404, "UnknownTurn")
    assert refused(saving.get("/api/turns/first/test")) == (404, "NotFound")


def test_writing_the_test(saving, tmp_path):
    turn(saving, LONG)
    code = saving.get("/api/turns/1/test").json()["code"]
    response = write(saving, path=SAVED)
    written = {"path": SAVED, "command": f"python -m pytest {SAVED}"}
    assert (response.status_code, response.json()) == (200, written)
    assert (tmp_path / SAVED).read_text(encoding="utf-8") == code
    again = saving.get("/api/turns/1/test").json()
    assert again["path"] == "tests/studio/test_one_two_three_four_2.py"


def test_the_browser_cannot_choose_what_is_written(saving, tmp_path):
    turn(saving, LONG)
    assert write(saving, path=SAVED, code="import os", text="import os").status_code == 200
    written = (tmp_path / SAVED).read_text(encoding="utf-8")
    assert written == saving.get("/api/turns/1/test").json()["code"]
    assert "import os" not in written


def test_a_file_of_the_studio_is_replaced_only_when_asked(saving, tmp_path):
    turn(saving, LONG)
    write(saving, path=SAVED)
    edited = '"""Saved from chatomon studio: edited by hand."""\n'
    (tmp_path / SAVED).write_text(edited, encoding="utf-8")
    response = write(saving, path=SAVED)
    assert refused(response) == (409, "Exists")
    assert response.json()["error"]["message"] == (
        f"{SAVED} exists and was saved by the studio: overwrite it?"
    )
    assert (tmp_path / SAVED).read_text(encoding="utf-8") == edited
    assert write(saving, path=SAVED, overwrite=True).status_code == 200
    assert "edited by hand" not in (tmp_path / SAVED).read_text(encoding="utf-8")


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_file_the_studio_did_not_write_is_never_replaced(saving, tmp_path, overwrite):
    turn(saving, LONG)
    (tmp_path / "app.py").write_text("atom = 1\n", encoding="utf-8")
    response = write(saving, path="app.py", overwrite=overwrite)
    assert refused(response) == (409, "Refused")
    assert response.json()["error"]["message"] == (
        "app.py was not written by the studio: choose another path"
    )
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "atom = 1\n"


@pytest.mark.parametrize(
    "path", ["", "../outside.py", "tests/../../outside.py", "tests/notes.txt", "tests"]
)
def test_a_path_the_studio_does_not_write_to(saving, tmp_path, path):
    turn(saving, LONG)
    assert refused(write(saving, path=path)) == (422, "InvalidPath")
    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path.parent / "outside.py").exists()


def test_an_absolute_path_is_refused(saving, tmp_path):
    turn(saving, LONG)
    assert refused(write(saving, path=str(tmp_path / "test_x.py"))) == (422, "InvalidPath")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"path": 3},
        {"path": SAVED, "alone": "yes"},
        {"path": SAVED, "overwrite": 1},
        {"path": SAVED, "turn": None},
    ],
)
def test_a_malformed_request_to_write(saving, tmp_path, body):
    turn(saving, LONG)
    assert refused(write(saving, **body)) == (422, "InvalidBody")
    assert list(tmp_path.iterdir()) == []


def test_writing_needs_the_studios_origin(saving, tmp_path):
    turn(saving, LONG)
    foreign = {"origin": "http://evil.example"}
    assert (
        saving.post("/api/turns/1/test", json={"path": SAVED}, headers=foreign).status_code == 403
    )
    assert saving.post("/api/turns/1/test", json={"path": SAVED}).status_code == 403
    assert list(tmp_path.iterdir()) == []
    # The same request from the studio's own page is the one that writes: it was the Origin.
    assert write(saving, path=SAVED).status_code == 200


def test_a_failed_turn_has_no_test(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model down")

    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
    )
    with serving(atom, namespace={"atom": atom}) as client:
        client.get(f"/?token={TOKEN}")
        turn(client, "hello")
        for response in (client.get("/api/turns/1/test"), write(client, path=SAVED)):
            assert refused(response) == (409, "Refused")
            assert response.json()["error"]["message"] == (
                "A turn that failed cannot be saved as a test."
            )
    assert list(tmp_path.iterdir()) == []


def test_started_from_python_the_test_is_shown_and_not_written(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    turn(client, "hello")  # the `client` fixture has no namespace, as with serve()
    body = client.get("/api/turns/1/test").json()
    assert (body["writable"], body["path"], body["root"]) == (False, None, None)
    assert refused(write(client, path=SAVED)) == (409, "Refused")
    assert list(tmp_path.iterdir()) == []


def test_a_test_is_read_and_written_while_a_turn_runs(saving, tmp_path, monkeypatch):
    turn(saving, LONG)
    monkeypatch.setattr(Session, "busy", property(lambda self: True))
    assert saving.get("/api/turns/1/test").status_code == 200
    assert write(saving, path=SAVED).status_code == 200


def test_a_path_the_system_refuses_is_a_message_not_a_failure(saving, tmp_path):
    turn(saving, LONG)
    response = write(saving, path="tests/" + "x" * 300 + ".py")
    assert refused(response) == (409, "Refused")
    assert response.json()["error"]["message"].startswith("could not write tests/xxx")
    assert refused(write(saving, path="~/test_x.py")) == (422, "InvalidPath")
    assert not (tmp_path / "~").exists()


def test_a_card_that_is_out_of_date_writes_nothing(saving, tmp_path):
    turn(saving, LONG)
    shown = saving.get("/api/turns/1/test").json()
    assert saving.post("/api/conversation/reset", headers=ORIGIN).status_code == 200
    turn(saving, "hello")  # the conversation was started again elsewhere: another turn 1
    response = write(saving, path=SAVED, turn=shown["turn"])
    assert refused(response) == (409, "Refused")
    assert response.json()["error"]["message"] == (
        "turn 1 is not the turn this test was made from: the conversation changed. "
        "Open the turn again"
    )
    assert list(tmp_path.iterdir()) == []


def test_the_command_to_run_a_test_quotes_what_a_shell_would_split(saving):
    turn(saving, LONG)
    written = write(saving, path="my tests/test_x.py").json()
    assert written == {
        "path": "my tests/test_x.py",
        "command": "python -m pytest 'my tests/test_x.py'",
    }


# --- Reload from code -----------------------------------------------------------------------

SHOP = """
    from pydantic_ai.models.test import TestModel
    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector
    from chatomon.orbitals import MessageLengthOrbital

    INSTRUCTIONS = "Be brief."
    limit = MessageLengthOrbital(max_words=60)
    extra = MessageLengthOrbital(name="extra", max_words=500)
    atom = Atom(
        name="shop",
        instructions=INSTRUCTIONS,
        connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
        input_orbitals=[limit, extra],
    )
"""


@pytest.fixture
def reloadable(sources):
    """A studio started as the command starts it, on a module the test can write again."""
    sources("shop.py", SHOP)
    session = Session(load("shop:atom"), source="shop:atom", namespace=namespace("shop:atom"))
    app = create_app(session, token=TOKEN, host="127.0.0.1", port=8016)
    with TestClient(app, base_url=BASE) as client:
        client.get(f"/?token={TOKEN}")
        yield client, sources


def reload(client):
    return client.post("/api/reload", headers=ORIGIN)


def test_the_state_says_whether_the_atom_can_be_reloaded(client, reloadable):
    assert reloadable[0].get("/api/state").json()["studio"]["reload"] is True
    assert client.get("/api/state").json()["studio"]["reload"] is False  # started from Python


def test_a_reload_answers_with_the_new_state_and_what_was_dropped(reloadable, capsys):
    client, sources = reloadable
    patch(client, "extra", enabled=False)
    patch(client, "message_length", settings={"max_words": 80})
    sources("shop.py", SHOP.replace("Be brief.", "Be kind.").replace("[limit, extra]", "[limit]"))
    response = reload(client)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"state", "dropped", "restarted"}
    assert body["dropped"] == [
        "'extra' is no longer in your code: what the studio changed on it was dropped."
    ]
    assert body["restarted"] is None
    assert body["state"] == client.get("/api/state").json()
    assert body["state"]["atom"]["instructions"] == "Be kind."
    assert [change["text"] for change in body["state"]["changes"]] == [
        "message_length.max_words changed"
    ]
    assert capsys.readouterr().err == ""


def test_a_reload_that_fails_says_why_and_keeps_the_traceback_in_the_terminal(reloadable, capsys):
    client, sources = reloadable
    before = client.get("/api/state").json()
    sources("shop.py", SHOP.replace('INSTRUCTIONS = "Be brief."', "SECRET = 'sk-123' +"))
    response = reload(client)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["type"] == "Refused"
    assert error["message"].startswith("could not import 'shop': invalid syntax")
    assert "sk-123" not in response.text and "Traceback" not in response.text
    err = capsys.readouterr().err
    assert "Traceback" in err and "SyntaxError" in err and "sk-123" in err
    assert err.rstrip().splitlines()[-1] == f"chatomon studio: reload failed: {error['message']}"
    assert client.get("/api/state").json() == before


def test_a_refusal_of_chatomons_own_has_no_traceback(reloadable, capsys):
    client, sources = reloadable
    sources("shop.py", SHOP.replace("[limit, extra]", "[limit, limit]"))
    response = reload(client)
    assert response.status_code == 409
    message = response.json()["error"]["message"]
    assert "appears twice" in message
    assert capsys.readouterr().err == f"chatomon studio: reload failed: {message}\n"


def test_the_notes_of_the_reloaded_atom_go_to_the_terminal(reloadable, capsys):
    client, sources = reloadable
    sources(
        "shop.py",
        SHOP.replace('custom_output_text="Hi")', 'custom_output_text="Hi"), deps_type=int'),
    )
    response = reload(client)
    assert response.status_code == 200
    err = capsys.readouterr().err
    assert err.startswith("note: this atom declares deps_type") and err.count("\n") == 1
    assert "deps_type" not in json.dumps(
        {key: response.json()[key] for key in ("dropped", "restarted")}
    )
    assert reload(client).status_code == 200
    assert capsys.readouterr().err == ""  # a note is told once


def test_a_reload_while_something_runs_is_busy(reloadable, monkeypatch):
    client, _ = reloadable
    monkeypatch.setattr(Session, "busy", property(lambda self: True))
    response = reload(client)
    assert response.status_code == 409
    assert response.json()["error"]["type"] == "Busy"


def test_a_studio_started_from_python_refuses_to_reload(client, capsys):
    response = reload(client)
    assert response.status_code == 409
    assert response.json()["error"] == {
        "type": "Refused",
        "message": (
            "the studio was started from Python and does not know where the atom lives: "
            "restart it to load your changes"
        ),
    }
    assert capsys.readouterr().err == ""


def test_a_reload_needs_the_studios_origin(reloadable):
    client, sources = reloadable
    sources("shop.py", SHOP.replace("Be brief.", "Be kind."))
    assert client.post("/api/reload", headers={"origin": "http://evil.example"}).status_code == 403
    assert client.post("/api/reload").status_code == 403
    assert client.get("/api/reload").status_code == 405
    assert sys.modules["shop"].INSTRUCTIONS == "Be brief."  # nothing was imported
    assert reload(client).status_code == 200


def test_an_error_chatomon_raises_in_the_developers_module_has_its_traceback(reloadable, capsys):
    """An orbital built with a wrong value: the traceback says which line of whose file."""
    client, sources = reloadable
    sources("shop.py", SHOP.replace("max_words=60", "max_words=-5"))
    response = reload(client)
    assert response.status_code == 409
    assert "max_words must be a positive integer" in response.json()["error"]["message"]
    assert "Traceback" not in response.text
    err = capsys.readouterr().err
    assert "Traceback" in err and "shop.py" in err and "max_words=-5" in err


def test_changes_of_the_studio_that_no_longer_build_have_no_traceback(reloadable, capsys):
    client, sources = reloadable
    reader = {"name": "tone", "model": "test:judge", "instructions": "Kind.", "reads": ["extra"]}
    body = {"side": "input", "kind": "JudgeOrbital", "settings": reader}
    assert client.post("/api/orbitals", json=body, headers=ORIGIN).status_code == 200
    sources("shop.py", SHOP.replace("[limit, extra]", "[limit]"))
    response = reload(client)
    assert response.status_code == 409
    message = response.json()["error"]["message"]
    assert message.startswith("the reloaded atom does not build with the studio's changes: ")
    assert capsys.readouterr().err == f"chatomon studio: reload failed: {message}\n"
