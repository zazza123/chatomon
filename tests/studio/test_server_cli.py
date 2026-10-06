import json
import logging
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

import chatomon.studio.server as server
from chatomon import Atom, ChatomonConfigError
from chatomon._cli import main
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital
from chatomon.studio import DEFAULT_PORT, serve
from chatomon.studio._session import Session

ATOM_MODULE = """
    from pydantic_ai.models.test import TestModel
    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector
    atom = Atom(name="t", instructions="Be brief.", connector=PydanticAIConnector(TestModel()))
"""


def make_atom() -> Atom:
    return Atom(name="t", instructions="Be brief.", connector=PydanticAIConnector(TestModel()))


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def runs(monkeypatch):
    """Replace uvicorn.run and the browser; record what serve() asked for."""
    calls: dict = {}
    monkeypatch.setattr(
        server.uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs)
    )
    monkeypatch.setattr(server.webbrowser, "open", lambda url: calls.setdefault("opened", url))
    monkeypatch.setattr(
        server.threading,
        "Timer",
        lambda delay, fn, args: type("T", (), {"start": lambda self: fn(*args), "daemon": True})(),
    )
    return calls


def test_the_default_port_is_oxygen_16():
    assert DEFAULT_PORT == 8016


def test_serve_runs_uvicorn_without_logging_setup(runs, capsys):
    port = free_port()
    serve(make_atom, port=port)
    assert runs["host"] == "127.0.0.1"
    assert runs["port"] == port
    assert runs["log_config"] is None
    assert runs["access_log"] is False
    err = capsys.readouterr().err
    assert f"http://127.0.0.1:{port}/?token=" in err
    assert runs["opened"] in err
    assert "warning" not in err


def test_serve_can_skip_the_browser(runs):
    serve(make_atom(), port=free_port(), open_browser=False)
    assert "opened" not in runs


def test_a_wildcard_host_is_refused(runs):
    with pytest.raises(ChatomonConfigError, match="listens on one address"):
        serve(make_atom(), host="0.0.0.0")


def test_another_host_prints_a_warning(runs, capsys, monkeypatch):
    monkeypatch.setattr(server, "_check_port", lambda host, port: None)
    serve(make_atom(), host="192.168.1.10", port=8125, open_browser=False)
    assert "warning" in capsys.readouterr().err


def test_a_port_in_use_is_reported_before_starting(runs):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(ChatomonConfigError, match=f"port {port} is in use"):
            serve(make_atom(), port=port)
    assert "app" not in runs


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [name for name in sys.modules if name == "cliapp"]:
        monkeypatch.delitem(sys.modules, name)

    def write(source: str) -> None:
        (tmp_path / "cliapp.py").write_text(textwrap.dedent(source), encoding="utf-8")

    return write


def test_the_command_starts_the_studio(project, monkeypatch):
    project(ATOM_MODULE)
    started: dict = {}
    monkeypatch.setattr(
        server, "_start", lambda atom, **kwargs: started.update(atom=atom, **kwargs)
    )
    assert main(["studio", "cliapp:atom", "--port", "9000", "--no-browser"]) == 0
    assert started["atom"].name == "t"
    assert started == {
        **started,
        "source": "cliapp:atom",
        "host": "127.0.0.1",
        "port": 9000,
        "open_browser": False,
    }
    # Copy as code looks names up in the module the atom was loaded from.
    assert started["namespace"]["atom"] is started["atom"]


def test_the_command_uses_the_default_port(project, monkeypatch):
    project(ATOM_MODULE)
    started: dict = {}
    monkeypatch.setattr(server, "_start", lambda atom, **kwargs: started.update(kwargs))
    assert main(["studio", "cliapp:atom"]) == 0
    assert started["port"] == DEFAULT_PORT


def test_a_bad_spec_exits_with_1(capsys):
    assert main(["studio", "nocolon"]) == 1
    assert "<module>:<attribute>" in capsys.readouterr().err


def test_an_import_error_prints_its_traceback(project, capsys):
    project("raise RuntimeError('boom')\n")
    assert main(["studio", "cliapp:atom"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert "RuntimeError: boom" in err


def test_a_missing_extra_exits_with_1(monkeypatch, capsys):
    import importlib.util

    for name in [name for name in sys.modules if name.startswith("chatomon.studio")]:
        monkeypatch.delitem(sys.modules, name)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "uvicorn" else real(name, *a)
    )
    assert main(["studio", "app:atom"]) == 1
    assert "pip install 'chatomon[studio]'" in capsys.readouterr().err


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_python_m_chatomon_studio_shows_the_help():
    result = subprocess.run(
        [sys.executable, "-m", "chatomon.studio", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "MODULE:ATTRIBUTE" in result.stdout
    assert "8016" in result.stdout
    assert "--preset FILE" in result.stdout


def test_a_port_in_time_wait_after_a_restart_is_not_reported_as_in_use(runs):
    """What uvicorn leaves behind when it stops with a browser tab still connected."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    accepted, _ = listener.accept()
    accepted.close()  # the server side closes first: its end stays in TIME_WAIT
    client.close()
    listener.close()
    serve(make_atom(), port=port, open_browser=False)
    assert runs["port"] == port


@pytest.mark.parametrize("port", [0, -1, 65536, 99999])
def test_a_port_out_of_range_is_refused_with_a_message(runs, port):
    with pytest.raises(ChatomonConfigError, match="between 1 and 65535"):
        serve(make_atom(), port=port)


def test_an_address_that_cannot_be_used_is_not_called_a_busy_port(runs):
    with pytest.raises(ChatomonConfigError, match=r"cannot listen on 192\.0\.2\.55") as error:
        serve(make_atom(), host="192.0.2.55", port=free_port(), open_browser=False)
    assert "in use" not in str(error.value)


@pytest.mark.parametrize("host", ["0", "0.0", "::0", "::ffff:0.0.0.0"])
def test_other_spellings_of_a_bind_all_host_are_refused(runs, host):
    with pytest.raises(ChatomonConfigError, match="listens on one address"):
        serve(make_atom(), host=host)


def test_stopping_does_not_wait_for_a_running_turn(runs, capsys):
    serve(make_atom(), port=free_port(), open_browser=False)
    assert runs["timeout_graceful_shutdown"] == 3
    assert "chatomon studio stopped" in capsys.readouterr().err


def test_an_atom_that_declares_deps_gets_a_note(runs, capsys):
    atom = Atom(
        name="t",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(), deps_type=int),
    )
    serve(atom, port=free_port(), open_browser=False)
    assert "without deps" in capsys.readouterr().err


def test_the_token_never_reaches_the_logs_of_the_real_server(monkeypatch, caplog, capsys):
    """Run uvicorn with exactly what serve() passes it and open the token link."""
    captured: dict = {}
    monkeypatch.setattr(
        server.uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs)
    )
    port = free_port()
    serve(make_atom(), port=port, open_browser=False)
    link = next(word for word in capsys.readouterr().err.split() if word.startswith("http://"))
    token = link.split("token=")[1]
    caplog.set_level(logging.DEBUG)
    app = captured.pop("app")
    running = server.uvicorn.Server(server.uvicorn.Config(app, **captured))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not running.started:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        browser = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        with browser.open(link) as response:
            assert response.status == 200
    finally:
        running.should_exit = True
        thread.join(timeout=10)
    assert caplog.records
    assert token not in caplog.text


def test_serve_has_no_module_to_look_names_up_in(monkeypatch):
    started: dict = {}
    monkeypatch.setattr(
        server, "_start", lambda atom, **kwargs: started.update(atom=atom, **kwargs)
    )
    server.serve(make_atom(), open_browser=False)
    assert "namespace" not in started


def test_a_function_that_needs_arguments_is_explained_without_a_traceback(project, capsys):
    project(ATOM_MODULE + "    def judge(messages, info):\n        return atom\n")
    assert main(["studio", "cliapp:judge"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "chatomon studio: cliapp:judge needs arguments (messages, info)" in err


TUNABLE_MODULE = """
    from pydantic_ai.models.test import TestModel
    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector
    from chatomon.orbitals import MessageLengthOrbital
    atom = Atom(
        name="t",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel()),
        input_orbitals=[MessageLengthOrbital(max_words=60)],
    )
"""
NOT_APPLIED = (
    "  - this preset is for the atom 'other', but the studio has 't' loaded\n"
    "  - 'message_length': max_words must be a positive integer, got 0"
)


def tunable_atom(max_words: int = 60) -> Atom:
    """The atom of `TUNABLE_MODULE`, built here."""
    return Atom(
        name="t",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel()),
        input_orbitals=[MessageLengthOrbital(max_words=max_words)],
    )


def preset(*, max_words: int | None = 80, atom: str = "t") -> str:
    """A preset exported from a session on `tunable_atom()`, as the page downloads it."""
    session = Session(tunable_atom(), source="app:atom")
    if max_words is not None:
        session.edit("input", "message_length", {"max_words": max_words})
    return json.dumps({**session.export_preset(), "atom": atom}, indent=2)


def broken_preset() -> str:
    """A preset with two problems: another atom's name and a value the orbital refuses."""
    data = json.loads(preset(atom="other"))
    data["input_orbitals"][0]["settings"] = {"max_words": 0}
    return json.dumps(data)


def state_of(runs: dict, err: str) -> dict:
    """Ask the app `serve()` built for its state, with the link it printed."""
    link = next(word for word in err.split() if word.startswith("http://"))
    base, _, query = link.partition("/?")
    with TestClient(runs["app"], base_url=base) as client:
        client.get(f"/?{query}")
        return client.get("/api/state").json()


@pytest.mark.parametrize("given", [str, Path], ids=["str", "Path"])
def test_serve_starts_with_a_preset_applied(runs, capsys, tmp_path, given):
    file = tmp_path / "t.preset.json"
    file.write_text(preset(), encoding="utf-8")
    serve(tunable_atom(), port=free_port(), open_browser=False, preset=given(file))
    err = capsys.readouterr().err
    assert "note:" not in err
    state = state_of(runs, err)
    assert [change["text"] for change in state["changes"]] == ["message_length.max_words changed"]
    settings = {s["key"]: s["value"] for s in state["orbitals"]["input"][0]["settings"]}
    assert settings["max_words"] == 80
    assert state["undo"] is None


def test_without_a_preset_the_studio_starts_as_the_code_has_it(runs, capsys):
    serve(tunable_atom(), port=free_port(), open_browser=False)
    assert state_of(runs, capsys.readouterr().err)["changes"] == []


def test_a_preset_with_problems_stops_serve_before_the_port_is_looked_at(runs, tmp_path):
    file = tmp_path / "old.preset.json"
    file.write_text(broken_preset(), encoding="utf-8")
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(ChatomonConfigError) as caught:
            serve(tunable_atom(), port=taken.getsockname()[1], preset=file)
    assert str(caught.value) == f"the preset {file} was not applied:\n{NOT_APPLIED}"
    assert "app" not in runs


def test_the_command_passes_the_preset_on(project, monkeypatch):
    project(ATOM_MODULE)
    started: dict = {}
    monkeypatch.setattr(server, "_start", lambda atom, **kwargs: started.update(kwargs))
    assert main(["studio", "cliapp:atom", "--preset", "tuned.preset.json"]) == 0
    assert started["preset"] == "tuned.preset.json"
    assert main(["studio", "cliapp:atom"]) == 0
    assert started["preset"] is None


def test_the_command_starts_with_a_preset_applied(project, tmp_path, runs, capsys):
    project(TUNABLE_MODULE)
    (tmp_path / "t.preset.json").write_text(preset(), encoding="utf-8")
    command = ["studio", "cliapp:atom", "--preset", "t.preset.json", "--no-browser"]
    assert main([*command, "--port", str(free_port())]) == 0
    state = state_of(runs, capsys.readouterr().err)
    assert [change["text"] for change in state["changes"]] == ["message_length.max_words changed"]


def test_a_preset_with_problems_exits_with_1_and_a_line_for_each(project, tmp_path, runs, capsys):
    project(TUNABLE_MODULE)
    (tmp_path / "old.preset.json").write_text(broken_preset(), encoding="utf-8")
    assert main(["studio", "cliapp:atom", "--preset", "old.preset.json", "--no-browser"]) == 1
    err = capsys.readouterr().err
    assert err == f"chatomon studio: the preset old.preset.json was not applied:\n{NOT_APPLIED}\n"
    assert "app" not in runs


def test_a_preset_file_that_is_not_there_exits_with_1(project, runs, capsys):
    project(TUNABLE_MODULE)
    assert main(["studio", "cliapp:atom", "--preset", "nowhere.json", "--no-browser"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("chatomon studio: cannot read the preset nowhere.json: ")
    assert len(err.splitlines()) == 1
    assert "app" not in runs


def test_a_directory_given_as_a_preset_is_explained_not_a_traceback(runs, tmp_path):
    with pytest.raises(ChatomonConfigError, match="cannot read the preset "):
        serve(tunable_atom(), port=free_port(), preset=tmp_path)
    assert "app" not in runs


def test_a_preset_file_that_is_not_text_is_explained(runs, tmp_path):
    file = tmp_path / "picture.preset.json"
    file.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ChatomonConfigError) as caught:
        serve(tunable_atom(), port=free_port(), preset=file)
    assert str(caught.value) == f"cannot read the preset {file}: it is not UTF-8 text"
    assert "app" not in runs


def test_an_empty_preset_file_is_a_problem_of_the_preset(runs, tmp_path):
    file = tmp_path / "empty.preset.json"
    file.write_text("", encoding="utf-8")
    with pytest.raises(ChatomonConfigError) as caught:
        serve(tunable_atom(), port=free_port(), preset=file)
    heading, problem = str(caught.value).splitlines()
    assert heading == f"the preset {file} was not applied:"
    assert problem.startswith("  - this is not valid JSON: ")


def test_a_preset_saved_with_a_byte_order_mark_is_read(runs, capsys, tmp_path):
    """What an editor on Windows can put in front of the text."""
    file = tmp_path / "t.preset.json"
    file.write_text(preset(), encoding="utf-8-sig")
    serve(tunable_atom(), port=free_port(), open_browser=False, preset=file)
    state = state_of(runs, capsys.readouterr().err)
    assert [change["text"] for change in state["changes"]] == ["message_length.max_words changed"]


def test_the_notes_of_a_preset_are_printed_before_the_link(runs, capsys, tmp_path):
    file = tmp_path / "t.preset.json"
    file.write_text(preset(max_words=None), encoding="utf-8")
    serve(tunable_atom(max_words=50), port=free_port(), open_browser=False, preset=file)
    err = capsys.readouterr().err
    note = (
        "note: 'message_length' differs from the exported preset: "
        "your code or chatomon changed since.\n"
    )
    assert note in err
    assert err.index(note) < err.index("chatomon studio is running at")
