import logging
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from http.cookiejar import CookieJar

import pytest
from pydantic_ai.models.test import TestModel

import chatomon.studio.server as server
from chatomon import Atom, ChatomonConfigError
from chatomon._cli import main
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio import DEFAULT_PORT, serve

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
