"""The studio's start in the terminal, as the documentation shows it."""

import io
import re
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import count

import pytest
from pydantic_ai.models.test import TestModel
from starlette.testclient import TestClient

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital
from chatomon.studio import server
from chatomon.studio._app import create_app
from chatomon.studio._console import Console
from chatomon.studio._session import Session
from tests.studio import shots, terminal

COMMAND = "chatomon studio speedypack.app:atom"
URL = "http://127.0.0.1:8016/?token=Xk3q…"
DURATIONS = ("1.4s", "0.2s", "1.1s")


def recorded() -> str:
    """Run the studio's console through a short session and return what it wrote."""
    ticks = count()
    clock = lambda: datetime(2026, 10, 14, 12, 4, 31) + timedelta(seconds=17 * next(ticks))  # noqa: E731
    out = io.StringIO()
    console = Console(out, palette="truecolor", width=80, clock=clock)
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
        input_orbitals=[MessageLengthOrbital(max_words=8)],
    )
    session = Session(atom, source="speedypack.app:atom")
    start = server._start_of(session, url=URL, local=True, browser=True)
    console.banner(replace(start, version="0.9.0", reload=True))
    app = create_app(session, token="t", host="127.0.0.1", port=8016, console=console)
    with TestClient(app, base_url="http://127.0.0.1:8016") as client:
        client.get("/?token=t")
        origin = {"origin": "http://127.0.0.1:8016"}
        client.post("/api/turns", json={"message": "Where is my parcel?"}, headers=origin)
        client.post("/api/turns", json={"message": "word " * 20}, headers=origin)
        client.patch(
            "/api/orbitals",
            headers=origin,
            json={"side": "input", "name": "message_length", "settings": {"max_words": 40}},
        )
        client.post("/api/turns", json={"message": "word " * 20}, headers=origin)
    # Durations of a fake model change at every run: the recording shows steady ones.
    steady = iter(DURATIONS)
    return re.sub(r"\d+(?:\.\d)?m?s(?=\x1b\[0m\n)", lambda _: next(steady), out.getvalue())


@pytest.mark.parametrize("theme", shots.THEMES)
def test_the_recording(theme, tmp_path):
    svg = terminal.render(COMMAND, recorded(), theme)
    elsewhere = svg.replace(URL, "").replace("http://www.w3.org/2000/svg", "")
    assert "<script" not in svg and "http://" not in elsewhere
    assert "prefers-reduced-motion" in svg
    assert "Xk3q…" in svg  # the placeholder, never a real token
    for color in terminal.TOKENS[theme].values():
        assert color in svg
    data = svg.encode()
    assert len(data) <= shots.MAX_FILE_BYTES
    folder = terminal.FOLDER if shots.folder(tmp_path) == shots.SHOTS else tmp_path
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"studio-{theme}.svg").write_bytes(data)


def test_the_tokens_are_the_brands():
    css = (shots.SHOTS.parents[1] / "stylesheets" / "chatomon.css").read_text(encoding="utf-8")
    for theme in shots.THEMES:
        for color in terminal.TOKENS[theme].values():
            assert color.lower() in css.lower(), f"{theme}: {color} is not a token of chatomon.css"


def test_the_kept_recordings_are_light_and_used():
    files = sorted(terminal.FOLDER.glob("*.svg"))
    assert [f.name for f in files] == ["studio-dark.svg", "studio-light.svg"]
    assert sum(f.stat().st_size for f in files) <= shots.MAX_FOLDER_BYTES
    page = (shots.SHOTS.parents[1] / "studio" / "index.md").read_text(encoding="utf-8")
    for theme in shots.THEMES:
        assert f"assets/terminal/studio-{theme}.svg#only-{theme}" in page


@pytest.mark.parametrize("theme", shots.THEMES)
def test_the_kept_recording_is_what_the_console_writes_now(theme):
    kept = (terminal.FOLDER / f"studio-{theme}.svg").read_text(encoding="utf-8")
    assert terminal.render(COMMAND, recorded(), theme) == kept, (
        "what the terminal shows changed: regenerate the recording with "
        "CHATOMON_SCREENSHOTS=1 uv run pytest tests/studio/test_terminal_recording.py"
    )
