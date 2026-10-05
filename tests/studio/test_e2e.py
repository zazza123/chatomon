"""The studio page in Chromium: turns, reports, switches and the ways a turn can go wrong."""

import asyncio
import contextlib
import http.client
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from http.cookiejar import CookieJar

import pytest
import uvicorn
from playwright.sync_api import Page, expect
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital
from chatomon.studio._app import create_app
from chatomon.studio._session import Session
from tests.fakes import ScriptedOrbital

pytestmark = pytest.mark.e2e
TOKEN = "e2e-token"
REPLY = "We are open Monday to Friday."


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answer every message; "boom" fails and "slow" takes three seconds."""
    prompts = [
        part.content
        for part in messages[-1].parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]
    if any("boom" in prompt for prompt in prompts):
        raise RuntimeError("model down")
    if any("slow" in prompt for prompt in prompts):
        await asyncio.sleep(3)
    return ModelResponse(parts=[TextPart(content=REPLY)])


@dataclass
class Studio:
    url: str
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@contextlib.contextmanager
def running(atom: Atom) -> Iterator[Studio]:
    port = free_port()
    app = create_app(Session(atom, source="tests:atom"), token=TOKEN, host="127.0.0.1", port=port)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the studio did not start"
        time.sleep(0.05)
    handle = Studio(url=f"http://127.0.0.1:{port}", server=server, thread=thread)
    try:
        yield handle
    finally:
        handle.stop()


@pytest.fixture
def studio() -> Iterator[Studio]:
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
        input_orbitals=[ScriptedOrbital(name="guard")],
        output_orbitals=[ScriptedOrbital(name="judge")],
    )
    with running(atom) as handle:
        yield handle


@pytest.fixture
def tunable() -> Iterator[Studio]:
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
        input_orbitals=[MessageLengthOrbital(max_words=60), ScriptedOrbital(name="scope")],
    )
    with running(atom) as handle:
        yield handle


def row(page: Page, name: str):
    return page.locator(f'#drawer li.orb[data-name="{name}"]')


def input_order(page: Page) -> list[str]:
    return page.locator("#list-input li.orb").evaluate_all("rows => rows.map(r => r.dataset.name)")


def open_studio(page: Page, studio: Studio, *, real_time: bool = True) -> None:
    page.goto(f"{studio.url}/?token={TOKEN}")
    if real_time:
        page.get_by_role("button", name="Real time").click()


def message_box(page: Page):
    return page.get_by_role("textbox", name="Message")


def test_a_turn_its_report_and_a_switch(page: Page, studio: Studio):
    open_studio(page, studio)
    message_box(page).fill('<img src=x onerror="window.pwned=1"> Are you open?\nAnd on Sunday?')
    page.get_by_role("button", name="Send").click()

    turn = page.locator("article.turn").first
    expect(turn).to_contain_text(REPLY)
    user_bubble = turn.locator(".user .bubble")
    expect(user_bubble).to_contain_text("<img src=x")
    expect(user_bubble).to_have_css("white-space", "pre-wrap")
    assert page.evaluate("window.pwned") is None

    lens = turn.get_by_role("button", name="Inspect turn 1")
    lens.click()
    report = page.locator("dialog[open]")
    expect(report).to_contain_text("Input chain")
    expect(report).to_contain_text("guard")
    report.get_by_role("button", name="Close").last.click()
    expect(page.get_by_role("button", name="Inspect turn 1")).to_be_focused()

    switch = page.locator('#drawer input[data-name="guard"]')
    switch.focus()
    page.keyboard.press("Space")
    expect(page.locator('#drawer input[data-name="guard"]')).not_to_be_checked()
    expect(page.locator('#drawer input[data-name="guard"]')).to_be_focused()

    message_box(page).fill("Again?")
    page.get_by_role("button", name="Send").click()
    second = page.locator("article.turn").nth(1)
    expect(second).to_contain_text(REPLY)
    second.get_by_role("button", name="Inspect turn 2").click()
    expect(page.locator("dialog[open]")).to_contain_text("No input orbitals were on.")


def test_enter_while_a_turn_runs_keeps_what_was_typed(page: Page, studio: Studio):
    open_studio(page, studio, real_time=False)
    message_box(page).fill("Are you open?")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_role("button", name="Send")).to_be_disabled()

    message_box(page).fill("my carefully typed second message")
    message_box(page).press("Enter")
    expect(message_box(page)).to_have_value("my carefully typed second message")

    expect(page.get_by_role("button", name="Send")).to_be_enabled(timeout=20_000)
    expect(page.locator("article.turn")).to_have_count(1)


def test_a_failed_turn_says_the_nucleus_failed(page: Page, studio: Studio):
    open_studio(page, studio)
    message_box(page).fill("boom")
    page.get_by_role("button", name="Send").click()

    turn = page.locator("article.turn").first
    expect(turn).to_contain_text("The turn failed: RuntimeError: model down")
    turn.get_by_role("button", name="Inspect turn 1").click()
    report = page.locator("dialog[open]")
    expect(report).to_contain_text("The nucleus was called and failed")
    expect(report).not_to_contain_text("Not called")


def test_a_studio_that_stops_answering_frees_the_page(page: Page, studio: Studio):
    open_studio(page, studio)
    expect(page.get_by_role("button", name="Send")).to_be_enabled()
    studio.stop()

    message_box(page).fill("Anyone there?")
    page.get_by_role("button", name="Send").click()
    expect(page.locator("#toast")).to_contain_text("not answering")
    expect(page.get_by_role("button", name="Send")).to_be_enabled()
    expect(page.locator("article.turn")).to_have_count(0)


def test_a_drawing_failure_does_not_freeze_the_page(page: Page, studio: Studio):
    open_studio(page, studio)
    page.evaluate(
        """() => {
          const root = document.querySelector('#atom .atom-root');
          const fail = () => { throw new Error('drawing failed'); };
          Object.defineProperty(root, 'innerHTML', { set: fail });
        }"""
    )
    message_box(page).fill("Are you open?")
    page.get_by_role("button", name="Send").click()
    expect(page.locator("article.turn").first).to_contain_text(REPLY)
    expect(page.get_by_role("button", name="Send")).to_be_enabled()


def test_reloading_during_a_turn_shows_it_running(page: Page, studio: Studio):
    open_studio(page, studio)
    message_box(page).fill("a slow question")
    page.get_by_role("button", name="Send").click()
    expect(page.locator(".typing")).to_be_visible()

    page.reload()
    expect(page.locator(".typing")).to_be_visible()
    expect(page.locator("article.turn .user .bubble")).to_contain_text("a slow question")
    expect(page.locator("#phase-title")).to_have_text("A turn is running")
    expect(page.get_by_role("button", name="Send")).to_be_disabled()

    expect(page.locator("article.turn").first).to_contain_text(REPLY, timeout=15_000)
    expect(page.get_by_role("button", name="Send")).to_be_enabled()
    expect(page.locator("#phase-title")).to_have_text("Waiting for a message")


def test_the_content_notice_opens_its_explanation(page: Page, studio: Studio):
    open_studio(page, studio)
    page.get_by_role("button", name="Content visible").click()
    notice = page.locator("dialog[open]")
    expect(notice).to_contain_text("capture_content")
    expect(notice).to_contain_text("only on this machine")


def test_the_docked_atom_does_not_cover_the_last_turn(page: Page, studio: Studio):
    page.set_viewport_size({"width": 1000, "height": 520})
    open_studio(page, studio)
    for text in ("First question", "Second question"):
        message_box(page).fill(text)
        page.get_by_role("button", name="Send").click()
        expect(page.get_by_role("button", name="Send")).to_be_enabled()
    # From the top (atom in view, no dock) to the end in one scroll, as after a new turn.
    page.evaluate("document.getElementById('play').scrollTo(0, 0)")
    expect(page.locator("#dock")).to_be_hidden()
    page.evaluate("const p = document.getElementById('play'); p.scrollTo(0, p.scrollHeight)")
    expect(page.locator("#dock")).to_be_visible()
    page.wait_for_timeout(300)
    meta = page.locator("article.turn .msg-meta").last.bounding_box()
    dock = page.locator("#dock").bounding_box()
    assert meta is not None and dock is not None
    assert meta["y"] + meta["height"] <= dock["y"]


SLOW_APP = """
    import asyncio

    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector


    async def respond(messages, info):
        await asyncio.sleep(60)
        return ModelResponse(parts=[TextPart(content="late")])


    atom = Atom(
        name="slow",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
    )
"""


def test_stopping_during_a_turn_does_not_wait_for_the_model(tmp_path):
    (tmp_path / "slowapp.py").write_text(textwrap.dedent(SLOW_APP), encoding="utf-8")
    port = free_port()
    command = [sys.executable, "-m", "chatomon.studio", "slowapp:atom", "--no-browser"]
    process = subprocess.Popen(
        [*command, "--port", str(port)], cwd=tmp_path, stderr=subprocess.PIPE, text=True
    )
    try:
        assert process.stderr is not None
        link = next(word for word in process.stderr.readline().split() if word.startswith("http"))
        base = link.split("/?")[0]
        browser = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        deadline = time.monotonic() + 10
        while True:
            try:
                browser.open(link).close()
                break
            except OSError:
                assert time.monotonic() < deadline, "the studio did not start"
                time.sleep(0.1)

        def ask() -> None:
            request = urllib.request.Request(
                f"{base}/api/turns",
                data=b'{"message": "hello"}',
                headers={"Content-Type": "application/json", "Origin": base},
            )
            # The server stops while the stream is open: the read ends with an error.
            with contextlib.suppress(OSError, http.client.HTTPException):
                browser.open(request).read()

        threading.Thread(target=ask, daemon=True).start()
        time.sleep(1)
        started = time.monotonic()
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
        assert time.monotonic() - started < 8
        assert "chatomon studio stopped" in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()


def test_the_drawer_counts_changes_and_resets_them(page: Page, tunable: Studio):
    open_studio(page, tunable)
    changes = page.locator("#changes-text")
    expect(changes).to_have_text("Same as tests:atom")
    expect(page.locator("#drawer-badge")).to_be_hidden()
    expect(row(page, "scope").locator(".mod-dot")).to_be_hidden()

    row(page, "scope").get_by_role("switch").focus()
    page.keyboard.press("Space")
    expect(changes).to_have_text("1 change vs code")
    expect(page.locator("#drawer-badge")).to_be_visible()
    expect(row(page, "scope").locator(".mod-dot")).to_be_visible()
    changes.click()
    expect(page.locator("#changes-list")).to_have_text("scope turned off")

    page.get_by_role("button", name="Reset to code").click()
    expect(changes).to_have_text("Same as tests:atom")
    toast = page.locator("#toast")
    expect(toast).to_contain_text("Reset to code")
    toast.get_by_role("button", name="Undo").click()
    expect(changes).to_have_text("1 change vs code")
    expect(row(page, "scope").get_by_role("switch")).not_to_be_checked()


def test_dragging_an_orbital_reorders_the_chain(page: Page, tunable: Studio):
    open_studio(page, tunable)
    assert input_order(page) == ["message_length", "scope"]
    row(page, "scope").locator(".orb-row").drag_to(
        row(page, "message_length").locator(".orb-row"), target_position={"x": 40, "y": 2}
    )
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    assert input_order(page) == ["scope", "message_length"]
    page.locator("#changes-text").click()
    expect(page.locator("#changes-list")).to_have_text("input order changed")


def test_a_page_that_is_out_of_date_reloads_the_state(page: Page, tunable: Studio):
    origin = {"origin": tunable.url}
    open_studio(page, tunable)  # the page's requests carry the cookie only after this
    added = page.request.post(
        f"{tunable.url}/api/orbitals",
        data={"side": "input", "kind": "TurnLimitOrbital", "settings": {"max_turns": 3}},
        headers=origin,
    )
    assert added.ok
    page.reload()
    expect(row(page, "turn_limit")).to_contain_text("added in the studio")

    # Another tab deletes the orbital; this page still shows it.
    gone = page.request.delete(
        f"{tunable.url}/api/orbitals", data={"side": "input", "name": "turn_limit"}, headers=origin
    )
    assert gone.ok
    row(page, "turn_limit").get_by_role("switch").click()
    expect(page.locator("#toast")).to_contain_text("no input orbital named 'turn_limit'")
    expect(row(page, "turn_limit")).to_have_count(0)


def test_deleting_an_added_orbital_and_undoing(page: Page, tunable: Studio):
    page.goto(f"{tunable.url}/?token={TOKEN}")
    page.request.post(
        f"{tunable.url}/api/orbitals",
        data={"side": "input", "kind": "TurnLimitOrbital", "settings": {"max_turns": 3}},
        headers={"origin": tunable.url},
    )
    page.reload()
    expect(row(page, "scope").get_by_role("button", name="Delete scope")).to_have_count(0)
    row(page, "turn_limit").get_by_role("button", name="Delete turn_limit").click()
    expect(row(page, "turn_limit")).to_have_count(0)
    toast = page.locator("#toast")
    expect(toast).to_contain_text("turn_limit deleted")
    toast.get_by_role("button", name="Undo").click()
    expect(row(page, "turn_limit")).to_have_count(1)
