"""The studio page in Chromium: turns, reports, switches and the ways a turn can go wrong."""

import asyncio
import contextlib
import http.client
import json
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
from typing import Any, Literal

import pytest
import uvicorn
from playwright.sync_api import Page, expect
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital
from chatomon.studio import _kinds as kinds
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
def running(atom: Atom, *, namespace: dict[str, object] | None = None) -> Iterator[Studio]:
    port = free_port()
    session = Session(atom, source="tests:atom", namespace=namespace)
    app = create_app(session, token=TOKEN, host="127.0.0.1", port=port)
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


def card(page: Page):
    return page.locator("dialog[open]")


def open_card(page: Page, name: str) -> None:
    # The row's button is named by its content (name and class); "Open <name>" is its tooltip.
    row(page, name).locator('[data-act="card"]').click()


def test_editing_a_setting_and_a_value_the_orbital_refuses(page: Page, tunable: Studio):
    open_studio(page, tunable)
    open_card(page, "message_length")
    words = card(page).get_by_label("max_words")
    expect(words).to_have_value("60")
    fingerprint = card(page).locator("#card-fp").get_attribute("title")
    reset = card(page).get_by_role("button", name="Reset to code")
    expect(reset).to_be_disabled()

    words.fill("80")
    words.press("Tab")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    expect(card(page).locator("#card-fp")).not_to_have_attribute("title", fingerprint or "")
    expect(reset).to_be_enabled()
    expect(row(page, "message_length").locator(".mod-dot")).to_be_visible()

    words.fill("")
    words.press("Tab")
    error = card(page).locator("#f-max_words-err")
    expect(error).to_have_text("set max_words, max_chars or both")
    expect(words).to_have_attribute("aria-invalid", "true")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")

    words.fill("1.5")
    words.press("Tab")
    expect(error).to_have_text("max_words must be a whole number")
    expect(error.locator("code")).to_have_text("max_words")

    reset.click()
    expect(card(page).get_by_label("max_words")).to_have_value("60")
    expect(page.locator("#changes-text")).to_have_text("Same as tests:atom")
    expect(page.locator("#toast")).to_contain_text("message_length reset to code")


def test_an_orbital_from_code_that_the_studio_does_not_edit(page: Page, tunable: Studio):
    open_studio(page, tunable)
    open_card(page, "scope")
    expect(card(page)).to_contain_text("ScriptedOrbital is your own class")
    expect(card(page).locator(".field")).to_have_count(0)
    expect(card(page).get_by_role("button", name="Delete orbital")).to_have_count(0)

    card(page).get_by_role("button", name="Move up").click()
    expect(card(page).locator("#card-pos")).to_have_text("1 of 2 in the input chain")
    expect(card(page).get_by_role("button", name="Move down")).to_be_focused()
    assert input_order(page) == ["scope", "message_length"]

    card(page).get_by_role("switch").click()
    expect(row(page, "scope").get_by_role("switch")).not_to_be_checked()
    expect(card(page)).to_be_visible()


def test_adding_an_orbital_with_a_name_renaming_and_deleting_it(page: Page, tunable: Studio):
    open_studio(page, tunable)
    page.locator('#drawer [data-act="add"][data-side="input"]').click()
    expect(card(page)).to_contain_text("Closes the conversation after a number of turns.")
    expect(card(page)).to_contain_text("added in code")
    card(page).get_by_role("button", name="TurnLimitOrbital").click()

    name = card(page).get_by_label("name", exact=True)
    expect(name).to_have_value("turn_limit")
    card(page).get_by_role("button", name="Add", exact=True).click()
    expect(card(page).locator("#f-max_turns-err")).to_have_text("max_turns needs a value")
    assert input_order(page) == ["message_length", "scope"]

    name.fill("limit")
    card(page).get_by_label("max_turns").fill("3")
    card(page).get_by_role("button", name="Add", exact=True).click()
    expect(card(page).locator("#card-title")).to_have_text("limit")
    expect(row(page, "limit")).to_contain_text("added in the studio")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")

    renamed = card(page).get_by_label("name", exact=True)
    renamed.fill("Limit")
    renamed.press("Enter")
    expect(card(page).locator("#f-name-err")).to_contain_text("invalid name 'Limit'")
    renamed.fill("turns")
    renamed.press("Enter")
    expect(card(page).locator("#card-title")).to_have_text("turns")
    expect(row(page, "turns")).to_have_count(1)

    card(page).get_by_role("button", name="Delete orbital").click()
    expect(row(page, "turns")).to_have_count(0)
    page.locator("#toast").get_by_role("button", name="Undo").click()
    expect(row(page, "turns")).to_have_count(1)


def test_copy_as_code_shows_the_lists_and_what_it_could_not_write(page: Page, tunable: Studio):
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    open_studio(page, tunable)
    open_card(page, "message_length")
    words = card(page).get_by_label("max_words")
    words.fill("80")
    words.press("Tab")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    card(page).get_by_role("button", name="Done").click()

    page.get_by_role("button", name="Copy as code").click()
    expect(card(page)).to_contain_text("Studio changes: message_length.max_words changed.")
    expect(card(page).locator("pre.code")).to_contain_text("MessageLengthOrbital(max_words=80),")
    expect(card(page)).to_contain_text("scope: a ScriptedOrbital from your code")
    expect(card(page).locator("p.note code")).to_have_text(["..."])
    card(page).get_by_role("button", name="Copy", exact=True).click()
    expect(page.locator("#toast")).to_contain_text("Copied")
    assert "MessageLengthOrbital(max_words=80)," in page.evaluate("navigator.clipboard.readText()")


def test_exporting_resetting_and_importing_restores_the_session(page: Page, tunable: Studio):
    open_studio(page, tunable)
    row(page, "scope").get_by_role("switch").click()
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")

    page.get_by_role("button", name="Export preset").click()
    expect(card(page).locator("pre.code")).to_contain_text('"chatomon_studio_preset": 1')
    text = card(page).locator("pre.code").inner_text()
    with page.expect_download() as download:
        card(page).get_by_role("link", name="Download").click()
    assert download.value.suggested_filename == "speedypack.preset.json"
    card(page).get_by_role("button", name="Close").first.click()

    page.get_by_role("button", name="Reset to code").click()
    expect(page.locator("#changes-text")).to_have_text("Same as tests:atom")

    page.get_by_role("button", name="Import preset").click()
    card(page).locator("#import-file").set_input_files(
        {"name": "speedypack.preset.json", "mimeType": "application/json", "buffer": text.encode()}
    )
    expect(card(page).get_by_label("Preset JSON")).to_have_value(text)
    card(page).get_by_role("button", name="Apply preset").click()
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    expect(row(page, "scope").get_by_role("switch")).not_to_be_checked()
    toast = page.locator("#toast")
    expect(toast).to_contain_text("Preset imported")
    toast.get_by_role("button", name="Undo").click()
    expect(page.locator("#changes-text")).to_have_text("Same as tests:atom")


def test_an_invalid_preset_lists_its_problems_and_changes_nothing(page: Page, tunable: Studio):
    open_studio(page, tunable)
    page.get_by_role("button", name="Import preset").click()
    card(page).get_by_label("Preset JSON").fill('{"chatomon_studio_preset": 2}')
    card(page).get_by_role("button", name="Apply preset").click()
    problems = card(page).locator("#import-problems")
    expect(problems).to_contain_text("The preset was not applied")
    expect(problems.locator("li")).to_have_text(
        ["this preset has schema version 2; this studio reads version 1"]
    )
    expect(page.locator("#changes-text")).to_have_text("Same as tests:atom")


def test_a_preset_applied_with_notes_lists_them(page: Page, tunable: Studio):
    open_studio(page, tunable)
    preset = page.request.get(f"{tunable.url}/api/preset").json()
    preset["input_orbitals"][0]["fingerprint"] = "sha256:another"
    del preset["input_orbitals"][1]
    page.get_by_role("button", name="Import preset").click()
    card(page).get_by_label("Preset JSON").fill(json.dumps(preset))
    card(page).get_by_role("button", name="Apply preset").click()
    expect(card(page)).to_contain_text("Preset applied")
    expect(card(page).locator("li")).to_have_text(
        [
            "'scope' is in your code but not in the preset: it was added at the end, turned on.",
            "'message_length' differs from the exported preset: "
            "your code or chatomon changed since.",
        ]
    )
    card(page).get_by_role("button", name="Undo").click()
    expect(card(page)).to_have_count(0)


@pytest.fixture
def readers() -> Iterator[Studio]:
    """An atom whose second orbital reads the first, so some changes are refused."""
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
        input_orbitals=[
            ScriptedOrbital(Verdict.allow(annotations={"x": 1}), name="first"),
            ScriptedOrbital(name="reader", reads=["first"]),
        ],
    )
    with running(atom) as handle:
        yield handle


def on_top(locator) -> bool:
    """Whether a click at the element's centre would reach it, not something over it."""
    return locator.evaluate(
        """el => {
          const box = el.getBoundingClientRect();
          const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
          return el === hit || el.contains(hit);
        }"""
    )


def test_undo_after_a_reset_from_the_card_can_be_clicked(page: Page, tunable: Studio):
    open_studio(page, tunable)
    open_card(page, "message_length")
    words = card(page).get_by_label("max_words")
    words.fill("80")
    words.press("Tab")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    card(page).get_by_role("button", name="Reset to code").click()
    expect(card(page).get_by_label("max_words")).to_have_value("60")

    undo = page.locator("#toast").get_by_role("button", name="Undo")
    assert on_top(undo)
    undo.click(timeout=3000)
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    expect(card(page).get_by_label("max_words")).to_have_value("80")


def test_a_refusal_from_the_card_can_be_read(page: Page, readers: Studio):
    open_studio(page, readers)
    open_card(page, "first")
    card(page).get_by_role("switch").click()
    reason = page.locator("#toast-text")
    expect(reason).to_contain_text("reads 'first'")
    assert on_top(reason)
    expect(card(page).get_by_role("switch")).to_be_checked()
    card(page).get_by_role("button", name="Move down").click()
    expect(reason).to_contain_text("reads 'first'")
    expect(card(page).locator("#card-pos")).to_have_text("1 of 2 in the input chain")


def test_a_refused_value_followed_by_done_is_reported(page: Page, tunable: Studio):
    open_studio(page, tunable)
    open_card(page, "message_length")
    card(page).get_by_label("max_words").fill("0")
    card(page).get_by_role("button", name="Done").click()
    expect(card(page)).to_have_count(0)
    expect(page.locator("#toast")).to_contain_text("max_words must be a positive integer, got 0")
    expect(page.locator("#changes-text")).to_have_text("Same as tests:atom")


def reply_on_the_server(page: Page, studio: Studio) -> str:
    state = page.request.get(f"{studio.url}/api/state").json()
    settings = state["orbitals"]["input"][0]["settings"]
    return next(setting["value"] for setting in settings if setting["key"] == "reply")


def test_enter_applies_a_text_setting_and_shift_enter_breaks_the_line(page: Page, tunable: Studio):
    open_studio(page, tunable)
    open_card(page, "message_length")
    reply = card(page).get_by_label("reply")
    reply.fill("Too long.")
    reply.press("Enter")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    expect(reply).to_have_value("Too long.")
    assert reply_on_the_server(page, tunable) == "Too long."

    reply.press("Shift+Enter")
    reply.press_sequentially("Shorten it.")
    expect(reply).to_have_value("Too long.\nShorten it.")
    assert reply_on_the_server(page, tunable) == "Too long."
    fingerprint = card(page).locator("#card-fp").get_attribute("title")
    reply.press("Enter")
    expect(card(page).locator("#card-fp")).not_to_have_attribute("title", fingerprint or "")
    assert reply_on_the_server(page, tunable) == "Too long.\nShorten it."


def test_text_that_is_not_a_number_is_refused_not_read_as_none(page: Page, tunable: Studio):
    open_studio(page, tunable)
    open_card(page, "message_length")
    chars = card(page).get_by_label("max_chars")
    chars.fill("500")
    chars.press("Tab")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    words = card(page).get_by_label("max_words")
    words.fill("")
    words.press_sequentially("3e")
    words.press("Tab")
    expect(card(page).locator("#f-max_words-err")).to_have_text("max_words must be a whole number")
    state = page.request.get(f"{tunable.url}/api/state").json()
    assert state["orbitals"]["input"][0]["settings"][0]["value"] == 60


def test_undo_stays_in_the_drawer_and_focus_is_never_lost(page: Page, tunable: Studio):
    open_studio(page, tunable)
    undo = page.locator("#undo")
    expect(undo).to_be_hidden()
    row(page, "scope").get_by_role("switch").click()
    page.get_by_role("button", name="Reset to code").click()
    expect(undo).to_have_text("Undo: Reset to code")
    expect(undo).to_be_focused()

    # The toast fades after five seconds; its button must not stay as an invisible stop.
    expect(page.locator("#toast-act")).to_be_hidden(timeout=8000)
    page.keyboard.press("Enter")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    expect(undo).to_be_hidden()
    expect(page.get_by_role("button", name="Reset to code")).to_be_focused()

    open_card(page, "message_length")
    words = card(page).get_by_label("max_words")
    words.fill("80")
    words.press("Tab")
    card(page).get_by_role("button", name="Done").click()
    expect(row(page, "message_length").locator('[data-act="card"]')).to_be_focused()


def test_selecting_text_past_the_edge_of_a_card_does_not_close_it(page: Page, tunable: Studio):
    open_studio(page, tunable)
    open_card(page, "message_length")
    box = card(page).get_by_label("reply").bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + 30, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(8, 8, steps=5)
    page.mouse.up()
    expect(card(page)).to_be_visible()
    page.mouse.click(8, 8)
    expect(card(page)).to_have_count(0)


def test_the_tab_key_skips_the_list_of_changes_when_there_are_none(page: Page, tunable: Studio):
    open_studio(page, tunable)
    last_add = page.locator('#drawer [data-act="add"][data-side="output"]')
    last_add.focus()
    page.keyboard.press("Tab")
    expect(page.get_by_role("button", name="Copy as code")).to_be_focused()

    row(page, "scope").get_by_role("switch").click()
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    last_add.focus()
    page.keyboard.press("Tab")
    expect(page.locator("#changes-text")).to_be_focused()


def test_each_add_button_says_which_chain_it_adds_to(page: Page, tunable: Studio):
    open_studio(page, tunable)
    page.get_by_role("button", name="Add an output orbital").click()
    expect(card(page).locator("#dlg-title")).to_have_text("Add an output orbital")
    card(page).get_by_role("button", name="Cancel").click()
    page.get_by_role("button", name="Add an input orbital").click()
    expect(card(page).locator("#dlg-title")).to_have_text("Add an input orbital")


class TriOrbital(ScriptedOrbital):
    """A choice and a switch that may be none: no built-in orbital has one yet."""

    def __init__(
        self,
        *,
        mode: Literal["strict", "loose"] | None = None,
        careful: bool | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.mode = mode
        self.careful = careful

    def config(self) -> dict[str, Any]:
        return {**super().config(), "mode": self.mode, "careful": self.careful}


@pytest.fixture
def optional(monkeypatch) -> Iterator[Studio]:
    """A studio that reads TriOrbital as one of chatomon's own, so its card can be edited."""
    builtin = kinds.is_builtin
    monkeypatch.setattr(kinds, "is_builtin", lambda cls: cls is TriOrbital or builtin(cls))
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
        input_orbitals=[TriOrbital(mode="strict", careful=True)],
    )
    with running(atom) as handle:
        yield handle


def test_a_choice_and_a_switch_that_may_be_none(page: Page, optional: Studio):
    def values() -> dict[str, Any]:
        state = page.request.get(f"{optional.url}/api/state").json()
        return {s["key"]: s["value"] for s in state["orbitals"]["input"][0]["settings"]}

    open_studio(page, optional)
    open_card(page, "tri")
    mode = card(page).get_by_label("mode")
    careful = card(page).get_by_label("careful")
    expect(mode).to_have_value("strict")
    expect(careful).to_have_value("true")

    mode.select_option("")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    careful.select_option("")
    expect(page.locator("#changes-text")).to_have_text("2 changes vs code")
    assert values() == {"mode": None, "careful": None}

    mode.select_option("loose")
    careful.select_option("false")
    expect(card(page).locator(".field.changed")).to_have_count(2)
    page.wait_for_function("() => document.querySelectorAll('[aria-invalid]').length === 0")
    assert values() == {"mode": "loose", "careful": False}


LONG = "one two three four five six"
SAVED = "tests/studio/test_one_two_three_four_five_six.py"


@pytest.fixture
def saving(tmp_path, monkeypatch) -> Iterator[Studio]:
    """A studio started from a module, in a directory of its own: it can write tests."""
    monkeypatch.chdir(tmp_path)
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
        input_orbitals=[
            MessageLengthOrbital(max_words=5, reply="Too long."),
            ScriptedOrbital(name="scope"),
        ],
    )
    with running(atom, namespace={"atom": atom}) as handle:
        yield handle


def send(page: Page, text: str, reply: str) -> None:
    """Send a message and wait for its turn to show the reply."""
    count = page.locator("article.turn").count()
    message_box(page).fill(text)
    page.get_by_role("button", name="Send").click()
    expect(page.locator("article.turn").nth(count)).to_contain_text(reply)


def open_report(page: Page, n: int) -> None:
    page.locator("article.turn").nth(n - 1).get_by_role("button", name=f"Inspect turn {n}").click()


def save_as_test(page: Page, n: int) -> None:
    open_report(page, n)
    card(page).get_by_role("button", name="Save as test").click()
    expect(card(page).locator("#dlg-title")).to_have_text(f"Save turn {n} as a test")
    # The report's button is gone with the report: the focus stays in the card, on its title.
    expect(card(page).locator("#dlg-title")).to_be_focused()


def test_saving_a_turn_writes_a_test_and_asks_before_replacing_it(page: Page, saving, tmp_path):
    open_studio(page, saving)
    send(page, LONG, "Too long.")
    save_as_test(page, 1)
    expect(card(page)).to_contain_text("No model is called in this test")
    expect(card(page).locator("pre.code")).to_contain_text("def test_one_two_three_four_five_six(")
    path = card(page).get_by_label("File, inside")
    expect(path).to_have_value(SAVED)

    card(page).get_by_role("button", name="Write file").click()
    done = card(page).locator("#test-done")
    expect(done).to_contain_text(f"Saved as {SAVED}")
    expect(done).to_contain_text(f"python -m pytest {SAVED}")
    assert (
        (tmp_path / SAVED).read_text(encoding="utf-8").startswith('"""Saved from chatomon studio')
    )

    (tmp_path / SAVED).write_text('"""Saved from chatomon studio: old."""\n', encoding="utf-8")
    card(page).get_by_role("button", name="Write file").click()
    message = card(page).locator("#test-path-msg")
    expect(message).to_contain_text("exists and was saved by the studio")
    expect(path).to_have_attribute("aria-invalid", "true")
    expect(done).to_be_hidden()
    assert "old" in (tmp_path / SAVED).read_text(encoding="utf-8")
    overwrite = message.get_by_role("button", name="Overwrite")
    expect(overwrite).to_be_focused()
    overwrite.click()
    expect(done).to_contain_text(f"Saved as {SAVED}")
    expect(done).to_be_focused()  # the button that had the focus is hidden again
    assert "old" not in (tmp_path / SAVED).read_text(encoding="utf-8")

    path.fill("../outside.py")
    path.press("Enter")
    expect(message).to_contain_text("the file must be inside")
    expect(path).to_be_focused()
    assert not (tmp_path.parent / "outside.py").exists()

    card(page).get_by_role("button", name="Back to the report").click()
    expect(card(page).locator("#dlg-title")).to_contain_text("Turn 1")
    expect(card(page).locator("#dlg-title")).to_be_focused()


def test_a_test_holds_the_conversation_or_the_turn_alone(page: Page, saving):
    open_studio(page, saving)
    send(page, "hello", REPLY)
    send(page, LONG, "Too long.")
    save_as_test(page, 2)
    earlier = card(page).get_by_role("checkbox", name="Replay the 1 turn before it")
    expect(earlier).to_be_checked()
    code = card(page).locator("pre.code")
    expect(code).to_contain_text('agent.run_sync("hello")')
    expect(card(page).locator("#dlg-sub")).to_contain_text("marked integration")
    card(page).get_by_label("File, inside").fill("tests/mine.py")

    earlier.click()
    expect(code).not_to_contain_text('agent.run_sync("hello")')
    expect(card(page).locator("#dlg-sub")).to_contain_text("Runs offline")
    expect(card(page).get_by_label("File, inside")).to_have_value("tests/mine.py")
    again = card(page).get_by_role("checkbox", name="Replay the 1 turn before it")
    expect(again).not_to_be_checked()
    expect(again).to_be_focused()


def test_a_turn_tried_with_studio_changes_offers_copy_as_code(page: Page, saving):
    open_studio(page, saving)
    open_card(page, "message_length")
    words = card(page).get_by_label("max_words")
    words.fill("80")
    words.press("Tab")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    card(page).get_by_role("button", name="Done").click()

    send(page, "hello", REPLY)
    open_report(page, 1)
    expect(card(page).locator("#dlg-sub")).to_contain_text("1 change vs code")
    card(page).get_by_role("button", name="Save as test").click()
    warning = card(page).locator(".note.warn")
    expect(warning).to_contain_text(
        "Turn 1 ran with 1 studio change (message_length.max_words changed)."
    )
    expect(warning).to_contain_text("it checks your code, not the studio")
    warning.get_by_role("button", name="Copy as code").click()
    expect(card(page).locator("#dlg-title")).to_have_text("Copy as code")
    expect(card(page).locator("pre.code")).to_contain_text("MessageLengthOrbital(max_words=80")
    card(page).get_by_role("button", name="Back to the test").click()
    expect(card(page).locator("#dlg-title")).to_have_text("Save turn 1 as a test")
    expect(card(page).get_by_role("button", name="Copy as code")).to_be_focused()

    page.keyboard.press("Escape")
    expect(page.get_by_role("button", name="Inspect turn 1")).to_be_focused()


def test_a_turn_with_no_changes_says_so_and_has_no_warning(page: Page, saving):
    open_studio(page, saving)
    send(page, "hello", REPLY)
    open_report(page, 1)
    expect(card(page).locator("#dlg-sub")).to_contain_text("same as code")
    card(page).get_by_role("button", name="Save as test").click()
    expect(card(page).locator("#dlg-title")).to_have_text("Save turn 1 as a test")
    expect(card(page).locator(".note.warn")).to_have_count(0)
    expect(card(page).get_by_role("checkbox")).to_have_count(0)


def test_a_failed_turn_cannot_be_saved(page: Page, saving):
    open_studio(page, saving)
    send(page, "boom", "The turn failed")
    open_report(page, 1)
    expect(card(page).get_by_role("button", name="Save as test")).to_be_disabled()
    expect(card(page)).to_contain_text("A turn that failed cannot be saved as a test.")


def test_started_from_python_the_test_is_only_copied(page: Page, studio: Studio):
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    open_studio(page, studio)
    send(page, "hello", REPLY)
    save_as_test(page, 1)
    expect(card(page)).to_contain_text("The studio was started from Python")
    expect(card(page).get_by_role("button", name="Write file")).to_have_count(0)
    expect(card(page).locator("#test-path")).to_have_count(0)
    card(page).get_by_role("button", name="Copy", exact=True).click()
    expect(page.locator("#toast")).to_contain_text("Copied")
    assert "def test_hello(" in page.evaluate("navigator.clipboard.readText()")


def test_the_test_card_fits_a_phone(page: Page, saving):
    page.set_viewport_size({"width": 390, "height": 800})
    open_studio(page, saving)
    send(page, LONG, "Too long.")
    save_as_test(page, 1)
    path = card(page).get_by_label("File, inside")
    path.fill("../outside.py")
    path.press("Enter")
    expect(card(page).locator("#test-path-msg")).to_contain_text("the file must be inside")
    # The directory's path is long and has no space: in the label and in the message it must
    # wrap, not push the card wider than the screen.
    past_the_edge = card(page).evaluate(
        """dialog => {
            const body = dialog.querySelector("#dlg-body");
            const padding = parseFloat(getComputedStyle(body).paddingRight);
            const limit = body.getBoundingClientRect().right - padding;
            return [...body.querySelectorAll("#test-path, #test-path-msg, pre.code, .field label")]
                .filter((el) => el.getBoundingClientRect().right > limit + 0.5)
                .map((el) => el.id || el.tagName.toLowerCase());
        }"""
    )
    assert past_the_edge == []
