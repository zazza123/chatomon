"""The studio page in Chromium: turns, reports, switches and the ways a turn can go wrong."""

import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import uvicorn
from playwright.sync_api import Page, expect
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
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


def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answer every message, except one that asks for a failure."""
    prompts = [
        part.content
        for part in messages[-1].parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]
    if any("boom" in prompt for prompt in prompts):
        raise RuntimeError("model down")
    return ModelResponse(parts=[TextPart(content=REPLY)])


@dataclass
class Studio:
    url: str
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def studio() -> Iterator[Studio]:
    port = free_port()
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
        input_orbitals=[ScriptedOrbital(name="guard")],
        output_orbitals=[ScriptedOrbital(name="judge")],
    )
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
    yield handle
    handle.stop()


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
