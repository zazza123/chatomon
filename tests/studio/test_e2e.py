"""The studio page in Chromium: a turn, its report, an orbital switched off."""

import socket
import threading
import time

import pytest
import uvicorn
from playwright.sync_api import Page, expect
from pydantic_ai.models.test import TestModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._app import create_app
from chatomon.studio._session import Session
from tests.fakes import ScriptedOrbital

pytestmark = pytest.mark.e2e
TOKEN = "e2e-token"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def studio():
    port = free_port()
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(
            TestModel(custom_output_text="We are open Monday to Friday.")
        ),
        input_orbitals=[ScriptedOrbital(name="guard")],
        output_orbitals=[ScriptedOrbital(name="judge")],
    )
    app = create_app(Session(atom, source="tests:atom"), token=TOKEN, host="127.0.0.1", port=port)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the studio did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_a_turn_its_report_and_a_switch(page: Page, studio: str):
    page.goto(f"{studio}/?token={TOKEN}")
    page.get_by_role("button", name="Real time").click()
    page.get_by_role("textbox", name="Message").fill(
        '<img src=x onerror="window.pwned=1"> Are you open?'
    )
    page.get_by_role("button", name="Send").click()

    turn = page.locator("article.turn").first
    expect(turn).to_contain_text("We are open Monday to Friday.")
    expect(turn.locator(".user .bubble")).to_contain_text("<img src=x")
    assert page.evaluate("window.pwned") is None

    turn.get_by_role("button", name="Inspect turn 1").click()
    report = page.locator("dialog[open]")
    expect(report).to_contain_text("Input chain")
    expect(report).to_contain_text("guard")
    report.get_by_role("button", name="Close").last.click()

    page.locator('#drawer input[data-name="guard"]').uncheck()
    page.get_by_role("textbox", name="Message").fill("Again?")
    page.get_by_role("button", name="Send").click()
    second = page.locator("article.turn").nth(1)
    expect(second).to_contain_text("We are open Monday to Friday.")
    second.get_by_role("button", name="Inspect turn 2").click()
    expect(page.locator("dialog[open]")).to_contain_text("No input orbitals were on.")
