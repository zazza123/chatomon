"""The Spectral add-on of the page in Chromium: the ghost, the chip and the card."""

import sys
from collections.abc import Iterator
from typing import Any

import pytest
from playwright.sync_api import Page, Route, expect
from pydantic_ai.models.function import FunctionModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._spectral import Bridge
from tests.fakes import ScriptedOrbital
from tests.studio.spectral.fake_relay import RelayThread
from tests.studio.test_e2e import REPLY, Studio, open_studio, respond, running

pytestmark = pytest.mark.e2e
MARKUP = '<img src=x onerror="window.pwned=1">'


def an_atom() -> Atom:
    return Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(FunctionModel(respond)),
        input_orbitals=[ScriptedOrbital(name="guard")],
        output_orbitals=[ScriptedOrbital(name="judge")],
    )


@pytest.fixture
def relay(monkeypatch: pytest.MonkeyPatch) -> Iterator[RelayThread]:
    """A fake relay in a thread of its own, and the environment of a studio that uses it."""
    thread = RelayThread()
    with thread as fake:
        monkeypatch.setenv("SPECTRAL_BRIDGE_API_KEY", fake.key)
        monkeypatch.setenv("SPECTRAL_BRIDGE_RELAY_URL", fake.url)
        monkeypatch.setattr(Bridge, "insecure", True)
        yield thread


@pytest.fixture
def studio(relay: RelayThread) -> Iterator[Studio]:
    with running(an_atom()) as handle:
        yield handle


def chip(page: Page):
    return page.locator("#spectral-chip")


def card(page: Page):
    return page.locator("dialog[open]")


def ghost(page: Page):
    return page.locator("#atom #gh")


def connect(page: Page) -> None:
    chip(page).click()
    card(page).get_by_role("button", name="Connect").click()
    expect(chip(page)).to_contain_text("connected")


def test_the_ghost_sleeps_on_the_stage_and_the_chip_opens_the_card(page: Page, studio: Studio):
    open_studio(page, studio)
    expect(chip(page)).to_have_text("Spectral")
    expect(chip(page)).to_have_attribute(
        "aria-label", "Spectral is not connected. Open the Spectral card"
    )
    expect(ghost(page)).to_have_class("ghost pose-sleep")
    expect(page.locator(".legend")).to_contain_text(
        "Click a dash, the nucleus or the ghost to open it"
    )
    expect(page.locator("#spectral-live")).to_have_text("Spectral is not connected")

    chip(page).focus()
    page.keyboard.press("Enter")
    expect(card(page).get_by_role("heading", name="Spectral")).to_be_visible()
    for row in ("Key", "Relay", "Adapter", "Atom"):
        expect(card(page).locator(".check .label", has_text=row)).to_be_visible()
    expect(card(page).get_by_label("Key")).to_have_value("SPECTRAL_BRIDGE_API_KEY")
    expect(card(page).locator("#sp-env-pill")).to_have_text("set")
    expect(card(page)).to_contain_text("While connected, Spectral can send messages to this atom.")
    expect(card(page)).to_contain_text("Closed. It opens when you connect.")
    expect(card(page).get_by_role("link", name="Open Spectral")).to_have_attribute(
        "href", "https://spectral.principled.app"
    )
    page.keyboard.press("Escape")
    expect(card(page)).to_have_count(0)
    expect(chip(page)).to_be_focused()


def test_the_ghost_on_the_stage_opens_the_card_too(page: Page, studio: Studio):
    open_studio(page, studio)
    # It never stops swaying: the click does not wait for it to stand still.
    ghost(page).click(force=True)
    expect(card(page).get_by_role("heading", name="Spectral")).to_be_visible()


def test_connecting_is_shown_by_the_chip_the_ghost_and_the_card(page: Page, studio: Studio):
    open_studio(page, studio)
    connect(page)
    expect(chip(page)).to_have_attribute("data-tone", "on")
    expect(page.locator("#spectral-live")).to_have_text("Connected to Spectral")
    expect(ghost(page)).to_have_class("ghost pose-orbit")
    expect(card(page).get_by_label("Key")).to_be_disabled()
    expect(card(page)).to_contain_text("Disconnect to change it.")
    expect(card(page)).to_contain_text("127.0.0.1:")
    expect(card(page)).to_contain_text("Spectral can reach this atom until you disconnect")
    expect(card(page)).to_contain_text("Next: in Spectral, open your target")
    expect(card(page).get_by_role("button", name="Disconnect")).to_be_visible()


def test_requests_move_the_counts_while_the_card_stays_open(
    page: Page, studio: Studio, relay: RelayThread
):
    open_studio(page, studio)
    connect(page)
    name = card(page).get_by_label("Key")
    page.evaluate("document.querySelector('#dlg-body').dataset.same = 'yes'")

    first = relay.ask("/v1/responses", {"input": "Are you open?"})
    again = {"input": [{"role": "user", "content": "And on Sunday?"}]}
    second = relay.ask("/v1/responses", {**again, "previous_response_id": first["body"]["id"]})
    assert (first["status"], second["status"]) == (200, 200)
    assert second["body"]["output"][0]["content"][0]["text"] == REPLY

    stats = card(page).locator("#sp-stats")
    expect(stats.locator(".stat-nums b").first).to_have_text("2")
    expect(stats).to_contain_text("answered 2")
    guard = stats.locator("tbody tr", has_text="guard")
    expect(guard.locator("td").first).to_have_text("2")
    expect(card(page).locator("#sp-kept")).to_contain_text("2 conversations are held in memory")
    expect(card(page).locator("#sp-next")).to_be_hidden()
    # The numbers changed in place: the card was not drawn again under the developer's hands.
    expect(page.locator("#dlg-body")).to_have_attribute("data-same", "yes")
    expect(name).to_be_disabled()

    card(page).get_by_role("button", name="Reset the counts").click()
    expect(stats.locator(".stat-nums b").first).to_have_text("0")
    expect(page.locator("#toast")).to_contain_text("The counts start again")

    card(page).get_by_role("button", name="Disconnect").click()
    expect(chip(page)).to_have_text("Spectral")
    expect(card(page).get_by_role("button", name="Connect")).to_be_visible()
    page.keyboard.press("Escape")
    expect(ghost(page)).to_have_class("ghost pose-sleep")


def test_while_requests_arrive_the_stage_says_what_the_atom_is_doing(
    page: Page, studio: Studio, relay: RelayThread
):
    open_studio(page, studio)
    connect(page)
    page.keyboard.press("Escape")
    relay.ask("/v1/responses", {"input": "Are you open?"})
    expect(chip(page)).to_contain_text("1 request")
    expect(ghost(page)).to_have_class("ghost pose-evaluate")
    expect(page.locator("#phase-title")).to_have_text("Answering Spectral")
    expect(page.locator("#phase-sub")).to_have_text("Its requests are counted, not shown here")
    # A moment after the last request the atom waits for a message again.
    expect(page.locator("#phase-title")).to_have_text("Waiting for a message", timeout=8000)
    expect(ghost(page)).to_have_class("ghost pose-orbit")
    # Nothing of the request is in the conversation of the page.
    expect(page.locator("article.turn")).to_have_count(0)


def test_a_name_that_is_not_set_is_said_under_the_field_and_stays_typed(page: Page, studio: Studio):
    open_studio(page, studio)
    chip(page).click()
    name = card(page).get_by_label("Key")
    name.fill("A_VARIABLE_NOBODY_SET")
    expect(card(page).locator("#sp-env-pill")).to_be_hidden()
    name.press("Enter")
    problem = card(page).locator("#sp-error")
    expect(problem).to_have_text("the environment variable that api_key_env names is not set")
    expect(name).to_have_value("A_VARIABLE_NOBODY_SET")
    expect(name).to_be_focused()
    expect(chip(page)).to_have_text("Spectral")
    # Typing again takes the message away.
    name.fill("SPECTRAL_BRIDGE_API_KEY")
    expect(problem).to_be_hidden()
    expect(card(page).locator("#sp-env-pill")).to_have_text("set")


def test_a_key_the_relay_refuses_startles_the_ghost(page: Page, studio: Studio, relay: RelayThread):
    relay.relay.refuse = "close-4001"
    open_studio(page, studio)
    chip(page).click()
    card(page).get_by_role("button", name="Connect").click()
    expect(chip(page)).to_contain_text("key refused")
    expect(chip(page)).to_have_attribute("data-tone", "block")
    expect(card(page)).to_contain_text(
        "Spectral refused the key. Check the variable and connect again."
    )
    expect(card(page).get_by_role("button", name="Connect")).to_be_visible()
    page.keyboard.press("Escape")
    expect(ghost(page)).to_have_class("ghost pose-refused")


def test_a_test_request_says_how_the_atom_answered(page: Page, studio: Studio):
    open_studio(page, studio)
    chip(page).click()
    card(page).get_by_role("button", name="Send a test request").click()
    atom_row = card(page).locator(".check", has_text="Atom")
    expect(atom_row.locator(".pill")).to_have_text("answered")
    # It is not a turn of the conversation, and it is not counted.
    expect(page.locator("article.turn")).to_have_count(0)
    expect(card(page).locator("#sp-stats")).to_have_count(0)


def spoiled(change: dict[str, Any]):
    """Answer the page with the state the server gives, changed as a test wants it."""

    def answer(route: Route) -> None:
        response = route.fetch()
        state = response.json()
        target = state["studio"]["spectral"] if "studio" in state else state
        target.update(change)
        route.fulfill(response=response, json=state)

    return answer


def test_what_the_server_says_is_shown_as_text(page: Page, studio: Studio):
    """Names come from the developer's code and addresses from the environment: none is markup."""
    row = {"name": MARKUP, "kind": MARKUP, "side": "input", "allow": 1, "block": 0, "respond": 0}
    row |= {"error": 0, "not_run": 0, "runs": 1, "duration_ms_total": 2.0}
    final = dict.fromkeys(("guided_refusal", "orbital_response", "blocked", "fallback_refusal"), 0)
    counts = {"requests": 1, "running": 0, "failed": 0, "rejected": 0, "duration_ms_total": 5.0}
    counts |= {"configurations": 1, "final": {"answered": 1, **final}, "orbitals": [row]}
    change = {
        "message": MARKUP,
        "relay": MARKUP,
        "notes": [MARKUP],
        "api_key_env": MARKUP,
        "test": {"outcome": "failed", "final_action": None, "message": MARKUP, "duration_ms": 1},
        "counts": counts,
        "site": "javascript:window.pwned=1",
    }
    page.route("**/api/state", spoiled(change))
    page.route("**/api/spectral", spoiled(change))
    open_studio(page, studio)
    chip(page).click()
    expect(card(page).locator("#sp-relay")).to_contain_text("<img src=x")
    expect(card(page).locator(".note li")).to_contain_text("<img src=x")
    expect(card(page).locator("tbody th b")).to_contain_text("<img src=x")
    expect(card(page).get_by_label("Key")).to_have_value(MARKUP)
    expect(card(page).locator("img")).to_have_count(0)
    # An address that is not a page's is no link at all.
    expect(card(page).get_by_role("link")).to_have_count(0)
    assert page.evaluate("window.pwned") is None


def test_on_a_network_address_the_card_says_why_it_cannot_connect(page: Page, studio: Studio):
    page.route("**/api/state", spoiled({"local": False}))
    page.route("**/api/spectral", spoiled({"local": False}))
    open_studio(page, studio)
    chip(page).click()
    expect(card(page)).to_contain_text(
        "Spectral can be connected only when the studio listens on this machine."
    )
    expect(card(page).get_by_role("button", name="Connect")).to_have_count(0)
    expect(card(page).get_by_label("Key")).to_have_count(0)


def test_without_the_extra_the_page_asks_for_nothing_of_the_add_on(
    page: Page, monkeypatch: pytest.MonkeyPatch
):
    for name in ("spectral_bridge", "spectral_bridge.cli", "spectral_bridge.cli.defaults"):
        monkeypatch.setitem(sys.modules, name, None)
    asked: list[str] = []
    page.on("request", lambda request: asked.append(request.url))
    with running(an_atom()) as handle:
        open_studio(page, handle)
        expect(page.locator(".legend")).to_contain_text("Click a dash or the nucleus to open it")
        expect(chip(page)).to_have_count(0)
        expect(ghost(page)).to_have_count(0)
    assert asked and not [url for url in asked if "spectral" in url]


def test_on_a_phone_the_chip_is_the_ghost_alone_and_nothing_overflows(page: Page, studio: Studio):
    page.set_viewport_size({"width": 400, "height": 800})
    open_studio(page, studio)
    expect(chip(page).locator("svg")).to_be_visible()
    expect(chip(page).locator(".txt")).to_be_hidden()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    chip(page).click()
    expect(card(page).get_by_role("button", name="Connect")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_with_reduced_motion_the_ghost_stands_still(page: Page, studio: Studio):
    page.emulate_media(reduced_motion="reduce")
    open_studio(page, studio)
    connect(page)
    page.keyboard.press("Escape")
    expect(ghost(page)).to_have_class("ghost pose-orbit")
    placed = ghost(page).locator("g").first
    before = placed.get_attribute("transform")
    page.wait_for_timeout(500)
    assert before and placed.get_attribute("transform") == before


def test_in_the_dark_theme_the_ghost_keeps_its_colours(page: Page, studio: Studio):
    page.emulate_media(color_scheme="dark")
    open_studio(page, studio)
    connect(page)
    page.keyboard.press("Escape")
    expect(ghost(page).locator(".g-body")).to_have_css("fill", "rgb(125, 247, 214)")
    expect(ghost(page).locator(".g-body")).to_have_css("stroke", "rgb(125, 247, 214)")
    expect(chip(page)).to_have_css("color", "rgb(125, 247, 214)")
