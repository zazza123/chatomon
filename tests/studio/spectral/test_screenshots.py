"""The screenshot of the Spectral add-on that the documentation shows."""

import re
from collections.abc import Iterator

import pytest
from playwright.sync_api import expect

from chatomon.studio._spectral import Bridge
from tests.studio.shots import Camera, reword
from tests.studio.spectral.fake_relay import RelayThread
from tests.studio.spectral.test_e2e import card, chip, connect
from tests.studio.test_e2e import running
from tests.studio.test_screenshots import HOURS, POEM, SOURCE, opened, speedypack

pytestmark = pytest.mark.e2e

RELAY = "bridge.spectral.principled.app"
ADAPTER = "127.0.0.1:52100"
SINCE = "since 10:42 AM"


@pytest.fixture
def relay(monkeypatch: pytest.MonkeyPatch) -> Iterator[RelayThread]:
    """A fake relay in a thread of its own, and the environment of a studio that uses it."""
    thread = RelayThread()
    with thread as fake:
        monkeypatch.setenv("SPECTRAL_BRIDGE_API_KEY", fake.key)
        monkeypatch.setenv("SPECTRAL_BRIDGE_RELAY_URL", fake.url)
        monkeypatch.setattr(Bridge, "insecure", True)
        yield thread


def test_spectral(camera: Camera, relay: RelayThread):
    with running(speedypack(), source=SOURCE) as studio:
        page = opened(camera, studio)
        connect(page)
        for message in (HOURS, HOURS, POEM, HOURS):
            assert relay.ask("/v1/responses", {"input": message})["status"] == 200
        expect(card(page).locator("#sp-stats .sp-nums b").first).to_have_text("4")
        # The chip counts the requests for two seconds after the last one.
        expect(chip(page)).to_contain_text("connected", timeout=8000)
        # The card asks the studio for news every second and writes again what differs
        # from it. The asking is stopped, so what is reworded below stays reworded.
        with page.expect_event(
            "requestfailed", lambda request: request.url.endswith("/api/spectral")
        ):
            page.route("**/api/spectral", lambda route: route.abort())
        # The relay is a fake one on this machine, the adapter listens on a port the
        # system chose and the connection was opened just now: the image shows the
        # relay of Spectral, one port and one time.
        said = card(page).inner_text()
        adapter, since = re.search(r"127\.0\.0\.1:\d+", said), re.search(r"since [\d:]+ ?\w*", said)
        assert adapter is not None and since is not None
        reword(page, adapter.group(), ADAPTER)
        reword(page, since.group(), SINCE)
        card(page).locator(".sp-what:has(#sp-relay) .mono").evaluate(
            "(shown, relay) => { shown.textContent = relay; }", RELAY
        )
        expect(card(page).locator(".sp-what:has(#sp-relay) .mono")).to_have_text(RELAY)
        expect(card(page).locator("#sp-relay")).to_have_text(f"Connected to Spectral · {SINCE}")
        expect(card(page)).to_contain_text(ADAPTER)
        camera.shoot("spectral", card(page))
