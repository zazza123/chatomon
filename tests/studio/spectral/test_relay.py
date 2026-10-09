"""The state of a connection, read from the real client of spectral-bridge against a fake relay."""

import logging

import pytest

from chatomon import MissingExtraError
from chatomon.studio._spectral import _relay
from chatomon.studio._spectral._relay import Relay
from tests.studio.spectral.fake_relay import FakeRelay
from tests.studio.spectral.helpers import eventually

SECRET = "the-key-of-the-adapter"
ADAPTER = f"http://127.0.0.1:9/{SECRET}"


class Ends:
    def __init__(self) -> None:
        self.times = 0

    def __call__(self) -> None:
        self.times += 1


@pytest.fixture
async def connect(relay: FakeRelay):
    """Open connections that are closed when the test ends."""
    opened: list[Relay] = []

    def open_(key: str = "good-key", *, url: str | None = None) -> tuple[Relay, Ends]:
        ends = Ends()
        link = Relay(url or relay.url, key, ADAPTER, secret=SECRET, on_end=ends, insecure=True)
        opened.append(link)
        return link, ends

    yield open_
    for link in opened:
        link.stop()
        link.stop()
        await link.wait(5)


async def test_a_connection_is_connecting_then_connected(relay, connect):
    link, ends = connect()
    assert link.state() == ("connecting", None)
    await eventually(lambda: link.state() == ("connected", None))
    assert (relay.connections, ends.times) == (1, 0)


@pytest.mark.parametrize("how", ["http-401", "close-4001"])
async def test_a_key_the_relay_refuses_ends_the_connection_with_its_reason(relay, connect, how):
    relay.refuse = how
    link, ends = connect()
    await eventually(lambda: link.state()[0] == "refused")
    assert link.state() == ("refused", "key")
    assert ends.times == 1


async def test_a_wrong_key_is_refused(relay, connect):
    link, _ = connect("wrong-key")
    await eventually(lambda: link.state() == ("refused", "key"))


async def test_another_version_of_the_protocol_is_refused_with_its_reason(relay, connect):
    relay.refuse = "close-4002"
    link, ends = connect()
    await eventually(lambda: link.state() == ("refused", "protocol"))
    assert ends.times == 1


async def test_a_relay_that_does_not_answer_is_tried_again(relay, connect):
    url = relay.url
    await relay.stop()
    link, ends = connect(url=url)
    await eventually(lambda: link.state() == ("connecting", "unreachable"))
    assert ends.times == 0
    await relay.start()
    await eventually(lambda: link.state() == ("connected", None), timeout=10)


async def test_a_connection_that_falls_is_lost_and_then_back(relay, connect):
    link, ends = connect()
    await eventually(lambda: link.state() == ("connected", None))
    await relay.stop()
    await eventually(lambda: link.state() == ("lost", None))
    assert ends.times == 0
    await relay.start()
    await eventually(lambda: link.state() == ("connected", None), timeout=10)
    assert relay.connections >= 2


async def test_a_stop_that_was_asked_for_closes_and_is_off(relay, connect):
    link, ends = connect()
    await eventually(lambda: link.state() == ("connected", None))
    link.stop()
    assert link.state() == ("closing", None)
    await link.wait(5)
    assert link.state() == ("off", None)
    assert ends.times == 1


async def test_a_stop_before_the_relay_answered_is_off_too(relay, connect):
    url = relay.url
    await relay.stop()
    link, ends = connect(url=url)
    await eventually(lambda: link.state() == ("connecting", "unreachable"))
    link.stop()
    await link.wait(5)
    assert (link.state(), ends.times) == (("off", None), 1)


async def test_a_client_that_stops_on_its_own_is_logged_by_the_class_of_its_error(
    relay, connect, monkeypatch, caplog
):
    async def broken(self: object) -> None:
        raise RuntimeError("the key is good-key")

    monkeypatch.setattr("spectral_bridge.RelayClient.run", broken)
    caplog.set_level("WARNING", logger="chatomon")
    link, ends = connect()
    await eventually(lambda: link.state() == ("refused", "stopped"))
    assert ends.times == 1
    assert "the client of spectral-bridge stopped with RuntimeError" in caplog.text
    assert "good-key" not in caplog.text


async def test_a_stop_that_was_asked_for_is_not_logged(relay, connect, caplog):
    caplog.set_level("WARNING", logger="chatomon")
    link, _ = connect()
    await eventually(lambda: link.state() == ("connected", None))
    link.stop()
    await link.wait(5)
    assert "stopped with" not in caplog.text


async def test_a_request_that_arrives_means_the_connection_was_there(relay, connect):
    url = relay.url
    await relay.stop()
    link, _ = connect(url=url)
    link.seen()
    assert link.state() == ("lost", None)


async def test_the_handler_is_on_the_clients_logger_only_while_connected(relay, connect):
    logger = logging.getLogger("spectral_bridge.client")
    before = list(logger.handlers)
    level = logger.level
    link, _ = connect()
    assert len(logger.handlers) == len(before) + 1
    assert (logger.level, logger.propagate) == (level, True)
    await eventually(lambda: link.state() == ("connected", None))
    link.stop()
    await link.wait(5)
    assert logger.handlers == before


async def test_the_key_of_the_adapter_is_hidden_in_what_httpx_logs(relay, connect, caplog):
    caplog.set_level("INFO", logger="httpx")
    logger = logging.getLogger("httpx")
    before = list(logger.filters)
    link, _ = connect()
    logger.info(
        'HTTP Request: %s %s "%s %d %s"', "POST", f"{ADAPTER}/v1/responses", "1.1", 200, "OK"
    )
    logger.info("HTTP Request: %s", "https://example.com/other")
    assert SECRET not in caplog.text
    assert 'POST http://127.0.0.1:9/***/v1/responses "1.1 200 OK"' in caplog.text
    assert "https://example.com/other" in caplog.text
    link.stop()
    await link.wait(5)
    # Once the connection is over the logger is as it was.
    assert logger.filters == before
    logger.info("HTTP Request: %s", ADAPTER)
    assert SECRET in caplog.text


async def test_a_plain_address_is_refused_unless_a_test_allows_it(relay):
    with pytest.raises(ValueError, match="wss://"):
        Relay(relay.url, "good-key", ADAPTER, secret=SECRET, on_end=lambda: None)


def test_the_relay_is_spectrals_unless_the_environment_names_another(monkeypatch):
    monkeypatch.delenv("SPECTRAL_BRIDGE_RELAY_URL", raising=False)
    assert _relay.relay_url() == "wss://bridge.spectral.principled.app/connect"
    monkeypatch.setenv("SPECTRAL_BRIDGE_RELAY_URL", "  wss://relay.example/connect ")
    assert _relay.relay_url() == "wss://relay.example/connect"


def test_the_extra_is_found_without_importing_it():
    assert _relay.available() is True


def test_without_the_extra_the_add_on_is_not_available(no_extra):
    assert _relay.available() is False


def test_without_the_extra_the_address_of_the_relay_names_it(no_extra, monkeypatch):
    monkeypatch.delenv("SPECTRAL_BRIDGE_RELAY_URL", raising=False)
    with pytest.raises(MissingExtraError, match=r"chatomon\[spectral\]") as error:
        _relay.relay_url()
    assert error.value.package == "spectral-bridge"


async def test_without_the_extra_a_connection_names_it(no_extra):
    with pytest.raises(MissingExtraError, match=r"chatomon\[spectral\]"):
        Relay(
            "wss://relay.example/connect", "good-key", ADAPTER, secret=SECRET, on_end=lambda: None
        )
