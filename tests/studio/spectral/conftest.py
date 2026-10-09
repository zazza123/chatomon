"""Fixtures of the add-on's tests: a fake relay, and a studio whose relay it is."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator

import pytest

from tests.studio.spectral.fake_relay import FakeRelay


@pytest.fixture
async def relay() -> AsyncIterator[FakeRelay]:
    """A fake relay that listens on loopback for the time of the test."""
    fake = FakeRelay()
    await fake.start()
    yield fake
    await fake.stop()


@pytest.fixture
def no_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """As when `spectral-bridge` is not installed: it is not found and cannot be imported."""
    for name in ("spectral_bridge", "spectral_bridge.cli", "spectral_bridge.cli.defaults"):
        monkeypatch.setitem(sys.modules, name, None)
