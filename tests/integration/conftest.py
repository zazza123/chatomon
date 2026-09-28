"""Helpers for tests that talk to real services (never run by default)."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable

import pytest

INTEGRATION_TIMEOUT = 60.0
"""Orbital timeout for real services (cold GPU starts are slow)."""

WARMUP_S = float(os.environ.get("CHATOMON_WARMUP_S", "600"))


def require_env(name: str) -> str:
    """Return an environment variable or skip the test."""
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run this integration test")
    return value


async def warm_up(check: Callable[[], Awaitable[object]]) -> None:
    """Retry `check` every 15 s until it succeeds or CHATOMON_WARMUP_S elapses."""
    deadline = time.monotonic() + WARMUP_S
    while True:
        try:
            await check()
            return
        except Exception:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(15)
