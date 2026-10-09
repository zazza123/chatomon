"""The bridge against Spectral's own relay, with the key of a real target.

    CHATOMON_INTEGRATION=1 SPECTRAL_BRIDGE_API_KEY=... \\
        uv run pytest tests/integration/test_spectral.py -m integration -v

The atom follows a script and calls no model. Nothing reaches Spectral but the
connection itself, which the test closes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._session import Session
from chatomon.studio._spectral import Bridge
from tests.integration.conftest import require_env

pytestmark = pytest.mark.integration


async def until(bridge: Bridge, status: str, *, timeout: float = 30) -> dict[str, Any]:
    """Wait for the bridge to say `status`, and fail with its sentence if it is refused."""
    deadline = time.monotonic() + timeout
    while (state := bridge.state())["status"] != status:
        assert state["status"] != "refused", state["message"]
        assert time.monotonic() < deadline, f"still {state['status']}: {state['message']}"
        await asyncio.sleep(0.1)
    return state


async def test_the_bridge_connects_to_spectral_and_disconnects():
    require_env("SPECTRAL_BRIDGE_API_KEY")
    connector = PydanticAIConnector(TestModel(custom_output_text="Hi"))
    atom = Atom(name="shop", instructions="Be brief.", connector=connector)
    bridge = Bridge(Session(atom, source="app:atom"))
    try:
        await bridge.connect("SPECTRAL_BRIDGE_API_KEY")
        state = await until(bridge, "connected")
        assert state["message"] == "Connected to Spectral"
        assert state["adapter"]["port"] > 0
        bridge.disconnect()
        await until(bridge, "off")
    finally:
        await bridge.close()
