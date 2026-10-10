"""The connection to the relay: the client of `spectral-bridge`, kept running and watched.

This is the only module that imports `spectral_bridge`. The client has no public
way to say whether it is connected, so its state is read from `_ws`, the
connection it holds while it has one, and the reason of a refusal from the
warnings and the errors it logs.

The client calls the adapter with `httpx`, which logs the address of every call:
while a connection is open, a filter on that logger hides the key the address holds.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
from collections.abc import Callable
from typing import Any

from chatomon.errors import MissingExtraError

logger = logging.getLogger(__name__)

KEY_ENV = "SPECTRAL_BRIDGE_API_KEY"
"""The variable that holds the key of a target, unless the page names another one."""

RELAY_ENV = "SPECTRAL_BRIDGE_RELAY_URL"
"""The variable that names another relay than Spectral's."""

_LOGGER = "spectral_bridge.client"
# The client's own sentences, as it gives them to its logger before they are filled in.
_REASONS = {
    "authentication failed": "key",
    "relay refused protocol version %d (%s), upgrade spectral-bridge": "protocol",
    "relay speaks protocol version %s, expected %d": "protocol",
}
_RETRYING = "disconnected (%s), reconnecting in %ds"
_HTTP_LOGGER = "httpx"
_HIDDEN = "***"


def available() -> bool:
    """Tell whether `spectral-bridge` is installed, without importing it.

    Returns:
        Whether the add-on can be used.
    """
    return importlib.util.find_spec("spectral_bridge") is not None


def relay_url() -> str:
    """Return the address of the relay: the one `RELAY_ENV` names, or Spectral's.

    Raises:
        MissingExtraError: If `spectral-bridge` is not installed.
    """
    given = os.environ.get(RELAY_ENV, "").strip()
    if given:
        return given
    try:
        from spectral_bridge.cli.defaults import SPECTRAL_RELAY_URL
    except ImportError as exc:
        raise MissingExtraError("spectral", "spectral-bridge") from exc
    return SPECTRAL_RELAY_URL


class _Reasons(logging.Handler):
    """Reads from the client's log why it stopped, and whether it is trying again.

    It prints nothing and changes no level: warnings and errors reach it as they
    reach any handler.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.reason: str | None = None
        self.retrying = False

    def emit(self, record: logging.LogRecord) -> None:
        template = record.msg if isinstance(record.msg, str) else ""
        self.reason = _REASONS.get(template, self.reason)
        if template == _RETRYING:
            self.retrying = True


class _Hidden(logging.Filter):
    """Hides the key of the adapter in what `httpx` logs of the calls to it.

    It lets every record through and changes only the ones that show the key.
    """

    def __init__(self, secret: str) -> None:
        super().__init__()
        self._secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(self._clean(arg) for arg in record.args)
        return True

    def _clean(self, arg: object) -> object:
        # A number stays a number: the message can ask for one.
        if isinstance(arg, (int, float)):
            return arg
        text = str(arg)
        return text.replace(self._secret, _HIDDEN) if self._secret in text else arg


class Relay:
    """One connection to the relay, from when it is asked for to when it ends.

    Args:
        url: The address of the relay.
        key: The key of the target.
        adapter_url: Where the client sends the requests it receives.
        secret: The key of the adapter, as `adapter_url` holds it: logs do not show it.
        on_end: Called once, when the client has stopped for any reason.
        insecure: Whether `ws://` is allowed. Only tests set it, in code.

    Raises:
        MissingExtraError: If `spectral-bridge` is not installed.
        ValueError: If the client refuses an address.
    """

    def __init__(
        self,
        url: str,
        key: str,
        adapter_url: str,
        *,
        secret: str,
        on_end: Callable[[], None],
        insecure: bool = False,
    ) -> None:
        try:
            from spectral_bridge import RelayClient
        except ImportError as exc:
            raise MissingExtraError("spectral", "spectral-bridge") from exc
        self._client: Any = RelayClient(url, key, adapter_url, insecure_relay=insecure)
        self._on_end = on_end
        self._asked = False
        self._was_connected = False
        self._reasons = _Reasons()
        self._hidden = _Hidden(secret)
        logging.getLogger(_LOGGER).addHandler(self._reasons)
        logging.getLogger(_HTTP_LOGGER).addFilter(self._hidden)
        self._task = asyncio.get_running_loop().create_task(self._client.run())
        self._task.add_done_callback(self._ended)

    def _ended(self, task: asyncio.Task[None]) -> None:
        logging.getLogger(_LOGGER).removeHandler(self._reasons)
        logging.getLogger(_HTTP_LOGGER).removeFilter(self._hidden)
        error = None if task.cancelled() else task.exception()
        if error is not None:
            # The class of the error, never its text: the state can only say that it stopped.
            logger.warning("the client of spectral-bridge stopped with %s", type(error).__name__)
        # The client holds the key: nothing keeps it once the connection is over.
        self._client = None
        self._on_end()

    def state(self) -> tuple[str, str | None]:
        """Say where the connection is, and why when there is a reason.

        Returns:
            `connecting` (with `unreachable` once the client is trying again),
            `connected`, `lost`, `closing`, `off` after a stop that was asked
            for, or `refused` with `key`, `protocol` or `stopped`. `off` and
            `refused` are said only after `on_end` was called.
        """
        # Over only once `_ended` ran, a moment after the client stopped: whoever reads
        # `off` or `refused` finds what `on_end` does already done.
        if self._client is None:
            return ("off", None) if self._asked else ("refused", self._reasons.reason or "stopped")
        if self._asked:
            return "closing", None
        if getattr(self._client, "_ws", None) is not None:
            self._was_connected = True
            return "connected", None
        if self._was_connected:
            return "lost", None
        return "connecting", "unreachable" if self._reasons.retrying else None

    def seen(self) -> None:
        """Note that a request arrived: the connection was there, whoever was looking."""
        self._was_connected = True

    def stop(self) -> None:
        """Ask the client to stop: requests that run finish first; asked again, it stops now."""
        self._asked = True
        if not self._task.done():
            self._client.shutdown()

    async def wait(self, timeout: float) -> None:
        """Wait for the client to have stopped, for at most `timeout` seconds."""
        await asyncio.wait({self._task}, timeout=timeout)
