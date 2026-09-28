"""Tracing and content-capture helpers shared by chatomon modules.

chatomon depends only on `opentelemetry-api`: without an SDK configured by the
application, every span is a no-op.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from pydantic_core import to_jsonable_python

from chatomon._version import __version__
from chatomon.errors import ChatomonConfigError
from chatomon.schemas import CaptureContent

__all__ = ["TRACER_NAME", "ContentPolicy", "get_tracer", "sha256_text", "stable_json"]

TRACER_NAME = "chatomon"
"""Instrumentation scope name of every chatomon span."""

_MODES: tuple[CaptureContent, ...] = ("none", "hash", "full")


def get_tracer(tracer_provider: trace.TracerProvider | None = None) -> trace.Tracer:
    """Return chatomon's tracer.

    Args:
        tracer_provider: Provider to use; the global one when `None`.

    Returns:
        A tracer named `chatomon`, versioned with the library version.
    """
    return trace.get_tracer(TRACER_NAME, __version__, tracer_provider=tracer_provider)


def stable_json(value: Any) -> str:
    """Serialize a value to canonical JSON: sorted keys, no spaces.

    Pydantic models and dataclasses are converted to JSON types; other objects
    fall back to their `repr`.

    Args:
        value: The value to serialize.

    Returns:
        The JSON text.
    """
    return json.dumps(
        to_jsonable_python(value, fallback=repr),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_text(text: str) -> str:
    """Hash a text with SHA-256.

    Args:
        text: The text to hash.

    Returns:
        `"sha256:"` followed by the hex digest.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class ContentPolicy:
    """Decides how much message content is recorded in logs, traces and reports.

    Args:
        mode: `"none"` records nothing, `"hash"` records SHA-256 digests,
            `"full"` records the text passed through `redactor`.
        redactor: Function applied to every text recorded in `"full"` mode.

    Raises:
        ChatomonConfigError: If `mode` is not one of the three values.

    Examples:
        ```python
        from chatomon.observability import ContentPolicy

        policy = ContentPolicy("full", redactor=lambda text: text.replace("@", " at "))
        assert policy.text("mail me at a@b.it") == "mail me at a at b.it"
        assert ContentPolicy().text("anything") is None
        ```
    """

    def __init__(
        self, mode: CaptureContent = "none", redactor: Callable[[str], str] | None = None
    ) -> None:
        if mode not in _MODES:
            raise ChatomonConfigError(
                f"capture_content must be one of {list(_MODES)}, got {mode!r}"
            )
        self._mode: CaptureContent = mode
        self._redactor = redactor

    @property
    def mode(self) -> CaptureContent:
        """The capture mode."""
        return self._mode

    def text(self, value: str | None) -> str | None:
        """Return what may be recorded for a text.

        Args:
            value: The text, or `None`.

        Returns:
            `None`, a digest or the (redacted) text, depending on the mode.
        """
        if value is None or self._mode == "none":
            return None
        if self._mode == "hash":
            return sha256_text(value)
        return self._redactor(value) if self._redactor is not None else value

    def data(self, value: Any) -> str | None:
        """Return what may be recorded for structured data, as JSON.

        Args:
            value: Any JSON-serializable value (strings are recorded as-is).

        Returns:
            `None`, a digest or the (redacted) JSON text, depending on the mode.
        """
        if value is None:
            return None
        return self.text(value if isinstance(value, str) else stable_json(value))
