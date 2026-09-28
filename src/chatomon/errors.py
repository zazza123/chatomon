"""Exceptions raised by chatomon.

This module imports nothing from chatomon, so every other module can use it.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "ChatomonConfigError",
    "ChatomonError",
    "MissingAnnotationsError",
    "MissingExtraError",
]


class ChatomonError(Exception):
    """Base class of every exception raised by chatomon."""


class ChatomonConfigError(ChatomonError, ValueError):
    """An atom, orbital or connector is configured incorrectly.

    The message always says what to change.
    """


class MissingExtraError(ChatomonError, ImportError):
    """An optional dependency needed by a chatomon module is not installed.

    Args:
        extra: Name of the chatomon extra that provides the dependency.
        package: Name of the missing distribution.
    """

    def __init__(self, extra: str, package: str) -> None:
        self.extra = extra
        self.package = package
        super().__init__(
            f"{package!r} is not installed. Install it with: pip install 'chatomon[{extra}]'"
        )


class MissingAnnotationsError(ChatomonError):
    """An orbital reads annotations that no earlier orbital produced in this turn.

    It is raised by the engine inside the orbital's execution, so the orbital's
    `on_error` policy applies.

    Args:
        orbital: Name of the orbital that needed the annotations.
        missing: Names of the orbitals whose annotations are missing.
    """

    def __init__(self, orbital: str, missing: Sequence[str]) -> None:
        self.orbital = orbital
        self.missing = tuple(missing)
        names = ", ".join(repr(name) for name in self.missing)
        super().__init__(f"orbital {orbital!r} reads annotations that are missing: {names}")
