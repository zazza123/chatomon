"""chatomon: declare a chat agent once and build it into native agent frameworks."""

from __future__ import annotations

import logging

from chatomon._version import __version__
from chatomon.errors import ChatomonConfigError, ChatomonError, MissingExtraError
from chatomon.knowledge import Knowledge
from chatomon.orbitals.base import Orbital, orbital
from chatomon.schemas import OrbitalContext, TurnReport, Verdict

logging.getLogger("chatomon").addHandler(logging.NullHandler())

__all__ = [
    "ChatomonConfigError",
    "ChatomonError",
    "Knowledge",
    "MissingExtraError",
    "Orbital",
    "OrbitalContext",
    "TurnReport",
    "Verdict",
    "__version__",
    "orbital",
]
