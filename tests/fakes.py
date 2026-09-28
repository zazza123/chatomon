"""Test doubles shared by the unit tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import ClassVar

from chatomon.orbitals.base import Orbital
from chatomon.schemas import ErrorPolicy, OrbitalContext, Side, Verdict


class ScriptedOrbital(Orbital):
    """Returns (or raises) the scripted steps in order; the last step repeats."""

    def __init__(
        self,
        *steps: Verdict | Exception,
        name: str | None = None,
        reads: Sequence[str] = (),
        timeout: float = 30.0,
        on_error: ErrorPolicy = "block",
    ) -> None:
        super().__init__(name=name, reads=reads, timeout=timeout, on_error=on_error)
        self.steps: list[Verdict | Exception] = list(steps) or [Verdict.allow()]
        self.contexts: list[OrbitalContext] = []

    async def check(self, ctx: OrbitalContext) -> Verdict:
        self.contexts.append(ctx)
        step = self.steps[min(len(self.contexts), len(self.steps)) - 1]
        if isinstance(step, Exception):
            raise step
        return step


class InputOnlyOrbital(ScriptedOrbital):
    """A scripted orbital that can only run on the input side."""

    sides: ClassVar[frozenset[Side]] = frozenset({"input"})


class SlowOrbital(Orbital):
    """Never answers in time."""

    async def check(self, ctx: OrbitalContext) -> Verdict:
        await asyncio.sleep(10)
        return Verdict.allow()
