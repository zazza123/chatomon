"""Test doubles shared by the unit tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from chatomon.atom import Atom
from chatomon.connectors.base import Connector
from chatomon.engine import Engine
from chatomon.orbitals.base import Orbital
from chatomon.schemas import ErrorPolicy, OrbitalContext, Side, Verdict


class ScriptedOrbital(Orbital):
    """Returns (or raises) the scripted steps in order; the last step repeats."""

    calls_model = False

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

    calls_model = False

    async def check(self, ctx: OrbitalContext) -> Verdict:
        await asyncio.sleep(10)
        return Verdict.allow()


class HangingOrbital(Orbital):
    """Starts and never answers, so a test can cancel the turn while it runs."""

    calls_model = False

    def __init__(self) -> None:
        super().__init__(name="hanging")
        self.started = asyncio.Event()

    async def check(self, ctx: OrbitalContext) -> Verdict:
        self.started.set()
        await asyncio.Event().wait()
        return Verdict.allow()


def make_engine(
    *, inputs: Sequence[Orbital] = (), outputs: Sequence[Orbital] = (), **kwargs: object
) -> Engine:
    """An engine for the test atom with the given chains."""
    return Engine(
        atom_name="test-atom",
        input_orbitals=inputs,
        output_orbitals=outputs,
        **kwargs,  # type: ignore[arg-type]
    )


ModelT = TypeVar("ModelT", bound=BaseModel)


class FakeAgent:
    """What FakeConnector.build returns: the atom and its engine."""

    def __init__(self, atom: Atom[FakeAgent], engine: Engine) -> None:
        self.atom = atom
        self.engine = engine


class FakeConnector(Connector[FakeAgent]):
    """A connector with no framework: records builds and scripted structured calls."""

    def __init__(self, decisions: Sequence[BaseModel | dict[str, Any]] = ()) -> None:
        self.decisions = list(decisions)
        self.built: list[FakeAgent] = []
        self.calls: list[dict[str, Any]] = []

    def build(self, atom: Atom[FakeAgent], engine: Engine) -> FakeAgent:
        agent = FakeAgent(atom, engine)
        self.built.append(agent)
        return agent

    async def generate_structured(
        self, model: str, *, system: str, prompt: str, schema: type[ModelT]
    ) -> ModelT:
        self.calls.append({"model": model, "system": system, "prompt": prompt, "schema": schema})
        decision = self.decisions.pop(0)
        payload = decision if isinstance(decision, dict) else decision.model_dump()
        return schema.model_validate(payload)
