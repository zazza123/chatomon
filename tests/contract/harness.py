"""Framework-neutral harness used by the connector contract suite."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chatomon import Atom, TurnReport
from chatomon.connectors import Connector

INSTRUCTIONS = "You are the customer assistant of SpeedyPack."


@dataclass(frozen=True)
class ToolCall:
    """Scripted model step: call the `lookup` tool with no arguments."""


Step = str | Exception | ToolCall


@dataclass
class Call:
    """What the scripted nucleus model received in one call."""

    system: str
    messages: list[tuple[str, str]] = field(default_factory=list)


def lookup() -> str:
    """Look up the opening hours."""
    return "Monday to Friday, 9-18"


class Harness(ABC):
    """Drives one framework through the same scenarios."""

    name: str

    @property
    @abstractmethod
    def calls(self) -> list[Call]:
        """Calls received by the scripted nucleus model, in order."""

    @abstractmethod
    def connector(
        self,
        script: Sequence[Step],
        *,
        tools: bool = False,
        judge_decisions: Sequence[dict[str, Any]] = (),
    ) -> Connector[Any]:
        """A connector whose nucleus replies with `script` and whose judge model
        returns `judge_decisions`."""

    @abstractmethod
    async def ask(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        """Run one turn asynchronously and return the reply text."""

    @abstractmethod
    def ask_sync(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        """Run one turn with the framework's sync API."""

    @abstractmethod
    def history(self, agent: Any, *, conversation: str = "c1") -> list[tuple[str, str]]:
        """Persisted user/assistant text messages of a conversation."""


def build_agent(
    harness: Harness,
    script: Sequence[Step],
    *,
    tools: bool = False,
    judge_decisions: Sequence[dict[str, Any]] = (),
    **atom_kwargs: Any,
) -> tuple[Any, list[TurnReport]]:
    """Build the test atom on the harness's framework; returns the agent and its reports."""
    reports: list[TurnReport] = []
    atom = Atom(
        name="speedypack",
        instructions=INSTRUCTIONS,
        connector=harness.connector(script, tools=tools, judge_decisions=judge_decisions),
        on_turn=reports.append,
        **atom_kwargs,
    )
    return atom.build(), reports
