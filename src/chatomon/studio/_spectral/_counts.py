"""What the orbitals decided on the requests that arrived, as numbers and nothing else."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from chatomon.schemas import Side, TurnReport

Chain = Mapping[Side, Sequence[tuple[str, str]]]
"""The orbitals a request met, by side and in order: the name and the class name of each."""

_FINAL = ("answered", "guided_refusal", "orbital_response", "blocked", "fallback_refusal")
_SIDES: tuple[Side, ...] = ("input", "output")
_COLUMNS = ("allow", "block", "respond", "error", "not_run", "runs")


class Counts:
    """Counts of requests, of how their turns ended and of what every orbital decided.

    A report can arrive from more than one turn at a time, so every change takes a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._running = 0
        self._failed = 0
        self._rejected = 0
        self._duration_ms = 0.0
        self._revisions: set[int] = set()
        self._final = dict.fromkeys(_FINAL, 0)
        self._orbitals: dict[tuple[Side, str], dict[str, Any]] = {}

    def rejected(self) -> None:
        """Count a request that was refused for its shape, before any turn."""
        with self._lock:
            self._rejected += 1

    def arrived(self) -> None:
        """Count a request that reached a turn, which is now running."""
        with self._lock:
            self._requests += 1
            self._running += 1

    def _row(self, side: Side, name: str, kind: str) -> dict[str, Any]:
        key = (side, name)
        if key not in self._orbitals:
            row: dict[str, Any] = {"name": name, "kind": kind, "side": side}
            row.update(dict.fromkeys(_COLUMNS, 0), duration_ms_total=0.0)
            self._orbitals[key] = row
        return self._orbitals[key]

    def done(
        self,
        *,
        revision: int,
        chain: Chain,
        report: TurnReport | None,
        failed: bool,
        duration_ms: float,
    ) -> None:
        """Count the end of a request that had arrived.

        Args:
            revision: The revision of the session the turn ran with.
            chain: The orbitals that were on for the turn.
            report: The report of the turn, when it gave one.
            failed: Whether the request was answered with an error.
            duration_ms: How long the turn took.
        """
        with self._lock:
            self._running -= 1
            self._duration_ms += duration_ms
            self._revisions.add(revision)
            if failed:
                self._failed += 1
            for side in _SIDES:
                for name, kind in chain.get(side, ()):
                    self._row(side, name, kind)
            ran: set[tuple[Side, str]] = set()
            if report is not None:
                output = [result for attempt in report.output for result in attempt.results]
                for result in [*report.input, *output]:
                    row = self._row(result.side, result.name, result.kind)
                    ran.add((result.side, result.name))
                    row["runs"] += 1
                    row["duration_ms_total"] += result.duration_ms
                    row["error" if result.error_type is not None else result.action] += 1
                if not failed and report.final_action in self._final:
                    self._final[report.final_action] += 1
            for side in _SIDES:
                for name, _ in chain.get(side, ()):
                    if (side, name) not in ran:
                        self._orbitals[side, name]["not_run"] += 1

    def to_json(self) -> dict[str, Any]:
        """Return the counts as JSON-ready data, input orbitals before output ones."""
        with self._lock:
            rows = sorted(self._orbitals.values(), key=lambda row: _SIDES.index(row["side"]))
            return {
                "requests": self._requests,
                "running": self._running,
                "failed": self._failed,
                "rejected": self._rejected,
                "duration_ms_total": round(self._duration_ms, 1),
                "configurations": len(self._revisions),
                "final": dict(self._final),
                "orbitals": [
                    {**row, "duration_ms_total": round(row["duration_ms_total"], 1)} for row in rows
                ],
            }
