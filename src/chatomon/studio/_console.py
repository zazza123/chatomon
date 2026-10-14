"""What the studio writes to the terminal: a banner at start, then a line for each action.

The lines carry labels only — names, decisions, reasons, durations — never a
message, a reply, a guidance, an annotation, the text of an exception or the
value of a setting, whatever the atom's `capture_content`.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Literal, TextIO

if TYPE_CHECKING:
    from chatomon.schemas import OrbitalResult, TurnReport

Palette = Literal["none", "256", "truecolor"]
"""How many colours the terminal shows; `none` writes no escape sequence."""

Tone = Literal["plain", "dim", "brand", "allowed", "blocked", "warning"]
"""What a piece of a line means, which decides its colour."""

# The brand's amber and the dark theme's allowed, blocked and warning of
# docs/public/stylesheets/chatomon.css: they read on light and dark terminals.
COLORS: dict[Tone, tuple[int, int, int]] = {
    "brand": (0xF2, 0xA5, 0x16),
    "allowed": (0x4C, 0xC3, 0x8A),
    "blocked": (0xFF, 0x70, 0x70),
    "warning": (0xFF, 0x8A, 0x5B),
}
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"
_CUBE = (0, 95, 135, 175, 215, 255)
_TEXT_WIDTH = 40


def plain(text: str) -> str:
    """Write as escapes the characters a terminal would act on.

    A name or a note can come from a preset someone else wrote: an escape
    sequence or a line break in it must not rewrite what the terminal shows.

    Returns:
        The text with every character that is not printable as its escape;
        letters of any alphabet stay as they are.
    """
    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode("ascii")
        for char in text
    )


def _palette(stream: TextIO) -> Palette:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return "none"
    try:
        tty = stream.isatty()
    except (AttributeError, OSError, ValueError):
        tty = False
    if not tty:
        return "none"
    depth = os.environ.get("COLORTERM", "").lower()
    return "truecolor" if depth in {"truecolor", "24bit"} else "256"


def _width(stream: TextIO) -> int:
    try:
        return os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, OSError, ValueError):
        return 80


def _nearest(level: int) -> int:
    return min(range(len(_CUBE)), key=lambda index: abs(_CUBE[index] - level))


def _duration(ms: float) -> str:
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


_ENDED: dict[str, tuple[str, Tone]] = {
    "blocked": ("blocked", "blocked"),
    "guided_refusal": ("refused", "blocked"),
    "fallback_refusal": ("refused", "blocked"),
    "orbital_response": ("answered by", "warning"),
}


def _who(result: OrbitalResult) -> str:
    return f"{result.name} ({result.reason})" if result.reason else result.name


def summary(report: TurnReport | None, error_type: str | None) -> tuple[str, Tone]:
    """Sum up how a turn went in labels, with the tone of its line.

    Args:
        report: The turn's report; `None` when the turn raised before it.
        error_type: The class name of what the turn raised, if it did.

    Returns:
        The text of the line and its tone; never a text of the conversation.
    """
    if error_type is not None:
        return f"failed · {error_type}", "blocked"
    if report is None:
        return "finished", "plain"
    if report.final_action == "aborted":
        return f"aborted · {report.error_type or 'error'}", "blocked"
    retried = [a for a in report.output if a.decision == "regenerate"]
    prefix = f"output retried {_count(len(retried), 'time')} · " if retried else ""
    if report.final_action == "answered":
        if not retried:
            return "input allowed · output allowed", "allowed"
        names = dict.fromkeys(r.name for a in retried for r in a.results if r.action != "allow")
        return f"{prefix}{', '.join(names)} · allowed", "warning"
    last = report.output[-1].results if report.output else ()
    stopper = next((r for r in (*report.input, *last) if r.action != "allow"), None)
    verb, tone = _ENDED.get(report.final_action, (report.final_action, "blocked"))
    if stopper is None:
        return f"{prefix}{verb}", tone
    if report.final_action == "orbital_response":
        return f"{prefix}{stopper.side} {verb} {_who(stopper)}", tone
    return f"{prefix}{stopper.side} {verb} · {_who(stopper)}", tone


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class Console:
    """Writes the studio's banner and feed to a stream, the terminal's stderr by default.

    Args:
        stream: Where to write; the `sys.stderr` of the moment of each write when `None`.
        palette: The colours to use; found from the stream and the environment when `None`.
        width: The terminal's width in columns; read from the stream when `None`.
        clock: The time of a line; the local wall clock when `None`.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        palette: Palette | None = None,
        width: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._stream = stream
        self.palette: Palette = palette if palette is not None else _palette(self._out())
        self.width = width if width is not None else _width(self._out())
        self._clock = clock if clock is not None else datetime.now
        self._lock = threading.Lock()
        self._off = False

    def _out(self) -> TextIO:
        return self._stream if self._stream is not None else sys.stderr

    def _write(self, text: str) -> None:
        # A report of Spectral can come from more than one turn at a time.
        with self._lock:
            if self._off:
                return
            try:
                out = self._out()
                out.write(text)
                out.flush()
            except (OSError, ValueError):
                # A closed stderr or a broken pipe: the page and the turns go on.
                self._off = True

    def paint(self, text: str, tone: Tone) -> str:
        """Return `text` in the colour of `tone`, or as it is without colour."""
        if self.palette == "none" or tone == "plain":
            return text
        if tone == "dim":
            return f"{_DIM}{text}{_RESET}"
        red, green, blue = COLORS[tone]
        if self.palette == "truecolor":
            return f"\x1b[38;2;{red};{green};{blue}m{text}{_RESET}"
        index = 16 + 36 * _nearest(red) + 6 * _nearest(green) + _nearest(blue)
        return f"\x1b[38;5;{index}m{text}{_RESET}"

    def line(
        self,
        symbol: str,
        label: str,
        text: str,
        *,
        tone: Tone = "plain",
        duration_ms: float | None = None,
    ) -> None:
        """Write one line of the feed: the time, a symbol, a label and a text.

        Args:
            symbol: One character that tells the kind of line.
            label: What the line is about, such as `turn 3` or `input`.
            text: What happened, labels only.
            tone: The colour of the symbol and the text.
            duration_ms: How long it took, shown after the text.
        """
        body = plain(text)
        tail = ""
        if duration_ms is not None:
            tail = " " * max(2, _TEXT_WIDTH + 2 - len(body)) + self.paint(
                _duration(duration_ms), "dim"
            )
        time = self.paint(self._clock().strftime("%H:%M:%S"), "dim")
        self._write(
            f"  {time}  {self.paint(symbol, tone)} {plain(label):<9} "
            f"{self.paint(body, tone)}{tail}\n"
        )

    def note(self, text: str) -> None:
        """Write what the developer should know: `note: …`, or `! …` in colour."""
        if self.palette == "none":
            self._write(f"note: {plain(text)}\n")
        else:
            self._write(f"  {self.paint('!', 'warning')} {self.paint(plain(text), 'warning')}\n")

    def stopped(self) -> None:
        """Write the line of a studio that stopped."""
        self._write("chatomon studio stopped.\n")

    def turn(
        self, n: int, report: TurnReport | None, *, error_type: str | None, duration_ms: float
    ) -> None:
        """Write how a turn of the page went."""
        text, tone = summary(report, error_type)
        self.line("●", f"turn {n}", text, tone=tone, duration_ms=duration_ms)

    def switched(self, side: str, name: str, enabled: bool) -> None:
        """Write that an orbital was turned on or off."""
        self.line("~", side, f"{name} {'on' if enabled else 'off'}")

    def edited(self, side: str, name: str) -> None:
        """Write that settings of an orbital changed; never their values."""
        self.line("~", side, f"{name} settings changed")

    def moved(self, side: str, name: str, position: int) -> None:
        """Write that an orbital moved; `position` is 0-based, the line counts from 1."""
        self.line("↕", side, f"{name} moved to {position + 1}")

    def added(self, side: str, name: str) -> None:
        """Write that an orbital was added."""
        self.line("+", side, f"{name} added", tone="allowed")

    def deleted(self, side: str, name: str) -> None:
        """Write that an orbital was deleted."""
        self.line("-", side, f"{name} removed", tone="warning")

    def reset(self, side: str | None, name: str | None) -> None:
        """Write that one orbital, or all, went back to the code."""
        if side is None or name is None:
            self.line("↺", "orbitals", "reset to the code")
        else:
            self.line("↺", side, f"{name} reset to the code")

    def undone(self) -> None:
        """Write that the last deletion, reset or import was undone."""
        self.line("↺", "orbitals", "undo")

    def preset_imported(self, changes: int) -> None:
        """Write that a preset was imported, with how many changes it left."""
        self.line("⇣", "preset", f"imported · {_count(changes, 'change')}")

    def cleared(self) -> None:
        """Write that the conversation was cleared."""
        self.line("○", "chat", "conversation cleared")

    def reloaded(self, changes: int, dropped: int) -> None:
        """Write that the code was loaded again, with the studio's changes kept and dropped."""
        parts = ["from code"]
        if changes:
            parts.append(f"{_count(changes, 'change')} kept")
        if dropped:
            parts.append(f"{_count(dropped, 'change')} dropped")
        self.line("↻", "reload", " · ".join(parts), tone="allowed")

    def reload_failed(self, message: str) -> None:
        """Write why a reload failed: the studio's sentence, after the traceback."""
        self.line("✗", "reload", f"failed: {message}", tone="blocked")

    def probed(
        self,
        side: str,
        name: str,
        outcome: str,
        action: str | None,
        reason: str | None,
        duration_ms: float,
    ) -> None:
        """Write how a test of an orbital went; never the sentence the page shows."""
        if outcome == "answered":
            text = f"{name} · {action}" + (f" ({reason})" if reason else "")
            tone: Tone = "allowed" if action == "allow" else "blocked"
        else:
            text, tone = f"{name} · {outcome}", "warning"
        self.line("▸", "probe", text, tone=tone, duration_ms=duration_ms)

    def test_written(self, path: str) -> None:
        """Write the path of a saved test."""
        self.line("✎", "test", f"saved {path}")
