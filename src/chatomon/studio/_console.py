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
from typing import Literal, TextIO

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
