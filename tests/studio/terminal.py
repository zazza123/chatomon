"""The studio's terminal as an animated SVG for the documentation.

It reads the escape sequences the studio's console writes — and only those —
and draws each line as text, in the colours of the documentation's theme. The
command is typed, then each line appears; the last frame stays. Under
`prefers-reduced-motion` the last frame is shown at once.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from tests.studio.shots import Theme

FOLDER = Path(__file__).resolve().parents[2] / "docs" / "public" / "assets" / "terminal"

# The console's colours, in the dark theme's values, and what each theme draws them with.
_CONSOLE = {
    "242;165;22": "brand",
    "76;195;138": "allowed",
    "255;112;112": "blocked",
    "255;138;91": "warning",
}
TOKENS: dict[Theme, dict[str, str]] = {
    "light": {
        "background": "#f8f8f9",
        "text": "#1a1c26",
        "brand": "#8f5a06",
        "allowed": "#18794e",
        "blocked": "#c93434",
        "warning": "#c2410c",
    },
    "dark": {
        "background": "#121319",
        "text": "#efeff3",
        "brand": "#f2a516",
        "allowed": "#4cc38a",
        "blocked": "#ff7070",
        "warning": "#ff8a5b",
    },
}
_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_CHAR_W, _LINE_H, _PAD, _BAR = 8.4, 18, 16, 28
_FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
_TYPE_S, _LINE_S, _HOLD_S = 1.6, 0.12, 4.0


def _spans(line: str) -> list[tuple[str, str | None]]:
    """Split a line of the console into texts and the class each is drawn with."""
    spans: list[tuple[str, str | None]] = []
    tone: str | None = None
    at = 0
    for match in _SGR.finditer(line):
        if match.start() > at:
            spans.append((line[at : match.start()], tone))
        code = match[1]
        if code in {"0", ""}:
            tone = None
        elif code == "2":
            tone = "dim"
        elif code.startswith("38;2;"):
            tone = _CONSOLE[code[5:]]
        else:
            raise ValueError(f"the console wrote an escape the recording does not know: {code}")
        at = match.end()
    spans.append((line[at:], tone))
    return [(text, tone) for text, tone in spans if text]


def render(command: str, output: str, theme: Theme) -> str:
    """Return the SVG of `command` typed, then of `output` line by line.

    Args:
        command: What the developer types after the prompt.
        output: What the console wrote, with its escape sequences.
        theme: The documentation's theme to draw with.

    Returns:
        The SVG document.
    """
    colors = TOKENS[theme]
    lines = output.rstrip("\n").split("\n")
    columns = max(80, len(command) + 2, *(len(_SGR.sub("", line)) for line in lines))
    width = round(_PAD * 2 + columns * _CHAR_W)
    height = _BAR + _PAD * 2 + (len(lines) + 1) * _LINE_H
    total = _TYPE_S + len(lines) * _LINE_S + _HOLD_S
    frames, rules = [], []
    typed = len(command) * _CHAR_W
    rules.append(
        f"@keyframes type{{0%{{transform:translateX(0)}}"
        f"{_TYPE_S / total * 100:.2f}%,100%{{transform:translateX({typed:.1f}px)}}}}"
        f".cover{{animation:type {total:.2f}s steps({len(command)}) infinite;"
        f"transform:translateX({typed:.1f}px)}}"
    )
    y0 = _BAR + _PAD + _LINE_H
    frames.append(
        f'<text x="{_PAD}" y="{y0}"><tspan class="brand">$ </tspan>{escape(command)}</text>'
        f'<rect class="cover" x="{_PAD + 2 * _CHAR_W}" y="{y0 - 13}" width="{typed + _CHAR_W:.1f}"'
        f' height="{_LINE_H}" fill="{colors["background"]}"/>'
    )
    for index, line in enumerate(lines, start=1):
        shown = (_TYPE_S + index * _LINE_S) / total * 100
        rules.append(
            f"@keyframes l{index}{{0%,{shown:.2f}%{{opacity:0}}"
            f"{shown + 0.01:.2f}%,97%{{opacity:1}}100%{{opacity:0}}}}"
            f".l{index}{{animation:l{index} {total:.2f}s infinite}}"
        )
        tspans = "".join(
            f'<tspan class="{tone}">{escape(text)}</tspan>' if tone else escape(text)
            for text, tone in _spans(line)
        )
        frames.append(
            f'<text class="l{index}" x="{_PAD}" y="{y0 + index * _LINE_H}">{tspans}</text>'
        )
    style = (
        f"text{{font-family:{_FONT};font-size:14px;white-space:pre;fill:{colors['text']}}}"
        + "".join(
            f".{tone}{{fill:{colors[tone]}}}" for tone in ("brand", "allowed", "blocked", "warning")
        )
        + ".dim{opacity:.6}"
        + "".join(rules)
        + "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
    )
    dots = "".join(
        f'<circle cx="{_PAD + 6 + i * 18}" cy="{_BAR / 2 + 2}" r="5" fill="{c}" opacity=".8"/>'
        for i, c in enumerate((colors["blocked"], colors["brand"], colors["allowed"]))
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="chatomon studio starting in a terminal, then writing a line for each action">'
        f"<style>{style}</style>"
        f'<rect width="{width}" height="{height}" rx="10" fill="{colors["background"]}" '
        f'stroke="{colors["text"]}" stroke-opacity=".15"/>{dots}' + "".join(frames) + "</svg>\n"
    )
