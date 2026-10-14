"""What the studio writes to the terminal."""

import io
from datetime import datetime

import pytest

from chatomon.studio._console import Console, Palette, plain

NOON = datetime(2026, 10, 14, 12, 4, 31)


def console(palette: Palette = "none", width: int = 80) -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console(out, palette=palette, width=width, clock=lambda: NOON), out


def test_a_line_has_the_time_a_symbol_a_label_and_the_text():
    con, out = console()
    con.line("~", "output", "tone_judge settings changed")
    assert out.getvalue() == "  12:04:31  ~ output    tone_judge settings changed\n"


def test_a_duration_is_aligned_after_the_text():
    con, out = console()
    con.line("●", "turn 1", "input allowed · output allowed", duration_ms=1432)
    con.line("●", "turn 2", "input blocked", duration_ms=212.4)
    assert out.getvalue().splitlines() == [
        "  12:04:31  ● turn 1    input allowed · output allowed            1.4s",
        "  12:04:31  ● turn 2    input blocked                             212ms",
    ]


def test_plain_writes_no_escape_sequence():
    con, out = console()
    con.line("●", "turn 1", "input blocked", tone="blocked")
    con.note("the nucleus calls a real model")
    con.stopped()
    assert "\x1b" not in out.getvalue()
    assert out.getvalue().splitlines()[1:] == [
        "note: the nucleus calls a real model",
        "chatomon studio stopped.",
    ]


def test_colour_uses_the_brand_tones():
    con, out = console("truecolor")
    con.line("●", "turn 2", "input blocked", tone="blocked")
    assert "\x1b[38;2;255;112;112m" in out.getvalue()
    con, out = console("256")
    con.line("●", "turn 1", "input allowed", tone="allowed")
    assert "\x1b[38;5;" in out.getvalue() and "\x1b[38;2;" not in out.getvalue()


def test_an_escape_sequence_from_outside_is_written_as_escapes():
    con, out = console("truecolor")
    con.line("~", "input", "x\x1b[2K\rchatomon studio is running at http://elsewhere\n  y")
    con.note("m\x9b2J")
    text = out.getvalue()
    assert "\x1b[2K" not in text and "\r" not in text and "\x9b" not in text
    assert text.count("\n") == 2
    assert "x\\x1b[2K\\rchatomon" in text


def test_letters_of_any_alphabet_stay_as_they_are():
    assert plain("Perché 📦 許可") == "Perché 📦 許可"


class Broken(io.StringIO):
    def write(self, text):
        raise BrokenPipeError


def test_a_stream_that_breaks_turns_the_console_off_without_raising():
    con = Console(Broken(), palette="none", width=80, clock=lambda: NOON)
    con.line("●", "turn 1", "input allowed")
    con.line("●", "turn 2", "input allowed")  # no second attempt, no exception


@pytest.mark.parametrize(
    ("env", "tty", "expected"),
    [
        ({}, False, "none"),
        ({"NO_COLOR": "1"}, True, "none"),
        ({"NO_COLOR": ""}, True, "256"),
        ({"TERM": "dumb"}, True, "none"),
        ({}, True, "256"),
        ({"COLORTERM": "truecolor"}, True, "truecolor"),
        ({"COLORTERM": "24bit"}, True, "truecolor"),
    ],
)
def test_the_palette_follows_the_terminal(monkeypatch, env, tty, expected):
    for name in ("NO_COLOR", "TERM", "COLORTERM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    stream = io.StringIO()
    stream.isatty = lambda: tty  # type: ignore[method-assign]
    assert Console(stream, width=80).palette == expected


def test_the_default_stream_is_the_stderr_of_the_moment(capsys):
    con = Console(palette="none", width=80, clock=lambda: NOON)
    con.stopped()
    assert capsys.readouterr().err == "chatomon studio stopped.\n"
