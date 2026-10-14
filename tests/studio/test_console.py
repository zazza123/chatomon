"""What the studio writes to the terminal."""

import io
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from chatomon.schemas import (
    FinalAction,
    OrbitalResult,
    OutputAttempt,
    OutputDecisionKind,
    Side,
    TurnReport,
    VerdictAction,
)
from chatomon.studio._console import Console, Palette, plain, summary

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


def result(
    name: str, side: Side, action: VerdictAction = "allow", reason: str | None = None, **extra: Any
) -> OrbitalResult:
    return OrbitalResult(
        name=name,
        side=side,
        kind="X",
        fingerprint="f",
        action=action,
        reason=reason,
        duration_ms=1.0,
        **extra,
    )


def report(
    final: FinalAction,
    inputs: Sequence[OrbitalResult] = (),
    outputs: Sequence[OutputAttempt] = (),
    **extra: Any,
) -> TurnReport:
    return TurnReport(
        turn_id="t",
        conversation_id=None,
        atom_name="shop",
        started_at=datetime(2026, 10, 14, tzinfo=UTC),
        duration_ms=10.0,
        final_action=final,
        escalated=False,
        input=tuple(inputs),
        output=tuple(outputs),
        **extra,
    )


def attempt(*results: OrbitalResult, decision: OutputDecisionKind = "accept") -> OutputAttempt:
    return OutputAttempt(results=tuple(results), decision=decision)


@pytest.mark.parametrize(
    ("turn", "error", "expected"),
    [
        (
            report("answered", [result("scope", "input")], [attempt(result("tone", "output"))]),
            None,
            ("input allowed · output allowed", "allowed"),
        ),
        (
            report("blocked", [result("scope", "input", "block", "off_topic")]),
            None,
            ("input blocked · scope (off_topic)", "blocked"),
        ),
        (
            report(
                "answered",
                [],
                [
                    attempt(result("tone", "output", "block", "rude"), decision="regenerate"),
                    attempt(result("tone", "output")),
                ],
            ),
            None,
            ("output retried 1 time · tone · allowed", "warning"),
        ),
        (
            report(
                "guided_refusal",
                [],
                [
                    attempt(result("tone", "output", "block"), decision="regenerate"),
                    attempt(result("tone", "output", "block"), decision="regenerate"),
                ],
            ),
            None,
            ("output retried 2 times · output refused · tone", "blocked"),
        ),
        (
            report("orbital_response", [result("faq", "input", "respond")]),
            None,
            ("input answered by faq", "warning"),
        ),
        (report("aborted", error_type="TimeoutError"), None, ("aborted · TimeoutError", "blocked")),
        (None, "RateLimitError", ("failed · RateLimitError", "blocked")),
    ],
)
def test_a_turn_is_summed_up_in_labels(turn, error, expected):
    assert summary(turn, error) == expected


SECRET = "the-user-wrote-this"


def test_a_turn_line_never_carries_content():
    turn = report(
        "blocked",
        [
            result(
                "scope",
                "input",
                "block",
                "off_topic",
                guidance=SECRET,
                error_message=SECRET,
                annotations=SECRET,
                reply=SECRET,
            )
        ],
        user_message=SECRET,
        final_reply=SECRET,
        error_message=SECRET,
    )
    con, out = console()
    con.turn(2, turn, error_type=None, duration_ms=212)
    con.turn(3, None, error_type="ValueError", duration_ms=5)
    assert SECRET not in out.getvalue()
    assert "● turn 2    input blocked · scope (off_topic)" in out.getvalue()


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda c: c.switched("input", "scope", False), "~ input     scope off"),
        (lambda c: c.switched("input", "scope", True), "~ input     scope on"),
        (lambda c: c.edited("output", "tone"), "~ output    tone settings changed"),
        (lambda c: c.moved("output", "tone", 0), "↕ output    tone moved to 1"),
        (lambda c: c.added("input", "keyword_block"), "+ input     keyword_block added"),
        (lambda c: c.deleted("input", "keyword_block"), "- input     keyword_block removed"),
        (lambda c: c.reset(None, None), "↺ orbitals  reset to the code"),
        (lambda c: c.reset("input", "scope"), "↺ input     scope reset to the code"),
        (lambda c: c.undone(), "↺ orbitals  undo"),
        (lambda c: c.preset_imported(2), "⇣ preset    imported · 2 changes"),
        (lambda c: c.cleared(), "○ chat      conversation cleared"),
        (lambda c: c.reloaded(1, 0), "↻ reload    from code · 1 change kept"),
        (lambda c: c.reloaded(0, 2), "↻ reload    from code · 2 changes dropped"),
        (
            lambda c: c.reload_failed("could not import 'shop'"),
            "✗ reload    failed: could not import 'shop'",
        ),
        (
            lambda c: c.test_written("tests/test_turn_3.py"),
            "✎ test      saved tests/test_turn_3.py",
        ),
        (
            lambda c: c.probed("output", "tone", "answered", "block", "rude", 900),
            "▸ probe     tone · block (rude)",
        ),
        (
            lambda c: c.probed("output", "tone", "timeout", None, None, 30000),
            "▸ probe     tone · timeout",
        ),
    ],
)
def test_each_action_has_its_line(call, expected):
    con, out = console()
    call(con)
    assert out.getvalue().startswith(f"  12:04:31  {expected}")
