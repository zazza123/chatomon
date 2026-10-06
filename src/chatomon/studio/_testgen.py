"""Save as test: a turn, or the conversation up to it, as a pytest file.

The test sends the messages to the atom from code again and checks how each turn
ended, which input orbitals stopped it and, when the reply is fixed in code, its
text. It runs the real atom: nothing the nucleus wrote is stored, so what the
output chain decides is checked only when nothing stopped a reply. Only messages,
orbital names, actions and replies fixed in code are written.
"""

from __future__ import annotations

import io
import keyword
import re
import sys
import textwrap
import tokenize
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chatomon.atom import Atom
from chatomon.schemas import TurnReport
from chatomon.studio import _codegen as codegen
from chatomon.studio._chains import RefusedError

if TYPE_CHECKING:
    from chatomon.studio._session import TurnRecord

HEADER = '"""Saved from chatomon studio'
"""How every saved test starts: the studio overwrites only files that start like this."""

CANNOT_SAVE = "A turn that failed cannot be saved as a test."
NO_MODULE = (
    "The studio was started from Python, so it does not know where the atom lives: "
    "complete the import and save the file yourself. To let the studio write it, start "
    "it with `chatomon studio module:attribute`."
)
_NOT_A_NAME = (
    "The atom's module cannot be written as an import, because its name is not a Python "
    "name: complete the import and save the file yourself."
)
_OFFLINE = "No model is called in this test: it runs offline, with {model} in place of the nucleus."
_MARKED = (
    "{who} {call}, so the test is marked `integration`: register the marker in your "
    "pytest configuration and run it when the credentials are set."
)
_OUTPUT = (
    "In turn {n} an output orbital stopped a reply of the nucleus. A real nucleus may write "
    "something else next time, so the test checks only how the turn ended."
)
_FUNCTION = (
    "`{name}` is a function of yours: the studio assumes it calls no model. If it does, "
    "declare it with `@orbital(calls_model=True)`."
)
_FAILED = (
    "`{name}` failed in turn {n} (`{error}`) and its `on_error` policy decided the result: "
    "the test expects the same."
)
_SKIPPED = "Turn {n} failed and left nothing in the conversation: the test leaves it out."
_ALONE = (
    "Turn {n} is saved alone: if it depended on what came before it, as a turn limit does, "
    "the test fails. Replay the earlier turns in that case."
)
_IN_THE_STUDIO = (
    "That is how the turn ran in the studio. Until its changes are in your code, your "
    "code's own orbitals run, and they may call a model."
)
_ORBITAL_NAME = re.compile(r"[A-Za-z0-9_.\-]+")
_CHANGES = (
    "{turns} ran with {count} ({changes}). The test builds the atom from your code, so it "
    "checks your code, not the studio: copy the configuration into your code first."
)
_CHANGED_SINCE = "The orbitals have changed since: Copy as code writes them as they are now."
_THEN = (
    "That is how the turn ran before the code was reloaded: the orbitals your code has now "
    "run, and they may call a model."
)
_RELOADED = (
    "{turns} ran before the code was reloaded: if the code changed what {they}, this test fails."
)
_NO_IMPORT = (
    "# The studio was started from Python, so it does not know where your atom lives.",
    "# Import it here, for example: from your_package.app import atom",
)
_KEEPS_OFFLINE = "# No model is called in this test: the test model keeps the nucleus offline."
_INPUT = 'assert [(r.name, r.action) for r in report.input if r.action != "allow"] == '
_NO_OUTPUT = (
    "assert [",
    '    (r.name, r.action) for a in report.output for r in a if r.action != "allow"',
    "] == []",
)
# Names the test itself uses: something imported under one of them gets an alias.
_USED = frozenset(
    {"atom", "agent", "reports", "report", "result", "monkeypatch", "pytest", "TurnReport"}
)
_FUNCTION_KIND = "FunctionOrbital"
_RESULT = re.compile(r"\bresult\b")


@dataclass(frozen=True)
class Call:
    """How the call that runs a turn is written around the message."""

    before: str = ""
    """What the first argument has before the message."""
    after: str = ""
    """What it has after the message."""
    arguments: tuple[str, ...] = ()
    """The arguments that follow the first."""


@dataclass(frozen=True)
class Offline:
    """What puts a stand-in in place of the nucleus in a test that calls no model."""

    label: str
    """How a note names the stand-in."""
    imports: tuple[str, ...] = ()
    """The import lines it needs."""
    definitions: tuple[str, ...] = ()
    """The lines after the imports that define it, when the framework ships none."""
    before_build: tuple[str, ...] = ()
    """The lines that put it in place before the atom is built."""
    after_build: tuple[str, ...] = ()
    """The lines that put it in place on the built agent."""
    notes: tuple[str, ...] = ()
    """What the developer should know about the stand-in."""
    comment: str = ""
    """The comment above the lines that put it in place, when it is not the usual one."""


@dataclass(frozen=True)
class Framework:
    """The lines of a saved test that depend on the agent framework."""

    run: str
    """What is called to run a turn; a call that starts with `await` makes the test async."""
    first: Call
    """The call of the first turn."""
    later: Call
    """The call of a turn after another, which carries the conversation."""
    reply: str
    """The expression of the reply the user received."""
    offline: Offline
    """What stands in for the nucleus when no model is needed."""
    imports: tuple[str, ...] = ()
    """The import lines every test needs."""
    setup: tuple[str, ...] = ()
    """The lines that follow the build of the agent."""
    marker: str = ""
    """A decorator every test takes."""
    notes: tuple[str, ...] = ()
    """What the developer should know about every test."""

    @property
    def awaited(self) -> bool:
        """Whether the turns are awaited, so the test is a coroutine."""
        return self.run.startswith("await ")

    def names(self) -> frozenset[str]:
        """Return the names these lines use, which nothing else in a test may take.

        Returns:
            Every identifier outside a string literal that is not an attribute.
        """
        offline = self.offline
        lines = [
            self.run,
            self.reply,
            self.marker,
            *self.imports,
            *self.setup,
            *offline.imports,
            *offline.definitions,
            *offline.before_build,
            *offline.after_build,
        ]
        for call in (self.first, self.later):
            lines += [call.before, call.after, *call.arguments]
        return frozenset(name for line in lines for name in _names(line))

    @property
    def carries(self) -> bool:
        """Whether the call of a later turn reads the result of the turn before."""
        later = self.later
        return _RESULT.search(" ".join([later.before, later.after, *later.arguments])) is not None


@dataclass(frozen=True)
class SavedTest:
    """A test and what the developer should know about it."""

    code: str
    name: str
    """The test function's name, which is also the file's."""
    integration: bool
    earlier: int
    """How many turns before the saved one the test can replay."""
    importable: bool
    """Whether the test imports the atom; when not, the import is left to complete."""
    notes: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class _Turn:
    """What a test says about one turn."""

    n: int
    message: str
    action: str
    stopped: list[tuple[str, str]]
    """The input orbitals that did not allow, with what they did."""
    nucleus: bool
    """Whether the nucleus was called."""
    output_stopped: bool
    """Whether an output orbital stopped a reply."""
    remote: list[str]
    """The orbitals that ran and call a model."""
    reply: str | None
    """The expected reply as code, when it is fixed in code."""
    reply_by: str | None
    """The function whose reply is checked, taken as calling no model."""
    functions: list[str]
    """The functions that ran and are taken as calling no model."""
    failed: list[tuple[str, str]]
    """The orbitals that failed, with the error's type."""


def refusal(turn: TurnRecord) -> str | None:
    """Say why a turn cannot be saved as a test.

    Args:
        turn: A finished turn.

    Returns:
        The sentence, or `None` when the turn can be saved.
    """
    report = turn.report
    if turn.error is not None or report is None or report.final_action == "aborted":
        return CANNOT_SAVE
    return None


def slug(message: str, n: int) -> str:
    """Turn a message into the words a test is named after.

    Args:
        message: The user message of the saved turn.
        n: The turn's number, used when the message leaves nothing.

    Returns:
        Up to six lower-case ASCII words joined by `_`, at most 60 characters.
    """
    plain = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", plain.lower())[:6]
    return "_".join(words)[:60].rstrip("_") or f"turn_{n}"


def _listed(items: Sequence[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _nucleus_called(report: TurnReport) -> bool:
    """Tell whether the nucleus wrote in this turn.

    Returns:
        False only when the input chain produced the reply.
    """
    return bool(report.output) or report.final_action not in ("blocked", "orbital_response")


def _read(turn: TurnRecord) -> _Turn:
    report = turn.report
    if report is None:
        raise RefusedError(CANNOT_SAVE)
    ran = [*report.input, *(result for attempt in report.output for result in attempt)]
    nucleus = _nucleus_called(report)
    reply: str | None = None
    reply_by: str | None = None
    if not nucleus and report.input:
        last = report.input[-1]
        if last.reply is None:
            # The verdict carried no reply: the atom's refusal was used.
            reply = "atom.refusal"
        elif turn.reply is not None and not any(r.name in turn.models for r in report.input):
            # Fixed in code only if no orbital that calls a model ran: the one that stopped
            # the turn may reply with what an earlier one left in its annotations.
            reply = codegen.text_literal(turn.reply)
            reply_by = last.name if last.kind == _FUNCTION_KIND else None
    return _Turn(
        n=turn.n,
        message=turn.message,
        action=report.final_action,
        stopped=[(r.name, r.action) for r in report.input if r.action != "allow"],
        nucleus=nucleus,
        output_stopped=any(r.action != "allow" for attempt in report.output for r in attempt),
        remote=list(dict.fromkeys(r.name for r in ran if r.name in turn.models)),
        reply=reply,
        reply_by=reply_by,
        functions=list(
            dict.fromkeys(
                r.name for r in ran if r.kind == _FUNCTION_KIND and r.name not in turn.models
            )
        ),
        failed=list(dict.fromkeys((r.name, r.error_type) for r in ran if r.error_type is not None)),
    )


def _atom(
    source: str, namespace: Mapping[str, object] | None, framework: Framework
) -> tuple[list[str], list[str], list[str]] | None:
    """Write how the test gets the atom.

    Returns:
        The lines among the imports, those after the imports and the first of the test;
        `None` when there is no module, or its name cannot be written in an import.
    """
    if namespace is None:
        return None
    module, _, attribute = source.partition(":")
    head, *rest = attribute.split(".")
    # importlib loads `my-shop`; an import statement cannot name it.
    if not all(_is_name(part) for part in (*module.split("."), head, *rest)):
        return None
    value: object = namespace.get(head)
    for part in rest:
        value = getattr(value, part, None)
    if isinstance(value, Atom) and not rest:
        return [f"from {module} import {head}" + ("" if head == "atom" else " as atom")], [], []
    local = "atom_source" if head in _USED | framework.names() else head
    line = f"from {module} import {head}" + ("" if local == head else f" as {local}")
    path = ".".join([local, *rest])
    if isinstance(value, Atom):
        return [line], [f"atom = {path}"], []
    return [line], [], [f"atom = {path}()"]


def _names(line: str) -> set[str]:
    """Read the identifiers of a line of code, leaving out attributes and keywords.

    Returns:
        The names; what is inside a string or a comment is not one.
    """
    found: set[str] = set()
    previous = ""
    try:
        for token in tokenize.generate_tokens(io.StringIO(line).readline):
            if token.type == tokenize.NAME and previous != ".":
                found.add(token.string)
            previous = token.string
    except (tokenize.TokenError, SyntaxError):
        # A line that opens a bracket another line closes: its names were read before the end.
        pass
    return {name for name in found if not keyword.iskeyword(name)}


def _stdlib(line: str) -> bool:
    """Tell whether an import line names a module of the standard library.

    Returns:
        True for `import uuid` and `from uuid import uuid4`.
    """
    return line.split()[1].partition(".")[0] in sys.stdlib_module_names


def _is_name(text: str) -> bool:
    return text.isidentifier() and not keyword.iskeyword(text)


def _definition(name: str, framework: Framework) -> list[str]:
    start = f"{'async ' if framework.awaited else ''}def {name}("
    line = f"{start}monkeypatch: pytest.MonkeyPatch) -> None:"
    if len(line) <= codegen.WIDTH:
        return [line]
    return [start, "    monkeypatch: pytest.MonkeyPatch,", ") -> None:"]


def _marker(callers: Sequence[str]) -> list[str]:
    # A comment takes a name only when it is an orbital's: anything else could end the comment.
    safe = [
        c if c == "the nucleus" or _ORBITAL_NAME.fullmatch(c) else "an orbital" for c in callers
    ]
    callers = list(dict.fromkeys(safe))
    call = "call real models" if len(callers) > 1 else "calls a real model"
    comment = f"{_listed(callers)} {call}"
    line = f"@pytest.mark.integration  # {comment}"
    if len(line) <= codegen.WIDTH:
        return [line]
    sentence = f"{comment[0].upper()}{comment[1:]}."
    wrapped = textwrap.wrap(
        sentence, codegen.WIDTH - 2, break_long_words=False, break_on_hyphens=False
    )
    return [*(f"# {line}" for line in wrapped), "@pytest.mark.integration"]


def _call(prefix: str, function: str, message: str, call: Call) -> list[str]:
    """Lay the call that runs a turn out on one line when it fits.

    Returns:
        The lines of the call, without the test's indentation.
    """
    text = codegen.text_literal(message)
    one = f"{prefix}{function}({', '.join([call.before + text + call.after, *call.arguments])})"
    if "\n" not in text and len(one) + 4 <= codegen.WIDTH:
        return [one]
    lines = [f"{prefix}{function}("]
    inline = f"    {call.before}{text}{call.after},"
    # What the framework writes around the message must not push it past the width.
    if "\n" in text or (call.before and len(inline) + 4 > codegen.WIDTH):
        # One literal per line, inside the parentheses the framework opens or in its own.
        opened = call.before.endswith("(")
        lines.append(f"    {call.before}{'' if opened else '('}")
        lines += [f"        {part}" for part in text.split("\n")]
        lines.append(f"    {'' if opened else ')'}{call.after},")
    else:
        lines.append(inline)
    return [*lines, *(f"    {argument}," for argument in call.arguments), ")"]


def _checks(turn: _Turn, *, first: bool, last: bool, framework: Framework) -> list[str]:
    """Write one turn: the call and what the test expects of it.

    Returns:
        The lines, without the test's indentation.
    """
    # No name is left unused: the result is kept for the check on the reply, or for the
    # call of the next turn when the framework carries the conversation in it.
    keep = turn.reply is not None or (not last and framework.carries)
    lines = [f"# Turn {turn.n}: {turn.action}"]
    lines += _call(
        "result = " if keep else "",
        framework.run,
        turn.message,
        framework.first if first else framework.later,
    )
    lines += [
        "report = reports[-1]",
        f"assert report.final_action == {codegen.literal(turn.action)}",
    ]
    if turn.stopped:
        pairs = (f"    ({codegen.literal(n)}, {codegen.literal(a)})," for n, a in turn.stopped)
        lines += [f"{_INPUT}[", *pairs, "]"]
    else:
        lines.append(f"{_INPUT}[]")
    if turn.nucleus and not turn.output_stopped:
        lines += _NO_OUTPUT
    if turn.reply is not None and "\n" in turn.reply:
        lines += [
            f"assert {framework.reply} == (",
            *(f"    {p}" for p in turn.reply.split("\n")),
            ")",
        ]
    elif turn.reply is not None:
        lines.append(f"assert {framework.reply} == {turn.reply}")
    return lines


def generate(
    turns: Sequence[TurnRecord],
    *,
    alone: bool,
    source: str,
    namespace: Mapping[str, object] | None,
    signature: object,
    code_signature: object,
    reloads: int,
    framework: Framework,
) -> SavedTest:
    """Write a turn, with the turns before it unless `alone`, as a pytest file.

    Args:
        turns: The conversation up to and including the turn to save.
        alone: Whether to leave the earlier turns out.
        source: How the studio was started: `module:attribute` or `serve()`.
        namespace: The variables of the atom's module, or `None` without a module.
        signature: What identifies the session's orbitals now, to compare with
            what each turn recorded.
        code_signature: What identifies the orbitals the code has now, with none
            of the studio's changes: a turn that ran with exactly these needs no
            copy into the code.
        reloads: How many times the code was loaded again so far, to compare
            with what each turn recorded.
        framework: The lines that depend on the agent framework.

    Returns:
        The code, the test's name, and what the developer should know.

    Raises:
        RefusedError: If the turn to save failed.
    """
    saved = turns[-1]
    why = refusal(saved)
    if why is not None:
        raise RefusedError(why)
    before = [turn for turn in turns[:-1] if refusal(turn) is None]
    skipped = [turn.n for turn in turns[:-1] if refusal(turn) is not None]
    records = [saved] if alone else [*before, saved]
    replayed = [_read(turn) for turn in records]
    offline = not any(turn.nucleus or turn.remote for turn in replayed)
    callers = ["the nucleus"] if any(turn.nucleus for turn in replayed) else []
    callers += dict.fromkeys(name for turn in replayed for name in turn.remote)
    name = f"test_{slug(saved.message, saved.n)}"

    numbers = [turn.n for turn in replayed]
    span = f"turn {numbers[0]}" if len(numbers) == 1 else f"turns {numbers[0]}-{numbers[-1]}"
    shown = source if re.fullmatch(r"[\w.:()]+", source) else "an atom"
    written = _atom(source, namespace, framework)
    imports, after, first = written if written is not None else (["", *_NO_IMPORT], [], [])
    stdlib = [line for line in framework.imports if _stdlib(line)]
    lines = [f'{HEADER}: {shown}, {span}."""', ""]
    if stdlib:
        lines += [*stdlib, ""]
    lines += ["import pytest", "", "from chatomon import TurnReport"]
    lines += [line for line in framework.imports if line not in stdlib]
    stand_in = framework.offline
    if offline:
        lines += stand_in.imports
    lines += imports
    if after:
        lines += ["", *after]
    if offline and stand_in.definitions:
        lines += ["", "", *stand_in.definitions]
    lines += ["", ""]
    if not offline:
        lines += _marker(callers)
    if framework.marker:
        lines.append(framework.marker)
    lines += _definition(name, framework)
    body = [*first, "reports: list[TurnReport] = []"]
    body.append('monkeypatch.setattr(atom, "on_turn", reports.append)')
    keeps_offline = stand_in.comment or _KEEPS_OFFLINE
    if offline and stand_in.before_build:
        body += [keeps_offline, *stand_in.before_build]
    body.append("agent = atom.build()")
    if offline and stand_in.after_build:
        body += [keeps_offline, *stand_in.after_build]
    body += framework.setup
    for index, turn in enumerate(replayed):
        body.append("")
        body += _checks(
            turn, first=index == 0, last=index == len(replayed) - 1, framework=framework
        )
    lines += [f"    {line}" if line else "" for line in body]

    # A turn whose orbitals are the code's of now ran with changes the code has since taken.
    changed = [turn for turn in records if turn.changes and turn.signature != code_signature]
    earlier = [str(turn.n) for turn in records if turn.reloads != reloads]
    notes: list[str] = []
    if offline:
        notes.append(_OFFLINE.format(model=stand_in.label))
        if changed:
            notes.append(_IN_THE_STUDIO)
        if earlier:
            notes.append(_THEN)
        notes += stand_in.notes
    else:
        who = _listed([c if c == "the nucleus" else f"`{c}`" for c in callers])
        call = "call real models" if len(callers) > 1 else "calls a real model"
        notes.append(_MARKED.format(who=who[0].upper() + who[1:], call=call))
    notes += framework.notes
    notes += [_OUTPUT.format(n=turn.n) for turn in replayed if turn.output_stopped]
    assumed = (
        [name for turn in replayed for name in turn.functions]
        if offline
        else [turn.reply_by for turn in replayed if turn.reply_by is not None]
    )
    notes += [_FUNCTION.format(name=name) for name in dict.fromkeys(assumed)]
    notes += [
        _FAILED.format(name=name, n=turn.n, error=error)
        for turn in replayed
        for name, error in turn.failed
    ]
    if not alone:
        notes += [_SKIPPED.format(n=n) for n in skipped]
    elif before:
        notes.append(_ALONE.format(n=saved.n))
    if namespace is None:
        notes.append(NO_MODULE)
    elif written is None:
        notes.append(_NOT_A_NAME)

    # Turns that ran with the same changes share one sentence.
    groups: dict[tuple[str, ...], list[str]] = {}
    for turn in changed:
        groups.setdefault(tuple(turn.changes), []).append(str(turn.n))
    warnings = [
        _CHANGES.format(
            turns=f"Turn{'' if len(numbers) == 1 else 's'} {_listed(numbers)}",
            count=f"{len(group)} studio change{'' if len(group) == 1 else 's'}",
            changes=", ".join(group),
        )
        for group, numbers in groups.items()
    ]
    # Compared by signature: the same sentence can stand for two values of one setting.
    if any(turn.signature != signature for turn in changed):
        warnings.append(_CHANGED_SINCE)
    # The test runs the code as it is now: what an earlier load of it did may not repeat.
    if earlier:
        one = len(earlier) == 1
        warnings.append(
            _RELOADED.format(
                turns=f"Turn{'' if one else 's'} {_listed(earlier)}",
                they="it checks" if one else "they check",
            )
        )
    return SavedTest(
        code="\n".join(lines) + "\n",
        name=name,
        integration=not offline,
        earlier=len(before),
        importable=written is not None,
        notes=notes,
        warnings=warnings,
    )
