"""Copy as code: the atom's two orbital lists as Python.

The studio sees objects, not source. An orbital the developer did not touch is
written with the name it has in the atom's module; one the studio can rebuild is
written as a constructor call with the arguments that differ from the defaults;
anything else is marked with `...` and listed. Only variable names and plain
settings are written, never the `repr` of an object.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import chatomon.orbitals as library
from chatomon.knowledge import Knowledge
from chatomon.orbitals.base import Orbital
from chatomon.schemas import Side
from chatomon.studio import _kinds as kinds
from chatomon.studio._chains import SIDES, Chains, Entry

WIDTH = 88
OFF = "# Off in the studio:"
EMPTY = "This atom has no orbitals."
NO_MODULE = (
    "The studio was started from Python, so it has no module to look names up in: "
    "objects from your code are marked `...`."
)
NO_NAME = (
    "What is marked `...` has no name in the atom's module: it is created inline or "
    "inside a function. Keep it as it is in your code."
)

Import = tuple[str, str]
Argument = tuple[str, str, str | None]
"""A keyword, its text and a trailing comment."""


@dataclass(frozen=True)
class Code:
    """The generated code and what the developer should know about it."""

    imports: str
    orbitals: str
    placeholders: list[str]
    notes: list[str]

    @property
    def code(self) -> str:
        """The imports and the two lists, as the card shows them."""
        parts = []
        if self.imports:
            parts.append(f"# Imports\n{self.imports}")
        if self.orbitals:
            parts.append(f"# Inside Atom(...)\n{self.orbitals}")
        return "\n\n".join(parts)


def names(namespace: Mapping[str, object] | None) -> dict[int, str]:
    """Map each object a namespace holds to the first name it has there.

    Plain values are left out: they are always written as literals.

    Args:
        namespace: A module's variables, or `None` when there is no module.

    Returns:
        Names by object id.
    """
    found: dict[int, str] = {}
    for name, value in (namespace or {}).items():
        if name.startswith("__") or value is None or isinstance(value, str | int | float):
            continue
        found.setdefault(id(value), name)
    return found


def literal(value: Any) -> str:
    """Write a plain value as a Python literal that reads back to the same value.

    Returns:
        The literal's text.
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return json.dumps(value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and not math.isfinite(value):
        return f'float("{value}")'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(literal(item) for item in value) + "]"
    return repr(value)


def _text(value: str) -> str:
    """Write text; with line breaks, one literal per line, to go inside parentheses.

    Returns:
        One literal, or several separated by line breaks.
    """
    if "\n" not in value:
        return literal(value)
    lines = value.split("\n")
    parts = [f"{line}\n" for line in lines[:-1]]
    if lines[-1]:
        parts.append(lines[-1])
    return "\n".join(literal(part) for part in parts)


def _import(cls: type) -> Import:
    public = getattr(library, cls.__name__, None) is cls
    return ("chatomon.orbitals" if public else cls.__module__, cls.__name__)


def _layout(name: str, args: list[Argument]) -> list[str]:
    """Lay a call out on one line when it fits, otherwise one argument per line.

    Returns:
        The lines of the call, without indentation.
    """
    simple = all(comment is None and "\n" not in text for _, text, comment in args)
    one = f"{name}({', '.join(f'{key}={text}' for key, text, _ in args)})"
    if simple and len(one) + 5 <= WIDTH:  # four spaces of indentation and the comma
        return [one]
    lines = [f"{name}("]
    for key, text, comment in args:
        if "\n" in text:
            lines += [f"    {key}=(", *(f"        {part}" for part in text.split("\n")), "    ),"]
        else:
            lines.append(f"    {key}={text}," + (f"  # {comment}" if comment else ""))
    return [*lines, ")"]


@dataclass
class _Writer:
    chains: Chains
    known: dict[int, str]
    placeholders: list[str] = field(default_factory=list)

    def entry(self, entry: Entry) -> tuple[list[str], str | None, set[Import]]:
        """Return an orbital's lines, a trailing comment and the imports they need."""
        orb = entry.orbital
        name = self.known.get(id(orb)) if orb is entry.base else None
        if name is not None:
            return [name], None, set()
        if self.chains.note(entry) is None:
            lines, needs = self._call(orb)
            return lines, None, needs
        kind = type(orb).__name__
        self.placeholders.append(f"{orb.name}: a {kind} from your code")
        return ["..."], f"keep {orb.name!r} ({kind}) from your code", set()

    def _object(self, owner: str, key: str, value: Any) -> tuple[str, str | None, set[Import]]:
        """Write an object: its name, a knowledge or a plain value as code, else a placeholder.

        Returns:
            The text, a trailing comment and the imports the text needs.
        """
        name = self.known.get(id(value))
        if name is not None:
            return name, None, set()
        if isinstance(value, Knowledge):
            if value.title is None:
                return _text(value.text), None, set()
            text = f"Knowledge({literal(value.text)}, title={literal(value.title)})"
            return text, None, {("chatomon", "Knowledge")}
        if isinstance(value, str | int | float):
            return literal(value), None, set()
        kind = type(value).__name__
        self.placeholders.append(f"{owner}: `{key}` is a {kind} from your code")
        return "...", f"your {kind}", set()

    def _call(self, orb: Orbital) -> tuple[list[str], set[Import]]:
        cls = type(orb)
        kind = kinds.describe(cls)
        needs: set[Import] = {_import(cls)}
        args: list[Argument] = []
        if orb.name != cls.default_name():
            args.append(("name", literal(orb.name), None))
        for key in kind.order:
            if key == "name":
                continue
            setting = kind.setting(key)
            if setting is None:
                value = kinds.read(orb, key, self.chains.connector)
                if value is None:
                    continue
                text, comment, more = self._object(orb.name, key, value)
            else:
                raw = getattr(orb, key)
                if not setting.required and kinds.plain(raw) == kinds.plain(setting.default):
                    continue
                if isinstance(raw, Knowledge):
                    text, comment, more = self._object(orb.name, key, raw)
                elif isinstance(raw, str):
                    text, comment, more = _text(raw), None, set()
                else:
                    text, comment, more = literal(kinds.plain(raw)), None, set()
            args.append((key, text, comment))
            needs |= more
        return _layout(cls.__name__, args), needs


def _side(side: Side, rendered: list[tuple[Entry, list[str], str | None]]) -> list[str]:
    """Write one list: on one line when every orbital is on and it fits.

    Returns:
        The lines of the list.
    """
    head = f"{side}_orbitals=["
    inline = all(e.enabled and len(lines) == 1 and c is None for e, lines, c in rendered)
    one = head + ", ".join(lines[0] for _, lines, _ in rendered) + "],"
    if inline and len(one) <= WIDTH:
        return [one]
    out = [head]
    for entry, lines, comment in rendered:
        body = [*lines[:-1], lines[-1] + "," + (f"  # {comment}" if comment else "")]
        if entry.enabled:
            out += [f"    {line}" for line in body]
        else:
            out += [f"    {OFF}", *(f"    # {line}" for line in body)]
    return [*out, "],"]


def generate(chains: Chains, namespace: Mapping[str, object] | None) -> Code:
    """Write the session's orbitals as the atom's two lists.

    Args:
        chains: The session's chains.
        namespace: The variables of the atom's module, or `None` without a module.

    Returns:
        The code, one sentence per placeholder, and the notes that apply.
    """
    writer = _Writer(chains, names(namespace))
    needs: set[Import] = set()
    blocks: list[str] = []
    for side in SIDES:
        rendered: list[tuple[Entry, list[str], str | None]] = []
        for entry in chains.work[side]:
            lines, comment, more = writer.entry(entry)
            rendered.append((entry, lines, comment))
            if entry.enabled:
                needs |= more
        if rendered:
            blocks.append("\n".join(_side(side, rendered)))
    modules = sorted({module for module, _ in needs})
    imports = "\n".join(
        f"from {module} import {', '.join(sorted(n for m, n in needs if m == module))}"
        for module in modules
    )
    notes: list[str] = []
    if not blocks:
        notes.append(EMPTY)
    if writer.placeholders:
        notes.append(NO_MODULE if namespace is None else NO_NAME)
    return Code(imports, "\n".join(blocks), writer.placeholders, notes)
