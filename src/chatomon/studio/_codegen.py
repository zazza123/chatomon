"""Copy as code: the atom's two orbital lists as Python.

The studio sees objects, not source. An orbital the developer did not touch is
written with the name it has in the atom's module; one the studio can rebuild is
written as a constructor call with the arguments that differ from the defaults;
anything else is marked with `...` and listed. Only variable names and plain
settings are written, never the `repr` of an object.
"""

from __future__ import annotations

import json
import keyword
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import chatomon.orbitals as library
from chatomon.knowledge import Knowledge
from chatomon.orbitals.base import Orbital
from chatomon.schemas import Side
from chatomon.studio import _kinds as kinds
from chatomon.studio._chains import SIDES, Chains, Entry
from chatomon.studio._values import OPAQUE, Setting, ValueType, plain, represented

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
# Characters that are valid inside a Python literal but that editors show as line breaks.
_BREAKS = {chr(code): f"\\u{code:04x}" for code in (0x85, 0x2028, 0x2029)}

Import = tuple[str, str]


class Argument(NamedTuple):
    """A keyword, its text and a trailing comment."""

    key: str
    text: str
    comment: str | None = None
    laid: bool = False
    """Whether the text is a call already laid out on its lines, not a text in parts."""


@dataclass(frozen=True)
class Code:
    """The generated code and what the developer should know about it."""

    imports: str
    setup: str
    """Assignments that go before the atom: orbitals the code puts in both chains."""
    orbitals: str
    placeholders: list[str]
    notes: list[str]

    @property
    def code(self) -> str:
        """The imports, the assignments and the two lists, as the card shows them."""
        parts = []
        if self.imports:
            parts.append(f"# Imports\n{self.imports}")
        if self.setup:
            parts.append(f"# Before the atom\n{self.setup}")
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


def _ascii(value: str) -> str:
    """Write text that holds a lone surrogate with escapes only.

    `json.dumps` writes a character past the basic plane as a pair of surrogates,
    which Python reads back as two characters.

    Returns:
        An ASCII literal that reads back to the same text.
    """
    parts: list[str] = []
    for char in value:
        code = ord(char)
        if code > 0xFFFF:
            parts.append(f"\\U{code:08x}")
        elif 0xD800 <= code <= 0xDFFF:
            parts.append(f"\\u{code:04x}")
        else:
            parts.append(json.dumps(char)[1:-1])
    return '"' + "".join(parts) + '"'


def literal(value: Any) -> str:
    """Write a plain value as a Python literal that reads back to the same value.

    Returns:
        The literal's text.
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return _ascii(value)
        written = json.dumps(value, ensure_ascii=False)
        for char, escape in _BREAKS.items():
            written = written.replace(char, escape)
        return written
    if isinstance(value, float) and not math.isfinite(value):
        return f'float("{value}")'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(literal(item) for item in value) + "]"
    return repr(value)


def text_literal(value: str) -> str:
    """Write text as code; with line breaks, one literal per line, to go inside parentheses.

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
    simple = all(arg.comment is None and "\n" not in arg.text for arg in args)
    one = f"{name}({', '.join(f'{arg.key}={arg.text}' for arg in args)})"
    if simple and len(one) + 5 <= WIDTH:  # four spaces of indentation and the comma
        return [one]
    lines = [f"{name}("]
    for arg in args:
        parts = arg.text.split("\n")
        if len(parts) == 1:
            lines.append(
                f"    {arg.key}={arg.text}," + (f"  # {arg.comment}" if arg.comment else "")
            )
        elif arg.laid:
            lines += [f"    {arg.key}={parts[0]}", *(f"    {part}" for part in parts[1:-1])]
            lines.append(f"    {parts[-1]},")
        else:
            lines += [f"    {arg.key}=(", *(f"        {part}" for part in parts), "    ),"]
    return [*lines, ")"]


def _bracketed(head: str, tail: str, items: list[list[str]], used: int) -> list[str]:
    """Lay out a call or a list: on one line when it fits, otherwise one item per line.

    Args:
        head: What opens it, such as `ApiBackend(` or `[`.
        tail: What closes it.
        items: The arguments or the entries, each as its own lines.
        used: The columns its first line has taken before it, with the comma after it.

    Returns:
        The lines, the first without indentation and the others relative to it.
    """
    one = head + ", ".join(item[0] for item in items) + tail
    if all(len(item) == 1 for item in items) and used + len(one) <= WIDTH:
        return [one]
    lines = [head]
    for item in items:
        lines += [f"    {line}" for line in item[:-1]]
        lines.append(f"    {item[-1]},")
    return [*lines, tail]


def _instance(
    found: ValueType, value: Any, indent: int, lead: int, needs: set[Import]
) -> list[str]:
    """Write a value type as a call: its required fields and those that differ from the default.

    The fields are read from the object, not from its JSON form, so a text the
    code gives where a list could be is written as that text.

    Args:
        found: The type as the studio reads it.
        value: The instance.
        indent: The indentation of the line the call starts on.
        lead: The characters between that indentation and the call.
        needs: The imports the code needs; the type's own is added.

    Returns:
        The lines of the call, relative to the line it starts on.
    """
    needs.add(_import(found.cls))
    items: list[list[str]] = []
    for item in found.fields:
        raw = getattr(value, item.key)
        if not item.required and raw == item.default:
            continue
        lines = _field(item, raw, indent + 4, len(item.key) + 1, needs)
        items.append([f"{item.key}={lines[0]}", *lines[1:]])
    # A field the studio does not read and that has no default has to be given, as it is here.
    items += [[f"{name}={literal(given)}"] for name, given, passed in found.unread if passed]
    return _bracketed(f"{found.cls.__name__}(", ")", items, indent + lead + 1)


def _field(item: Setting, raw: Any, indent: int, lead: int, needs: set[Import]) -> list[str]:
    """Write the value of one field: a plain value, or a list of texts and entries.

    Returns:
        The lines of the value, relative to the line it starts on.
    """
    if not isinstance(raw, list | tuple):
        return [literal(raw)]
    entries: list[list[str]] = []
    for entry in raw:
        known = next((kind for kind in item.types if type(entry) is kind.cls), None)
        if known is None:
            entries.append([literal(entry)])
        else:
            entries.append(_instance(known, entry, indent + 4, 0, needs))
    return _bracketed("[", "]", entries, indent + lead + 1)


def written(setting: Setting, value: Any) -> str:
    """Write a block on one line, as the argument Copy as code would write.

    Args:
        setting: The block.
        value: What the orbital holds for it: a text or an instance of a value type.

    Returns:
        The argument, such as `guard=ApiBackend(api_url="https://…")`.
    """
    found = next((known for known in setting.types if type(value) is known.cls), None)
    if found is None:
        return f"{setting.key}={literal(value)}"
    # An indentation far to the left: whatever its length, the call stays on one line.
    return f"{setting.key}={_instance(found, value, -(10**6), 0, set())[0]}"


@dataclass
class _Writer:
    chains: Chains
    known: dict[int, str]
    shared: dict[int, str] = field(default_factory=dict)
    """Variables for the orbitals written once and used in both chains, by object id."""
    placeholders: list[str] = field(default_factory=list)

    def entry(self, entry: Entry) -> tuple[list[str], str | None, set[Import]]:
        """Return an orbital's lines, a trailing comment and the imports they need."""
        orb = entry.orbital
        name = self.known.get(id(orb)) if orb is entry.base else None
        if name is None:
            name = self.shared.get(id(orb))
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
                return text_literal(value.text), None, set()
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
            args.append(Argument("name", literal(orb.name)))
        for key in kind.order:
            if key == "name":
                continue
            setting = kind.setting(key)
            if setting is not None and setting.type == "block":
                more = self._block(orb, setting, args)
            elif setting is None:
                value = kinds.read(orb, key, self.chains.connector)
                if value is None:
                    continue
                text, comment, more = self._object(orb.name, key, value)
                args.append(Argument(key, text, comment))
            else:
                raw = getattr(orb, key)
                if not setting.required and plain(raw) == plain(setting.default):
                    continue
                if isinstance(raw, Knowledge):
                    text, comment, more = self._object(orb.name, key, raw)
                elif isinstance(raw, str):
                    text, comment, more = text_literal(raw), None, set()
                else:
                    text, comment, more = literal(plain(raw)), None, set()
                args.append(Argument(key, text, comment))
            needs |= more
        return _layout(cls.__name__, args), needs

    def _block(self, orb: Orbital, setting: Setting, args: list[Argument]) -> set[Import]:
        """Write a block: the code's own object by its name, a text, or a call of its type.

        Returns:
            The imports the argument needs; nothing is added for a block that is none.
        """
        value = getattr(orb, setting.key)
        if value is None:
            return set()
        found = next((known for known in setting.types if type(value) is known.cls), None)
        nameless = id(value) not in self.known
        if found is None or not nameless or represented(setting, value) is OPAQUE:
            # A name in the module, one text, or an object the studio cannot write.
            text, comment, more = self._object(orb.name, setting.key, value)
            args.append(Argument(setting.key, text, comment))
            return more
        needs: set[Import] = set()
        lines = _instance(found, value, 8, len(setting.key) + 1, needs)
        args.append(Argument(setting.key, "\n".join(lines), laid=True))
        return needs


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


def _variable(name: str, taken: set[str]) -> str:
    """Turn an orbital's name into a variable that nothing else uses.

    Returns:
        A Python identifier, added to `taken`.
    """
    base = re.sub(r"\W", "_", name)
    if not base or base[0].isdigit():
        base = f"_{base}"
    candidate, number = base, 2
    while candidate in taken or keyword.iskeyword(candidate):
        candidate, number = f"{base}_{number}", number + 1
    taken.add(candidate)
    return candidate


def _assignments(writer: _Writer, taken: set[str]) -> tuple[list[str], set[Import]]:
    """Write once every orbital that both chains hold and that needs a constructor call.

    Two calls would build two orbitals with one name, which an atom refuses: the
    orbital is assigned to a variable and both lists use it.

    Returns:
        The assignment lines and the imports they need.
    """
    chains = writer.chains
    first: dict[int, Entry] = {id(entry.orbital): entry for entry in chains.work["input"]}
    lines: list[str] = []
    needs: set[Import] = set()
    for entry in chains.work["output"]:
        orb = entry.orbital
        twin = first.get(id(orb))
        named = orb is entry.base and id(orb) in writer.known
        if twin is None or named or chains.note(entry) is not None or id(orb) in writer.shared:
            continue
        call, more = writer._call(orb)
        variable = _variable(orb.name, taken)
        writer.shared[id(orb)] = variable
        lines += [f"{variable} = {call[0]}", *call[1:]]
        needs |= more
    return lines, needs


def _imported(module: str, names: list[str]) -> str:
    """Write one import: on one line when it fits, otherwise one name per line.

    Returns:
        The import statement.
    """
    one = f"from {module} import {', '.join(names)}"
    if len(one) <= WIDTH:
        return one
    return "\n".join([f"from {module} import (", *(f"    {name}," for name in names), ")"])


def generate(chains: Chains, namespace: Mapping[str, object] | None) -> Code:
    """Write the session's orbitals as the atom's two lists.

    Args:
        chains: The session's chains.
        namespace: The variables of the atom's module, or `None` without a module.

    Returns:
        The code, one sentence per placeholder, and the notes that apply.
    """
    writer = _Writer(chains, names(namespace))
    setup, needs = _assignments(writer, set(namespace or ()))
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
        _imported(module, sorted(name for found, name in needs if found == module))
        for module in modules
    )
    notes: list[str] = []
    if not blocks:
        notes.append(EMPTY)
    if writer.placeholders:
        notes.append(NO_MODULE if namespace is None else NO_NAME)
    return Code(imports, "\n".join(setup), "\n".join(blocks), writer.placeholders, notes)
