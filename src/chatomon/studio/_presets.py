"""Presets: a session's orbital changes as a JSON layer over the atom from code.

A preset never holds an atom. It says which orbitals are on, their order, the
settings the studio changed on the orbitals from code, and the orbitals added
in the studio. Applying it starts from the atom's own chains, so the code stays
the source of truth for whatever the preset does not mention.

A setting that is a block is written whole, as its type and the fields that
differ from the type's defaults. It never holds a secret: a block names the
environment variable that does, and a preset that names one is applied with a
note that shows the block.

The same layer takes a session's changes over code that was loaded again:
[carried][chatomon.studio._presets.carried] applies it entry by entry and
drops, instead of refusing, what no longer applies.
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

from chatomon._version import __version__
from chatomon.errors import ChatomonError
from chatomon.schemas import Side
from chatomon.studio import _codegen as codegen
from chatomon.studio import _kinds as kinds
from chatomon.studio import _values as values
from chatomon.studio._chains import SIDES, Chains, Entry, RefusedError, UnknownOrbitalError

VERSION = 1
"""The schema version this studio writes and reads."""

_TOP = {"chatomon_studio_preset", "chatomon", "atom", "input_orbitals", "output_orbitals"}
_ENTRY = {"name", "kind", "from", "enabled", "settings", "fingerprint"}
_FIELDS: dict[str, tuple[type, str]] = {
    "name": (str, "text"),
    "kind": (str, "text"),
    "enabled": (bool, "true or false"),
    "fingerprint": (str, "text"),
}
_NOT_PRESET = 'this is not a studio preset: it has no "chatomon_studio_preset" version'


class InvalidPresetError(ChatomonError):
    """A preset that cannot be applied; `problems` has one sentence per problem."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("the preset was not applied")
        self.problems = problems


class _GoneError(RefusedError):
    """The code does not define the orbital an entry names, on that side."""


class _OtherClassError(RefusedError):
    """The code defines the orbital an entry names as another class."""

    def __init__(self, message: str, *, now: str) -> None:
        super().__init__(message)
        self.now = now


def _entry(chains: Chains, entry: Entry) -> dict[str, Any]:
    orb = entry.orbital
    data: dict[str, Any] = {
        "name": orb.name,
        "kind": type(orb).__name__,
        "from": entry.origin,
        "enabled": entry.enabled,
    }
    if chains.note(entry) is None:
        kind = kinds.describe(type(orb))
        now = kinds.values(orb)
        # An orbital from code carries what differs from the code; an added one, everything.
        before = kinds.values(entry.base) if entry.base is not None else None
        settings: dict[str, Any] = {}
        for key, value in now.items():
            if before is not None and value == before.get(key):
                continue
            setting = kind.setting(key)
            block = setting is not None and setting.type == "block"
            settings[key] = values.compact(setting, value) if setting and block else value
        if settings:
            data["settings"] = settings
    data["fingerprint"] = orb.fingerprint()
    return data


def exported(chains: Chains, *, atom: str) -> dict[str, Any]:
    """Return the session's chains as a preset.

    Args:
        chains: The session's chains.
        atom: The atom's name.

    Returns:
        JSON-ready data: names, kinds, plain settings and fingerprints only.
    """
    preset: dict[str, Any] = {
        "chatomon_studio_preset": VERSION,
        "chatomon": __version__,
        "atom": atom,
    }
    for side in SIDES:
        preset[f"{side}_orbitals"] = [_entry(chains, entry) for entry in chains.work[side]]
    return preset


def _parsed(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        where = f"(line {exc.lineno}, column {exc.colno})"
        raise InvalidPresetError([f"this is not valid JSON: {exc.msg} {where}"]) from exc
    except RecursionError as exc:
        raise InvalidPresetError(["this is not valid JSON: it is nested too deeply"]) from exc
    except ValueError as exc:
        # Python refuses to read an integer of thousands of digits.
        raise InvalidPresetError(
            ["this is not valid JSON: it has a number too large to read"]
        ) from exc
    try:
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        # A lone surrogate from a JSON escape could not be sent back in a message.
        raise InvalidPresetError(["this preset has text that is not valid Unicode"]) from exc
    version = data.get("chatomon_studio_preset") if isinstance(data, dict) else None
    if version is None:
        raise InvalidPresetError([_NOT_PRESET])
    if isinstance(version, bool) or version != VERSION:
        raise InvalidPresetError(
            [f"this preset has schema version {version!r}; this studio reads version {VERSION}"]
        )
    return data


def _malformed(raw: Any, where: str) -> list[str]:
    if not isinstance(raw, dict):
        return [f"{where} must be an object"]
    problems = [f"{where} has an unknown key {key!r}" for key in sorted(set(raw) - _ENTRY)]
    for key, (kind, word) in _FIELDS.items():
        if not isinstance(raw.get(key), kind):
            problems.append(f"{where} needs {key!r} as {word}")
    if raw.get("from") not in ("code", "studio"):
        problems.append(f'{where} needs \'from\' as "code" or "studio"')
    if "settings" in raw and not isinstance(raw["settings"], dict):
        problems.append(f"{where} needs 'settings' as an object")
    return problems


def _from_code(
    result: Chains,
    side: Side,
    raw: dict[str, Any],
    settings: dict[str, Any],
    shared: dict[int, dict[str, Any]],
) -> Chains:
    """Apply an entry to the orbital the code defines with that side and name.

    Returns:
        The chains with the entry's settings applied.

    Raises:
        _GoneError: If the code does not define it on that side.
        _OtherClassError: If the code defines it as another class.
        RefusedError: With any other reason the entry cannot be applied.
    """
    name = raw["name"]
    try:
        where, index = result.find(side, name)
        base = result.work[where][index].base
    except UnknownOrbitalError:
        base = None
    if base is None:
        raise _GoneError(
            "it is not in your code, so a preset cannot bring it back: add it to your code first"
        )
    if type(base).__name__ != raw["kind"]:
        raise _OtherClassError(
            f"it is a {type(base).__name__} in your code, a {raw['kind']} in the preset",
            now=type(base).__name__,
        )
    if id(base) in shared:
        if shared[id(base)] != settings:
            raise RefusedError(
                "it is one orbital in both chains of your code, "
                "but the preset gives it different settings"
            )
        return result
    shared[id(base)] = settings
    return result.edited(side, name, settings) if settings else result


def _entry_applied(
    result: Chains, side: Side, raw: dict[str, Any], shared: dict[int, dict[str, Any]]
) -> Chains:
    """Apply one well-formed entry: its settings, or the orbital it adds, and its switch.

    Returns:
        The chains with the entry applied.

    Raises:
        ChatomonError: With the reason the entry cannot be applied.
    """  # noqa: DOC502 - raised by the helpers
    name, settings = raw["name"], raw.get("settings") or {}
    if raw["from"] == "code":
        result = _from_code(result, side, raw, settings, shared)
    else:
        result = result.added(side, raw["kind"], {**settings, "name": name})
    return result if raw["enabled"] else result.switched(side, name, False)


class Notes(NamedTuple):
    """What applying a preset has to say."""

    connections: list[str]
    """Where the preset makes an orbital connect, one sentence for each connection."""
    warnings: list[str]
    """What differs from the exported session."""


_CHECK = "Check where it points before you send a message."


def _connections(result: Chains, side: Side, raw: dict[str, Any]) -> list[str]:
    """Say where an entry's blocks make its orbital connect, showing each block.

    A preset decides where a key and the conversation are sent: the note shows the
    block as Copy as code writes it, so the developer sees all of it. A block with
    a field for an environment variable is a connection also when the field is
    empty: a client can read a key from the environment on its own.

    Returns:
        One sentence per variable named and one per block that leaves it out; none
        for an entry whose blocks have no such field.
    """
    name = raw["name"]
    where, index = result.find(side, name)
    orb = result.work[where][index].orbital
    kind = kinds.describe(type(orb))
    notes: list[str] = []
    for key, value in (raw.get("settings") or {}).items():
        setting = kind.setting(key)
        if setting is None or setting.type != "block":
            continue
        # A letter of another alphabet can make an address look like one the developer
        # trusts: whatever is not ASCII is written as an escape.
        shown = codegen.written(setting, getattr(orb, key))
        shown = shown.encode("ascii", "backslashreplace").decode("ascii")
        notes += [
            f"{name!r} uses the environment variable {variable}: {shown}. {_CHECK}"
            for variable in values.variables(setting, value)
        ]
        if values.unnamed(setting, value):
            notes.append(
                f"{name!r} names no environment variable for its key: {shown}. Its client "
                f"can still send a key it reads from the environment on its own. {_CHECK}"
            )
    return notes


def applied(text: str, chains: Chains, *, atom: str) -> tuple[Chains, Notes]:
    """Apply a preset to the atom's own chains.

    Args:
        text: The preset, as JSON text.
        chains: The session's chains; only the atom's own are used.
        atom: The loaded atom's name.

    Returns:
        The new chains and the notes: where the preset makes an orbital connect,
        and apart from it the warnings, which are the orbitals from code the preset
        does not mention and those that differ from the exported ones.

    Raises:
        InvalidPresetError: With every problem found; nothing is applied.
    """
    data = _parsed(text)
    problems = [f"unknown key {key!r}" for key in sorted(set(data) - _TOP)]
    if data.get("atom") != atom:
        problems.append(
            f"this preset is for the atom {data.get('atom')!r}, but the studio has {atom!r} loaded"
        )
    result = chains.initial()
    wanted: dict[Side, list[str]] = {"input": [], "output": []}
    expected: list[tuple[Side, str, str]] = []
    connected: list[tuple[Side, dict[str, Any]]] = []
    shared: dict[int, dict[str, Any]] = {}
    for side in SIDES:
        entries = data.get(f"{side}_orbitals", [])
        if not isinstance(entries, list):
            problems.append(f"{side}_orbitals must be a list")
            continue
        for index, raw in enumerate(entries):
            malformed = _malformed(raw, f"{side}_orbitals[{index}]")
            if malformed:
                problems += malformed
                continue
            name, settings = raw["name"], raw.get("settings") or {}
            if name in wanted[side]:
                problems.append(f"{name!r} appears twice in {side}_orbitals")
                continue
            if raw["from"] == "studio" and "name" in settings:
                problems.append(f"{name!r}: `name` is the entry's own field, not a setting")
                continue
            try:
                result = _entry_applied(result, side, raw, shared)
            except ChatomonError as exc:
                problems.append(f"{name!r}: {exc}")
                continue
            wanted[side].append(name)
            expected.append((side, name, raw["fingerprint"]))
            connected.append((side, raw))
    if problems:
        raise InvalidPresetError(problems)
    connections: list[str] = []
    for side, raw in connected:
        connections += _connections(result, side, raw)
    warnings: list[str] = []
    for side in SIDES:
        missing = [e.orbital.name for e in result.work[side] if e.orbital.name not in wanted[side]]
        for position, name in enumerate([*wanted[side], *missing]):
            result = result.moved(side, name, position)
        warnings += [
            f"{name!r} is in your code but not in the preset: it was added at the end, turned on."
            for name in missing
        ]
    differing: list[str] = []
    for side, name, fingerprint in expected:
        where, index = result.find(side, name)
        if result.work[where][index].orbital.fingerprint() != fingerprint and name not in differing:
            differing.append(name)
    warnings += [
        f"{name!r} differs from the exported preset: your code or chatomon changed since."
        for name in differing
    ]
    return result, Notes(connections, warnings)


_LOST = "what the studio changed on it was dropped."


def carried(session: Chains, code: Chains) -> tuple[Chains, list[str]]:
    """Lay a session's changes over chains built from code that was loaded again.

    The rules of an entry are those of [applied][chatomon.studio._presets.applied],
    but an entry that no longer applies is dropped and said, and the others are
    kept. An orbital from code that the session had not changed is not looked
    at: whether it is still there is the code's matter.

    Args:
        session: The session's chains before the code was loaded again.
        code: The chains of the atom the code gives now.

    Returns:
        The new chains, and one sentence for each entry that was dropped.
    """
    result = code.initial()
    in_code = {orb.name for side in SIDES for orb in code.code[side]}
    dropped: list[str] = []
    shared: dict[int, dict[str, Any]] = {}
    for side in SIDES:
        for entry in session.work[side]:
            raw = _entry(session, entry)
            name = raw["name"]
            if entry.base is not None and raw["enabled"] and "settings" not in raw:
                continue
            if entry.base is None and name in in_code:
                dropped.append(
                    f"{name!r} is now in your code: the one added in the studio was dropped."
                )
                continue
            try:
                result = _entry_applied(result, side, raw, shared)
            except _GoneError:
                if name in in_code:
                    # It left this chain only: on the other one it is still the code's.
                    dropped.append(
                        f"{name!r} is no longer in the {side} chain of your code: "
                        "what the studio changed on it there was dropped."
                    )
                else:
                    dropped.append(f"{name!r} is no longer in your code: {_LOST}")
            except _OtherClassError as exc:
                dropped.append(f"{name!r} is a {exc.now} in your code now: {_LOST}")
            except ChatomonError as exc:
                if entry.base is not None:
                    dropped.append(f"what the studio changed on {name!r} was dropped: {exc}")
                else:
                    dropped.append(f"{name!r}, added in the studio, was dropped: {exc}")
    for side in SIDES:
        result = _ordered(result, session, side)
    return result, dropped


def _ordered(result: Chains, session: Chains, side: Side) -> Chains:
    """Give a chain its order after the code was loaded again.

    A chain the studio had reordered keeps the session's order, and what is new
    in the code goes last, as in a preset. Any other chain takes the order of
    the code, and an orbital added in the studio stays after the entry it
    followed, or after the nearest earlier one that is still there.

    Returns:
        The chains with that side in order.
    """
    present = [entry.orbital.name for entry in result.work[side]]
    was = [entry.orbital.name for entry in session.work[side]]
    if any(change.type == "order" and change.side == side for change in session.changes()):
        order = [name for name in was if name in present]
        order += [name for name in present if name not in order]
    else:
        added = {entry.orbital.name for entry in result.work[side] if entry.base is None}
        order = [name for name in present if name not in added]
        anchor: str | None = None
        for name in was:
            if name in added:
                order.insert(order.index(anchor) + 1 if anchor is not None else 0, name)
            if name in present:
                anchor = name
    for position, name in enumerate(order):
        result = result.moved(side, name, position)
    return result
