"""Presets: a session's orbital changes as a JSON layer over the atom from code.

A preset never holds an atom. It says which orbitals are on, their order, the
settings the studio changed on the orbitals from code, and the orbitals added
in the studio. Applying it starts from the atom's own chains, so the code stays
the source of truth for whatever the preset does not mention.
"""

from __future__ import annotations

import json
from typing import Any

from chatomon._version import __version__
from chatomon.errors import ChatomonError
from chatomon.schemas import Side
from chatomon.studio import _kinds as kinds
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


def _entry(chains: Chains, entry: Entry) -> dict[str, Any]:
    orb = entry.orbital
    data: dict[str, Any] = {
        "name": orb.name,
        "kind": type(orb).__name__,
        "from": entry.origin,
        "enabled": entry.enabled,
    }
    if chains.note(entry) is None:
        now = kinds.values(orb)
        # An orbital from code carries what differs from the code; an added one, everything.
        before = kinds.values(entry.base) if entry.base is not None else None
        settings = {k: v for k, v in now.items() if before is None or v != before[k]}
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
        RefusedError: With the reason the entry cannot be applied.
    """
    name = raw["name"]
    try:
        where, index = result.find(side, name)
        base = result.work[where][index].base
    except UnknownOrbitalError:
        base = None
    if base is None:
        raise RefusedError(
            "it is not in your code, so a preset cannot bring it back: add it to your code first"
        )
    if type(base).__name__ != raw["kind"]:
        raise RefusedError(
            f"it is a {type(base).__name__} in your code, a {raw['kind']} in the preset"
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


def applied(text: str, chains: Chains, *, atom: str) -> tuple[Chains, list[str]]:
    """Apply a preset to the atom's own chains.

    Args:
        text: The preset, as JSON text.
        chains: The session's chains; only the atom's own are used.
        atom: The loaded atom's name.

    Returns:
        The new chains and the warnings: orbitals that differ from the exported
        ones, and orbitals from code the preset does not mention.

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
                if raw["from"] == "code":
                    result = _from_code(result, side, raw, settings, shared)
                else:
                    result = result.added(side, raw["kind"], {**settings, "name": name})
                if not raw["enabled"]:
                    result = result.switched(side, name, False)
            except ChatomonError as exc:
                problems.append(f"{name!r}: {exc}")
                continue
            wanted[side].append(name)
            expected.append((side, name, raw["fingerprint"]))
    if problems:
        raise InvalidPresetError(problems)
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
    return result, warnings
