"""The orbitals of a studio session: the atom's own chains and the working ones.

Pure data and rules, with no conversation and no event loop. `Chains` is never
changed in place: every operation returns a changed copy, so the session can try
it, keep the previous one for undo, and drop it if the atom does not build.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from chatomon.atom import Atom
from chatomon.connectors.base import Connector
from chatomon.errors import ChatomonError
from chatomon.orbitals.base import Orbital
from chatomon.schemas import Side
from chatomon.studio import _kinds as kinds
from chatomon.studio._kinds import InvalidSettingError

SIDES: tuple[Side, ...] = ("input", "output")


class UnknownOrbitalError(ChatomonError):
    """The atom has no orbital with that side and name."""


class RefusedError(ChatomonError):
    """The orbital, or the state of the session, does not allow the operation."""


class InvalidRequestError(ChatomonError):
    """The request asks for something that cannot exist, such as a position past the end."""


@dataclass(frozen=True)
class Entry:
    """One orbital of a working chain."""

    orbital: Orbital
    enabled: bool = True
    base: Orbital | None = None
    """The developer's instance; `None` for an orbital added in the studio."""

    @property
    def origin(self) -> str:
        """`"code"` for an orbital the atom defines, `"studio"` for an added one."""
        return "code" if self.base is not None else "studio"


@dataclass(frozen=True)
class Change:
    """One difference between the session and the code."""

    type: str
    side: Side
    text: str
    name: str | None = None
    setting: str | None = None

    def to_json(self) -> dict[str, str]:
        """Return the change as the page receives it."""
        data = {"type": self.type, "side": self.side, "text": self.text}
        if self.name is not None:
            data["name"] = self.name
        if self.setting is not None:
            data["setting"] = self.setting
        return data


def _initial(code: Mapping[Side, tuple[Orbital, ...]]) -> dict[Side, tuple[Entry, ...]]:
    return {side: tuple(Entry(orb, True, orb) for orb in code[side]) for side in SIDES}


@dataclass(frozen=True)
class Chains:
    """The atom's chains and the session's working entries."""

    code: Mapping[Side, tuple[Orbital, ...]]
    work: Mapping[Side, tuple[Entry, ...]]
    connector: Connector[Any]
    notes: Mapping[int, str]
    """Why an orbital from code is read-only, by the id of the developer's instance."""

    @classmethod
    def from_atom(cls, atom: Atom[Any]) -> Chains:
        """Start from the atom's chains, every orbital on.

        Args:
            atom: The developer's atom; it is read, never changed.

        Returns:
            The chains as the code has them.
        """
        code: dict[Side, tuple[Orbital, ...]] = {
            "input": atom.input_orbitals,
            "output": atom.output_orbitals,
        }
        notes: dict[int, str] = {}
        for orb in (*atom.input_orbitals, *atom.output_orbitals):
            note = kinds.read_only(orb, atom.connector)
            if note is not None:
                notes[id(orb)] = note
        return cls(code, _initial(code), atom.connector, notes)

    def initial(self) -> Chains:
        """Return the chains as the code has them."""
        return replace(self, work=_initial(self.code))

    def enabled(self, side: Side) -> list[Orbital]:
        """Return the orbitals of a side that are on, in order."""
        return [entry.orbital for entry in self.work[side] if entry.enabled]

    def note(self, entry: Entry) -> str | None:
        """Say why an entry is read-only.

        Returns:
            The sentence the card shows, or `None` when the entry can be edited.
        """
        return None if entry.base is None else self.notes.get(id(entry.base))

    def modified(self, entry: Entry) -> bool:
        """Say whether an entry differs from the code.

        Returns:
            `True` when it was added, is off, or is no longer the developer's instance.
        """
        return entry.base is None or not entry.enabled or entry.orbital is not entry.base

    def find(self, side: str, name: str) -> tuple[Side, int]:
        """Return the side and the position of an orbital.

        Args:
            side: `"input"` or `"output"`.
            name: The orbital's name.

        Returns:
            The side and the 0-based position in its chain.

        Raises:
            UnknownOrbitalError: If there is no such orbital.
        """
        if side in SIDES:
            where = cast("Side", side)
            for index, entry in enumerate(self.work[where]):
                if entry.orbital.name == name:
                    return where, index
        raise UnknownOrbitalError(f"the atom has no {side} orbital named {name!r}")

    def free_name(self, cls: type[Orbital]) -> str:
        """Return the first name a new orbital of that class can take."""
        taken = {entry.orbital.name for side in SIDES for entry in self.work[side]}
        base = cls.default_name()
        name, number = base, 2
        while name in taken:
            name, number = f"{base}_{number}", number + 1
        return name

    def _require_free(self, name: str, *, besides: Orbital | None = None) -> None:
        for side in SIDES:
            for entry in self.work[side]:
                if entry.orbital.name == name and entry.orbital is not besides:
                    raise InvalidSettingError(
                        f"an orbital named {name!r} is already in the atom", setting="name"
                    )

    def _swapped(self, old: Orbital, new: Orbital) -> Chains:
        """Replace an instance wherever it is, so one shared by both chains stays shared.

        Returns:
            The chains with the new instance.
        """
        work = {
            side: tuple(
                replace(entry, orbital=new) if entry.orbital is old else entry
                for entry in self.work[side]
            )
            for side in SIDES
        }
        return replace(self, work=work)

    def _placed(self, side: Side, entries: tuple[Entry, ...]) -> Chains:
        return replace(self, work={**self.work, side: entries})

    def switched(self, side: str, name: str, enabled: bool) -> Chains:
        """Turn an orbital on or off.

        Returns:
            The chains with the orbital switched.

        Raises:
            UnknownOrbitalError: If there is no such orbital.
        """  # noqa: DOC502 - raised by find()
        where, index = self.find(side, name)
        entries = list(self.work[where])
        entries[index] = replace(entries[index], enabled=enabled)
        return self._placed(where, tuple(entries))

    def edited(self, side: str, name: str, settings: Mapping[str, Any]) -> Chains:
        """Change some settings of an orbital, or the name of an added one.

        Returns:
            The chains with a new instance in place of the orbital.

        Raises:
            UnknownOrbitalError: If there is no such orbital.
            RefusedError: If the orbital is read-only, or `name` is changed on an
                orbital from code.
            InvalidSettingError: If a value is refused.
        """  # noqa: DOC502 - UnknownOrbitalError and InvalidSettingError come from the helpers
        where, index = self.find(side, name)
        entry = self.work[where][index]
        note = self.note(entry)
        if note is not None:
            raise RefusedError(f"{name!r} is read-only in the studio. {note}")
        if "name" in settings:
            if entry.base is not None:
                raise RefusedError(f"{name!r} is defined in your code: rename it there")
            self._require_free(kinds.check_name(settings["name"]), besides=entry.orbital)
        new = kinds.rebuilt(entry.orbital, settings, self.connector)
        if entry.base is not None and kinds.values(new) == kinds.values(entry.base):
            new = entry.base
        return self._swapped(entry.orbital, new)

    def moved(self, side: str, name: str, position: int) -> Chains:
        """Move an orbital to a 0-based position in its chain.

        Returns:
            The chains with the orbital moved.

        Raises:
            UnknownOrbitalError: If there is no such orbital.
            InvalidRequestError: If the position is outside the chain.
        """  # noqa: DOC502 - UnknownOrbitalError is raised by find()
        where, index = self.find(side, name)
        entries = list(self.work[where])
        if not 0 <= position < len(entries):
            raise InvalidRequestError(f"position must be between 0 and {len(entries) - 1}")
        entries.insert(position, entries.pop(index))
        return self._placed(where, tuple(entries))

    def added(self, side: str, kind: str, settings: Mapping[str, Any]) -> Chains:
        """Add an orbital of a built-in class at the end of a chain, on.

        Args:
            side: `"input"` or `"output"`.
            kind: The class name.
            settings: Its settings; `name` is optional.

        Returns:
            The chains with the new orbital.

        Raises:
            InvalidRequestError: If the side does not exist.
            RefusedError: If the class cannot be added, or not on that side.
            InvalidSettingError: If the name or a value is refused.
        """  # noqa: DOC502 - InvalidSettingError comes from the helpers
        if side not in SIDES:
            raise InvalidRequestError("side must be 'input' or 'output'")
        where = cast("Side", side)
        found = next((k for k in kinds.addable() if k.cls.__name__ == kind), None)
        if found is None:
            raise RefusedError(f"{kind} cannot be added from the studio: add it in your code")
        if where not in found.cls.sides:
            raise RefusedError(f"{kind} cannot run on the {where} side")
        values = dict(settings)
        name = values.pop("name", None)
        if name is None:
            name = self.free_name(found.cls)
        self._require_free(kinds.check_name(name))
        new = kinds.created(found, name, values)
        return self._placed(where, (*self.work[where], Entry(new)))

    def deleted(self, side: str, name: str) -> Chains:
        """Delete an orbital that was added in the studio.

        Returns:
            The chains without the orbital.

        Raises:
            UnknownOrbitalError: If there is no such orbital.
            RefusedError: If the orbital is defined in the code.
        """  # noqa: DOC502 - UnknownOrbitalError is raised by find()
        where, index = self.find(side, name)
        entries = self.work[where]
        if entries[index].base is not None:
            raise RefusedError(f"{name!r} is defined in your code: turn it off instead")
        return self._placed(where, entries[:index] + entries[index + 1 :])

    def reset(self, side: str | None = None, name: str | None = None) -> Chains:
        """Go back to the code: one orbital with `side` and `name`, otherwise all.

        Returns:
            The chains with the developer's instance back, and on.

        Raises:
            UnknownOrbitalError: If there is no such orbital.
            RefusedError: If the orbital was added in the studio.
        """  # noqa: DOC502 - UnknownOrbitalError is raised by find()
        if side is None or name is None:
            return self.initial()
        where, index = self.find(side, name)
        entry = self.work[where][index]
        if entry.base is None:
            raise RefusedError(f"{name!r} was added in the studio: delete it instead")
        chains = self._swapped(entry.orbital, entry.base)
        entries = list(chains.work[where])
        entries[index] = replace(entries[index], enabled=True)
        return chains._placed(where, tuple(entries))

    def changes(self) -> list[Change]:
        """List how the session differs from the code, in chain order.

        Returns:
            One change per difference; empty when the session is the code.
        """
        found: list[Change] = []
        told: set[int] = set()
        for side in SIDES:
            for entry in self.work[side]:
                name = entry.orbital.name
                if entry.base is None:
                    found.append(Change("added", side, f"{name} added", name))
                    continue
                if not entry.enabled:
                    found.append(Change("enabled", side, f"{name} turned off", name))
                if entry.orbital is not entry.base and id(entry.orbital) not in told:
                    # An instance in both chains is one orbital: its settings are listed once.
                    told.add(id(entry.orbital))
                    now, before = kinds.values(entry.orbital), kinds.values(entry.base)
                    found.extend(
                        Change("setting", side, f"{name}.{key} changed", name, key)
                        for key in now
                        if now[key] != before[key]
                    )
            kept = [id(entry.base) for entry in self.work[side] if entry.base is not None]
            if kept != [id(orb) for orb in self.code[side]]:
                found.append(Change("order", side, f"{side} order changed"))
        return found
