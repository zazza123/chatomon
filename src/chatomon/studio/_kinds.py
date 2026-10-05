"""What the studio can edit in an orbital, read from the orbital's constructor.

Nothing here lists the orbitals or their settings. The classes exported by
`chatomon.orbitals` are read one parameter at a time: a keyword parameter with
a plain annotation is a setting, any other parameter is an object that comes
from the developer's code, and every value is read back from the public
attribute named like its parameter. The exceptions to that convention are the
only hand-written part.
"""

from __future__ import annotations

import inspect
import math
import re
import sys
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

import chatomon.orbitals as library
from chatomon.connectors.base import Connector
from chatomon.errors import ChatomonConfigError, ChatomonError
from chatomon.knowledge import Knowledge
from chatomon.orbitals.base import FunctionOrbital, Orbital
from chatomon.orbitals.judge import JudgeOrbital

SettingType = Literal["int", "number", "text", "bool", "choice", "names"]
Shape = tuple[SettingType, bool, tuple[str, ...]]

FUNCTION = "Written with @orbital: change it in your code."
UNREADABLE = "The studio cannot rebuild this orbital from its settings: change it in your code."

_NAME = re.compile(r"[a-z0-9_.\-]+")
_PLAIN: dict[type, SettingType] = {int: "int", float: "number", str: "text", bool: "bool"}
_EXPECTED: dict[SettingType, str] = {
    "int": "a whole number",
    "number": "a number",
    "text": "text",
    "bool": "true or false",
    "names": "a list of orbital names",
}
# What a constructor may raise on a value it does not accept.
_REFUSALS = (ChatomonConfigError, TypeError, ValueError, OverflowError)

# Settings whose annotation alone would make them objects.
_SETTINGS: dict[tuple[type, str], Shape] = {(JudgeOrbital, "context"): ("text", True, ())}


class InvalidSettingError(ChatomonError):
    """A value that the studio or the orbital's constructor refuses."""

    def __init__(self, message: str, *, setting: str | None = None) -> None:
        super().__init__(message)
        self.setting = setting


@dataclass(frozen=True)
class Setting:
    """One plain constructor parameter of an orbital class."""

    key: str
    type: SettingType
    optional: bool
    required: bool
    default: Any
    options: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        """Return the setting as the page receives it."""
        data: dict[str, Any] = {
            "key": self.key,
            "type": self.type,
            "optional": self.optional,
            "required": self.required,
            "default": plain(self.default),
        }
        if self.options:
            data["options"] = list(self.options)
        return data


@dataclass(frozen=True)
class Kind:
    """An orbital class as the studio reads it."""

    cls: type[Orbital]
    settings: tuple[Setting, ...]
    objects: tuple[str, ...]
    order: tuple[str, ...]
    readable: bool
    addable: bool

    def setting(self, key: str) -> Setting | None:
        """Return the setting with that key, if the class has it."""
        return next((setting for setting in self.settings if setting.key == key), None)


def plain(value: Any) -> Any:
    """Return a setting's value as JSON data: a knowledge as its text, a tuple as a list."""
    if isinstance(value, Knowledge):
        return value.text
    if isinstance(value, tuple):
        return list(value)
    return value


def _hint(function: Any, annotation: Any) -> Any:
    """Resolve one annotation on its own; one that does not resolve is an object.

    Returns:
        The annotation as an object, or `None` when it cannot be resolved.
    """
    if not isinstance(annotation, str):
        return annotation
    try:
        # The text is an annotation of the library's own constructor, never user input.
        return eval(annotation, getattr(function, "__globals__", {}))
    except Exception:
        return None


def _shape(hint: Any) -> Shape | None:
    """Return the setting type of a plain annotation, or `None` for an object."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (types.UnionType, typing.Union):
        rest = [arg for arg in args if arg is not type(None)]
        if len(rest) != 1 or len(rest) == len(args):
            return None
        inner = _shape(rest[0])
        return None if inner is None else (inner[0], True, inner[2])
    if origin is Literal and args and all(isinstance(arg, str) for arg in args):
        return ("choice", False, tuple(args))
    if origin in (Sequence, list) and args == (str,):
        return ("names", False, ())
    if origin is tuple and args == (str, ...):
        return ("names", False, ())
    kind = _PLAIN.get(hint) if isinstance(hint, type) else None
    return None if kind is None else (kind, False, ())


@cache
def describe(cls: type[Orbital]) -> Kind:
    """Read an orbital class's settings and objects from its constructor.

    Args:
        cls: The orbital class.

    Returns:
        The class as the studio reads it.
    """
    init = cls.__init__
    parameters = list(inspect.signature(init).parameters.values())[1:]
    readable = all(p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in parameters)
    addable = readable and not inspect.isabstract(cls)
    settings: list[Setting] = []
    objects: list[str] = []
    for parameter in parameters:
        if parameter.name == "name":
            continue
        required = parameter.default is inspect.Parameter.empty
        shape = _SETTINGS.get((cls, parameter.name)) or _shape(_hint(init, parameter.annotation))
        if shape is None:
            objects.append(parameter.name)
            addable = addable and not required
            continue
        default = None if required else parameter.default
        settings.append(Setting(parameter.name, shape[0], shape[1], required, default, shape[2]))
    order = tuple(parameter.name for parameter in parameters)
    return Kind(cls, tuple(settings), tuple(objects), order, readable, addable)


def is_builtin(cls: type) -> bool:
    """Say whether a class is one of chatomon's own exported orbitals.

    Returns:
        `True` for a class a `chatomon.orbitals` module exports.
    """
    module = sys.modules.get(cls.__module__)
    return cls.__module__.startswith("chatomon.orbitals.") and cls.__name__ in getattr(
        module, "__all__", ()
    )


@cache
def addable() -> tuple[Kind, ...]:
    """Return the exported classes whose required parameters are all settings."""
    found: list[Kind] = []
    for name in library.__all__:
        value = getattr(library, name)
        if isinstance(value, type) and issubclass(value, Orbital) and describe(value).addable:
            found.append(describe(value))
    return tuple(found)


def summary(cls: type) -> str:
    """Return the first line of a class's docstring, the sentence the picker shows."""
    doc = inspect.getdoc(cls) or ""
    return doc.splitlines()[0] if doc else ""


def _judge_connector(orb: Any, connector: Connector[Any]) -> Any:
    try:
        own = orb.connector
    except ChatomonConfigError:
        return None
    return None if own is connector else own


def _scope_mapping(orb: Any, connector: Connector[Any]) -> Any:
    defaults = sys.modules[type(orb).__module__].DEFAULT_SCOPE_RULES
    return None if orb.rules == dict(defaults) else orb.rules


# Parameters of built-in classes that are not kept in an attribute of the same name.
_READERS: dict[tuple[str, str], Callable[[Any, Connector[Any]], Any]] = {
    ("JudgeOrbital", "connector"): _judge_connector,
    ("ScopeGuardOrbital", "mapping"): _scope_mapping,
}


def read(orb: Orbital, key: str, connector: Connector[Any]) -> Any:
    """Return the constructor argument that gives `orb` its `key`.

    Args:
        orb: The orbital.
        key: A constructor parameter.
        connector: The atom's connector.

    Returns:
        The value, read from the attribute named like the parameter.
    """
    reader = _READERS.get((type(orb).__name__, key))
    return reader(orb, connector) if reader is not None else getattr(orb, key)


def arguments(orb: Orbital, connector: Connector[Any]) -> dict[str, Any]:
    """Return the constructor arguments that rebuild `orb`.

    Args:
        orb: The orbital.
        connector: The atom's connector.

    Returns:
        One value per constructor parameter.
    """
    return {key: read(orb, key, connector) for key in describe(type(orb)).order}


def values(orb: Orbital) -> dict[str, Any]:
    """Return the orbital's settings as JSON data, in constructor order."""
    return {s.key: plain(getattr(orb, s.key)) for s in describe(type(orb)).settings}


def read_only(orb: Orbital, connector: Connector[Any]) -> str | None:
    """Say why the studio cannot edit an orbital, or `None` when it can.

    The studio edits an orbital only when rebuilding it from what it reads gives
    the same fingerprint.

    Args:
        orb: The orbital.
        connector: The atom's connector.

    Returns:
        The sentence the card shows, or `None`.
    """
    cls = type(orb)
    if cls is FunctionOrbital:
        return FUNCTION
    if not is_builtin(cls):
        return f"{cls.__name__} is your own class: change its settings in your code."
    try:
        twin = cls(**arguments(orb, connector))
        same = describe(cls).readable and twin.fingerprint() == orb.fingerprint()
    except Exception:
        same = False
    return None if same else UNREADABLE


def check_name(name: Any) -> str:
    """Check a name set in the studio.

    Args:
        name: The name.

    Returns:
        The name.

    Raises:
        InvalidSettingError: If it is not lowercase letters, digits, `_`, `-` or `.`.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise InvalidSettingError(
            f"invalid name {name!r}: use lowercase letters, digits, '_', '-' or '.'",
            setting="name",
        )
    return name


def _is_text(value: Any) -> bool:
    """Say whether a value is text the studio can send back as UTF-8.

    Returns:
        `False` for anything that is not a string, or that cannot be encoded.
    """
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate from a JSON escape would break every later response.
        return False
    return True


def check_shape(setting: Setting, value: Any) -> None:
    """Check that a JSON value has the shape its setting needs.

    Args:
        setting: The setting.
        value: The value from the request or the preset.

    Raises:
        InvalidSettingError: With the studio's own message.
    """
    key = setting.key
    if value is None:
        if setting.optional:
            return
        raise InvalidSettingError(f"`{key}` needs a value", setting=key)
    number = isinstance(value, int | float) and not isinstance(value, bool)
    if setting.type == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif setting.type == "number":
        ok = number and (isinstance(value, int) or math.isfinite(value))
    elif setting.type == "text":
        ok = _is_text(value)
    elif setting.type == "bool":
        ok = isinstance(value, bool)
    elif setting.type == "choice":
        ok = isinstance(value, str) and value in setting.options
    else:
        ok = isinstance(value, list) and all(_is_text(item) for item in value)
    if ok:
        return
    expected = _EXPECTED.get(setting.type) or f"one of: {', '.join(setting.options)}"
    raise InvalidSettingError(f"`{key}` must be {expected}", setting=key)


def _setting(kind: Kind, key: str) -> Setting:
    if key in kind.objects:
        raise InvalidSettingError(
            f"`{key}` is an object from your code: change it there", setting=key
        )
    setting = kind.setting(key)
    if setting is None:
        raise InvalidSettingError(f"`{kind.cls.__name__}` has no setting `{key}`", setting=key)
    return setting


def _written(orb: Orbital, key: str, value: Any) -> Any:
    """Turn a setting's new value into the constructor argument.

    A knowledge the developer shared stays the same object while its text is
    unchanged, and keeps its title when the text changes.

    Returns:
        The value to pass to the constructor.
    """
    current = getattr(orb, key, None)
    if isinstance(current, Knowledge) and isinstance(value, str):
        if value == current.text:
            return current
        if current.title is not None:
            return Knowledge(value, title=current.title)
    return value


def rebuilt(orb: Orbital, changes: Mapping[str, Any], connector: Connector[Any]) -> Orbital:
    """Build a new orbital like `orb` with some settings changed.

    Args:
        orb: The orbital to start from; it is not changed.
        changes: Setting keys and their new values; `name` renames it.
        connector: The atom's connector.

    Returns:
        The new orbital.

    Raises:
        InvalidSettingError: If a value has the wrong shape or the constructor
            refuses it; the message is the constructor's own in the second case.
    """
    kind = describe(type(orb))
    for key, value in changes.items():
        if key == "name":
            check_name(value)
        else:
            check_shape(_setting(kind, key), value)
    only = next(iter(changes)) if len(changes) == 1 else None
    try:
        args = arguments(orb, connector)
        for key, value in changes.items():
            args[key] = value if key == "name" else _written(orb, key, value)
        return type(orb)(**args)
    except _REFUSALS as exc:
        raise InvalidSettingError(str(exc), setting=only) from exc


def created(kind: Kind, name: str, settings: Mapping[str, Any]) -> Orbital:
    """Build a new orbital of a class that can be added.

    Args:
        kind: The class, from [addable][chatomon.studio._kinds.addable].
        name: The orbital's name.
        settings: Setting keys and values; required ones must be there.

    Returns:
        The new orbital.

    Raises:
        InvalidSettingError: If a value is missing, has the wrong shape, or the
            constructor refuses it.
    """
    check_name(name)
    for key, value in settings.items():
        check_shape(_setting(kind, key), value)
    for setting in kind.settings:
        if setting.required and settings.get(setting.key) is None:
            raise InvalidSettingError(f"`{setting.key}` needs a value", setting=setting.key)
    try:
        return kind.cls(name=name, **settings)
    except _REFUSALS as exc:
        raise InvalidSettingError(str(exc)) from exc
