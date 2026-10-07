"""What the studio can edit in an orbital, read from the orbital's constructor.

Nothing here lists the orbitals or their settings. The modules of
`chatomon.orbitals` are walked for the classes they export, and each class is
read one parameter at a time: a keyword parameter with a plain annotation is a
setting, one that accepts a value made of plain fields is a block (see
`_values.py`), any other parameter is an object that comes from the developer's
code, and every value is read back from the public attribute named like its
parameter. The exceptions to that convention are the only hand-written part.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import chatomon.orbitals as library
from chatomon.connectors.base import Connector
from chatomon.errors import ChatomonConfigError, MissingExtraError
from chatomon.knowledge import Knowledge
from chatomon.orbitals.base import FunctionOrbital, Orbital
from chatomon.orbitals.judge import JudgeOrbital
from chatomon.studio import _values
from chatomon.studio._values import (
    OPAQUE,
    REFUSALS,
    InvalidSettingError,
    Setting,
    Shape,
    check_shape,
    represented,
)

FUNCTION = "Written with @orbital: change it in your code."
UNREADABLE = "The studio cannot rebuild this orbital from its settings: change it in your code."

_NAME = re.compile(r"[a-z0-9_.\-]+")

# Settings whose annotation alone would make them objects.
_SETTINGS: dict[tuple[type, str], Shape] = {(JudgeOrbital, "context"): ("text", True, ())}


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
        annotation = _values.hint(init, parameter.annotation)
        shape = _SETTINGS.get((cls, parameter.name)) or _values.shape(annotation)
        if shape is not None:
            default = None if required else parameter.default
            settings.append(
                Setting(parameter.name, shape[0], shape[1], required, default, shape[2])
            )
            continue
        block = _values.block(parameter.name, annotation, required)
        if block is not None:
            settings.append(block)
            continue
        objects.append(parameter.name)
        addable = addable and not required
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


@dataclass(frozen=True)
class _Library:
    """What walking the modules of `chatomon.orbitals` found."""

    classes: tuple[type[Orbital], ...]
    extras: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]


@cache
def _library() -> _Library:
    """Import the modules of `chatomon.orbitals` and take the orbital classes they export.

    Returns:
        The classes, those of `chatomon.orbitals` first; the modules that need an
        extra, each with it; and a note for every other module that did not import.
    """
    names = [library.__name__]
    names += sorted(
        f"{library.__name__}.{module.name}"
        for module in pkgutil.iter_modules(library.__path__)
        if not module.name.startswith("_")
    )
    classes: list[type[Orbital]] = []
    extras: list[tuple[str, str]] = []
    notes: list[str] = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except MissingExtraError as exc:
            extras.append((name, exc.extra))
            continue
        except Exception as exc:
            notes.append(f"{name} could not be imported: {type(exc).__name__}: {exc}")
            continue
        for exported in getattr(module, "__all__", ()):
            value = getattr(module, exported, None)
            if isinstance(value, type) and issubclass(value, Orbital) and value not in classes:
                classes.append(value)
    return _Library(tuple(classes), tuple(extras), tuple(notes))


def classes() -> tuple[type[Orbital], ...]:
    """Return every orbital class the modules of `chatomon.orbitals` export."""
    return _library().classes


def addable() -> tuple[Kind, ...]:
    """Return the exported classes whose required parameters are all settings or blocks."""
    return tuple(describe(cls) for cls in classes() if describe(cls).addable)


def extras() -> tuple[tuple[str, str], ...]:
    """Return the modules of `chatomon.orbitals` that need an extra, each with its extra."""
    return _library().extras


def import_notes() -> tuple[str, ...]:
    """Return one sentence for each module of `chatomon.orbitals` that failed to import."""
    return _library().notes


def unknown(kind: str) -> str:
    """Say that a class cannot be added, naming the extras that would bring more classes.

    Args:
        kind: The class name that was asked for.

    Returns:
        The refusal. The studio does not know which module an unknown class is in.
    """
    missing = extras()
    if not missing:
        return f"{kind} cannot be added from the studio: add it in your code"
    hints = " ".join(
        f"The orbitals of {module} need: pip install 'chatomon[{extra}]'."
        for module, extra in missing
    )
    return f"{kind} is not an orbital this studio can add. {hints}"


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


# Parameters of built-in classes that are not kept in an attribute of the same name.
_READERS: dict[tuple[str, str], Callable[[Any, Connector[Any]], Any]] = {
    ("JudgeOrbital", "connector"): _judge_connector,
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
    """Return the orbital's settings as JSON data, in constructor order.

    A block whose value the studio could not write back, such as a client, is
    left out: for this orbital it is an object from the developer's code.
    """
    found: dict[str, Any] = {}
    for setting in describe(type(orb)).settings:
        value = represented(setting, getattr(orb, setting.key))
        if value is not OPAQUE:
            found[setting.key] = value
    return found


def opaque(orb: Orbital) -> tuple[str, ...]:
    """Return the blocks of an orbital that hold an object from the developer's code."""
    return tuple(
        setting.key
        for setting in describe(type(orb)).settings
        if represented(setting, getattr(orb, setting.key)) is OPAQUE
    )


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


def _setting(kind: Kind, key: str) -> Setting:
    if key in kind.objects:
        raise InvalidSettingError(
            f"`{key}` is an object from your code: change it there", setting=key
        )
    setting = kind.setting(key)
    if setting is None:
        raise InvalidSettingError(f"`{kind.cls.__name__}` has no setting `{key}`", setting=key)
    return setting


def _written(orb: Orbital, setting: Setting, value: Any, base: Orbital | None) -> Any:
    """Turn a setting's new value into the constructor argument.

    A knowledge the developer shared is the same object again whenever the text
    is the one it has, now or in the code, and keeps its title when the text
    changes. A block that reads like the one the orbital has, now or in the
    code, is that object again.

    Returns:
        The value to pass to the constructor.

    Raises:
        InvalidSettingError: If the constructor of a block's type refuses it.
    """  # noqa: DOC502 - raised by built()
    key = setting.key
    held = [getattr(source, key, None) for source in (orb, base) if source is not None]
    if setting.type == "block":
        new = _values.built(setting, value, getattr(orb, key, None))
        wanted = represented(setting, new)
        # The developer's object first: back to the code's values is back to the code's object.
        return next((old for old in reversed(held) if represented(setting, old) == wanted), new)
    if not isinstance(value, str):
        return value
    shared = [knowledge for knowledge in held if isinstance(knowledge, Knowledge)]
    for knowledge in shared:
        if value == knowledge.text:
            return knowledge
    title = next((knowledge.title for knowledge in shared if knowledge.title is not None), None)
    return Knowledge(value, title=title) if title is not None else value


def rebuilt(
    orb: Orbital,
    changes: Mapping[str, Any],
    connector: Connector[Any],
    *,
    base: Orbital | None = None,
) -> Orbital:
    """Build a new orbital like `orb` with some settings changed.

    Args:
        orb: The orbital to start from; it is not changed.
        changes: Setting keys and their new values; `name` renames it.
        connector: The atom's connector.
        base: The developer's instance, when `orb` is an edited copy of it.

    Returns:
        The new orbital.

    Raises:
        InvalidSettingError: If a value has the wrong shape or a constructor
            refuses it; the message is the constructor's own in the second case.
    """
    kind = describe(type(orb))
    found: dict[str, Setting] = {}
    for key, value in changes.items():
        if key == "name":
            check_name(value)
            continue
        setting = _setting(kind, key)
        if setting.type == "block" and key in opaque(orb):
            raise InvalidSettingError(
                f"`{key}` is an object from your code: change it there", setting=key
            )
        check_shape(setting, value)
        found[key] = setting
    only = next(iter(changes)) if len(changes) == 1 else None
    try:
        args = arguments(orb, connector)
        for key, value in changes.items():
            args[key] = value if key == "name" else _written(orb, found[key], value, base)
        return type(orb)(**args)
    except InvalidSettingError as exc:
        raise InvalidSettingError(str(exc), setting=only) from exc
    except REFUSALS as exc:
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
        InvalidSettingError: If a value is missing, has the wrong shape, or a
            constructor refuses it.
    """
    check_name(name)
    found: dict[str, Setting] = {}
    for key, value in settings.items():
        found[key] = _setting(kind, key)
        check_shape(found[key], value)
    for setting in kind.settings:
        if setting.required and settings.get(setting.key) is None:
            raise InvalidSettingError(f"`{setting.key}` needs a value", setting=setting.key)
    made: dict[str, Any] = {}
    for key, value in settings.items():
        if found[key].type != "block":
            made[key] = value
            continue
        try:
            made[key] = _values.built(found[key], value)
        except InvalidSettingError as exc:
            # The constructor of the block's type refused it: the refusal is the block's.
            raise InvalidSettingError(str(exc), setting=key) from exc
    try:
        return kind.cls(name=name, **made)
    except REFUSALS as exc:
        raise InvalidSettingError(str(exc)) from exc
