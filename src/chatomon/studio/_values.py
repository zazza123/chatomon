"""The values the studio reads and writes: plain settings, and blocks made of plain fields.

A setting is one parameter of a constructor. A plain setting holds a number, a
text or a choice. A block holds a value that has fields of its own: a dataclass
or a pydantic model the parameter accepts, read one field at a time. Nothing
here names a type: what a parameter accepts is read from its annotation.

The page and the presets see a block as JSON: one text is a string, a value
type is `{"type": "<class name>", "fields": {...}}`, a field of lines is a list
of strings, and a list of entries holds strings and values of the same form.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import sys
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from chatomon.errors import ChatomonConfigError, ChatomonError
from chatomon.knowledge import Knowledge

SettingType = Literal[
    "int", "number", "text", "bool", "choice", "names", "env", "lines", "items", "block"
]
Shape = tuple[SettingType, bool, tuple[str, ...]]

_PLAIN: dict[type, SettingType] = {int: "int", float: "number", str: "text", bool: "bool"}
_EXPECTED: dict[str, str] = {
    "int": "a whole number",
    "number": "a number",
    "text": "text",
    "env": "text",
    "bool": "true or false",
    "names": "a list of orbital names",
    "lines": "a list of texts",
}
# What a constructor may raise on a value it does not accept.
REFUSALS = (ChatomonConfigError, TypeError, ValueError, OverflowError)


class _Opaque:
    def __repr__(self) -> str:
        return "OPAQUE"


OPAQUE: Any = _Opaque()
"""Stands for a value the studio cannot write back: an object from the developer's code."""


class InvalidSettingError(ChatomonError):
    """A value that the studio or a constructor refuses."""

    def __init__(self, message: str, *, setting: str | None = None) -> None:
        super().__init__(message)
        self.setting = setting


@dataclass(frozen=True)
class Setting:
    """One parameter of a constructor that the studio reads and writes."""

    key: str
    type: SettingType
    optional: bool
    required: bool
    default: Any
    options: tuple[str, ...] = ()
    text: bool = False
    """For a block, whether one text is accepted; for a list of entries, whether sentences are."""
    types: tuple[ValueType, ...] = ()
    """For a block or a list of entries, the value types it accepts."""

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
        if self.type in ("block", "items"):
            data["text"] = self.text
            data["types"] = [found.schema() for found in self.types]
        return data


@dataclass(frozen=True)
class ValueType:
    """A dataclass or a pydantic model as the studio reads it."""

    cls: type
    fields: tuple[Setting, ...]
    unread: tuple[tuple[str, Any, bool], ...]
    """The fields the studio does not read: the name, the value it must hold for the
    studio to write the type, and whether that value has to be passed."""

    def field(self, key: str) -> Setting | None:
        """Return the field with that key, if the type has it."""
        return next((found for found in self.fields if found.key == key), None)

    def schema(self) -> dict[str, Any]:
        """Return the type as the page receives it."""
        return {"type": self.cls.__name__, "fields": [found.schema() for found in self.fields]}


def plain(value: Any) -> Any:
    """Return a plain setting's value as JSON data: a knowledge as its text, a tuple as a list."""
    if isinstance(value, Knowledge):
        return value.text
    if isinstance(value, tuple):
        return list(value)
    return value


def hint(owner: Any, annotation: Any) -> Any:
    """Resolve one annotation on its own; one that does not resolve is an object.

    Args:
        owner: The function or the class the annotation belongs to.
        annotation: The annotation, as an object or as text.

    Returns:
        The annotation as an object, or `None` when it cannot be resolved.
    """
    if not isinstance(annotation, str):
        return annotation
    scope = getattr(owner, "__globals__", None)
    if scope is None:
        module = sys.modules.get(getattr(owner, "__module__", ""))
        scope = vars(module) if module is not None else {}
    try:
        # The text is an annotation written in a library's code, never user input.
        return eval(annotation, scope)
    except Exception:
        return None


def members(annotation: Any) -> tuple[list[Any], bool]:
    """Split an annotation into what it admits.

    Returns:
        The members of a union, or the annotation alone, and whether `None` is among them.
    """
    if typing.get_origin(annotation) in (types.UnionType, typing.Union):
        args = typing.get_args(annotation)
        rest = [arg for arg in args if arg is not type(None)]
        return rest, len(rest) != len(args)
    return [annotation], False


def shape(annotation: Any) -> Shape | None:
    """Return the setting type of a plain annotation, or `None` for anything else."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (types.UnionType, typing.Union):
        rest, optional = members(annotation)
        if len(rest) != 1 or not optional:
            return None
        inner = shape(rest[0])
        return None if inner is None else (inner[0], True, inner[2])
    if origin is Literal and args and all(isinstance(arg, str) for arg in args):
        return ("choice", False, tuple(args))
    if origin in (Sequence, list) and args == (str,):
        return ("names", False, ())
    if origin is tuple and args == (str, ...):
        return ("names", False, ())
    kind = _PLAIN.get(annotation) if isinstance(annotation, type) else None
    return None if kind is None else (kind, False, ())


def _entries(annotation: Any) -> list[Any] | None:
    """Return what the entries of a list annotation admit, or `None` when it is not a list."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (Sequence, list) and len(args) == 1:
        return members(args[0])[0]
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        return members(args[0])[0]
    return None


def _field(name: str, annotation: Any, required: bool, default: Any) -> Setting | None:
    """Read one field of a value type.

    Returns:
        The field, or `None` when its annotation is not one the studio reads.
    """
    found = shape(annotation)
    if found is not None:
        kind, optional, options = found
        if kind == "names":
            kind = "lines"
        elif kind == "text" and name.endswith("_env"):
            kind = "env"
        return Setting(name, kind, optional, required, default, options)
    admitted, optional = members(annotation)
    lists = [member for member in admitted if _entries(member) is not None]
    if len(lists) != 1 or any(member is not str for member in admitted if member is not lists[0]):
        return None
    entries = _entries(lists[0]) or []
    if all(entry is str for entry in entries):
        return Setting(name, "lines", optional, required, default)
    found_types = [value_type(entry) for entry in entries if entry is not str]
    known = tuple(found for found in found_types if found is not None)
    if not known or len(known) != len(found_types):
        return None
    if any(inner.type == "items" for found in known for inner in found.fields):
        # The page draws the entries of a list, not the lists inside an entry.
        return None
    return Setting(name, "items", optional, required, default, text=str in entries, types=known)


def _is_plain_default(value: Any) -> bool:
    if isinstance(value, list | tuple):
        return all(isinstance(item, str) for item in value)
    return value is None or isinstance(value, bool | int | float | str)


_reading: list[type] = []


def value_type(cls: Any) -> ValueType | None:
    """Read a dataclass or a pydantic model whose fields are plain.

    Args:
        cls: A member of a parameter's annotation.

    Returns:
        The type as the studio reads it, or `None` when the class is not a
        dataclass or a pydantic model, or has a field the studio can neither
        read nor leave at a default.
    """
    if not (
        isinstance(cls, type) and (dataclasses.is_dataclass(cls) or issubclass(cls, BaseModel))
    ):
        return None
    if cls in _reading:
        # A type that holds itself has no end to read.
        return None
    # Inside another type the reading can be cut where a circle closes, so only a
    # type read from the outside is remembered: the order of the calls changes nothing.
    return _guarded(cls) if _reading else _outermost(cls)


@cache
def _outermost(cls: type) -> ValueType | None:
    return _guarded(cls)


def _guarded(cls: type) -> ValueType | None:
    _reading.append(cls)
    try:
        return _read(cls)
    finally:
        _reading.pop()


def _default(cls: type, parameter: inspect.Parameter) -> Any:
    """Return a parameter's default, also when the type makes it with a factory.

    Returns:
        What an instance holds when the argument is left out; the signature's
        own default when the factory cannot be called without arguments.
    """
    factory: Any = None
    if dataclasses.is_dataclass(cls):
        declared = {found.name: found for found in dataclasses.fields(cls)}.get(parameter.name)
        if declared is not None and declared.default_factory is not dataclasses.MISSING:
            factory = declared.default_factory
    elif issubclass(cls, BaseModel):
        info = cls.model_fields.get(parameter.name)
        factory = info.default_factory if info is not None else None
    if factory is None:
        return parameter.default
    try:
        return factory()
    except Exception:
        return parameter.default


def _read(cls: type) -> ValueType | None:
    try:
        parameters = list(inspect.signature(cls).parameters.values())
    except (TypeError, ValueError):
        return None
    fields: list[Setting] = []
    unread: list[tuple[str, Any, bool]] = []
    for parameter in parameters:
        if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            return None
        required = parameter.default is inspect.Parameter.empty
        annotation = hint(cls, parameter.annotation)
        default = None if required else _default(cls, parameter)
        found = _field(parameter.name, annotation, required, default)
        if found is not None and _is_plain_default(default):
            fields.append(found)
        elif not required:
            unread.append((parameter.name, default, False))
        elif members(annotation)[1]:
            # No default, but none is accepted: the studio gives none.
            unread.append((parameter.name, None, True))
        else:
            return None
    return ValueType(cls, tuple(fields), tuple(unread)) if fields else None


def block(name: str, annotation: Any, required: bool, default: Any = None) -> Setting | None:
    """Read a parameter that accepts a value made of plain fields.

    Args:
        name: The parameter's name.
        annotation: Its annotation, resolved.
        required: Whether it has no default.
        default: Its default, when it has one.

    Returns:
        A setting of type `block`, or `None` when the annotation admits no value
        type. Its default is the JSON form of the parameter's; none when the
        studio could not write that default back, as for a client.
    """
    admitted, optional = members(annotation)
    found = tuple(known for member in admitted if (known := value_type(member)) is not None)
    if not found:
        return None
    setting = Setting(name, "block", optional, required, None, text=str in admitted, types=found)
    shown = represented(setting, default)
    return setting if shown is OPAQUE else dataclasses.replace(setting, default=shown)


def _fields_of(value: Any) -> Mapping[str, Any]:
    """Return the fields a JSON value gives to its type; none when it gives none.

    Returns:
        The mapping under `fields`, or an empty one.
    """
    return value.get("fields", {}) if isinstance(value, dict) else {}


def named(setting: Setting, value: Any) -> ValueType | None:
    """Return the value type a JSON value names, among those a setting accepts."""
    if not isinstance(value, dict) or not set(value) <= {"type", "fields"}:
        return None
    if not isinstance(value.get("type"), str) or not isinstance(value.get("fields", {}), dict):
        return None
    return next((found for found in setting.types if found.cls.__name__ == value["type"]), None)


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


def _is_line(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip() and "\n" not in value


def _fits(setting: Setting, value: Any) -> bool:
    """Say whether a JSON value has the shape a plain setting needs.

    Returns:
        `True` when the value can be given to the constructor as it is.
    """
    number = isinstance(value, int | float) and not isinstance(value, bool)
    if setting.type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if setting.type == "number":
        return number and (isinstance(value, int) or math.isfinite(value))
    if setting.type in ("text", "env"):
        return _is_text(value)
    if setting.type == "bool":
        return isinstance(value, bool)
    if setting.type == "choice":
        return isinstance(value, str) and value in setting.options
    return isinstance(value, list) and all(_is_text(item) for item in value)


def _field_json(field: Setting, raw: Any) -> Any:
    """Return a field's value as JSON data, or `OPAQUE` when the studio could not write it back."""
    if raw is OPAQUE:
        return OPAQUE
    if field.type == "lines":
        if raw is None:
            return []
        if isinstance(raw, str):
            # A text is shown as its lines; untouched, it stays the text it is.
            return [line.strip() for line in raw.split("\n") if line.strip()]
        if isinstance(raw, list | tuple) and all(_is_line(item) for item in raw):
            return list(raw)
        return OPAQUE
    if field.type == "items":
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw] if field.text else OPAQUE
        if not isinstance(raw, list | tuple):
            return OPAQUE
        entries: list[Any] = []
        for entry in raw:
            if isinstance(entry, str):
                got = entry if field.text else OPAQUE
            else:
                found = next((known for known in field.types if type(entry) is known.cls), None)
                got = OPAQUE if found is None else _dumped(found, entry)
            if got is OPAQUE:
                return OPAQUE
            entries.append(got)
        return entries
    if raw is None:
        return None
    return raw if _fits(field, raw) else OPAQUE


def _dumped(found: ValueType, value: Any) -> Any:
    fields: dict[str, Any] = {}
    for field in found.fields:
        got = _field_json(field, getattr(value, field.key, OPAQUE))
        if got is OPAQUE:
            return OPAQUE
        fields[field.key] = got
    for name, expected, _given in found.unread:
        try:
            same = bool(getattr(value, name, OPAQUE) == expected)
        except Exception:
            same = False
        if not same:
            return OPAQUE
    return {"type": found.cls.__name__, "fields": fields}


def represented(setting: Setting, value: Any) -> Any:
    """Return a setting's value as JSON data.

    Args:
        setting: The setting.
        value: What the orbital holds in the attribute named like it.

    Returns:
        The JSON form, with every field of a value type; `OPAQUE` for a block
        whose value the studio could not write back, such as a client.
    """
    if setting.type != "block":
        return plain(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value if setting.text else OPAQUE
    found = next((known for known in setting.types if type(value) is known.cls), None)
    return OPAQUE if found is None else _dumped(found, value)


def compact(setting: Setting, value: Any) -> Any:
    """Leave out of a block's JSON the fields that hold their type's default.

    Args:
        setting: The block, or the list of entries, the value belongs to.
        value: The JSON form of the value.

    Returns:
        The same value with its required fields and those that differ from the default.
    """
    found = named(setting, value)
    if found is None:
        return value
    fields: dict[str, Any] = {}
    for field in found.fields:
        got = _fields_of(value).get(field.key)
        if field.type == "items" and got:
            got = [entry if isinstance(entry, str) else compact(field, entry) for entry in got]
        if field.required or got != _field_json(field, field.default):
            fields[field.key] = got
    return {"type": value["type"], "fields": fields}


def variables(setting: Setting, value: Any) -> list[str]:
    """Return the environment variables a setting's JSON names; for a block, at any depth."""
    if setting.type == "env":
        return [value] if isinstance(value, str) and value else []
    found = named(setting, value)
    if found is None:
        return []
    names: list[str] = []
    for field in found.fields:
        got = _fields_of(value).get(field.key)
        if field.type == "env" and isinstance(got, str) and got:
            names.append(got)
        elif field.type == "items" and isinstance(got, list):
            for entry in got:
                names += variables(field, entry)
    return names


def unnamed(setting: Setting, value: Any) -> bool:
    """Return whether a block's JSON has a field for an environment variable left empty."""
    found = named(setting, value)
    if found is None:
        return False
    for field in found.fields:
        got = _fields_of(value).get(field.key)
        if field.type == "env" and not got:
            return True
        if field.type == "items" and isinstance(got, list) and any(unnamed(field, e) for e in got):
            return True
    return False


def _accepted(setting: Setting) -> str:
    """Say what a block or a list of entries accepts, for a message.

    Returns:
        The words that follow "must be".
    """
    names = [found.cls.__name__ for found in setting.types]
    if setting.type == "items":
        entries = " or ".join(names)
        return f"a list of texts and {entries} entries" if setting.text else f"a list of {entries}"
    parts = ["text"] if setting.text else []
    parts += [f"{'an' if name[:1] in 'AEIOU' else 'a'} {name}" for name in names]
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} or {parts[-1]}"


def _check_fields(found: ValueType, fields: Mapping[str, Any], owner: str, *, entry: bool) -> None:
    kind = found.cls.__name__
    for name in fields:
        if found.field(name) is None:
            raise InvalidSettingError(f"`{kind}` has no field `{name}`", setting=owner)
    for field in found.fields:
        value = fields.get(field.key)
        empty = value is None or (isinstance(value, str) and not value.strip())
        listed = field.type in ("lines", "items")
        if (field.required and (empty or (listed and not value))) or (
            field.key in fields and empty and not (field.optional or listed)
        ):
            asked = f"every {kind} needs `{field.key}`" if entry else f"`{field.key}` needs a value"
            raise InvalidSettingError(asked, setting=owner)
        if value is None:
            continue
        if field.type == "items":
            _check_items(field, value, owner)
        elif not _fits(field, value):
            expected = _EXPECTED.get(field.type) or f"one of: {', '.join(field.options)}"
            raise InvalidSettingError(f"`{field.key}` must be {expected}", setting=owner)
        elif field.type == "lines" and not all(_is_line(item) for item in value):
            # The page writes these one per line: an entry it could not show again
            # would turn the whole block into an object the studio cannot edit.
            raise InvalidSettingError(
                f"every entry of `{field.key}` must be one line of text, with no space around it",
                setting=owner,
            )


def _check_items(field: Setting, value: Any, owner: str) -> None:
    refusal = InvalidSettingError(f"`{field.key}` must be {_accepted(field)}", setting=owner)
    if not isinstance(value, list):
        raise refusal
    for entry in value:
        if isinstance(entry, str):
            if not (field.text and _is_text(entry)):
                raise refusal
            continue
        found = named(field, entry)
        if found is None:
            raise refusal
        _check_fields(found, _fields_of(entry), owner, entry=True)


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
    if setting.type != "block":
        if _fits(setting, value):
            return
        expected = _EXPECTED.get(setting.type) or f"one of: {', '.join(setting.options)}"
        raise InvalidSettingError(f"`{key}` must be {expected}", setting=key)
    if isinstance(value, str) and setting.text and _is_text(value):
        if value.strip():
            return
        raise InvalidSettingError(f"`{key}` needs a value", setting=key)
    found = named(setting, value)
    if found is None:
        raise InvalidSettingError(f"`{key}` must be {_accepted(setting)}", setting=key)
    _check_fields(found, _fields_of(value), key, entry=False)


def _field_value(field: Setting, incoming: Any) -> Any:
    """Turn a field's JSON into what the type's constructor receives.

    Returns:
        The value: an empty list of an optional field is none.
    """
    if field.type == "lines":
        return None if not incoming and field.optional else list(incoming or [])
    if field.type == "items":
        entries = [
            entry
            if isinstance(entry, str)
            else _instance(_known(field, entry), _fields_of(entry), None)
            for entry in incoming or []
        ]
        return None if not entries and field.optional else entries
    return incoming


def _known(setting: Setting, value: Any) -> ValueType:
    found = named(setting, value)
    if found is None:
        raise InvalidSettingError(f"`{setting.key}` must be {_accepted(setting)}")
    return found


def _instance(found: ValueType, fields: Mapping[str, Any], current: Any) -> Any:
    arguments: dict[str, Any] = {}
    for field in found.fields:
        held = getattr(current, field.key, OPAQUE) if current is not None else OPAQUE
        if field.key not in fields:
            # Left out, as in a preset: the type's own default, or the value held when it
            # reads the same, such as an empty list where the default is none.
            if held is not OPAQUE and _field_json(field, held) == _field_json(field, field.default):
                arguments[field.key] = held
            continue
        incoming = fields[field.key]
        if held is not OPAQUE and _field_json(field, held) == incoming:
            # Untouched: a text shown as lines stays the text it is.
            arguments[field.key] = held
        else:
            arguments[field.key] = _field_value(field, incoming)
    for name, value, given in found.unread:
        if given:
            arguments[name] = value
    try:
        return found.cls(**arguments)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise InvalidSettingError(f"{where}: {first['msg']}") from exc
    except REFUSALS as exc:
        raise InvalidSettingError(str(exc)) from exc


def built(setting: Setting, value: Any, current: Any = None) -> Any:
    """Turn a block's JSON value into what the orbital's constructor receives.

    The shape is checked before. A field whose JSON equals that of the value
    the orbital holds keeps that value.

    Args:
        setting: The block.
        value: Its JSON form: `None`, one text, or a type and its fields.
        current: What the orbital holds now, when there is an orbital.

    Returns:
        `None`, the text, or an instance of the value type.

    Raises:
        InvalidSettingError: With the message of the type's constructor.
    """  # noqa: DOC502 - raised by the helpers
    if value is None or isinstance(value, str):
        return value
    found = _known(setting, value)
    held = current if type(current) is found.cls else None
    return _instance(found, _fields_of(value), held)
