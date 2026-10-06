"""Find the atom the studio opens: `module:attribute`, an instance or a factory."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import Any

from chatomon.atom import Atom
from chatomon.errors import ChatomonConfigError


def load(spec: str) -> Atom[Any]:
    """Import `module:attribute` from the current directory and return its atom.

    Args:
        spec: `module:attribute`; the attribute may be dotted and may be a factory.

    Returns:
        The atom.

    Raises:
        ChatomonConfigError: If the spec, the module, the attribute or the value is wrong.
    """
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ChatomonConfigError(f"expected <module>:<attribute>, got {spec!r}") from None
    cwd = os.getcwd()
    if not sys.path or sys.path[0] != cwd:
        sys.path.insert(0, cwd)
    try:
        value: Any = importlib.import_module(module_name)
    except Exception as exc:
        raise ChatomonConfigError(f"could not import {module_name!r}: {exc}") from exc
    for part in attribute.split("."):
        try:
            value = getattr(value, part)
        except AttributeError:
            raise ChatomonConfigError(f"{module_name!r} has no attribute {attribute!r}") from None
    return resolve(value, label=spec)


def namespace(spec: str) -> dict[str, object]:
    """Return the variables of the module a spec names, where Copy as code looks names up.

    Call it after [load][chatomon.studio._loading.load], which imports the module.

    Args:
        spec: `module:attribute`.

    Returns:
        The module's namespace; empty when the module is not imported.
    """
    module = sys.modules.get(spec.partition(":")[0])
    return vars(module) if module is not None else {}


def _required(function: Any) -> list[str]:
    """Name the arguments a callable cannot be called without.

    Returns:
        The names, empty when it can be called with no arguments or its
        signature cannot be read.
    """
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return []
    plain = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return [p.name for p in parameters if p.default is p.empty and p.kind not in plain]


def resolve(value: object, *, label: str) -> Atom[Any]:
    """Return `value` if it is an atom, or call it and return the atom it builds.

    Args:
        value: An `Atom` or a function with no arguments that returns one.
        label: How the value is named in error messages.

    Returns:
        The atom.

    Raises:
        ChatomonConfigError: If the value is neither, or the factory fails.
    """
    if isinstance(value, Atom):
        return value
    if not callable(value):
        raise ChatomonConfigError(
            f"{label} is a {type(value).__name__}, not an Atom or a function that returns one"
        )
    needed = _required(value)
    if needed:
        # Calling it would only fail with a traceback about missing arguments.
        raise ChatomonConfigError(
            f"{label} needs arguments ({', '.join(needed)}): the studio calls it with none. "
            "Point at an Atom, or at a function with no arguments that returns one"
        )
    try:
        result = value()
    except Exception as exc:
        raise ChatomonConfigError(f"{label}() raised {type(exc).__name__}: {exc}") from exc
    if not isinstance(result, Atom):
        raise ChatomonConfigError(f"{label}() returned a {type(result).__name__}, not an Atom")
    return result
