"""Find the atom the studio opens: `module:attribute`, an instance or a factory.

The studio can also load it again while it runs:
[reimporting][chatomon.studio._loading.reimporting] makes the modules of the
developer's project be imported anew.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from chatomon.atom import Atom
from chatomon.errors import ChatomonConfigError

_INSTALLED = {"site-packages", "dist-packages"}
_OWN = Path(__file__).resolve().parents[1]
"""chatomon's own package: importing it again would break the studio that is running."""


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def project_modules(root: Path) -> dict[str, ModuleType]:
    """Return the imported modules that belong to the project in a directory.

    They are read from `sys.modules` every time: nothing is listed or remembered.
    A module belongs to the project when its file is under `root`. Left out: what
    is installed (an environment inside the directory, a `site-packages` folder),
    chatomon's own modules, and `__main__`.

    Args:
        root: The directory the studio was started from.

    Returns:
        The modules by name.
    """
    root = root.resolve()
    prefixes = {Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    # Only an environment inside the project hides files of it: with a system
    # interpreter in /usr, a project in /usr/src/app is still the project.
    environments = [prefix for prefix in prefixes if root in prefix.parents]
    found: dict[str, ModuleType] = {}
    for name, module in list(sys.modules.items()):
        if name == "__main__" or not isinstance(module, ModuleType):
            continue
        # A package without an `__init__.py` has no file: it is where its folders are.
        file = getattr(module, "__file__", None)
        places = [file] if file else list(getattr(module, "__path__", ()))
        for place in places:
            # A library can give a module it builds a bare file name: it is not on disk here.
            if not isinstance(place, str) or not os.path.isabs(place):
                continue
            path = Path(place).resolve()
            if (
                root in path.parents
                and not _INSTALLED.intersection(path.relative_to(root).parts)
                and not any(_inside(path, environment) for environment in environments)
                and not _inside(path, _OWN)
            ):
                found[name] = module
                break
    return found


@contextmanager
def reimporting(root: Path) -> Iterator[None]:
    """Make the project's modules be imported again inside the block, or not at all.

    On entry the modules of the project leave `sys.modules`, so an import inside
    the block runs their code again and sees what changed in any of them. If the
    block raises, what it imported is removed and the modules that were there come
    back, the same objects: the code that was running is still what is imported.

    Args:
        root: The directory the studio was started from.

    Yields:
        Nothing: the block imports what it needs, usually with
        [load][chatomon.studio._loading.load].
    """
    removed = project_modules(root)
    for name in removed:
        del sys.modules[name]
    # A file created since the last import is not in the finders' caches.
    importlib.invalidate_caches()
    try:
        yield
    except BaseException:
        for name in project_modules(root):
            del sys.modules[name]
        sys.modules.update(removed)
        raise


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
