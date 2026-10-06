"""Write the files the studio saves: inside its directory, never over someone else's.

The page drives the studio, so nothing it sends is trusted here: a file is
written only under the directory the studio was started from and only as a
Python file, and an existing file is replaced only when the studio wrote it.
"""

from __future__ import annotations

from pathlib import Path

from chatomon.errors import ChatomonError
from chatomon.studio._chains import RefusedError

FOLDER = "tests/studio"
"""Where a saved test goes unless the developer says otherwise."""


class InvalidPathError(ChatomonError):
    """The path is not one the studio writes to."""


class OwnFileExistsError(ChatomonError):
    """The file exists and the studio wrote it: it is replaced only when asked."""


def proposed(root: Path, name: str) -> str:
    """Return the first free path for a test called `name`.

    Args:
        root: The directory the studio was started from.
        name: The test's name, without the extension.

    Returns:
        A path relative to the root, with `/` between its parts.
    """
    number = 1
    while True:
        candidate = f"{FOLDER}/{name}.py" if number == 1 else f"{FOLDER}/{name}_{number}.py"
        if not (root / candidate).exists():
            return candidate
        number += 1


def _target(root: Path, path: str) -> Path:
    if not path.strip() or chr(0) in path:
        raise InvalidPathError(f"give a path inside {root}, such as {FOLDER}/test_turn.py")
    given = Path(path.strip())
    if given.is_absolute() or given.drive:
        raise InvalidPathError(f"give a path relative to {root}")
    # Resolved, so that `..` and symbolic links cannot lead outside the root.
    target = (root / given).resolve()
    if not target.is_relative_to(root):
        raise InvalidPathError(f"the file must be inside {root}")
    if target.suffix != ".py":
        raise InvalidPathError("the file must end in .py")
    if target.is_dir():
        raise InvalidPathError(f"{target.relative_to(root).as_posix()} is a directory")
    return target


def _written_by_the_studio(target: Path, header: str) -> bool:
    try:
        with target.open(encoding="utf-8") as file:
            return file.read(len(header)) == header
    except (OSError, UnicodeDecodeError):
        return False


def write(root: Path, path: str, text: str, *, header: str, overwrite: bool) -> str:
    """Write a file the studio saves.

    Args:
        root: The directory the studio was started from.
        path: Where to write, relative to the root.
        text: The file's content, which starts with `header`.
        header: How every file the studio writes starts.
        overwrite: Whether to replace a file the studio wrote earlier.

    Returns:
        The written path, relative to the root, with `/` between its parts.

    Raises:
        InvalidPathError: If the path is absolute, leaves the root, is a directory
            or does not end in `.py`.
        OwnFileExistsError: If the file exists, the studio wrote it and
            `overwrite` is false.
        RefusedError: If the file exists and the studio did not write it, or the
            system refuses the write.
    """  # noqa: DOC502 - InvalidPathError is raised by _target()
    root = root.resolve()
    target = _target(root, path)
    shown = target.relative_to(root).as_posix()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive: a file created since the check is never replaced by accident.
            with target.open("x", encoding="utf-8", newline="\n") as file:
                file.write(text)
            return shown
        except FileExistsError:
            pass
        if not _written_by_the_studio(target, header):
            raise RefusedError(f"{shown} was not written by the studio: choose another path")
        if not overwrite:
            raise OwnFileExistsError(f"{shown} exists and was saved by the studio: overwrite it?")
        target.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise RefusedError(f"could not write {shown}: {exc.strerror or exc}") from exc
    return shown
