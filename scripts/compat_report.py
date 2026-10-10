"""Write the report of a failed compatibility run, as the body of an issue.

The weekly run of ``.github/workflows/compat.yml`` upgrades the lockfile in the
runner and runs the suite on the newest versions the ranges of ``pyproject.toml``
allow. When it fails, this script reads what ``uv lock --upgrade`` and ``pytest``
printed and writes which packages are newer than the committed lockfile and which
tests failed (CONTRIBUTING.md, "The weekly compatibility run").

The report is a list of names: no text of a failure is copied, the log has it.

Input: the files with the output of ``uv lock --upgrade`` and of ``pytest``, as
arguments, either of which may be missing; the ``RUN_URL`` environment variable.
Output: the report, in Markdown. Exit code: 0.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path

MAX_TESTS = 30
"""How many failed tests the report lists before it counts the rest."""

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
"""Where the dependencies chatomon declares are named."""

_UPDATED_RE = re.compile(r"^Updated (?P<name>\S+) v(?P<old>\S+) -> v(?P<new>\S+)$", re.MULTILINE)
# A test's name can hold spaces, between the brackets of its parameters.
_FAILED_RE = re.compile(r"^(?:FAILED|ERROR) (?P<test>.+?)(?: - .*)?$", re.MULTILINE)
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _alike(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def direct_names(pyproject: str) -> frozenset[str]:
    """Return the packages a ``pyproject.toml`` names, in every table of dependencies.

    Args:
        pyproject: The content of the file.

    Returns:
        The names, written alike: lower case, with ``-`` between their parts.
    """
    data = tomllib.loads(pyproject)
    project = data.get("project", {})
    tables: list[Iterable[object]] = [
        project.get("dependencies", []),
        *project.get("optional-dependencies", {}).values(),
        *data.get("dependency-groups", {}).values(),
    ]
    names: set[str] = set()
    for requirement in (entry for table in tables for entry in table):
        found = _NAME_RE.match(requirement) if isinstance(requirement, str) else None
        if found:
            names.add(_alike(found.group()))
    return frozenset(names)


def _newer(upgraded: str, direct: frozenset[str]) -> list[str]:
    updates = list(_UPDATED_RE.finditer(upgraded))
    if not updates:
        return ["No package is newer than the lockfile: the failure is not a new release."]
    named = [u for u in updates if _alike(u["name"]) in direct]
    others = len(updates) - len(named)
    lines = [f"- {u['name']} {u['old']} → {u['new']}" for u in named]
    if others:
        what = "package" if others == 1 else "packages"
        lines.append(
            f"- and {others} {what} that they depend on"
            if named
            else f"- {others} {what} that the ones `pyproject.toml` names depend on"
        )
    return lines


def report(run_url: str, upgraded: str, pytest_log: str, direct: frozenset[str]) -> str:
    """Return the body of the issue for a failed run.

    Args:
        run_url: The address of the workflow run.
        upgraded: What ``uv lock --upgrade`` printed.
        pytest_log: What ``pytest`` printed.
        direct: The packages ``pyproject.toml`` names, as `direct_names` returns
            them: they are listed, the others are counted.

    Returns:
        The report in Markdown, ending with a line break.
    """
    newer = _newer(upgraded, direct)
    tests = [match["test"] for match in _FAILED_RE.finditer(pytest_log)]
    lines = [
        f"The weekly compatibility run failed: {run_url}",
        "",
        "It runs the suite on the newest versions the ranges of `pyproject.toml` allow. "
        'What to do is in CONTRIBUTING.md, "The weekly compatibility run".',
        "",
        "### Newer than the lockfile",
        "",
        *newer,
        "",
        "### Failed",
        "",
    ]
    if tests:
        lines += [f"- `{test}`" for test in tests[:MAX_TESTS]]
        if len(tests) > MAX_TESTS:
            lines.append(f"- and {len(tests) - MAX_TESTS} more")
    else:
        lines.append(
            "The run did not reach the summary of the tests: its log says where it stopped."
        )
    return "\n".join(lines) + "\n"


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def main(argv: Sequence[str]) -> int:
    """Print the report for the two files given as arguments.

    Args:
        argv: The file with the output of ``uv lock --upgrade`` and the one with
            the output of ``pytest``.

    Returns:
        The exit code.
    """
    upgraded, pytest_log = (_read(path) for path in argv[:2])
    direct = direct_names(_read(str(PYPROJECT)))
    sys.stdout.write(report(os.environ.get("RUN_URL", ""), upgraded, pytest_log, direct))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
