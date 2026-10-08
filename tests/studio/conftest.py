"""Fixtures of the studio's tests."""

from __future__ import annotations

import itertools
import os
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.studio import frameworks


@pytest.fixture(params=frameworks.NAMES)
def framework(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> frameworks.StudioHarness:
    """Run the test once for every framework the studio opens."""
    return frameworks.harness(request.param, monkeypatch)


@pytest.fixture(autouse=True)
def the_suite_stays_imported() -> Iterator[None]:
    """Fail a test that leaves the suite's own modules forgotten.

    A reload forgets the modules under the directory its session was started in.
    Started in the repository, that is the suite itself: a test that runs later
    would import a second copy of a module other tests already hold.
    """
    before = {name: module for name, module in sys.modules.items() if name.split(".")[0] == "tests"}
    yield
    lost = sorted(name for name, module in before.items() if sys.modules.get(name) is not module)
    sys.modules.update(before)
    assert not lost, f"this test left modules of the suite forgotten or replaced: {lost}"


def _forget_modules_under(directory: Path) -> None:
    """Take out of `sys.modules` what was imported from a directory."""
    directories = {Path(os.path.abspath(directory)), directory.resolve()}
    for name, module in list(sys.modules.items()):
        places = [getattr(module, "__file__", None), *getattr(module, "__path__", ())]
        # As it was imported and with its links followed: a folder of a project can be a link.
        forms = [form for place in places if place for form in (Path(place), Path(place).resolve())]
        if any(base in form.parents for form in forms for base in directories):
            del sys.modules[name]


@pytest.fixture
def sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[str, str], None]]:
    """A temporary project as the current directory; `sources(path, text)` writes one of its files.

    A file can be written again, as a developer edits it. Every write gets a later
    modification time: Python reads a source that was rewritten within the same
    second, with the same length, from the bytecode of the version before.
    """
    # A module another test imported from its own temporary project would be found first.
    _forget_modules_under(tmp_path.parent)
    monkeypatch.chdir(tmp_path)
    # As the command does: the project is imported from the current directory.
    monkeypatch.setattr(sys, "path", [os.getcwd(), *sys.path])
    start = time.time()
    seconds = itertools.count(1)

    def write(path: str, text: str) -> None:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(textwrap.dedent(text), encoding="utf-8")
        stamp = start + next(seconds)
        os.utime(file, (stamp, stamp))

    yield write
    _forget_modules_under(tmp_path)
