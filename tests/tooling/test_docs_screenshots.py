"""The screenshots of the studio in the docs: named in both themes, all used, and light."""

import re

from tests.studio import shots

DOCS = shots.SHOTS.parents[1]
FILE_RE = re.compile(r"(?P<scene>[\w-]+)-(?P<theme>light|dark)\.webp")
NAMED_RE = re.compile(r"assets/studio/(?P<file>[^\s)\"'#]+)(?P<only>#only-(?:light|dark))?")


def named() -> dict[str, set[tuple[str, str]]]:
    """For every page that shows screenshots, the files it names and how it shows each."""
    pages: dict[str, set[tuple[str, str]]] = {}
    for path in sorted(DOCS.rglob("*.md")):
        found = {
            (match["file"], match["only"] or "")
            for match in NAMED_RE.finditer(path.read_text(encoding="utf-8"))
        }
        if found:
            pages[path.relative_to(DOCS).as_posix()] = found
    return pages


def test_the_documentation_shows_screenshots():
    assert named(), "no page names a screenshot: has the folder moved?"


def test_every_screenshot_a_page_names_exists():
    missing = sorted(
        f"{page}: {file}"
        for page, found in named().items()
        for file, _ in found
        if not (shots.SHOTS / file).is_file()
    )
    assert missing == []


def test_a_screenshot_is_shown_once_in_each_theme():
    wrong = []
    for page, found in named().items():
        shown: dict[str, set[str]] = {}
        for file, only in sorted(found):
            match = FILE_RE.fullmatch(file)
            if match is None or only != f"#only-{match['theme']}":
                wrong.append(f"{page}: {file}{only}")
            else:
                shown.setdefault(match["scene"], set()).add(match["theme"])
        wrong += [
            f"{page}: {scene} is shown in one theme only"
            for scene, themes in shown.items()
            if themes != set(shots.THEMES)
        ]
    assert wrong == []


def test_every_screenshot_is_named_by_a_page():
    used = {file for found in named().values() for file, _ in found}
    kept = {path.name for path in shots.SHOTS.iterdir()}
    assert kept - used == set()


def test_the_screenshots_are_light():
    files = sorted(shots.SHOTS.iterdir())
    assert [path.name for path in files if path.suffix != ".webp"] == []
    heavy = {path.name: path.stat().st_size for path in files}
    assert {name: size for name, size in heavy.items() if size > shots.MAX_FILE_BYTES} == {}
    assert sum(heavy.values()) <= shots.MAX_FOLDER_BYTES
