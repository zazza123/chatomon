"""The screenshots of the studio in the docs: named in both themes, all used, and light."""

import re
from collections import Counter
from pathlib import Path

from tests.studio import shots

DOCS = shots.SHOTS.parents[1]
FILE_RE = re.compile(r"(?P<scene>[\w-]+)-(?P<theme>light|dark)\.webp")
NAMED_RE = re.compile(r"assets/studio/(?P<file>[^\s)\"'#]+)(?P<only>#only-(?:light|dark))?")


def named(docs: Path = DOCS) -> dict[str, list[tuple[str, str]]]:
    """For every page that shows screenshots, each file it names and how it shows it."""
    pages: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(docs.rglob("*.md")):
        found = [
            (match["file"], match["only"] or "")
            for match in NAMED_RE.finditer(path.read_text(encoding="utf-8"))
        ]
        if found:
            pages[path.relative_to(docs).as_posix()] = found
    return pages


def misnamed(pages: dict[str, list[tuple[str, str]]]) -> list[str]:
    """What is wrong in how pages name images: each is named once, in each theme, for that theme."""
    wrong = []
    for page, found in pages.items():
        shown: dict[str, set[str]] = {}
        for (file, only), times in sorted(Counter(found).items()):
            match = FILE_RE.fullmatch(file)
            if match is None or only != f"#only-{match['theme']}":
                wrong.append(f"{page}: {file}{only}")
                continue
            shown.setdefault(match["scene"], set()).add(match["theme"])
            if times > 1:
                wrong.append(f"{page}: {file}{only} is named {times} times")
        wrong += [
            f"{page}: {scene} is shown in one theme only"
            for scene, themes in shown.items()
            if themes != set(shots.THEMES)
        ]
    return wrong


def kept() -> list[Path]:
    """The files of the folder, but for the hidden ones a system leaves, such as `.DS_Store`."""
    return sorted(path for path in shots.SHOTS.iterdir() if not path.name.startswith("."))


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
    assert misnamed(named()) == []


def test_every_screenshot_is_named_by_a_page():
    used = {file for found in named().values() for file, _ in found}
    assert {path.name for path in kept()} - used == set()


def test_the_screenshots_are_light():
    files = kept()
    assert [path.name for path in files if path.suffix != ".webp"] == []
    heavy = {path.name: path.stat().st_size for path in files}
    assert {name: size for name, size in heavy.items() if size > shots.MAX_FILE_BYTES} == {}
    assert sum(heavy.values()) <= shots.MAX_FOLDER_BYTES


def test_a_hidden_file_the_system_leaves_is_not_a_screenshot(tmp_path, monkeypatch):
    (tmp_path / ".DS_Store").write_bytes(b"")
    (tmp_path / "overview-light.webp").write_bytes(b"")
    monkeypatch.setattr(shots, "SHOTS", tmp_path)
    assert [path.name for path in kept()] == ["overview-light.webp"]


def test_an_image_named_twice_in_a_page_is_reported(tmp_path):
    line = "![The page](assets/studio/overview-{0}.webp#only-{0}){{ .cm-shot }}\n"
    (tmp_path / "index.md").write_text(line.format("light") * 2 + line.format("dark"))
    assert misnamed(named(tmp_path)) == [
        "index.md: overview-light.webp#only-light is named 2 times"
    ]
