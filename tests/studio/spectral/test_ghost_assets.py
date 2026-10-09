"""The ghost the page draws and the ghost of the brand's files are the same one."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GHOST = ROOT / "src" / "chatomon" / "studio" / "static" / "js" / "spectral" / "ghost.js"
FILES = ROOT / "docs" / "public" / "assets" / "brand" / "mascot" / "spectral"
THEMES = ("light", "dark")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def poses() -> list[str]:
    match = re.search(r"export const GHOST_POSES = \[(?P<names>[^\]]*)\]", GHOST.read_text("utf-8"))
    assert match, "ghost.js must export GHOST_POSES"
    return re.findall(r'"([a-z]+)"', match["names"])


def body() -> str:
    match = re.search(r'export const GHOST_BODY = "(?P<path>[^"]+)"', GHOST.read_text("utf-8"))
    assert match, "ghost.js must export GHOST_BODY"
    return match["path"]


def test_the_poses_are_the_six_of_the_design():
    assert poses() == ["sleep", "connect", "orbit", "evaluate", "lost", "refused"]


def test_every_pose_has_still_and_animated_files_in_both_themes():
    missing = [
        str(path.relative_to(FILES))
        for pose in poses()
        for folder in (FILES, FILES / "animated")
        for theme in THEMES
        if not (path := folder / f"{pose}-{theme}.svg").is_file()
    ]
    assert missing == []


def test_there_is_no_file_of_a_pose_the_page_does_not_have():
    names = {path.stem.rsplit("-", 1)[0] for path in FILES.rglob("*.svg")}
    assert names == set(poses())


def test_every_file_draws_the_shape_the_page_draws():
    files = sorted(FILES.rglob("*.svg"))
    assert len(files) == 24
    for path in files:
        assert f'd="{body()}"' in path.read_text(encoding="utf-8"), path.name


def test_every_file_stands_alone():
    """No font, no stylesheet and no address: a file is shown as it is wherever it is put."""
    files = sorted(FILES.rglob("*.svg"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "<text" not in text and "font" not in text, path.name
        assert re.findall(r"https?://[^\s\"')]+", text) == [SVG_NAMESPACE], path.name


def test_the_animated_files_stop_under_reduced_motion():
    files = sorted((FILES / "animated").glob("*.svg"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "@keyframes" in text, path.name
        assert "prefers-reduced-motion" in text, path.name


def test_the_still_files_do_not_move():
    files = sorted(FILES.glob("*.svg"))
    assert files
    for path in files:
        assert "animation" not in path.read_text(encoding="utf-8"), path.name
