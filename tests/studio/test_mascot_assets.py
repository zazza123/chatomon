import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASCOT = ROOT / "docs" / "public" / "assets" / "brand" / "mascot"
STAGE = ROOT / "src" / "chatomon" / "studio" / "static" / "js" / "stage.js"


def poses() -> list[str]:
    match = re.search(
        r"export const POSES = \[(?P<names>[^\]]*)\]", STAGE.read_text(encoding="utf-8")
    )
    assert match, "stage.js must export POSES"
    return re.findall(r'"([a-z-]+)"', match["names"])


def test_poses_are_the_brand_names():
    assert poses() == [
        "rest",
        "listen",
        "think",
        "review",
        "block",
        "block-output",
        "reply",
        "idle",
    ]


def test_every_pose_has_static_and_animated_assets():
    missing = [
        str(path.relative_to(MASCOT))
        for pose in poses()
        for folder in (MASCOT, MASCOT / "animated")
        for theme in ("light", "dark")
        if not (path := folder / f"{pose}-{theme}.svg").is_file()
    ]
    assert missing == []


def test_the_turn_loop_exists():
    for theme in ("light", "dark"):
        assert (MASCOT / "animated" / f"turn-{theme}.svg").is_file()


def test_animations_stop_under_reduced_motion():
    files = sorted((MASCOT / "animated").glob("*.svg"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "@keyframes" in text, path.name
        assert "prefers-reduced-motion" in text, path.name
