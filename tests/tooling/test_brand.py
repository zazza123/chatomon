"""The brand assets: every file the site and the README point to exists and is a valid SVG."""

import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))["project"]
DOCS = ROOT / CONFIG["docs_dir"]
BRAND = DOCS / "assets" / "brand"
SVG_NS = "{http://www.w3.org/2000/svg}"
ASSET_RE = re.compile(r"'(?P<path>assets/[^']+)'")


def test_theme_logo_and_favicon_exist():
    theme = CONFIG["theme"]
    assert (DOCS / theme["logo"]).is_file()
    assert (DOCS / theme["favicon"]).is_file()


def test_theme_overrides_point_to_existing_assets():
    overrides = ROOT / CONFIG["theme"]["custom_dir"]
    assets = [
        match["path"]
        for path in overrides.rglob("*.html")
        for match in ASSET_RE.finditer(path.read_text(encoding="utf-8"))
    ]
    assert assets
    assert [asset for asset in assets if not (DOCS / asset).is_file()] == []


def test_every_extra_stylesheet_exists():
    missing = [path for path in CONFIG.get("extra_css", []) if not (DOCS / path).is_file()]
    assert CONFIG.get("extra_css")
    assert missing == []


def brand_svgs():
    return sorted(BRAND.rglob("*.svg"))


def test_brand_has_svgs():
    assert brand_svgs()


@pytest.mark.parametrize("path", brand_svgs(), ids=lambda path: path.relative_to(BRAND).as_posix())
def test_brand_svg_is_well_formed(path):
    root = ET.parse(path).getroot()
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("viewBox")
    assert root.find(f"{SVG_NS}title") is not None


def test_self_hosted_fonts_ship_their_license():
    fonts = DOCS / "assets" / "fonts"
    assert list(fonts.glob("*.woff2"))
    assert (fonts / "OFL.txt").is_file()
