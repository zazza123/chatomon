"""The brand assets: every file the site and the README point to exists and is a valid SVG."""

import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))["project"]
DOCS = ROOT / CONFIG["docs_dir"]
BRAND = DOCS / "assets" / "brand"
SVG_NS = "{http://www.w3.org/2000/svg}"


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
