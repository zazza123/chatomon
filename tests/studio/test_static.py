import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "chatomon" / "studio" / "static"
DOCS_CSS = ROOT / "docs" / "public" / "stylesheets" / "chatomon.css"
BRAND_TOKEN_RE = re.compile(r"--cm-(?P<name>ink|paper|night|snow|amber|link|allowed|blocked|warning):\s*(?P<value>#[0-9a-fA-F]{3,8})")


def text_files():
    return [path for path in STATIC.rglob("*") if path.suffix in {".html", ".css", ".js", ".svg"}]


def tokens(css: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for match in BRAND_TOKEN_RE.finditer(css):
        found.setdefault(match["name"], set()).add(match["value"].lower())
    return found


def test_the_page_has_its_files():
    for name in ["index.html", "studio.css", "favicon.svg", "js/main.js", "js/stage.js"]:
        assert (STATIC / name).is_file(), name


def test_no_external_url():
    offenders = [
        f"{path.relative_to(STATIC)}: {match}"
        for path in text_files()
        for match in re.findall(r"https?://[^\s\"')]+", path.read_text(encoding="utf-8"))
        if not match.startswith("http://www.w3.org/")
    ]
    assert offenders == []


def test_brand_tokens_match_the_docs():
    assert tokens((STATIC / "studio.css").read_text(encoding="utf-8")) == tokens(DOCS_CSS.read_text(encoding="utf-8"))


def test_fredoka_and_the_favicon_are_the_docs_copies():
    brand = ROOT / "docs" / "public" / "assets"
    assert (STATIC / "fonts" / "Fredoka.woff2").read_bytes() == (brand / "fonts" / "Fredoka.woff2").read_bytes()
    assert (STATIC / "favicon.svg").read_bytes() == (brand / "brand" / "favicon.svg").read_bytes()


def test_every_font_ships_its_licence():
    fonts = STATIC / "fonts"
    for font in fonts.glob("*.woff2"):
        assert (fonts / f"OFL-{font.stem}.txt").is_file(), font.name
