import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "chatomon" / "studio" / "static"
DOCS_CSS = ROOT / "docs" / "public" / "stylesheets" / "chatomon.css"
BRAND_TOKEN_RE = re.compile(
    r"--cm-(?P<name>ink|paper|night|snow|amber|link|allowed|blocked|warning):\s*(?P<value>#[0-9a-fA-F]{3,8})"
)


def text_files():
    return [path for path in STATIC.rglob("*") if path.suffix in {".html", ".css", ".js", ".svg"}]


MINT_TOKEN_RE = re.compile(r"--cm-(?P<name>mint|mint-deep):\s*(?P<value>#[0-9a-fA-F]{3,8})")


def tokens(css: str, pattern: re.Pattern[str] = BRAND_TOKEN_RE) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for match in pattern.finditer(css):
        found.setdefault(match["name"], set()).add(match["value"].lower())
    return found


def test_the_page_has_its_files():
    for name in [
        "index.html",
        "studio.css",
        "favicon.svg",
        "js/main.js",
        "js/stage.js",
        "js/orbital-card.js",
        "js/blocks.js",
        "js/export-cards.js",
    ]:
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
    assert tokens((STATIC / "studio.css").read_text(encoding="utf-8")) == tokens(
        DOCS_CSS.read_text(encoding="utf-8")
    )


def test_the_mint_of_the_spectral_add_on_matches_the_docs():
    """Spectral's colour is a brand token like the others: one value in the studio and the docs."""
    studio = tokens((STATIC / "spectral.css").read_text(encoding="utf-8"), MINT_TOKEN_RE)
    assert studio == tokens(DOCS_CSS.read_text(encoding="utf-8"), MINT_TOKEN_RE)
    assert studio == {"mint": {"#7df7d6"}, "mint-deep": {"#0a7560", "#7df7d6"}}


def classes(css: str) -> set[str]:
    """The class names in the selectors of a stylesheet."""
    plain = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    selectors = [
        text for text in re.findall(r"([^{}]+)\{", plain) if not text.strip().startswith("@")
    ]
    return {
        name for selector in selectors for name in re.findall(r"\.(-?[_a-zA-Z][\w-]*)", selector)
    }


def test_the_styles_of_the_spectral_add_on_name_only_its_own_classes():
    """Loaded beside the studio's, they must leave every card of the studio as it was."""
    own = classes((STATIC / "spectral.css").read_text(encoding="utf-8"))
    studio = classes((STATIC / "studio.css").read_text(encoding="utf-8"))
    assert own and studio
    assert own & studio == set()


def test_fredoka_and_the_favicon_are_the_docs_copies():
    brand = ROOT / "docs" / "public" / "assets"
    assert (STATIC / "fonts" / "Fredoka.woff2").read_bytes() == (
        brand / "fonts" / "Fredoka.woff2"
    ).read_bytes()
    assert (STATIC / "favicon.svg").read_bytes() == (brand / "brand" / "favicon.svg").read_bytes()


def test_every_font_ships_its_licence():
    fonts = STATIC / "fonts"
    for font in fonts.glob("*.woff2"):
        assert (fonts / f"OFL-{font.stem}.txt").is_file(), font.name


def test_the_class_of_a_block_of_fields_is_written_with_its_element():
    # `block` is also the class of what marks a blocking decision (`.lg.block`,
    # `.trail i.block`, `.res-dash.block`): a rule or a query meant for a block of fields
    # names the fieldset, so it never reaches those marks.
    offenders = [
        f"{path.relative_to(STATIC)}:{number}"
        for path in text_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"(?<![\w.\]-])\.block(?![\w-])", line)
    ]
    assert offenders == []


def test_the_page_says_what_a_link_on_a_network_address_gives():
    # No more than that: there the page cannot say where an orbital connects.
    for name, given in (
        ("js/cards.js", "can read this conversation and use the atom's models."),
        ("js/main.js", "can read this conversation."),
    ):
        text = (STATIC / name).read_text(encoding="utf-8")
        (line,) = [line for line in text.splitlines() if "listens on a network address" in line]
        assert given in line and "address of their choice" not in line, name
