"""The docs site: every page is in the nav and every public module has a reference page."""

import importlib
import pkgutil
import re
import tomllib
from pathlib import Path

import chatomon

ROOT = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))["project"]
DOCS = ROOT / CONFIG["docs_dir"]
DIRECTIVE_RE = re.compile(r"^::: (?P<module>[\w.]+)\s*$", re.MULTILINE)


def nav_pages(entries):
    for entry in entries:
        for target in entry.values():
            if isinstance(target, list):
                yield from nav_pages(target)
            elif not target.startswith(("http://", "https://")):
                yield target


def is_public(name):
    return not any(part.startswith("_") for part in name.split("."))


def public_modules():
    for module in pkgutil.walk_packages(chatomon.__path__, "chatomon."):
        if not module.ispkg and is_public(module.name):
            yield module.name


def public_packages():
    yield "chatomon"
    for module in pkgutil.walk_packages(chatomon.__path__, "chatomon."):
        if module.ispkg and is_public(module.name):
            yield module.name


def test_every_nav_entry_exists():
    missing = [page for page in nav_pages(CONFIG["nav"]) if not (DOCS / page).is_file()]
    assert missing == []


def test_every_page_is_in_the_nav():
    pages = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.md")}
    assert pages - set(nav_pages(CONFIG["nav"])) == set()


def test_every_public_module_has_one_reference_page():
    directives = [
        match["module"]
        for path in (DOCS / "reference").glob("*.md")
        for match in DIRECTIVE_RE.finditer(path.read_text(encoding="utf-8"))
    ]
    assert sorted(directives) == sorted(public_modules())


def test_every_package_export_is_listed_in_the_reference_index():
    index = (DOCS / "reference" / "index.md").read_text(encoding="utf-8")
    section = index.split("\n## Exports\n", 1)[1].split("\n## ", 1)[0]
    exports = {
        f"{package}.{name}"
        for package in public_packages()
        for name in importlib.import_module(package).__all__
        if name != "__version__"
    }
    assert {export for export in exports if f"`{export.rsplit('.', 1)[1]}`" not in section} == set()
