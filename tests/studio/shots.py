"""Screenshots of the studio for the documentation: preparing a page and writing an image.

A scene is a test that brings the page to a state and calls `Camera.shoot`. The
images are written under pytest's temporary directory, and under `SHOTS` only when
`CHATOMON_SCREENSHOTS=1`: see "Studio screenshots" in `CONTRIBUTING.md`.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from playwright.sync_api import Locator, Page

Theme = Literal["light", "dark"]

SHOTS = Path(__file__).resolve().parents[2] / "docs" / "public" / "assets" / "studio"
THEMES: tuple[Theme, ...] = ("light", "dark")
VIEWPORT = {"width": 1120, "height": 940}
SCALE = 1.5
QUALITY = 75
MARGIN = 12
ADDRESS = "127.0.0.1:8016"

# A deliberate ceiling, not a default: the images are kept in the repository and
# sent to every reader of the documentation. Raising a limit is a decision to
# take with the maintainer, not the way to make a test pass.
MAX_FILE_BYTES = 150_000
MAX_FOLDER_BYTES = 1_000_000

# What changes at every run is made constant, so that a scene whose screen did not
# change writes the same file again. The address shows a port the system chose. A
# duration is rounded to a tenth of a second, and one under 50 ms, which is how
# long a check that calls nothing takes, is shown as 0.2 ms.
_STEADY = """
([address]) => {
  const shown = document.querySelector("#addr");
  if (shown) shown.textContent = address;
  const duration = /\\b(\\d+(?:\\.\\d+)?) (ms|s)\\b/g;
  const steady = (text) => text.replace(duration, (_, value, unit) => {
    const ms = Number(value) * (unit === "s" ? 1000 : 1);
    if (ms < 50) return "0.2 ms";
    const tenths = Math.round(ms / 100);
    return tenths < 10 ? `${tenths * 100} ms` : `${(tenths / 10).toFixed(1)} s`;
  });
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = steady(node.nodeValue);
    if (text !== node.nodeValue) node.nodeValue = text;
  }
}
"""
_REWORD = """
([text, shown]) => {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeValue.includes(text)) node.nodeValue = node.nodeValue.replaceAll(text, shown);
  }
  for (const field of document.querySelectorAll("input, textarea")) {
    if (field.value.includes(text)) field.value = field.value.replaceAll(text, shown);
  }
}
"""
_SHOWN = """
() => {
  const fields = [...document.querySelectorAll("input, textarea")].map((field) => field.value);
  return [document.body.innerText, ...fields].join("\\n");
}
"""
_NO_CARET = "*, *::before, *::after { caret-color: transparent !important; }"


def writing() -> bool:
    """Whether this run writes the images of the documentation."""
    return os.environ.get("CHATOMON_SCREENSHOTS") == "1"


def reword(page: Page, text: str, shown: str) -> None:
    """Show `shown` wherever the page shows `text`, in its texts and in its fields.

    For what a scene cannot make constant otherwise: the address of a fake service,
    the temporary directory of a test. The page is not told, so nothing is sent.

    Args:
        page: The page of the scene.
        text: What the page shows.
        shown: What the image shows in its place.
    """
    page.evaluate(_REWORD, [text, shown])


@dataclass
class Camera:
    """Takes the image of a scene, in the theme of its page."""

    page: Page
    theme: Theme
    folder: Path
    private: tuple[str, ...] = ()
    """Texts no image may show: the paths that say whose machine took it."""

    def shoot(self, scene: str, target: Locator | None = None) -> Path:
        """Write `<scene>-<theme>.webp`: the whole page, or `target` with a margin.

        Args:
            scene: The name of the scene, as the pages of the documentation name it.
            target: The element to frame; the whole page when `None`.

        Returns:
            The file written.
        """
        self.page.add_style_tag(content=_NO_CARET)
        self.page.evaluate("document.fonts.ready")
        self.page.evaluate(_STEADY, [ADDRESS])
        shown = self.page.evaluate(_SHOWN)
        told = [text for text in self.private if text in shown]
        assert not told, (
            f"{scene}: the page shows {told}, a path of this machine: reword it before the image"
        )
        size = self.page.viewport_size
        assert size is not None
        clip = {"x": 0, "y": 0, **size}
        if target is not None:
            box = target.bounding_box()
            assert box is not None, f"{scene}: the element to frame is not on the page"
            left, top = max(box["x"] - MARGIN, 0), max(box["y"] - MARGIN, 0)
            right = min(box["x"] + box["width"] + MARGIN, size["width"])
            bottom = min(box["y"] + box["height"] + MARGIN, size["height"])
            clip = {"x": left, "y": top, "width": right - left, "height": bottom - top}
        session = self.page.context.new_cdp_session(self.page)
        shot = session.send(
            "Page.captureScreenshot",
            {"format": "webp", "quality": QUALITY, "clip": {**clip, "scale": SCALE}},
        )
        session.detach()
        image = base64.b64decode(shot["data"])
        assert len(image) <= MAX_FILE_BYTES, (
            f"{scene}-{self.theme}.webp weighs {len(image)} bytes, over the {MAX_FILE_BYTES} "
            "an image of the documentation may weigh: frame less, or show less"
        )
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{scene}-{self.theme}.webp"
        path.write_bytes(image)
        return path
