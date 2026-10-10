"""The screenshots of the studio that the documentation shows, one scene per image.

Every scene runs with the browser tests, so a change to the page that breaks one
is seen at once. With `CHATOMON_SCREENSHOTS=1` the images are written where the
pages read them: see "Studio screenshots" in `CONTRIBUTING.md`.
"""

from pathlib import Path

import pytest
from playwright.sync_api import expect

from tests.studio import shots
from tests.studio.shots import Camera, reword

pytestmark = pytest.mark.e2e


def test_what_changes_at_every_run_is_shown_constant(camera: Camera, tmp_path, monkeypatch):
    monkeypatch.setattr(camera, "folder", tmp_path)
    camera.page.set_content(
        '<span id="addr">127.0.0.1:50123</span>'
        "<p id=times>0.1 ms · 49 ms · 312 ms · 964 ms · 1.28 s</p>"
        "<p id=text>Open 9 to 18. 2 turns, 3 s later, in 5 seconds.</p>"
    )
    camera.shoot("steady")
    expect(camera.page.locator("#addr")).to_have_text(shots.ADDRESS)
    expect(camera.page.locator("#times")).to_have_text("0.2 ms · 0.2 ms · 300 ms · 1.0 s · 1.3 s")
    expect(camera.page.locator("#text")).to_have_text(
        "Open 9 to 18. 2 turns, 3.0 s later, in 5 seconds."
    )


def test_an_image_that_shows_a_path_of_this_machine_is_refused(
    camera: Camera, tmp_path, monkeypatch
):
    monkeypatch.setattr(camera, "folder", tmp_path)
    camera.page.set_content(f'<input value="{Path.home()}/speedypack">')
    with pytest.raises(AssertionError, match="a path of this machine"):
        camera.shoot("private")
    assert not (tmp_path / f"private-{camera.theme}.webp").exists()
    reword(camera.page, str(Path.home()), "/home/ada")
    assert camera.shoot("private").is_file()


def test_an_image_over_the_ceiling_is_refused(camera: Camera, tmp_path, monkeypatch):
    monkeypatch.setattr(camera, "folder", tmp_path)
    monkeypatch.setattr(shots, "MAX_FILE_BYTES", 100)
    camera.page.set_content("<p>More than a hundred bytes of image.</p>")
    with pytest.raises(AssertionError, match="over the 100"):
        camera.shoot("heavy")
    assert not (tmp_path / f"heavy-{camera.theme}.webp").exists()


@pytest.mark.parametrize(("value", "written"), [("1", True), ("0", False), (None, False)])
def test_images_reach_the_documentation_only_when_asked(monkeypatch, value, written):
    monkeypatch.delenv("CHATOMON_SCREENSHOTS", raising=False)
    if value is not None:
        monkeypatch.setenv("CHATOMON_SCREENSHOTS", value)
    assert shots.writing() is written
