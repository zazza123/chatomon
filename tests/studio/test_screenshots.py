"""The screenshots of the studio that the documentation shows, one scene per image.

Every scene runs with the browser tests, so a change to the page that breaks one
is seen at once. With `CHATOMON_SCREENSHOTS=1` the images are written where the
pages read them: see "Studio screenshots" in `CONTRIBUTING.md`.
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import expect
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom, Knowledge
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import JudgeOrbital, MessageLengthOrbital
from tests.fake_service import FakeService
from tests.studio import shots
from tests.studio.shots import Camera, reword
from tests.studio.test_e2e import (
    Studio,
    block_of,
    card,
    open_card,
    open_report,
    open_studio,
    running,
    save_as_test,
    send,
    shown,
)

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


def test_an_image_has_the_scale_of_the_camera_and_the_page_keeps_it(
    camera: Camera, tmp_path, monkeypatch
):
    monkeypatch.setattr(camera, "folder", tmp_path)
    camera.page.set_content("<h1>Opening hours</h1><p>Monday to Friday, 9 to 18.</p>")
    image = camera.shoot("scale").read_bytes()
    # Chromium writes the extended form of WebP, which holds each side, less one, in three bytes.
    assert image[12:16] == b"VP8X"
    width, height = (int.from_bytes(image[at : at + 3], "little") + 1 for at in (24, 27))
    assert (width, height) == tuple(int(side * shots.SCALE) for side in shots.VIEWPORT.values())
    # Taking the image must not undo the scale of the page: the next image has it too.
    assert camera.page.evaluate("devicePixelRatio") == shots.SCALE


@pytest.mark.parametrize(("value", "written"), [("1", True), ("0", False), (None, False)])
def test_images_reach_the_documentation_only_when_asked(monkeypatch, value, written):
    monkeypatch.delenv("CHATOMON_SCREENSHOTS", raising=False)
    if value is not None:
        monkeypatch.setenv("CHATOMON_SCREENSHOTS", value)
    assert shots.writing() is written


MODEL = "openai:gpt-5-mini"
SOURCE = "speedypack.app:atom"
HOURS = "When are you open on Saturday?"
CLOSED = "We are closed on Saturday. Our offices are open Monday to Friday, 9 to 18."
POEM = "Can you write a poem about my neighbour?"
REFUSAL = "I can only help with SpeedyPack parcels and deliveries."
PROJECT = "/home/ada/speedypack"
SCOPE_GUARD_URL = "https://scope-guard.speedypack.example"
SCOPE_GUARD_KEY = "SCOPE_GUARD_API_KEY"


def _text(messages: list[ModelMessage]) -> str:
    """The text of the last request: what the user wrote, or what a judge reviews."""
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    return "".join(
        part.content
        for part in request.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


async def nucleus(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answer as SpeedyPack's assistant would, after the time a model takes.

    A blocked message does not reach the nucleus: what is not the question about
    Saturday is the request it is guided to refuse.
    """
    await asyncio.sleep(0.66)
    return ModelResponse(parts=[TextPart(CLOSED if "Saturday" in _text(messages) else REFUSAL)])


async def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Decide as the judges of the atom would: a request for a poem is out of scope."""
    await asyncio.sleep(0.3)
    reviewed = _text(messages).split("<text_under_review>")[-1]
    decision: dict[str, str] = {"action": "allow", "reason": "in scope"}
    if "poem" in reviewed:
        decision = {
            "action": "block",
            "guidance": "Say that you only help with parcels and deliveries.",
            "reason": "out of scope",
            "explanation": "The message asks for a poem, which is not about parcels.",
        }
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, decision)])


def speedypack() -> Atom:
    """The atom every scene shows: SpeedyPack's assistant, with fake models behind it."""
    hours = Knowledge("Opening hours: Monday to Friday, 9 to 18. Closed on Saturday and Sunday.")
    return Atom(
        name="speedypack",
        instructions="You are the customer assistant of SpeedyPack, a parcel courier.",
        knowledge=hours,
        connector=PydanticAIConnector(
            FunctionModel(nucleus, model_name=MODEL),
            resolve_model=lambda name: FunctionModel(judge, model_name=name),
        ),
        input_orbitals=[
            MessageLengthOrbital(max_words=60),
            JudgeOrbital(
                name="scope",
                model=MODEL,
                instructions="Block messages that are not about parcels, deliveries or SpeedyPack.",
            ),
        ],
        output_orbitals=[
            JudgeOrbital(
                name="facts",
                model=MODEL,
                instructions="Block replies that state facts missing from the reference.",
                context=hours,
            ),
        ],
    )


@pytest.fixture
def studio() -> Iterator[Studio]:
    with running(speedypack(), source=SOURCE) as handle:
        yield handle


def talk(camera: Camera, studio: Studio) -> None:
    """Open the page and hold the conversation of the scenes: an answer, then a refusal."""
    open_studio(camera.page, studio)
    send(camera.page, HOURS, CLOSED)
    send(camera.page, POEM, REFUSAL)
    expect(camera.page.get_by_role("button", name="Send")).to_be_enabled()


def test_overview(camera: Camera, studio: Studio):
    talk(camera, studio)
    camera.shoot("overview")


def test_report(camera: Camera, studio: Studio):
    talk(camera, studio)
    open_report(camera.page, 2)
    camera.shoot("report", card(camera.page))


def test_orbital_card(camera: Camera, studio: Studio):
    page = camera.page
    open_studio(page, studio)
    open_card(page, "message_length")
    words = card(page).get_by_label("max_words")
    words.fill("40")
    words.press("Tab")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    camera.shoot("orbital-card", card(page))


def test_copy_as_code(camera: Camera, studio: Studio):
    page = camera.page
    open_studio(page, studio)
    open_card(page, "message_length")
    words = card(page).get_by_label("max_words")
    words.fill("40")
    words.press("Tab")
    expect(page.locator("#changes-text")).to_have_text("1 change vs code")
    card(page).get_by_role("button", name="Done").click()
    page.get_by_role("button", name="Copy as code").click()
    expect(card(page).locator("pre.code")).to_contain_text("MessageLengthOrbital(max_words=40),")
    camera.shoot("copy-as-code", card(page))


def test_save_as_test(camera: Camera, tmp_path, monkeypatch):
    # A studio writes tests only when it was started from a module, in the directory it is in.
    monkeypatch.chdir(tmp_path)
    atom = speedypack()
    with running(atom, source=SOURCE, namespace={"atom": atom}) as studio:
        talk(camera, studio)
        save_as_test(camera.page, 2)
        expect(card(camera.page).locator("pre.code")).to_contain_text("def test_")
        reword(camera.page, str(tmp_path.resolve()), PROJECT)
        camera.shoot("save-as-test", card(camera.page))


def test_add_orbital(camera: Camera, studio: Studio, service: FakeService, monkeypatch):
    page = camera.page
    monkeypatch.setenv(SCOPE_GUARD_KEY, "a-key-that-never-reaches-the-page")
    service.key = "a-key-that-never-reaches-the-page"
    service.delay = 0.3
    open_studio(page, studio)
    page.locator('#drawer [data-act="add"][data-side="input"]').click()
    card(page).get_by_role("button", name="ScopeGuardOrbital").click()
    address = block_of(page, "guard").get_by_label("api_url")
    address.fill(service.url)
    block_of(page, "guard").get_by_label("api_key_env").fill(SCOPE_GUARD_KEY)
    block_of(page, "description").locator("[data-blk-text]").fill(
        "SpeedyPack is a parcel courier. The assistant tracks parcels and explains deliveries."
    )
    card(page).get_by_role("button", name="Test connection").click()
    expect(shown(page).locator(".pill")).to_have_text("allow")
    # The fake service listens on a port the system chose: the image shows an address
    # a reader could have.
    reword(page, service.url, SCOPE_GUARD_URL)
    camera.shoot("add-orbital", card(page))
