"""Where the studio writes a file, and when it replaces one."""

from pathlib import Path

import pytest

from chatomon.studio import _files as files
from chatomon.studio._chains import RefusedError

HEADER = '"""Saved from chatomon studio'
TEXT = HEADER + ': app:atom, turn 1."""\n\nimport pytest\n'
MINE = "print('written by hand')\n"


def write(root: Path, path: str, *, text: str = TEXT, overwrite: bool = False) -> str:
    return files.write(root, path, text, header=HEADER, overwrite=overwrite)


def test_a_new_file_is_written_with_its_folders(tmp_path):
    assert write(tmp_path, "tests/studio/test_x.py") == "tests/studio/test_x.py"
    assert (tmp_path / "tests/studio/test_x.py").read_text(encoding="utf-8") == TEXT


def test_the_written_path_is_the_resolved_one(tmp_path):
    assert write(tmp_path, " tests/./deep/../test_x.py ") == "tests/test_x.py"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("", "give a path inside"),
        ("   ", "give a path inside"),
        ("../x.py", "the file must be inside"),
        ("tests/../../x.py", "the file must be inside"),
        ("tests/notes.txt", r"the file must end in \.py"),
        ("tests/test_x", r"the file must end in \.py"),
        ("tests/test_x.py" + chr(0), "give a path inside"),
    ],
)
def test_a_path_the_studio_does_not_write_to_is_refused(tmp_path, path, message):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(files.InvalidPathError, match=message):
        write(root, path)
    assert list(root.iterdir()) == []
    assert [entry.name for entry in tmp_path.iterdir()] == ["project"]


def test_an_absolute_path_is_refused(tmp_path):
    with pytest.raises(files.InvalidPathError, match="give a path relative to"):
        write(tmp_path, str(tmp_path / "test_x.py"))
    assert list(tmp_path.iterdir()) == []


def test_a_folder_that_links_outside_is_refused(tmp_path):
    outside, root = tmp_path / "outside", tmp_path / "project"
    outside.mkdir()
    root.mkdir()
    try:
        (root / "tests").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not available here")
    with pytest.raises(files.InvalidPathError, match="the file must be inside"):
        write(root, "tests/test_x.py")
    assert list(outside.iterdir()) == []


def test_a_directory_is_not_a_file(tmp_path):
    (tmp_path / "test_x.py").mkdir()
    with pytest.raises(files.InvalidPathError, match=r"test_x\.py is a directory"):
        write(tmp_path, "test_x.py")


def test_a_file_of_the_studio_is_replaced_only_when_asked(tmp_path):
    write(tmp_path, "tests/test_x.py")
    newer = TEXT + "# newer\n"
    with pytest.raises(files.OwnFileExistsError, match=r"tests/test_x\.py exists and was saved"):
        write(tmp_path, "tests/test_x.py", text=newer)
    assert (tmp_path / "tests/test_x.py").read_text(encoding="utf-8") == TEXT
    assert write(tmp_path, "tests/test_x.py", text=newer, overwrite=True) == "tests/test_x.py"
    assert (tmp_path / "tests/test_x.py").read_text(encoding="utf-8") == newer


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_file_the_studio_did_not_write_is_never_replaced(tmp_path, overwrite):
    (tmp_path / "shop.py").write_text(MINE, encoding="utf-8")
    with pytest.raises(
        RefusedError, match=r"shop\.py was not written by the studio: choose another path"
    ):
        write(tmp_path, "shop.py", overwrite=overwrite)
    assert (tmp_path / "shop.py").read_text(encoding="utf-8") == MINE


def test_a_file_that_is_not_text_is_not_the_studios(tmp_path):
    (tmp_path / "data.py").write_bytes(bytes([0xFF, 0xFE, 0x00]))
    with pytest.raises(RefusedError, match="was not written by the studio"):
        write(tmp_path, "data.py", overwrite=True)


def test_a_folder_that_cannot_be_made_is_reported_not_raised(tmp_path):
    (tmp_path / "tests").write_text("a file where a folder is needed\n", encoding="utf-8")
    with pytest.raises(RefusedError, match=r"could not write tests/studio/test_x\.py"):
        write(tmp_path, "tests/studio/test_x.py")


def test_the_proposed_path_is_the_first_free_one(tmp_path):
    assert files.proposed(tmp_path, "test_x") == "tests/studio/test_x.py"
    write(tmp_path, "tests/studio/test_x.py")
    assert files.proposed(tmp_path, "test_x") == "tests/studio/test_x_2.py"
    write(tmp_path, "tests/studio/test_x_2.py")
    assert files.proposed(tmp_path, "test_x") == "tests/studio/test_x_3.py"


def test_what_the_system_refuses_is_reported_not_raised(tmp_path):
    with pytest.raises(RefusedError, match="could not write"):
        write(tmp_path, "tests/" + "x" * 300 + ".py")
    try:
        (tmp_path / "loop").symlink_to(tmp_path / "loop")
    except OSError:
        pytest.skip("symbolic links are not available here")
    with pytest.raises(RefusedError, match="could not write"):
        write(tmp_path, "loop/test_x.py")


def test_the_home_sign_is_not_a_folder(tmp_path):
    with pytest.raises(files.InvalidPathError, match="~ is not expanded here"):
        write(tmp_path, "~/test_x.py")
    with pytest.raises(files.InvalidPathError, match="~ is not expanded here"):
        write(tmp_path, "~someone/tests/test_x.py")
    assert list(tmp_path.iterdir()) == []
    assert write(tmp_path, "tests/~notes/test_x.py") == "tests/~notes/test_x.py"  # only the start
