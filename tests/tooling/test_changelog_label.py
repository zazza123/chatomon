"""Tests for the changelog label derived from a pull request title."""

import pytest
from scripts import changelog_label as changelog
from scripts.changelog_label import CHANGELOG_LABELS, changelog_label
from scripts.check_pr_conventions import COMMIT_CODES
from scripts.derive_issue_labels import LABELS_FILE, load_label_names


@pytest.mark.parametrize(
    ("title", "label"),
    [
        ("ADD: ADK connector (#12)", "changelog:feature"),
        ("UPD: Stream guidance chunks (#13)", "changelog:feature"),
        ("FIX: Keep the guidance when streaming (#14)", "changelog:fix"),
        ("SEC: Refuse oversized judge prompts (#15)", "changelog:security"),
        ("DOC: Fix a typo in the connectors guide", "changelog:docs"),
        ("DEL: Remove the legacy verdict alias (#16)", "changelog:maintenance"),
        ("REF: Extract the span helper (#17)", "changelog:maintenance"),
        ("TST: Pin streaming behaviour (#18)", "changelog:maintenance"),
        ("STY: Reformat (#19)", "changelog:maintenance"),
        ("CFG: Take the version from the tag (#20)", "changelog:maintenance"),
        ("DEP: Bump the python group with 3 updates", "changelog:maintenance"),
        ("MRG: Bring main into the release branch (#21)", "changelog:maintenance"),
    ],
)
def test_the_title_code_decides_the_label(title, label):
    assert changelog_label(title) == label


@pytest.mark.parametrize("title", ["WIP: halfway", "Bump x", "add: lower case", "ADD:no space", ""])
def test_titles_without_a_known_code_get_no_label(title):
    assert changelog_label(title) is None


def test_every_commit_code_has_a_label():
    assert set(CHANGELOG_LABELS) == set(COMMIT_CODES)


def test_every_changelog_label_exists():
    labels = load_label_names(LABELS_FILE.read_text(encoding="utf-8"))
    assert set(CHANGELOG_LABELS.values()) <= labels


def test_main_prints_the_label(capsys):
    assert changelog.main({"PR_TITLE": "FIX: Something (#1)"}) == 0
    assert capsys.readouterr().out == "changelog:fix\n"


def test_main_prints_nothing_without_a_code(capsys):
    assert changelog.main({"PR_TITLE": "Bump x"}) == 0
    assert capsys.readouterr().out == ""
