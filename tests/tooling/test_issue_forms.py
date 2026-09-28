"""The issue forms, the label parser and labels.yml must agree with each other."""

import re
from pathlib import Path

import pytest
import yaml
from scripts.derive_issue_labels import AXIS_HEADINGS, LABELS_FILE, load_label_names

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"
FORMS = sorted(path for path in TEMPLATES.glob("*.yml") if path.name != "config.yml")
LABEL_TEXT = LABELS_FILE.read_text(encoding="utf-8")
LABELS = load_label_names(LABEL_TEXT)
OPTION_LABEL_RE = re.compile(r"(?P<label>[a-z-]+:[a-z0-9-]+) — ")


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dropdowns(form):
    return [item for item in form["body"] if item["type"] == "dropdown"]


def option_labels(dropdown):
    matches = (OPTION_LABEL_RE.match(option) for option in dropdown["attributes"]["options"])
    return {match["label"] for match in matches if match}


def test_the_label_parser_agrees_with_a_yaml_parser():
    assert {entry["name"] for entry in yaml.safe_load(LABEL_TEXT)} == LABELS


def test_blank_issues_are_disabled():
    assert load(TEMPLATES / "config.yml")["blank_issues_enabled"] is False


@pytest.mark.parametrize("path", FORMS, ids=lambda path: path.name)
def test_a_form_applies_exactly_one_existing_type_label(path):
    labels = load(path)["labels"]
    assert set(labels) <= LABELS
    assert sum(label.startswith("type:") for label in labels) == 1


@pytest.mark.parametrize("path", FORMS, ids=lambda path: path.name)
def test_menu_options_name_existing_labels_of_their_axis(path):
    for dropdown in dropdowns(load(path)):
        found = option_labels(dropdown)
        if not found:
            continue
        heading = dropdown["attributes"]["label"]
        assert heading in AXIS_HEADINGS, f"{path.name}: menu {heading!r} is unknown to the parser"
        assert found <= LABELS, f"{path.name}: {sorted(found - LABELS)} not in labels.yml"
        assert all(label.startswith(f"{AXIS_HEADINGS[heading]}:") for label in found)


@pytest.mark.parametrize("path", FORMS, ids=lambda path: path.name)
def test_every_axis_label_is_offered_by_every_form(path):
    menus = {dropdown["attributes"]["label"]: dropdown for dropdown in dropdowns(load(path))}
    for heading, axis in AXIS_HEADINGS.items():
        expected = {label for label in LABELS if label.startswith(f"{axis}:")}
        assert heading in menus, f"{path.name}: no {heading!r} menu"
        assert option_labels(menus[heading]) == expected, f"{path.name}: {heading!r} menu drifted"
