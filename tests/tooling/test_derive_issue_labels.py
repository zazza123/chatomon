"""Tests for deriving axis labels from an issue form's body."""

import pytest
from scripts import derive_issue_labels as derive
from scripts.derive_issue_labels import derive_labels, load_label_names

KNOWN = frozenset(
    {
        "type:fix",
        "area:core",
        "area:connectors",
        "area:observability",
        "framework:pydantic-ai",
        "framework:langchain",
    }
)

# GitHub writes every answer of a multi-select menu on one line, comma-separated.
AREA_ANSWER = (
    "area:connectors — Shared connector layer, contract suite, "
    "area:observability — Logging, OpenTelemetry, turn reports"
)

BUG_BODY = f"""### What happens

The guidance disappears. Mentioning area:core here must not label anything.

### What should happen

It stays.

### Minimal reproducer

```python
print("area:core")
```

### chatomon version

0.1.0

### Area

{AREA_ANSWER}

### Framework

framework:langchain — LangChain / LangGraph
"""


def test_labels_come_from_the_classification_sections_only():
    assert derive_labels(BUG_BODY, KNOWN) == [
        "area:connectors",
        "area:observability",
        "framework:langchain",
    ]


def test_unanswered_menus_give_no_labels():
    body = "### Area\n\n_No response_\n\n### Framework\n\nNone / don't know\n"
    assert derive_labels(body, KNOWN) == []


def test_a_body_without_form_headings_gives_no_labels():
    assert derive_labels("Opened from the CLI: area:core", KNOWN) == []


def test_unknown_labels_are_dropped():
    assert derive_labels("### Area\n\narea:rag — Retrieval\n", KNOWN) == []


def test_the_prefix_must_match_the_heading():
    assert derive_labels("### Area\n\nframework:langchain — LangChain\n", KNOWN) == []


def test_a_token_glued_to_another_word_is_ignored():
    assert derive_labels("### Area\n\nnot-area:core\n", KNOWN) == []


def test_windows_line_endings_are_handled():
    assert derive_labels("### Area\r\n\r\narea:core — Core\r\n", KNOWN) == ["area:core"]


def test_labels_are_not_repeated():
    body = "### Area\n\narea:core\n\n### Area\n\narea:core, area:connectors\n"
    assert derive_labels(body, KNOWN) == ["area:core", "area:connectors"]


def test_label_names_are_read_quoted_or_not():
    text = '- name: "type:feat"\n  color: "a2eeef"\n- name: \'good first issue\'\n- name: python\n'
    assert load_label_names(text) == {"type:feat", "good first issue", "python"}


def test_main_prints_one_label_per_line(tmp_path, capsys):
    labels_file = tmp_path / "labels.yml"
    labels_file.write_text('- name: "area:core"\n- name: "framework:langchain"\n')
    body = "### Area\n\narea:core\n\n### Framework\n\nframework:langchain\n"
    assert derive.main({"ISSUE_BODY": body}, labels_file) == 0
    assert capsys.readouterr().out == "area:core\nframework:langchain\n"


def test_main_prints_nothing_for_an_empty_body(tmp_path, capsys):
    labels_file = tmp_path / "labels.yml"
    labels_file.write_text('- name: "area:core"\n')
    assert derive.main({}, labels_file) == 0
    assert capsys.readouterr().out == ""


def test_main_exits_2_without_the_labels_file(tmp_path, capsys):
    assert derive.main({"ISSUE_BODY": "### Area\n\narea:core\n"}, tmp_path / "missing.yml") == 2
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.parametrize("heading", sorted(derive.AXIS_HEADINGS))
def test_every_axis_heading_maps_to_a_label_prefix(heading):
    assert derive.AXIS_HEADINGS[heading] in {"area", "framework"}
