"""Derive the axis labels of an issue from the body an issue form wrote.

A dropdown in a GitHub issue form cannot apply a label: GitHub writes the chosen
options into the issue body, under the dropdown's ``### <label>`` heading. Every
option of the *Area* and *Framework* menus starts with its label, so this script
reads those sections back and prints the labels to add, one per line
(CONTRIBUTING.md, "How a menu becomes a label").

Only labels listed in ``.github/labels.yml`` are printed, so text a reporter
types under a look-alike heading cannot invent labels.

Input: the ``ISSUE_BODY`` environment variable. Exit codes: 0 success, 2 when the
labels file cannot be read.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

AXIS_HEADINGS: dict[str, str] = {"Area": "area", "Framework": "framework"}
"""The heading of each classification menu in the issue forms, and its label prefix."""

LABELS_FILE = Path(__file__).resolve().parent.parent / ".github" / "labels.yml"
"""The canonical label list."""

_HEADING_RE = re.compile(r"^### (?P<heading>.+?)[ \t]*$", re.MULTILINE)
_NAME_RE = re.compile(r"""^- name: *(?P<quote>["']?)(?P<name>.+?)(?P=quote)[ \t]*$""", re.MULTILINE)


def load_label_names(text: str) -> frozenset[str]:
    """Return the label names declared in the text of a ``labels.yml`` file.

    Args:
        text: The content of ``.github/labels.yml``.

    Returns:
        The value of every ``- name:`` entry.
    """
    return frozenset(match["name"] for match in _NAME_RE.finditer(text))


def derive_labels(body: str, known: frozenset[str]) -> list[str]:
    """Return the axis labels named in the classification sections of an issue body.

    Args:
        body: The issue body as GitHub stores it.
        known: The labels that exist; any other token is ignored.

    Returns:
        The labels in the order they appear, without duplicates.
    """
    text = body.replace("\r\n", "\n")
    headings = list(_HEADING_RE.finditer(text))
    found: list[str] = []
    for index, heading in enumerate(headings):
        axis = AXIS_HEADINGS.get(heading["heading"])
        if axis is None:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[heading.end() : end]
        for token in re.finditer(rf"(?<![\w:-]){axis}:[a-z0-9][a-z0-9-]*", section):
            label = token.group(0)
            if label in known and label not in found:
                found.append(label)
    return found


def main(env: Mapping[str, str] | None = None, labels_file: Path = LABELS_FILE) -> int:
    """Print the labels derived from ``ISSUE_BODY``, one per line.

    Args:
        env: The environment to read; defaults to ``os.environ``.
        labels_file: The canonical label list.

    Returns:
        0 on success, 2 when the labels file cannot be read.
    """
    env = os.environ if env is None else env
    try:
        known = load_label_names(labels_file.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"cannot read {labels_file}: {error}", file=sys.stderr)
        return 2
    for label in derive_labels(env.get("ISSUE_BODY", ""), known):
        print(label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
