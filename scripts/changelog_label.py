"""Map a pull request title to the changelog label release-drafter groups it by.

The title's commit code decides the section of the release notes
(CONTRIBUTING.md, "Versioning and releases"). The workflow that calls this
replaces any other ``changelog:`` label, because the title — not the label — is
the source of truth.

Input: the ``PR_TITLE`` environment variable. Prints the label, or nothing when
the title carries no known code. Always exits 0.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping

_MAINTENANCE = "changelog:maintenance"

CHANGELOG_LABELS: dict[str, str] = {
    "ADD": "changelog:feature",
    "UPD": "changelog:feature",
    "FIX": "changelog:fix",
    "SEC": "changelog:security",
    "DOC": "changelog:docs",
    "DEL": _MAINTENANCE,
    "REF": _MAINTENANCE,
    "TST": _MAINTENANCE,
    "STY": _MAINTENANCE,
    "CFG": _MAINTENANCE,
    "DEP": _MAINTENANCE,
    "MRG": _MAINTENANCE,
}
"""The changelog label for each commit code."""

_CODE_RE = re.compile(r"(?P<code>[A-Z]{3}): ")


def changelog_label(title: str) -> str | None:
    """Return the changelog label for a pull request title.

    Args:
        title: The pull request title.

    Returns:
        The label, or ``None`` when the title does not start with a known code.
    """
    match = _CODE_RE.match(title)
    return None if match is None else CHANGELOG_LABELS.get(match["code"])


def main(env: Mapping[str, str] | None = None) -> int:
    """Print the changelog label for ``PR_TITLE``.

    Args:
        env: The environment to read; defaults to ``os.environ``.

    Returns:
        Always 0.
    """
    env = os.environ if env is None else env
    label = changelog_label(env.get("PR_TITLE", ""))
    if label is not None:
        print(label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
