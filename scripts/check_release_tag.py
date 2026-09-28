"""Release tag check run by publish.yml before anything ships.

A release tag is ``v<major>.<minor>.<patch>``, optionally followed by ``a``, ``b``
or ``rc`` and a number (CONTRIBUTING.md, "Versioning and releases"). The tag and
the release's *pre-release* flag must agree: a release candidate published as a
final release would become GitHub's Latest release and deploy the documentation,
and a final version marked as a pre-release would not.

Inputs are the ``TAG`` and ``PRERELEASE`` (``true``/``false``) environment
variables. Exit codes: 0 valid, 1 invalid, 2 misuse.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping

TAG_RE = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+(?P<pre>(?:a|b|rc)[0-9]+)?")


def check_release_tag(tag: str, *, prerelease: bool) -> list[str]:
    """Return every problem with a release tag.

    Args:
        tag: The release's tag name.
        prerelease: Whether the release is marked as a pre-release.

    Returns:
        One message per problem; empty when the tag can be released.
    """
    match = TAG_RE.fullmatch(tag)
    if match is None:
        return [f"tag {tag!r} is not v<major>.<minor>.<patch>, optionally followed by a|b|rc<N>"]
    if match["pre"] and not prerelease:
        return [
            f"tag {tag} is a pre-release version, but the release is not marked as a "
            "pre-release: tick 'Set as a pre-release', or it becomes Latest and deploys the docs"
        ]
    if not match["pre"] and prerelease:
        return [
            f"tag {tag} is a final version, but the release is marked as a pre-release: "
            "untick 'Set as a pre-release', or use an a|b|rc tag"
        ]
    return []


def main(env: Mapping[str, str] | None = None) -> int:
    """Check the release described by the environment and report the result.

    Args:
        env: The environment to read; defaults to ``os.environ``.

    Returns:
        0 when the tag can be released, 1 when it cannot, 2 when ``TAG`` is missing.
    """
    env = os.environ if env is None else env
    tag = env.get("TAG", "")
    if not tag:
        print("usage: set TAG (and PRERELEASE=true|false)", file=sys.stderr)
        return 2
    problems = check_release_tag(tag, prerelease=env.get("PRERELEASE") == "true")
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        return 1
    print(f"Tag {tag} can be released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
