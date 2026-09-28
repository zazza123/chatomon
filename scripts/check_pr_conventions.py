"""Pull request convention gate for chatomon (CONTRIBUTING.md, "Pull requests").

What is checked depends on where the pull request's branch lives:

- the branch name ``<type>/<issue>-<slug>`` — branches of this repository only;
- the title ``CODE: prose (#issue)``, where the issue reference is mandatory for
  branches of this repository and, from forks, for ``ADD``, ``UPD``, ``REF`` and
  ``SEC`` only;
- exactly one ``type:`` label on the linked issue (R1);
- the ``CODE: Title`` subject of every commit — branches of this repository only,
  because pull requests from forks are squash-merged.

Dependabot's pull requests are exempt, recognised by their author — never by the
branch name, which whoever opens the pull request chooses.

Inputs are environment variables, so the gate runs without a live pull request:
``PR_TITLE``, ``PR_BRANCH``, ``PR_AUTHOR``, ``PR_BASE_REPO`` and ``PR_NUMBER`` are
required; ``PR_HEAD_REPO`` is empty when the fork was deleted. Every violation is
collected and reported in one run. Exit codes: 0 clean, 1 violations, 2 misuse.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

COMMIT_CODES = ("ADD", "UPD", "FIX", "DEL", "DOC", "REF", "TST", "STY", "CFG", "SEC", "DEP", "MRG")
"""The action codes of the git-commit-convention skill that may reach ``main``."""

FORK_ISSUE_REQUIRED = frozenset({"ADD", "UPD", "REF", "SEC"})
"""Codes whose pull requests from forks must reference an issue."""

BRANCH_TYPES = ("feat", "fix", "refactor", "docs", "chore")
DEPENDABOT = "dependabot[bot]"
REQUIRED_ENV = ("PR_TITLE", "PR_BRANCH", "PR_AUTHOR", "PR_BASE_REPO", "PR_NUMBER")

_CODES = "|".join(COMMIT_CODES)
BRANCH_RE = re.compile(rf"(?:{'|'.join(BRANCH_TYPES)})/[0-9]+-[a-z0-9]+(?:-[a-z0-9]+)*")
TITLE_RE = re.compile(rf"(?P<code>{_CODES}): (?P<prose>\S.*?)(?: \(#(?P<issue>[0-9]+)\))?")
SUBJECT_RE = re.compile(rf"(?:{_CODES}): \S.*")


@dataclass(frozen=True)
class PullRequest:
    """The facts about a pull request that the gate checks.

    Attributes:
        title: The pull request title.
        branch: The head branch name.
        author: The login of the pull request's author.
        is_fork: Whether the head branch lives in another repository.
    """

    title: str
    branch: str
    author: str
    is_fork: bool


@dataclass(frozen=True)
class Issue:
    """What the gate needs to know about the issue a title references.

    Attributes:
        labels: The issue's label names.
        is_pull_request: Whether the number belongs to a pull request — GitHub
            numbers issues and pull requests from the same sequence.
    """

    labels: tuple[str, ...]
    is_pull_request: bool = False


def check(
    pr: PullRequest,
    issue: Callable[[int], Issue | None],
    commit_subjects: Callable[[], Sequence[str] | None],
) -> list[str]:
    """Return every convention the pull request violates.

    Args:
        pr: The pull request to check.
        issue: Returns the issue with a given number, or ``None`` when it cannot
            be read.
        commit_subjects: Returns the subject line of every commit of the pull
            request, or ``None`` when they cannot be read.

    Returns:
        One message per violation; empty when the pull request is clean.
    """
    if pr.author == DEPENDABOT:
        return []
    # Checked first and alone: bash-style line-oriented matching once let a
    # well-formed first line carry anything after it. str.isprintable() rejects
    # every line break (LF, VT, FF, CR, NEL, LS, PS) and every control character.
    if not (pr.title.isprintable() and pr.branch.isprintable()):
        return ["the branch name or the title contains a line break or a control character"]

    violations: list[str] = []
    if not pr.is_fork and BRANCH_RE.fullmatch(pr.branch) is None:
        violations.append(
            f"branch name {pr.branch!r}: expected <type>/<issue>-<slug>, "
            f"type in {'|'.join(BRANCH_TYPES)}"
        )
    match = TITLE_RE.fullmatch(pr.title)
    if match is None:
        violations.append(
            f"title {pr.title!r}: expected 'CODE: prose (#issue)', "
            f"CODE in {', '.join(COMMIT_CODES)}"
        )
    else:
        violations.extend(_check_issue(match["code"], match["issue"], pr.is_fork, issue))
    if not pr.is_fork:
        violations.extend(_check_commits(commit_subjects))
    return violations


def _check_issue(
    code: str,
    number: str | None,
    is_fork: bool,
    issue: Callable[[int], Issue | None],
) -> list[str]:
    if number is None:
        if not is_fork:
            return ["the title carries no issue reference: append ' (#<issue>)'"]
        if code in FORK_ISSUE_REQUIRED:
            return [
                f"a {code} pull request from a fork must reference an issue: open one to "
                "agree on the change, then append ' (#<issue>)' to the title"
            ]
        return []
    found = issue(int(number))
    if found is None:
        return [
            f"could not read the labels of issue #{number} (does it exist?): the gate fails "
            "closed when the GitHub API does not answer — check the number, then re-run it"
        ]
    if found.is_pull_request:
        return [
            f"#{number} is a pull request, not an issue: "
            "reference the issue the pull request closes"
        ]
    types = sum(label.startswith("type:") for label in found.labels)
    if types != 1:
        return [f"issue #{number} carries {types} type: labels: exactly one is required (R1)"]
    return []


def _check_commits(commit_subjects: Callable[[], Sequence[str] | None]) -> list[str]:
    subjects = commit_subjects()
    if subjects is None:
        return ["could not read the commits of the pull request: re-run the gate"]
    return [
        _commit_violation(subject)
        for subject in subjects
        if not (subject.isprintable() and SUBJECT_RE.fullmatch(subject))
    ]


def _commit_violation(subject: str) -> str:
    if subject.startswith("Merge "):
        return (
            f"commit subject {subject!r}: a merge of another branch into this one — drop it "
            "with git rebase main, which is how a branch is kept current"
        )
    return (
        f"commit subject {subject!r}: expected 'CODE: Title' (reword it with an interactive "
        "rebase; update the branch with git rebase main, never by merging main into it)"
    )


def _gh(*args: str) -> str | None:
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def fetch_issue(repo: str, number: int) -> Issue | None:
    """Read an issue's labels, and whether it is a pull request, with the GitHub CLI.

    Args:
        repo: The repository, as ``owner/name``.
        number: The issue number.

    Returns:
        The issue, or ``None`` when the API call fails.
    """
    output = _gh(
        "api",
        f"repos/{repo}/issues/{number}",
        "--jq",
        "{labels: [.labels[].name], pull_request: (.pull_request != null)}",
    )
    if output is None:
        return None
    data = json.loads(output)
    return Issue(labels=tuple(data["labels"]), is_pull_request=data["pull_request"])


def fetch_commit_subjects(repo: str, number: str) -> list[str] | None:
    """Read the subject line of every commit of a pull request with the GitHub CLI.

    Args:
        repo: The repository, as ``owner/name``.
        number: The pull request number.

    Returns:
        The subjects, oldest first, or ``None`` when the API call fails.
    """
    output = _gh(
        "api",
        "--paginate",
        f"repos/{repo}/pulls/{number}/commits",
        "--jq",
        '.[].commit.message | split("\\n")[0]',
    )
    return None if output is None else output.splitlines()


def main(env: Mapping[str, str] | None = None) -> int:
    """Check the pull request described by the environment and report the result.

    Args:
        env: The environment to read; defaults to ``os.environ``.

    Returns:
        0 when the pull request is clean, 1 when it violates a convention, 2 when
        a required variable is missing.
    """
    env = os.environ if env is None else env
    missing = [name for name in REQUIRED_ENV if not env.get(name)]
    if missing:
        print(f"usage: set {', '.join(missing)}", file=sys.stderr)
        return 2
    repo, number = env["PR_BASE_REPO"], env["PR_NUMBER"]
    pr = PullRequest(
        title=env["PR_TITLE"],
        branch=env["PR_BRANCH"],
        author=env["PR_AUTHOR"],
        is_fork=env.get("PR_HEAD_REPO", "") != repo,
    )
    violations = check(
        pr,
        issue=lambda issue: fetch_issue(repo, issue),
        commit_subjects=lambda: fetch_commit_subjects(repo, number),
    )
    for violation in violations:
        print(f"VIOLATION: {violation}")
    if violations:
        print(f"\n{len(violations)} convention violation(s). See CONTRIBUTING.md, 'Pull requests'.")
        return 1
    print("PR conventions OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
