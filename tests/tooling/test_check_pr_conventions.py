"""Tests for the pull request convention gate."""

import pytest
from scripts import check_pr_conventions as gate
from scripts.check_pr_conventions import Issue, PullRequest, check

ONE_TYPE = ["type:feat", "area:connectors"]


def labels_of(mapping):
    def lookup(number):
        labels = mapping.get(number)
        return None if labels is None else Issue(labels=tuple(labels))

    return lookup


def no_issue_lookup(number):
    raise AssertionError(f"issue #{number} must not be looked up")


def commits(*subjects):
    return lambda: list(subjects)


def no_commit_lookup():
    raise AssertionError("the commits must not be read")


def internal(title="ADD: ADK connector (#12)", branch="feat/12-adk-connector", author="zazza123"):
    return PullRequest(title=title, branch=branch, author=author, is_fork=False)


def fork(title="DOC: Fix a typo in the connectors guide", branch="patch-1", author="someone"):
    return PullRequest(title=title, branch=branch, author=author, is_fork=True)


def test_a_clean_internal_pull_request_passes():
    violations = check(
        internal(),
        labels_of({12: ONE_TYPE}),
        commits("ADD: ADK connector module", "TST: Contract harness for ADK"),
    )
    assert violations == []


@pytest.mark.parametrize(
    "branch",
    [
        "connectors-v0.2",
        "chore/bump-deps",
        "feat/12_adk",
        "Feat/12-adk",
        "epic/12-adk",
        "feat/12-",
        "feat/12-adk--connector",
        "feat/x-adk",
    ],
)
def test_an_internal_branch_must_be_type_issue_slug(branch):
    violations = check(internal(branch=branch), labels_of({12: ONE_TYPE}), commits())
    assert len(violations) == 1
    assert "branch name" in violations[0]


@pytest.mark.parametrize(
    "title",
    [
        "Add ADK connector (#12)",
        "WIP: ADK connector (#12)",
        "ADD:ADK connector (#12)",
        "add: ADK connector (#12)",
        "feat(connectors): ADK connector (#12)",
        "ADD:   ",
    ],
)
def test_the_title_must_start_with_a_commit_code(title):
    violations = check(internal(title=title), labels_of({12: ONE_TYPE}), commits())
    assert len(violations) == 1
    assert "title" in violations[0]


def test_an_internal_title_must_reference_an_issue():
    violations = check(internal(title="ADD: ADK connector"), no_issue_lookup, commits())
    assert violations == ["the title carries no issue reference: append ' (#<issue>)'"]


@pytest.mark.parametrize(
    "hidden", ["\n", "\r", "\x0b", "\x0c", "\x85", "\u2028", "\u2029", "\x00", "\t"]
)
def test_line_breaks_and_control_characters_are_rejected_before_anything_else(hidden):
    title = f"ADD: ADK connector (#12){hidden}injected"
    violations = check(internal(title=title), no_issue_lookup, no_commit_lookup)
    assert violations == [
        "the branch name or the title contains a line break or a control character"
    ]


def test_a_control_character_in_the_branch_is_rejected():
    violations = check(internal(branch="feat/12-adk\n"), no_issue_lookup, no_commit_lookup)
    assert len(violations) == 1


def test_printable_non_ascii_text_passes():
    title = "UPD: Handle naïve datetimes — and emoji 🚀 (#12)"
    assert check(internal(title=title), labels_of({12: ONE_TYPE}), commits()) == []


@pytest.mark.parametrize("labels", [["area:core"], ["type:feat", "type:fix"]])
def test_the_linked_issue_needs_exactly_one_type_label(labels):
    violations = check(internal(), labels_of({12: labels}), commits())
    assert violations == [
        f"issue #12 carries {sum(label.startswith('type:') for label in labels)} type: labels: "
        "exactly one is required (R1)"
    ]


def test_unreadable_issue_labels_fail_closed():
    violations = check(internal(), labels_of({}), commits())
    assert len(violations) == 1
    assert "could not read the labels of issue #12" in violations[0]
    assert "does it exist?" in violations[0]


def test_a_fork_branch_name_and_commits_are_not_checked():
    violations = check(fork(branch="whatever you like"), no_issue_lookup, no_commit_lookup)
    assert violations == []


@pytest.mark.parametrize("code", ["DOC", "FIX", "TST", "STY", "CFG", "DEP", "DEL"])
def test_a_fork_may_omit_the_issue_for_small_changes(code):
    assert check(fork(title=f"{code}: Small change"), no_issue_lookup, no_commit_lookup) == []


@pytest.mark.parametrize("code", ["ADD", "UPD", "REF", "SEC"])
def test_a_fork_must_reference_an_issue_for_substantial_changes(code):
    violations = check(fork(title=f"{code}: Big change"), no_issue_lookup, no_commit_lookup)
    assert len(violations) == 1
    assert f"a {code} pull request from a fork must reference an issue" in violations[0]


def test_a_fork_that_references_an_issue_gets_it_checked():
    violations = check(
        fork(title="ADD: ADK connector (#12)"),
        labels_of({12: ["type:feat", "type:epic"]}),
        no_commit_lookup,
    )
    assert violations == ["issue #12 carries 2 type: labels: exactly one is required (R1)"]


@pytest.mark.parametrize(
    "subject",
    [
        "wip",
        "WIP: halfway there",
        "Fix the thing",
        "ADD:no space",
        "Merge branch 'main' into feat/12-adk",
    ],
)
def test_every_internal_commit_subject_must_follow_the_convention(subject):
    violations = check(internal(), labels_of({12: ONE_TYPE}), commits("ADD: Fine", subject))
    assert len(violations) == 1
    assert repr(subject) in violations[0]
    assert "git rebase main" in violations[0]


def test_unreadable_commits_fail_closed():
    violations = check(internal(), labels_of({12: ONE_TYPE}), lambda: None)
    assert len(violations) == 1
    assert "could not read the commits" in violations[0]


def test_dependabot_is_exempt():
    pr = PullRequest(
        title="Bump the python group with 3 updates",
        branch="dependabot/uv/python-abc",
        author="dependabot[bot]",
        is_fork=False,
    )
    assert check(pr, no_issue_lookup, no_commit_lookup) == []


def test_a_dependabot_looking_branch_from_anyone_else_is_checked():
    pr = PullRequest(title="Bump x", branch="dependabot/uv/x", author="mallory", is_fork=False)
    violations = check(pr, no_issue_lookup, commits())
    assert len(violations) == 2


def test_all_violations_are_reported_together():
    violations = check(
        internal(title="Add ADK", branch="adk"),
        no_issue_lookup,
        commits("bad subject"),
    )
    assert len(violations) == 3


BASE_ENV = {
    "PR_TITLE": "ADD: ADK connector (#12)",
    "PR_BRANCH": "feat/12-adk-connector",
    "PR_AUTHOR": "zazza123",
    "PR_HEAD_REPO": "zazza123/chatomon",
    "PR_BASE_REPO": "zazza123/chatomon",
    "PR_NUMBER": "7",
}


def fake_github(monkeypatch, labels=ONE_TYPE, subjects=("ADD: ADK connector module",)):
    seen = []

    def issue(repo, number):
        seen.append(("issue", repo, number))
        return Issue(labels=tuple(labels))

    def commit_subjects(repo, number):
        seen.append(("commits", repo, number))
        return list(subjects)

    monkeypatch.setattr(gate, "fetch_issue", issue)
    monkeypatch.setattr(gate, "fetch_commit_subjects", commit_subjects)
    return seen


def test_main_passes_a_clean_pull_request(monkeypatch, capsys):
    seen = fake_github(monkeypatch)
    assert gate.main(BASE_ENV) == 0
    assert ("issue", "zazza123/chatomon", 12) in seen
    assert ("commits", "zazza123/chatomon", "7") in seen
    assert "PR conventions OK" in capsys.readouterr().out


def test_main_reports_violations_and_exits_1(monkeypatch, capsys):
    fake_github(monkeypatch)
    assert gate.main({**BASE_ENV, "PR_BRANCH": "adk"}) == 1
    out = capsys.readouterr().out
    assert "VIOLATION: branch name 'adk'" in out
    assert "1 convention violation(s)" in out


def test_main_treats_a_different_head_repository_as_a_fork(monkeypatch):
    seen = fake_github(monkeypatch)
    env = {**BASE_ENV, "PR_HEAD_REPO": "someone/chatomon", "PR_BRANCH": "patch-1"}
    assert gate.main(env) == 0
    assert all(kind != "commits" for kind, *_ in seen)


def test_main_treats_a_deleted_head_repository_as_a_fork(monkeypatch):
    fake_github(monkeypatch)
    env = {**BASE_ENV, "PR_HEAD_REPO": "", "PR_BRANCH": "patch-1"}
    assert gate.main(env) == 0


@pytest.mark.parametrize(
    "missing", ["PR_TITLE", "PR_BRANCH", "PR_AUTHOR", "PR_BASE_REPO", "PR_NUMBER"]
)
def test_main_exits_2_when_an_input_is_missing(missing, capsys):
    env = {name: value for name, value in BASE_ENV.items() if name != missing}
    assert gate.main(env) == 2
    assert missing in capsys.readouterr().err


def test_a_pull_request_number_in_the_title_is_named_as_such():
    def lookup(number):
        return Issue(labels=(), is_pull_request=True)

    violations = check(internal(), lookup, commits())
    assert violations == [
        "#12 is a pull request, not an issue: reference the issue the pull request closes"
    ]


def test_a_merge_commit_is_to_be_dropped_not_reworded():
    subject = "Merge branch 'main' into feat/12-adk"
    violations = check(internal(), labels_of({12: ONE_TYPE}), commits(subject))
    assert len(violations) == 1
    assert "drop it with git rebase main" in violations[0]
    assert "reword" not in violations[0]
