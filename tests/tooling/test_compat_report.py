"""Tests for the report of a failed compatibility run."""

from scripts import compat_report
from scripts.compat_report import direct_names, report

RUN = "https://github.com/zazza123/chatomon/actions/runs/1"
DIRECT = frozenset({"agent-framework-core", "pydantic-ai-slim", "pytest"})

UPGRADED = """Resolved 149 packages in 354ms
Updated agent-framework-core v1.19.0 -> v1.21.0
Updated pydantic-ai-slim v2.51.0 -> v2.55.0
Updated pydantic-graph v2.51.0 -> v2.55.0
Updated typer v0.27.2 -> v0.27.3
Added regex v2026.9.29
"""

PYTEST = """tests/contract/test_input.py F.....                                      [  1%]
=========================== short test summary info ============================
SKIPPED [1] tests/contract/test_lifecycle.py:43: MAF exposes only an asynchronous native run API
FAILED tests/contract/test_input.py::test_plain_answer[pydantic_ai] - AssertionError: aborted
FAILED tests/connectors/test_review_fixes.py::test_a - assert ['a - b'] == []
FAILED tests/studio/test_events.py::test_event_sequence[guided refusal] - assert 1 == 2
ERROR tests/studio - ModuleNotFoundError: The package agent-framework-foundry is required
========== 3 failed, 11 passed, 1 skipped, 1 error in 1.09s ==========
"""


def test_the_report_names_the_run_the_newer_packages_and_the_failed_tests():
    text = report(RUN, UPGRADED, PYTEST, DIRECT)
    assert RUN in text
    # The packages pyproject.toml names, then how many of the others: those are the
    # releases to look at first.
    assert (
        "- agent-framework-core 1.19.0 → 1.21.0\n"
        "- pydantic-ai-slim 2.51.0 → 2.55.0\n"
        "- and 2 packages that they depend on\n"
    ) in text
    assert (
        "- `tests/contract/test_input.py::test_plain_answer[pydantic_ai]`\n"
        "- `tests/connectors/test_review_fixes.py::test_a`\n"
        "- `tests/studio/test_events.py::test_event_sequence[guided refusal]`\n"
        "- `tests/studio`\n"
    ) in text


def test_the_report_carries_no_text_of_a_failure():
    # What a test compared can be long and is in the log: the report is a list of names.
    text = report(RUN, UPGRADED, PYTEST, DIRECT)
    assert "AssertionError" not in text
    assert "aborted" not in text
    assert "SKIPPED" not in text and "test_lifecycle" not in text
    assert "regex" not in text and "typer" not in text


def test_a_long_list_of_failed_tests_is_cut():
    lines = "".join(f"FAILED tests/test_x.py::test_{n} - assert False\n" for n in range(45))
    text = report(RUN, UPGRADED, lines, DIRECT)
    assert "- `tests/test_x.py::test_29`\n" in text
    assert "test_30" not in text
    assert "and 15 more" in text


def test_a_run_that_stopped_before_the_summary_says_so():
    text = report(RUN, UPGRADED, "collecting ...\n", DIRECT)
    assert "did not reach the summary of the tests" in text


def test_a_run_with_no_newer_package_says_so():
    text = report(RUN, "Resolved 149 packages in 354ms\n", PYTEST, DIRECT)
    assert "No package is newer than the lockfile" in text


def test_a_run_where_only_other_packages_are_newer_says_how_many():
    text = report(RUN, "Updated typer v0.27.2 -> v0.27.3\n", PYTEST, DIRECT)
    assert "- 1 package that the ones `pyproject.toml` names depend on\n" in text


def test_the_names_of_pyproject_are_read_from_every_table_and_made_alike():
    text = """
[project]
dependencies = ["pydantic>=2.10", "opentelemetry-api>=1.28"]

[project.optional-dependencies]
pydantic-ai = ["pydantic-ai-slim>=2.14,<3"]
spectral = ["chatomon[studio]", "spectral-bridge>=0.5,<0.6"]

[dependency-groups]
dev = ["pytest>=9", "Agent_Framework.Core>=1.19,<2", {include-group = "docs"}]
docs = ["zensical==0.0.66"]
"""
    assert direct_names(text) == frozenset(
        {
            "pydantic",
            "opentelemetry-api",
            "pydantic-ai-slim",
            "chatomon",
            "spectral-bridge",
            "pytest",
            "agent-framework-core",
            "zensical",
        }
    )


def test_this_repository_s_pyproject_names_the_frameworks():
    names = direct_names(compat_report.PYPROJECT.read_text(encoding="utf-8"))
    assert {"pydantic-ai-slim", "langchain", "agent-framework-core", "google-adk"} <= names


def test_the_command_reads_the_files_it_is_given_and_does_without_a_missing_one(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "pytest.txt").write_text(PYTEST, encoding="utf-8")
    monkeypatch.setenv("RUN_URL", RUN)
    code = compat_report.main([str(tmp_path / "upgraded.txt"), str(tmp_path / "pytest.txt")])
    out = capsys.readouterr().out
    assert code == 0
    assert RUN in out
    assert "No package is newer than the lockfile" in out
    assert "`tests/studio`" in out
