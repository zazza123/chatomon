"""Security policy every GitHub Actions workflow in this repository follows."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
TRUSTED_OWNERS = {"actions", "github", "pypa"}
PULL_REQUEST_TARGET_ALLOWED = {"changelog-label.yml"}
SHA_RE = re.compile(r"[0-9a-f]{40}")


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def triggers(workflow):
    # PyYAML follows YAML 1.1 and reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True))


def steps(workflow):
    for job in workflow["jobs"].values():
        yield from job.get("steps", [])


def test_there_are_workflows_to_check():
    assert WORKFLOWS


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_a_workflow_grants_no_permission_by_default(path):
    assert load(path)["permissions"] == {}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_third_party_actions_are_pinned_by_commit_sha(path):
    for step in steps(load(path)):
        uses = step.get("uses")
        if uses is None or uses.startswith("./"):
            continue
        action, _, ref = uses.partition("@")
        if action.split("/")[0] not in TRUSTED_OWNERS:
            assert SHA_RE.fullmatch(ref), f"{path.name}: pin {action} by commit SHA"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_pull_request_target_is_used_only_where_documented(path):
    on = triggers(load(path))
    if isinstance(on, dict) and "pull_request_target" in on:
        assert path.name in PULL_REQUEST_TARGET_ALLOWED


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_run_steps_never_interpolate_expressions(path):
    for step in steps(load(path)):
        assert "${{" not in step.get("run", ""), f"{path.name}: pass values to run: through env:"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_job_on_a_runner_has_a_time_limit(path):
    # Without one a step that hangs holds its job until GitHub's limit of six hours.
    for name, job in load(path)["jobs"].items():
        if "uses" in job:
            # A job that calls a workflow takes the limits of that workflow's jobs.
            continue
        limit = job.get("timeout-minutes")
        assert isinstance(limit, int), f"{path.name}: give the job {name} timeout-minutes"
        assert limit <= 30, f"{path.name}: {name} may hang for {limit} minutes"


def test_dependabot_never_raises_the_declared_lower_bounds():
    # The floors in pyproject.toml were found by bisection and are checked by the
    # `lowest` CI job, which would keep passing on silently raised values.
    config = load(ROOT / ".github" / "dependabot.yml")
    uv = next(update for update in config["updates"] if update["package-ecosystem"] == "uv")
    assert uv.get("versioning-strategy") == "widen"
