"""Tests for the release tag check run before anything is published."""

import pytest
from scripts import check_release_tag as release
from scripts.check_release_tag import check_release_tag


@pytest.mark.parametrize(
    ("tag", "prerelease"),
    [
        ("v0.1.0", False),
        ("v1.12.3", False),
        ("v0.2.0rc1", True),
        ("v0.2.0a1", True),
        ("v0.2.0b2", True),
    ],
)
def test_a_well_formed_tag_that_agrees_with_the_flag_passes(tag, prerelease):
    assert check_release_tag(tag, prerelease=prerelease) == []


@pytest.mark.parametrize(
    "tag", ["0.1.0", "v0.1", "v0.1.0-rc.1", "v0.1.0.post1", "v0.1.0rc", "release-0.1.0", "v0.1.0\n"]
)
def test_a_malformed_tag_is_rejected(tag):
    violations = check_release_tag(tag, prerelease=False)
    assert len(violations) == 1
    assert "is not v<major>.<minor>.<patch>" in violations[0]


def test_a_release_candidate_published_as_final_is_rejected():
    assert check_release_tag("v0.2.0rc1", prerelease=False) == [
        "tag v0.2.0rc1 is a pre-release version, but the release is not marked as a "
        "pre-release: tick 'Set as a pre-release', or it becomes Latest and deploys the docs"
    ]


def test_a_final_version_published_as_prerelease_is_rejected():
    assert check_release_tag("v0.2.0", prerelease=True) == [
        "tag v0.2.0 is a final version, but the release is marked as a pre-release: "
        "untick 'Set as a pre-release', or use an a|b|rc tag"
    ]


def test_main_reads_the_environment(capsys):
    assert release.main({"TAG": "v0.2.0rc1", "PRERELEASE": "true"}) == 0
    assert "v0.2.0rc1" in capsys.readouterr().out
    assert release.main({"TAG": "v0.2.0rc1", "PRERELEASE": "false"}) == 1


def test_main_exits_2_without_a_tag(capsys):
    assert release.main({"PRERELEASE": "false"}) == 2
    assert "TAG" in capsys.readouterr().err
