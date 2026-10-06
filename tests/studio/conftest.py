"""Fixtures of the studio's tests."""

from __future__ import annotations

import pytest

from tests.studio import frameworks


@pytest.fixture(params=frameworks.NAMES)
def framework(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> frameworks.StudioHarness:
    """Run the test once for every framework the studio opens."""
    return frameworks.harness(request.param, monkeypatch)
