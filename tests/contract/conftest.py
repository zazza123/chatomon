"""Run every contract test once per connector."""

from __future__ import annotations

import pytest

from tests.contract.harness import Harness

HARNESSES = ["pydantic_ai"]


@pytest.fixture(params=HARNESSES)
def harness(request: pytest.FixtureRequest) -> Harness:
    if request.param == "pydantic_ai":
        from tests.contract.harness_pydantic_ai import PydanticAIHarness

        return PydanticAIHarness()
    from tests.contract.harness_langchain import LangChainHarness  # pyright: ignore[reportMissingImports]

    return LangChainHarness()
