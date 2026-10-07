"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests import fake_service

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration and e2e tests unless their environment variable is set."""
    rules = [
        (
            "integration",
            "CHATOMON_INTEGRATION",
            "integration test: set CHATOMON_INTEGRATION=1 to run",
        ),
        ("e2e", "CHATOMON_E2E", "browser test: set CHATOMON_E2E=1 to run"),
    ]
    for marker, variable, reason in rules:
        if os.environ.get(variable) == "1":
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def spans() -> SimpleNamespace:
    """A tracer whose finished spans are kept in memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return SimpleNamespace(
        tracer=provider.get_tracer("tests"), exporter=exporter, provider=provider
    )


@pytest.fixture
def service() -> Iterator[fake_service.FakeService]:
    """The fake Principled Intelligence services, listening on loopback."""
    with fake_service.running() as running:
        yield running
