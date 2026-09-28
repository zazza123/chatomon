"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless CHATOMON_INTEGRATION=1."""
    if os.environ.get("CHATOMON_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test: set CHATOMON_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
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
