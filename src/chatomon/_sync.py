"""Run chatomon's async engine from synchronous framework hooks."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

__all__ = ["run_sync"]

T = TypeVar("T")


def run_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine to completion from synchronous code.

    With no event loop running in this thread it uses `asyncio.run`. Inside a
    running loop (for example a sync framework call made from async code) it runs
    the coroutine in a fresh loop on a worker thread, carrying over the caller's
    context variables so OpenTelemetry spans keep their parent.

    Args:
        factory: Called once to create the coroutine.

    Returns:
        The coroutine's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(context.run, asyncio.run, factory()).result()
