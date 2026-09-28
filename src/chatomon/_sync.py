"""Run chatomon's async engine from synchronous framework hooks."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

__all__ = ["run_sync"]

T = TypeVar("T")

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _background_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is None or _thread is None or not _thread.is_alive():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, name="chatomon-sync", daemon=True)
            _thread.start()
        return _loop


def run_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine to completion from synchronous code.

    Every call runs on one long-lived event loop in a daemon thread, so async
    clients that bind to the loop they first run on (HTTP clients, SDK clients
    cached by orbitals) keep working across turns. The coroutine runs with a copy
    of the caller's context variables, so OpenTelemetry spans keep their parent.

    Args:
        factory: Called once, on the background loop, to create the coroutine.

    Returns:
        The coroutine's result.
    """
    loop = _background_loop()
    context = contextvars.copy_context()
    if threading.current_thread() is _thread:
        # Called from the background loop itself: waiting on it would deadlock.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(context.run, asyncio.run, factory()).result()

    done: Future[T] = Future()

    def start() -> None:
        task = context.run(loop.create_task, factory())

        def settle(finished: asyncio.Task[T]) -> None:
            if finished.cancelled():
                done.cancel()
            elif (error := finished.exception()) is not None:
                done.set_exception(error)
            else:
                done.set_result(finished.result())

        task.add_done_callback(settle)

    loop.call_soon_threadsafe(start)
    return done.result()
