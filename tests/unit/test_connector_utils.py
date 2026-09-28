import asyncio
import threading
from contextvars import ContextVar

from chatomon._sync import run_sync
from chatomon.connectors import render_guidance

VALUE: ContextVar[str] = ContextVar("VALUE")


async def _thread_ident() -> int:
    return threading.get_ident()


async def _read_value() -> str:
    return VALUE.get()


def test_render_guidance_lists_every_item():
    assert render_guidance(["Be brief.", "No prices."]) == (
        "<guidance>\n- Be brief.\n- No prices.\n</guidance>"
    )


def test_run_sync_without_a_running_loop():
    assert run_sync(lambda: asyncio.sleep(0, result=5)) == 5


async def test_run_sync_inside_a_running_loop_uses_a_worker_thread():
    assert run_sync(_thread_ident) != threading.get_ident()


def test_run_sync_keeps_context_variables():
    VALUE.set("outside")
    assert run_sync(_read_value) == "outside"


async def test_run_sync_keeps_context_variables_inside_a_running_loop():
    VALUE.set("inside")
    assert run_sync(_read_value) == "inside"
