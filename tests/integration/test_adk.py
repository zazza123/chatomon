"""`ADKConnector` against a real Gemini model.

Run with Google credentials in the environment, on Vertex AI:
    CHATOMON_INTEGRATION=1 GOOGLE_GENAI_USE_VERTEXAI=true \
        GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=<location> \
        uv run pytest tests/integration/test_adk.py -m integration -v

or with a key of the Gemini API in `GOOGLE_API_KEY`. `CHATOMON_ADK_MODEL` names
the model; without it the tests call the one ADK defaults to.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable
from typing import TypeVar

import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

from chatomon import Atom, Knowledge, OrbitalContext, TurnReport, Verdict, orbital
from chatomon.connectors.adk import ADKConnector
from chatomon.orbitals import JudgeOrbital
from tests.contract.harness_adk import reply
from tests.integration.conftest import INTEGRATION_TIMEOUT

pytestmark = pytest.mark.integration

T = TypeVar("T")

HOURS = "Opening hours: Monday to Friday, 9 to 18. Closed at weekends."


def model() -> str:
    """Name the Gemini model to call, or skip when no Google credentials are set."""
    if not (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_API_KEY")):
        pytest.skip("set GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY to run this integration test")
    return os.environ.get("CHATOMON_ADK_MODEL") or LlmAgent(name="default").canonical_model.model


async def in_time(call: Awaitable[T]) -> T:
    """Wait for a call to a real service, and fail instead of waiting for ever."""
    return await asyncio.wait_for(call, timeout=INTEGRATION_TIMEOUT)


async def ask(atom: Atom, message: str) -> str:
    runner = InMemoryRunner(app=atom.build())
    return reply(await in_time(runner.run_debug(message, quiet=True)))


async def test_a_judge_gets_its_decision_from_a_real_model():
    name = model()
    judge = JudgeOrbital(
        model=name,
        instructions="Block replies that state facts the reference does not support.",
        context=Knowledge(HOURS),
        timeout=INTEGRATION_TIMEOUT,
    )
    judge.bind(ADKConnector(name))

    def reply_under_check(text: str) -> OrbitalContext:
        return OrbitalContext(side="output", message=text, turn_id="integration")

    invented = await in_time(judge.check(reply_under_check("We are open 24 hours a day.")))
    faithful = await in_time(judge.check(reply_under_check("We are open Monday to Friday.")))
    assert (invented.action, faithful.action) == ("block", "allow")


async def test_guidance_reaches_the_model_and_a_rejected_reply_is_written_again():
    @orbital
    def banana(ctx: OrbitalContext) -> Verdict:
        return Verdict.allow("End every reply with the word BANANA.")

    @orbital
    def pineapple(ctx: OrbitalContext) -> Verdict:
        if "PINEAPPLE" in ctx.message.upper():
            return Verdict.allow()
        return Verdict.block(guidance="Reply with the single word PINEAPPLE and nothing else.")

    reports: list[TurnReport] = []
    atom = Atom(
        name="speedypack",
        instructions="You are the customer assistant of SpeedyPack. Be brief.",
        connector=ADKConnector(model()),
        input_orbitals=[banana],
        output_orbitals=[pineapple],
        on_turn=reports.append,
        capture_content="full",
    )
    assert "PINEAPPLE" in (await ask(atom, "Hello, are you there?")).upper()
    first, last = reports[0].output[0], reports[0].output[-1]
    # The first reply followed the input guidance, and was sent back with the output's.
    assert "BANANA" in (first.nucleus_reply or "").upper()
    assert (first.decision, last.decision) == ("regenerate", "accept")
    assert reports[0].final_action == "answered"


async def test_a_turn_with_a_tool_checks_only_the_final_reply():
    called: list[str] = []
    checked: list[str] = []

    def opening_hours() -> str:
        """Look up the opening hours of SpeedyPack."""
        called.append("opening_hours")
        return "Monday to Friday, 9 to 18"

    @orbital
    def seen(ctx: OrbitalContext) -> Verdict:
        checked.append(ctx.message)
        return Verdict.allow()

    reports: list[TurnReport] = []
    atom = Atom(
        name="speedypack",
        instructions="You are the customer assistant of SpeedyPack. Always look the opening "
        "hours up with the tool before you answer about them.",
        connector=ADKConnector(model(), tools=[opening_hours]),
        output_orbitals=[seen],
        on_turn=reports.append,
    )
    answer = await ask(atom, "What are your opening hours?")
    # What the model writes is its own: the test looks at what ran and what was checked.
    assert called == ["opening_hours"]
    assert answer and checked == [answer]
    assert [attempt.decision for attempt in reports[0].output] == ["accept"]
