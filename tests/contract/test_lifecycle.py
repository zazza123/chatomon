import asyncio

import pytest

from chatomon import OrbitalContext, Verdict, orbital
from tests.contract.harness import build_agent
from tests.fakes import HangingOrbital, ScriptedOrbital


async def test_model_error_aborts_the_turn_and_the_next_turn_works(harness):
    agent, reports = build_agent(harness, [RuntimeError("model down"), "Back online."])
    with pytest.raises(RuntimeError, match="model down"):
        await harness.ask(agent, "hello")
    assert reports[0].final_action == "aborted"
    assert reports[0].error_type == "RuntimeError"
    assert await harness.ask(agent, "hello again") == "Back online."
    assert reports[1].final_action == "answered"


async def test_a_turn_cancelled_while_an_output_orbital_runs_reports_the_attempt(harness):
    hanging = HangingOrbital()
    agent, reports = build_agent(
        harness,
        ["Open 24/7"],
        output_orbitals=[ScriptedOrbital(name="first"), hanging],
        capture_content="full",
    )
    running = asyncio.ensure_future(harness.ask(agent, "hours?"))
    await hanging.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert [report.final_action for report in reports] == ["aborted"]
    [attempt] = reports[0].output
    # The reply under check and the orbitals that had answered are in the report.
    assert attempt.nucleus_reply == "Open 24/7"
    assert [result.name for result in attempt.results] == ["first"]
    assert attempt.decision is None


def test_sync_api(harness):
    if not harness.supports_sync:
        pytest.skip("MAF exposes only an asynchronous native run API")
    agent, reports = build_agent(harness, ["Hi!"], output_orbitals=[ScriptedOrbital(name="out")])
    assert harness.ask_sync(agent, "hello") == "Hi!"
    assert reports[0].final_action == "answered"


async def test_concurrent_conversations_are_isolated(harness):
    seen = ScriptedOrbital(Verdict.allow(), name="seen")
    agent, reports = build_agent(harness, ["one", "two"], input_orbitals=[seen])
    await asyncio.gather(
        harness.ask(agent, "first", conversation="a"),
        harness.ask(agent, "second", conversation="b"),
    )
    assert sorted(ctx.message for ctx in seen.contexts) == ["first", "second"]
    assert len({report.turn_id for report in reports}) == 2
    assert len({report.conversation_id for report in reports}) == 2
    assert all(report.final_action == "answered" for report in reports)


async def test_chatomon_spans_are_emitted(harness, spans):
    agent, _ = build_agent(
        harness,
        ["ok"],
        input_orbitals=[ScriptedOrbital(name="i")],
        output_orbitals=[ScriptedOrbital(name="o")],
        tracer_provider=spans.provider,
    )
    await harness.ask(agent, "hello")
    names = {span.name for span in spans.exporter.get_finished_spans()}
    assert {
        "chatomon.input",
        "chatomon.output",
        "chatomon.orbital i",
        "chatomon.orbital o",
    } <= names


async def test_concurrent_conversations_do_not_share_guidance(harness):
    @orbital
    def personal_hint(ctx: OrbitalContext) -> Verdict:
        return Verdict.allow(f"hint for {ctx.message}")

    agent, _ = build_agent(harness, ["one", "two"], input_orbitals=[personal_hint])
    await asyncio.gather(
        harness.ask(agent, "first", conversation="a"),
        harness.ask(agent, "second", conversation="b"),
    )
    for call in harness.calls:
        message = call.messages[-1][1]
        other = "second" if message == "first" else "first"
        assert f"hint for {message}" in call.system
        assert f"hint for {other}" not in call.system
