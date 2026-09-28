"""Engine lifecycle: guidance accumulation, aborted turns, on_turn failures."""

import logging

import pytest

from chatomon import TurnReport, Verdict
from chatomon.observability import ContentPolicy
from chatomon.schemas import Message
from tests.fakes import ScriptedOrbital, make_engine

QUESTION = "Where is my parcel?"


async def test_on_turn_failure_is_logged_without_traceback(caplog):
    caplog.set_level(logging.DEBUG, logger="chatomon")

    def broken(report: TurnReport) -> None:
        raise RuntimeError("secret in the callback error")

    engine = make_engine(on_turn=broken)
    turn = engine.start_turn()
    await engine.run_input(turn, message=QUESTION)
    await engine.run_output(turn, reply="ok")
    await engine.finish_turn(turn)
    failures = [r for r in caplog.records if "on_turn" in r.getMessage()]
    assert failures
    assert failures[0].exc_info is None
    assert "RuntimeError" in failures[0].getMessage()
    assert all("secret in the callback error" not in r.getMessage() for r in caplog.records)


async def test_regeneration_guidance_accumulates_within_the_turn():
    judge = ScriptedOrbital(
        Verdict.block(guidance="Fix the hours."),
        Verdict.block(guidance="Do not mention prices."),
        Verdict.allow(),
        name="judge",
    )
    engine = make_engine(outputs=[judge])
    turn = engine.start_turn()
    await engine.run_input(turn, message=QUESTION)
    first = await engine.run_output(turn, reply="a")
    second = await engine.run_output(turn, reply="b")
    assert first.guidance == ("Fix the hours.",)
    assert second.guidance == ("Fix the hours.", "Do not mention prices.")


async def test_abort_turn_produces_an_aborted_report_without_content():
    received: list[TurnReport] = []
    engine = make_engine(on_turn=received.append)
    turn = engine.start_turn(conversation_id="c1")
    await engine.run_input(turn, message=QUESTION, history=(Message(role="user", content="hi"),))
    report = await engine.abort_turn(turn, RuntimeError("model crashed on secret text"))
    assert report.final_action == "aborted"
    assert report.error_type == "RuntimeError"
    assert report.error_message is None
    assert "secret text" not in report.model_dump_json()
    assert received == [report]
    with pytest.raises(RuntimeError, match="already"):
        await engine.finish_turn(turn)


async def test_abort_turn_records_the_message_only_with_full_capture():
    engine = make_engine(content=ContentPolicy("full"))
    turn = engine.start_turn()
    report = await engine.abort_turn(turn, ValueError("boom"))
    assert report.error_message == "boom"
    assert report.input == ()


async def test_reopen_output_allows_checking_a_new_reply_after_acceptance():
    out = ScriptedOrbital(name="out")
    engine = make_engine(outputs=[out])
    turn = engine.start_turn()
    await engine.run_input(turn, message=QUESTION)
    await engine.run_output(turn, reply="first")
    engine.reopen_output(turn)
    assert turn.final_action is None
    decision = await engine.run_output(turn, reply="second")
    assert decision.kind == "accept"
    assert turn.final_reply == "second"
    assert [ctx.attempt for ctx in out.contexts] == [1, 2]


async def test_reopen_output_is_rejected_after_an_input_reply_or_a_finished_turn():
    engine = make_engine(inputs=[ScriptedOrbital(Verdict.respond("FAQ"), name="faq")])
    turn = engine.start_turn()
    await engine.run_input(turn, message=QUESTION)
    with pytest.raises(RuntimeError, match="input chain"):
        engine.reopen_output(turn)
    await engine.finish_turn(turn)
    with pytest.raises(RuntimeError, match="finished"):
        engine.reopen_output(turn)
