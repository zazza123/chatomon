import logging
import time

import pytest

from chatomon import Verdict, orbital
from chatomon.engine import Engine, TurnState
from chatomon.schemas import DEFAULT_REFUSAL, InputDecision, Message, OrbitalContext
from tests.fakes import ScriptedOrbital, SlowOrbital, make_engine

HISTORY = (Message(role="user", content="hi"), Message(role="assistant", content="hello!"))


async def run_input(
    engine: Engine, message: str = "Where is my parcel?"
) -> tuple[TurnState, InputDecision]:
    turn = engine.start_turn(conversation_id="conv-1")
    decision = await engine.run_input(turn, message=message, history=HISTORY)
    return turn, decision


async def test_no_input_orbitals_proceeds_without_guidance():
    _, decision = await run_input(make_engine())
    assert decision.kind == "proceed"
    assert decision.guidance == ()
    assert not decision.redact_user_message


async def test_allow_guidance_accumulates_in_order():
    engine = make_engine(
        inputs=[
            ScriptedOrbital(Verdict.allow("first"), name="a"),
            ScriptedOrbital(Verdict.allow(), name="b"),
            ScriptedOrbital(Verdict.allow("second"), name="c"),
        ]
    )
    _, decision = await run_input(engine)
    assert decision.kind == "proceed"
    assert decision.guidance == ("first", "second")


async def test_respond_stops_the_chain_and_sets_the_final_reply():
    later = ScriptedOrbital(name="later")
    engine = make_engine(
        inputs=[ScriptedOrbital(Verdict.respond("We open at 9."), name="faq"), later]
    )
    turn, decision = await run_input(engine)
    assert decision.kind == "reply"
    assert decision.reply == "We open at 9."
    assert not decision.redact_user_message
    assert later.contexts == []
    assert turn.final_action == "orbital_response"
    assert turn.final_reply == "We open at 9."


async def test_block_with_reply_returns_it_and_redacts_the_message():
    engine = make_engine(inputs=[ScriptedOrbital(Verdict.block("Only parcels, sorry."), name="g")])
    turn, decision = await run_input(engine)
    assert decision.kind == "reply"
    assert decision.reply == "Only parcels, sorry."
    assert decision.redact_user_message
    assert turn.final_action == "blocked"


async def test_block_without_reply_uses_the_refusal():
    engine = make_engine(inputs=[ScriptedOrbital(Verdict.block(), name="g")], refusal="Nope.")
    turn, decision = await run_input(engine)
    assert decision.reply == "Nope."
    assert turn.final_reply == "Nope."


async def test_block_with_guidance_is_a_guided_refusal_with_only_that_guidance():
    engine = make_engine(
        inputs=[
            ScriptedOrbital(Verdict.allow("earlier hint"), name="a"),
            ScriptedOrbital(Verdict.block(guidance="Decline politely."), name="b"),
        ]
    )
    turn, decision = await run_input(engine)
    assert decision.kind == "guided_refusal"
    assert decision.guidance == ("Decline politely.",)
    assert decision.redact_user_message
    assert turn.final_action is None


async def test_orbitals_receive_context_and_earlier_annotations():
    first = ScriptedOrbital(Verdict.allow(annotations={"topic": "tracking"}), name="first")
    second = ScriptedOrbital(name="second", reads=["first"])
    turn, _ = await run_input(make_engine(inputs=[first, second]), message="Where is it?")
    ctx = second.contexts[0]
    assert ctx.side == "input"
    assert ctx.message == "Where is it?"
    assert ctx.history == HISTORY
    assert ctx.annotations == {"first": {"topic": "tracking"}}
    assert ctx.attempt == 1
    assert ctx.turn_id == turn.turn_id
    assert ctx.conversation_id == "conv-1"
    assert first.contexts[0].annotations == {}


async def test_escalate_marks_the_turn():
    engine = make_engine(
        inputs=[ScriptedOrbital(Verdict.respond("A human will call you.", escalate=True), name="g")]
    )
    turn, _ = await run_input(engine)
    assert turn.escalated


async def test_failing_orbital_blocks_by_default():
    turn, decision = await run_input(
        make_engine(inputs=[ScriptedOrbital(RuntimeError("boom"), name="g")])
    )
    assert decision.kind == "reply"
    assert decision.reply == DEFAULT_REFUSAL
    result = turn.input_results[0]
    assert result.error_type == "RuntimeError"
    assert result.error_policy == "block"
    assert result.reason == "orbital_error"
    assert result.action == "block"


async def test_failing_orbital_with_on_error_allow_lets_the_message_through():
    engine = make_engine(inputs=[ScriptedOrbital(RuntimeError("boom"), name="g", on_error="allow")])
    turn, decision = await run_input(engine)
    assert decision.kind == "proceed"
    assert turn.input_results[0].error_policy == "allow"


async def test_timeout_is_an_error():
    turn, decision = await run_input(make_engine(inputs=[SlowOrbital(name="slow", timeout=0.05)]))
    assert decision.reply == DEFAULT_REFUSAL
    assert turn.input_results[0].error_type == "TimeoutError"


async def test_sync_function_orbital_that_blocks_still_times_out():
    @orbital(timeout=0.05)
    def stuck(ctx: OrbitalContext) -> Verdict:
        time.sleep(0.5)
        return Verdict.allow()

    started = time.perf_counter()
    turn, decision = await run_input(make_engine(inputs=[stuck]))
    assert time.perf_counter() - started < 0.4
    assert turn.input_results[0].error_type == "TimeoutError"
    assert decision.kind == "reply"


async def test_returning_something_else_than_a_verdict_is_an_error():
    @orbital
    async def broken(ctx: OrbitalContext) -> Verdict:
        return "allow"  # type: ignore[return-value]

    turn, decision = await run_input(make_engine(inputs=[broken]))
    assert turn.input_results[0].error_type == "TypeError"
    assert decision.kind == "reply"


async def test_reader_fails_when_the_orbital_it_reads_failed():
    producer = ScriptedOrbital(RuntimeError("down"), name="claims", on_error="allow")
    reader = ScriptedOrbital(name="judge", reads=["claims"])
    turn, decision = await run_input(make_engine(inputs=[producer, reader]))
    assert reader.contexts == []
    assert turn.input_results[1].error_type == "MissingAnnotationsError"
    assert decision.kind == "reply"


async def test_run_input_twice_is_rejected():
    engine = make_engine()
    turn, _ = await run_input(engine)
    with pytest.raises(RuntimeError, match="already"):
        await engine.run_input(turn, message="again")


def test_negative_max_retries_is_rejected():
    with pytest.raises(ValueError, match="max_retries"):
        make_engine(max_retries=-1)


async def test_spans_describe_the_chain_and_each_orbital(spans):
    engine = make_engine(
        inputs=[
            ScriptedOrbital(Verdict.allow("hint"), name="a"),
            ScriptedOrbital(RuntimeError("x"), name="b", on_error="allow"),
        ],
        tracer=spans.tracer,
    )
    turn, _ = await run_input(engine)
    finished = {span.name: span for span in spans.exporter.get_finished_spans()}
    chain = finished["chatomon.input"]
    a = finished["chatomon.orbital a"]
    b = finished["chatomon.orbital b"]
    assert a.parent is not None and a.parent.span_id == chain.context.span_id
    assert chain.attributes["chatomon.decision.kind"] == "proceed"
    assert chain.attributes["chatomon.turn.id"] == turn.turn_id
    assert chain.attributes["chatomon.conversation.id"] == "conv-1"
    assert a.attributes["chatomon.verdict.action"] == "allow"
    assert a.attributes["chatomon.verdict.has_guidance"] is True
    assert a.attributes["chatomon.orbital.fingerprint"].startswith("sha256:")
    assert b.attributes["error.type"] == "RuntimeError"
    assert b.status.status_code.name == "ERROR"


async def test_failures_are_logged_as_warnings(caplog):
    caplog.set_level(logging.DEBUG, logger="chatomon")
    await run_input(make_engine(inputs=[ScriptedOrbital(RuntimeError("x"), name="b")]))
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings
    assert warnings[0].__dict__["chatomon"]["orbital"] == "b"
    assert warnings[0].__dict__["chatomon"]["error_type"] == "RuntimeError"
