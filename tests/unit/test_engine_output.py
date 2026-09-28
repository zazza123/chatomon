import logging

import pytest

from chatomon import TurnReport, Verdict
from chatomon.engine import Engine, TurnState
from chatomon.observability import ContentPolicy, sha256_text
from chatomon.orbitals.base import Orbital
from chatomon.schemas import DEFAULT_REFUSAL, Message, OrbitalContext
from tests.fakes import ScriptedOrbital, make_engine

QUESTION = "Where is my parcel?"
HISTORY = (Message(role="user", content="hi"), Message(role="assistant", content="hello!"))
OUT_HISTORY = (*HISTORY, Message(role="user", content=QUESTION))


async def start(engine: Engine, message: str = QUESTION) -> TurnState:
    turn = engine.start_turn()
    await engine.run_input(turn, message=message, history=HISTORY)
    return turn


async def test_no_output_orbitals_accepts_the_reply():
    engine = make_engine()
    turn = await start(engine)
    decision = await engine.run_output(turn, reply="It is in Rome.", history=OUT_HISTORY)
    assert decision.kind == "accept"
    assert turn.final_action == "answered"
    assert turn.final_reply == "It is in Rome."


async def test_respond_replaces_the_reply():
    engine = make_engine(outputs=[ScriptedOrbital(Verdict.respond("Call us."), name="o")])
    turn = await start(engine)
    decision = await engine.run_output(turn, reply="draft", history=OUT_HISTORY)
    assert decision.kind == "replace"
    assert decision.text == "Call us."
    assert turn.final_action == "orbital_response"


async def test_block_with_reply_replaces_it():
    engine = make_engine(outputs=[ScriptedOrbital(Verdict.block("Not allowed."), name="o")])
    turn = await start(engine)
    decision = await engine.run_output(turn, reply="draft", history=OUT_HISTORY)
    assert (decision.kind, decision.text) == ("replace", "Not allowed.")
    assert turn.final_action == "blocked"


async def test_block_without_reply_uses_the_refusal():
    engine = make_engine(outputs=[ScriptedOrbital(Verdict.block(), name="o")])
    turn = await start(engine)
    decision = await engine.run_output(turn, reply="draft", history=OUT_HISTORY)
    assert decision.text == DEFAULT_REFUSAL


async def test_output_orbitals_see_the_reply_history_and_input_annotations():
    producer = ScriptedOrbital(Verdict.allow(annotations=["track parcel"]), name="intents")
    reader = ScriptedOrbital(name="judge", reads=["intents"])
    engine = make_engine(inputs=[producer], outputs=[reader])
    turn = await start(engine)
    await engine.run_output(turn, reply="It is in Rome.", history=OUT_HISTORY)
    ctx = reader.contexts[0]
    assert ctx.side == "output"
    assert ctx.message == "It is in Rome."
    assert ctx.history == OUT_HISTORY
    assert ctx.annotations == {"intents": ["track parcel"]}
    assert ctx.attempt == 1


async def test_block_with_guidance_asks_for_a_regeneration_then_accepts():
    judge = ScriptedOrbital(
        Verdict.block(guidance="Fix the opening hours."), Verdict.allow(), name="judge"
    )
    engine = make_engine(outputs=[judge])
    turn = await start(engine)
    first = await engine.run_output(turn, reply="Open 24/7.", history=OUT_HISTORY)
    assert first.kind == "regenerate"
    assert first.guidance == ("Fix the opening hours.",)
    assert turn.final_action is None
    second = await engine.run_output(turn, reply="Open 9-18.", history=OUT_HISTORY)
    assert second.kind == "accept"
    assert turn.final_reply == "Open 9-18."
    assert [ctx.attempt for ctx in judge.contexts] == [1, 2]
    assert len(turn.output_results) == 2


@pytest.mark.parametrize(("max_retries", "regenerations"), [(0, 0), (1, 1), (2, 2)])
async def test_retries_are_capped_then_the_refusal_is_used(max_retries, regenerations):
    engine = make_engine(
        outputs=[ScriptedOrbital(Verdict.block(guidance="Try again."), name="judge")],
        max_retries=max_retries,
        refusal="Sorry.",
    )
    turn = await start(engine)
    kinds = []
    decision = None
    for _ in range(regenerations + 1):
        decision = await engine.run_output(turn, reply="bad", history=OUT_HISTORY)
        kinds.append(decision.kind)
    assert kinds == ["regenerate"] * regenerations + ["replace"]
    assert decision is not None and decision.text == "Sorry."
    assert turn.final_action == "fallback_refusal"


async def test_guidance_on_output_allow_is_ignored_with_a_warning(caplog):
    caplog.set_level(logging.WARNING, logger="chatomon")
    engine = make_engine(outputs=[ScriptedOrbital(Verdict.allow("ignored"), name="o")])
    turn = await start(engine)
    decision = await engine.run_output(turn, reply="fine", history=OUT_HISTORY)
    assert decision.kind == "accept"
    assert any("ignored" in record.getMessage() for record in caplog.records)


class FlakyProducer(Orbital):
    """Produces annotations on the first call, fails afterwards."""

    def __init__(self) -> None:
        super().__init__(name="claims", on_error="allow")
        self.calls = 0

    async def check(self, ctx: OrbitalContext) -> Verdict:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("extractor down")
        return Verdict.allow(annotations={"claims": ["open 24/7"]})


async def test_output_annotations_do_not_leak_between_attempts():
    reader = ScriptedOrbital(
        Verdict.block(guidance="Fix it."), Verdict.allow(), name="judge", reads=["claims"]
    )
    engine = make_engine(outputs=[FlakyProducer(), reader])
    turn = await start(engine)
    first = await engine.run_output(turn, reply="Open 24/7.", history=OUT_HISTORY)
    assert first.kind == "regenerate"
    second = await engine.run_output(turn, reply="Open 9-18.", history=OUT_HISTORY)
    assert len(reader.contexts) == 1
    assert turn.output_results[1][1].error_type == "MissingAnnotationsError"
    assert second.kind == "replace"
    assert second.text == DEFAULT_REFUSAL


async def test_guided_refusal_accepted_is_recorded_as_such():
    engine = make_engine(inputs=[ScriptedOrbital(Verdict.block(guidance="Decline."), name="g")])
    turn = await start(engine)
    decision = await engine.run_output(
        turn, reply="I can't help with that, sorry.", history=OUT_HISTORY
    )
    assert decision.kind == "accept"
    assert turn.final_action == "guided_refusal"


async def test_run_output_is_rejected_before_input_and_after_an_input_reply():
    engine = make_engine(inputs=[ScriptedOrbital(Verdict.respond("FAQ"), name="faq")])
    fresh = engine.start_turn()
    with pytest.raises(RuntimeError, match="run_input"):
        await engine.run_output(fresh, reply="x")
    turn = await start(engine)
    with pytest.raises(RuntimeError, match="already produced the reply"):
        await engine.run_output(turn, reply="x")


async def test_output_span_records_attempt_and_decision(spans):
    engine = make_engine(
        outputs=[ScriptedOrbital(Verdict.block("no"), name="o")], tracer=spans.tracer
    )
    turn = await start(engine)
    await engine.run_output(turn, reply="draft", history=OUT_HISTORY)
    output = next(
        span for span in spans.exporter.get_finished_spans() if span.name == "chatomon.output"
    )
    assert output.attributes["chatomon.attempt"] == 1
    assert output.attributes["chatomon.decision.kind"] == "replace"


async def test_finish_turn_reports_the_whole_turn():
    engine = make_engine(
        inputs=[ScriptedOrbital(Verdict.allow("be brief"), name="i")],
        outputs=[ScriptedOrbital(Verdict.block(guidance="Shorter."), Verdict.allow(), name="o")],
    )
    turn = engine.start_turn(conversation_id="conv-9")
    await engine.run_input(turn, message=QUESTION, history=HISTORY)
    await engine.run_output(turn, reply="long", history=OUT_HISTORY)
    await engine.run_output(turn, reply="short", history=OUT_HISTORY)
    report = await engine.finish_turn(turn)
    assert isinstance(report, TurnReport)
    assert report.turn_id == turn.turn_id
    assert report.conversation_id == "conv-9"
    assert report.atom_name == "test-atom"
    assert report.final_action == "answered"
    assert [r.name for r in report.input] == ["i"]
    assert [[r.action for r in attempt] for attempt in report.output] == [["block"], ["allow"]]
    assert report.user_message is None
    assert report.final_reply is None
    assert report.duration_ms >= 0


async def test_finish_turn_after_an_input_reply_has_no_output_attempts():
    engine = make_engine(
        inputs=[ScriptedOrbital(Verdict.respond("FAQ", escalate=True), name="faq")]
    )
    turn = await start(engine)
    report = await engine.finish_turn(turn)
    assert report.final_action == "orbital_response"
    assert report.output == ()
    assert report.escalated is True


async def test_finish_turn_guards_its_preconditions():
    engine = make_engine()
    turn = await start(engine)
    with pytest.raises(RuntimeError, match="no final reply"):
        await engine.finish_turn(turn)
    await engine.run_output(turn, reply="ok", history=OUT_HISTORY)
    await engine.finish_turn(turn)
    with pytest.raises(RuntimeError, match="already"):
        await engine.finish_turn(turn)


async def test_on_turn_receives_the_report_sync_and_async():
    received: list[TurnReport] = []

    async def async_sink(report: TurnReport) -> None:
        received.append(report)

    for sink in (received.append, async_sink):
        engine = make_engine(on_turn=sink)
        turn = await start(engine)
        await engine.run_output(turn, reply="ok", history=OUT_HISTORY)
        report = await engine.finish_turn(turn)
        assert received[-1] is report
    assert len(received) == 2


async def test_failing_on_turn_is_logged_and_does_not_break_the_turn(caplog):
    caplog.set_level(logging.ERROR, logger="chatomon")

    def broken(report: TurnReport) -> None:
        raise RuntimeError("database down")

    engine = make_engine(on_turn=broken)
    turn = await start(engine)
    await engine.run_output(turn, reply="ok", history=OUT_HISTORY)
    report = await engine.finish_turn(turn)
    assert report.final_action == "answered"
    assert any("on_turn" in record.getMessage() for record in caplog.records)


async def test_hash_capture_records_digests_only():
    engine = make_engine(content=ContentPolicy("hash"))
    turn = await start(engine)
    await engine.run_output(turn, reply="It is in Rome.", history=OUT_HISTORY)
    report = await engine.finish_turn(turn)
    assert report.user_message == sha256_text(QUESTION)
    assert report.final_reply == sha256_text("It is in Rome.")


async def test_full_capture_applies_the_redactor_everywhere():
    policy = ContentPolicy("full", redactor=lambda text: text.replace("Rome", "[CITY]"))
    engine = make_engine(
        inputs=[ScriptedOrbital(Verdict.allow("mention Rome"), name="i")],
        outputs=[ScriptedOrbital(RuntimeError("Rome timeout"), name="o", on_error="allow")],
        content=policy,
    )
    turn = await start(engine)
    await engine.run_output(turn, reply="It is in Rome.", history=OUT_HISTORY)
    report = await engine.finish_turn(turn)
    assert report.final_reply == "It is in [CITY]."
    assert report.input[0].guidance == "mention [CITY]"
    assert report.output[0][0].error_message == "[CITY] timeout"


async def test_default_capture_keeps_content_out_of_logs_spans_and_reports(caplog, spans):
    secret = "IBAN IT60X0542811101000000123456"
    caplog.set_level(logging.DEBUG, logger="chatomon")
    engine = make_engine(
        inputs=[
            ScriptedOrbital(
                Verdict.allow(f"user shared {secret}", annotations={"raw": secret}), name="a"
            )
        ],
        outputs=[
            ScriptedOrbital(RuntimeError(secret), name="b", on_error="allow"),
            ScriptedOrbital(Verdict.allow(), name="c"),
        ],
        tracer=spans.tracer,
    )
    turn = engine.start_turn()
    await engine.run_input(turn, message=secret)
    await engine.run_output(
        turn, reply=f"Your {secret}", history=(Message(role="user", content=secret),)
    )
    report = await engine.finish_turn(turn)
    assert secret not in report.model_dump_json()
    for record in caplog.records:
        assert secret not in record.getMessage()
        assert secret not in repr(getattr(record, "chatomon", ""))
    for span in spans.exporter.get_finished_spans():
        assert all(secret not in str(value) for value in (span.attributes or {}).values())
        for event in span.events:
            assert all(secret not in str(value) for value in (event.attributes or {}).values())
