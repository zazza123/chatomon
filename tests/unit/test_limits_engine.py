import logging

import pytest

from chatomon import Atom, ChatomonConfigError
from chatomon.observability import ContentPolicy
from chatomon.orbitals import MessageLengthOrbital, TurnLimitOrbital
from chatomon.schemas import DEFAULT_REFUSAL, Message
from tests.fakes import FakeConnector, make_engine

LONG = "one two three four"
LENGTH_REPLY = "Your message is too long. Please shorten it and try again."
TURN_REPLY = "This conversation has reached its limit. Please start a new one."
ONE_TURN = (Message(role="user", content="hi"), Message(role="assistant", content="hello"))


async def test_long_message_ends_the_turn_blocked():
    engine = make_engine(inputs=[MessageLengthOrbital(max_words=3)])
    turn = engine.start_turn()
    decision = await engine.run_input(turn, message=LONG)
    assert (decision.kind, decision.reply, decision.redact_user_message) == (
        "reply",
        LENGTH_REPLY,
        True,
    )
    report = await engine.finish_turn(turn)
    assert report.final_action == "blocked"
    assert report.input[0].reason == "message_too_long"


async def test_long_reply_regenerates_then_falls_back_to_the_refusal():
    engine = make_engine(outputs=[MessageLengthOrbital(max_words=3)], max_retries=1)
    turn = engine.start_turn()
    await engine.run_input(turn, message="hi")
    first = await engine.run_output(turn, reply=LONG)
    assert first.kind == "regenerate"
    assert first.guidance == ("Your previous reply was too long. Keep it to at most 3 words.",)
    second = await engine.run_output(turn, reply=LONG)
    assert (second.kind, second.text) == ("replace", DEFAULT_REFUSAL)
    report = await engine.finish_turn(turn)
    assert report.final_action == "fallback_refusal"
    assert [attempt.results[0].reason for attempt in report.output] == [
        "reply_too_long",
        "reply_too_long",
    ]


async def test_long_reply_without_retries_is_refused_at_once():
    engine = make_engine(outputs=[MessageLengthOrbital(max_words=3)], max_retries=0)
    turn = engine.start_turn()
    await engine.run_input(turn, message="hi")
    decision = await engine.run_output(turn, reply=LONG)
    assert (decision.kind, decision.text) == ("replace", DEFAULT_REFUSAL)
    report = await engine.finish_turn(turn)
    assert report.final_action == "fallback_refusal"
    assert len(report.output) == 1


async def test_shorter_regenerated_reply_is_answered():
    engine = make_engine(outputs=[MessageLengthOrbital(max_words=3)])
    turn = engine.start_turn()
    await engine.run_input(turn, message="hi")
    assert (await engine.run_output(turn, reply=LONG)).kind == "regenerate"
    assert (await engine.run_output(turn, reply="short")).kind == "accept"
    assert (await engine.finish_turn(turn)).final_action == "answered"


async def test_turn_limit_answers_without_the_nucleus_and_keeps_the_message():
    engine = make_engine(inputs=[TurnLimitOrbital(max_turns=1)])
    turn = engine.start_turn()
    decision = await engine.run_input(turn, message="again", history=ONE_TURN)
    assert (decision.kind, decision.reply, decision.redact_user_message) == (
        "reply",
        TURN_REPLY,
        False,
    )
    report = await engine.finish_turn(turn)
    assert report.final_action == "orbital_response"
    assert report.input[0].reason == "turn_limit"


async def test_turn_warning_reaches_the_nucleus_as_guidance():
    engine = make_engine(inputs=[TurnLimitOrbital(max_turns=2, warn_before=1)])
    turn = engine.start_turn()
    decision = await engine.run_input(turn, message="again", history=ONE_TURN)
    assert decision.kind == "proceed"
    assert decision.guidance == ("The conversation will end soon: help the user wrap up.",)


async def test_turn_limit_stops_the_chain_before_later_orbitals():
    later = MessageLengthOrbital(max_words=1)
    engine = make_engine(inputs=[TurnLimitOrbital(max_turns=1), later])
    turn = engine.start_turn()
    await engine.run_input(turn, message="two words", history=ONE_TURN)
    report = await engine.finish_turn(turn)
    assert [result.name for result in report.input] == ["turn_limit"]


def test_one_length_orbital_can_guard_both_sides():
    limit = MessageLengthOrbital(max_chars=10)
    atom = Atom(
        name="a",
        instructions="Be brief.",
        connector=FakeConnector(),
        input_orbitals=[limit],
        output_orbitals=[limit],
    )
    atom.validate()


async def test_one_length_orbital_reports_each_side_with_its_reason():
    limit = MessageLengthOrbital(max_words=3)
    engine = make_engine(inputs=[limit], outputs=[limit], max_retries=0)
    turn = engine.start_turn()
    await engine.run_input(turn, message="hi")
    await engine.run_output(turn, reply=LONG)
    report = await engine.finish_turn(turn)
    assert (report.input[0].action, report.input[0].reason) == ("allow", None)
    assert report.output[0].results[0].reason == "reply_too_long"


def test_turn_limit_cannot_guard_the_output():
    atom = Atom(
        name="a",
        instructions="Be brief.",
        connector=FakeConnector(),
        output_orbitals=[TurnLimitOrbital(max_turns=3)],
    )
    with pytest.raises(ChatomonConfigError, match="cannot run on the output side"):
        atom.validate()


async def test_reasons_reach_the_spans_without_content(spans):
    engine = make_engine(
        inputs=[TurnLimitOrbital(max_turns=1, reply="Goodbye for now.")], tracer=spans.tracer
    )
    turn = engine.start_turn()
    await engine.run_input(turn, message="secret question", history=ONE_TURN)
    await engine.finish_turn(turn)
    finished = spans.exporter.get_finished_spans()
    orbital_span = next(s for s in finished if s.name == "chatomon.orbital turn_limit")
    assert orbital_span.attributes["chatomon.verdict.reason"] == "turn_limit"
    assert orbital_span.attributes["chatomon.verdict.action"] == "respond"
    values = [str(value) for span in finished for value in (span.attributes or {}).values()]
    assert not any("secret" in value or "Goodbye" in value for value in values)


async def test_nothing_but_reasons_is_recorded_by_default(caplog):
    caplog.set_level(logging.DEBUG, logger="chatomon")
    engine = make_engine(inputs=[MessageLengthOrbital(max_words=3)])
    turn = engine.start_turn()
    await engine.run_input(turn, message=LONG)
    report = await engine.finish_turn(turn)
    result = report.input[0]
    assert result.reason == "message_too_long"
    assert (result.annotations, result.reply, result.guidance) == (None, None, None)
    assert (report.user_message, report.final_reply) == (None, None)
    for record in caplog.records:
        dumped = str(record.__dict__)
        assert LONG not in dumped
        assert LENGTH_REPLY not in dumped
        assert "'words'" not in dumped


async def test_full_capture_records_the_measurements_and_texts():
    engine = make_engine(inputs=[MessageLengthOrbital(max_words=3)], content=ContentPolicy("full"))
    turn = engine.start_turn()
    await engine.run_input(turn, message=LONG)
    result = (await engine.finish_turn(turn)).input[0]
    assert result.annotations == '{"chars":18,"words":4}'
    assert result.reply == LENGTH_REPLY
