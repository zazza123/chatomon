"""The attempt records of a turn report: one per reply the output chain checked."""

import asyncio
import logging

import pytest

from chatomon import TurnReport, Verdict
from chatomon.engine import Engine, TurnState
from chatomon.observability import REDACTION_FAILED, ContentPolicy, sha256_text
from chatomon.orbitals.base import Orbital
from chatomon.schemas import DEFAULT_REFUSAL, OrbitalContext
from tests.fakes import ScriptedOrbital, make_engine

QUESTION = "Where is my parcel?"
AGAIN = Verdict.block(guidance="Fix the opening hours.")


async def start(engine: Engine) -> TurnState:
    turn = engine.start_turn()
    await engine.run_input(turn, message=QUESTION)
    return turn


def replies(report: TurnReport) -> list[str | None]:
    return [attempt.nucleus_reply for attempt in report.output]


def decisions(report: TurnReport) -> list[str | None]:
    return [attempt.decision for attempt in report.output]


class HangingOrbital(Orbital):
    """Starts and never answers, so the test can cancel the turn while it runs."""

    calls_model = False

    def __init__(self) -> None:
        super().__init__(name="hanging")
        self.started = asyncio.Event()

    async def check(self, ctx: OrbitalContext) -> Verdict:
        self.started.set()
        await asyncio.Event().wait()
        return Verdict.allow()


async def test_every_attempt_is_in_the_report_with_its_reply_and_decision():
    judge = ScriptedOrbital(AGAIN, AGAIN, Verdict.allow(), name="judge")
    engine = make_engine(outputs=[judge], content=ContentPolicy("full"))
    turn = await start(engine)
    for reply in ("Open 24/7.", "Open 9-18, 5 euro.", "Open 9-18."):
        await engine.run_output(turn, reply=reply)
    report = await engine.finish_turn(turn)
    assert replies(report) == ["Open 24/7.", "Open 9-18, 5 euro.", "Open 9-18."]
    assert decisions(report) == ["regenerate", "regenerate", "accept"]
    assert [[r.action for r in attempt.results] for attempt in report.output] == [
        ["block"],
        ["block"],
        ["allow"],
    ]
    assert report.final_reply == "Open 9-18."


async def test_an_attempt_has_the_number_its_results_carry():
    judge = ScriptedOrbital(AGAIN, AGAIN, Verdict.allow(), name="judge")
    engine = make_engine(outputs=[ScriptedOrbital(name="first"), judge])
    turn = await start(engine)
    for reply in ("one", "two", "three"):
        await engine.run_output(turn, reply=reply)
    report = await engine.finish_turn(turn)
    assert [[r.attempt for r in attempt.results] for attempt in report.output] == [
        [1, 1],
        [2, 2],
        [3, 3],
    ]


async def test_default_capture_records_no_reply_of_any_attempt(caplog, spans):
    secret = "IBAN IT60X0542811101000000123456"
    caplog.set_level(logging.DEBUG, logger="chatomon")
    judge = ScriptedOrbital(AGAIN, Verdict.allow(), name="judge")
    engine = make_engine(outputs=[judge], tracer=spans.tracer)
    turn = await start(engine)
    await engine.run_output(turn, reply=f"Your {secret}")
    await engine.run_output(turn, reply="Ask your bank.")
    report = await engine.finish_turn(turn)
    assert replies(report) == [None, None]
    # Labels are not content: they are recorded with any capture mode.
    assert decisions(report) == ["regenerate", "accept"]
    assert secret not in report.model_dump_json()
    assert secret not in repr(turn)
    for record in caplog.records:
        assert secret not in record.getMessage()
        assert secret not in repr(getattr(record, "chatomon", ""))
    for span in spans.exporter.get_finished_spans():
        assert all(secret not in str(value) for value in (span.attributes or {}).values())
        for event in span.events:
            assert all(secret not in str(value) for value in (event.attributes or {}).values())


async def test_hash_capture_records_one_digest_per_attempt():
    judge = ScriptedOrbital(AGAIN, AGAIN, Verdict.allow(), name="judge")
    engine = make_engine(outputs=[judge], content=ContentPolicy("hash"))
    turn = await start(engine)
    for reply in ("Closed on Mondays.", "Closed on Mondays.", "Open 9-18."):
        await engine.run_output(turn, reply=reply)
    report = await engine.finish_turn(turn)
    # The same reply twice gives the same digest: attempts can be compared without their text.
    assert replies(report) == [
        sha256_text("Closed on Mondays."),
        sha256_text("Closed on Mondays."),
        sha256_text("Open 9-18."),
    ]
    assert report.final_reply == sha256_text("Open 9-18.")
    assert "Mondays" not in report.model_dump_json()
    # The state of the turn keeps no plain text of a reply that was not sent.
    assert "Mondays" not in repr(turn)


async def test_full_capture_passes_every_reply_through_the_redactor():
    policy = ContentPolicy("full", redactor=lambda text: text.replace("Rome", "[CITY]"))
    judge = ScriptedOrbital(AGAIN, Verdict.allow(), name="judge")
    engine = make_engine(outputs=[judge], content=policy)
    turn = await start(engine)
    await engine.run_output(turn, reply="It left Rome.")
    await engine.run_output(turn, reply="It is in Rome.")
    report = await engine.finish_turn(turn)
    assert replies(report) == ["It left [CITY].", "It is in [CITY]."]
    assert "Rome" not in report.model_dump_json()


async def test_a_redactor_that_fails_on_a_rejected_reply_leaves_the_marker_and_the_turn_goes_on():
    def broken(text: str) -> str:
        raise ValueError(text)

    judge = ScriptedOrbital(AGAIN, Verdict.allow(), name="judge")
    engine = make_engine(outputs=[judge], content=ContentPolicy("full", redactor=broken))
    turn = await start(engine)
    first = await engine.run_output(turn, reply="Open 24/7.")
    assert first.kind == "regenerate"
    second = await engine.run_output(turn, reply="Open 9-18.")
    assert second.kind == "accept"
    report = await engine.finish_turn(turn)
    assert report.final_action == "answered"
    assert replies(report) == [REDACTION_FAILED, REDACTION_FAILED]
    assert decisions(report) == ["regenerate", "accept"]


@pytest.mark.parametrize(("mode", "recorded"), [("full", ""), ("hash", sha256_text(""))])
async def test_an_empty_reply_is_recorded_as_empty_not_as_missing(mode, recorded):
    engine = make_engine(content=ContentPolicy(mode))
    turn = await start(engine)
    await engine.run_output(turn, reply="")
    report = await engine.finish_turn(turn)
    assert replies(report) == [recorded]


@pytest.mark.parametrize(
    ("verdict", "final_action", "sent"),
    [
        (Verdict.respond("Call us."), "orbital_response", "Call us."),
        (Verdict.block("Not allowed."), "blocked", "Not allowed."),
        (Verdict.block(), "blocked", DEFAULT_REFUSAL),
    ],
)
async def test_a_replaced_attempt_keeps_what_the_nucleus_wrote(verdict, final_action, sent):
    engine = make_engine(
        outputs=[ScriptedOrbital(verdict, name="o")], content=ContentPolicy("full")
    )
    turn = await start(engine)
    await engine.run_output(turn, reply="draft")
    report = await engine.finish_turn(turn)
    assert report.final_action == final_action
    assert (replies(report), decisions(report)) == (["draft"], ["replace"])
    # The text that replaced it is where it already was.
    assert report.final_reply == sent
    assert report.output[0].results[0].reply == verdict.reply


async def test_retries_used_up_end_with_a_replaced_attempt():
    engine = make_engine(
        outputs=[ScriptedOrbital(AGAIN, name="judge")],
        max_retries=1,
        refusal="Sorry.",
        content=ContentPolicy("full"),
    )
    turn = await start(engine)
    await engine.run_output(turn, reply="Open 24/7.")
    await engine.run_output(turn, reply="Open all day.")
    report = await engine.finish_turn(turn)
    assert report.final_action == "fallback_refusal"
    assert replies(report) == ["Open 24/7.", "Open all day."]
    assert decisions(report) == ["regenerate", "replace"]
    assert report.final_reply == "Sorry."


async def test_an_attempt_with_no_output_orbital_is_recorded_as_accepted():
    engine = make_engine(content=ContentPolicy("full"))
    turn = await start(engine)
    await engine.run_output(turn, reply="It is in Rome.")
    report = await engine.finish_turn(turn)
    assert len(report.output) == 1
    attempt = report.output[0]
    assert (attempt.nucleus_reply, attempt.results, attempt.decision) == (
        "It is in Rome.",
        (),
        "accept",
    )


async def test_a_reopened_turn_keeps_the_attempt_that_was_accepted():
    engine = make_engine(outputs=[ScriptedOrbital(name="out")], content=ContentPolicy("full"))
    turn = await start(engine)
    await engine.run_output(turn, reply="first")
    engine.reopen_output(turn)
    await engine.run_output(turn, reply="second")
    report = await engine.finish_turn(turn)
    assert replies(report) == ["first", "second"]
    assert decisions(report) == ["accept", "accept"]
    assert report.final_reply == "second"


async def test_a_reopened_turn_keeps_the_attempt_that_was_replaced():
    out = ScriptedOrbital(Verdict.respond("Call us."), Verdict.allow(), name="out")
    engine = make_engine(outputs=[out], content=ContentPolicy("full"))
    turn = await start(engine)
    await engine.run_output(turn, reply="draft")
    engine.reopen_output(turn)
    await engine.run_output(turn, reply="second")
    report = await engine.finish_turn(turn)
    assert replies(report) == ["draft", "second"]
    assert decisions(report) == ["replace", "accept"]
    # The text that took the first reply's place did not end the turn: it is in the
    # orbital's result, and the final reply is the one accepted afterwards.
    assert report.output[0].results[0].reply == "Call us."
    assert (report.final_action, report.final_reply) == ("answered", "second")


async def test_a_turn_aborted_after_an_attempt_keeps_the_attempt():
    engine = make_engine(
        outputs=[ScriptedOrbital(AGAIN, name="judge")], content=ContentPolicy("full")
    )
    turn = await start(engine)
    await engine.run_output(turn, reply="Open 24/7.")
    report = await engine.abort_turn(turn, RuntimeError("the model is down"))
    assert report.final_action == "aborted"
    assert (replies(report), decisions(report)) == (["Open 24/7."], ["regenerate"])
    assert report.final_reply is None


async def test_a_turn_interrupted_while_the_chain_runs_records_the_attempt_without_a_decision():
    hanging = HangingOrbital()
    engine = make_engine(
        outputs=[ScriptedOrbital(name="first"), hanging, ScriptedOrbital(name="never")],
        content=ContentPolicy("full"),
    )
    turn = await start(engine)
    running = asyncio.ensure_future(engine.run_output(turn, reply="draft"))
    await hanging.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    report = await engine.abort_turn(turn, asyncio.CancelledError())
    assert len(report.output) == 1
    attempt = report.output[0]
    assert attempt.nucleus_reply == "draft"
    assert [r.name for r in attempt.results] == ["first"]
    assert attempt.decision is None


async def test_a_refused_call_leaves_no_attempt():
    engine = make_engine(content=ContentPolicy("full"))
    turn = engine.start_turn()
    with pytest.raises(RuntimeError, match="run_input before run_output"):
        await engine.run_output(turn, reply="too early")
    assert turn.output_attempts == []
    await engine.run_input(turn, message=QUESTION)
    await engine.run_output(turn, reply="fine")
    with pytest.raises(RuntimeError, match="already has its final reply"):
        await engine.run_output(turn, reply="one more")
    report = await engine.finish_turn(turn)
    assert replies(report) == ["fine"]
