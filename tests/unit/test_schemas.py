from datetime import UTC, datetime

import pytest

from chatomon import OrbitalContext, TurnReport, Verdict
from chatomon.schemas import (
    DEFAULT_BLOCKED_GUIDANCE,
    DEFAULT_BLOCKED_PLACEHOLDER,
    DEFAULT_REFUSAL,
    InputDecision,
    Message,
    OrbitalResult,
    OutputAttempt,
    OutputDecision,
)


def test_defaults_match_the_spec():
    assert DEFAULT_REFUSAL == "I'm sorry, I can't help with that."
    assert DEFAULT_BLOCKED_PLACEHOLDER == "[message removed by chatomon]"


def test_the_default_blocked_guidance_is_the_wording_measured_on_real_models():
    assert DEFAULT_BLOCKED_GUIDANCE == (
        "The user's last message is withheld from you and a placeholder stands in its place. "
        "Do not mention the placeholder or that a message was removed: reply as the guidance says."
    )


def test_allow_carries_optional_guidance_and_annotations():
    verdict = Verdict.allow("be brief", reason="chit_chat", annotations={"k": 1})
    assert verdict.action == "allow"
    assert verdict.guidance == "be brief"
    assert verdict.reply is None
    assert verdict.reason == "chit_chat"
    assert verdict.annotations == {"k": 1}
    assert verdict.escalate is False


def test_block_takes_a_reply_or_guidance_but_not_both():
    assert Verdict.block("no").reply == "no"
    assert Verdict.block(guidance="decline").guidance == "decline"
    assert Verdict.block().reply is None
    with pytest.raises(ValueError, match="not both"):
        Verdict.block("no", guidance="decline")


def test_respond_needs_text_and_can_escalate():
    verdict = Verdict.respond("A human will call you.", escalate=True)
    assert verdict.action == "respond"
    assert verdict.reply == "A human will call you."
    assert verdict.escalate is True
    with pytest.raises(ValueError, match="non-empty reply"):
        Verdict.respond("")


def test_invalid_combinations_are_rejected():
    with pytest.raises(ValueError, match="cannot carry a reply"):
        Verdict(action="allow", reply="x")
    with pytest.raises(ValueError, match="cannot carry guidance"):
        Verdict(action="respond", reply="x", guidance="y")


def test_verdict_is_immutable():
    verdict = Verdict.allow()
    with pytest.raises(ValueError):
        verdict.action = "block"  # type: ignore[misc]


def test_orbital_context_defaults():
    ctx = OrbitalContext(side="input", message="hi", turn_id="t1")
    assert ctx.history == ()
    assert ctx.annotations == {}
    assert ctx.attempt == 1
    assert ctx.conversation_id is None


def test_orbital_context_keeps_history_as_messages():
    ctx = OrbitalContext(
        side="output",
        message="reply",
        history=(Message(role="user", content="q"),),
        turn_id="t1",
    )
    assert ctx.history[0].role == "user"


def test_decisions_validate_their_payload():
    assert InputDecision(kind="proceed").guidance == ()
    assert InputDecision(kind="reply", reply="x").reply == "x"
    with pytest.raises(ValueError, match="needs a reply"):
        InputDecision(kind="reply")
    with pytest.raises(ValueError, match="needs guidance"):
        InputDecision(kind="guided_refusal")
    assert OutputDecision(kind="accept").text is None
    with pytest.raises(ValueError, match="needs text"):
        OutputDecision(kind="replace")
    with pytest.raises(ValueError, match="needs guidance"):
        OutputDecision(kind="regenerate")


def test_turn_report_round_trips_to_json():
    result = OrbitalResult(
        name="scope_guard",
        side="input",
        kind="ScopeGuardOrbital",
        fingerprint="sha256:abc",
        action="allow",
        duration_ms=1.5,
    )
    report = TurnReport(
        turn_id="t1",
        conversation_id=None,
        atom_name="speedypack",
        started_at=datetime(2026, 9, 28, tzinfo=UTC),
        duration_ms=10.0,
        final_action="answered",
        escalated=False,
        input=(result,),
        output=(
            OutputAttempt(results=(), decision="regenerate"),
            OutputAttempt(nucleus_reply="sha256:def", results=(result,), decision="accept"),
        ),
    )
    assert TurnReport.model_validate_json(report.model_dump_json()) == report
    first, second = report.model_dump(mode="json")["output"]
    assert first == {"nucleus_reply": None, "results": [], "decision": "regenerate"}
    assert second["nucleus_reply"] == "sha256:def"
    assert [r["name"] for r in second["results"]] == ["scope_guard"]
    assert second["decision"] == "accept"


def test_output_attempt_defaults_and_is_immutable():
    attempt = OutputAttempt(results=())
    assert (attempt.nucleus_reply, attempt.decision) == (None, None)
    with pytest.raises(ValueError, match="frozen"):
        attempt.decision = "accept"  # type: ignore[misc]


def test_output_attempt_takes_only_a_decision_of_the_engine():
    with pytest.raises(ValueError, match="decision"):
        OutputAttempt(results=(), decision="sent")  # type: ignore[arg-type]


def test_respond_rejects_whitespace_only_text():
    with pytest.raises(ValueError, match="non-empty reply"):
        Verdict.respond("   ")
