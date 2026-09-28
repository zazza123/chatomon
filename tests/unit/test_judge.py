import pytest

from chatomon import ChatomonConfigError, Knowledge
from chatomon.orbitals import JudgeOrbital
from chatomon.orbitals.judge import JudgeDecision
from chatomon.schemas import Message, OrbitalContext, Side
from tests.fakes import FakeConnector


def ctx(
    side: Side = "output", message="We are open 24/7.", history=(), annotations=None
) -> OrbitalContext:
    return OrbitalContext(
        side=side,
        message=message,
        history=tuple(history),
        annotations=annotations or {},
        turn_id="t1",
    )


def judge(bound: FakeConnector | None = None, **overrides) -> JudgeOrbital:
    settings = {
        "model": "openai:gpt-5-mini",
        "instructions": "Block replies that contradict the reference.",
        "context": Knowledge("Opening hours: 9-18."),
    }
    settings.update(overrides)
    orb = JudgeOrbital(**settings)
    if bound is not None:
        orb.bind(bound)
    return orb


async def test_check_calls_the_model_and_returns_the_verdict():
    connector = FakeConnector(
        [{"action": "block", "guidance": "Hours are 9-18.", "reason": "wrong_hours"}]
    )
    verdict = await judge(connector).check(ctx())
    assert verdict.action == "block"
    assert verdict.guidance == "Hours are 9-18."
    assert verdict.reason == "judge_block"
    assert verdict.annotations["label"] == "wrong_hours"
    call = connector.calls[0]
    assert call["model"] == "openai:gpt-5-mini"
    assert call["schema"] is JudgeDecision


def test_system_prompt_contains_frame_instructions_and_context():
    orb = judge()
    output_prompt = orb.system_prompt("output")
    assert "a reply written by the assistant" in output_prompt
    assert "Block replies that contradict the reference." in output_prompt
    assert "<knowledge>\nOpening hours: 9-18.\n</knowledge>" in output_prompt
    assert 'You may set "guidance"' not in output_prompt
    assert 'You may set "guidance"' in orb.system_prompt("input")
    assert "a message written by the user" in orb.system_prompt("input")


def test_user_prompt_has_a_history_window_annotations_and_the_text():
    history = [Message(role="user", content=f"m{i}") for i in range(10)]
    orb = judge(history_window=2, reads=["claims"])
    prompt = orb.user_prompt(
        ctx(history=history, annotations={"claims": {"claims": ["open 24/7"]}})
    )
    assert "user: m8\nuser: m9" in prompt
    assert "m7" not in prompt
    assert '<annotations from="claims">\n{"claims":["open 24/7"]}\n</annotations>' in prompt
    assert prompt.endswith("<text_under_review>\nWe are open 24/7.\n</text_under_review>")


def test_history_window_zero_sends_no_history():
    prompt = judge(history_window=0).user_prompt(ctx(history=[Message(role="user", content="old")]))
    assert "<history>" not in prompt


@pytest.mark.parametrize(
    ("decision", "action", "reply", "guidance"),
    [
        ({"action": "allow", "reason": "ok"}, "allow", None, None),
        ({"action": "allow", "guidance": "be brief", "reason": "ok"}, "allow", None, "be brief"),
        ({"action": "block", "reply": "No.", "reason": "off"}, "block", "No.", None),
        (
            {"action": "respond", "reply": "We open at 9.", "reason": "faq"},
            "respond",
            "We open at 9.",
            None,
        ),
    ],
)
def test_decisions_map_to_verdicts(decision, action, reply, guidance):
    verdict = JudgeDecision.model_validate(decision).to_verdict()
    assert (verdict.action, verdict.reply, verdict.guidance) == (action, reply, guidance)


def test_explanation_becomes_annotations():
    verdict = JudgeDecision(
        action="allow", reason="ok", explanation="all claims supported"
    ).to_verdict()
    assert verdict.annotations == {"label": "ok", "explanation": "all claims supported"}


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "respond", "reason": "faq"},
        {"action": "block", "reply": "No.", "guidance": "Decline.", "reason": "off"},
    ],
)
async def test_invalid_decision_raises_so_the_engine_fails_closed(decision):
    with pytest.raises(ValueError):
        await judge(FakeConnector([decision])).check(ctx())


async def test_unbound_judge_is_a_config_error():
    with pytest.raises(ChatomonConfigError, match="no connector"):
        await judge().check(ctx())


async def test_explicit_connector_wins_over_the_bound_one():
    own = FakeConnector([{"action": "allow", "reason": "ok"}])
    orb = judge(connector=own)
    orb.bind(FakeConnector())
    await orb.check(ctx())
    assert len(own.calls) == 1


async def test_check_adds_an_evaluation_event_to_the_current_span(spans):
    orb = judge(
        FakeConnector(
            [{"action": "block", "reply": "No.", "reason": "off", "explanation": "secret"}]
        )
    )
    with spans.tracer.start_as_current_span("chatomon.orbital judge"):
        await orb.check(ctx())
    (span,) = spans.exporter.get_finished_spans()
    (event,) = span.events
    assert event.name == "gen_ai.evaluation.result"
    assert event.attributes["gen_ai.evaluation.name"] == "judge"
    assert event.attributes["gen_ai.evaluation.score.label"] == "block"
    assert event.attributes["gen_ai.request.model"] == "openai:gpt-5-mini"
    assert "secret" not in str(dict(event.attributes))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model": " "}, "model"),
        ({"instructions": ""}, "instructions"),
        ({"history_window": -1}, "history_window"),
    ],
)
def test_invalid_settings_are_rejected(overrides, match):
    with pytest.raises(ChatomonConfigError, match=match):
        judge(**overrides)


def test_fingerprint_changes_with_the_instructions():
    assert judge().fingerprint() == judge().fingerprint()
    assert judge().fingerprint() != judge(instructions="Something else.").fingerprint()


async def test_model_written_reason_is_kept_out_of_the_audit_reason():
    secret = "user shared IBAN IT60X0542811101000000123456"
    connector = FakeConnector([{"action": "block", "reply": "No.", "reason": secret}])
    verdict = await judge(connector).check(ctx())
    assert verdict.reason == "judge_block"
    assert verdict.annotations["label"] == secret


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (
            {"action": "block", "reply": "", "guidance": "Fix it.", "reason": "x"},
            ("block", None, "Fix it."),
        ),
        ({"action": "allow", "guidance": "  ", "reason": "x"}, ("allow", None, None)),
    ],
)
def test_blank_fields_from_the_model_are_ignored(decision, expected):
    verdict = JudgeDecision.model_validate(decision).to_verdict()
    assert (verdict.action, verdict.reply, verdict.guidance) == expected


def test_whitespace_only_respond_is_invalid():
    with pytest.raises(ValueError):
        JudgeDecision(action="respond", reply="   ", reason="x").to_verdict()
