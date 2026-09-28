from chatomon import Verdict
from chatomon.connectors import render_guidance
from chatomon.orbitals import JudgeOrbital
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER as PLACEHOLDER
from chatomon.schemas import Message
from tests.contract.harness import ToolCall, build_agent
from tests.fakes import ScriptedOrbital


async def test_output_respond_replaces_the_reply(harness):
    out = ScriptedOrbital(Verdict.respond("Please call us."), name="out")
    agent, reports = build_agent(harness, ["draft"], output_orbitals=[out])
    assert await harness.ask(agent, "hello") == "Please call us."
    assert harness.history(agent) == [("user", "hello"), ("assistant", "Please call us.")]
    assert reports[0].final_action == "orbital_response"


async def test_regeneration_accumulates_guidance_and_keeps_only_the_final_reply(harness):
    judge = ScriptedOrbital(
        Verdict.block(guidance="Fix the hours."),
        Verdict.block(guidance="No prices."),
        Verdict.allow(),
        name="judge",
    )
    agent, reports = build_agent(
        harness, ["Open 24/7", "Open 9-18, 5 euro", "Open 9-18"], output_orbitals=[judge]
    )
    assert await harness.ask(agent, "hours?") == "Open 9-18"
    assert len(harness.calls) == 3
    assert render_guidance(["Fix the hours.", "No prices."]) in harness.calls[2].system
    assert harness.history(agent) == [("user", "hours?"), ("assistant", "Open 9-18")]
    assert [[r.action for r in attempt] for attempt in reports[0].output] == [
        ["block"],
        ["block"],
        ["allow"],
    ]


async def test_exhausted_retries_fall_back_to_the_refusal(harness):
    judge = ScriptedOrbital(Verdict.block(guidance="Again."), name="judge")
    agent, reports = build_agent(
        harness, ["bad", "bad"], output_orbitals=[judge], max_retries=1, refusal="Sorry."
    )
    assert await harness.ask(agent, "hello") == "Sorry."
    assert len(harness.calls) == 2
    assert reports[0].final_action == "fallback_refusal"


async def test_output_orbitals_see_the_reply_and_the_user_message(harness):
    out = ScriptedOrbital(name="out")
    agent, _ = build_agent(harness, ["It is in Rome."], output_orbitals=[out])
    await harness.ask(agent, "Where is it?")
    ctx = out.contexts[0]
    assert ctx.side == "output"
    assert ctx.message == "It is in Rome."
    assert ctx.history[-1] == Message(role="user", content="Where is it?")


async def test_tool_call_steps_are_not_checked(harness):
    out = ScriptedOrbital(name="out")
    agent, reports = build_agent(
        harness, [ToolCall(), "Open 9-18"], tools=True, output_orbitals=[out]
    )
    assert await harness.ask(agent, "hours?") == "Open 9-18"
    assert [ctx.message for ctx in out.contexts] == ["Open 9-18"]
    assert reports[0].final_action == "answered"


async def test_guided_refusal_reply_goes_through_output_orbitals(harness):
    guard = ScriptedOrbital(Verdict.block(guidance="Decline."), name="guard")
    out = ScriptedOrbital(name="out")
    agent, _ = build_agent(
        harness, ["I can't help with that."], input_orbitals=[guard], output_orbitals=[out]
    )
    await harness.ask(agent, "bad request")
    assert len(out.contexts) == 1
    assert out.contexts[0].history[-1].content == PLACEHOLDER


async def test_judge_orbital_uses_the_atom_connector(harness):
    judge = JudgeOrbital(model="judge-model", instructions="Keep replies short.")
    agent, reports = build_agent(
        harness,
        ["A very long reply", "Short."],
        output_orbitals=[judge],
        judge_decisions=[
            {"action": "block", "guidance": "Shorter.", "reason": "too_long"},
            {"action": "allow", "reason": "ok"},
        ],
    )
    assert await harness.ask(agent, "hello") == "Short."
    assert render_guidance(["Shorter."]) in harness.calls[1].system
    assert reports[0].output[0][0].reason == "judge_block"
