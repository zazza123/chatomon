"""What the counts say after requests that ended in every way."""

from datetime import UTC, datetime
from typing import Any

from chatomon.schemas import OrbitalResult, OutputAttempt, TurnReport
from chatomon.studio._spectral._counts import Chain, Counts

CHAIN: Chain = {
    "input": [("length", "MessageLengthOrbital"), ("scope", "ScopeGuardOrbital")],
    "output": [("judge", "JudgeOrbital")],
}


def result(name: str, action: str, *, side: str = "input", error: str | None = None) -> Any:
    kind = dict([*CHAIN["input"], *CHAIN["output"]]).get(name, "Orbital")
    return OrbitalResult(
        name=name,
        side=side,  # type: ignore[arg-type]
        kind=kind,
        fingerprint="sha256:0",
        action=action,  # type: ignore[arg-type]
        duration_ms=2.0,
        error_type=error,
    )


def report(final: str, *inputs: Any, attempts: tuple[tuple[Any, ...], ...] = ()) -> TurnReport:
    return TurnReport(
        turn_id="t",
        conversation_id=None,
        atom_name="shop",
        started_at=datetime.now(UTC),
        duration_ms=10.0,
        final_action=final,  # type: ignore[arg-type]
        escalated=False,
        input=inputs,
        output=tuple(OutputAttempt(results=results) for results in attempts),
    )


def done(
    counts: Counts, turn: TurnReport | None, *, failed: bool = False, revision: int = 1
) -> None:
    counts.arrived()
    counts.done(revision=revision, chain=CHAIN, report=turn, failed=failed, duration_ms=100.0)


def rows(counts: Counts) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in counts.to_json()["orbitals"]}


def test_new_counts_are_all_zero():
    assert Counts().to_json() == {
        "requests": 0,
        "running": 0,
        "failed": 0,
        "rejected": 0,
        "duration_ms_total": 0.0,
        "configurations": 0,
        "final": {
            "answered": 0,
            "guided_refusal": 0,
            "orbital_response": 0,
            "blocked": 0,
            "fallback_refusal": 0,
        },
        "orbitals": [],
    }


def test_a_request_is_running_until_it_is_done():
    counts = Counts()
    counts.arrived()
    assert (counts.to_json()["requests"], counts.to_json()["running"]) == (1, 1)
    counts.done(revision=1, chain=CHAIN, report=None, failed=True, duration_ms=5.0)
    after = counts.to_json()
    assert (after["requests"], after["running"], after["failed"]) == (1, 0, 1)


def test_an_answered_request_counts_its_final_action_and_every_orbital():
    counts = Counts()
    answered = report(
        "answered",
        result("length", "allow"),
        result("scope", "allow"),
        attempts=((result("judge", "allow", side="output"),),),
    )
    done(counts, answered)
    data = counts.to_json()
    assert data["final"]["answered"] == 1
    assert data["duration_ms_total"] == 100.0
    assert [(row["name"], row["side"], row["kind"]) for row in data["orbitals"]] == [
        ("length", "input", "MessageLengthOrbital"),
        ("scope", "input", "ScopeGuardOrbital"),
        ("judge", "output", "JudgeOrbital"),
    ]
    assert all((row["allow"], row["runs"], row["not_run"]) == (1, 1, 0) for row in data["orbitals"])
    assert data["orbitals"][0]["duration_ms_total"] == 2.0


def test_an_orbital_that_did_not_run_is_counted_as_not_run():
    counts = Counts()
    done(counts, report("blocked", result("length", "block")))
    assert counts.to_json()["final"]["blocked"] == 1
    assert (rows(counts)["length"]["block"], rows(counts)["length"]["not_run"]) == (1, 0)
    assert (rows(counts)["scope"]["runs"], rows(counts)["scope"]["not_run"]) == (0, 1)
    assert (rows(counts)["judge"]["runs"], rows(counts)["judge"]["not_run"]) == (0, 1)


def test_an_orbital_that_failed_is_an_error_whatever_was_applied():
    counts = Counts()
    failed = result("scope", "block", error="TimeoutError")
    done(counts, report("blocked", result("length", "allow"), failed))
    scope = rows(counts)["scope"]
    assert (scope["error"], scope["block"], scope["runs"]) == (1, 0, 1)


def test_an_output_orbital_runs_once_for_every_attempt():
    counts = Counts()
    judge_block = result("judge", "block", side="output")
    judge_allow = result("judge", "allow", side="output")
    twice = report(
        "answered",
        result("length", "allow"),
        result("scope", "allow"),
        attempts=((judge_block,), (judge_allow,)),
    )
    done(counts, twice)
    judge = rows(counts)["judge"]
    assert (judge["block"], judge["allow"], judge["runs"], judge["not_run"]) == (1, 1, 2, 0)
    assert counts.to_json()["requests"] == 1


def test_an_orbital_that_answers_is_counted_as_a_respond():
    counts = Counts()
    done(counts, report("orbital_response", result("length", "allow"), result("scope", "respond")))
    assert rows(counts)["scope"]["respond"] == 1
    assert counts.to_json()["final"]["orbital_response"] == 1


def test_a_failed_request_keeps_what_its_orbitals_did_and_adds_no_final_action():
    counts = Counts()
    done(
        counts, report("aborted", result("length", "allow"), result("scope", "allow")), failed=True
    )
    data = counts.to_json()
    assert data["failed"] == 1
    assert sum(data["final"].values()) == 0
    assert rows(counts)["length"]["allow"] == 1
    assert rows(counts)["judge"]["not_run"] == 1


def test_an_answer_that_was_sent_as_an_error_is_not_counted_as_answered():
    """A reply that came back empty is a failed request, whatever its report says."""
    counts = Counts()
    done(counts, report("answered", result("length", "allow")), failed=True)
    assert (counts.to_json()["failed"], counts.to_json()["final"]["answered"]) == (1, 0)


def test_a_request_without_a_report_counts_every_orbital_as_not_run():
    counts = Counts()
    done(counts, None, failed=True)
    assert [row["not_run"] for row in counts.to_json()["orbitals"]] == [1, 1, 1]


def test_the_counts_say_how_many_configurations_answered():
    counts = Counts()
    done(counts, report("answered"), revision=3)
    done(counts, report("answered"), revision=3)
    assert counts.to_json()["configurations"] == 1
    done(counts, report("answered"), revision=4)
    assert counts.to_json()["configurations"] == 2


def test_a_request_refused_for_its_shape_is_counted_apart():
    counts = Counts()
    counts.rejected()
    data = counts.to_json()
    assert (data["rejected"], data["requests"], data["failed"]) == (1, 0, 0)


def test_an_orbital_that_is_no_longer_in_the_chain_keeps_its_row():
    counts = Counts()
    done(counts, report("answered", result("length", "allow"), result("scope", "allow")))
    counts.arrived()
    shorter: Chain = {"input": [("length", "MessageLengthOrbital")], "output": []}
    turn = report("answered", result("length", "allow"))
    counts.done(revision=2, chain=shorter, report=turn, failed=False, duration_ms=1.0)
    assert (rows(counts)["scope"]["allow"], rows(counts)["scope"]["not_run"]) == (1, 0)
    assert rows(counts)["length"]["allow"] == 2
