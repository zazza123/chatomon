"""Regression tests for the minor findings of the Plan 1 final review."""

import importlib
import importlib.util
import sys

import pytest

from chatomon import Atom, ChatomonConfigError, MissingExtraError, Verdict
from chatomon.observability import ContentPolicy
from chatomon.orbitals import JudgeOrbital
from tests.fakes import FakeConnector, ScriptedOrbital, make_engine

QUESTION = "Where is my parcel?"


def test_none_mode_never_serializes_annotations():
    circular: list = []
    circular.append(circular)
    assert ContentPolicy().data(circular) is None


def test_failing_redactor_records_a_marker_instead_of_raising():
    def broken(text: str) -> str:
        raise RuntimeError("redactor down")

    assert ContentPolicy("full", redactor=broken).text("secret") == "[redaction failed]"


@pytest.mark.parametrize("returned", [None, 42, b"bytes", ["a", "list"]])
def test_a_redactor_that_returns_no_text_records_a_marker(returned):
    policy = ContentPolicy("full", redactor=lambda text: returned)
    assert policy.text("secret") == "[redaction failed]"
    assert policy.data({"k": "secret"}) == "[redaction failed]"


async def test_a_redactor_that_returns_no_text_does_not_cost_the_turn_its_report():
    engine = make_engine(
        inputs=[ScriptedOrbital(Verdict.allow("be brief"), name="i")],
        outputs=[ScriptedOrbital(Verdict.block(guidance="Shorter."), Verdict.allow(), name="o")],
        content=ContentPolicy("full", redactor=lambda text: 42),  # type: ignore[arg-type,return-value]
    )
    turn = engine.start_turn()
    await engine.run_input(turn, message="hello")
    await engine.run_output(turn, reply="a long reply")
    await engine.run_output(turn, reply="short")
    report = await engine.finish_turn(turn)
    assert report.final_action == "answered"
    marker = "[redaction failed]"
    assert (report.user_message, report.final_reply) == (marker, marker)
    assert report.input[0].guidance == marker
    assert [attempt.nucleus_reply for attempt in report.output] == [marker, marker]
    assert report.output[0].results[0].guidance == marker


def test_unserializable_annotations_record_a_marker_instead_of_raising():
    circular: list = []
    circular.append(circular)
    assert ContentPolicy("full").data(circular) == "[unserializable]"


def test_a_judge_cannot_be_bound_to_two_different_connectors():
    judge = JudgeOrbital(model="m", instructions="Judge.")
    first, second = FakeConnector(), FakeConnector()
    Atom(name="a", instructions="i", connector=first, output_orbitals=[judge]).build()
    Atom(name="b", instructions="i", connector=first, output_orbitals=[judge]).build()
    with pytest.raises(ChatomonConfigError, match="already bound"):
        Atom(name="c", instructions="i", connector=second, output_orbitals=[judge]).build()


def test_broken_orbitals_dependency_is_not_reported_as_missing_extra(monkeypatch):
    monkeypatch.delitem(sys.modules, "chatomon.orbitals.principled", raising=False)
    monkeypatch.setitem(sys.modules, "orbitals.scope_guard_v2", None)
    with pytest.raises(ImportError) as info:
        importlib.import_module("chatomon.orbitals.principled")
    assert not isinstance(info.value, MissingExtraError)


def test_absent_orbitals_package_is_reported_as_missing_extra(monkeypatch):
    monkeypatch.delitem(sys.modules, "chatomon.orbitals.principled", raising=False)
    monkeypatch.setitem(sys.modules, "orbitals.scope_guard_v2", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "orbitals" else real_find_spec(name, *args),
    )
    with pytest.raises(MissingExtraError, match="chatomon\\[principled\\]"):
        importlib.import_module("chatomon.orbitals.principled")
