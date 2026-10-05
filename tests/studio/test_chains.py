from typing import Any

import pytest

from chatomon import Atom
from chatomon.orbitals import JudgeOrbital, MessageLengthOrbital
from chatomon.studio._chains import (
    Chains,
    InvalidRequestError,
    RefusedError,
    UnknownOrbitalError,
)
from chatomon.studio._kinds import InvalidSettingError
from tests.fakes import FakeConnector, ScriptedOrbital


def make_atom(**kwargs: Any) -> Atom:
    kwargs.setdefault("connector", FakeConnector())
    return Atom(name="speedypack", instructions="Be brief.", **kwargs)


def names(chains: Chains, side: str = "input") -> list[str]:
    return [entry.orbital.name for entry in chains.work[side]]  # type: ignore[index]


def texts(chains: Chains) -> list[str]:
    return [change.text for change in chains.changes()]


@pytest.fixture
def chains() -> Chains:
    return Chains.from_atom(
        make_atom(
            input_orbitals=[MessageLengthOrbital(max_words=60), ScriptedOrbital(name="scope")],
            output_orbitals=[
                JudgeOrbital(name="faithfulness", model="test:judge", instructions="Check facts.")
            ],
        )
    )


def test_the_chains_start_as_the_code_has_them(chains):
    assert names(chains) == ["message_length", "scope"]
    assert names(chains, "output") == ["faithfulness"]
    for entry in (*chains.work["input"], *chains.work["output"]):
        assert (entry.origin, entry.enabled, chains.modified(entry)) == ("code", True, False)
        assert entry.orbital is entry.base
    assert chains.changes() == []
    limit, scope = chains.work["input"]
    assert chains.note(limit) is None
    assert (
        chains.note(scope) == "ScriptedOrbital is your own class: change its settings in your code."
    )


def test_an_operation_returns_new_chains_and_leaves_the_old_ones(chains):
    before = dict(chains.work)
    changed = chains.switched("input", "scope", False)
    assert changed is not chains
    assert dict(chains.work) == before
    assert [entry.enabled for entry in changed.work["input"]] == [True, False]
    assert [o.name for o in changed.enabled("input")] == ["message_length"]
    assert texts(changed) == ["scope turned off"]
    assert texts(changed.switched("input", "scope", True)) == []


def test_an_edit_builds_a_new_orbital_and_keeps_the_developers(chains):
    original = chains.work["input"][0].orbital
    edited = chains.edited("input", "message_length", {"max_words": 80, "reply": "Shorter."})
    entry = edited.work["input"][0]
    assert entry.orbital is not original and entry.base is original
    assert isinstance(entry.orbital, MessageLengthOrbital)
    assert (entry.orbital.max_words, original.max_words) == (80, 60)  # type: ignore[attr-defined]
    assert edited.modified(entry)
    assert texts(edited) == [
        "message_length.max_words changed",
        "message_length.reply changed",
    ]
    change = edited.changes()[0]
    assert change.to_json() == {
        "type": "setting",
        "side": "input",
        "name": "message_length",
        "setting": "max_words",
        "text": "message_length.max_words changed",
    }


def test_back_to_the_codes_values_is_back_to_the_codes_object(chains):
    original = chains.work["input"][0].orbital
    edited = chains.edited("input", "message_length", {"max_words": 80})
    back = edited.edited("input", "message_length", {"max_words": 60})
    assert back.work["input"][0].orbital is original
    assert back.changes() == []


def test_what_an_edit_may_not_do(chains):
    with pytest.raises(RefusedError, match=r"'scope' is read-only in the studio\. ScriptedOrbital"):
        chains.edited("input", "scope", {"timeout": 5})
    with pytest.raises(RefusedError, match="'message_length' is defined in your code: rename it"):
        chains.edited("input", "message_length", {"name": "len"})
    with pytest.raises(InvalidSettingError, match="set max_words, max_chars or both"):
        chains.edited("input", "message_length", {"max_words": None})
    with pytest.raises(UnknownOrbitalError, match="no input orbital named 'ghost'"):
        chains.edited("input", "ghost", {"timeout": 5})
    with pytest.raises(UnknownOrbitalError, match="no sideways orbital named 'scope'"):
        chains.switched("sideways", "scope", False)


def test_moving_an_orbital(chains):
    moved = chains.moved("input", "scope", 0)
    assert names(moved) == ["scope", "message_length"]
    assert texts(moved) == ["input order changed"]
    assert moved.changes()[0].to_json() == {
        "type": "order",
        "side": "input",
        "text": "input order changed",
    }
    assert texts(moved.moved("input", "scope", 1)) == []
    with pytest.raises(InvalidRequestError, match="position must be between 0 and 1"):
        chains.moved("input", "scope", 2)


def test_adding_an_orbital(chains):
    settings = {"model": "test:judge", "instructions": "Be kind."}
    added = chains.added("output", "JudgeOrbital", settings)
    assert names(added, "output") == ["faithfulness", "judge"]
    entry = added.work["output"][1]
    assert (entry.origin, entry.enabled, entry.base, added.modified(entry)) == (
        "studio",
        True,
        None,
        True,
    )
    assert added.note(entry) is None
    assert texts(added) == ["judge added"]
    again = added.added("input", "JudgeOrbital", settings)
    assert names(again) == ["message_length", "scope", "judge_2"]
    named = chains.added("output", "JudgeOrbital", {**settings, "name": "tone"})
    assert names(named, "output") == ["faithfulness", "tone"]
    assert chains.free_name(MessageLengthOrbital) == "message_length_2"


def test_what_cannot_be_added(chains):
    settings = {"model": "test:judge", "instructions": "Be kind."}
    with pytest.raises(RefusedError, match="FunctionOrbital cannot be added from the studio"):
        chains.added("input", "FunctionOrbital", {})
    with pytest.raises(RefusedError, match="Nope cannot be added from the studio"):
        chains.added("input", "Nope", {})
    with pytest.raises(RefusedError, match="TurnLimitOrbital cannot run on the output side"):
        chains.added("output", "TurnLimitOrbital", {"max_turns": 3})
    with pytest.raises(InvalidRequestError, match="side must be 'input' or 'output'"):
        chains.added("sideways", "JudgeOrbital", settings)
    with pytest.raises(InvalidSettingError, match="`instructions` needs a value"):
        chains.added("output", "JudgeOrbital", {"model": "test:judge"})
    with pytest.raises(InvalidSettingError, match="invalid name 'Tone'"):
        chains.added("output", "JudgeOrbital", {**settings, "name": "Tone"})
    with pytest.raises(
        InvalidSettingError, match="an orbital named 'scope' is already in the atom"
    ):
        chains.added("output", "JudgeOrbital", {**settings, "name": "scope"})


def test_an_added_orbital_can_be_edited_renamed_and_deleted(chains):
    settings = {"model": "test:judge", "instructions": "Be kind."}
    added = chains.added("output", "JudgeOrbital", settings)
    renamed = added.edited("output", "judge", {"name": "tone", "history_window": 2})
    assert names(renamed, "output") == ["faithfulness", "tone"]
    assert texts(renamed) == ["tone added"]
    with pytest.raises(
        InvalidSettingError, match="an orbital named 'scope' is already in the atom"
    ):
        added.edited("output", "judge", {"name": "scope"})
    assert names(renamed.deleted("output", "tone"), "output") == ["faithfulness"]
    with pytest.raises(RefusedError, match="'scope' is defined in your code: turn it off instead"):
        chains.deleted("input", "scope")


def test_resetting_one_orbital_and_all_of_them(chains):
    original = chains.work["input"][0].orbital
    settings = {"model": "test:judge", "instructions": "Be kind."}
    changed = (
        chains.edited("input", "message_length", {"max_words": 80})
        .switched("input", "message_length", False)
        .moved("input", "message_length", 1)
        .added("output", "JudgeOrbital", settings)
    )
    one = changed.reset("input", "message_length")
    entry = one.work["input"][1]
    assert entry.orbital is original and entry.enabled is True
    assert names(one) == ["scope", "message_length"]  # its position stays
    assert texts(one) == ["input order changed", "judge added"]
    with pytest.raises(RefusedError, match="'judge' was added in the studio: delete it instead"):
        changed.reset("output", "judge")
    everything = changed.reset()
    assert everything.changes() == []
    assert names(everything) == ["message_length", "scope"]
    assert names(everything, "output") == ["faithfulness"]
    assert changed.initial().changes() == []


def test_an_instance_in_both_chains_stays_one_orbital():
    both = MessageLengthOrbital(max_words=60)
    chains = Chains.from_atom(make_atom(input_orbitals=[both], output_orbitals=[both]))
    edited = chains.edited("input", "message_length", {"max_words": 80})
    assert edited.work["input"][0].orbital is edited.work["output"][0].orbital
    assert edited.work["output"][0].base is both
    assert texts(edited) == ["message_length.max_words changed"]  # listed once
    off = edited.switched("output", "message_length", False)
    assert [e.enabled for e in (off.work["input"][0], off.work["output"][0])] == [True, False]
    reset = off.reset("input", "message_length")
    assert reset.work["input"][0].orbital is both and reset.work["output"][0].orbital is both
    assert texts(reset) == ["message_length turned off"]  # the other side keeps its switch
