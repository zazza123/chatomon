import json
from collections.abc import Callable
from typing import Any

import pytest
from orbitals.scope_guard_v2 import AsyncScopeGuardV2
from orbitals.types import AIServiceDescriptionV2
from pydantic_ai.models.test import TestModel

from chatomon import Atom, ChatomonConfigError, Knowledge, Verdict
from chatomon._version import __version__
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.errors import MissingExtraError
from chatomon.orbitals import JudgeOrbital, MessageLengthOrbital, TurnLimitOrbital
from chatomon.orbitals.base import Orbital
from chatomon.orbitals.principled import ScopeGuardOrbital
from chatomon.schemas import Side
from chatomon.studio import _kinds as kinds
from chatomon.studio._chains import Chains, RefusedError
from chatomon.studio._presets import InvalidPresetError, carried
from chatomon.studio._runners import pydantic_ai as pydantic_ai_runner
from chatomon.studio._session import Session
from tests.fakes import ScriptedOrbital

TONE = {"name": "tone", "model": "test:judge", "instructions": "Be kind."}


def make_session(
    *,
    context: str = "Open 9-18.",
    extra: list[Any] | None = None,
    preset: str | None = None,
    deps: bool = False,
) -> Session:
    """A new session on the same code: new objects, the same configuration.

    Args:
        context: The text of the knowledge the code gives to the judge.
        extra: Input orbitals the code has after the others.
        preset: A preset the session starts with.
        deps: Whether the atom declares `deps_type`, which the runner has a note for.
    """
    faq = Knowledge(context)
    model = TestModel(custom_output_text="Hi")
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        knowledge=faq,
        connector=PydanticAIConnector(model, deps_type=int) if deps else PydanticAIConnector(model),
        input_orbitals=[
            MessageLengthOrbital(max_words=60),
            ScriptedOrbital(name="scope"),
            *(extra or []),
        ],
        output_orbitals=[
            JudgeOrbital(
                name="faithfulness", model="test:judge", instructions="Check.", context=faq
            )
        ],
    )
    return Session(atom, source="app:atom", preset=preset)


def snapshot(session: Session) -> Any:
    state = session.state()
    orbitals = [
        (o["side"], o["name"], o["enabled"], o["origin"], o["fingerprint"], o["settings"])
        for side in ("input", "output")
        for o in state["orbitals"][side]
    ]
    return orbitals, state["changes"]


def nothing(session: Session) -> None:
    pass


def tuned(session: Session) -> None:
    session.edit("input", "message_length", {"max_words": 80, "reply": "Shorter.\nPlease."})
    session.switch("input", "scope", False)
    session.edit("output", "faithfulness", {"context": "Open 9-17.", "history_window": 2})


def rearranged(session: Session) -> None:
    session.move("input", "scope", 0)
    session.add("input", "TurnLimitOrbital", {"max_turns": 3, "warn_before": 1})
    session.add("output", "JudgeOrbital", {**TONE, "reads": ["faithfulness"]})
    session.move("input", "turn_limit", 0)
    session.switch("output", "tone", False)


def everything(session: Session) -> None:
    tuned(session)
    rearranged(session)


@pytest.mark.parametrize("change", [nothing, tuned, rearranged, everything])
def test_exporting_and_importing_restores_the_session(change: Callable[[Session], None]):
    first = make_session()
    change(first)
    text = json.dumps(first.export_preset())
    second = make_session()
    assert second.import_preset(text) == ([], [])
    assert snapshot(second) == snapshot(first)


def test_a_preset_carries_differences_for_code_orbitals_and_everything_for_added_ones():
    session = make_session()
    session.edit("input", "message_length", {"max_words": 80})
    session.switch("input", "scope", False)
    session.add("output", "JudgeOrbital", TONE)
    preset = session.export_preset()
    assert list(preset) == [
        "chatomon_studio_preset",
        "chatomon",
        "atom",
        "input_orbitals",
        "output_orbitals",
    ]
    assert (preset["chatomon_studio_preset"], preset["chatomon"], preset["atom"]) == (
        1,
        __version__,
        "speedypack",
    )
    limit, scope = preset["input_orbitals"]
    assert limit == {
        "name": "message_length",
        "kind": "MessageLengthOrbital",
        "from": "code",
        "enabled": True,
        "settings": {"max_words": 80},
        "fingerprint": session.chains.work["input"][0].orbital.fingerprint(),
    }
    assert scope == {
        "name": "scope",
        "kind": "ScriptedOrbital",
        "from": "code",
        "enabled": False,
        "fingerprint": session.chains.work["input"][1].orbital.fingerprint(),
    }
    faithfulness, tone = preset["output_orbitals"]
    assert "settings" not in faithfulness
    assert tone["from"] == "studio"
    assert set(tone["settings"]) == {
        "model",
        "instructions",
        "context",
        "history_window",
        "reads",
        "timeout",
        "on_error",
    }
    text = json.dumps(preset)
    assert "Open 9-18." not in text  # what the studio did not touch stays in the code
    assert "PydanticAIConnector" not in text and "config" not in text


def base() -> dict[str, Any]:
    return make_session().export_preset()


def with_entry(side: str = "input", index: int = 0, **changes: Any) -> dict[str, Any]:
    preset = base()
    preset[f"{side}_orbitals"][index].update(changes)
    return preset


def with_added(side: str, **entry: Any) -> dict[str, Any]:
    preset = base()
    added = {"kind": "JudgeOrbital", "from": "studio", "enabled": True, "fingerprint": "sha256:x"}
    preset[f"{side}_orbitals"].append({**added, **entry})
    return preset


def without_key(key: str) -> dict[str, Any]:
    preset = base()
    del preset["input_orbitals"][0][key]
    return preset


READ_ONLY = (
    "'scope': 'scope' is read-only in the studio. "
    "ScriptedOrbital is your own class: change its settings in your code."
)
JUDGE = {"model": "test:judge", "instructions": "Be kind."}
FROM_DISK: dict[str, Any] = {"from": "disk"}

PROBLEMS: list[tuple[str, Any, list[str]]] = [
    ("no version", [], ['this is not a studio preset: it has no "chatomon_studio_preset" version']),
    (
        "no version key",
        {"atom": "speedypack"},
        ['this is not a studio preset: it has no "chatomon_studio_preset" version'],
    ),
    (
        "newer version",
        {**base(), "chatomon_studio_preset": 2},
        ["this preset has schema version 2; this studio reads version 1"],
    ),
    (
        "true as version",
        {**base(), "chatomon_studio_preset": True},
        ["this preset has schema version True; this studio reads version 1"],
    ),
    (
        "another atom",
        {**base(), "atom": "other"},
        ["this preset is for the atom 'other', but the studio has 'speedypack' loaded"],
    ),
    ("unknown top key", {**base(), "extra": 1}, ["unknown key 'extra'"]),
    ("chain not a list", {**base(), "input_orbitals": "x"}, ["input_orbitals must be a list"]),
    (
        "entry not an object",
        {**base(), "input_orbitals": ["message_length"]},
        ["input_orbitals[0] must be an object"],
    ),
    (
        "enabled as text",
        with_entry(enabled="yes"),
        ["input_orbitals[0] needs 'enabled' as true or false"],
    ),
    ("name missing", without_key("name"), ["input_orbitals[0] needs 'name' as text"]),
    (
        "fingerprint missing",
        without_key("fingerprint"),
        ["input_orbitals[0] needs 'fingerprint' as text"],
    ),
    (
        "unknown entry key",
        with_entry(colour="red"),
        ["input_orbitals[0] has an unknown key 'colour'"],
    ),
    (
        "from somewhere else",
        with_entry(**FROM_DISK),
        ['input_orbitals[0] needs \'from\' as "code" or "studio"'],
    ),
    (
        "settings not an object",
        with_entry(settings=[1]),
        ["input_orbitals[0] needs 'settings' as an object"],
    ),
    (
        "gone from the code",
        with_entry(name="ghost"),
        [
            "'ghost': it is not in your code, so a preset cannot bring it back: "
            "add it to your code first"
        ],
    ),
    (
        "another class in the code",
        with_entry(index=1, kind="JudgeOrbital"),
        ["'scope': it is a ScriptedOrbital in your code, a JudgeOrbital in the preset"],
    ),
    (
        "a kind that cannot be added",
        with_added("input", name="x", kind="Nope"),
        ["'x': Nope cannot be added from the studio: add it in your code"],
    ),
    (
        "a kind on the wrong side",
        with_added("output", name="limit", kind="TurnLimitOrbital", settings={"max_turns": 3}),
        ["'limit': TurnLimitOrbital cannot run on the output side"],
    ),
    (
        "a name with capitals",
        with_added("output", name="Tone", settings=JUDGE),
        ["'Tone': invalid name 'Tone': use lowercase letters, digits, '_', '-' or '.'"],
    ),
    (
        "a name the code uses",
        with_added("output", name="scope", settings=JUDGE),
        ["'scope': an orbital named 'scope' is already in the atom"],
    ),
    (
        "a name among the settings",
        with_added("output", name="tone", settings={**JUDGE, "name": "other"}),
        ["'tone': `name` is the entry's own field, not a setting"],
    ),
    (
        "the same orbital twice",
        {**base(), "input_orbitals": [base()["input_orbitals"][1]] * 2},
        ["'scope' appears twice in input_orbitals"],
    ),
    (
        "an unknown setting",
        with_entry(settings={"nope": 1}),
        ["'message_length': `MessageLengthOrbital` has no setting `nope`"],
    ),
    ("a setting on a read-only orbital", with_entry(index=1, settings={"timeout": 5}), [READ_ONLY]),
    (
        "a number where text is expected",
        with_entry(settings={"reply": 3}),
        ["'message_length': `reply` must be text"],
    ),
    (
        "a value the constructor refuses",
        with_entry(settings={"max_words": 0}),
        ["'message_length': max_words must be a positive integer, got 0"],
    ),
    (
        "two problems",
        {**with_entry(settings={"max_words": 0}), "atom": "other", "extra": 1},
        [
            "unknown key 'extra'",
            "this preset is for the atom 'other', but the studio has 'speedypack' loaded",
            "'message_length': max_words must be a positive integer, got 0",
        ],
    ),
]


@pytest.mark.parametrize(
    ("preset", "problems"), [case[1:] for case in PROBLEMS], ids=[c[0] for c in PROBLEMS]
)
def test_an_invalid_preset_is_refused_with_one_message_per_problem(preset, problems):
    session = make_session()
    session.edit("input", "message_length", {"max_words": 70})
    before = snapshot(session)
    with pytest.raises(InvalidPresetError) as caught:
        session.import_preset(json.dumps(preset))
    assert caught.value.problems == problems
    assert str(caught.value) == "the preset was not applied"
    assert snapshot(session) == before
    assert session.state()["undo"] is None


def test_text_that_is_not_json_says_where_it_breaks():
    session = make_session()
    with pytest.raises(InvalidPresetError) as caught:
        session.import_preset('{"chatomon_studio_preset": 1,\n  oops}')
    (problem,) = caught.value.problems
    assert problem.startswith("this is not valid JSON: ")
    assert problem.endswith("(line 2, column 3)")
    with pytest.raises(InvalidPresetError, match="the preset was not applied"):
        session.import_preset("[" * 100_000)


def reading_atom() -> Atom:
    """An atom whose second orbital reads the first: turning the first off is refused."""
    first = ScriptedOrbital(Verdict.allow(annotations={"x": 1}), name="first")
    reader = ScriptedOrbital(name="reader", reads=["first"])
    return Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel()),
        input_orbitals=[first, reader],
    )


def test_a_chain_the_atom_refuses_is_a_problem():
    session = Session(reading_atom(), source="app:atom")
    preset = session.export_preset()
    preset["input_orbitals"][0]["enabled"] = False
    with pytest.raises(InvalidPresetError) as caught:
        session.import_preset(json.dumps(preset))
    assert len(caught.value.problems) == 1
    assert "reads 'first'" in caught.value.problems[0]
    assert session.state()["changes"] == []


def test_a_preset_from_other_code_is_applied_with_notes():
    old = make_session()
    old.switch("input", "scope", False)
    text = json.dumps(old.export_preset())
    new = make_session(context="Open 9-17.", extra=[ScriptedOrbital(name="new_guard")])
    assert new.import_preset(text).warnings == [
        "'new_guard' is in your code but not in the preset: it was added at the end, turned on.",
        "'faithfulness' differs from the exported preset: your code or chatomon changed since.",
    ]
    orbitals = new.state()["orbitals"]["input"]
    assert [(o["name"], o["enabled"]) for o in orbitals] == [
        ("message_length", True),
        ("scope", False),
        ("new_guard", True),
    ]


def test_an_import_replaces_the_session_and_can_be_undone():
    session = make_session()
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    before = snapshot(session)
    session.import_preset(json.dumps(base()))
    assert session.state()["changes"] == []
    assert session.state()["undo"] == "Preset imported"
    session.undo()
    assert snapshot(session) == before


def shared_session() -> Session:
    both = MessageLengthOrbital(max_words=60)
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel()),
        input_orbitals=[both],
        output_orbitals=[both],
    )
    return Session(atom, source="app:atom")


def test_an_instance_in_both_chains_takes_one_set_of_settings():
    first = shared_session()
    first.edit("output", "message_length", {"max_words": 80})
    preset = first.export_preset()
    assert preset["input_orbitals"][0]["settings"] == {"max_words": 80}
    assert preset["output_orbitals"][0]["settings"] == {"max_words": 80}
    second = shared_session()
    assert second.import_preset(json.dumps(preset)) == ([], [])
    assert snapshot(second) == snapshot(first)
    preset["output_orbitals"][0]["settings"] = {"max_words": 90}
    with pytest.raises(InvalidPresetError) as caught:
        shared_session().import_preset(json.dumps(preset))
    assert caught.value.problems == [
        "'message_length': it is one orbital in both chains of your code, "
        "but the preset gives it different settings"
    ]


@pytest.mark.parametrize(
    "odd",
    [None, 3, "text", [], {}, True, 1.5, [None], {"a": {"b": []}}],
    ids=repr,
)
def test_a_preset_edited_by_hand_never_crashes_the_import(odd):
    """Whatever replaces a value, the answer is a list of problems or an applied preset."""
    template = with_added("output", name="tone", settings=JUDGE)
    entry_keys = ["name", "kind", "from", "enabled", "settings", "fingerprint"]
    candidates = [odd, {**template, "input_orbitals": odd}, {**template, "atom": odd}]
    for key in entry_keys:
        preset = json.loads(json.dumps(template))
        preset["output_orbitals"][-1][key] = odd
        candidates.append(preset)
    preset = json.loads(json.dumps(template))
    preset["output_orbitals"][-1]["settings"]["reads"] = odd
    candidates.append(preset)
    for candidate in candidates:
        session = make_session()
        try:
            session.import_preset(json.dumps(candidate))
        except InvalidPresetError as exc:
            assert exc.problems
            assert all(isinstance(problem, str) for problem in exc.problems)


def test_text_the_studio_cannot_read_or_send_back_is_a_problem_not_a_failure():
    session = make_session()
    with pytest.raises(InvalidPresetError) as caught:
        session.import_preset('{"chatomon_studio_preset": ' + "9" * 5000 + "}")
    assert caught.value.problems == ["this is not valid JSON: it has a number too large to read"]
    odd = with_added("output", name="tone", kind="\ud800", settings=JUDGE)
    with pytest.raises(InvalidPresetError) as caught:
        session.import_preset(json.dumps(odd))
    assert caught.value.problems == ["this preset has text that is not valid Unicode"]
    assert session.state()["changes"] == []


@pytest.mark.parametrize("change", [nothing, tuned, rearranged, everything])
def test_a_session_started_with_a_preset_is_the_one_that_exported_it(
    change: Callable[[Session], None],
):
    first = make_session()
    change(first)
    started = make_session(preset=json.dumps(first.export_preset()))
    assert snapshot(started) == snapshot(first)
    assert started.notes == []


def test_a_start_with_a_preset_leaves_nothing_to_undo_and_a_reset_brings_it_back():
    first = make_session()
    everything(first)
    started = make_session(preset=json.dumps(first.export_preset()))
    assert started.state()["undo"] is None
    with pytest.raises(RefusedError, match="nothing to undo"):
        started.undo()
    started.reset_orbitals()
    assert started.state()["changes"] == []
    assert started.state()["undo"] == "Reset to code"
    started.undo()
    assert snapshot(started) == snapshot(first)


async def test_turns_run_with_the_orbitals_of_the_preset_the_session_started_with():
    first = make_session()
    first.edit("input", "message_length", {"max_words": 2, "reply": "Too long."})
    started = make_session(preset=json.dumps(first.export_preset()))
    queue = started.start_turn("one two three")
    while await queue.get() is not None:
        pass
    (turn,) = started.turns
    assert turn.report is not None
    assert (turn.report.final_action, turn.reply) == ("blocked", "Too long.")
    assert turn.changes == ["message_length.max_words changed", "message_length.reply changed"]


def test_the_warnings_of_a_preset_applied_at_start_are_notes_after_the_runners():
    old = make_session()
    old.switch("input", "scope", False)
    started = make_session(
        context="Open 9-17.",
        extra=[ScriptedOrbital(name="new_guard")],
        preset=json.dumps(old.export_preset()),
        deps=True,
    )
    runner_note, *warnings = started.notes
    assert "without deps" in runner_note
    assert warnings == [
        "'new_guard' is in your code but not in the preset: it was added at the end, turned on.",
        "'faithfulness' differs from the exported preset: your code or chatomon changed since.",
    ]
    orbitals = started.state()["orbitals"]["input"]
    assert [(o["name"], o["enabled"]) for o in orbitals] == [
        ("message_length", True),
        ("scope", False),
        ("new_guard", True),
    ]


@pytest.mark.parametrize(
    ("preset", "problems"), [case[1:] for case in PROBLEMS], ids=[c[0] for c in PROBLEMS]
)
def test_an_invalid_preset_stops_the_session_from_starting(preset, problems):
    with pytest.raises(InvalidPresetError) as caught:
        make_session(preset=json.dumps(preset))
    assert caught.value.problems == problems
    assert str(caught.value) == "the preset was not applied"


def test_a_chain_the_atom_refuses_stops_the_start_as_a_problem_of_the_preset():
    preset = Session(reading_atom(), source="app:atom").export_preset()
    preset["input_orbitals"][0]["enabled"] = False
    with pytest.raises(InvalidPresetError) as caught:
        Session(reading_atom(), source="app:atom", preset=json.dumps(preset))
    assert len(caught.value.problems) == 1
    assert "reads 'first'" in caught.value.problems[0]


def test_an_atom_that_does_not_build_keeps_its_own_error_when_a_preset_is_given():
    reader = ScriptedOrbital(name="reader", reads=["missing"])
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel()),
        input_orbitals=[reader],
    )
    entry = {
        "name": "reader",
        "kind": "ScriptedOrbital",
        "from": "code",
        "enabled": True,
        "fingerprint": reader.fingerprint(),
    }
    preset = {**base(), "input_orbitals": [entry], "output_orbitals": []}
    with pytest.raises(ChatomonConfigError) as caught:
        Session(atom, source="app:atom", preset=json.dumps(preset))
    assert not isinstance(caught.value, InvalidPresetError)
    with pytest.raises(ChatomonConfigError) as plain:
        Session(atom, source="app:atom")
    assert str(caught.value) == str(plain.value)


def test_starting_with_a_preset_leaves_the_developers_atom_untouched():
    first = make_session()
    everything(first)
    text = json.dumps(first.export_preset())
    faq = Knowledge("Open 9-18.")
    limit = MessageLengthOrbital(max_words=60)
    judge = JudgeOrbital(
        name="faithfulness", model="test:judge", instructions="Check.", context=faq
    )
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        knowledge=faq,
        connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
        input_orbitals=[limit, ScriptedOrbital(name="scope")],
        output_orbitals=[judge],
    )
    fingerprints = [orb.fingerprint() for orb in (*atom.input_orbitals, *atom.output_orbitals)]
    started = Session(atom, source="app:atom", preset=text)
    assert started.state()["changes"]
    assert atom.input_orbitals[0] is limit and atom.output_orbitals == (judge,)
    assert [
        orb.fingerprint() for orb in (*atom.input_orbitals, *atom.output_orbitals)
    ] == fingerprints
    assert limit.max_words == 60


def test_the_list_of_notes_a_runner_returns_is_not_changed_by_a_preset(monkeypatch):
    kept = ["a note the runner keeps"]
    monkeypatch.setattr(pydantic_ai_runner, "notes", lambda connector: kept)
    old = make_session()
    old.switch("input", "scope", False)
    started = make_session(context="Open 9-17.", preset=json.dumps(old.export_preset()))
    assert started.notes == [
        "a note the runner keeps",
        "'faithfulness' differs from the exported preset: your code or chatomon changed since.",
    ]
    assert kept == ["a note the runner keeps"]


# --- The changes of a session carried over code that was loaded again -----------------------


def limit(max_words: int | None = 60, max_chars: int | None = None) -> MessageLengthOrbital:
    return MessageLengthOrbital(max_words=max_words, max_chars=max_chars)


def scope() -> ScriptedOrbital:
    return ScriptedOrbital(name="scope")


def refunds() -> ScriptedOrbital:
    return ScriptedOrbital(name="no_refunds")


def judge(name: str = "faithfulness") -> JudgeOrbital:
    return JudgeOrbital(name=name, model="test:judge", instructions="Check.")


def code(inputs: list[Orbital], outputs: list[Orbital] | None = None) -> Session:
    """A session on code with these orbitals: what the studio has after a load."""
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
        input_orbitals=inputs,
        output_orbitals=outputs or [],
    )
    return Session(atom, source="app:atom")


def names(chains: Chains, side: Side = "input") -> list[str]:
    return [entry.orbital.name for entry in chains.work[side]]


def changes(chains: Chains) -> list[str]:
    return [change.text for change in chains.changes()]


def test_a_setting_changed_in_the_studio_goes_over_the_new_values_of_the_code():
    session = code([limit(), scope()])
    session.edit("input", "message_length", {"max_words": 80})
    after = code([limit(max_chars=500), scope()]).chains
    result, dropped = carried(session.chains, after)
    assert dropped == []
    values = kinds.values(result.work["input"][0].orbital)
    assert (values["max_words"], values["max_chars"]) == (80, 500)
    assert result.work["input"][0].base is after.code["input"][0]
    assert changes(result) == ["message_length.max_words changed"]


def test_a_setting_the_code_now_has_gives_back_the_codes_object():
    """What pasting Copy as code into the code and loading it again gives."""
    session = code([limit(), scope()])
    session.edit("input", "message_length", {"max_words": 80})
    after = code([limit(max_words=80), scope()]).chains
    result, dropped = carried(session.chains, after)
    assert dropped == []
    assert result.work["input"][0].orbital is after.code["input"][0]
    assert changes(result) == []


def test_an_orbital_that_was_off_stays_off_also_when_it_is_read_only():
    session = code([limit(), scope()])
    session.switch("input", "scope", False)
    session.switch("input", "message_length", False)
    result, dropped = carried(session.chains, code([limit(), scope()]).chains)
    assert dropped == []
    assert [entry.enabled for entry in result.work["input"]] == [False, False]
    assert changes(result) == ["message_length turned off", "scope turned off"]


def test_an_orbital_added_in_the_studio_is_carried_with_its_settings():
    session = code([limit()], [judge()])
    session.add("output", "JudgeOrbital", TONE)
    before = session.chains.work["output"][1].orbital
    result, dropped = carried(session.chains, code([limit()], [judge()]).chains)
    assert dropped == []
    tone = result.work["output"][1]
    assert (tone.origin, tone.orbital.fingerprint()) == ("studio", before.fingerprint())
    assert changes(result) == ["tone added"]


def test_a_changed_orbital_that_left_the_code_is_dropped_and_said():
    session = code([limit(), scope(), refunds()])
    session.switch("input", "scope", False)
    session.edit("input", "message_length", {"max_words": 80})
    result, dropped = carried(session.chains, code([refunds()]).chains)
    assert dropped == [
        "'message_length' is no longer in your code: what the studio changed on it was dropped.",
        "'scope' is no longer in your code: what the studio changed on it was dropped.",
    ]
    assert names(result) == ["no_refunds"]
    assert changes(result) == []


def test_an_untouched_orbital_that_left_the_code_says_nothing():
    session = code([limit(), scope()])
    session.edit("input", "message_length", {"max_words": 80})
    result, dropped = carried(session.chains, code([limit()]).chains)
    assert dropped == []
    assert names(result) == ["message_length"]
    assert changes(result) == ["message_length.max_words changed"]


def test_an_orbital_that_became_another_class_loses_its_changes():
    session = code([limit(), scope()])
    session.edit("input", "message_length", {"max_words": 80})
    after = code([ScriptedOrbital(name="message_length"), scope()]).chains
    result, dropped = carried(session.chains, after)
    assert dropped == [
        "'message_length' is a ScriptedOrbital in your code now: "
        "what the studio changed on it was dropped."
    ]
    assert result.work["input"][0].orbital is after.code["input"][0]
    assert changes(result) == []


def test_an_added_orbital_the_code_now_has_is_dropped_for_the_codes():
    session = code([limit()], [judge()])
    session.add("output", "JudgeOrbital", TONE)
    after = code([limit()], [judge(), judge("tone")]).chains
    result, dropped = carried(session.chains, after)
    assert dropped == ["'tone' is now in your code: the one added in the studio was dropped."]
    assert [entry.origin for entry in result.work["output"]] == ["code", "code"]
    assert changes(result) == []


def test_changes_the_new_code_refuses_are_dropped_with_the_reason():
    session = code([limit(max_chars=100)])
    session.edit("input", "message_length", {"max_words": None})
    result, dropped = carried(session.chains, code([limit()]).chains)
    assert dropped == [
        "what the studio changed on 'message_length' was dropped: set max_words, max_chars or both"
    ]
    assert changes(result) == []


def test_an_added_orbital_that_can_no_longer_be_added_is_dropped_with_the_reason(monkeypatch):
    session = code([limit()])
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    others = tuple(kind for kind in kinds.addable() if kind.cls is not TurnLimitOrbital)
    monkeypatch.setattr(kinds, "addable", lambda: others)
    result, dropped = carried(session.chains, code([limit()]).chains)
    assert dropped == [
        "'turn_limit', added in the studio, was dropped: "
        "TurnLimitOrbital cannot be added from the studio: add it in your code"
    ]
    assert names(result) == ["message_length"]


def test_a_chain_the_studio_never_reordered_takes_the_order_of_the_new_code():
    session = code([limit(), scope(), refunds()], [judge()])
    session.switch("input", "scope", False)
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    after = code([ScriptedOrbital(name="first"), refunds(), scope(), limit()], [judge()]).chains
    result, dropped = carried(session.chains, after)
    assert dropped == []
    # turn_limit followed no_refunds, and still does.
    assert names(result) == ["first", "no_refunds", "turn_limit", "scope", "message_length"]
    assert changes(result) == ["turn_limit added", "scope turned off"]


def test_an_added_orbital_stays_after_the_entry_it_followed():
    session = code([limit(), scope(), refunds()])
    session.add("input", "TurnLimitOrbital", {"max_turns": 3, "name": "a"})
    session.add("input", "TurnLimitOrbital", {"max_turns": 4, "name": "b"})
    session.add("input", "TurnLimitOrbital", {"max_turns": 5, "name": "c"})
    session.move("input", "a", 1)  # after message_length
    session.move("input", "b", 2)  # after a
    session.move("input", "c", 0)  # first
    assert names(session.chains) == ["c", "message_length", "a", "b", "scope", "no_refunds"]
    assert "input order changed" not in changes(session.chains)
    after = code([ScriptedOrbital(name="first"), scope(), limit(), refunds()]).chains
    result, dropped = carried(session.chains, after)
    assert dropped == []
    assert names(result) == ["c", "first", "scope", "message_length", "a", "b", "no_refunds"]


def test_an_added_orbital_whose_neighbour_left_goes_after_the_one_before_it():
    session = code([limit(), scope(), refunds()])
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    session.move("input", "turn_limit", 2)  # after scope
    result, dropped = carried(session.chains, code([limit(), refunds()]).chains)
    assert dropped == []
    assert names(result) == ["message_length", "turn_limit", "no_refunds"]
    result, _ = carried(session.chains, code([refunds()]).chains)
    assert names(result) == ["turn_limit", "no_refunds"]


def test_a_chain_the_studio_reordered_keeps_its_order_and_new_orbitals_go_last():
    session = code([limit(), scope(), refunds()])
    session.move("input", "no_refunds", 0)
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    after = code([ScriptedOrbital(name="first"), limit(), scope(), refunds()]).chains
    result, dropped = carried(session.chains, after)
    assert dropped == []
    assert names(result) == ["no_refunds", "message_length", "scope", "turn_limit", "first"]
    assert "input order changed" in changes(result)


def test_an_instance_in_both_chains_is_carried_as_one():
    def both() -> Session:
        shared = limit()
        return code([shared], [shared])

    session = both()
    session.edit("output", "message_length", {"max_words": 80})
    result, dropped = carried(session.chains, both().chains)
    assert dropped == []
    assert result.work["input"][0].orbital is result.work["output"][0].orbital
    assert kinds.values(result.work["input"][0].orbital)["max_words"] == 80
    assert changes(result) == ["message_length.max_words changed"]


def test_carrying_changes_never_touches_the_chains_it_reads():
    session = code([limit(), scope()])
    session.edit("input", "message_length", {"max_words": 80})
    after = code([scope(), limit()]).chains
    before = (dict(session.chains.work), dict(after.work))
    carried(session.chains, after)
    assert (dict(session.chains.work), dict(after.work)) == before


def test_an_orbital_the_code_moved_to_the_other_chain_is_not_said_to_have_left_the_code():
    session = code([limit(), scope()])
    session.switch("input", "message_length", False)
    result, dropped = carried(session.chains, code([scope()], [limit()]).chains)
    assert dropped == [
        "'message_length' is no longer in the input chain of your code: "
        "what the studio changed on it there was dropped."
    ]
    assert [entry.enabled for entry in result.work["output"]] == [True]


def test_an_orbital_of_both_chains_that_left_one_keeps_its_changes_on_the_other():
    shared = limit()
    session = code([shared], [shared])
    session.edit("output", "message_length", {"max_words": 80})
    result, dropped = carried(session.chains, code([limit()]).chains)
    assert dropped == [
        "'message_length' is no longer in the output chain of your code: "
        "what the studio changed on it there was dropped."
    ]
    assert changes(result) == ["message_length.max_words changed"]


# --- Blocks: connections and service descriptions -----------------------------

KEY = "key-0123456789-not-for-the-preset"
BACKEND = {
    "type": "ApiBackend",
    "fields": {"api_url": "https://scope.example", "api_key_env": "SCOPE_KEY"},
}
PAIRS = {
    "type": "AIServiceDescriptionV2",
    "fields": {
        "identity_role": "Courier",
        "context": "Italy",
        "predefined_responses": [
            {
                "type": "PredefinedResponse",
                "fields": {"trigger": "phone", "response": "800 123 456"},
            },
            "Greet the user back.",
        ],
    },
}
NOTE = (
    "'hosted' uses the environment variable SCOPE_KEY: "
    'guard=ApiBackend(api_url="https://scope.example", api_key_env="SCOPE_KEY"). '
    "Check where it points before you send a message."
)
UNNAMED = (
    "'claim_extractor' names no environment variable for its key: "
    'extractor=ApiBackend(api_url="https://claims.example"). Its client can still send a key it '
    "reads from the environment on its own. Check where it points before you send a message."
)


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("SCOPE_KEY", KEY)
    return monkeypatch


def service_session(preset: str | None = None) -> Session:
    """A session on code whose scope guard has a client and a description of its own."""
    client = AsyncScopeGuardV2(backend="api", api_url="http://scope-guard.test", api_key="k")
    described = AIServiceDescriptionV2(
        identity_role="Courier", context="Italy", constraints="No refunds"
    )
    atom = Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
        input_orbitals=[ScopeGuardOrbital(name="scope", guard=client, description=described)],
    )
    return Session(atom, source="app:atom", preset=preset)


def with_services(session: Session) -> None:
    session.add(
        "input", "ScopeGuardOrbital", {"name": "hosted", "guard": BACKEND, "description": PAIRS}
    )
    session.add(
        "output",
        "ClaimExtractorOrbital",
        {"extractor": {"type": "ApiBackend", "fields": {"api_url": "https://claims.example"}}},
    )
    shown = session.state()["orbitals"]["input"][0]["settings"][0]["value"]
    shown["fields"]["context"] = "Italy and France"
    session.edit("input", "scope", {"description": shown})


def test_exporting_and_importing_restores_a_session_with_blocks(key):
    first = service_session()
    with_services(first)
    exported = first.export_preset()
    text = json.dumps(exported)
    assert KEY not in text
    second = service_session()
    # Where each orbital connects is said apart from what differs from the exported session.
    assert second.import_preset(text) == ([NOTE, UNNAMED], [])
    assert snapshot(second) == snapshot(first)
    started = service_session(preset=text)
    assert snapshot(started) == snapshot(first)
    assert started.notes == [NOTE, UNNAMED]
    # The page shows, when it opens, where a preset given at the start makes orbitals connect.
    assert started.state()["studio"]["preset_connections"] == [NOTE, UNNAMED]
    assert second.state()["studio"]["preset_connections"] == []


def test_a_preset_writes_a_block_whole_with_the_fields_that_differ_from_the_default(key):
    session = service_session()
    with_services(session)
    preset = session.export_preset()
    scope, hosted = preset["input_orbitals"]
    # An orbital from code carries the setting that changed, whole.
    assert scope["settings"] == {
        "description": {
            "type": "AIServiceDescriptionV2",
            "fields": {
                "identity_role": "Courier",
                "context": "Italy and France",
                "constraints": ["No refunds"],
            },
        }
    }
    assert (hosted["settings"]["guard"], hosted["settings"]["description"]) == (BACKEND, PAIRS)
    claims = preset["output_orbitals"][0]
    assert claims["settings"]["extractor"] == {
        "type": "ApiBackend",
        "fields": {"api_url": "https://claims.example"},
    }
    assert claims["settings"]["description"] is None


def added(**settings: Any) -> str:
    preset = service_session().export_preset()
    entry = {
        "name": "hosted",
        "kind": "ScopeGuardOrbital",
        "from": "studio",
        "enabled": True,
        "settings": {"guard": BACKEND, "description": "A courier.", **settings},
        "fingerprint": "sha256:x",
    }
    preset["input_orbitals"].append(entry)
    return json.dumps(preset)


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        (
            {"guard": {"type": "Tunnel", "fields": {}}},
            "'hosted': `guard` must be an ApiBackend",
        ),
        (
            {"guard": {"type": "ApiBackend", "fields": {"api_url": "https://a", "port": 1}}},
            "'hosted': `ApiBackend` has no field `port`",
        ),
        (
            {"guard": {"type": "ApiBackend", "fields": {"api_url": "a"}}},
            "'hosted': api_url must start with http:// or https:// and name a host",
        ),
        (
            {
                "description": {
                    "type": "AIServiceDescription",
                    "fields": {"identity_role": "r", "context": "c"},
                }
            },
            "'hosted': ScopeGuard V2 needs an AIServiceDescriptionV2 (or a string) as description",
        ),
        (
            {"description": {"type": "AIServiceDescriptionV2", "fields": {"identity_role": "r"}}},
            "'hosted': `context` needs a value",
        ),
    ],
)
def test_a_block_a_preset_gets_wrong_is_a_problem(key, settings, problem):
    session = service_session()
    before = snapshot(session)
    with pytest.raises(InvalidPresetError) as caught:
        session.import_preset(added(**settings))
    assert caught.value.problems == [problem]
    assert snapshot(session) == before


def test_a_variable_that_is_not_set_is_a_problem_of_the_preset(monkeypatch):
    monkeypatch.delenv("SCOPE_KEY", raising=False)
    session = service_session()
    before = snapshot(session)
    with pytest.raises(InvalidPresetError) as caught:
        session.import_preset(added())
    assert caught.value.problems == ["'hosted': the environment variable SCOPE_KEY is not set"]
    assert snapshot(session) == before
    with pytest.raises(InvalidPresetError):
        service_session(preset=added())


def test_a_class_whose_module_lacks_its_extra_is_a_problem_that_names_the_extra(key):
    text = added()
    session = make_session()
    real = kinds.importlib.import_module

    def fake(name: str, package: str | None = None) -> Any:
        if name == "chatomon.orbitals.principled":
            raise MissingExtraError("principled", "orbitals")
        return real(name, package)

    key.setattr(kinds.importlib, "import_module", fake)
    kinds._library.cache_clear()
    try:
        with pytest.raises(InvalidPresetError) as caught:
            session.import_preset(text.replace('"scope"', '"message_length"'))
    finally:
        key.undo()
        kinds._library.cache_clear()
    assert (
        "'hosted': ScopeGuardOrbital is not an orbital this studio can add. The orbitals of "
        "chatomon.orbitals.principled need: pip install 'chatomon[principled]'."
    ) in caught.value.problems


def test_a_connection_that_names_no_variable_is_noted_too(key):
    # Without a name the client reads a key from the environment on its own, as the
    # library's does with PRINCIPLED_API_KEY: the address a preset sets is always said.
    session = service_session()
    plain = {"type": "ApiBackend", "fields": {"api_url": "https://scope.example"}}
    assert session.import_preset(added(guard=plain)).connections == [
        "'hosted' names no environment variable for its key: "
        'guard=ApiBackend(api_url="https://scope.example"). Its client can still send a key it '
        "reads from the environment on its own. Check where it points before you send a message."
    ]


def test_a_block_that_is_not_a_connection_gives_no_note(key):
    session = service_session()
    preset = session.export_preset()
    preset["input_orbitals"][0]["settings"] = {"description": "A courier."}
    notes = session.import_preset(json.dumps(preset))
    assert notes.connections == []
    assert notes.warnings == [
        "'scope' differs from the exported preset: your code or chatomon changed since."
    ]


class TokenOrbital(ScriptedOrbital):
    """The name of a variable as a parameter of its own: no built-in orbital has one yet."""

    def __init__(self, *, token_env: str | None = None, name: str | None = None) -> None:
        super().__init__(name=name)
        self.token_env = token_env

    def config(self) -> dict[str, Any]:
        return {**super().config(), "token_env": self.token_env}


def test_a_plain_setting_that_names_a_variable_is_noted_like_a_connection(monkeypatch):
    builtin = kinds.is_builtin
    monkeypatch.setattr(kinds, "is_builtin", lambda cls: cls is TokenOrbital or builtin(cls))

    def session() -> Session:
        atom = Atom(
            name="speedypack",
            instructions="Be brief.",
            connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
            input_orbitals=[TokenOrbital(name="token")],
        )
        return Session(atom, source="app:atom")

    first = session()
    first.edit("input", "token", {"token_env": "SOME_TOKEN"})
    assert session().import_preset(json.dumps(first.export_preset())) == (
        [
            "'token' uses the environment variable SOME_TOKEN: token_env=\"SOME_TOKEN\". "
            "Check where it points before you send a message."
        ],
        [],
    )


def test_a_connection_note_writes_letters_that_are_not_ascii_as_escapes(key):
    # An address with a letter of another alphabet can look like one the developer trusts.
    session = service_session()
    lookalike = {
        "type": "ApiBackend",
        "fields": {"api_url": "https://sc\u043epe.example", "api_key_env": "SCOPE_KEY"},
    }
    (note,) = session.import_preset(added(guard=lookalike)).connections
    assert 'api_url="https://sc\\u043epe.example"' in note
    assert note.isascii()


def test_blocks_are_carried_over_code_that_was_loaded_again(key):
    session = service_session()
    with_services(session)
    before = [entry.orbital.fingerprint() for entry in session.chains.work["input"]]
    result, dropped = carried(session.chains, service_session().chains)
    assert dropped == []
    assert [entry.orbital.fingerprint() for entry in result.work["input"]] == before
    assert changes(result) == [
        "scope.description changed",
        "hosted added",
        "claim_extractor added",
    ]


def test_an_added_orbital_whose_variable_is_gone_is_dropped_and_said(key):
    session = service_session()
    with_services(session)
    key.delenv("SCOPE_KEY")
    result, dropped = carried(session.chains, service_session().chains)
    assert dropped == [
        "'hosted', added in the studio, was dropped: the environment variable SCOPE_KEY is not set"
    ]
    assert names(result) == ["scope"]
    assert changes(result) == ["scope.description changed", "claim_extractor added"]
