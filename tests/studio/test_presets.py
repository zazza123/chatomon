import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

from chatomon import Atom, ChatomonConfigError, Knowledge, Verdict
from chatomon._version import __version__
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import JudgeOrbital, MessageLengthOrbital
from chatomon.studio._chains import RefusedError
from chatomon.studio._presets import InvalidPresetError
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
    assert second.import_preset(text) == []
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
    assert new.import_preset(text) == [
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
    assert second.import_preset(json.dumps(preset)) == []
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
