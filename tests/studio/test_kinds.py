import inspect
import math
from typing import Any

import pytest

import chatomon.orbitals as library
from chatomon import Knowledge, Verdict, orbital
from chatomon.orbitals import FunctionOrbital, JudgeOrbital, MessageLengthOrbital, TurnLimitOrbital
from chatomon.orbitals.base import Orbital
from chatomon.schemas import OrbitalContext
from chatomon.studio import _kinds as kinds
from chatomon.studio._kinds import InvalidSettingError
from tests.fakes import FakeConnector, ScriptedOrbital

CONNECTOR = FakeConnector()

# Constructor arguments for every orbital class the library exports. A new class
# needs an entry here: the studio must be able to rebuild it from what it reads.
SAMPLES: dict[type[Orbital], list[dict[str, Any]]] = {
    MessageLengthOrbital: [
        {"max_words": 60},
        {
            "max_chars": 500,
            "reply": "Too long.",
            "guidance": "Shorter.",
            "name": "len",
            "timeout": 5,
            "on_error": "allow",
        },
    ],
    TurnLimitOrbital: [
        {"max_turns": 8},
        {"max_turns": 8, "warn_before": 2, "warning": "Wrap up.", "reply": "Bye."},
    ],
    JudgeOrbital: [
        {"model": "test:judge", "instructions": "Block rude replies."},
        {
            "model": "test:judge",
            "instructions": "Check facts.",
            "context": Knowledge("Open 9-18.", title="FAQ"),
            "connector": FakeConnector(),
            "history_window": 2,
            "reads": ["claims"],
            "name": "facts",
        },
    ],
    FunctionOrbital: [],  # read-only by decision
}


def exported() -> list[type[Orbital]]:
    found = []
    for name in library.__all__:
        value = getattr(library, name)
        if isinstance(value, type) and issubclass(value, Orbital) and not inspect.isabstract(value):
            found.append(value)
    return found


def shapes(cls: type[Orbital]) -> dict[str, tuple[str, bool, bool]]:
    return {s.key: (s.type, s.optional, s.required) for s in kinds.describe(cls).settings}


def test_settings_come_from_the_constructor():
    common = {"timeout": ("number", False, False), "on_error": ("choice", False, False)}
    assert shapes(MessageLengthOrbital) == {
        "max_words": ("int", True, False),
        "max_chars": ("int", True, False),
        "reply": ("text", False, False),
        "guidance": ("text", True, False),
        **common,
    }
    assert shapes(TurnLimitOrbital) == {
        "max_turns": ("int", False, True),
        "reply": ("text", False, False),
        "warn_before": ("int", True, False),
        "warning": ("text", False, False),
        **common,
    }
    assert shapes(JudgeOrbital) == {
        "model": ("text", False, True),
        "instructions": ("text", False, True),
        "context": ("text", True, False),
        "history_window": ("int", False, False),
        "reads": ("names", False, False),
        **common,
    }
    assert kinds.describe(JudgeOrbital).objects == ("connector",)
    assert kinds.describe(FunctionOrbital).objects == ("func",)


def test_defaults_options_and_summaries_are_read_not_copied():
    limit = kinds.describe(TurnLimitOrbital)
    on_error = limit.setting("on_error")
    reply = limit.setting("reply")
    assert on_error is not None and on_error.options == ("block", "allow")
    assert reply is not None
    assert reply.default == inspect.signature(TurnLimitOrbital.__init__).parameters["reply"].default
    assert kinds.summary(TurnLimitOrbital) == "Closes the conversation after a number of turns."
    assert reply.schema() == {
        "key": "reply",
        "type": "text",
        "optional": False,
        "required": False,
        "default": reply.default,
    }
    reads = kinds.describe(JudgeOrbital).setting("reads")
    assert reads is not None and reads.schema()["default"] == []


def test_the_classes_that_can_be_added_are_those_built_from_plain_values():
    assert [kind.cls for kind in kinds.addable()] == [
        JudgeOrbital,
        MessageLengthOrbital,
        TurnLimitOrbital,
    ]


def test_every_exported_orbital_has_a_sample():
    missing = [cls.__name__ for cls in exported() if cls not in SAMPLES]
    assert missing == [], f"add constructor arguments for {missing} to SAMPLES in this file"


@pytest.mark.parametrize(
    ("cls", "arguments"),
    [(cls, arguments) for cls, samples in SAMPLES.items() for arguments in samples],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
def test_an_orbital_rebuilt_from_what_the_studio_reads_is_the_same(cls, arguments):
    original = cls(**arguments)
    assert kinds.read_only(original, CONNECTOR) is None
    twin = cls(**kinds.arguments(original, CONNECTOR))
    assert twin is not original
    assert twin.fingerprint() == original.fingerprint()


def test_a_judge_keeps_its_own_connector_and_drops_the_atoms():
    own = FakeConnector()
    judge = JudgeOrbital(model="m", instructions="i", connector=own)
    assert kinds.arguments(judge, CONNECTOR)["connector"] is own
    assert kinds.arguments(judge, own)["connector"] is None
    unbound = JudgeOrbital(model="m", instructions="i")
    assert kinds.arguments(unbound, CONNECTOR)["connector"] is None
    unbound.bind(CONNECTOR)
    assert kinds.arguments(unbound, CONNECTOR)["connector"] is None


def test_a_judges_context_is_edited_as_text():
    faq = Knowledge("Open 9-18.", title="FAQ")
    judge = JudgeOrbital(model="m", instructions="i", context=faq)
    assert kinds.values(judge)["context"] == "Open 9-18."
    same = kinds.rebuilt(judge, {"context": "Open 9-18."}, CONNECTOR)
    assert isinstance(same, JudgeOrbital) and same.context is faq
    changed = kinds.rebuilt(judge, {"context": "Open 9-17."}, CONNECTOR)
    assert isinstance(changed, JudgeOrbital) and changed.context is not None
    assert (changed.context.text, changed.context.title) == ("Open 9-17.", "FAQ")
    cleared = kinds.rebuilt(judge, {"context": None}, CONNECTOR)
    assert isinstance(cleared, JudgeOrbital) and cleared.context is None
    with pytest.raises(InvalidSettingError, match="Knowledge text must not be empty"):
        kinds.rebuilt(judge, {"context": "  "}, CONNECTOR)


def test_orbitals_the_studio_does_not_edit_say_why():
    @orbital
    def check(ctx: OrbitalContext) -> Verdict:
        return Verdict.allow()

    class Mine(JudgeOrbital):
        pass

    assert kinds.read_only(check, CONNECTOR) == kinds.FUNCTION
    assert kinds.read_only(ScriptedOrbital(), CONNECTOR) == (
        "ScriptedOrbital is your own class: change its settings in your code."
    )
    assert kinds.read_only(Mine(model="m", instructions="i"), CONNECTOR) == (
        "Mine is your own class: change its settings in your code."
    )


def test_an_orbital_the_studio_cannot_read_back_is_read_only(monkeypatch):
    limit = MessageLengthOrbital(max_words=60)
    limit.max_words = "sixty"  # type: ignore[assignment]
    assert kinds.read_only(limit, CONNECTOR) == kinds.UNREADABLE

    class Hides(ScriptedOrbital):
        def __init__(self, *, limit: int = 3) -> None:
            super().__init__()
            self._limit = limit

    monkeypatch.setattr(kinds, "is_builtin", lambda cls: True)
    assert kinds.read_only(Hides(limit=5), CONNECTOR) == kinds.UNREADABLE


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("max_words", "60", "`max_words` must be a whole number"),
        ("max_words", True, "`max_words` must be a whole number"),
        ("max_words", 1.5, "`max_words` must be a whole number"),
        ("timeout", "soon", "`timeout` must be a number"),
        ("timeout", math.inf, "`timeout` must be a number"),
        ("timeout", None, "`timeout` needs a value"),
        ("reply", 3, "`reply` must be text"),
        ("reply", "broken \ud800 text", "`reply` must be text"),
        ("on_error", "ignore", "`on_error` must be one of: block, allow"),
        ("nope", 1, "`MessageLengthOrbital` has no setting `nope`"),
    ],
)
def test_a_value_of_the_wrong_shape_is_refused_by_the_studio(key, value, message):
    limit = MessageLengthOrbital(max_words=60)
    with pytest.raises(InvalidSettingError) as caught:
        kinds.rebuilt(limit, {key: value}, CONNECTOR)
    assert str(caught.value) == message
    assert caught.value.setting == key


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("reads", "claims", "`reads` must be a list of orbital names"),
        ("reads", [1], "`reads` must be a list of orbital names"),
        ("connector", "x", "`connector` is an object from your code: change it there"),
    ],
)
def test_lists_and_objects_have_their_own_messages(key, value, message):
    judge = JudgeOrbital(model="m", instructions="i")
    with pytest.raises(InvalidSettingError) as caught:
        kinds.rebuilt(judge, {key: value}, CONNECTOR)
    assert str(caught.value) == message


def test_the_constructor_has_the_last_word_with_its_own_message():
    limit = MessageLengthOrbital(max_words=60)
    with pytest.raises(InvalidSettingError) as caught:
        kinds.rebuilt(limit, {"max_words": None}, CONNECTOR)
    assert str(caught.value) == "set max_words, max_chars or both"
    assert caught.value.setting == "max_words"
    with pytest.raises(InvalidSettingError, match="max_words must be a positive integer, got 0"):
        kinds.rebuilt(limit, {"max_words": 0}, CONNECTOR)
    with pytest.raises(InvalidSettingError) as several:
        kinds.rebuilt(limit, {"max_words": None, "reply": "Shorter."}, CONNECTOR)
    assert several.value.setting is None


def test_a_number_too_large_for_the_constructor_is_refused_not_raised():
    limit = MessageLengthOrbital(max_words=60)
    with pytest.raises(InvalidSettingError):
        kinds.rebuilt(limit, {"timeout": 10**400}, CONNECTOR)


def test_an_accepted_change_builds_a_new_orbital():
    limit = MessageLengthOrbital(max_words=60, name="len")
    longer = kinds.rebuilt(limit, {"max_words": 80}, CONNECTOR)
    assert isinstance(longer, MessageLengthOrbital)
    assert (longer.max_words, longer.name, limit.max_words) == (80, "len", 60)
    assert longer.fingerprint() != limit.fingerprint()


@pytest.mark.parametrize("name", ["Tone", "has space", "", "tóne", 3, None])
def test_names_set_in_the_studio_are_lowercase(name):
    with pytest.raises(InvalidSettingError) as caught:
        kinds.check_name(name)
    assert str(caught.value) == (
        f"invalid name {name!r}: use lowercase letters, digits, '_', '-' or '.'"
    )
    assert caught.value.setting == "name"


def test_a_valid_name_is_returned():
    assert kinds.check_name("fact-check.v2_a") == "fact-check.v2_a"


def test_creating_an_orbital_asks_for_what_is_required():
    judge = kinds.describe(JudgeOrbital)
    with pytest.raises(InvalidSettingError) as caught:
        kinds.created(judge, "tone", {"model": "test:judge"})
    assert (str(caught.value), caught.value.setting) == (
        "`instructions` needs a value",
        "instructions",
    )
    made = kinds.created(judge, "tone", {"model": "test:judge", "instructions": "Be kind."})
    assert isinstance(made, JudgeOrbital) and made.name == "tone"
    with pytest.raises(InvalidSettingError, match="set max_words, max_chars or both"):
        kinds.created(kinds.describe(MessageLengthOrbital), "len", {})


def test_principled_orbitals_are_read_through_their_instances():
    pytest.importorskip("orbitals")
    from orbitals.claim_extractor import AsyncClaimExtractor
    from orbitals.scope_guard_v2 import AsyncScopeGuardV2

    from chatomon.orbitals.principled import ClaimExtractorOrbital, ScopeGuardOrbital

    client = AsyncScopeGuardV2(backend="api", api_url="http://scope-guard.test", api_key="k")
    guard = ScopeGuardOrbital(guard=client, description="Parcel tracking")
    assert shapes(ScopeGuardOrbital) == {
        "history_window": ("int", False, False),
        "timeout": ("number", False, False),
        "on_error": ("choice", False, False),
    }
    assert kinds.describe(ScopeGuardOrbital).objects == (
        "guard",
        "description",
        "mapping",
        "headers",
    )
    assert kinds.describe(ScopeGuardOrbital).addable is False
    assert kinds.read_only(guard, CONNECTOR) is None
    assert kinds.arguments(guard, CONNECTOR)["mapping"] is None
    overrides: dict[Any, Any] = {"Chit Chat": Verdict.block()}
    strict = ScopeGuardOrbital(guard=client, description="Parcel tracking", mapping=overrides)
    assert kinds.read_only(strict, CONNECTOR) is None
    # Every argument is read from the attribute named like it: the developer's own dict.
    assert kinds.arguments(strict, CONNECTOR)["mapping"] is overrides
    shorter = kinds.rebuilt(strict, {"history_window": 2}, CONNECTOR)
    assert isinstance(shorter, ScopeGuardOrbital)
    assert (shorter.guard, shorter.rules, shorter.history_window) == (client, strict.rules, 2)

    extractor = ClaimExtractorOrbital(
        extractor=AsyncClaimExtractor(backend="api", api_url="http://claims.test", api_key="k")
    )
    assert shapes(ClaimExtractorOrbital)["intents_only"] == ("bool", False, False)
    assert kinds.read_only(extractor, CONNECTOR) is None


def test_a_switch_takes_true_or_false():
    flag = kinds.Setting("intents_only", "bool", False, False, False)
    kinds.check_shape(flag, True)
    kinds.check_shape(flag, False)
    for value in ("yes", 1, 0):
        with pytest.raises(InvalidSettingError) as caught:
            kinds.check_shape(flag, value)
        assert str(caught.value) == "`intents_only` must be true or false"
