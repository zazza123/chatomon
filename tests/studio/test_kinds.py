import importlib
import inspect
import math
from typing import Any

import pytest
from orbitals.claim_extractor import AsyncClaimExtractor
from orbitals.scope_guard_v2 import AsyncScopeGuardV2
from orbitals.types import (
    AIServiceDescription,
    AIServiceDescriptionV2,
    PredefinedResponse,
    Principle,
)

from chatomon import Knowledge, Verdict, orbital
from chatomon.errors import MissingExtraError
from chatomon.orbitals import FunctionOrbital, JudgeOrbital, MessageLengthOrbital, TurnLimitOrbital
from chatomon.orbitals.base import Orbital
from chatomon.orbitals.principled import ApiBackend, ClaimExtractorOrbital, ScopeGuardOrbital
from chatomon.schemas import OrbitalContext
from chatomon.studio import _kinds as kinds
from chatomon.studio import _values as values
from chatomon.studio._values import InvalidSettingError
from tests.fakes import FakeConnector, ScriptedOrbital

CONNECTOR = FakeConnector()
CLIENT = AsyncScopeGuardV2(backend="api", api_url="http://scope-guard.test", api_key="k")
SERVICE = AIServiceDescriptionV2(
    identity_role="Parcel tracking assistant",
    context="SpeedyPack, an Italian courier",
    constraints=["Never discuss refunds"],
    predefined_responses=[
        PredefinedResponse(trigger="The user asks for the phone number", response="800 123 456"),
        "Greet the user back.",
    ],
)

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
    ScopeGuardOrbital: [
        {"guard": ApiBackend(api_url="https://scope.example"), "description": "A courier."},
        {
            "guard": CLIENT,
            "description": SERVICE,
            "mapping": {"Chit Chat": Verdict.block()},
            "history_window": 2,
            "name": "scope",
        },
    ],
    ClaimExtractorOrbital: [
        {"extractor": ApiBackend(api_url="https://claims.example", model="claims-q")},
        {
            "extractor": AsyncClaimExtractor(backend="api", api_url="http://claims.test"),
            "description": AIServiceDescription(
                identity_role="Parcel tracking assistant",
                context="SpeedyPack",
                principles=[
                    Principle(title="Honest", description="No guesses", supporting_materials=None),
                    "Be brief.",
                ],
            ),
            "intents_only": True,
            "reads": ["scope"],
        },
    ],
}


def exported() -> list[type[Orbital]]:
    return [cls for cls in kinds.classes() if not inspect.isabstract(cls)]


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
    # A function's own argument; the studio still shows functions read-only.
    assert shapes(FunctionOrbital)["calls_model"] == ("bool", False, False)


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


def test_the_classes_are_found_by_walking_the_modules_of_the_library():
    # Those of `chatomon.orbitals` first, in the order it exports them, then the other modules'.
    assert [cls.__name__ for cls in kinds.classes()] == [
        "FunctionOrbital",
        "JudgeOrbital",
        "MessageLengthOrbital",
        "Orbital",
        "TurnLimitOrbital",
        "ClaimExtractorOrbital",
        "ScopeGuardOrbital",
    ]
    assert (kinds.extras(), kinds.import_notes()) == ((), ())


def test_the_classes_that_can_be_added_are_those_built_from_settings_and_blocks():
    assert [kind.cls for kind in kinds.addable()] == [
        JudgeOrbital,
        MessageLengthOrbital,
        TurnLimitOrbital,
        ClaimExtractorOrbital,
        ScopeGuardOrbital,
    ]


@pytest.fixture
def without(monkeypatch):
    """Make the import of one module of the library fail, and forget what was found."""

    def fail(module: str, error: Exception) -> None:
        real = importlib.import_module

        def fake(name: str, package: str | None = None) -> Any:
            if name == module:
                raise error
            return real(name, package)

        monkeypatch.setattr(kinds.importlib, "import_module", fake)
        kinds._library.cache_clear()

    yield fail
    monkeypatch.undo()
    kinds._library.cache_clear()


def test_a_module_that_needs_an_extra_is_remembered_with_it(without):
    without("chatomon.orbitals.principled", MissingExtraError("principled", "orbitals"))
    assert [kind.cls for kind in kinds.addable()] == [
        JudgeOrbital,
        MessageLengthOrbital,
        TurnLimitOrbital,
    ]
    assert kinds.extras() == (("chatomon.orbitals.principled", "principled"),)
    assert kinds.import_notes() == ()
    assert kinds.unknown("ScopeGuardOrbital") == (
        "ScopeGuardOrbital is not an orbital this studio can add. The orbitals of "
        "chatomon.orbitals.principled need: pip install 'chatomon[principled]'."
    )


def test_a_module_that_fails_to_import_is_a_note(without):
    without("chatomon.orbitals.principled", RuntimeError("broken install"))
    assert ScopeGuardOrbital not in kinds.classes()
    assert kinds.extras() == ()
    assert kinds.import_notes() == (
        "chatomon.orbitals.principled could not be imported: RuntimeError: broken install",
    )


def test_a_class_the_studio_does_not_know_is_refused_plainly():
    assert kinds.unknown("Nope") == "Nope cannot be added from the studio: add it in your code"


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


def test_a_scope_guard_is_read_with_its_blocks():
    assert shapes(ScopeGuardOrbital) == {
        "guard": ("block", False, True),
        "description": ("block", False, True),
        "history_window": ("int", False, False),
        "timeout": ("number", False, False),
        "on_error": ("choice", False, False),
    }
    kind = kinds.describe(ScopeGuardOrbital)
    assert (kind.objects, kind.addable) == (("mapping", "headers"), True)
    guard, description = kind.setting("guard"), kind.setting("description")
    assert guard is not None and description is not None
    assert (guard.text, [found.cls for found in guard.types]) == (False, [ApiBackend])
    assert (description.text, [found.cls for found in description.types]) == (
        True,
        [AIServiceDescription, AIServiceDescriptionV2],
    )
    connection = guard.types[0]
    assert [(field.key, field.type, field.required) for field in connection.fields] == [
        ("api_url", "text", True),
        ("api_key_env", "env", False),
        ("model", "text", False),
    ]


def test_a_claim_extractor_is_read_with_its_blocks():
    kind = kinds.describe(ClaimExtractorOrbital)
    assert (kind.objects, kind.addable) == (("headers",), True)
    assert shapes(ClaimExtractorOrbital)["extractor"] == ("block", False, True)
    assert shapes(ClaimExtractorOrbital)["description"] == ("block", True, False)
    assert shapes(ClaimExtractorOrbital)["intents_only"] == ("bool", False, False)
    description = kind.setting("description")
    assert description is not None
    principles = description.types[0].field("principles")
    assert principles is not None and (principles.type, principles.text) == ("items", True)
    # `supporting_materials` has no default and accepts none: the studio gives none.
    assert principles.types[0].unread == (("supporting_materials", None, True),)


def test_a_client_from_code_is_an_object_and_a_description_is_a_block():
    overrides: dict[Any, Any] = {"Chit Chat": Verdict.block()}
    guard = ScopeGuardOrbital(guard=CLIENT, description=SERVICE, mapping=overrides)
    assert kinds.read_only(guard, CONNECTOR) is None
    assert kinds.opaque(guard) == ("guard",)
    read = kinds.values(guard)
    assert "guard" not in read
    assert read["description"]["type"] == "AIServiceDescriptionV2"
    assert read["description"]["fields"]["predefined_responses"] == [
        {
            "type": "PredefinedResponse",
            "fields": {"trigger": "The user asks for the phone number", "response": "800 123 456"},
        },
        "Greet the user back.",
    ]
    # Every argument is read from the attribute named like it: the developer's own objects.
    assert kinds.arguments(guard, CONNECTOR)["mapping"] is overrides
    assert kinds.arguments(guard, CONNECTOR)["guard"] is CLIENT
    shorter = kinds.rebuilt(guard, {"history_window": 2}, CONNECTOR)
    assert isinstance(shorter, ScopeGuardOrbital)
    assert (shorter.guard, shorter.description, shorter.rules, shorter.history_window) == (
        CLIENT,
        SERVICE,
        guard.rules,
        2,
    )
    with pytest.raises(InvalidSettingError) as caught:
        kinds.rebuilt(
            guard, {"guard": {"type": "ApiBackend", "fields": {"api_url": "http://a"}}}, CONNECTOR
        )
    assert (str(caught.value), caught.value.setting) == (
        "`guard` is an object from your code: change it there",
        "guard",
    )
    with pytest.raises(InvalidSettingError, match="`mapping` is an object from your code"):
        kinds.rebuilt(guard, {"mapping": {}}, CONNECTOR)


def test_a_block_is_edited_whole_and_untouched_fields_keep_their_values():
    text = AIServiceDescriptionV2(
        identity_role="Courier", context="Italy", constraints="No refunds"
    )
    guard = ScopeGuardOrbital(guard=CLIENT, description=text)
    shown = kinds.values(guard)["description"]
    assert shown["fields"]["constraints"] == ["No refunds"]
    shown["fields"]["context"] = "Italy and France"
    edited = kinds.rebuilt(guard, {"description": shown}, CONNECTOR)
    assert isinstance(edited, ScopeGuardOrbital) and edited.description is not text
    assert isinstance(edited.description, AIServiceDescriptionV2)
    # Shown as lines, the text is still the text the code has.
    assert (edited.description.context, edited.description.constraints) == (
        "Italy and France",
        "No refunds",
    )
    assert edited.fingerprint() != guard.fingerprint()

    shown["fields"]["context"] = "Italy"
    back = kinds.rebuilt(edited, {"description": shown}, CONNECTOR, base=guard)
    assert isinstance(back, ScopeGuardOrbital) and back.description is text

    as_text = kinds.rebuilt(guard, {"description": "A courier."}, CONNECTOR)
    assert isinstance(as_text, ScopeGuardOrbital) and as_text.description == "A courier."


def test_a_block_the_constructors_refuse_carries_their_messages():
    guard = ScopeGuardOrbital(guard=ApiBackend(api_url="https://scope.example"), description="x")
    cases = [
        ({"guard": {"type": "ApiBackend", "fields": {}}}, "`api_url` needs a value"),
        (
            {"guard": {"type": "ApiBackend", "fields": {"api_url": "scope.example"}}},
            "api_url must start with http:// or https:// and name a host",
        ),
        (
            {
                "description": {
                    "type": "AIServiceDescription",
                    "fields": {"identity_role": "r", "context": "c"},
                }
            },
            "ScopeGuard V2 needs an AIServiceDescriptionV2 (or a string) as description",
        ),
        (
            {"description": 3},
            "`description` must be text, an AIServiceDescription or an AIServiceDescriptionV2",
        ),
    ]
    for changes, message in cases:
        with pytest.raises(InvalidSettingError) as caught:
            kinds.rebuilt(guard, changes, CONNECTOR)
        assert (str(caught.value), caught.value.setting) == (message, next(iter(changes)))


def test_creating_an_orbital_with_blocks(monkeypatch):
    kind = kinds.describe(ScopeGuardOrbital)
    backend = {"type": "ApiBackend", "fields": {"api_url": "https://scope.example"}}
    with pytest.raises(InvalidSettingError) as caught:
        kinds.created(kind, "scope_guard", {"guard": backend})
    assert (str(caught.value), caught.value.setting) == (
        "`description` needs a value",
        "description",
    )
    made = kinds.created(
        kind,
        "scope_guard",
        {
            "guard": backend,
            "description": {
                "type": "AIServiceDescriptionV2",
                "fields": {
                    "identity_role": "Courier",
                    "context": "Italy",
                    "predefined_responses": [
                        {"type": "PredefinedResponse", "fields": {"trigger": "t", "response": "r"}}
                    ],
                },
            },
        },
    )
    assert isinstance(made, ScopeGuardOrbital)
    assert made.guard == ApiBackend(api_url="https://scope.example")
    assert made.description == AIServiceDescriptionV2(
        identity_role="Courier",
        context="Italy",
        predefined_responses=[PredefinedResponse(trigger="t", response="r")],
    )
    assert kinds.read_only(made, CONNECTOR) is None and kinds.opaque(made) == ()

    # What the type of a block refuses is said under that block; what the orbital refuses
    # once it has every value belongs to no setting.
    with pytest.raises(InvalidSettingError) as typed:
        kinds.created(
            kind,
            "scope_guard",
            {"guard": {"type": "ApiBackend", "fields": {"api_url": "x"}}, "description": "x"},
        )
    assert (str(typed.value), typed.value.setting) == (
        "api_url must start with http:// or https:// and name a host",
        "guard",
    )
    old = {"type": "AIServiceDescription", "fields": {"identity_role": "r", "context": "c"}}
    with pytest.raises(InvalidSettingError) as whole:
        kinds.created(kind, "scope_guard", {"guard": backend, "description": old})
    assert whole.value.setting is None

    monkeypatch.delenv("SCOPE_KEY", raising=False)
    named = {
        "type": "ApiBackend",
        "fields": {"api_url": "https://scope.example", "api_key_env": "SCOPE_KEY"},
    }
    with pytest.raises(InvalidSettingError, match="the environment variable SCOPE_KEY is not set"):
        kinds.created(kind, "scope_guard", {"guard": named, "description": "A courier."})

    extractor = kinds.created(
        kinds.describe(ClaimExtractorOrbital),
        "claims",
        {
            "extractor": {"type": "ApiBackend", "fields": {"api_url": "https://claims.example"}},
            "description": {
                "type": "AIServiceDescription",
                "fields": {
                    "identity_role": "Courier",
                    "context": "Italy",
                    "principles": [
                        {
                            "type": "Principle",
                            "fields": {"title": "Honest", "description": "No guesses"},
                        }
                    ],
                },
            },
        },
    )
    assert isinstance(extractor, ClaimExtractorOrbital)
    assert isinstance(extractor.description, AIServiceDescription)
    assert extractor.description.principles == [
        Principle(title="Honest", description="No guesses", supporting_materials=None)
    ]


def test_a_switch_takes_true_or_false():
    flag = values.Setting("intents_only", "bool", False, False, False)
    values.check_shape(flag, True)
    values.check_shape(flag, False)
    for value in ("yes", 1, 0):
        with pytest.raises(InvalidSettingError) as caught:
            values.check_shape(flag, value)
        assert str(caught.value) == "`intents_only` must be true or false"
