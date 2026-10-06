import pytest

from chatomon import ChatomonConfigError, Orbital, Verdict, orbital
from chatomon.orbitals import (
    FunctionOrbital,
    JudgeOrbital,
    MessageLengthOrbital,
    TurnLimitOrbital,
)
from chatomon.schemas import OrbitalContext
from tests.fakes import ScriptedOrbital


class ScopeGuardOrbital(Orbital):
    calls_model = False

    async def check(self, ctx: OrbitalContext) -> Verdict:
        return Verdict.allow()


def make_ctx(message: str = "hi") -> OrbitalContext:
    return OrbitalContext(side="input", message=message, turn_id="t1")


def test_default_name_is_snake_case_without_suffix():
    assert ScopeGuardOrbital().name == "scope_guard"
    assert ScriptedOrbital().name == "scripted"


def test_defaults():
    orb = ScopeGuardOrbital()
    assert orb.reads == ()
    assert orb.timeout == 30.0
    assert orb.on_error == "block"
    assert orb.sides == frozenset({"input", "output"})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "has space"}, "invalid orbital name"),
        ({"name": ""}, "invalid orbital name"),
        ({"timeout": 0}, "timeout must be positive"),
        ({"on_error": "ignore"}, "on_error"),
        ({"reads": "claims"}, "list of orbital names"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, match):
    with pytest.raises(ChatomonConfigError, match=match):
        ScopeGuardOrbital(**kwargs)


def test_fingerprint_is_stable_and_config_sensitive():
    assert ScopeGuardOrbital().fingerprint() == ScopeGuardOrbital().fingerprint()
    assert ScopeGuardOrbital().fingerprint() != ScopeGuardOrbital(timeout=5).fingerprint()
    assert ScopeGuardOrbital().fingerprint().startswith("sha256:")


def test_bind_is_a_no_op_by_default():
    assert ScopeGuardOrbital().bind(object()) is None  # type: ignore[arg-type]


async def test_decorator_without_arguments_uses_the_function_name():
    @orbital
    async def no_competitors(ctx: OrbitalContext) -> Verdict:
        return Verdict.block(guidance="no") if "amazon" in ctx.message else Verdict.allow()

    assert isinstance(no_competitors, FunctionOrbital)
    assert no_competitors.name == "no_competitors"
    assert (await no_competitors.check(make_ctx("amazon"))).action == "block"


async def test_decorator_with_arguments_and_a_sync_function():
    @orbital(name="length", timeout=2.0, on_error="allow", reads=["claims"])
    def length_limit(ctx: OrbitalContext) -> Verdict:
        return Verdict.block() if len(ctx.message) > 3 else Verdict.allow()

    assert length_limit.name == "length"
    assert length_limit.timeout == 2.0
    assert length_limit.on_error == "allow"
    assert length_limit.reads == ("claims",)
    assert (await length_limit.check(make_ctx("long message"))).action == "block"


async def test_function_orbital_rejects_non_verdict_results():
    @orbital
    async def broken(ctx: OrbitalContext) -> Verdict:
        return "allow"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="expected Verdict"):
        await broken.check(make_ctx())


def test_function_orbital_config_names_the_function():
    @orbital
    def checker(ctx: OrbitalContext) -> Verdict:
        return Verdict.allow()

    assert checker.config()["function"].endswith("checker")


def test_the_default_name_is_known_without_an_instance():
    assert ScopeGuardOrbital.default_name() == "scope_guard"
    assert ScriptedOrbital.default_name() == "scripted"
    assert JudgeOrbital.default_name() == "judge"
    assert MessageLengthOrbital.default_name() == "message_length"


def test_a_class_can_choose_its_default_name():
    class Renamed(ScriptedOrbital):
        @classmethod
        def default_name(cls) -> str:
            return "custom"

    assert Renamed().name == "custom"


def test_a_class_must_say_whether_it_calls_a_model():
    class Undeclared(Orbital):
        async def check(self, ctx: OrbitalContext) -> Verdict:
            return Verdict.allow()

    with pytest.raises(
        ChatomonConfigError,
        match=r"Undeclared must say whether it calls a model: "
        r"add calls_model = True or calls_model = False to the class",
    ):
        Undeclared()


def test_calls_model_is_true_or_false_and_nothing_else():
    class Vague(Orbital):
        calls_model = "sometimes"  # type: ignore[assignment]

        async def check(self, ctx: OrbitalContext) -> Verdict:
            return Verdict.allow()

    with pytest.raises(ChatomonConfigError, match="Vague must say whether it calls a model"):
        Vague()


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (MessageLengthOrbital, False),
        (TurnLimitOrbital, False),
        (FunctionOrbital, False),
        (JudgeOrbital, True),
    ],
)
def test_the_librarys_orbitals_say_whether_they_call_a_model(cls, expected):
    assert cls.calls_model is expected


def test_a_subclass_keeps_what_its_parent_declared():
    class Mine(MessageLengthOrbital):
        pass

    assert Mine(max_words=3).calls_model is False


def test_a_function_calls_no_model_unless_it_says_so():
    @orbital
    def plain(ctx: OrbitalContext) -> Verdict:
        return Verdict.allow()

    @orbital(calls_model=True)
    def remote(ctx: OrbitalContext) -> Verdict:
        return Verdict.allow()

    assert plain.calls_model is False
    assert remote.calls_model is True
    inline = FunctionOrbital(lambda ctx: Verdict.allow(), name="inline", calls_model=True)
    assert inline.calls_model is True
    assert FunctionOrbital.calls_model is False  # one function does not change the class


def test_calls_model_is_not_part_of_the_configuration():
    def check(ctx: OrbitalContext) -> Verdict:
        return Verdict.allow()

    quiet, remote = FunctionOrbital(check), FunctionOrbital(check, calls_model=True)
    assert "calls_model" not in remote.config()
    assert quiet.fingerprint() == remote.fingerprint()
