from typing import Any

import pytest
from orbitals.claim_extractor import AsyncClaimExtractor, Claim, ClaimExtractorOutput, Extractions
from orbitals.scope_guard import AsyncScopeGuard, ScopeGuardOutput
from orbitals.scope_guard import ScopeClass as ScopeClassV1
from orbitals.scope_guard_v2 import AsyncScopeGuardV2, ScopeClass, ScopeGuardV2, ScopeGuardV2Output
from orbitals.types import AIServiceDescription, AIServiceDescriptionV2

from chatomon import Atom, ChatomonConfigError, Verdict
from chatomon.orbitals.principled import (
    CHIT_CHAT_GUIDANCE,
    OUT_OF_SCOPE_GUIDANCE,
    V2_OUTPUT_FIELDS,
    ClaimExtractorOrbital,
    ScopeGuardOrbital,
)
from chatomon.schemas import Message, OrbitalContext
from tests.fakes import FakeConnector

DESCRIPTION = AIServiceDescriptionV2(
    identity_role="Parcel tracking assistant", context="SpeedyPack, an Italian courier"
)
HISTORY = tuple(
    Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}") for i in range(8)
)


def ctx(
    message: str = "Where is my parcel?", *, side: str = "input", history=HISTORY
) -> OrbitalContext:
    return OrbitalContext(side=side, message=message, history=tuple(history), turn_id="t1")  # type: ignore[arg-type]


def v2_guard(
    scope: ScopeClass, *, suggested: str | None = "Here is the answer.", calls=None
) -> AsyncScopeGuardV2:
    guard = AsyncScopeGuardV2(backend="api", api_url="http://scope-guard.test", api_key="api-key")

    async def fake_validate(conversation: Any, **kwargs: Any) -> ScopeGuardV2Output:
        if calls is not None:
            calls.append(
                {"conversation": conversation, "headers": dict(guard.custom_headers), **kwargs}
            )
        return ScopeGuardV2Output(
            scope_class=scope,
            evidences=["Parcel tracking assistant"],
            reasoning="because",
            suggested_response=suggested,
            model="scope-guard-v2",
        )

    guard.__dict__["_validate"] = fake_validate
    return guard


@pytest.mark.parametrize(
    ("scope", "action", "guidance", "reply", "escalate"),
    [
        (ScopeClass.DIRECTLY_SUPPORTED, "allow", None, None, False),
        (ScopeClass.POTENTIALLY_SUPPORTED, "allow", None, None, False),
        (ScopeClass.CHIT_CHAT, "allow", CHIT_CHAT_GUIDANCE, None, False),
        (ScopeClass.PREDEFINED_ANSWER, "respond", None, "Here is the answer.", False),
        (ScopeClass.HUMAN_OVERSIGHT, "respond", None, "Here is the answer.", True),
        (ScopeClass.OUT_OF_SCOPE, "block", OUT_OF_SCOPE_GUIDANCE, None, False),
        (ScopeClass.RESTRICTED, "block", None, None, False),
    ],
)
async def test_default_mapping(scope, action, guidance, reply, escalate):
    verdict = await ScopeGuardOrbital(guard=v2_guard(scope), description=DESCRIPTION).check(ctx())
    assert (verdict.action, verdict.guidance, verdict.reply, verdict.escalate) == (
        action,
        guidance,
        reply,
        escalate,
    )
    assert verdict.reason == scope.value
    assert verdict.annotations["scope_class"] == scope.value
    assert verdict.annotations["reasoning"] == "because"


async def test_request_contains_window_description_and_output_fields():
    calls: list[dict[str, Any]] = []
    orb = ScopeGuardOrbital(
        guard=v2_guard(ScopeClass.DIRECTLY_SUPPORTED, calls=calls),
        description=DESCRIPTION,
        history_window=2,
    )
    await orb.check(ctx("And now?"))
    call = calls[0]
    assert [(m.role, m.content) for m in call["conversation"]] == [
        ("user", "m6"),
        ("assistant", "m7"),
        ("user", "And now?"),
    ]
    assert call["ai_service_description"] == DESCRIPTION
    assert tuple(call["output_fields"]) == V2_OUTPUT_FIELDS


async def test_predefined_answer_without_suggested_response_is_an_error():
    orb = ScopeGuardOrbital(
        guard=v2_guard(ScopeClass.PREDEFINED_ANSWER, suggested=None), description=DESCRIPTION
    )
    with pytest.raises(ValueError, match="suggested_response"):
        await orb.check(ctx())


async def test_mapping_overrides_accept_enums_strings_and_callables():
    orb = ScopeGuardOrbital(
        guard=v2_guard(ScopeClass.CHIT_CHAT),
        description=DESCRIPTION,
        mapping={
            ScopeClass.CHIT_CHAT: lambda result: Verdict.respond(f"Hi! ({result.model})"),
            "Out of Scope": Verdict.block("Only parcels."),
        },
    )
    verdict = await orb.check(ctx())
    assert verdict.reply == "Hi! (scope-guard-v2)"
    assert orb.rules["Out of Scope"] == Verdict.block("Only parcels.")


def test_unknown_mapping_key_is_rejected():
    with pytest.raises(ChatomonConfigError, match="unknown ScopeGuard class"):
        ScopeGuardOrbital(
            guard=v2_guard(ScopeClass.CHIT_CHAT),
            description=DESCRIPTION,
            mapping={"Maybe": Verdict.allow()},
        )


async def test_v1_guard_is_supported():
    guard = AsyncScopeGuard(backend="api", api_url="http://scope-guard.test")
    calls: list[dict[str, Any]] = []

    async def fake_validate(conversation: Any, **kwargs: Any) -> ScopeGuardOutput:
        calls.append(kwargs)
        return ScopeGuardOutput(
            scope_class=ScopeClassV1.OUT_OF_SCOPE, evidences=None, model="v1", usage=None
        )

    guard.__dict__["_validate"] = fake_validate
    description = AIServiceDescription(
        identity_role="Parcel tracking assistant", context="SpeedyPack"
    )
    orb = ScopeGuardOrbital(guard=guard, description=description)
    verdict = await orb.check(ctx())
    assert not orb.is_v2
    assert verdict.guidance == OUT_OF_SCOPE_GUIDANCE
    assert "output_fields" not in calls[0]


async def test_sync_guard_runs_in_a_thread():
    guard = ScopeGuardV2(backend="api", api_url="http://scope-guard.test")

    def fake_validate(conversation: Any, **kwargs: Any) -> ScopeGuardV2Output:
        return ScopeGuardV2Output(scope_class=ScopeClass.RESTRICTED, reasoning="r", model="m")

    guard.__dict__["_validate"] = fake_validate
    verdict = await ScopeGuardOrbital(guard=guard, description=DESCRIPTION).check(ctx())
    assert verdict.action == "block"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_headers_are_refreshed_before_every_call(asynchronous):
    calls: list[dict[str, Any]] = []
    tokens = iter(["token-1", "token-2"])

    def sync_headers() -> dict[str, str]:
        return {"Authorization": f"Bearer {next(tokens)}"}

    async def async_headers() -> dict[str, str]:
        return sync_headers()

    orb = ScopeGuardOrbital(
        guard=v2_guard(ScopeClass.DIRECTLY_SUPPORTED, calls=calls),
        description=DESCRIPTION,
        headers=async_headers if asynchronous else sync_headers,
    )
    await orb.check(ctx())
    await orb.check(ctx())
    assert [call["headers"]["Authorization"] for call in calls] == [
        "Bearer token-1",
        "Bearer token-2",
    ]
    assert all(call["headers"]["X-API-Key"] == "api-key" for call in calls)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"guard": object()}, "ScopeGuard client"),
        (
            {"description": AIServiceDescription(identity_role="x", context="y")},
            "AIServiceDescriptionV2",
        ),
        ({"history_window": -1}, "history_window"),
    ],
)
def test_invalid_scope_guard_settings(kwargs, match):
    settings: dict[str, Any] = {"guard": v2_guard(ScopeClass.CHIT_CHAT), "description": DESCRIPTION}
    settings.update(kwargs)
    with pytest.raises(ChatomonConfigError, match=match):
        ScopeGuardOrbital(**settings)


def test_headers_need_an_api_client():
    guard = v2_guard(ScopeClass.CHIT_CHAT)
    del guard.__dict__["custom_headers"]
    with pytest.raises(ChatomonConfigError, match="backend='api'"):
        ScopeGuardOrbital(guard=guard, description=DESCRIPTION, headers=lambda: {})


def test_scope_guard_cannot_be_an_output_orbital():
    orb = ScopeGuardOrbital(guard=v2_guard(ScopeClass.CHIT_CHAT), description=DESCRIPTION)
    atom = Atom(name="a", instructions="i", connector=FakeConnector(), output_orbitals=[orb])
    with pytest.raises(ChatomonConfigError, match="cannot run on the output side"):
        atom.build()


def test_fingerprint_follows_the_description():
    guard = v2_guard(ScopeClass.CHIT_CHAT)
    same = ScopeGuardOrbital(guard=guard, description=DESCRIPTION).fingerprint()
    assert (
        same
        == ScopeGuardOrbital(
            guard=v2_guard(ScopeClass.CHIT_CHAT), description=DESCRIPTION
        ).fingerprint()
    )
    other = AIServiceDescriptionV2(identity_role="Other", context="Other")
    assert same != ScopeGuardOrbital(guard=guard, description=other).fingerprint()


def claim_extractor(calls: list[dict[str, Any]]) -> AsyncClaimExtractor:
    extractor = AsyncClaimExtractor(backend="api", api_url="http://claims.test")

    async def fake_extract(conversation: Any, **kwargs: Any) -> ClaimExtractorOutput:
        calls.append({"conversation": conversation, **kwargs})
        return ClaimExtractorOutput(
            extractions=Extractions(claims=[Claim(subtype="Factoid", content="Open 24/7")]),
            model="claim-extractor",
            usage=None,
        )

    extractor.__dict__["_extract"] = fake_extract
    return extractor


@pytest.mark.parametrize(("side", "role"), [("input", "user"), ("output", "assistant")])
async def test_claim_extractor_annotates_the_last_message(side, role):
    calls: list[dict[str, Any]] = []
    orb = ClaimExtractorOrbital(
        extractor=claim_extractor(calls), history_window=1, intents_only=True
    )
    verdict = await orb.check(ctx("We are open 24/7.", side=side))
    assert verdict.action == "allow"
    assert isinstance(verdict.annotations, Extractions)
    assert verdict.annotations.claims[0].content == "Open 24/7"
    conversation = calls[0]["conversation"]
    assert [(m.role, m.content) for m in conversation] == [
        ("assistant", "m7"),
        (role, "We are open 24/7."),
    ]
    assert calls[0]["intents_only"] is True
    assert orb.name == "claim_extractor"
