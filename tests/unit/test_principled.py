import dataclasses
from typing import Any

import aiohttp
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
    ApiBackend,
    ClaimExtractorOrbital,
    ScopeGuardOrbital,
)
from chatomon.schemas import Message, OrbitalContext
from tests.fake_service import CLAIM_EXTRACTOR, SCOPE_GUARD, closed_url
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


async def test_whitespace_suggested_response_is_an_error():
    orb = ScopeGuardOrbital(
        guard=v2_guard(ScopeClass.PREDEFINED_ANSWER, suggested="  "), description=DESCRIPTION
    )
    with pytest.raises(ValueError, match="suggested_response"):
        await orb.check(ctx())


async def test_slow_sync_headers_provider_is_cut_off_by_the_timeout():
    import time

    from tests.fakes import make_engine

    def slow_headers() -> dict[str, str]:
        time.sleep(0.5)
        return {"Authorization": "Bearer t"}

    orb = ScopeGuardOrbital(
        guard=v2_guard(ScopeClass.DIRECTLY_SUPPORTED),
        description=DESCRIPTION,
        headers=slow_headers,
        timeout=0.1,
    )
    engine = make_engine(inputs=[orb])
    turn = engine.start_turn()
    started = time.perf_counter()
    decision = await engine.run_input(turn, message="hi")
    assert time.perf_counter() - started < 0.4
    assert turn.input_results[0].error_type == "TimeoutError"
    assert decision.kind == "reply"


async def test_refreshed_headers_reach_the_http_request(monkeypatch):
    """Pins the orbitals-internal detail spec §8.4 relies on: custom_headers is read per request."""
    import aiohttp

    sent: list[dict[str, str]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        async def json(self) -> dict[str, Any]:
            return {"scope_class": "Directly Supported", "model": "scope-guard-v2"}

    async def fake_post(self: Any, url: str, **kwargs: Any) -> FakeResponse:
        sent.append(dict(kwargs["headers"]))
        return FakeResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "post", fake_post)
    tokens = iter(["token-1", "token-2"])
    orb = ScopeGuardOrbital(
        guard=AsyncScopeGuardV2(backend="api", api_url="http://scope-guard.test", api_key="k"),
        description=DESCRIPTION,
        headers=lambda: {"Authorization": f"Bearer {next(tokens)}"},
    )
    await orb.check(ctx())
    await orb.check(ctx())
    assert [headers["Authorization"] for headers in sent] == ["Bearer token-1", "Bearer token-2"]
    assert all(headers["X-API-Key"] == "k" for headers in sent)


def test_scope_guard_keeps_the_mapping_it_was_given():
    overrides: dict[Any, Any] = {"Chit Chat": Verdict.block()}
    guard = v2_guard(ScopeClass.CHIT_CHAT)
    orbital = ScopeGuardOrbital(guard=guard, description=DESCRIPTION, mapping=overrides)
    assert orbital.mapping is overrides
    assert orbital.rules is not overrides  # the rules are its own, with the defaults
    assert ScopeGuardOrbital(guard=guard, description=DESCRIPTION).mapping is None


def test_the_principled_orbitals_call_a_model():
    assert ScopeGuardOrbital.calls_model is True
    assert ClaimExtractorOrbital.calls_model is True


# --- ApiBackend: a service named by plain values ------------------------------

KEY = "key-0123456789-not-for-the-page"


def test_api_backend_is_a_frozen_value_with_keyword_arguments():
    backend = ApiBackend(api_url="https://scope.example", api_key_env="SCOPE_KEY", model="m")
    assert (backend.api_url, backend.api_key_env, backend.model) == (
        "https://scope.example",
        "SCOPE_KEY",
        "m",
    )
    assert backend == ApiBackend(
        api_url="https://scope.example", api_key_env="SCOPE_KEY", model="m"
    )
    assert ApiBackend(api_url="http://127.0.0.1:8080").api_key_env is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        backend.api_url = "https://other.example"  # type: ignore[misc]
    with pytest.raises(TypeError):
        ApiBackend("https://scope.example")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "match", "secret"),
    [
        ({"api_url": "scope.example"}, "must start with http:// or https://", "scope.example"),
        ({"api_url": "ftp://scope.example"}, "must start with http:// or https://", "ftp"),
        ({"api_url": "https://"}, "name a host", None),
        ({"api_url": "http://host:port"}, "name a host", "port"),
        ({"api_url": "https://scope.example/a b"}, "name a host", "a b"),
        ({"api_url": "https://scope.example/\x1b[2J"}, "name a host", "2J"),
        ({"api_url": 8080}, "must start with http:// or https://", None),
        ({"api_url": "https://user:hunter2@scope.example"}, "user or a password", "hunter2"),
        ({"api_url": "ftp://user:hunter2@scope.example"}, "user or a password", "hunter2"),
        ({"api_url": "https://scope.example", "api_key_env": KEY}, "name of an", KEY),
        ({"api_url": "https://scope.example", "api_key_env": ""}, "name of an", None),
        ({"api_url": "https://scope.example", "api_key_env": 3}, "name of an", None),
        ({"api_url": "https://scope.example", "model": 3}, "model must be text", None),
    ],
)
def test_api_backend_refuses_without_quoting_what_it_refuses(kwargs, match, secret):
    with pytest.raises(ChatomonConfigError, match=match) as info:
        ApiBackend(**kwargs)
    if secret is not None:
        assert secret not in str(info.value)


@pytest.mark.parametrize(
    ("kwargs", "message", "secret"),
    [
        (
            # A port that is not a number does not hide the user and the password.
            {"api_url": "http://user:hunter2@scope.example:abc"},
            "api_url must not hold a user or a password: "
            "name an environment variable in api_key_env",
            "hunter2",
        ),
        (
            # It could not work, the client writes its path after it, and it would be shown.
            {"api_url": "https://scope.example/api?code=hunter2"},
            "api_url must not hold a query or a fragment",
            "hunter2",
        ),
        (
            {"api_url": "https://scope.example/api#hunter2"},
            "api_url must not hold a query or a fragment",
            "hunter2",
        ),
        (
            {"api_url": " https://scope.example"},
            "api_url must not start or end with a space",
            None,
        ),
        (
            {"api_url": "https://scope.example\n"},
            "api_url must not start or end with a space",
            None,
        ),
        (
            {"api_url": "https://scope.example", "api_key_env": "SCOPE_KEY "},
            "api_key_env must not start or end with a space",
            None,
        ),
    ],
    ids=["port", "query", "fragment", "before", "after", "name"],
)
def test_api_backend_says_what_is_wrong_with_what_it_refuses(kwargs, message, secret):
    with pytest.raises(ChatomonConfigError) as info:
        ApiBackend(**kwargs)
    assert str(info.value) == message
    if secret is not None:
        assert secret not in str(info.value)


def test_a_name_that_is_not_set_is_named_also_when_it_is_a_key_written_there(monkeypatch):
    """Pins a limit: a key that reads like a name cannot be told from one, and is repeated."""
    pasted = "a3f9c0d1e5b7"
    monkeypatch.delenv(pasted, raising=False)
    with pytest.raises(ChatomonConfigError) as info:
        ScopeGuardOrbital(
            guard=ApiBackend(api_url="https://scope.example", api_key_env=pasted),
            description="A courier.",
        )
    assert str(info.value) == f"the environment variable {pasted} is not set"


async def test_a_predefined_answer_of_the_service_is_the_orbitals_reply(service, monkeypatch):
    monkeypatch.setenv("SCOPE_KEY", KEY)
    service.key = KEY
    service.scope_class = "Predefined Answer"
    service.suggested_response = "800 123 456"
    orb = ScopeGuardOrbital(
        guard=ApiBackend(api_url=service.url, api_key_env="SCOPE_KEY"), description=DESCRIPTION
    )
    verdict = await orb.check(ctx("What is your phone number?"))
    assert (verdict.action, verdict.reason, verdict.reply) == (
        "respond",
        "Predefined Answer",
        "800 123 456",
    )


def test_api_backend_does_not_read_the_environment(monkeypatch):
    monkeypatch.delenv("SCOPE_KEY", raising=False)
    assert ApiBackend(api_url="https://scope.example", api_key_env="SCOPE_KEY").api_key_env


async def test_scope_guard_built_from_an_api_backend_calls_the_service(service, monkeypatch):
    monkeypatch.setenv("SCOPE_KEY", KEY)
    service.key = KEY
    service.scope_class = "Chit Chat"
    backend = ApiBackend(api_url=service.url, api_key_env="SCOPE_KEY", model="scope-guard-q")
    orb = ScopeGuardOrbital(guard=backend, description=DESCRIPTION)
    assert orb.guard is backend
    assert orb.is_v2 is True
    verdict = await orb.check(ctx("Hi"))
    assert (verdict.action, verdict.reason, verdict.guidance) == (
        "allow",
        "Chit Chat",
        CHIT_CHAT_GUIDANCE,
    )
    request = service.requests[0]
    assert request["path"] == SCOPE_GUARD
    assert request["headers"]["X-API-Key"] == KEY
    assert request["body"]["model"] == "scope-guard-q"
    assert request["body"]["ai_service_description"] == DESCRIPTION.model_dump()
    assert request["body"]["conversation"][-1] == {"role": "user", "content": "Hi"}


async def test_the_variable_is_read_when_the_orbital_is_built(service, monkeypatch):
    monkeypatch.setenv("SCOPE_KEY", KEY)
    service.key = KEY
    orb = ScopeGuardOrbital(
        guard=ApiBackend(api_url=service.url, api_key_env="SCOPE_KEY"), description="A courier."
    )
    monkeypatch.delenv("SCOPE_KEY")
    assert (await orb.check(ctx())).action == "allow"


@pytest.mark.parametrize("value", [None, ""])
@pytest.mark.parametrize("kind", ["scope_guard", "claim_extractor"])
def test_a_variable_that_is_not_set_is_refused_by_name(monkeypatch, value, kind):
    if value is None:
        monkeypatch.delenv("SCOPE_KEY", raising=False)
    else:
        monkeypatch.setenv("SCOPE_KEY", value)
    backend = ApiBackend(api_url="https://scope.example", api_key_env="SCOPE_KEY")
    with pytest.raises(ChatomonConfigError, match="the environment variable SCOPE_KEY is not set"):
        if kind == "scope_guard":
            ScopeGuardOrbital(guard=backend, description="A courier.")
        else:
            ClaimExtractorOrbital(extractor=backend)


async def test_without_a_variable_the_client_behaves_as_the_library_does(service, monkeypatch):
    monkeypatch.delenv("PRINCIPLED_API_KEY", raising=False)
    plain = ScopeGuardOrbital(guard=ApiBackend(api_url=service.url), description="A courier.")
    await plain.check(ctx())
    assert "X-API-Key" not in service.requests[0]["headers"]
    assert "model" not in service.requests[0]["body"]
    monkeypatch.setenv("PRINCIPLED_API_KEY", KEY)
    library = ScopeGuardOrbital(guard=ApiBackend(api_url=service.url), description="A courier.")
    await library.check(ctx())
    assert service.requests[1]["headers"]["X-API-Key"] == KEY


def test_an_api_backend_gives_the_fingerprint_of_a_hand_made_client(monkeypatch):
    monkeypatch.setenv("SCOPE_KEY", KEY)
    url = "https://scope.example"
    backend = ApiBackend(api_url=url, api_key_env="SCOPE_KEY", model="scope-guard-q")
    short = ScopeGuardOrbital(guard=backend, description=DESCRIPTION)
    by_hand = ScopeGuardOrbital(
        guard=AsyncScopeGuardV2(backend="api", api_url=url, model="scope-guard-q", api_key="other"),
        description=DESCRIPTION,
    )
    assert short.fingerprint() == by_hand.fingerprint()
    assert short.config()["guard"] == {
        "type": "AsyncAPIScopeGuardV2",
        "model": "scope-guard-q",
        "api_url": url,
    }
    written = str(short.config())
    assert KEY not in written and "SCOPE_KEY" not in written
    renamed = ScopeGuardOrbital(
        guard=ApiBackend(api_url=url, model="scope-guard-q"), description=DESCRIPTION
    )
    assert renamed.fingerprint() == short.fingerprint()


async def test_headers_and_mapping_work_next_to_an_api_backend(service):
    service.scope_class = "Chit Chat"
    orb = ScopeGuardOrbital(
        guard=ApiBackend(api_url=service.url),
        description="A courier.",
        headers=lambda: {"Authorization": "Bearer token-1"},
        mapping={"Chit Chat": Verdict.block("Only parcels.")},
    )
    verdict = await orb.check(ctx("Hi"))
    assert (verdict.action, verdict.reply) == ("block", "Only parcels.")
    assert service.requests[0]["headers"]["Authorization"] == "Bearer token-1"


def test_an_api_backend_is_a_v2_guard():
    with pytest.raises(ChatomonConfigError, match="AIServiceDescriptionV2"):
        ScopeGuardOrbital(
            guard=ApiBackend(api_url="https://scope.example"),
            description=AIServiceDescription(identity_role="x", context="y"),
        )


def test_a_guard_that_is_neither_a_client_nor_a_backend_is_refused():
    with pytest.raises(ChatomonConfigError, match="or an ApiBackend, got str"):
        ScopeGuardOrbital(guard="https://scope.example", description="x")  # type: ignore[arg-type]
    with pytest.raises(ChatomonConfigError, match="or an ApiBackend, got str"):
        ClaimExtractorOrbital(extractor="https://claims.example")  # type: ignore[arg-type]


@pytest.mark.parametrize("model", [None, "claims-q"])
async def test_claim_extractor_built_from_an_api_backend(service, monkeypatch, model):
    # Without a model the library's own default is used, whatever name it maps it to.
    sent = model or AsyncClaimExtractor(backend="api", api_url=service.url).default_model
    monkeypatch.setenv("CLAIMS_KEY", KEY)
    service.key = KEY
    backend = ApiBackend(api_url=service.url, api_key_env="CLAIMS_KEY", model=model)
    orb = ClaimExtractorOrbital(extractor=backend, description="A courier.")
    assert orb.extractor is backend
    verdict = await orb.check(ctx("We are open 24/7.", side="output"))
    assert verdict.action == "allow"
    assert isinstance(verdict.annotations, Extractions)
    assert verdict.annotations.claims[0].content == "Open 24/7"
    request = service.requests[0]
    assert request["path"] == CLAIM_EXTRACTOR
    assert request["headers"]["X-API-Key"] == KEY
    assert request["body"]["model"] == sent
    assert request["body"]["ai_service_description"] == "A courier."
    assert orb.config()["extractor"] == {
        "type": "AsyncAPIClaimExtractor",
        "model": sent,
        "api_url": service.url,
    }


async def test_the_text_of_a_refused_key_does_not_quote_the_key(service, monkeypatch):
    """Pins what the studio relies on: the client's error text holds no key."""
    monkeypatch.setenv("SCOPE_KEY", KEY)
    service.key = "another-key"
    orb = ScopeGuardOrbital(
        guard=ApiBackend(api_url=service.url, api_key_env="SCOPE_KEY"), description="A courier."
    )
    with pytest.raises(aiohttp.ClientResponseError) as info:
        await orb.check(ctx())
    assert info.value.status == 401
    assert KEY not in str(info.value)


async def test_the_text_of_a_refused_connection_does_not_quote_the_key(monkeypatch):
    monkeypatch.setenv("SCOPE_KEY", KEY)
    orb = ScopeGuardOrbital(
        guard=ApiBackend(api_url=closed_url(), api_key_env="SCOPE_KEY"), description="A courier."
    )
    with pytest.raises(OSError) as info:
        await orb.check(ctx())
    assert KEY not in str(info.value) and KEY not in repr(info.value)
