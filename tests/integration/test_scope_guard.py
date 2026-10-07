"""ScopeGuard V2 against a real deployment.

Run with, for example, the Cloud Run proxy of the deployment open:
    CHATOMON_INTEGRATION=1 CHATOMON_SCOPE_GUARD_URL=http://127.0.0.1:8080 \
        uv run pytest tests/integration -m integration -v
"""

from __future__ import annotations

import subprocess

import aiohttp
import pytest
from orbitals.scope_guard_v2 import AsyncScopeGuardV2
from orbitals.types import AIServiceDescriptionV2, PredefinedResponse
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom, TurnReport
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals.principled import ApiBackend, ScopeGuardOrbital
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER, OrbitalContext
from tests.integration.conftest import INTEGRATION_TIMEOUT, require_env, warm_up

pytestmark = pytest.mark.integration

DESCRIPTION = AIServiceDescriptionV2(
    identity_role="Customer assistant of SpeedyPack, an Italian parcel courier",
    context="Customers write in Italian or English about their shipments.",
    knowledge_scope="Parcel tracking, delivery times and opening hours of SpeedyPack points.",
    functionalities=["Track a parcel", "Explain delivery times", "Give opening hours"],
    constraints=["Never discuss refunds or compensation"],
    predefined_responses=[
        PredefinedResponse(
            trigger="The user asks for the customer service phone number",
            response="You can call SpeedyPack at 800 123 456, Monday to Friday 9-18.",
        )
    ],
    escalation_criteria=["The user reports a lost or damaged parcel"],
)

_warm: set[str] = set()


async def _scope_guard(url: str, **kwargs: object) -> ScopeGuardOrbital:
    orb = ScopeGuardOrbital(
        guard=AsyncScopeGuardV2(backend="api", api_url=url),
        description=DESCRIPTION,
        timeout=INTEGRATION_TIMEOUT,
        **kwargs,  # type: ignore[arg-type]
    )
    if url not in _warm:
        await warm_up(lambda: orb.check(_ctx("Where is my parcel?")))
        _warm.add(url)
    return orb


def _ctx(message: str) -> OrbitalContext:
    return OrbitalContext(side="input", message=message, turn_id="integration")


@pytest.mark.parametrize(
    ("message", "actions"),
    [
        ("Where is my parcel 123456?", {"allow"}),
        ("What's your customer service phone number?", {"respond"}),
        ("Can I get a refund for the late delivery?", {"block"}),
        ("Write me a poem about the sea.", {"block"}),
        ("thanks, bye!", {"allow"}),
        ("My parcel arrived damaged.", {"respond"}),
    ],
)
async def test_real_classification_maps_to_verdicts(message, actions):
    orb = await _scope_guard(require_env("CHATOMON_SCOPE_GUARD_URL"))
    verdict = await orb.check(_ctx(message))
    assert verdict.action in actions, (message, verdict.reason)
    if verdict.action == "respond":
        assert verdict.reply and verdict.reply.strip()
    assert verdict.annotations["reasoning"]


def _gcloud_identity_token() -> dict[str, str]:
    token = subprocess.run(
        ["gcloud", "auth", "print-identity-token"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"Authorization": f"Bearer {token}"}


async def test_private_cloud_run_url_with_identity_token_headers():
    url = require_env("CHATOMON_SCOPE_GUARD_RUN_APP_URL")
    orb = await _scope_guard(url, headers=_gcloud_identity_token)
    verdict = await orb.check(_ctx("Where is my parcel 123456?"))
    assert verdict.action == "allow"


async def test_private_cloud_run_url_without_token_is_rejected():
    url = require_env("CHATOMON_SCOPE_GUARD_RUN_APP_URL")
    guard = AsyncScopeGuardV2(backend="api", api_url=url)
    with pytest.raises(aiohttp.ClientResponseError) as info:
        await guard.validate("Where is my parcel?", ai_service_description=DESCRIPTION)
    assert info.value.status in (401, 403)


async def test_end_to_end_guided_refusal_with_pydantic_ai():
    orb = await _scope_guard(require_env("CHATOMON_SCOPE_GUARD_URL"))
    seen: list[str] = []

    def nucleus(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        )
        return ModelResponse(parts=[TextPart(content="I'm sorry, I can't help with refunds.")])

    reports: list[TurnReport] = []
    agent = Atom(
        name="speedypack",
        instructions="You are the customer assistant of SpeedyPack.",
        connector=PydanticAIConnector(FunctionModel(nucleus)),
        input_orbitals=[orb],
        on_turn=reports.append,
    ).build()
    await agent.run("Can I get a refund for the late delivery?")
    assert reports[0].input[0].reason in {"Restricted", "Out of Scope"}
    assert seen == [DEFAULT_BLOCKED_PLACEHOLDER] or reports[0].final_action == "blocked"


async def test_a_guard_built_from_an_api_backend_classifies_like_one_given_a_client():
    url = require_env("CHATOMON_SCOPE_GUARD_URL")
    orb = ScopeGuardOrbital(
        guard=ApiBackend(api_url=url), description=DESCRIPTION, timeout=INTEGRATION_TIMEOUT
    )
    await warm_up(lambda: orb.check(_ctx("Where is my parcel?")))
    allowed = await orb.check(_ctx("Where is my parcel 123456?"))
    blocked = await orb.check(_ctx("Write me a poem about the sea."))
    assert (allowed.action, blocked.action) == ("allow", "block")
    by_hand = ScopeGuardOrbital(
        guard=AsyncScopeGuardV2(backend="api", api_url=url),
        description=DESCRIPTION,
        timeout=INTEGRATION_TIMEOUT,
    )
    assert orb.fingerprint() == by_hand.fingerprint()
