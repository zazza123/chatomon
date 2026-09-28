"""ClaimExtractor against a real deployment (e.g. a vast.ai instance through an SSH tunnel).

    CHATOMON_INTEGRATION=1 CHATOMON_CLAIM_EXTRACTOR_URL=http://127.0.0.1:8100 \
        uv run pytest tests/integration/test_claim_extractor.py -m integration -v
"""

from __future__ import annotations

import os

import pytest
from orbitals.claim_extractor import AsyncClaimExtractor, Extractions

from chatomon import OrbitalContext, Verdict, orbital
from chatomon.orbitals.principled import ClaimExtractorOrbital
from chatomon.schemas import Message
from tests.fakes import make_engine
from tests.integration.conftest import INTEGRATION_TIMEOUT, require_env, warm_up

pytestmark = pytest.mark.integration


@orbital(reads=["claims"])
def no_round_the_clock_promises(ctx: OrbitalContext) -> Verdict:
    extractions: Extractions = ctx.annotations["claims"]
    if any("24" in claim.content for claim in extractions.claims):
        return Verdict.block(guidance="Opening hours are Monday to Friday, 9-18.")
    return Verdict.allow()


async def test_claims_from_a_real_reply_drive_a_regeneration():
    extractor = AsyncClaimExtractor(
        backend="api",
        api_url=require_env("CHATOMON_CLAIM_EXTRACTOR_URL"),
        model=os.environ.get("CHATOMON_CLAIM_EXTRACTOR_MODEL", "claim-extractor"),
    )
    claims = ClaimExtractorOrbital(name="claims", extractor=extractor, timeout=INTEGRATION_TIMEOUT)
    warm_ctx = OrbitalContext(side="output", message="We are open.", turn_id="warm-up")
    await warm_up(lambda: claims.check(warm_ctx))

    engine = make_engine(outputs=[claims, no_round_the_clock_promises])
    turn = engine.start_turn()
    await engine.run_input(turn, message="When are you open?")
    decision = await engine.run_output(
        turn,
        reply="We are open 24 hours a day, 7 days a week.",
        history=(Message(role="user", content="When are you open?"),),
    )
    assert decision.kind == "regenerate"
    assert isinstance(turn.annotations["claims"], Extractions)
    assert turn.annotations["claims"].claims
