# Principled Intelligence Orbitals

## Overview

`src/chatomon/orbitals/principled.py` wraps the `orbitals` library of Principled Intelligence as chatomon orbitals. It needs the `principled` extra (`pip install 'chatomon[principled]'`). You configure the library's client yourself — local, self-hosted or API backend — and give each orbital its own service description.

## ScopeGuard

`ScopeGuardOrbital` works on the input side with ScopeGuard V1 or V2 and maps the returned class to a verdict:

| ScopeGuard class | Default verdict |
|------------------|-----------------|
| Directly Supported, Potentially Supported | allow |
| Chit Chat | allow, with guidance to reply briefly and steer back to the service |
| Predefined Answer (V2) | respond with ScopeGuard's suggested response |
| Human Oversight (V2) | respond with the suggested response and `escalate` |
| Out of Scope | block with guidance to explain what the service can help with |
| Restricted | block with the default refusal (the nucleus never sees the message) |

Any entry can be overridden with the `mapping` setting, using a fixed verdict or a function of the ScopeGuard result.

!!! note "Needs a running ScopeGuard server"
    The example below is not run by the test suite. It configures a ScopeGuard V2 orbital with its own service description and one override.

```{.python test="skip"}
from orbitals.scope_guard_v2 import AsyncScopeGuardV2, ScopeClass
from orbitals.types import AIServiceDescriptionV2

from chatomon import Verdict
from chatomon.orbitals.principled import ScopeGuardOrbital

scope_guard = ScopeGuardOrbital(
    guard=AsyncScopeGuardV2(backend="api", api_url="http://127.0.0.1:8080"),
    description=AIServiceDescriptionV2(
        identity_role="Customer assistant of SpeedyPack, a parcel courier",
        context="Customers ask about their shipments",
        constraints=["Never discuss refunds"],
    ),
    mapping={ScopeClass.CHIT_CHAT: Verdict.allow()},
    timeout=60,
)
```

## ClaimExtractor

`ClaimExtractorOrbital` works on both sides and never blocks: it extracts claims and intents from the message and leaves them as annotations for a later orbital, typically a [judge](judge.md) that checks the claims against reference material.

## Credentials

Both orbitals accept a `headers` provider, called before every request, to refresh credentials such as a Cloud Run identity token — see [ScopeGuard on Cloud Run](../deployment/cloud_run.md).

## Related Documentation

- [Orbitals](index.md) — chaining, errors and timeouts
- [Deployment](../deployment/index.md) — running guard models behind private endpoints
- [API Reference: orbitals.principled](../reference/orbitals_principled.md) — every parameter

---
**Last Updated:** September 2026
**Status:** Implemented
