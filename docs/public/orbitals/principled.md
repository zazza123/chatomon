# Principled Intelligence Orbitals

## Overview

`src/chatomon/orbitals/principled.py` wraps the `orbitals` library of Principled Intelligence as chatomon orbitals. It needs the `principled` extra (`pip install 'chatomon[principled]'`). Each orbital gets a client of the library and its own service description, which is never derived from the atom.

## Connecting to a Service

An orbital takes its client in one of two ways.

| You give | The orbital |
|----------|-------------|
| A client of the `orbitals` library that you configured | Calls it as it is: any backend (local, self-hosted or API), sync or async, V1 or V2, with its own headers |
| An `ApiBackend` | Builds the asynchronous client of the `api` backend itself; for ScopeGuard, the V2 one |

`ApiBackend` is a shortcut for the common case, a service reached over HTTP. It holds three plain values: `api_url`, where the service answers; `api_key_env`, the name of the environment variable that holds the API key; and `model`, for a service that hosts more than one. For anything else, pass a client: the choice is open, and an `ApiBackend` is only one more thing `guard` and `extractor` accept.

!!! note "Needs the variable to be set"
    The example below is not run by the test suite. It builds a ScopeGuard orbital from the address of its service and the name of the variable that holds the key.

```{.python test="skip"}
from chatomon.orbitals.principled import ApiBackend, ScopeGuardOrbital

scope_guard = ScopeGuardOrbital(
    guard=ApiBackend(
        api_url="https://scope-guard.example.run.app",
        api_key_env="PRINCIPLED_API_KEY",
    ),
    description="Customer assistant of SpeedyPack, a parcel courier. Never discuss refunds.",
)
```

Because an `ApiBackend` is made of plain values, the [studio](../studio/tuning.md#blocks) can add an orbital built from one, edit it and write it back as code.

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

An `ApiBackend` never holds a key, only the name of the variable that does. The orbital reads the variable when it is created, so a variable that is not set, or is empty, is a `ChatomonConfigError` that names it, raised where the atom is defined and not at the first message. The name ends up in your code; the key does not, and neither is part of the orbital's fingerprint. An address that holds a user and a password is refused for the same reason: the key belongs in a variable. An address with a query or a fragment is refused too: the client writes its own path after the address.

The name is not a secret, and it is shown as it is: in the message for a variable that is not set, in the studio, in a preset and in the code. A key written where the name goes is refused without being repeated when it cannot be a name, as one with a hyphen; a key made only of letters, digits and underscores reads like a name, and the message that says it is not set repeats it. Write the name there, never the key.

Without `api_key_env` no key is passed, and the client of the `orbitals` library does what it always does: it uses `PRINCIPLED_API_KEY` when that variable is set.

Both orbitals also accept a `headers` provider, called before every request, to refresh credentials such as a Cloud Run identity token — see [ScopeGuard on Cloud Run](../deployment/cloud_run.md). It works next to an `ApiBackend` as next to a client of the `api` backend.

## Related Documentation

- [Orbitals](index.md) — chaining, errors and timeouts
- [Deployment](../deployment/index.md) — running guard models behind private endpoints
- [Tuning Orbitals](../studio/tuning.md) — adding these orbitals from the studio and testing their connection
- [API Reference: `chatomon.orbitals.principled`](../reference/orbitals_principled.md) — every parameter

---
**Last Updated:** October 2026
**Status:** Implemented
