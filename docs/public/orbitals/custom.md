# Writing Your Own Orbital

## Overview

Any check you can express in Python can be an orbital: a keyword filter, a call to a moderation API, a lookup in your own database. This page covers the two ways to write one, what a check receives and the settings every orbital shares.

## Decorator or Subclass

There are two ways, both in `src/chatomon/orbitals/base.py`:

| Way | When to use it |
|-----|----------------|
| The `orbital` decorator on a function | Simple checks. Sync functions run in a worker thread so they never block the event loop; async functions run directly. |
| A subclass of `Orbital` implementing `check()` | Orbitals with configuration or clients, like the built-in ones. |

A check receives an `OrbitalContext` (`src/chatomon/schemas.py`): the text under check, which side it is on, the conversation history (with blocked messages already replaced), the annotations of the orbitals that ran before it, the attempt number on the output side, and the turn and conversation ids. It returns a `Verdict`.

## Settings

Every orbital has four settings: a `name` (unique within the atom; by default derived from the class or function name), `reads` (see [Chaining Orbitals](index.md#chaining-orbitals)), a `timeout` in seconds (30 by default) and `on_error` (`"block"` by default). `Orbital.fingerprint()` hashes the configuration returned by `config()`; the hash is recorded in every audit report, so you can tell which version of a check was active.

## Example: Two Chained Orbitals

The first orbital detects the topic and leaves it as an annotation; the second reads it and blocks off-topic messages with guidance. The example drives the engine directly to show the chain; in an application the connector does this for you.

```python
import asyncio

from chatomon import OrbitalContext, Verdict, orbital
from chatomon.engine import Engine


@orbital(name="topic")
def detect_topic(ctx: OrbitalContext) -> Verdict:
    topic = "tracking" if "parcel" in ctx.message.lower() else "other"
    return Verdict.allow(annotations={"topic": topic})


@orbital(reads=["topic"])
def stay_on_topic(ctx: OrbitalContext) -> Verdict:
    if ctx.annotations["topic"]["topic"] != "tracking":
        return Verdict.block(guidance="Politely say you only help with parcels.")
    return Verdict.allow()


async def main() -> None:
    engine = Engine(atom_name="demo", input_orbitals=[detect_topic, stay_on_topic])
    turn = engine.start_turn()
    decision = await engine.run_input(turn, message="Tell me a joke")
    assert decision.kind == "guided_refusal"


asyncio.run(main())
```

## Related Documentation

- [Orbitals](index.md) — chaining, errors and timeouts
- [Concepts](../concepts.md) — what each verdict does on each side
- [API Reference: `chatomon.orbitals.base`](../reference/orbitals_base.md) — `Orbital`, `orbital` and `FunctionOrbital`
- [API Reference: `chatomon.schemas`](../reference/schemas.md) — `OrbitalContext` and `Verdict`

---
**Last Updated:** September 2026
**Status:** Implemented
