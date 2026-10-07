# Observability and Audit

## Overview

![](assets/brand/mascot/think-light.svg#only-light){ .cm-page-mascot }
![](assets/brand/mascot/think-dark.svg#only-dark){ .cm-page-mascot }
Every turn leaves three kinds of records: log records, OpenTelemetry spans and a structured `TurnReport`. All three are content-free by default: they describe what happened (which orbital decided what, how long it took, what failed) without the text of the conversation, unless you explicitly allow it.

## Logging

chatomon logs through the standard `logging` module, with loggers under the `chatomon` namespace (for example `chatomon.engine`). The library only attaches a `NullHandler`; enabling output is the application's choice, by configuring logging and setting the level of the `chatomon` logger. Records carry structured fields under the `chatomon` attribute (atom name, turn id, orbital, side, action, reason, error type).

```python
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("chatomon").setLevel(logging.INFO)
```

| Level | What is logged |
|-------|----------------|
| DEBUG | Every orbital verdict, accepted decisions, answered turns |
| INFO | Decisions other than proceed/accept, turns that did not end with a plain answer |
| WARNING | Orbital failures and timeouts, guidance ignored on the output side, aborted turns |
| ERROR | A failing `on_turn` callback (type only) |

## OpenTelemetry

chatomon depends only on `opentelemetry-api`: without an SDK configured by the application every span is a no-op. `Atom(tracer_provider=...)` selects a provider; by default the global one is used. With the SDK installed (`opentelemetry-sdk`):

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
```

| Span | Attributes |
|------|------------|
| `chatomon.input` | atom name, turn id, conversation id, decision kind |
| `chatomon.output` (one per attempt) | the same, plus the attempt number |
| `chatomon.orbital {name}` | orbital name, side, configuration fingerprint, verdict action and reason, whether it carried guidance, escalation, `error.type` on failure |

A `JudgeOrbital` also adds the `gen_ai.evaluation.result` event of the OpenTelemetry GenAI conventions. The conventions for guardrails are still a proposal; chatomon uses its own `chatomon.*` names until they are stable. How spans nest in each framework is described in [Connectors](connectors/index.md#at-a-glance).

## Turn Reports

`Atom(on_turn=...)` receives a `TurnReport` (`src/chatomon/schemas.py`) at the end of every turn; the callback may be sync or async, and an exception it raises is logged and never breaks the turn. Storing and retaining the reports is up to the application.

```python
from pydantic_ai.models.test import TestModel

from chatomon import Atom, TurnReport
from chatomon.connectors.pydantic_ai import PydanticAIConnector

audit_log: list[dict] = []


def save_audit_record(report: TurnReport) -> None:
    audit_log.append(report.model_dump(mode="json"))


agent = Atom(
    name="speedypack",
    instructions="Be brief.",
    connector=PydanticAIConnector(TestModel(custom_output_text="Hi!")),
    on_turn=save_audit_record,
    capture_content="hash",
).build()
agent.run_sync("hello")
assert audit_log[0]["final_action"] == "answered"
assert audit_log[0]["user_message"].startswith("sha256:")
attempt = audit_log[0]["output"][0]
assert attempt["nucleus_reply"].startswith("sha256:")
assert attempt["decision"] == "accept"
```

| Field | Meaning |
|-------|---------|
| `final_action` | `answered`, `orbital_response`, `blocked`, `guided_refusal`, `fallback_refusal` or `aborted` |
| `escalated` | Whether any orbital asked for a human |
| `input` | One `OrbitalResult` per input orbital, with action, reason, fingerprint, timing, error type and error policy |
| `output` | One `OutputAttempt` per reply of the nucleus that the output chain checked: the reply (`nucleus_reply`), one `OrbitalResult` per output orbital (`results`) and what chatomon decided about the reply (`decision`) |
| `error_type` | The exception type of an aborted turn |
| `user_message`, `final_reply`, the `nucleus_reply` of each attempt and the content fields of each result | Only according to `capture_content` |

A turn has one attempt for every reply the nucleus wrote and the output chain checked, in order. A turn the input chain answered has none, and a step in which the nucleus calls a tool is not an attempt. Each attempt keeps what the nucleus wrote, also when that text never reached the user: `final_reply` is what the user received, and a reply that an orbital gave instead is in that orbital's result too.

| `decision` | What chatomon did with the reply |
|------------|----------------------------------|
| `accept` | Took it as the reply of the turn |
| `replace` | Sent another text instead: the reply of an orbital, or the atom's refusal |
| `regenerate` | Asked the nucleus to write it again with the orbital's guidance; another attempt follows |
| none | Nothing: the turn was aborted while the output chain was running |

An accepted attempt is the last one unless the framework asks the model again afterwards, for example when its own validation of the output fails.

The ready-made orbitals use fixed `reason` labels that contain no user text, so they are safe to record without content capture and an application can act on them — for example [the limit orbitals](orbitals/limits.md#decisions-and-audit) (`message_too_long`, `turn_limit`, ...), such as closing a chat when `turn_limit` appears. Your own orbitals choose their labels: keep user text out of them.

## Content Capture

One setting, `Atom(capture_content=...)`, governs every piece of conversation content chatomon records: the user message, the reply of the nucleus at every attempt, the final reply, guidance, annotations, judge explanations and exception messages. Content is recorded in turn reports only. Log records and spans carry labels — names, actions, reasons, timings and error types — whatever the setting.

| Value | Effect |
|-------|--------|
| `"none"` (default) | No content is recorded |
| `"hash"` | Only SHA-256 digests, useful to correlate without storing text |
| `"full"` | Full content, passed through `redactor` when one is configured; if the redactor fails the marker `[redaction failed]` is recorded, and unserializable data becomes `[unserializable]` |

The policy is implemented by `ContentPolicy` in `src/chatomon/observability.py`.

## Compliance Notes

These notes are orientation, not legal advice.

- **EU AI Act, article 50** — telling users they are talking to an AI is the application's job, in its user interface.
- **EU AI Act, articles 12, 19 and 26** — automatic logging kept for at least six months applies to high-risk systems (for example chatbots used in hiring, credit or education). Turn reports make it possible; retention is the application's responsibility.
- **GDPR** — user messages are personal data. Content-free defaults implement privacy by default (article 25); enable content capture only with a legal basis and a retention policy.

## Related Documentation

- [Concepts](concepts.md) — the final actions of a turn
- [Connectors](connectors/index.md) — span nesting per framework
- [API Reference](reference/index.md) — `TurnReport`, `OutputAttempt`, `OrbitalResult` and `ContentPolicy`

---
**Last Updated:** October 2026
**Status:** Implemented
