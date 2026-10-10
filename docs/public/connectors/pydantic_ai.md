# PydanticAI

## Overview

`PydanticAIConnector` (`src/chatomon/connectors/pydantic_ai.py`) builds an atom into a `pydantic_ai.Agent` with text output and installs chatomon's checks as a capability around the model requests. It needs the `pydantic-ai` extra (`pip install 'chatomon[pydantic-ai]'`).

## Building the Agent

`Atom.build()` returns a `pydantic_ai.Agent`; every extra keyword argument of the connector goes to it (`tools`, `deps_type`, `capabilities`, ...). chatomon sets the prompt and name from the atom, so `instructions`, `system_prompt`, `output_type` and `name` are refused with `ChatomonConfigError`. Your own capabilities run inside chatomon's layer.

## Hooks

| Hook | What chatomon does there |
|------|--------------------------|
| `before_model_request`, `wrap_model_request` | The first of the two that PydanticAI runs opens the turn: it runs the input chain and redacts a blocked message |
| `wrap_model_request` | Sends a canned reply without calling the model, puts chatomon's wrapper around the model of the request, and refuses a reply that did not pass the checks |
| The model wrapper | Adds the guidance, runs the output chain and asks the model again when an orbital rejects a reply |
| `after_model_request` | Refuses a reply that a capability changed after the checks |
| `wrap_run` | Finishes the turn, or aborts it if the run fails |

Both `run` and `run_sync` are supported.

PydanticAI changed the order of these hooks in version 2.54: `wrap_model_request` now runs around `before_model_request`, and may call the model once. The connector does not depend on that order and behaves the same on every version the extra allows.

## Capabilities of Your Own

Your `capabilities` run inside chatomon's layer, so they see the user message after the input orbitals checked it. Two things follow from where the reply is checked:

- **A reply written again is one model request.** The model is asked again inside the wrapper, so a capability of yours sees one model request for each reply that leaves it, and the run's usage counts only that reply.
- **The reply must come from the atom's model.** chatomon checks what the model writes. A capability that answers in place of the model, skips it with `SkipModelRequest` or rewrites its reply makes the run fail with `ChatomonConfigError`, and the turn is closed as `aborted`: no text leaves the run unchecked. A capability that replaces the model of a request is refused in the same way from PydanticAI 2.54; on earlier versions the model it puts in place is checked like the atom's.

## Conversations

The history lives in the `message_history` you pass to the next run. The conversation id of every report is the run's `conversation_id`, stable across runs that share the history, and a blocked message is replaced by the placeholder in `all_messages()`.

```python
from pydantic_ai.models.test import TestModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector

agent = Atom(
    name="speedypack",
    instructions="Be brief.",
    connector=PydanticAIConnector(TestModel(custom_output_text="Hi!")),
).build()
first = agent.run_sync("hello")
second = agent.run_sync("and now?", message_history=first.all_messages())
assert len(second.all_messages()) == 4
```

## Streaming

With input orbitals only, tokens stream normally.

!!! warning "`run_stream` is refused when the atom has output orbitals"
    PydanticAI streams text before it can be checked, so `run_stream` raises `ChatomonConfigError` before the model is called. Use `run` or `run_sync`.

## Tracing

PydanticAI propagates the OpenTelemetry context through its hooks, so chatomon's spans nest under the agent run automatically.

## Related Documentation

- [Connectors](index.md) — shared connector behavior
- [LangChain / LangGraph](langchain.md) — the LangChain connector
- [Microsoft Agent Framework](maf.md) — the MAF connectors
- [API Reference: `chatomon.connectors.pydantic_ai`](../reference/connectors_pydantic_ai.md) — every parameter and a runnable example

---
**Last Updated:** October 2026
**Status:** Implemented
