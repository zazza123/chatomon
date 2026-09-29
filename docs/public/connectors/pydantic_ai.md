# PydanticAI

## Overview

`PydanticAIConnector` (`src/chatomon/connectors/pydantic_ai.py`) builds an atom into a `pydantic_ai.Agent` with text output and installs chatomon's checks as a capability around the model requests. It needs the `pydantic-ai` extra (`pip install 'chatomon[pydantic-ai]'`).

## Building the Agent

`Atom.build()` returns a `pydantic_ai.Agent`; every extra keyword argument of the connector goes to it (`tools`, `deps_type`, `capabilities`, ...). chatomon sets the prompt and name from the atom, so `instructions`, `system_prompt`, `output_type` and `name` are refused with `ChatomonConfigError`. Your own capabilities run inside chatomon's layer.

## Hooks

| Hook | What chatomon does there |
|------|--------------------------|
| `before_model_request` | Runs the input chain and redacts a blocked message |
| `wrap_model_request` | Adds the guidance, sends a canned reply, runs the output chain and the regenerations |
| `wrap_run` | Finishes the turn, or aborts it if the run fails |

Both `run` and `run_sync` are supported.

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

- [Connectors](index.md) — what the two connectors share
- [LangChain / LangGraph](langchain.md) — the other connector
- [API Reference: `chatomon.connectors.pydantic_ai`](../reference/connectors_pydantic_ai.md) — every parameter and a runnable example

---
**Last Updated:** September 2026
**Status:** Implemented
