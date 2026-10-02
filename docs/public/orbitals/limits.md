# Limit Orbitals

## Overview

`MessageLengthOrbital` and `TurnLimitOrbital` (`src/chatomon/orbitals/limits.py`) cap a conversation without writing your own orbital: the first limits the length of the user message or of the nucleus reply, the second the number of turns. Neither calls a model, so they cost nothing and belong early in a chain, before a judge.

## Message Length

`MessageLengthOrbital` counts words (the text split on whitespace) and characters; a message is too long when it exceeds any limit you set. Words are the natural unit to write a rule in; characters also hold for languages written without spaces and for text pasted without any.

On the input side a long message never reaches the nucleus: the user receives the orbital's `reply`, and the message is replaced by the placeholder in the history, so the text does not stay in the model's context.

```python
from pydantic_ai.models.test import TestModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital

agent = Atom(
    name="speedypack",
    instructions="You are the customer assistant of SpeedyPack.",
    connector=PydanticAIConnector(TestModel(custom_output_text="Hi!")),
    input_orbitals=[
        MessageLengthOrbital(max_words=200, reply="Please keep your message under 200 words.")
    ],
).build()

assert agent.run_sync("word " * 201).output == "Please keep your message under 200 words."
```

### Long Replies: Regeneration, Then Refusal

On the output side the orbital asks the nucleus to rewrite a reply that is too long, with guidance that states the limits ("Your previous reply was too long. Keep it to at most 150 words."); pass `guidance` to word it yourself. The engine repeats this up to the atom's `max_retries`, then sends the atom's `refusal`.

- **Cost** — each rewrite is one more model call: with the default `max_retries=2`, a turn can take three.
- **Outcome** — when the retries run out the user receives the refusal, and the turn report records `fallback_refusal`, with one `reply_too_long` result per attempt.
- **Prevention** — state the limit in the atom's instructions too, so the first reply usually fits and the rewrite stays an exception.

### Pairing It with `max_tokens`

The model's own `max_tokens` is a hard cap against runaway generations, set in the connector. Set it comfortably above the orbital's limit — for example 400 tokens for a 150-word limit: a reply cut by `max_tokens` then also exceeds the orbital's limit and is rewritten, instead of reaching the user cut mid-sentence.

| Connector | Where to set `max_tokens` |
|-----------|---------------------------|
| PydanticAI | `PydanticAIConnector(model, model_settings={"max_tokens": 400})` |
| LangChain | On the chat model you pass to `LangChainConnector` (for example `max_tokens` on `ChatOpenAI`) |
| Microsoft Agent Framework | `MAFConnector(client, default_options={"max_tokens": 400})` |

## Turn Limit

`TurnLimitOrbital` counts the user messages in the history; the current message is turn *count + 1*, and blocked messages count too. Past `max_turns` every message is answered with the orbital's `reply` without calling the nucleus. With `warn_before`, the nucleus receives the `warning` as guidance in the last turns before the limit, so it can help the user wrap up.

```python
from pydantic_ai.models.test import TestModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import TurnLimitOrbital

agent = Atom(
    name="speedypack",
    instructions="You are the customer assistant of SpeedyPack.",
    connector=PydanticAIConnector(TestModel(custom_output_text="Hi!")),
    input_orbitals=[TurnLimitOrbital(max_turns=2, reply="Please start a new chat.")],
).build()

first = agent.run_sync("hello")
second = agent.run_sync("and then?", message_history=first.all_messages())
third = agent.run_sync("one more", message_history=second.all_messages())
assert (second.output, third.output) == ("Hi!", "Please start a new chat.")
```

- **Closing the session** — chatomon never runs the agent, so it cannot end the conversation: it keeps answering with the closing reply. Your application reads the `turn_limit` reason in the turn report and closes the chat.
- **Full history required** — the count comes from the history your application passes. If it trims the history (for example a sliding window), the count starts over.
- **Input side only** — placing it among the output orbitals is a configuration error.

## Decisions and Audit

| Orbital | Side | Outcome | Reason | Turn report |
|---------|------|---------|--------|-------------|
| `MessageLengthOrbital` | input | blocked with `reply` | `message_too_long` | `blocked` |
| `MessageLengthOrbital` | output | rewritten, then refused | `reply_too_long` | `answered` after a successful rewrite, otherwise `fallback_refusal` |
| `TurnLimitOrbital` | input | answered with `reply` | `turn_limit` | `orbital_response` |
| `TurnLimitOrbital` | input | warning guidance | `turn_limit_warning` | the nucleus answers |

The reasons are recorded in spans, logs and turn reports with the default `capture_content="none"`. The measurements — `{"words", "chars"}` and `{"turn", "max_turns"}` — are annotations: later orbitals can read them through `reads`, and they reach reports only when content capture allows it.

## Related Documentation

- [Orbitals](index.md) — chaining, errors and timeouts
- [Observability and Audit](../observability.md) — turn reports and content capture
- [API Reference: `chatomon.orbitals.limits`](../reference/orbitals_limits.md) — every parameter and a runnable example

---
**Last Updated:** October 2026
**Status:** Implemented
