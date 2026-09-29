# LangChain / LangGraph

## Overview

`LangChainConnector` (`src/chatomon/connectors/langchain.py`) builds an atom into a LangGraph `CompiledStateGraph` with `langchain.agents.create_agent` and installs chatomon's checks as middleware. It needs the `langchain` extra (`pip install 'chatomon[langchain]'`).

## Building the Agent

`Atom.build()` returns the compiled graph; every extra keyword argument of the connector goes to `create_agent` (`tools`, `checkpointer`, `middleware`, ...). chatomon sets the prompt and name from the atom, so `system_prompt`, `response_format` and `name` are refused with `ChatomonConfigError`. Your own middleware runs inside chatomon's layer.

## Hooks

| Hook | What chatomon does there |
|------|--------------------------|
| `before_agent` | Runs the input chain, redacts a blocked message by id, jumps to the end when an orbital replies |
| `wrap_model_call` | Adds the guidance, runs the output chain and the regenerations; rebuilds a turn resumed by another process or agent instance |
| `wrap_tool_call` | Closes the turn as `aborted` when a tool raises |
| `after_agent` | Finishes the turn and clears chatomon's private state |

Every hook has a sync and an async variant, so both `invoke` and `ainvoke` are supported.

## Conversations

The checkpointer keeps the conversation and its `thread_id` becomes the conversation id of every report; a blocked message is replaced by the placeholder in the latest checkpoint.

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from chatomon import Atom, TurnReport
from chatomon.connectors.langchain import LangChainConnector

reports: list[TurnReport] = []
model = GenericFakeChatModel(messages=iter([AIMessage(content="Hi!"), AIMessage(content="Bye!")]))
agent = Atom(
    name="speedypack",
    instructions="Be brief.",
    connector=LangChainConnector(model, checkpointer=InMemorySaver()),
    on_turn=reports.append,
).build()
config = {"configurable": {"thread_id": "customer-42"}}
agent.invoke({"messages": [HumanMessage(content="hello")]}, config)
agent.invoke({"messages": [HumanMessage(content="thanks")]}, config)
assert [report.conversation_id for report in reports] == ["customer-42", "customer-42"]
```

!!! warning "Earlier checkpoints keep the original message"
    A LangGraph checkpointer keeps every earlier snapshot of the thread, and the snapshot written before chatomon runs still contains the original user message. The nucleus never sees it again — replaying a thread runs the input orbitals again — but it remains in storage: pruning and retention of checkpoints are the application's responsibility, as for any other personal data.

## Streaming

With input orbitals only, tokens stream normally. With output orbitals the model is tagged `nostream` and checked inside the model call, so the graph's output streams (`stream` / `astream` with `messages`, `updates` or `values`) emit only the approved reply.

!!! danger "`astream_events` and callbacks are not a safe user-facing channel"
    `astream_events` and callback handlers observe every internal model call, so when output orbitals exist they also see drafts that the output orbitals later reject. Use them for tracing, not to feed a user interface.

## Tracing

LangChain does not propagate the OpenTelemetry context through its hooks. `LangChainConnector` accepts a `parent_span` callable that returns the span chatomon's spans should nest under; without it they attach to the caller's current span.

## Failures and Human in the Loop

Failures of the model call and of tools close the turn as `aborted` (see [Failures](index.md#failures)). A turn interrupted by `interrupt()` stays open and is closed when the graph resumes; if it is resumed by another process or agent instance, chatomon rebuilds the turn and runs the input orbitals again before the model is called, so the reply is always checked. A turn left behind by a failure elsewhere is closed as `aborted` when the next message arrives on the same thread.

## Related Documentation

- [Connectors](index.md) — what the two connectors share
- [PydanticAI](pydantic_ai.md) — the other connector
- [API Reference: `chatomon.connectors.langchain`](../reference/connectors_langchain.md) — every parameter and a runnable example

---
**Last Updated:** September 2026
**Status:** Implemented
