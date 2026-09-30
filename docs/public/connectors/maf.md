# Microsoft Agent Framework

## Overview

The MAF connectors build native asynchronous agents from an [Atom][chatomon.Atom]. Install `chatomon[maf]` for a client you supply, or `chatomon[maf-foundry]` to use Foundry clients and agents. The core extra does not choose a model provider; install that provider's MAF package separately when needed.

| Connector | Native result | Output orbitals |
|-----------|---------------|-----------------|
| [`MAFConnector`][chatomon.connectors.maf.MAFConnector] | `agent_framework.Agent`, including one backed by `FoundryChatClient` | Supported for non-streaming runs |
| [`MAFFoundryConnector`][chatomon.connectors.maf.MAFFoundryConnector] | `agent_framework.foundry.FoundryAgent` connected to an existing PromptAgent or HostedAgent | Awaiting a real Foundry continuity test |

## Building an Agent

`MAFConnector` accepts a MAF chat client and forwards its other arguments to `Agent`. `MAFFoundryConnector` forwards Foundry settings such as `project_endpoint`, `agent_name`, `agent_version`, and credentials to `FoundryAgent`. The atom supplies the local agent's name, instructions, and knowledge; conflicting arguments fail at build time. Both paths require a chat client with `ChatMiddlewareLayer`, so a raw client that bypasses chat middleware is refused.

For output checks, place application middleware in the connector's `middleware` argument. Preconfigured chat middleware on a client could rewrite a checked reply after chatomon's hook, so `MAFConnector` refuses that combination.

```{.python test="skip"}
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

from chatomon import Atom
from chatomon.connectors.maf import MAFConnector

client = FoundryChatClient(
    project_endpoint="https://example.services.ai.azure.com",
    model="my-deployment",
    credential=DefaultAzureCredential(),
)
agent = Atom(
    name="support",
    instructions="Answer customer questions briefly.",
    connector=MAFConnector(client),
).build()
session = agent.create_session()
reply = await agent.run("Hello", session=session)
```

For an existing Foundry PromptAgent or HostedAgent, replace the connector with `MAFFoundryConnector`. The local atom instructions are passed to the Foundry agent wrapper; its remote configuration remains controlled in Foundry.

```{.python test="skip"}
from azure.identity import DefaultAzureCredential

from chatomon import Atom
from chatomon.connectors.maf import MAFFoundryConnector

agent = Atom(
    name="support",
    instructions="Answer customer questions briefly.",
    connector=MAFFoundryConnector(
        project_endpoint="https://example.services.ai.azure.com",
        agent_name="support-agent",
        credential=DefaultAzureCredential(),
    ),
).build()
reply = await agent.run("Hello", session=agent.create_session())
```

## Checks and Local History

The outer `AgentMiddleware` reads the local session history, runs input orbitals, and replaces a blocked user message with the atom's placeholder before other agent hooks or model calls. `ChatMiddleware` adds guidance to the current request, checks final text replies, and substitutes or regenerates rejected drafts before MAF writes the reply to local history. Intermediate tool calls are passed through until a final text reply exists. The connector adds `InMemoryHistoryProvider` when no loading history provider is supplied.

```mermaid
flowchart LR
    A[Agent.run with local AgentSession] --> B[AgentMiddleware: input checks and redaction]
    B --> C[MAF context providers and client]
    C --> D[ChatMiddleware: guidance and final output checks]
    D --> E[Approved reply in local history]
```

Pass the same `AgentSession` from `agent.create_session()` to subsequent turns. Its `session_id` appears as the conversation id in chatomon reports. The connector sets `store=False` and rejects `store=True`, remote conversation IDs, explicit Foundry agent session IDs, and sessions with `service_session_id`. A provider that returns a remote conversation ID despite `store=False` causes the run to abort. A Foundry HostedAgent may still create an internal infrastructure session ID in its local state; Foundry's own retention and logging policies govern that service behavior.

The `resolve_client(model_id)` callback is required when a `JudgeOrbital` uses this connector. It returns a MAF chat client for the judge's model ID; the judge call uses a separate agent with `store=False`. If the callback is missing, `Atom.build()` fails with a configuration error.

## Streaming and Failures

MAF's native API is asynchronous. With input orbitals only, `run(..., stream=True)` returns a MAF `ResponseStream`; completed and cancelled streams produce a turn report. If the atom has output orbitals, streaming raises `ChatomonConfigError` before the model call because tokens would otherwise leave before the final reply can be checked. Use the non-streaming `run()` path for checked output.

Model and middleware failures abort the turn and preserve the exception type in the report. Message content, guidance, and judge explanations are absent from chatomon logs and reports by default; the atom's `capture_content` policy controls optional capture.

`MAFFoundryConnector` currently supports input orbitals only. Before enabling its output orbitals, a real Foundry test must show that a rejected draft is absent from later turns of the same agent session. `MAFConnector` with `FoundryChatClient` already supports checked output under the local-history policy above.

## Related Documentation

- [Connectors](index.md) — shared connector behavior
- [JudgeOrbital](../orbitals/judge.md) — model-based checks and model resolution
- [Observability and audit](../observability.md) — reports and content capture
- [MAF API reference](../reference/connectors_maf.md) — parameters, return values, and exceptions

---
**Last Updated:** September 2026
**Status:** Implemented
