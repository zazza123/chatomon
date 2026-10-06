# Microsoft Agent Framework

## Overview

The MAF connectors build native asynchronous agents from an [Atom][chatomon.Atom]. Install `chatomon[maf]` for a client you supply, or `chatomon[maf-foundry]` to use Foundry clients and agents. The core extra does not choose a model provider; install that provider's MAF package separately when needed. See the [official MAF documentation](https://learn.microsoft.com/en-us/agent-framework/) and [MAF source repository](https://github.com/microsoft/agent-framework) for the framework's clients and agent API.

| Connector | Native result | Output orbitals |
|-----------|---------------|-----------------|
| [`MAFConnector`][chatomon.connectors.maf.MAFConnector] | `agent_framework.Agent`, including one backed by `FoundryChatClient` | Supported for non-streaming runs |
| [`MAFFoundryConnector`][chatomon.connectors.maf.MAFFoundryConnector] | `agent_framework.foundry.FoundryAgent` connected to an existing PromptAgent or HostedAgent | Supported for non-streaming runs |

## Building an Agent

`MAFConnector` accepts a MAF chat client and forwards its other arguments to `Agent`. `MAFFoundryConnector` forwards Foundry settings such as `project_endpoint`, `agent_name`, `agent_version`, and credentials to `FoundryAgent`. Use the full project endpoint ending in `/api/projects/<project-name>`; a PromptAgent also needs its version. The atom supplies the local agent's name, instructions, and knowledge; conflicting arguments fail at build time. Both paths require a chat client with `ChatMiddlewareLayer`. `MAFFoundryConnector` selects its client type to preserve the atom's instructions and refuses a caller-supplied `client_type`.

For output checks, place application middleware in the connector's `middleware` argument. Preconfigured chat middleware on a client could rewrite a checked reply after chatomon's hook, so `MAFConnector` refuses that combination.

### Agent with a chat client

This example uses `OpenAIChatClient` to show the provider-independent `MAFConnector` path. Install `chatomon[maf]` and `agent-framework-openai`, then set `OPENAI_API_KEY` and `MAF_MODEL`. The input orbital blocks requests containing a secret; the output orbital replaces a reply containing the example marker. Both turns use one local session. The `AsyncOpenAI` context closes the provider client.

```{.python test="skip"}
import asyncio
import os

from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI

from chatomon import Atom, OrbitalContext, Verdict, orbital
from chatomon.connectors.maf import MAFConnector


@orbital
async def no_secrets(ctx: OrbitalContext) -> Verdict:
    if "secret" in ctx.message.lower():
        return Verdict.block(reply="Please remove secrets from your request.")
    return Verdict.allow()


@orbital
async def no_internal_details(ctx: OrbitalContext) -> Verdict:
    if "internal-only" in ctx.message.lower():
        return Verdict.block(reply="I cannot share internal details.")
    return Verdict.allow()


async def main() -> None:
    async with AsyncOpenAI() as provider:
        client = OpenAIChatClient(model=os.environ["MAF_MODEL"], async_client=provider)
        agent = Atom(
            name="support",
            instructions="Answer customer questions briefly.",
            connector=MAFConnector(client),
            input_orbitals=[no_secrets],
            output_orbitals=[no_internal_details],
        ).build()
        session = agent.create_session()
        first = await agent.run("Hello", session=session)
        second = await agent.run("What did I just say?", session=session)
        print(first.text, second.text)


asyncio.run(main())
```

### Existing Foundry agent

For an existing [Foundry agent](https://learn.microsoft.com/en-us/azure/foundry/agents/overview), use `MAFFoundryConnector`. Install `chatomon[maf-foundry]`, then set `FOUNDRY_PROJECT_ENDPOINT` to the full project endpoint and `FOUNDRY_AGENT_NAME` and `FOUNDRY_AGENT_VERSION` for the agent. The two-turn example below targets a PromptAgent; HostedAgent runtime sessions have the limitation described under [Checks and Local History](#checks-and-local-history).

Foundry controls the remote agent configuration. The connector sends the atom's instructions and knowledge as a developer message on each model call, since Foundry rejects the request-level `instructions` field when an agent is specified. Guidance joins that message for the current call only.

```{.python test="skip"}
import asyncio
import os

from azure.identity.aio import DefaultAzureCredential

from chatomon import Atom, OrbitalContext, Verdict, orbital
from chatomon.connectors.maf import MAFFoundryConnector


@orbital
async def no_internal_details(ctx: OrbitalContext) -> Verdict:
    if "internal-only" in ctx.message.lower():
        return Verdict.block(reply="I cannot share internal details.")
    return Verdict.allow()


async def main() -> None:
    async with DefaultAzureCredential() as credential:
        agent = Atom(
            name="support",
            instructions="Answer customer questions briefly.",
            connector=MAFFoundryConnector(
                project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                agent_name=os.environ["FOUNDRY_AGENT_NAME"],
                agent_version=os.environ["FOUNDRY_AGENT_VERSION"],
                credential=credential,
            ),
            output_orbitals=[no_internal_details],
        ).build()
        try:
            session = agent.create_session()
            first = await agent.run("Hello", session=session)
            second = await agent.run("Please clarify.", session=session)
            print(first.text, second.text)
        finally:
            await agent.client.close()


asyncio.run(main())
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

Pass the same `AgentSession` from `agent.create_session()` to subsequent turns. Its `session_id` appears as the conversation id in chatomon reports. The connector sets `store=False` and rejects `store=True`, remote conversation IDs, explicit Foundry agent session IDs, and sessions with `service_session_id`. A provider that returns a remote conversation ID despite `store=False` causes the run to abort. A Foundry HostedAgent may still create an internal infrastructure session ID in its local state; chatomon refuses to send that ID on a later turn. Foundry's own retention and logging policies govern service-side records.

The `resolve_client(model_id)` callback is required when a `JudgeOrbital` uses this connector. It returns a MAF chat client for the judge's model ID; the judge call uses a separate agent with `store=False`. If the callback is missing, `Atom.build()` fails with a configuration error. Keep ownership of the returned clients and close them with their underlying provider clients.

The following configuration-only example builds an agent without sending a request. The judge's `model` is a lookup key for `resolve_client`; each MAF client holds its provider model setting.

```python
import asyncio

from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI

from chatomon import Atom
from chatomon.connectors.maf import MAFConnector
from chatomon.orbitals import JudgeOrbital


async def main() -> None:
    async with AsyncOpenAI(api_key="example-only") as provider:
        primary = OpenAIChatClient(model="example-primary", async_client=provider)
        reviewer = OpenAIChatClient(model="example-reviewer", async_client=provider)
        clients = {"review": reviewer}
        agent = Atom(
            name="reviewed-support",
            instructions="Answer briefly.",
            connector=MAFConnector(primary, resolve_client=clients.__getitem__),
            output_orbitals=[
                JudgeOrbital(model="review", instructions="Reject unsupported claims.")
            ],
        ).build()
        assert agent.name == "reviewed-support"


asyncio.run(main())
```

## Streaming and Failures

MAF's native API is asynchronous. With input orbitals only, `run(..., stream=True)` returns a MAF `ResponseStream`; completed and cancelled streams produce a turn report. If the atom has output orbitals, streaming raises `ChatomonConfigError` before the model call because tokens would otherwise leave before the final reply can be checked. Use the non-streaming `run()` path for checked output.

Model and middleware failures abort the turn and preserve the exception type in the report. Message content, guidance, and judge explanations are absent from chatomon logs and reports by default; the atom's `capture_content` policy controls optional capture.

With `store=False`, output checks happen before replies enter local history. Replacement and regeneration retain only approved replies. HostedAgent runtime sessions may require a remote `agent_session_id`, which chatomon refuses to reuse.

## Related Documentation

- [Connectors](index.md) — shared connector behavior
- [JudgeOrbital](../orbitals/judge.md) — model-based checks and model resolution
- [Observability and audit](../observability.md) — reports and content capture
- [Frameworks in the Studio](../studio/frameworks.md) — trying an atom, a Foundry agent included, in the local studio
- [MAF API reference](../reference/connectors_maf.md) — parameters, return values, and exceptions

---
**Last Updated:** October 2026
**Status:** Implemented
