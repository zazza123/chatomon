# Frameworks in the Studio

## Overview

The studio opens an atom built with any of chatomon's connectors, and the page is the same for all of them: turns, reports, tuning and export work alike. What differs is underneath: how a conversation is carried from one turn to the next, what the atom card names as the model, and what a turn can rely on. This page says what changes with each connector.

An atom on a connector you wrote yourself is refused when the studio starts, with a message that names the connector.

## At a Glance

| Connector | Install | The conversation is kept as | The atom card shows |
|-----------|---------|-----------------------------|---------------------|
| `PydanticAIConnector` | `chatomon[studio,pydantic-ai]` | PydanticAI's messages | The model string, or the name of the model object |
| `LangChainConnector` | `chatomon[studio,langchain]` | LangChain's messages, given again at every turn on a thread of its own | The model string, or the chat model's own name; its class when it has none |
| `MAFConnector` | `chatomon[studio,maf]` | A session of the agent | The client's model; the client's class when it has none |
| `MAFFoundryConnector` | `chatomon[studio,maf-foundry]` | A session of the agent | The name and the version of the hosted agent |
| `ADKConnector` | `chatomon[studio,adk]` | A session of ADK, kept in memory | The model's name, or the name the model object declares |

When the studio starts, the command prints a note for anything about your atom that can make a turn fail, or that the studio writes to. The sections below say which notes each connector has.

In every framework the atom's tools run for real, and replies arrive whole. A [reload from code](reload.md) keeps the conversation with every connector; code that moves to a connector of another framework starts a new one.

## PydanticAI

Turns run without `deps`. An atom that declares `deps_type` still opens, and the note at start says that tools and instructions that use the dependencies fail.

## LangChain and LangGraph

The studio keeps the conversation itself. At every turn it gives the agent the messages so far together with the new one, reads the conversation back from the result, and runs the turn on a thread of its own.

This matters when the atom has a checkpointer. A checkpointer keeps the state of a thread, including the user message of a turn that failed, and the nucleus would see that message again in the next turn. With a thread for each turn a failed turn leaves nothing behind, as in every other framework, and a saved test replays the conversation the studio ran. The price is that the studio writes one thread to your checkpointer for every turn you try, which the note at start says. With a database as the checkpointer, point the atom at one you can clear.

The thread is also the conversation id of a LangChain atom, so the id that orbitals and the turn report see is new at every turn. In the other frameworks it is one for the whole conversation.

An atom that declares a `context_schema` opens too. Turns run without a context, so tools and middleware that read it fail, and the note at start says so.

## Microsoft Agent Framework

The conversation is a session of the agent. The studio creates it at the first turn and keeps it until **New conversation**, through every change to the orbitals. The session is where a blocked message stays replaced by its placeholder.

The framework writes to the session while a turn runs, so the studio runs each turn on a copy and keeps the copy only when the turn ends well. A turn that fails leaves the conversation as it was.

The connector keeps the history in memory unless you gave it a history provider of your own. When you did, the studio's conversations are written to that provider, and the note at start names it.

The studio sends a message as your code does, so what holds for these connectors holds here: the framework does not store a message that repeats one already in the history word for word, be it yours, the placeholder of a blocked message or a fixed reply of an orbital. The report of the next turn shows the history the atom was given. [Limits on Length and Turns](../orbitals/limits.md) says what this does to a turn limit.

## A Foundry Agent

An atom on `MAFFoundryConnector` wraps an agent that lives in Foundry. In the studio the orbitals, the conversation and the turn report are local. The nucleus is the hosted agent, with the model and the tools configured in Foundry: every turn that passes the input chain calls it with your credentials, and its tools run. A turn the input chain stops never reaches the service. The note at start says all this.

Opening the atom needs what your application needs: the name of the agent and a credential, or a project client. A module like this one is enough:

```{.python test="skip"}
import os

from azure.identity.aio import DefaultAzureCredential

from chatomon import Atom
from chatomon.connectors.maf import MAFFoundryConnector

atom = Atom(
    name="assistant",
    instructions="Be concise.",
    connector=MAFFoundryConnector(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        agent_name=os.environ["FOUNDRY_AGENT_NAME"],
        agent_version=os.environ["FOUNDRY_AGENT_VERSION"],
        credential=DefaultAzureCredential(),
    ),
)
```

Saved as `app.py`, it opens with `chatomon studio app:atom`. The atom card shows the agent's name and version as the model, also when they come from the `FOUNDRY_AGENT_NAME` and `FOUNDRY_AGENT_VERSION` environment variables instead of the code.

## Google ADK

The studio runs the `App` the atom builds on ADK's in-memory runner. In ADK the session service belongs to the runner, not to the app, so the studio never reaches the session store of your application: there is nothing to point elsewhere and nothing to clear. Artifacts and memory are in memory too and last one turn: a tool finds nothing that a tool saved in an earlier turn.

The conversation is an ADK session. ADK keeps the message of a turn that failed in the session it ran on, so the studio runs each turn on a copy of the session and keeps the copy only when the turn ends well. The id of the session stays the same, and it is the conversation id that orbitals and the turn report see.

The plugins you gave the connector run on every turn of the studio too, after chatomon's, and write where they usually write: one that sends analytics sends the studio's turns. The note at start names them.

## Limits

- **A turn that waits for an answer is shown as failed.** The studio cannot answer a LangGraph interrupt, approve a tool call in Microsoft Agent Framework, nor answer a long-running tool or a request for a confirmation in Google ADK. The conversation goes on as it was before that turn. With a history provider of your own, what the framework wrote to it before stopping stays there.
- **A turn that ends with no reply is left out.** When the nucleus writes nothing, or a LangGraph run ends on the result of a tool, the turn is shown as aborted and does not enter the conversation.
- **The agent the studio replaces is not closed.** Every change to the orbitals builds the agent again, and so does a reload from code. A Foundry atom built from an endpoint and a credential leaves a client open for each of them, until the studio stops.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `PydanticAIConnector` | `src/chatomon/connectors/pydantic_ai.py` | Builds an atom into a PydanticAI agent |
| `LangChainConnector` | `src/chatomon/connectors/langchain.py` | Builds an atom into a LangGraph agent |
| `MAFConnector`, `MAFFoundryConnector` | `src/chatomon/connectors/maf/` | Build an atom into a Microsoft Agent Framework agent, with a chat client or on an agent hosted in Foundry |
| `ADKConnector` | `src/chatomon/connectors/adk.py` | Builds an atom into a Google ADK app |
| The studio's runners | `src/chatomon/studio/_runners/` | One module per framework: runs a turn, names the model, says what to know at start |

## Related Documentation

- [Studio](index.md) — installing and starting the studio
- [Saving a Turn as a Test](testing.md) — the test the studio writes for each framework
- [Connectors](../connectors/index.md) — how each connector builds a native agent
- [LangChain / LangGraph](../connectors/langchain.md) — conversations, checkpointers and human in the loop
- [Microsoft Agent Framework](../connectors/maf.md) — sessions, local history and Foundry agents
- [Google ADK](../connectors/adk.md) — the app, sessions, streaming and ADK's own telemetry

---
**Last Updated:** October 2026
**Status:** Implemented
