# chatomon

## Overview

chatomon ("chat" + "atom") lets you declare a chat agent once — an **atom** made of instructions, knowledge and a connector — surround it with **orbitals** that check every message going in and every reply coming out, and build it into a **native agent** of the framework you already use. chatomon is not a chatbot framework: it never runs the agent itself, it installs its checks inside PydanticAI or LangChain/LangGraph and gets out of the way.

## Key Features

- **One declaration, native agents** — `Atom.build()` returns a real `pydantic_ai.Agent` or a LangGraph `CompiledStateGraph`; every framework feature (tools, checkpointers, streaming, tracing) keeps working.
- **Orbitals** — ordered input and output guardrails with four outcomes: let the message through, block it with a fixed reply, block it and let the model write a guided refusal, or answer in place of the model.
- **Guidance instead of hard blocks** — orbitals can steer the model for one turn, or make it regenerate a reply, without the guidance ever entering the conversation history.
- **Fail-closed by default** — an orbital that fails or times out blocks the message unless configured otherwise.
- **Guard models included** — an LLM judge and wrappers for Principled Intelligence's ScopeGuard (V1 and V2) and ClaimExtractor.
- **Audit and privacy** — a structured report for every turn, OpenTelemetry spans and logs that carry no message content unless you allow it.

## Installation

chatomon is installed with the extra of the framework you use: `pip install 'chatomon[pydantic-ai]'` or `pip install 'chatomon[langchain]'`. The Principled Intelligence orbitals need `pip install 'chatomon[principled]'`. The core package depends only on pydantic and the OpenTelemetry API.

## Quickstart

An atom with one input orbital that turns refund requests into a guided refusal. The fake models stand in for real ones; replace them with a model string such as `"openai:gpt-5"`.

=== "PydanticAI"

    ```python
    from pydantic_ai.models.test import TestModel

    from chatomon import Atom, Knowledge, OrbitalContext, Verdict, orbital
    from chatomon.connectors.pydantic_ai import PydanticAIConnector


    @orbital
    def no_refunds(ctx: OrbitalContext) -> Verdict:
        if "refund" in ctx.message.lower():
            return Verdict.block(guidance="Explain that refunds are handled by email.")
        return Verdict.allow()


    atom = Atom(
        name="speedypack",
        instructions="You are the customer assistant of SpeedyPack, a parcel courier.",
        knowledge=Knowledge("Opening hours: Monday to Friday, 9-18."),
        connector=PydanticAIConnector(TestModel(custom_output_text="Refunds are handled by email.")),
        input_orbitals=[no_refunds],
    )
    agent = atom.build()
    result = agent.run_sync("I want a refund")
    assert result.output == "Refunds are handled by email."
    ```

=== "LangChain"

    ```python
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    from chatomon import Atom, Knowledge, OrbitalContext, Verdict, orbital
    from chatomon.connectors.langchain import LangChainConnector


    @orbital
    def no_refunds(ctx: OrbitalContext) -> Verdict:
        if "refund" in ctx.message.lower():
            return Verdict.block(guidance="Explain that refunds are handled by email.")
        return Verdict.allow()


    model = GenericFakeChatModel(messages=iter([AIMessage(content="Refunds are handled by email.")]))
    atom = Atom(
        name="speedypack",
        instructions="You are the customer assistant of SpeedyPack, a parcel courier.",
        knowledge=Knowledge("Opening hours: Monday to Friday, 9-18."),
        connector=LangChainConnector(model),
        input_orbitals=[no_refunds],
    )
    agent = atom.build()
    result = agent.invoke({"messages": [HumanMessage(content="I want a refund")]})
    assert result["messages"][-1].content == "Refunds are handled by email."
    ```

## Table of Contents

<div class="grid cards" markdown>

-   :lucide-atom:{ .lg .middle } __Concepts__

    ---

    Atoms, orbitals, verdicts and what happens to a message during a turn.

    [:lucide-arrow-right: Concepts](concepts.md)

-   :lucide-shield-check:{ .lg .middle } __Orbitals__

    ---

    Chaining and failures, writing your own, the LLM judge and the Principled Intelligence guard models.

    [:lucide-arrow-right: Orbitals](orbitals/index.md)

-   :lucide-plug:{ .lg .middle } __Connectors__

    ---

    How atoms become PydanticAI agents and LangChain graphs: conversations, streaming, tools and tracing.

    [:lucide-arrow-right: Connectors](connectors/index.md)

-   :lucide-activity:{ .lg .middle } __Observability and audit__

    ---

    Logs, OpenTelemetry spans, turn reports, content capture and compliance notes.

    [:lucide-arrow-right: Observability](observability.md)

-   :lucide-server:{ .lg .middle } __Deployment__

    ---

    Running guard models behind private, authenticated endpoints.

    [:lucide-arrow-right: Deployment](deployment/index.md)

-   :lucide-book-open:{ .lg .middle } __API reference__

    ---

    Every public module, generated from the code, with runnable examples.

    [:lucide-arrow-right: API reference](reference/index.md)

</div>

The [Changelog](changelog.md) explains where release notes are published and how versions are numbered.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `Atom` | `src/chatomon/atom.py` | Declares the agent, validates the orbital chains and builds the native agent |
| `Knowledge` | `src/chatomon/knowledge.py` | Plain-text knowledge added to the nucleus prompt |
| `Orbital`, `orbital` | `src/chatomon/orbitals/base.py` | The orbital extension point and the function decorator |
| `Verdict`, `TurnReport` | `src/chatomon/schemas.py` | The result of an orbital and the audit record of a turn |
| `Engine` | `src/chatomon/engine.py` | Runs the orbital chains and turns verdicts into decisions |
| `PydanticAIConnector` | `src/chatomon/connectors/pydantic_ai.py` | Builds atoms into PydanticAI agents |
| `LangChainConnector` | `src/chatomon/connectors/langchain.py` | Builds atoms into LangChain agents (LangGraph graphs) |
| `JudgeOrbital` | `src/chatomon/orbitals/judge.py` | An LLM that judges messages with its own instructions |
| `ScopeGuardOrbital`, `ClaimExtractorOrbital` | `src/chatomon/orbitals/principled.py` | Principled Intelligence guard models as orbitals |

## Related Documentation

- [Concepts](concepts.md) — start here for the mental model
- [API Reference](reference/index.md) — parameters and runnable examples for every public object

---
**Last Updated:** September 2026
**Status:** Implemented
