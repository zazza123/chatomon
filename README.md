<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/zazza123/chatomon/main/docs/public/assets/brand/hero-dark.svg">
    <img alt="chatomon" src="https://raw.githubusercontent.com/zazza123/chatomon/main/docs/public/assets/brand/hero-light.svg" width="420">
  </picture>
</h1>

Declare a chat agent once — its instructions, its knowledge and the **orbitals** (input/output guardrails) around it — and build it into a native agent of the framework you already use: PydanticAI, LangChain/LangGraph, Microsoft Agent Framework, or Google ADK.

> Status: pre-alpha, under active development.

## Install

```bash
pip install 'chatomon[pydantic-ai]'   # or 'chatomon[langchain]'
pip install 'chatomon[maf]'           # MAF core; add your provider's MAF client package
pip install 'chatomon[maf-foundry]'   # FoundryChatClient and FoundryAgent
pip install 'chatomon[adk]'           # Google ADK 2.10 to 2.12
pip install 'chatomon[principled]'    # Principled Intelligence ScopeGuard / ClaimExtractor orbitals
pip install 'chatomon[studio,pydantic-ai]'  # the local studio: chatomon studio module:atom
pip install 'chatomon[spectral,pydantic-ai]'  # the studio with its Spectral add-on: evaluation at scale
```

Both MAF connectors support input and output orbitals on non-streaming runs. The [MAF guide](docs/public/connectors/maf.md) covers local Foundry sessions and connector configuration.

## Quickstart

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
agent = atom.build()  # a native pydantic_ai.Agent
assert agent.run_sync("I want a refund").output == "Refunds are handled by email."
```

Replace `TestModel(...)` with a real model, for example `"openai:gpt-5"`.

To try an atom in the browser, run `chatomon studio speedypack.app:atom` and open the printed link: the studio shows every orbital deciding, turn by turn, and lets you tune the orbitals and copy the result back as code.

## Documentation

The documentation site is built with Zensical from `docs/public/`: run `uv run zensical serve` and open http://localhost:8000. It covers concepts, orbitals, [connectors including MAF](docs/public/connectors/maf.md), observability and deployment, plus the API reference generated from the code.

## Contributing

Bug reports, ideas and pull requests are welcome — start with
[`CONTRIBUTING.md`](CONTRIBUTING.md). Security issues: see
[`SECURITY.md`](SECURITY.md).
