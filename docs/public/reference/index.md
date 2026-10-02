# API Reference

## Overview

The public API of chatomon, generated from the docstrings of the code. There is one page per public module or connector package; each starts with a summary of its members, and every class and function includes its parameters, return values, raised exceptions and, for the main entry points, runnable examples.

## Table of Contents

### Core

| Page | Module | Contents |
|------|--------|----------|
| [Atom](atom.md) | `chatomon.atom` | Declares an agent, validates its orbital chains and builds the native agent |
| [Knowledge](knowledge.md) | `chatomon.knowledge` | Plain-text knowledge added to the nucleus prompt |
| [Schemas](schemas.md) | `chatomon.schemas` | Verdicts, the orbital context, engine decisions and turn reports |
| [Errors](errors.md) | `chatomon.errors` | The exceptions chatomon raises |

### Engine

| Page | Module | Contents |
|------|--------|----------|
| [Engine](engine.md) | `chatomon.engine` | Runs the orbital chains and turns verdicts into decisions for the connectors |

### Orbitals

| Page | Module | Contents |
|------|--------|----------|
| [Base](orbitals_base.md) | `chatomon.orbitals.base` | The `Orbital` extension point, the `orbital` decorator and `FunctionOrbital` |
| [Judge](orbitals_judge.md) | `chatomon.orbitals.judge` | `JudgeOrbital` and its structured `JudgeDecision` |
| [Limits](orbitals_limits.md) | `chatomon.orbitals.limits` | `MessageLengthOrbital` and `TurnLimitOrbital`: caps on length and turns |
| [Principled](orbitals_principled.md) | `chatomon.orbitals.principled` | ScopeGuard and ClaimExtractor orbitals (extra `principled`) |

### Connectors

| Page | Module | Contents |
|------|--------|----------|
| [Base](connectors_base.md) | `chatomon.connectors.base` | The `Connector` extension point and `render_guidance` |
| [PydanticAI](connectors_pydantic_ai.md) | `chatomon.connectors.pydantic_ai` | `PydanticAIConnector` (extra `pydantic-ai`) |
| [LangChain](connectors_langchain.md) | `chatomon.connectors.langchain` | `LangChainConnector` (extra `langchain`) |
| [Microsoft Agent Framework](connectors_maf.md) | `chatomon.connectors.maf` | `MAFConnector` and `MAFFoundryConnector` (extras `maf`, `maf-foundry`) |

### Observability

| Page | Module | Contents |
|------|--------|----------|
| [Observability](observability.md) | `chatomon.observability` | The tracer, the content policy and the hashing helpers |

## Exports

The packages re-export their main names, so `from chatomon import Atom` works as well as `from chatomon.atom import Atom`. Each name is documented in the module that defines it.

### `chatomon`

| Name | Documented in |
|------|---------------|
| `Atom` | [Atom](atom.md) |
| `Knowledge` | [Knowledge](knowledge.md) |
| `Verdict`, `OrbitalContext`, `TurnReport` | [Schemas](schemas.md) |
| `ChatomonError`, `ChatomonConfigError`, `MissingExtraError` | [Errors](errors.md) |
| `Orbital`, `orbital` | [Orbitals / Base](orbitals_base.md) |

`chatomon.__version__` holds the installed version as a string.

### `chatomon.orbitals`

| Name | Documented in |
|------|---------------|
| `Orbital`, `orbital`, `FunctionOrbital`, `OrbitalFunction` | [Orbitals / Base](orbitals_base.md) |
| `JudgeOrbital`, `JudgeDecision` | [Orbitals / Judge](orbitals_judge.md) |
| `MessageLengthOrbital`, `TurnLimitOrbital` | [Orbitals / Limits](orbitals_limits.md) |

The Principled Intelligence orbitals are not re-exported: import them from `chatomon.orbitals.principled`, which needs the `principled` extra.

### `chatomon.connectors`

| Name | Documented in |
|------|---------------|
| `Connector`, `render_guidance` | [Connectors / Base](connectors_base.md) |

The framework connectors are not re-exported: import them from `chatomon.connectors.pydantic_ai`, `chatomon.connectors.langchain`, or `chatomon.connectors.maf`, which need the matching extra.

### `chatomon.connectors.maf`

| Name | Documented in |
|------|---------------|
| `MAFConnector`, `MAFFoundryConnector` | [Connectors / Microsoft Agent Framework](connectors_maf.md) |

## Editing the Reference

These pages are generated from the docstrings in `src/chatomon/`: to change them, edit the docstrings, not the files in `docs/public/reference/`.

## Related Documentation

- [Concepts](../concepts.md) — the mental model behind these objects
- [Home](../index.md) — overview and table of contents of the documentation

---
**Last Updated:** September 2026
**Status:** Implemented
