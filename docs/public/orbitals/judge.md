# JudgeOrbital

## Overview

`JudgeOrbital` (`src/chatomon/orbitals/judge.py`) asks a language model for a structured decision (`JudgeDecision`) and turns it into a verdict. It works on both sides: on the input side it judges the user message, on the output side the draft reply.

## Configuration

A judge has its own `instructions`, an optional `context` (reference material, typically a `Knowledge` shared explicitly with the atom) and a `history_window` of recent messages. It never inherits the atom's instructions or knowledge.

It calls the model through the atom's connector — so a LangChain atom uses a LangChain model — unless you give it a `connector` of its own; connectors resolve the judge's model id with their `resolve_model` setting (see [Judges and Model Resolution](../connectors.md#judges-and-model-resolution)).

## Decisions and Audit

The model's own label and explanation may quote the user, so they are stored as annotations (treated as content by the privacy settings); the verdict's audit reason is the fixed label `judge_allow`, `judge_block` or `judge_respond`. A malformed decision (for example `respond` without a reply) makes the orbital fail, and `on_error` applies.

A judge often reads the annotations of an earlier orbital — for example the claims extracted by [`ClaimExtractorOrbital`](principled.md#claimextractor) — through `reads`.

## Related Documentation

- [Orbitals](index.md) — chaining, errors and timeouts
- [Observability and Audit](../observability.md) — content capture and the `gen_ai.evaluation.result` event
- [API Reference: orbitals.judge](../reference/orbitals_judge.md) — every parameter and a runnable example

---
**Last Updated:** September 2026
**Status:** Implemented
