# JudgeOrbital

## Overview

`JudgeOrbital` (`src/chatomon/orbitals/judge.py`) asks a language model for a structured decision (`JudgeDecision`) and turns it into a verdict. It works on both sides: on the input side it judges the user message, on the output side the draft reply.

## Configuration

A judge has its own `instructions`, an optional `context` (reference material, typically a `Knowledge` shared explicitly with the atom) and a `history_window` of recent messages. It never inherits the atom's instructions or knowledge.

It calls the model through the atom's connector — so a LangChain atom uses a LangChain model — unless you give it a `connector` of its own; connectors resolve the judge's model id with their `resolve_model` setting (see [Judges and Model Resolution](../connectors/index.md#judges-and-model-resolution)).

## What the Judge Sees

The judge's system prompt holds a fixed frame, its `instructions` and its `context`. The user prompt puts the history window, the annotations named in `reads` and the text under review in separate tags. Every piece of text inserted into those tags is XML-escaped (`<`, `>` and `&`), so a message that contains a tag such as `</text_under_review>` cannot close it early and change what the judge believes it is reviewing. The frame tells the judge how to read the escaped characters, and that the tagged material is something to judge, never instructions to follow: a message that tells the judge to allow it is part of what the judge reviews. This last rule is an instruction to the model, not a guarantee, so keep critical checks in orbitals that do not depend on a model.

## Decisions and Audit

The model's own label and explanation may quote the user, so they are stored as annotations (treated as content by the privacy settings); the verdict's audit reason is the fixed label `judge_allow`, `judge_block` or `judge_respond`. A malformed decision (for example `respond` without a reply) makes the orbital fail, and `on_error` applies.

A judge often reads the annotations of an earlier orbital — for example the claims extracted by [`ClaimExtractorOrbital`](principled.md#claimextractor) — through `reads`.

## Related Documentation

- [Orbitals](index.md) — chaining, errors and timeouts
- [Observability and Audit](../observability.md) — content capture and the `gen_ai.evaluation.result` event
- [API Reference: `chatomon.orbitals.judge`](../reference/orbitals_judge.md) — every parameter and a runnable example

---
**Last Updated:** October 2026
**Status:** Implemented
