# Tuning Orbitals

## Overview

Calibrating a guardrail means changing it and sending the same message again. In the studio you change orbitals from their cards and from the drawer: settings, order, on and off, and which of chatomon's own orbitals are in the chain. Every change takes effect from the next turn and lives in the studio session; your code is never written. [Back to Code](export.md) explains how to take the result with you.

## What the Studio Can Change

The studio reads an orbital's settings from its constructor: an argument that takes a plain value (a number, text, a choice, a list of orbital names) is a setting, and an argument that takes an object, such as a client or a connector, comes from your code and stays as it is.

| Orbital | Settings | On, off and order | Delete |
|---------|----------|-------------------|--------|
| chatomon's own, defined in your code | Plain values are edited; objects stay as in your code | Yes | No: turn it off |
| `@orbital` functions and classes you wrote | Read-only, with the reason | Yes | No: turn it off |
| Added in the studio | Edited, name included | Yes | Yes, with **Undo** |

chatomon's own orbitals are those of `chatomon.orbitals` and `chatomon.orbitals.principled`: `MessageLengthOrbital`, `TurnLimitOrbital`, `JudgeOrbital`, `ScopeGuardOrbital` and `ClaimExtractorOrbital`. A class of yours that inherits from one of them counts as your own.

## Editing a Setting

Clicking an orbital opens its card, with one field per setting. A field is applied when you leave it or press ++enter++; a choice or a switch is applied at once. In a text field ++shift+enter++ starts a new line. The fingerprint at the bottom of the card changes with every accepted setting, and a field that differs from your code is marked as changed.

A value the orbital does not accept is refused: the message appears under the field, which keeps what you typed, and the session stays as it was. If you close the card before the answer arrives, the message appears at the bottom of the page instead. The message is the orbital's own, the one its constructor raises in your code, for example `set max_words, max_chars or both`. When two settings depend on each other, change them in the order the message suggests: to lower `max_turns` below `warn_before`, lower `warn_before` first.

Arguments that are objects are listed as **from code**, with the name of the variable that holds them when the studio finds it in the atom's module. Change them in your code.

Two details are worth knowing:

- **A judge's `context` is edited as text.** While you leave it alone, the judge keeps the `Knowledge` object from your code; once you change it, the judge has its own text and no longer follows the atom's knowledge.
- **An orbital that your code puts in both chains is one orbital.** A change to its settings applies to both sides; its switch and its position are per chain.

## Changing the Order

Drag an orbital by its grip to another place in the same chain, or open its card and use **Move up** and **Move down**, which also work from the keyboard and on a phone. An order the atom would not accept, such as an orbital placed before one whose annotations it reads, is refused with the reason.

## Adding and Deleting

**Add**, next to each chain, lists the orbitals that can be built from plain values: `JudgeOrbital`, `MessageLengthOrbital` and `TurnLimitOrbital`, each on the sides it supports. Orbitals that need a client or a function are added in your code.

Choosing one opens a form with its name and its settings. The name is proposed and can be changed: lowercase letters, digits, `_`, `-` and `.`, unique in the atom. Settings without a default, such as a judge's `model` and `instructions`, must be filled in; nothing is created until **Add** is accepted. A judge added in the studio calls its model through the atom's connector, so `model` is an identifier in that connector's format.

An added orbital goes at the end of its chain, can be renamed from its card, and can be deleted from the card or from its row in the drawer. Deleting offers **Undo**. Orbitals defined in your code cannot be deleted or renamed here: turn them off instead.

## Changes Against the Code

The foot of the drawer counts how the session differs from your code and lists the differences: an orbital turned off, a setting changed, an orbital added, a chain in another order. A dot marks every orbital that was added, turned off or edited, and another marks the **Atom** button in the header.

**Reset to code** in the drawer goes back to the atom as your code defines it. The same button in a card resets that orbital alone: its settings and its switch, not its position. Both offer **Undo**, as deleting an orbital and importing a preset do: in the message at the bottom of the page for a few seconds, and at the foot of the drawer until the next change.

Stopping the studio discards the session. To keep what you tuned, copy it as code or export a preset first.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `Orbital` | `src/chatomon/orbitals/base.py` | The constructor arguments the studio reads as settings; `config()` and `fingerprint()` shown in the cards |
| `Atom.validate` | `src/chatomon/atom.py` | The chain rules a change must keep: unique names, `reads` pointing backwards, supported sides |

## Related Documentation

- [Reading the Screen](screen.md) — the drawer, the stage and the turn reports
- [Back to Code](export.md) — Copy as code and presets
- [Orbitals](../orbitals/index.md) — chaining, `reads` and the settings every orbital has
- [JudgeOrbital](../orbitals/judge.md) and [Limits](../orbitals/limits.md) — the orbitals that can be added from the studio

---
**Last Updated:** October 2026
**Status:** Implemented
