# Saving a Turn as a Test

## Overview

A turn you tried in the studio can become a regression test. **Save as test**, in the report of a turn, writes a pytest file that sends the same messages to the atom from your code and checks that each turn ends the way it did in the studio. The test runs the real atom: the nucleus and the orbitals call their own models, so most saved tests need the credentials your application needs.

## What a Test Checks

A saved test builds the atom from your code, collects its turn reports and sends one message per turn, carrying the conversation from one turn to the next. This one holds two turns: the first was answered, the second was stopped by an orbital called `no_refunds`.

```{.python test="skip"}
"""Saved from chatomon studio: speedypack.app:atom, turns 1-2."""

import pytest

from chatomon import TurnReport
from speedypack.app import atom


@pytest.mark.integration  # the nucleus and scope call real models
def test_i_want_a_refund(monkeypatch: pytest.MonkeyPatch) -> None:
    reports: list[TurnReport] = []
    monkeypatch.setattr(atom, "on_turn", reports.append)
    agent = atom.build()

    # Turn 1: answered
    result = agent.run_sync("Where is my parcel?")
    report = reports[-1]
    assert report.final_action == "answered"
    assert [(r.name, r.action) for r in report.input if r.action != "allow"] == []
    assert [
        (r.name, r.action) for a in report.output for r in a if r.action != "allow"
    ] == []

    # Turn 2: blocked
    result = agent.run_sync("I want a refund", message_history=result.all_messages())
    report = reports[-1]
    assert report.final_action == "blocked"
    assert [(r.name, r.action) for r in report.input if r.action != "allow"] == [
        ("no_refunds", "block"),
    ]
    assert result.output == "We don't handle refunds in chat."
```

`monkeypatch` puts the atom back as it was when the test ends, and your own `on_turn` is not called while it runs. The atom keeps its `capture_content`: the checks need names and actions only.

| The test checks | When |
|-----------------|------|
| How the turn ended: the `final_action` of its report | Always |
| The input orbitals that stopped the turn, with what they did | Always. An orbital that let the message through is not listed, so adding an orbital to your code does not break the tests you saved |
| That no output orbital stopped a reply | When none did in the studio |
| The text of the reply | When the input chain produced it and no model can have written it: no orbital that calls a model ran in the turn, or the reply is the atom's refusal |

A test never checks what the nucleus writes, and it cannot repeat what an output orbital decided on a reply: a real nucleus writes something different every time. For a turn in which an output orbital stopped a reply, the test checks only how the turn ended, and the card says so. For a turn in which none did, the test expects none to. With an output judge that is close to its limit on that reply, such a test can pass one run and fail the next: the judge needs tuning, not the test.

A turn that failed cannot be saved.

## The Test for Each Framework

The example above is the test of a PydanticAI atom. What a test checks is the same for every framework; the lines that run a turn follow the framework of the atom.

| | PydanticAI | LangChain | Microsoft Agent Framework |
|---|---|---|---|
| A turn | `agent.run_sync("…")` | `await agent.ainvoke({"messages": [HumanMessage("…")]})` | `await agent.run("…", session=session)` |
| The conversation | `message_history=result.all_messages()` | The messages of the turn before go in front of the new one | A session, created once the atom is built |
| The reply | `result.output` | `result["messages"][-1].content` | `result.text` |
| Offline, in place of the nucleus | `TestModel` | `OfflineNucleus`, a chat model the test defines | `OfflineNucleus`, a client the test defines |

**Asynchronous tests.** The studio runs a LangChain agent asynchronously, and Microsoft Agent Framework has no other way, so the tests of these two frameworks are `async def` with `@pytest.mark.asyncio(loop_scope="session")` and need `pytest-asyncio` 0.24 or later. A LangChain test runs the agent as the studio did, which is what lets tools and middleware that are asynchronous only work in the test too. The loop is the session's because a client your module creates when it is imported keeps its connections on one loop: with a loop for each test, the second test that reaches a model would fail.

**LangChain.** With a checkpointer every call also takes a thread of its own, as every turn does in the studio. Offline, the test defines a chat model that takes the atom's tools and calls none, so a message your code lets through fails the test on what it expects. That stand-in replaces the model of the connector: a middleware of yours that chooses the model itself can still reach a real one, and the card says so whenever the atom has middleware.

**Microsoft Agent Framework**, an agent hosted in Foundry included. Offline, the test defines a small client and puts it on the agent it built, in place of the nucleus, so a Foundry atom never reaches its hosted agent. That client is written on the framework's own client classes: save the test again if they change.

Offline means the nucleus is not called. The module of your atom is still imported, with whatever it reads when it is, and in Microsoft Agent Framework the context providers of the atom still run on a turn the input chain stops: one that calls a service calls it.

## The Conversation or the Turn Alone

A turn may depend on the ones before it: a turn limit counts them, and a judge reads the history. The studio cannot tell, so a test holds the whole conversation up to the saved turn, and every turn in it is checked. When the earlier turns have nothing to do with the one you are saving, clear **Replay the turns before it** and the test holds that turn alone. Turns that failed are left out: they left nothing in the conversation. A turn saved alone that did depend on what came before it gives a test that fails, and the card reminds you of it.

Each replayed turn is a real call, so a long conversation makes a slow test. **New conversation** before the turn you care about keeps tests short.

## Offline and Integration

A test runs offline only when no model is needed: in every turn the input chain produced the reply, and no orbital that ran calls a model. Such a test puts a stand-in in place of the nucleus: PydanticAI's test model, or for the other frameworks a small one the test defines. The nucleus then needs no credentials, the stand-in calls none of the atom's tools, and a change in your code that lets the message through fails the test instead of calling a model. The module of your atom must still import without credentials for the test to run with none.

Every other test is marked `integration`, with a comment that names what calls real models. Register the marker in your pytest configuration, and leave these tests out where there are no credentials:

```toml
[tool.pytest.ini_options]
markers = ["integration: calls real models"]
```

| Command | Runs |
|---------|------|
| `python -m pytest tests/studio` | Every saved test |
| `python -m pytest tests/studio -m "not integration"` | Only the tests that run offline |

Run them with `python -m pytest`, from the directory you start the studio in: it puts that directory on the import path, where the module of your atom is.

The studio knows whether an orbital calls a model from the orbital itself: see [Saying Whether It Calls a Model](../orbitals/custom.md#saying-whether-it-calls-a-model). A function written with `@orbital` is taken as calling no model unless it says otherwise, and the card names the functions it relied on.

## Changes That Are Not in Your Code

A saved test builds the atom from your code. When a turn ran with changes you made in the studio, the test describes a turn your code cannot produce yet, and it fails until those changes are in the code. The card warns about it, lists the changes each turn ran with, and opens [Copy as code](export.md#copy-as-code) so you can paste the orbitals first. Saved before the code changes and passing after, the test shows that the change did what you tried. The same holds for a test that runs offline: it is offline for the turn as it ran in the studio, and until the changes are in your code, the orbitals of your code run, which may call a model.

## Turns from Before a Reload

A saved test runs your code as it is now. A turn that ran before the last [reload](reload.md) ran on the code of that moment, and the test checks what happened then. When a test holds such turns the card says which: if the code changed what they check, the test fails. Run the turn again with **Re-run** to see it on the code you have now.

Two more things follow from the code having changed. A test that is offline was read from what ran then: if your code now has an orbital that calls a model in front of the one that stopped the turn, the test calls it, and the card says that it may. And a turn you tried with studio changes is no longer asked to be copied into the code once your code has exactly those orbitals.

## Where the File Goes

The card proposes `tests/studio/test_<words of the message>.py`, one file per test. The path can be changed, within these limits:

- the file stays inside the directory the studio was started from;
- its name ends in `.py`, and the path does not start with `~`, which is not expanded;
- an existing file is replaced only when the studio wrote it, and only after you confirm. Any other file is refused. The studio knows its files by their first line, so a saved test you edited by hand is replaced too, and your edits are lost.

The page chooses the turn and the path. What is written is always the test the studio generated, and the card then shows the command that runs it.

## Limits

- **Decisions of the output chain are not repeated.** A test cannot make the nucleus write the reply that an output orbital stopped.
- **Started with `serve()`.** The studio does not know which module the atom lives in. The card shows the test with its import left to complete, and copies it; it does not write the file. The same holds for a module whose name is not a Python name, such as `my-shop`, which no import can name.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `Orbital.calls_model` | `src/chatomon/orbitals/base.py` | Whether an orbital calls a model, which decides if a test can run offline |
| `TurnReport` | `src/chatomon/schemas.py` | The record of a turn that a saved test reads |

## Related Documentation

- [Reading the Screen](screen.md) — the report of a turn, where a test is saved from
- [Back to Code](export.md) — Copy as code, for changes that are not in the code yet
- [Frameworks in the Studio](frameworks.md) — how the studio runs the atoms of each framework
- [Writing Your Own Orbital](../orbitals/custom.md) — `calls_model` for classes and functions
- [Observability and Audit](../observability.md) — the turn report and its final actions

---
**Last Updated:** October 2026
**Status:** Implemented
