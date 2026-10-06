# Studio

## Overview

![](../assets/brand/mascot/animated/turn-light.svg#only-light){ .cm-page-mascot }
![](../assets/brand/mascot/animated/turn-dark.svg#only-dark){ .cm-page-mascot }
The studio is a local web page for trying an atom before it goes anywhere. You chat with it, watch each orbital decide while the turn runs, tune the orbitals until they behave, and take the result back to your code. It is a development tool: the atom stays in your code, the studio never changes your code, and it is not meant to be deployed or to serve users.

## Key Features

- **Live turns** — the atom on stage shows every orbital deciding, and each turn keeps its report.
- **Tuning** — switch orbitals on and off, change their settings, reorder them and add chatomon's own orbitals. Changes live in the studio session.
- **Back to code** — the tuned orbitals as Python to paste into your atom, or as a preset file to load again later.
- **Tests** — a turn, or the conversation up to it, saved as a pytest file that runs against the atom from your code.

## Table of Contents

| Page | What it covers |
|------|----------------|
| [Reading the Screen](screen.md) | The atom drawer, the atom on stage, turns and their reports |
| [Tuning Orbitals](tuning.md) | Editing settings, changing the order, adding and deleting orbitals, changes against the code |
| [Back to Code](export.md) | Copy as code, presets and the preset format |
| [Saving a Turn as a Test](testing.md) | What a saved test checks, offline and integration tests, where the file goes |

## Installing and Starting

Install the `studio` extra together with the extra of the atom's framework: `pip install 'chatomon[studio,pydantic-ai]'`. Then give the command the atom's location as `module:attribute`:

```bash
chatomon studio speedypack.app:atom
```

The command imports the module from the current directory, prints a link and opens it in the browser. Stop it with ++ctrl+c++; a turn that is still waiting for its model does not hold the exit. `python -m chatomon.studio` takes the same arguments.

| Option | Default | Purpose |
|--------|---------|---------|
| `--host` | `127.0.0.1` | The address to listen on |
| `--port` | `8016` | The port to listen on |
| `--no-browser` | off | Print the link without opening it |

The default port is 8016 for oxygen-16, the most common oxygen atom: the "o" of chatomon is the letter the atom replaces in the wordmark.

When a function builds the atom, for example because it reads configuration or creates clients first, point at the function instead: `chatomon studio speedypack.app:make_atom`. The studio calls it with no arguments; a function that needs some is refused with a message that names them.

From Python, `serve` in `chatomon.studio` does the same and blocks until interrupted:

```{.python test="skip"}
from chatomon.studio import serve

from speedypack.app import atom

serve(atom)
```

Started this way the studio does not know which module the atom comes from, which limits what [Copy as code](export.md#copy-as-code) can name.

## Content in the Studio

The header shows **Content visible · local only**, or **network address** when you started the studio on one; clicking the notice explains it. The studio runs a copy of your atom with `capture_content="full"`, because showing guidance, replies and judge explanations is what it is for.

Your atom keeps its own setting. The copy also has its own turn callback and tracer: your `on_turn`, your tracer provider and your redactor are not called, so chatomon's reports and spans of a studio session stay in the studio.

Two things still go where your application sends them. Tracing you set up outside chatomon, such as PydanticAI's own instrumentation, records a studio turn like any other, messages included if you configured it so. chatomon's log records also follow your logging setup; they carry labels, never message text.

Model calls are real. With a real model, a turn in the studio costs what a turn costs.

## Security

The studio shows conversation text and calls models with your credentials, so it protects itself in five ways:

- It listens on `127.0.0.1`, which only your machine can reach.
- The link it prints carries a key for this run. The page keeps it in a cookie; opened without it, the page asks for the link from the terminal. Treat the link like a password.
- It answers only on its own address and accepts changes only from its own page, so other sites open in your browser cannot start turns or change orbitals.
- Its pages run only the studio's own files and cannot be shown inside another page.
- With `--host` and a network address it prints a warning: anyone who reaches that address with the link can read the conversation and use your models. Addresses that listen everywhere, such as `0.0.0.0`, are refused.

## Limits of This Version

- **PydanticAI atoms only.** An atom built with another connector is refused with a message that says so.
- **No streaming.** Replies arrive whole, because output orbitals check a reply before it is sent.
- **Your own orbitals are read-only.** The studio edits the settings of chatomon's own orbitals. `@orbital` functions and orbital classes you wrote can be turned off and moved; their settings are changed in your code.
- **Changes live in the session.** Stopping the studio discards them: copy the code or export a preset first.
- **Restart to reload.** After changing the code, stop the command and start it again.
- **No dependencies.** Turns run without PydanticAI `deps`: an atom whose tools or instructions need them fails when it uses them, and the command says so when it starts.
- **A saved test does not repeat what the output chain decided.** It runs the real nucleus, which writes something else every time: for a turn in which an output orbital stopped a reply, the test checks only how the turn ended.
- **Rejected replies are not shown.** When an output orbital makes the nucleus write a reply again, the report shows the orbital's guidance, not the text it rejected.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `serve`, `DEFAULT_PORT` | `src/chatomon/studio/server.py` | Starts the studio for an atom |
| `chatomon studio` | `src/chatomon/_cli.py` | The command: loads `module:attribute` and starts the studio |

## Related Documentation

- [Concepts](../concepts.md) — what happens to a message during a turn, and the final actions
- [Observability and Audit](../observability.md) — turn reports and content capture
- [Orbitals](../orbitals/index.md) — chaining, failures and annotations
- [API Reference: `chatomon.studio.server`](../reference/studio.md) — `serve` and its parameters

---
**Last Updated:** October 2026
**Status:** Implemented
