# Studio

## Overview

![](assets/brand/mascot/animated/turn-light.svg#only-light){ .cm-page-mascot }
![](assets/brand/mascot/animated/turn-dark.svg#only-dark){ .cm-page-mascot }
The studio is a local web page for trying an atom before it goes anywhere. You chat with it, watch each orbital decide while the turn runs, and switch orbitals on and off to see what changes. It is a development tool: the atom stays in your code, the studio never edits it, and it is not meant to be deployed or to serve users.

## Installing and Starting

Install the `studio` extra together with the extra of the atom's framework: `pip install 'chatomon[studio,pydantic-ai]'`. Then give the command the atom's location as `module:attribute`:

```bash
chatomon studio speedypack.app:atom
```

The command imports the module from the current directory, prints a link and opens it in the browser. Stop it with ++ctrl+c++. `python -m chatomon.studio` takes the same arguments.

| Option | Default | Purpose |
|--------|---------|---------|
| `--host` | `127.0.0.1` | The address to listen on |
| `--port` | `8016` | The port to listen on |
| `--no-browser` | off | Print the link without opening it |

The default port is 8016 for oxygen-16, the most common oxygen atom: the "o" of chatomon is the letter the atom replaces in the wordmark.

When a function builds the atom, for example because it reads configuration or creates clients first, point at the function instead: `chatomon studio speedypack.app:make_atom`. The studio calls it with no arguments.

From Python, `serve` in `chatomon.studio` does the same and blocks until interrupted:

```{.python test="skip"}
from chatomon.studio import serve

from speedypack.app import atom

serve(atom)
```

## Reading the Screen

### The Atom Drawer

The drawer on the left lists the atom in the order a turn runs: input orbitals, the nucleus, output orbitals. The **Atom** button in the header shows and hides it.

Every orbital has a switch. Turning one off takes effect from the next turn and writes nothing to your code. A change that would make the atom invalid, such as turning off an orbital whose annotations another one reads, is refused with the reason.

Clicking an orbital opens its card: its class, the configuration returned by `Orbital.config()` and its fingerprint. Clicking the atom or the nucleus opens the atom's card: instructions, knowledge, connector and model, retries and refusal texts. The dashes and the nucleus of the atom on stage open the same cards.

### The Atom on Stage

The atom in the middle shows the turn as it happens. The dashes on its left are the input orbitals, in order from the top; the dashes on its right are the output orbitals. A dash lights up when its orbital lets the text through, turns red when it blocks, and stays dim when it did not run.

| Pose | | When |
|------|---|------|
| rest | ![](assets/brand/mascot/animated/rest-light.svg#only-light){ width="64" }![](assets/brand/mascot/animated/rest-dark.svg#only-dark){ width="64" } | Waiting for a message |
| listen | ![](assets/brand/mascot/animated/listen-light.svg#only-light){ width="64" }![](assets/brand/mascot/animated/listen-dark.svg#only-dark){ width="64" } | An input orbital is checking the message |
| think | ![](assets/brand/mascot/animated/think-light.svg#only-light){ width="64" }![](assets/brand/mascot/animated/think-dark.svg#only-dark){ width="64" } | The nucleus is writing a reply, or writing it again |
| review | ![](assets/brand/mascot/animated/review-light.svg#only-light){ width="64" }![](assets/brand/mascot/animated/review-dark.svg#only-dark){ width="64" } | An output orbital is checking the reply |
| block | ![](assets/brand/mascot/animated/block-light.svg#only-light){ width="64" }![](assets/brand/mascot/animated/block-dark.svg#only-dark){ width="64" } | An input orbital blocked the message |
| block-output | ![](assets/brand/mascot/animated/block-output-light.svg#only-light){ width="64" }![](assets/brand/mascot/animated/block-output-dark.svg#only-dark){ width="64" } | An output orbital blocked the reply |
| reply | ![](assets/brand/mascot/animated/reply-light.svg#only-light){ width="64" }![](assets/brand/mascot/animated/reply-dark.svg#only-dark){ width="64" } | The reply was sent |
| idle | ![](assets/brand/mascot/animated/idle-light.svg#only-light){ width="64" }![](assets/brand/mascot/animated/idle-dark.svg#only-dark){ width="64" } | Nothing happened for 45 seconds |

**Slow motion**, the default, gives every step time to be seen: the checks of local orbitals take a fraction of a millisecond. **Real time** shows the events as they arrive. When the atom scrolls out of view, a small copy of it stays above the message box.

### Turns and Reports

The conversation flows under the atom. Every turn is a block with your message, the reply, a trail with one mark per orbital and a dot for the nucleus, the final action and the duration. A `↻` counts the times the nucleus wrote the reply again.

A message with a dashed red outline was blocked: it is kept out of the conversation history, where the atom's placeholder replaces it. **Re-run** sends the same message again with the orbitals as they are now, which is how you compare a turn with and without an orbital. **New conversation** clears the history.

The lens of a turn opens its report: every orbital with its action, reason, duration, guidance, annotations and fingerprint, then the attempts of the nucleus and the reply that was sent. It is the turn's `TurnReport`, described in [Observability and Audit](observability.md).

## Content in the Studio

The header shows **Content visible · local only**. The studio runs a copy of your atom with `capture_content="full"`, because showing guidance, replies and judge explanations is what it is for.

Your atom keeps its own setting. The copy also has its own turn callback and tracer, so your `on_turn`, your tracer provider and your redactor are not called: nothing from a studio session reaches your audit or tracing systems.

Model calls are real. With a real model, a turn in the studio costs what a turn costs.

## Security

The studio shows conversation text and calls models with your credentials, so it protects itself in four ways:

- It listens on `127.0.0.1`, which only your machine can reach.
- The link it prints carries a key for this run. The page keeps it in a cookie; opened without it, the page asks for the link from the terminal. Treat the link like a password.
- It answers only on its own address and accepts changes only from its own page, so other sites open in your browser cannot start turns.
- With `--host` and a network address it prints a warning: anyone who reaches that address with the link can read the conversation and use your models. Addresses that listen everywhere, such as `0.0.0.0`, are refused.

## Limits of This Version

- **PydanticAI atoms only.** An atom built with another connector is refused with a message that says so.
- **No streaming.** Replies arrive whole, because output orbitals check a reply before it is sent.
- **Read-only settings.** The cards show an orbital's configuration; change it in your code.
- **Restart to reload.** After changing the code, stop the command and start it again.
- **Rejected replies are not shown.** When an output orbital makes the nucleus write a reply again, the report shows the orbital's guidance, not the text it rejected.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `serve`, `DEFAULT_PORT` | `src/chatomon/studio/server.py` | Starts the studio for an atom |
| `chatomon studio` | `src/chatomon/_cli.py` | The command: loads `module:attribute` and starts the studio |

## Related Documentation

- [Concepts](concepts.md) — what happens to a message during a turn, and the final actions
- [Observability and Audit](observability.md) — turn reports and content capture
- [Orbitals](orbitals/index.md) — chaining, failures and annotations
- [API Reference: `chatomon.studio.server`](reference/studio.md) — `serve` and its parameters

---
**Last Updated:** October 2026
**Status:** Implemented
