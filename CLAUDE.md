# chatomon

chatomon ("chat" + "atom") is a Python library to declare a chat agent — an **Atom** (instructions, knowledge, connector) — wrapped by ordered chains of **Orbitals** (input/output guardrails), and build it into a **native** agent of existing frameworks (PydanticAI, LangChain/LangGraph, Microsoft Agent Framework, Google ADK). It is not a chatbot framework: it never runs the agent itself, except in the studio (`chatomon.studio`), a local development tool.

Architecture invariants are stated in this file. The maintainer keeps the design spec outside the repository; any rule a contributor must follow lives here or in `CONTRIBUTING.md`.

## Contributing

`CONTRIBUTING.md` is the source of truth for process — issues, labels, branches, commits, pull requests, merging and releases. This file covers how the library is built. Opening an issue, creating a branch, opening or merging a pull request, triaging and releasing follow the `contributing-workflow` skill.

## Language

- Code, docstrings, library documentation, commit messages, issues and pull requests: English.

## Architecture rules

- The engine (`engine.py`) holds all orbital-chain semantics and is pure Python — no framework imports. Connectors only translate `InputDecision`/`OutputDecision` into native hooks.
- `schemas.py` imports nothing from chatomon.
- Framework and `orbitals` imports live only in `connectors/<framework>.py` or `connectors/<framework>/`, `studio/_runners/<framework>.py` and `orbitals/principled.py`, imported lazily; a missing extra raises an error naming the extra to install. `spectral_bridge` imports live only in `studio/_spectral/_relay.py`, imported lazily. Web (`starlette`, `uvicorn`) and OpenTelemetry SDK imports live only in `studio/`; everywhere else chatomon depends only on `opentelemetry-api`.
- `Orbital` and `Connector` are the only abstract extension points. Don't add abstractions without a second concrete use.
- Never derive an orbital's definition/config from the Atom. Sharing (e.g. the same `Knowledge` given to a judge) is always explicit.
- Defaults are fail-closed (`on_error="block"`) and content-free (`capture_content="none"`).

## Logging and telemetry

- Loggers are `logging.getLogger(__name__)` under `chatomon.*`; only a `NullHandler` on `chatomon`. Never configure handlers.
- The one handler chatomon adds to another library's logger is the one `studio/_spectral/_relay.py` puts on `spectral_bridge.client` while a connection is open: it reads why a connection was refused, emits nothing, changes no level or propagation, and is removed when the connection closes. For the same time it puts one filter on `httpx`, which that client calls the adapter with and which logs the address of every call: the filter lets every record through and hides the adapter's key in the ones that show it.
- Never log, trace or report message content, guidance, annotations or judge explanations unless `capture_content` allows it.
- Depend only on `opentelemetry-api`; only the studio also uses the SDK, to collect the spans it shows. Open and close spans within the same hook — never across hooks or across `yield`.

## Docstrings (API reference is generated from them with Zensical + mkdocstrings)

- Google style on every public module, class, function and method (ruff `D`, `convention = "google"`).
- Types go in signatures, not repeated in `Args:`.
- Document `Raises:` for exceptions the caller should handle.
- Public entry points get an `Examples:` section with fenced code blocks; examples must run under `pytest-examples`. An example that cannot run in tests (network, real models) is fenced as ```` ```{.python test="skip"} ```` — the brace form is the only one that both pytest-examples and the docs renderer (superfences + attr_list) understand.
- Cross-reference with mkdocstrings syntax: `[Atom][chatomon.Atom]`.
- Write docstrings together with the code, not afterwards.

## Functional and technical documentation

- Hand-written usage docs (concepts, orbitals, connectors, observability, end-to-end examples) live in `docs/public/` (the Zensical `docs_dir`) and are always written or updated with the `writing-documentation` skill.
- They explain how to use chatomon and link to the generated API reference for details; don't duplicate parameter tables that docstrings already cover.
- Exception to the skill's no-code rule (approved by the user): usage pages may include short code examples when they genuinely help. They must run under `pytest-examples` (collected from `docs/public/`), or be fenced as ```` ```{.python test="skip"} ```` when they need real services.
- Any change that alters behavior updates the affected pages in the same milestone.

## Testing

- TDD: write the failing test first.
- Every engine rule has a unit test with fake orbitals.
- Every connector must pass the shared contract suite in `tests/contract/`, using the framework's fake models — no network.
- Real-service tests live in `tests/integration/`, are marked `integration` and are skipped unless their environment variables are set.
- A fake service that listens on loopback (`tests/fake_service.py`) is not a network: a test that needs a real HTTP call uses it.

## Studio (`chatomon.studio`)

- **Architecture.** The studio talks to the core only through the public API and the spans the engine already opens; it never changes the engine to serve the UI. It never mutates the developer's atom: it builds a copy with its own `on_turn`, `capture_content="full"` and `tracer_provider`. A change to an orbital builds a new instance through its constructor: orbitals are never changed in place, and a change is kept only if the atom copy builds. It loads the developer's code again by importing anew the modules under the directory it was started from, never chatomon's own or installed ones; a reload replaces the atom, its chains and its agent together or changes nothing, and what the studio changed is carried over it with the preset rules of `studio/_presets.py`. The server computes, the browser renders: orbital semantics, code generation, presets and tests are Python. The state and event JSON shapes are a contract tested in `tests/studio/`. `serve()` configures no logging; what the studio prints goes to stderr: its URL, the notes about the atom, the warning for a network address, what a failed reload raised and a line when it stops.
- **Extending.** A new connector gets a studio runner in `studio/_runners/<framework>.py`, named like the connector's module: that name is how the studio finds it, so there is no list of runners to keep. A runner is a module with the five functions of the `Runner` protocol in `studio/_runners/__init__.py`: it runs one turn with the conversation it is given and leaves that conversation as it was when the model or a tool fails; the conversation it returns can be given to an agent built from another instance of the same connector, which is what a reload from code does; it runs one turn on a conversation given as plain texts, which is how a request from outside the page reaches the atom; it names the connector and its model; it says what the developer should know when the studio starts — a turn that can fail, or a place the studio writes to; and it supplies its framework's part of a saved test. In a saved test a stand-in for the nucleus calls no tool and is put in place through a public attribute: a runner supplies the one of an offline test, and the one that repeats the replies recorded in the studio, which fails the test when it is asked for more than were recorded. A runner has an entry in `tests/studio/frameworks.py` and passes, with the framework's fake models, the scenarios of `tests/studio/test_events.py`, `test_runners.py` and `test_testgen.py`, and the browser smoke test of `tests/studio/test_e2e.py`. A new orbital needs no studio code: the studio shows any orbital through `config()` and `fingerprint()`, finds the orbital classes by walking the modules of `chatomon.orbitals` (a module behind an extra raises `MissingExtraError`, which is how the studio names the extra), and reads what it can edit from the constructor (`studio/_kinds.py`), never from a list of its own. An orbital exported by `chatomon.orbitals` or `chatomon.orbitals.principled` keeps to this convention: a keyword parameter annotated with a plain type (`int`, `float`, `str`, `bool`, a `Literal` of strings, `Sequence[str]`, each optionally `| None`) is a setting, a parameter that accepts a value type is a block, any other parameter is an object from the developer's code, and every parameter is kept in a public attribute of the same name. A value type is a dataclass or a pydantic model made of plain fields, lists of texts and lists of value types whose own fields are plain or lists of texts: the studio reads its fields from its signature (`studio/_values.py`) and edits the block whole. Every orbital class sets `calls_model`, which the studio reads to tell a saved test that runs offline from one that needs real services; it never keeps a list of its own. A secret is never a setting or a field: a parameter or a field whose name ends in `_env` holds the name of an environment variable, and the orbital reads its value when it is built; any other secret reaches the orbital inside an object. Constructors are cheap and have no side effects, because the studio calls them when it starts and on every change. Exceptions are listed in `studio/_kinds.py`; `tests/studio/test_kinds.py` rebuilds every exported orbital from what the studio reads and compares fingerprints. An orbital can be added from the studio when its required parameters are settings or blocks; orbitals that need other objects (functions, clients configured by hand) are added in code. Secrets never reach the browser, presets or generated code; only environment variable names do.
- **Add-ons.** An add-on of the studio lives in its own folders (`studio/_<name>/`, `static/js/<name>/`, `tests/studio/<name>/`) and is loaded only when its extra is installed: without it the page, the state and the routes are the same. The session knows nothing of an add-on; an add-on reads the session's public attributes and builds on `revision`, `build_copy()` and `settled()`. `spectral` is the first: it connects the atom to Spectral through `spectral-bridge`, runs the requests on a copy of the atom built with `capture_content="none"`, and keeps counts, never text.
- **Frontend.** No build step: static HTML, CSS and ES modules in `studio/static/`; no Node, framework, bundler or CDN, and no external URL. Fonts ship with their OFL licences. The brand tokens equal those of `docs/public/stylesheets/chatomon.css`, and every mascot pose or animation the studio uses is also saved, light and dark, in `docs/public/assets/brand/mascot/` (animations in `animated/`); tests check both. Both themes, keyboard access, visible focus, `aria-live` for turn phases, `prefers-reduced-motion`, phone width.
- **Local tools.** Any chatomon tool that opens a server listens on `127.0.0.1` by default (another address only explicitly, with a printed warning; never a bind-all address), checks `Host` on every request and `Origin` on every state-changing one, requires a per-run token kept in an `HttpOnly`, `SameSite=Strict` cookie and never logged, sends a `Content-Security-Policy` that allows only its own files and forbids framing, and says on screen when content is captured. A tool that writes files writes only under the directory it was started from, and never over a file it did not write; what it writes is computed by the server, never sent by the page, and when that holds text a model wrote the page says so before it writes. A tool that calls a service because its page asked calls only what a turn would call, with text the server wrote, and reports the outcome in its own words, never with the text of an exception. When it listens on another address than loopback, its page cannot set where a call goes or which variable is sent: that comes from the developer's code or from what the command was started with. A server a tool opens for another program on the same machine, not for a browser, listens on `127.0.0.1` on a port the system chooses and only while it is needed; it takes a per-run key in its path in place of the cookie, never logged, checks `Host`, and refuses every request that carries `Origin`. A tool that connects to a service outside the machine because its page asked does so only when it listens on loopback, at the address its command's environment names, with the key of the variable the page names; it prints a line when the connection opens and when it closes.
- **Testing.** Studio tests live in `tests/studio/`. Browser checks are Playwright smoke tests marked `e2e`, skipped unless `CHATOMON_E2E=1`, run in their own CI job.

## Workflow

- Every commit follows the `git-commit-convention` skill; commit at the end of each step with green tests.
- Issues, branches, pull requests, merging, triage and releases follow the `contributing-workflow` skill and `CONTRIBUTING.md`.

## Skills (in `.claude/skills/`)

- **git-commit-convention** — every commit.
- **contributing-workflow** — opening an issue, creating a branch, opening or merging a pull request, triaging, releasing.
- **writing-documentation** — every change to `docs/public/`.
