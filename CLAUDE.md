# chatomon

chatomon ("chat" + "atom") is a Python library to declare a chat agent — an **Atom** (instructions, knowledge, connector) — wrapped by ordered chains of **Orbitals** (input/output guardrails), and build it into a **native** agent of existing frameworks (PydanticAI, LangChain/LangGraph; Google ADK next). It is not a chatbot framework: it never runs the agent itself.

The design source of truth is `docs/superpowers/specs/2026-09-27-chatomon-design.md`. Read it before changing architecture; update it when a decision changes.

## Language

- Talk with the user in Italian.
- Code, docstrings, library documentation, commit messages: English.

## Architecture rules

- The engine (`engine.py`) holds all orbital-chain semantics and is pure Python — no framework imports. Connectors only translate `InputDecision`/`OutputDecision` into native hooks.
- `schemas.py` imports nothing from chatomon.
- Framework and `orbitals` imports live only in `connectors/<framework>.py` and `orbitals/principled.py`, imported lazily; a missing extra raises an error naming the extra to install.
- `Orbital` and `Connector` are the only abstract extension points. Don't add abstractions without a second concrete use.
- Never derive an orbital's definition/config from the Atom. Sharing (e.g. the same `Knowledge` given to a judge) is always explicit.
- Defaults are fail-closed (`on_error="block"`) and content-free (`capture_content="none"`).

## Logging and telemetry

- Loggers are `logging.getLogger(__name__)` under `chatomon.*`; only a `NullHandler` on `chatomon`. Never configure handlers.
- Never log, trace or report message content, guidance, annotations or judge explanations unless `capture_content` allows it.
- Depend only on `opentelemetry-api`. Open and close spans within the same hook — never across hooks or across `yield`.

## Docstrings (API reference is generated from them with Zensical + mkdocstrings)

- Google style on every public module, class, function and method (ruff `D`, `convention = "google"`).
- Types go in signatures, not repeated in `Args:`.
- Document `Raises:` for exceptions the caller should handle.
- Public entry points get an `Examples:` section with fenced code blocks; examples must run under `pytest-examples`. An example that cannot run in tests (network, real models) is fenced as ```` ```{.python test="skip"} ```` — the brace form is the only one that both pytest-examples and the docs renderer (superfences + attr_list) understand.
- Cross-reference with mkdocstrings syntax: `[Atom][chatomon.Atom]`.
- Write docstrings together with the code, not afterwards.

## Functional and technical documentation

- Hand-written usage docs (concepts, orbitals, connectors, observability, end-to-end examples) live in `docs/public/` (the Zensical `docs_dir`; `docs/superpowers/` is internal and never published) and are always written or updated with the `writing-documentation` skill.
- They explain how to use chatomon and link to the generated API reference for details; don't duplicate parameter tables that docstrings already cover.
- Any change that alters behavior updates the affected pages in the same milestone.

## Testing

- TDD: write the failing test first.
- Every engine rule has a unit test with fake orbitals.
- Every connector must pass the shared contract suite in `tests/contract/`, using the framework's fake models — no network.
- Real-service tests live in `tests/integration/`, are marked `integration` and are skipped unless their environment variables are set.

## Workflow

- Commit at the end of each milestone (spec §15) with green tests, following the `git-commit-convention` skill.
