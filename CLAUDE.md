# chatomon

chatomon ("chat" + "atom") is a Python library to declare a chat agent — an **Atom** (instructions, knowledge, connector) — wrapped by ordered chains of **Orbitals** (input/output guardrails), and build it into a **native** agent of existing frameworks (PydanticAI, LangChain/LangGraph, Microsoft Agent Framework; Google ADK next). It is not a chatbot framework: it never runs the agent itself.

Architecture invariants are stated in this file. The maintainer keeps the design spec outside the repository; any rule a contributor must follow lives here or in `CONTRIBUTING.md`.

## Contributing

`CONTRIBUTING.md` is the source of truth for process — issues, labels, branches, commits, pull requests, merging and releases. This file covers how the library is built. Opening an issue, creating a branch, opening or merging a pull request, triaging and releasing follow the `contributing-workflow` skill.

## Language

- Code, docstrings, library documentation, commit messages, issues and pull requests: English.

## Architecture rules

- The engine (`engine.py`) holds all orbital-chain semantics and is pure Python — no framework imports. Connectors only translate `InputDecision`/`OutputDecision` into native hooks.
- `schemas.py` imports nothing from chatomon.
- Framework and `orbitals` imports live only in `connectors/<framework>.py` or `connectors/<framework>/` and `orbitals/principled.py`, imported lazily; a missing extra raises an error naming the extra to install.
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

- Hand-written usage docs (concepts, orbitals, connectors, observability, end-to-end examples) live in `docs/public/` (the Zensical `docs_dir`) and are always written or updated with the `writing-documentation` skill.
- They explain how to use chatomon and link to the generated API reference for details; don't duplicate parameter tables that docstrings already cover.
- Exception to the skill's no-code rule (approved by the user): usage pages may include short code examples when they genuinely help. They must run under `pytest-examples` (collected from `docs/public/`), or be fenced as ```` ```{.python test="skip"} ```` when they need real services.
- Any change that alters behavior updates the affected pages in the same milestone.

## Testing

- TDD: write the failing test first.
- Every engine rule has a unit test with fake orbitals.
- Every connector must pass the shared contract suite in `tests/contract/`, using the framework's fake models — no network.
- Real-service tests live in `tests/integration/`, are marked `integration` and are skipped unless their environment variables are set.

## Workflow

- Every commit follows the `git-commit-convention` skill; commit at the end of each step with green tests.
- Issues, branches, pull requests, merging, triage and releases follow the `contributing-workflow` skill and `CONTRIBUTING.md`.

## Skills (in `.claude/skills/`)

- **git-commit-convention** — every commit.
- **contributing-workflow** — opening an issue, creating a branch, opening or merging a pull request, triaging, releasing.
- **writing-documentation** — every change to `docs/public/`.
