# Contributing to chatomon

Thanks for your interest in chatomon. This file is the single source of truth
for **how work moves forward** — from an idea to a released version. It covers
process; how the library is built is written elsewhere.

## Where things are written down

| Source | Answers |
|---|---|
| [`README.md`](README.md) and the [documentation](docs/public/index.md) | What chatomon is and how to use it |
| `CONTRIBUTING.md` | How work moves forward — this file |
| [`CLAUDE.md`](CLAUDE.md) | How the library is built: architecture invariants, logging, docstrings, testing |
| [`.claude/skills/`](.claude/skills/) | How to carry out a procedure, step by step — written for coding agents, readable by anyone |

When something is already written in one of the others, this file links to it
instead of copying it.

## Ways to contribute

- **Report a bug or propose a feature** — open an issue with one of the
  [forms](https://github.com/zazza123/chatomon/issues/new/choose).
- **Ask a question or float an idea** — use
  [Discussions](https://github.com/zazza123/chatomon/discussions).
- **Report a vulnerability** — privately, never in public. See
  [`SECURITY.md`](SECURITY.md).
- **Send a pull request** — read the rest of this file first. For anything
  beyond a small fix, open an issue first, so we agree on the approach before
  you write the code.

## Development setup

chatomon supports Python 3.11 and later and uses
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zazza123/chatomon.git   # or your fork
cd chatomon
uv sync
```

The checks a pull request must pass — CI runs the same ones:

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
uv run zensical build --strict   # when documentation or docstrings changed
```

`pytest` includes the connector contract suite and runs the code examples in
`docs/public/` and in docstrings. Tests in `tests/integration/` talk to real
services: they are skipped unless `CHATOMON_INTEGRATION=1` and their
credentials are set, and the maintainer runs them when a change needs it.

To preview the documentation site: `uv run zensical serve`.

## Issues

### Titles

Issue titles are **English prose with no prefix**:

```
✅  LangChain connector drops the guidance when streaming
✅  ADK connector
❌  [Bug]: streaming broken
❌  FIX: guidance lost
```

`ADD:`, `FIX:` and the other codes are the vocabulary of commits and pull
requests. **An issue states what we want; a commit states what was done.**

Children of an epic prefix the epic's name with an em-dash —
`<Epic> — <description>` — and are attached to the epic as sub-issues.

### Opening one

Pick a form: **Bug**, **Feature**, or **Epic** (maintainers). Blank issues are
disabled. Do not add labels by hand: the form applies the `type:` label, and
your answers to its *Area* and *Framework* menus become labels automatically.

### Labels

Issues are classified along independent axes. Only `type:` is mandatory.

| Axis | How many | Values |
|---|---|---|
| `type:` | exactly 1 on open issues | `feat` · `fix` · `refactor` · `docs` · `chore` · `epic` |
| `area:` | 0–N | `core` · `connectors` · `orbitals` · `observability` · `docs` · `ci` |
| `framework:` | 0–N | `pydantic-ai` · `langchain` |
| `priority:` | 0–1 | `high` · `low` — absent means normal |
| `status:` | 0–1 | `needs-triage` · `needs-info` · `blocked` |

**`area:` is where the code lives, and which documentation page to update:**

| `area:` | Code | Documentation |
|---|---|---|
| `core` | `atom.py`, `knowledge.py`, `schemas.py`, `engine.py`, `errors.py`, `_sync.py` | `concepts.md` |
| `connectors` | `connectors/`, `tests/contract/` | `connectors.md` |
| `orbitals` | `orbitals/` | `orbitals.md`, `deployment/` |
| `observability` | `observability.py` | `observability.md` |
| `docs` | `docs/public/`, `zensical.toml`, docstrings | — |
| `ci` | `.github/`, `scripts/`, `tests/tooling/` | — |

**`framework:` is which agent framework is involved**, independently of
`area:` — an OpenTelemetry problem only LangChain shows is
`area:observability` + `framework:langchain`.

Outside the axes:

- `breaking` — a backwards-incompatible change; it drives the version number.
- `security` — a security-related change.
- `good first issue`, `help wanted` — for newcomers, and where help is welcome.
- `changelog:*`, `skip-changelog` — pull requests only; see
  [Versioning and releases](#versioning-and-releases).
- `dependencies`, `python`, `github-actions` — applied by Dependabot; leave
  them alone.

Rules:

1. **R1** — every open issue carries exactly one `type:`. The forms apply it;
   the pull request gate checks it on the linked issue.
2. **R2** — `area:` and `framework:` are optional and may repeat.
3. **R3** — at most one `priority:`. Do not use `priority:high` to mean "I am
   doing this next".
4. **R4** — at most one `status:`. `needs-triage` is removed once triage is
   done.
5. **R5** — a new `area:` or `framework:` value is added only when a real
   second instance exists. No speculative labels.
6. **R6** — [`.github/labels.yml`](.github/labels.yml) is the canonical list
   and is synced automatically. A label that is not in that file does not
   exist.

### How a menu becomes a label

A form menu cannot apply a label: GitHub writes the answer into the issue body.
Every menu option starts with its label — `area:connectors — Shared connector
layer, contract suite` — and a workflow reads the body and adds those labels.

**The body is the source of truth, and the workflow only ever adds.** Removing
a label by hand while the body still names it does not stick: the next edit of
the issue puts it back. To change an axis, edit the answer in the body.

### Triage

New issues arrive with `status:needs-triage`. A maintainer confirms `area:` and
`framework:`, adds `priority:`, `good first issue` or `help wanted` when they
apply, and removes `status:needs-triage`.

When more information is needed, the issue gets a question and then
`status:needs-info`. Any answer moves it back to `status:needs-triage`; after
30 days without one it is closed, and a comment with the missing details is
enough to have it reopened.

Response times are best effort: we aim to acknowledge bug reports within a
week. Feature requests get an answer, without a promised date.

## Branches

`main` is the only long-lived branch and is always releasable. Work starts from
`main` and returns to it.

With write access to this repository, branches are named

```
<type>/<issue>-<kebab-english-slug>
```

where `type` is `feat`, `fix`, `refactor`, `docs` or `chore` — the `type:`
labels minus `epic` — and the issue number is mandatory:

```
✅  feat/12-adk-connector
✅  fix/31-langchain-empty-ai-message
❌  connectors-v0.2        no type, no issue
❌  chore/bump-deps        no issue
```

Branches in forks can be named as you like.

Keep a branch current with `git rebase main`, not by merging `main` into it.

## Commits

A commit message is `CODE: Title`, followed by a body that explains what
changed and why:

```
FIX: Fail closed when a LangChain model call returns no AI message

The connector assumed the last message was an AIMessage. A tool-only turn
returned none, and the output chain was skipped instead of blocking.
```

| Code | Use for | Code | Use for |
|---|---|---|---|
| `ADD` | new features, files, capabilities | `TST` | tests only |
| `UPD` | changes to existing behaviour | `STY` | formatting only |
| `FIX` | bug fixes | `CFG` | configuration, build, CI |
| `DEL` | removals, dead code | `SEC` | security fixes |
| `DOC` | documentation only | `DEP` | dependency changes |
| `REF` | restructuring, no behaviour change | `MRG` | an unavoidable merge |

The title is imperative and at most about 72 characters. One code per commit:
split work that spans several. The full convention, with a decision tree for
choosing the code, is the
[`git-commit-convention`](.claude/skills/git-commit-convention/SKILL.md) skill.
`WIP` commits never reach `main`.

Pull requests from forks are squash-merged, so there only the pull request
title has to follow the format.

## Pull requests

### Title

```
CODE: <English prose> (#issue)
```

`CODE` is one of the commit codes above. Work in progress is opened as a
**draft pull request**, never with a `WIP` title.

### What CI checks

A gate checks every pull request. What it checks depends on where the branch
lives:

| Check | Branch in this repository | Branch in a fork |
|---|---|---|
| Branch name `<type>/<issue>-<slug>` | ✅ | — |
| Title `CODE: prose` | ✅ | ✅ |
| `(#issue)` at the end of the title | required | required for `ADD`, `UPD`, `REF`, `SEC`; optional otherwise |
| The linked issue has exactly one `type:` label | ✅ | ✅ when an issue is linked |
| Every commit subject is `CODE: Title` | ✅ | — |

So a typo fix from a fork needs no issue — `DOC: Fix a typo in the
connectors guide` is enough — while a new feature starts from an issue where
the approach was agreed. Branch names and titles must be single-line printable
text.

Linting, type checking, the test suite on every supported Python, a run
against the oldest dependency versions `pyproject.toml` allows, and the
documentation build run in CI too. All of them must be green before merging.

### The template

Fill in the pull request template. It is not paperwork: it is the record of
what was verified — the checks you ran and the definition of done below.

### Contributions written with AI assistance

They are welcome. Whoever opens the pull request answers for it exactly as if
they had typed every line: they ran the checks, they understand the code, and
they respond to review. Pull requests that were generated and not verified are
closed.

## Review and merging

- The maintainer reviews every pull request from a fork; `CODEOWNERS` requests
  the review automatically.
- Pull requests from branches of this repository may be self-merged once CI is
  green.
- **Branch in this repository → merge commit.** Its message is the pull
  request's title and description, and the individual commits stay in the
  history — which is why every one of them must follow the convention.
- **Branch in a fork → squash merge.** Its message is the pull request title.
- Rebase merging is disabled, and merged branches are deleted automatically.

When a second maintainer joins, one approving review from a code owner becomes
required on every pull request.

## Definition of done

Two obligations apply to every piece of work.

**Dead code.** If a file you are touching contains dead code, remove it — the
perimeter is the files already in your diff — in a separate `DEL:` commit, so
the cleanup can be reviewed and reverted on its own. Open a `type:refactor`
issue instead when the cleanup is larger than the change, reaches other files,
or you are not certain the code is dead (anything public is part of the API).

**Documentation.** A change in behaviour updates the affected pages of
`docs/public/` in the same pull request; the `area:` label names the page.
Docstrings are written together with the code. The rules are in
[`CLAUDE.md`](CLAUDE.md) and in the
[`writing-documentation`](.claude/skills/writing-documentation/SKILL.md) skill.

## Adding a connector

Support for a new agent framework — Google ADK, Microsoft Agent Framework and
others — is the most likely large contribution, so its path is written down:

1. **Request.** A `type:feat` issue with `area:connectors`, titled
   `<Framework> connector`. There is no `framework:` label for it yet (R5).
   Further requests for the same framework are closed as duplicates of this
   one.
2. **Decision.** When the work is scheduled the issue becomes an epic, and a
   `framework:<name>` label is added to `.github/labels.yml` together with an
   option in the *Framework* menu of every issue form. A test fails if the two
   drift apart.
3. **Done when:**
   - `src/chatomon/connectors/<framework>.py` imports the framework lazily and
     raises an error naming the extra when it is missing;
   - `pyproject.toml` has an extra for it, with the framework's major version
     pinned;
   - a harness in `tests/contract/` passes the whole contract suite, using the
     framework's fake models and no network;
   - `docs/public/connectors.md` has a section for it.

What a connector may and may not do is in [`CLAUDE.md`](CLAUDE.md).

## Versioning and releases

chatomon follows [Semantic Versioning](https://semver.org/). While the version
is `0.x`:

- a backwards-incompatible change (label `breaking`) bumps the **minor**
  version: `0.1.x` → `0.2.0`;
- everything else bumps the **patch** version.

The version comes from the git tag; there is no version number to edit.

Release notes are drafted continuously. Every merged pull request becomes a
line, grouped by a `changelog:` label set automatically from the code in its
title — `ADD`/`UPD` → Features, `FIX` → Fixes, `SEC` → Security, `DOC` →
Documentation, anything else → Maintenance. Pull requests labelled `breaking`
are listed first. Add `skip-changelog` to leave a pull request out.

Releasing (maintainers):

1. Check that `main` is green.
2. Review the draft release: reword lines for users, check that `breaking` is
   where it belongs, check the proposed version.
3. **Publish** the release. Its tag — `v0.2.0`, or `v0.2.0rc1` marked as a
   pre-release — starts the pipeline: it verifies that the tag is on `main`,
   runs the full CI, builds, publishes to PyPI and, for final releases,
   deploys the documentation. Pushing a tag on its own does nothing.
4. Check that the pipeline is green and the PyPI page is right.

Only the latest release receives fixes.

## Code of conduct

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By contributing, you agree to uphold it.
