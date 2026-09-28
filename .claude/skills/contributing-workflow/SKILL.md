---
name: contributing-workflow
description: Use when opening a GitHub issue, creating a branch, opening or merging a pull request, triaging an issue or cutting a release in this repository - enforces the conventions in CONTRIBUTING.md
---

# Contributing Workflow

## Overview

`CONTRIBUTING.md` is the source of truth for the label axes, the title formats,
the branch format and the merge rules. **Read the section you need before
acting.** This skill holds the procedures only; it deliberately does not
restate the vocabularies, because two copies of a vocabulary always diverge.

**Core principle:** an issue states what we want; a commit states what was
done; a pull request states what was verified.

## The Iron Law

```
NO BRANCH WITHOUT AN ISSUE. NO PULL REQUEST WITHOUT ITS GATES RUN.
```

CI enforces the conventions (`.github/workflows/pr-conventions.yml`). A
violation is a failed check, not a review comment.

## Opening an issue

1. Read "Issues" in `CONTRIBUTING.md`.
2. Issue forms can only be filled in a browser. From the CLI, pass every label
   explicitly — the label workflow reads the forms' `### Area` and
   `### Framework` headings, which a CLI body does not have:

   ```bash
   gh issue create --title "<English prose, no prefix>" --body-file <path> \
     --label "type:feat" --label "area:connectors" --label "framework:langchain"
   ```

   An issue a maintainer opens needs no `status:needs-triage`.
3. Child of an epic: title `<Epic> — <description>`, then attach it:

   ```bash
   child_id=$(gh api repos/zazza123/chatomon/issues/<child> --jq .id)
   gh api repos/zazza123/chatomon/issues/<epic>/sub_issues -F sub_issue_id="$child_id"
   ```

**Check before you finish:** no `ADD:` or `[Bug]:` prefix in the title, and
exactly one `type:` label.

## Creating a branch

```bash
git switch main && git pull
git switch -c <type>/<issue>-<kebab-english-slug>
```

`<type>` is the issue's `type:` label minus `epic`.

## Committing

Follow the `git-commit-convention` skill for every commit. Dead code found in a
file you touch goes in its own `DEL:` commit. Update the branch with
`git rebase main`; never merge `main` into it.

## Before opening the pull request

Run every gate and read its output — never tick a gate you did not run:

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
uv run zensical build --strict   # docs/public/, zensical.toml or a docstring changed
```

If the change alters behaviour, update `docs/public/` with the
`writing-documentation` skill.

## Opening the pull request

```bash
git push -u origin HEAD
gh pr create --title "<CODE>: <English prose> (#<issue>)" --body-file <path>
```

- The body is `.github/PULL_REQUEST_TEMPLATE.md` filled in: `Closes #<issue>`,
  what changes, the gates actually run, the definition of done.
- Not ready → `gh pr create --draft`. Never a `WIP` title.
- Backwards-incompatible → `gh pr edit <n> --add-label breaking`.
- Wait for the checks: `gh pr checks <n> --watch`.

## Merging

The method depends on where the branch lives ("Review and merging" in
`CONTRIBUTING.md`):

| Branch | Command |
|---|---|
| this repository | `gh pr merge <n> --merge --delete-branch` |
| a fork | `gh pr merge <n> --squash` |
| Dependabot | `gh pr merge <n> --squash --delete-branch` |

Never merge with a red check. Merge a fork's pull request only after reviewing
it.

## Triaging an issue

1. Read "Labels" and "Triage" in `CONTRIBUTING.md`.
2. Confirm or correct `area:` and `framework:`. When the issue came from a form,
   edit the answer in its body — which adds the new label — **and** remove the
   old label with `gh issue edit <n> --remove-label <old>`: the workflow only
   adds, and re-adds any label the body still names, so both steps are needed.
   Otherwise use `gh issue edit <n> --add-label/--remove-label`.
3. Add `priority:`, `good first issue` or `help wanted` when they apply.
4. Remove `status:needs-triage`.
5. Missing information: comment with the question and finish every other
   label change **first**; add `status:needs-info` **last**, swapping it for
   `status:needs-triage` in one command
   (`gh issue edit <n> --remove-label status:needs-triage --add-label status:needs-info`).
   Any update after the label — a comment, an edit, another label — counts as
   the answer and removes it.

## Releasing

Follow "Versioning and releases" in `CONTRIBUTING.md`: review the draft
release, then publish it. Publishing is the act that ships.

## Red Flags — STOP

- Creating a branch with no issue number
- A pull request title without `(#issue)` from a branch of this repository
- Ticking a gate in the template that was not run
- Squash-merging a branch of this repository, or merge-committing a fork
- Adding a label that is not in `.github/labels.yml`
- Copying a vocabulary from `CONTRIBUTING.md` into another file
