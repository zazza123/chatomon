Closes #

## What changes



## Gates run

Tick only what you actually ran. See CONTRIBUTING.md, "Development setup".

- [ ] `uv run ruff check` · `uv run ruff format --check`
- [ ] `uv run pyright`
- [ ] `uv run pytest`
- [ ] `uv run zensical build --strict` — docs or docstrings changed
- [ ] integration tests (`CHATOMON_INTEGRATION=1`) — a Principled orbital changed and you have access

## Definition of done

- [ ] dead code found in the files I touched was removed in a separate `DEL:` commit — or none found
- [ ] `docs/public/` updated per the `writing-documentation` skill — or not required, because:
- [ ] this is a breaking change (the maintainer adds the `breaking` label)
