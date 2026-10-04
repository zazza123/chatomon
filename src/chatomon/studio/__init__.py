"""A local web studio: chat with an atom and watch its orbitals decide.

Install it with the extra of the atom's framework, for example
`pip install 'chatomon[studio,pydantic-ai]'`, then run `chatomon studio module:atom`.
"""

from __future__ import annotations

import importlib.util

from chatomon.errors import MissingExtraError

for _module, _package in (
    ("starlette", "starlette"),
    ("uvicorn", "uvicorn"),
    ("opentelemetry.sdk", "opentelemetry-sdk"),
):
    if importlib.util.find_spec(_module) is None:
        raise MissingExtraError("studio", _package)

from chatomon.studio.server import DEFAULT_PORT, serve  # noqa: E402 - after the extra check

__all__ = ["DEFAULT_PORT", "serve"]
