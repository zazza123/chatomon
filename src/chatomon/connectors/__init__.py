"""Connectors: build an atom into a native agent of a framework.

Framework connectors live in their own modules (`chatomon.connectors.pydantic_ai`,
`chatomon.connectors.langchain`, `chatomon.connectors.maf`,
`chatomon.connectors.adk`) and need the matching extra.
"""

from chatomon.connectors.base import Connector, render_guidance

__all__ = ["Connector", "render_guidance"]
