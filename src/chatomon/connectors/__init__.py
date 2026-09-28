"""Connectors: build an atom into a native agent of a framework.

Framework connectors live in their own modules (`chatomon.connectors.pydantic_ai`,
`chatomon.connectors.langchain`) and need the matching extra.
"""

from chatomon.connectors.base import Connector

__all__ = ["Connector"]
