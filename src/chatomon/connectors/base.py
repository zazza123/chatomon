"""The connector extension point: how an atom becomes a native agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from chatomon.atom import Atom
    from chatomon.engine import Engine

__all__ = ["AgentT", "Connector", "ModelT"]

AgentT = TypeVar("AgentT")
"""Type of the native agent a connector builds."""

ModelT = TypeVar("ModelT", bound=BaseModel)
"""Type of a structured model answer."""


class Connector(ABC, Generic[AgentT]):
    """Builds an atom into a native agent of one framework.

    A connector installs chatomon's hooks as the outermost layer of the native
    agent and translates the engine's decisions into the framework's mechanisms.
    To support a new framework, subclass it in a new module and make it pass the
    shared contract suite.
    """

    @abstractmethod
    def build(self, atom: Atom[AgentT], engine: Engine) -> AgentT:
        """Return the native agent with chatomon's hooks installed.

        Args:
            atom: The validated atom.
            engine: The engine that runs the atom's orbitals.

        Returns:
            The framework's agent object.
        """

    @abstractmethod
    async def generate_structured(
        self, model: str, *, system: str, prompt: str, schema: type[ModelT]
    ) -> ModelT:
        """Make one model call whose answer must match `schema`.

        Used by orbitals that call a model, such as the judge.

        Args:
            model: Model identifier in the framework's format.
            system: System prompt.
            prompt: User prompt.
            schema: Pydantic model the answer must validate against.

        Returns:
            The validated answer.
        """
