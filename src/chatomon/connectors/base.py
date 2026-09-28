"""The connector extension point: how an atom becomes a native agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from chatomon.atom import Atom
    from chatomon.engine import Engine

__all__ = ["AgentT", "Connector", "ModelT", "render_guidance"]

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

    Examples:
        A toy connector whose "native agent" is a dictionary:

        ```python
        from typing import Any

        from chatomon import Atom
        from chatomon.connectors import Connector


        class DictConnector(Connector[dict[str, Any]]):
            def build(self, atom, engine):
                return {"system_prompt": atom.system_prompt, "engine": engine}

            async def generate_structured(self, model, *, system, prompt, schema):
                raise NotImplementedError


        agent = Atom(name="demo", instructions="Be brief.", connector=DictConnector()).build()
        assert agent["system_prompt"] == "Be brief."
        ```
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


def render_guidance(guidance: Sequence[str]) -> str:
    r"""Render guidance as the section connectors add to one model call.

    Args:
        guidance: Guidance items, in order.

    Returns:
        A `<guidance>` block with one bullet per item.

    Examples:
        ```python
        from chatomon.connectors import render_guidance

        assert render_guidance(["Be brief."]) == "<guidance>\n- Be brief.\n</guidance>"
        ```
    """
    items = "\n".join(f"- {item}" for item in guidance)
    return f"<guidance>\n{items}\n</guidance>"
