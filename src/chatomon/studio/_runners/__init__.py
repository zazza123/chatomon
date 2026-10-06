"""Run studio turns on the native agent of the atom's connector.

One module per framework, named like the connector's module: a connector
defined in `chatomon.connectors.<framework>` is run by
`chatomon.studio._runners.<framework>`. Each module has the four functions of
[Runner][chatomon.studio._runners.Runner].
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from chatomon.connectors.base import Connector
from chatomon.errors import ChatomonConfigError, ChatomonError

if TYPE_CHECKING:
    from chatomon.studio._testgen import Framework

_CONNECTORS = "chatomon.connectors."


@dataclass(frozen=True)
class RunOutcome:
    """The reply of a turn and what the runner needs to run the next one."""

    reply: str
    conversation: Any


class InterruptedTurnError(ChatomonError):
    """The agent stopped to wait for an answer the studio cannot give."""


class Runner(Protocol):
    """What the studio needs from an agent framework; a module with these functions is one."""

    async def run_turn(self, agent: Any, message: str, conversation: Any) -> RunOutcome:
        """Run one turn without streaming.

        Args:
            agent: The native agent built from the studio's copy of the atom.
            message: The user message.
            conversation: `None` in the first turn of a conversation, then what
                the turn before returned.

        Returns:
            The reply and the conversation after it, blocked messages already
            replaced by the placeholder.

        Raises:
            InterruptedTurnError: If the agent stopped to wait for an answer.
            Exception: Whatever the model or a tool raised; the conversation that
                was given is then left as it was.
        """
        ...

    def describe(self, connector: Connector[Any], agent: Any) -> dict[str, str]:
        """Name the connector and its model for the atom card.

        Args:
            connector: The atom's connector.
            agent: The native agent, for what the framework resolves when it builds.

        Returns:
            `{"connector": ..., "model": ...}`.
        """
        ...

    def notes(self, connector: Connector[Any]) -> list[str]:
        """Say what the developer should know before trying this atom in the studio.

        Args:
            connector: The atom's connector.

        Returns:
            One sentence per turn that can fail and per place the studio writes to.
        """
        ...

    def saved_test(self, connector: Connector[Any]) -> Framework:
        """Return the framework's part of a saved test.

        Args:
            connector: The atom's connector.

        Returns:
            The lines of a saved test that depend on the framework.
        """
        ...


def runner_for(connector: Connector[Any]) -> Runner:
    """Return the runner of the framework a connector builds for.

    The connector's classes are searched for the first one chatomon defines; a
    subclass written by the developer is run like the connector it extends.

    Args:
        connector: The atom's connector.

    Returns:
        The module `chatomon.studio._runners.<framework>`.

    Raises:
        ChatomonConfigError: For a connector the studio has no runner for.
    """
    for cls in type(connector).__mro__:
        if not cls.__module__.startswith(_CONNECTORS):
            continue
        framework = cls.__module__[len(_CONNECTORS) :].partition(".")[0]
        name = f"{__name__}.{framework}"
        if importlib.util.find_spec(name) is not None:
            return cast("Runner", importlib.import_module(name))
    raise ChatomonConfigError(
        f"the studio has no runner for {type(connector).__name__}: "
        "it opens atoms built with chatomon's own connectors"
    )
