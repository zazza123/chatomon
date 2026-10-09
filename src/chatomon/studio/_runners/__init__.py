"""Run studio turns on the native agent of the atom's connector.

One module per framework, named like the connector's module: a connector
defined in `chatomon.connectors.<framework>` is run by
`chatomon.studio._runners.<framework>`. Each module has the five functions of
[Runner][chatomon.studio._runners.Runner].
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from chatomon.connectors.base import Connector
from chatomon.errors import ChatomonConfigError, ChatomonError

if TYPE_CHECKING:
    from chatomon.studio._testgen import Framework

_CONNECTORS = "chatomon.connectors."

History = Sequence[tuple[Literal["user", "assistant"], str]]
"""A conversation as plain texts, oldest first: who spoke, and what was said."""


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

        The conversation that is given is never changed: a turn that fails, or
        that the session does not keep, leaves it as it was.

        Returns:
            The reply and the conversation after it, blocked messages already
            replaced by the placeholder.

        Raises:
            InterruptedTurnError: If the agent stopped to wait for an answer.
            Exception: Whatever the model or a tool raised.
        """
        ...

    async def run_from_history(self, agent: Any, history: History, message: str) -> str:
        """Run one turn, without streaming, on a conversation given as plain texts.

        This is how a turn that does not come from the page reaches the atom: there
        is no conversation of the framework's to go on from, only what was said.

        Args:
            agent: A native agent built from the atom.
            history: What was said before the message, oldest first. It is not changed.
            message: The user message.

        Texts of the same speaker that follow one another are given in order. A
        framework that reads what the assistant said in a row as one message gets
        it joined by line breaks, so that the words of one text do not run into
        the next.

        Returns:
            The reply; for a message an orbital stops, the reply the user gets. An
            empty text when the turn ended with no reply: nothing else says so here.

        Raises:
            InterruptedTurnError: If the agent stopped to wait for an answer.
            Exception: Whatever the model or a tool raised.
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
        module = cls.__module__ or ""  # a class made at run time can have none
        if not module.startswith(_CONNECTORS):
            continue
        framework = module[len(_CONNECTORS) :].partition(".")[0]
        name = f"{__name__}.{framework}"
        if importlib.util.find_spec(name) is not None:
            return cast("Runner", importlib.import_module(name))
    raise ChatomonConfigError(
        f"the studio has no runner for {type(connector).__name__}: "
        "it opens atoms built with chatomon's own connectors"
    )
