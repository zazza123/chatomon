"""Studio turns on atoms built with `PydanticAIConnector`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ModelMessage

from chatomon.connectors.base import Connector


@dataclass(frozen=True)
class RunOutcome:
    """The reply of a turn and the conversation history after it."""

    reply: str
    history: list[ModelMessage]


async def run_turn(agent: Any, message: str, history: Sequence[ModelMessage]) -> RunOutcome:
    """Run one turn without streaming, as the connector requires with output orbitals.

    Args:
        agent: The native `pydantic_ai.Agent` built from the studio's copy of the atom.
        message: The user message.
        history: The conversation so far, as PydanticAI messages.

    Returns:
        The reply and the full history, blocked messages already redacted.
    """
    result = await agent.run(message, message_history=list(history))
    return RunOutcome(reply=result.output, history=result.all_messages())


def describe(connector: Connector[Any]) -> dict[str, str]:
    """Name the connector and its model for the atom card.

    Args:
        connector: A `PydanticAIConnector`.

    Returns:
        `{"connector": ..., "model": ...}`.
    """
    model = getattr(connector, "model", None)
    if isinstance(model, str):
        name = model
    elif model is None:
        name = "unknown"
    else:
        name = str(getattr(model, "model_name", type(model).__name__))
    return {"connector": type(connector).__name__, "model": name}
