"""Save as test: a turn, or the conversation up to it, as a pytest file.

The test sends the messages to the atom from code again and checks how each turn
ended, which input orbitals stopped it and, when the reply is fixed in code, its
text. It runs the real atom: nothing the nucleus wrote is stored, so what the
output chain decides is checked only when nothing stopped a reply. Only messages,
orbital names, actions and replies fixed in code are written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatomon.studio._session import TurnRecord

CANNOT_SAVE = "A turn that failed cannot be saved as a test."


def refusal(turn: TurnRecord) -> str | None:
    """Say why a turn cannot be saved as a test.

    Args:
        turn: A finished turn.

    Returns:
        The sentence, or `None` when the turn can be saved.
    """
    report = turn.report
    if turn.error is not None or report is None or report.final_action == "aborted":
        return CANNOT_SAVE
    return None
