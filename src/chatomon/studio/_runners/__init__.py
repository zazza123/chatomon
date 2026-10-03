"""Run studio turns on the native agent of a supported connector.

One module per framework. The studio supports PydanticAI only for now; a
shared runner protocol comes with the second framework.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from chatomon.connectors.base import Connector
from chatomon.errors import ChatomonConfigError


def check_supported(connector: Connector[Any]) -> None:
    """Raise unless the studio can run turns on this connector's agents.

    Args:
        connector: The atom's connector.

    Raises:
        ChatomonConfigError: For a connector the studio does not support yet.
    """
    if importlib.util.find_spec("pydantic_ai") is not None:
        from chatomon.connectors.pydantic_ai import PydanticAIConnector

        if isinstance(connector, PydanticAIConnector):
            return
    raise ChatomonConfigError(
        f"the studio supports PydanticAI atoms for now; this atom uses {type(connector).__name__}"
    )
