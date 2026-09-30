"""Build an atom into a native Microsoft Agent Framework agent.

Install with ``pip install 'chatomon[maf]'`` or ``chatomon[maf-foundry]``.
"""

from chatomon.connectors.maf._core import MAFConnector
from chatomon.connectors.maf._foundry import MAFFoundryConnector

__all__ = ["MAFConnector", "MAFFoundryConnector"]
