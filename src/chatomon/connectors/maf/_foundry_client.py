"""The chat client of a Foundry agent built by chatomon.

Its base class is in the Foundry package, so this module is imported only when a
Foundry agent is built: `chatomon[maf]` alone does not install that package.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_framework import ChatMiddlewareLayer, FunctionInvocationLayer, Message
from agent_framework.foundry import RawFoundryAgentChatClient
from agent_framework.observability import ChatTelemetryLayer


class _ChatomonFoundryChatClient(
    FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, RawFoundryAgentChatClient
):
    """Keep developer messages in the request; MAF strips them by default."""

    def _prepare_messages_for_azure_ai(
        self, messages: Sequence[Message]
    ) -> tuple[list[Message], str | None]:
        return list(messages), None
