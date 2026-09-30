"""Build native Foundry agents already configured in the service."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from agent_framework import (
    ChatMiddlewareLayer,
    FunctionInvocationLayer,
    Message,
)
from agent_framework.foundry import RawFoundryAgentChatClient
from agent_framework.observability import ChatTelemetryLayer

from chatomon.connectors.base import Connector, ModelT
from chatomon.connectors.maf._middleware import (
    _ChatomonAgentMiddleware,
    _ChatomonChatMiddleware,
)
from chatomon.connectors.maf._shared import _generate_structured, _prepare_kwargs, _require_resolver
from chatomon.errors import ChatomonConfigError, MissingExtraError

if TYPE_CHECKING:
    from agent_framework.foundry import FoundryAgent

    from chatomon.atom import Atom
    from chatomon.engine import Engine


class _ChatomonFoundryChatClient(
    FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, RawFoundryAgentChatClient
):
    """Keep developer messages in the request; MAF strips them by default."""

    def _prepare_messages_for_azure_ai(
        self, messages: Sequence[Message]
    ) -> tuple[list[Message], str | None]:
        return list(messages), None


class MAFFoundryConnector(Connector["FoundryAgent"]):
    """Build an atom into a native `agent_framework.foundry.FoundryAgent`.

    This connector attaches to an agent already configured in Foundry. Input
    orbitals are supported. Output orbitals are blocked until a real Foundry
    test proves rejected drafts are not reused in later turns. Atom instructions
    and knowledge are sent as a developer message because Foundry rejects the
    request-level `instructions` field when an agent is specified.

    Args:
        resolve_client: Maps a `JudgeOrbital` model ID to a MAF chat client.
        **foundry_kwargs: Forwarded to `FoundryAgent`; include the project
            endpoint, agent name and credential or project client. `client_type`
            is managed by this connector.

    Raises:
        ChatomonConfigError: If an argument conflicts with atom identity or
            enables remote conversation continuity.
        MissingExtraError: If `chatomon[maf-foundry]` is not installed.

    Examples:
        ```{.python test="skip"}
        from azure.identity import DefaultAzureCredential
        from chatomon import Atom
        from chatomon.connectors.maf import MAFFoundryConnector

        atom = Atom(name="assistant", instructions="Be concise.",
                    connector=MAFFoundryConnector(
                        project_endpoint="https://example.services.ai.azure.com",
                        agent_name="my-agent", credential=DefaultAzureCredential()))
        agent = atom.build()
        reply = await agent.run("Hello", session=agent.create_session())
        ```
    """

    def __init__(
        self,
        *,
        resolve_client: Callable[[str], Any] | None = None,
        **foundry_kwargs: Any,
    ) -> None:
        if "client_type" in foundry_kwargs:
            raise ChatomonConfigError(
                "MAFFoundryConnector sets client_type to preserve atom instructions "
                "and chat middleware"
            )
        self.resolve_client = resolve_client
        self.foundry_kwargs = _prepare_kwargs(foundry_kwargs)

    def build(self, atom: Atom[FoundryAgent], engine: Engine) -> FoundryAgent:
        """Build a native Foundry agent with input checks and local history.

        Args:
            atom: The validated atom.
            engine: The engine that runs the atom's orbitals.

        Returns:
            A native Foundry agent.

        Raises:
            ChatomonConfigError: If output orbitals or a raw client are supplied.
            MissingExtraError: If the Foundry extra is not installed.
        """
        if engine.output_orbitals:
            raise ChatomonConfigError(
                "FoundryAgent output orbitals need a real-service test proving rejected "
                "drafts are not reused in later turns; use MAFConnector(FoundryChatClient(...)) "
                "for checked output"
            )
        _require_resolver(atom, self, self.resolve_client)
        try:
            from agent_framework.foundry import FoundryAgent as NativeFoundryAgent
        except ImportError as exc:
            raise MissingExtraError("maf-foundry", "agent-framework-foundry") from exc
        kwargs = dict(self.foundry_kwargs)
        middleware = [
            _ChatomonAgentMiddleware(engine, atom.blocked_placeholder),
            _ChatomonChatMiddleware(engine, foundry_instructions=atom.system_prompt),
            *(kwargs.pop("middleware", ()) or ()),
        ]
        agent = NativeFoundryAgent(
            name=atom.name,
            instructions=atom.system_prompt,
            client_type=_ChatomonFoundryChatClient,
            middleware=middleware,
            **kwargs,
        )
        if not isinstance(agent.client, ChatMiddlewareLayer):
            raise ChatomonConfigError(
                "MAFFoundryConnector requires a Foundry client with ChatMiddlewareLayer"
            )
        return agent

    async def generate_structured(
        self, model: str, *, system: str, prompt: str, schema: type[ModelT]
    ) -> ModelT:
        """Run a structured judge call with a resolved MAF chat client.

        Args:
            model: The judge model identifier.
            system: Judge instructions.
            prompt: Judge input.
            schema: Required Pydantic result type.

        Returns:
            The validated judge result.

        Raises:
            ChatomonConfigError: If `resolve_client` is missing or invalid.
            ValueError: If the judge returned no structured answer.
        """  # noqa: DOC502 - delegated to _generate_structured
        return await _generate_structured(
            self.resolve_client, model, system=system, prompt=prompt, schema=schema
        )
