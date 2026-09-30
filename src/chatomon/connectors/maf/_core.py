"""Build native MAF agents backed by a chat client."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from chatomon.connectors.base import Connector, ModelT
from chatomon.connectors.maf._middleware import (
    _ChatomonAgentMiddleware,
    _ChatomonChatMiddleware,
)
from chatomon.connectors.maf._shared import (
    Agent,
    ChatMiddlewareLayer,
    _generate_structured,
    _prepare_kwargs,
    _require_resolver,
)
from chatomon.errors import ChatomonConfigError

if TYPE_CHECKING:
    from chatomon.atom import Atom
    from chatomon.engine import Engine


class MAFConnector(Connector[Agent[Any]]):
    """Build an atom into a native `agent_framework.Agent`.

    The client can be any MAF chat client that implements `ChatMiddlewareLayer`,
    including `FoundryChatClient`. Provider clients such as `OpenAIChatClient`
    need their own MAF distribution in addition to `chatomon[maf]`. The
    connector places chatomon middleware
    outside agent middleware supplied by the caller. Local `AgentSession` history
    is used with `store=False`; pass a session from `agent.create_session()` to
    keep turns together. MAF's native run API is asynchronous.

    Args:
        client: The nucleus MAF chat client.
        resolve_client: Maps a `JudgeOrbital` model ID to another MAF chat client.
        **agent_kwargs: Forwarded to `agent_framework.Agent`.

    Raises:
        ChatomonConfigError: If an argument conflicts with atom identity, enables
            remote conversation continuity, or bypasses the chat middleware.

    Examples:
        ```{.python test="skip"}
        import asyncio
        import os

        from agent_framework.openai import OpenAIChatClient
        from openai import AsyncOpenAI

        from chatomon import Atom
        from chatomon.connectors.maf import MAFConnector


        async def main() -> None:
            async with AsyncOpenAI() as provider:
                client = OpenAIChatClient(
                    model=os.environ["MAF_MODEL"], async_client=provider
                )
                agent = Atom(
                    name="assistant",
                    instructions="Be concise.",
                    connector=MAFConnector(client),
                ).build()
                reply = await agent.run("Hello", session=agent.create_session())
                print(reply.text)


        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        client: Any,
        *,
        resolve_client: Callable[[str], Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self.client = client
        self.resolve_client = resolve_client
        self.agent_kwargs = _prepare_kwargs(agent_kwargs)

    def build(self, atom: Atom[Agent[Any]], engine: Engine) -> Agent[Any]:
        """Build the native MAF agent with chatomon's middleware installed.

        Args:
            atom: The validated atom.
            engine: The engine that runs the atom's orbitals.

        Returns:
            A native MAF agent.

        Raises:
            ChatomonConfigError: If the client cannot run chat middleware or
                preconfigured client middleware could rewrite checked output.
        """
        _require_resolver(atom, self, self.resolve_client)
        if not isinstance(self.client, ChatMiddlewareLayer):
            raise ChatomonConfigError(
                "MAFConnector requires a client with ChatMiddlewareLayer so chatomon "
                "can check replies before local history is saved"
            )
        if engine.output_orbitals and self.client.chat_middleware:
            raise ChatomonConfigError(
                "preconfigured client chat middleware can rewrite checked output; "
                "move it to MAFConnector(middleware=...)"
            )
        kwargs = dict(self.agent_kwargs)
        middleware = [
            _ChatomonAgentMiddleware(engine, atom.blocked_placeholder),
            _ChatomonChatMiddleware(engine),
            *(kwargs.pop("middleware", ()) or ()),
        ]
        return Agent(
            client=cast(Any, self.client),
            name=atom.name,
            instructions=atom.system_prompt,
            middleware=middleware,
            **kwargs,
        )

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
