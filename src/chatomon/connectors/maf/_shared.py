"""Shared validation and judge support for MAF connectors."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from chatomon.connectors.base import Connector, ModelT
from chatomon.errors import ChatomonConfigError, MissingExtraError
from chatomon.orbitals.judge import JudgeOrbital

try:
    from agent_framework import Agent, ChatMiddlewareLayer, HistoryProvider, InMemoryHistoryProvider
except ImportError as exc:
    if importlib.util.find_spec("agent_framework") is None:
        raise MissingExtraError("maf", "agent-framework-core") from exc
    raise

if TYPE_CHECKING:
    from chatomon.atom import Atom

__all__ = ["Agent", "ChatMiddlewareLayer", "HistoryProvider", "InMemoryHistoryProvider"]


_RESERVED = frozenset({"client", "name", "instructions"})
_REMOTE_IDS = ("conversation_id", "previous_response_id", "conversation")
_TURN_KEY = "_chatomon_maf_turn"
_NO_FINAL_REPLY = "the MAF run ended without a final text reply"
_STREAMING_REFUSED = (
    "MAF streams model output before chatomon's output orbitals can check it; "
    "use run(..., stream=False) when the atom has output orbitals"
)


def _reject_remote(options: Mapping[str, Any] | None) -> None:
    if options is None:
        return
    store = options.get("store")
    if store is not None and store is not False:
        raise ChatomonConfigError("MAF requires store=False for local session history")
    for key in _REMOTE_IDS:
        if options.get(key) is not None:
            raise ChatomonConfigError(
                f"MAF does not accept {key}; use a local AgentSession instead"
            )
    extra_body = options.get("extra_body")
    if isinstance(extra_body, Mapping) and extra_body.get("agent_session_id") is not None:
        raise ChatomonConfigError(
            "MAF does not accept an explicit extra_body.agent_session_id; "
            "use a local AgentSession instead"
        )


def _prepare_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    reserved = sorted(_RESERVED & kwargs.keys())
    if reserved:
        raise ChatomonConfigError(
            f"{', '.join(reserved)} is set by chatomon from the atom; remove it from the connector"
        )
    if kwargs.get("require_per_service_call_history_persistence"):
        raise ChatomonConfigError(
            "require_per_service_call_history_persistence is incompatible with "
            "chatomon's per-turn output checks"
        )
    default_options = dict(kwargs.get("default_options") or {})
    if "instructions" in default_options:
        raise ChatomonConfigError("default_options.instructions is set by chatomon from the atom")
    _reject_remote(default_options)
    default_options["store"] = False
    result = dict(kwargs)
    result["default_options"] = default_options
    providers: list[Any] = list(result.pop("context_providers", ()) or ())
    if not any(
        isinstance(provider, HistoryProvider) and provider.load_messages for provider in providers
    ):
        providers.insert(0, InMemoryHistoryProvider())
    result["context_providers"] = providers
    return result


def _require_resolver(atom: Atom[Any], connector: Connector[Any], resolve_client: Any) -> None:
    if resolve_client is None and any(
        isinstance(orb, JudgeOrbital) and orb.connector is connector
        for orb in (*atom.input_orbitals, *atom.output_orbitals)
    ):
        raise ChatomonConfigError(
            "JudgeOrbital with MAF requires resolve_client(model_id) on the connector"
        )


async def _generate_structured(
    resolve_client: Callable[[str], Any] | None,
    model: str,
    *,
    system: str,
    prompt: str,
    schema: type[ModelT],
) -> ModelT:
    if resolve_client is None:
        raise ChatomonConfigError(
            "JudgeOrbital with MAF requires resolve_client(model_id) on the connector"
        )
    client = resolve_client(model)
    if not callable(getattr(client, "get_response", None)):
        raise ChatomonConfigError("resolve_client must return a MAF chat client")
    agent = Agent(client=client, instructions=system, default_options={"store": False})
    response = await agent.run(prompt, options={"response_format": schema, "store": False})
    value = response.value
    if value is None:
        raise ValueError("the MAF judge returned no structured response")
    return schema.model_validate(value)
