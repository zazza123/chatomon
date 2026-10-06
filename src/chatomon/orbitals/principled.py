"""Orbitals backed by Principled Intelligence models (the `orbitals` library).

Install them with `pip install 'chatomon[principled]'`. Each orbital receives a
client you configure yourself (local, self-hosted or API backend) and its own
service description: it is never derived from the atom.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import Enum
from typing import Any, ClassVar, cast

from pydantic import BaseModel

from chatomon.errors import ChatomonConfigError, MissingExtraError
from chatomon.orbitals.base import Orbital
from chatomon.schemas import ErrorPolicy, Message, OrbitalContext, Side, Verdict

try:
    from orbitals.claim_extractor import AsyncClaimExtractor, ClaimExtractor
    from orbitals.scope_guard import AsyncScopeGuard, ScopeGuard, ScopeGuardOutput
    from orbitals.scope_guard_v2 import AsyncScopeGuardV2, ScopeGuardV2, ScopeGuardV2Output
    from orbitals.types import AIServiceDescription, AIServiceDescriptionV2
except ImportError as exc:
    if importlib.util.find_spec("orbitals") is None:
        raise MissingExtraError("principled", "orbitals") from exc
    raise

__all__ = [
    "CHIT_CHAT_GUIDANCE",
    "DEFAULT_SCOPE_RULES",
    "OUT_OF_SCOPE_GUIDANCE",
    "V2_OUTPUT_FIELDS",
    "ClaimExtractorOrbital",
    "HeadersProvider",
    "ScopeGuardOrbital",
    "ScopeRule",
]

HeadersProvider = Callable[[], Mapping[str, str] | Awaitable[Mapping[str, str]]]
"""Sync or async function returning HTTP headers, called before every request."""

ScopeGuardClient = AsyncScopeGuardV2 | ScopeGuardV2 | AsyncScopeGuard | ScopeGuard
ScopeGuardResult = ScopeGuardV2Output | ScopeGuardOutput

ScopeRule = Verdict | Callable[[ScopeGuardResult], Verdict]
"""A fixed verdict, or a function building one from the ScopeGuard result."""

CHIT_CHAT_GUIDANCE = "Reply briefly and steer the conversation back to the service."
"""Guidance used for the Chit Chat class."""

OUT_OF_SCOPE_GUIDANCE = (
    "The request is outside your scope. Politely explain what you can help with."
)
"""Guidance used for the Out of Scope class."""

V2_OUTPUT_FIELDS = ("scope_class", "evidences", "reasoning", "suggested_response")
"""Output fields always requested from ScopeGuard V2."""


def _suggested_response(result: ScopeGuardResult, *, escalate: bool) -> Verdict:
    text = getattr(result, "suggested_response", None)
    if not text or not text.strip():
        raise ValueError(
            f"ScopeGuard returned {result.scope_class.value!r} without a suggested_response"
        )
    return Verdict.respond(text, escalate=escalate)


def _predefined_answer(result: ScopeGuardResult) -> Verdict:
    return _suggested_response(result, escalate=False)


def _human_oversight(result: ScopeGuardResult) -> Verdict:
    return _suggested_response(result, escalate=True)


DEFAULT_SCOPE_RULES: Mapping[str, ScopeRule] = {
    "Directly Supported": Verdict.allow(),
    "Potentially Supported": Verdict.allow(),
    "Chit Chat": Verdict.allow(CHIT_CHAT_GUIDANCE),
    "Predefined Answer": _predefined_answer,
    "Human Oversight": _human_oversight,
    "Out of Scope": Verdict.block(guidance=OUT_OF_SCOPE_GUIDANCE),
    "Restricted": Verdict.block(),
}
"""Default verdict for every ScopeGuard class (V2 classes include V1 ones)."""


def _scope_key(key: str | Enum) -> str:
    value = key.value if isinstance(key, Enum) else key
    if value not in DEFAULT_SCOPE_RULES:
        raise ChatomonConfigError(
            f"unknown ScopeGuard class {value!r}; expected one of {sorted(DEFAULT_SCOPE_RULES)}"
        )
    return value


def _window(history: Sequence[Message], size: int) -> tuple[Message, ...]:
    return tuple(history[-size:]) if size else ()


def _conversation(messages: Sequence[Message]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _check_headers(client: object, headers: HeadersProvider | None) -> None:
    if headers is not None and not hasattr(client, "custom_headers"):
        raise ChatomonConfigError("headers= needs a client created with backend='api'")


async def _refresh_headers(client: Any, headers: HeadersProvider | None) -> None:
    if headers is None:
        return
    if inspect.iscoroutinefunction(headers):
        value = await headers()
    else:
        # Sync providers (e.g. google-auth token fetches) do blocking I/O: run them in a
        # thread so the orbital's timeout still applies and the event loop keeps running.
        value = await asyncio.to_thread(headers)
    if inspect.isawaitable(value):
        value = await value
    client.custom_headers = {**client.custom_headers, **cast("Mapping[str, str]", value)}


async def _call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    return await asyncio.to_thread(method, *args, **kwargs)


def _describe_client(client: object) -> dict[str, Any]:
    return {
        "type": type(client).__qualname__,
        "model": getattr(client, "default_model", None),
        "api_url": getattr(client, "api_url", None),
    }


def _describe_rule(rule: ScopeRule) -> Any:
    if isinstance(rule, Verdict):
        return rule.model_dump()
    module = getattr(rule, "__module__", "?")
    return f"{module}.{getattr(rule, '__qualname__', repr(rule))}"


class ScopeGuardOrbital(Orbital):
    """Classifies the user message with ScopeGuard (V1 or V2) and maps the class to a verdict.

    Input side only: ScopeGuard classifies the last user message. Default mapping:

    | Class | Verdict |
    |---|---|
    | Directly / Potentially Supported | allow |
    | Chit Chat | allow with a short "steer back" guidance |
    | Predefined Answer (V2) | respond with `suggested_response` |
    | Human Oversight (V2) | respond with `suggested_response`, escalate |
    | Out of Scope | block with "explain what you can help with" guidance |
    | Restricted | block (default refusal) |

    Args:
        guard: A ScopeGuard client you configured (`AsyncScopeGuardV2`,
            `ScopeGuardV2`, `AsyncScopeGuard` or `ScopeGuard`). Async clients are
            recommended: sync ones run in a worker thread, and a request that
            hangs keeps that thread busy after the timeout (the sync client sets
            no HTTP timeout of its own).
        description: The guard's own service description (a string, or
            `AIServiceDescriptionV2` for V2 / `AIServiceDescription` for V1).
        mapping: Overrides for some classes; keys are class enums or their
            string values, values are verdicts or functions of the result.
        headers: Called before every request to refresh HTTP headers (for
            example a Cloud Run identity token). Needs an `api` backend client.
        history_window: Number of history messages sent with the user message.
        name: Unique name inside the atom; defaults to `"scope_guard"`.
        timeout: Seconds before the check is treated as failed.
        on_error: `"block"` (default) or `"allow"`.

    Raises:
        ChatomonConfigError: If a setting is invalid.

    Examples:
        ```{.python test="skip"}
        from orbitals.scope_guard_v2 import AsyncScopeGuardV2
        from orbitals.types import AIServiceDescriptionV2

        from chatomon.orbitals.principled import ScopeGuardOrbital

        scope_guard = ScopeGuardOrbital(
            guard=AsyncScopeGuardV2(backend="api", api_url="https://scope-guard.example.run.app"),
            description=AIServiceDescriptionV2(
                identity_role="Parcel tracking assistant of SpeedyPack",
                context="Italian courier; customers ask about their shipments",
                constraints=["Never discuss refunds"],
            ),
        )
        ```
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"input"})

    def __init__(
        self,
        *,
        guard: ScopeGuardClient,
        description: str | AIServiceDescription | AIServiceDescriptionV2,
        mapping: Mapping[str | Enum, ScopeRule] | None = None,
        headers: HeadersProvider | None = None,
        history_window: int = 6,
        name: str | None = None,
        timeout: float = 30.0,
        on_error: ErrorPolicy = "block",
    ) -> None:
        super().__init__(name=name, timeout=timeout, on_error=on_error)
        if not isinstance(guard, AsyncScopeGuardV2 | ScopeGuardV2 | AsyncScopeGuard | ScopeGuard):
            raise ChatomonConfigError(
                f"guard must be a ScopeGuard client from the orbitals library, "
                f"got {type(guard).__name__}"
            )
        self.is_v2 = isinstance(guard, AsyncScopeGuardV2 | ScopeGuardV2)
        if self.is_v2 and isinstance(description, AIServiceDescription):
            raise ChatomonConfigError(
                "ScopeGuard V2 needs an AIServiceDescriptionV2 (or a string) as description"
            )
        if not self.is_v2 and isinstance(description, AIServiceDescriptionV2):
            raise ChatomonConfigError(
                "ScopeGuard V1 needs an AIServiceDescription (or a string) as description"
            )
        _check_headers(guard, headers)
        if history_window < 0:
            raise ChatomonConfigError(f"history_window must be >= 0, got {history_window!r}")
        self.guard = guard
        self.description = description
        self.headers = headers
        self.history_window = history_window
        # The overrides as they were given; `rules` is what the orbital applies.
        self.mapping = mapping
        self.rules: dict[str, ScopeRule] = dict(DEFAULT_SCOPE_RULES)
        for key, rule in (mapping or {}).items():
            self.rules[_scope_key(key)] = rule

    async def check(self, ctx: OrbitalContext) -> Verdict:
        """Classify the user message and apply the rule of its class.

        Args:
            ctx: The orbital context.

        Returns:
            The verdict, with the class as reason and ScopeGuard's output as annotations.
        """
        await _refresh_headers(self.guard, self.headers)
        messages = (
            *_window(ctx.history, self.history_window),
            Message(role="user", content=ctx.message),
        )
        kwargs: dict[str, Any] = {"ai_service_description": self.description}
        if self.is_v2:
            kwargs["output_fields"] = list(V2_OUTPUT_FIELDS)
        result = await _call(self.guard.validate, _conversation(messages), **kwargs)
        scope = result.scope_class.value
        rule = self.rules[scope]
        verdict = rule(result) if callable(rule) else rule
        annotations = verdict.annotations
        if annotations is None:
            annotations = {
                "scope_class": scope,
                "evidences": result.evidences,
                "reasoning": getattr(result, "reasoning", None),
            }
        return verdict.model_copy(
            update={"reason": verdict.reason or scope, "annotations": annotations}
        )

    def config(self) -> dict[str, Any]:
        """Return the configuration: client, description, rules and window.

        Returns:
            A JSON-serializable dictionary.
        """
        description = (
            self.description.model_dump()
            if isinstance(self.description, BaseModel)
            else self.description
        )
        rules = {key: _describe_rule(rule) for key, rule in sorted(self.rules.items())}
        return {
            **super().config(),
            "guard": _describe_client(self.guard),
            "description": description,
            "rules": rules,
            "history_window": self.history_window,
        }


class ClaimExtractorOrbital(Orbital):
    """Extracts claims and intents from the last message; never blocks.

    Its annotations (an `Extractions` object with `claims` and `intents`) are meant
    for a later orbital that reads them, typically a
    [JudgeOrbital][chatomon.orbitals.judge.JudgeOrbital] with `reads=[...]`. Works
    on both sides: on the output side it analyses the nucleus reply.

    Args:
        extractor: A ClaimExtractor client you configured (`AsyncClaimExtractor`
            or `ClaimExtractor`).
        description: Optional service description (string or `AIServiceDescription`).
        intents_only: Extract only intents.
        headers: Called before every request to refresh HTTP headers.
        history_window: Number of history messages sent with the last message.
        name: Unique name inside the atom; defaults to `"claim_extractor"`.
        reads: Earlier orbitals whose annotations this orbital needs.
        timeout: Seconds before the check is treated as failed.
        on_error: `"block"` (default) or `"allow"`.

    Raises:
        ChatomonConfigError: If a setting is invalid.

    Examples:
        ```{.python test="skip"}
        from orbitals.claim_extractor import AsyncClaimExtractor

        from chatomon import Knowledge
        from chatomon.orbitals import JudgeOrbital
        from chatomon.orbitals.principled import ClaimExtractorOrbital

        faq = Knowledge("Opening hours: Monday to Friday, 9 to 18.")
        output_orbitals = [
            ClaimExtractorOrbital(
                name="claims",
                extractor=AsyncClaimExtractor(
                    backend="api", api_url="https://claims.example.run.app"
                ),
            ),
            JudgeOrbital(
                name="fact_check",
                model="openai:gpt-5-mini",
                instructions="Block replies whose claims are not supported by the reference.",
                context=faq,
                reads=["claims"],
            ),
        ]
        ```
    """

    def __init__(
        self,
        *,
        extractor: AsyncClaimExtractor | ClaimExtractor,
        description: str | AIServiceDescription | None = None,
        intents_only: bool = False,
        headers: HeadersProvider | None = None,
        history_window: int = 6,
        name: str | None = None,
        reads: Sequence[str] = (),
        timeout: float = 30.0,
        on_error: ErrorPolicy = "block",
    ) -> None:
        super().__init__(name=name, reads=reads, timeout=timeout, on_error=on_error)
        if not isinstance(extractor, AsyncClaimExtractor | ClaimExtractor):
            raise ChatomonConfigError(
                f"extractor must be a ClaimExtractor client from the orbitals library, "
                f"got {type(extractor).__name__}"
            )
        _check_headers(extractor, headers)
        if history_window < 0:
            raise ChatomonConfigError(f"history_window must be >= 0, got {history_window!r}")
        self.extractor = extractor
        self.description = description
        self.intents_only = intents_only
        self.headers = headers
        self.history_window = history_window

    async def check(self, ctx: OrbitalContext) -> Verdict:
        """Extract claims and intents from the message under check.

        Args:
            ctx: The orbital context.

        Returns:
            An `allow` verdict whose annotations are the extractions.
        """
        await _refresh_headers(self.extractor, self.headers)
        role = "user" if ctx.side == "input" else "assistant"
        messages = (
            *_window(ctx.history, self.history_window),
            Message(role=role, content=ctx.message),
        )
        result = await _call(
            self.extractor.extract,
            _conversation(messages),
            ai_service_description=self.description,
            intents_only=self.intents_only,
        )
        return Verdict.allow(annotations=result.extractions)

    def config(self) -> dict[str, Any]:
        """Return the configuration: client, description and options.

        Returns:
            A JSON-serializable dictionary.
        """
        description = (
            self.description.model_dump()
            if isinstance(self.description, BaseModel)
            else self.description
        )
        return {
            **super().config(),
            "extractor": _describe_client(self.extractor),
            "description": description,
            "intents_only": self.intents_only,
            "history_window": self.history_window,
        }
