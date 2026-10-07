"""Orbitals backed by Principled Intelligence models (the `orbitals` library).

Install them with `pip install 'chatomon[principled]'`. Each orbital receives a
client you configure yourself (local, self-hosted or API backend) and its own
service description: it is never derived from the atom. For a service reached
over HTTP, [ApiBackend][chatomon.orbitals.principled.ApiBackend] is a shorter
way to name it than building the client.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, cast
from urllib.parse import urlsplit

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
    "ApiBackend",
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


_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NOT_AN_ADDRESS = "api_url must start with http:// or https:// and name a host"


@dataclass(frozen=True, kw_only=True)
class ApiBackend:
    """A Principled Intelligence service reached over HTTP: the `api` backend of `orbitals`.

    A shortcut for the common case. Give it to
    [ScopeGuardOrbital][chatomon.orbitals.principled.ScopeGuardOrbital] as `guard`
    or to [ClaimExtractorOrbital][chatomon.orbitals.principled.ClaimExtractorOrbital]
    as `extractor`, and the orbital builds the asynchronous client itself. For
    anything else, such as a local backend or a client with its own settings,
    pass a client you configured.

    It holds the name of the variable, never the key: the orbital reads the
    variable when it is created.

    Args:
        api_url: Where the service answers, starting with `http://` or `https://`.
        api_key_env: Name of the environment variable that holds the API key. When
            `None` no key is given, and the client of the `orbitals` library uses
            `PRINCIPLED_API_KEY` if that is set.
        model: Model the service should use; the library's default when `None`.

    Raises:
        ChatomonConfigError: If the address is not an HTTP address, holds a user or
            a password, or `api_key_env` is not the name of a variable. The message
            never quotes the value it refuses.

    Examples:
        ```python
        from chatomon.orbitals.principled import ApiBackend

        backend = ApiBackend(api_url="http://127.0.0.1:8080", api_key_env="SCOPE_GUARD_KEY")
        assert backend.api_key_env == "SCOPE_GUARD_KEY"  # the name: the value is never stored
        ```
    """

    api_url: str
    api_key_env: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        url = self.api_url
        try:
            parts = urlsplit(url) if isinstance(url, str) else None
            host = parts.hostname if parts is not None else None
            # Reading the port is what refuses one that is not a number.
            port_is_valid = parts is None or parts.port is None or parts.port >= 0
        except ValueError:
            parts, host, port_is_valid = None, None, False
        if parts is not None and (parts.username is not None or parts.password is not None):
            raise ChatomonConfigError(
                "api_url must not hold a user or a password: "
                "name an environment variable in api_key_env"
            )
        plain = isinstance(url, str) and url.isprintable() and not any(c.isspace() for c in url)
        if parts is None or parts.scheme not in ("http", "https") or not host:
            raise ChatomonConfigError(_NOT_AN_ADDRESS)
        if not plain or not port_is_valid:
            raise ChatomonConfigError(_NOT_AN_ADDRESS)
        name = self.api_key_env
        if name is not None and not (isinstance(name, str) and _VARIABLE.fullmatch(name)):
            raise ChatomonConfigError(
                "api_key_env takes the name of an environment variable, not its value"
            )
        if self.model is not None and not isinstance(self.model, str):
            raise ChatomonConfigError("model must be text")


def _api_key(backend: ApiBackend) -> str | None:
    """Read the key of a backend from the environment.

    Returns:
        The key, or `None` when the backend names no variable.

    Raises:
        ChatomonConfigError: If the variable is not set or is empty.
    """
    if backend.api_key_env is None:
        return None
    key = os.environ.get(backend.api_key_env)
    if not key:
        raise ChatomonConfigError(f"the environment variable {backend.api_key_env} is not set")
    return key


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
            `ScopeGuardV2`, `AsyncScopeGuard` or `ScopeGuard`), or an
            [ApiBackend][chatomon.orbitals.principled.ApiBackend], from which the
            orbital builds an `AsyncScopeGuardV2` itself. Async clients are
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
        ChatomonConfigError: If a setting is invalid, or the environment variable an
            `ApiBackend` names is not set.

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

        The same service named by plain values, with its key in a variable:

        ```{.python test="skip"}
        from chatomon.orbitals.principled import ApiBackend, ScopeGuardOrbital

        scope_guard = ScopeGuardOrbital(
            guard=ApiBackend(
                api_url="https://scope-guard.example.run.app",
                api_key_env="PRINCIPLED_API_KEY",
            ),
            description="Parcel tracking assistant of SpeedyPack, an Italian courier.",
        )
        ```
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"input"})

    calls_model = True
    """Every check is a call to the service."""

    def __init__(
        self,
        *,
        guard: ScopeGuardClient | ApiBackend,
        description: str | AIServiceDescription | AIServiceDescriptionV2,
        mapping: Mapping[str | Enum, ScopeRule] | None = None,
        headers: HeadersProvider | None = None,
        history_window: int = 6,
        name: str | None = None,
        timeout: float = 30.0,
        on_error: ErrorPolicy = "block",
    ) -> None:
        super().__init__(name=name, timeout=timeout, on_error=on_error)
        if isinstance(guard, ApiBackend):
            client: ScopeGuardClient = AsyncScopeGuardV2(
                backend="api", api_url=guard.api_url, model=guard.model, api_key=_api_key(guard)
            )
        elif isinstance(guard, AsyncScopeGuardV2 | ScopeGuardV2 | AsyncScopeGuard | ScopeGuard):
            client = guard
        else:
            raise ChatomonConfigError(
                f"guard must be a ScopeGuard client from the orbitals library or an ApiBackend, "
                f"got {type(guard).__name__}"
            )
        self.is_v2 = isinstance(client, AsyncScopeGuardV2 | ScopeGuardV2)
        if self.is_v2 and isinstance(description, AIServiceDescription):
            raise ChatomonConfigError(
                "ScopeGuard V2 needs an AIServiceDescriptionV2 (or a string) as description"
            )
        if not self.is_v2 and isinstance(description, AIServiceDescriptionV2):
            raise ChatomonConfigError(
                "ScopeGuard V1 needs an AIServiceDescription (or a string) as description"
            )
        _check_headers(client, headers)
        if history_window < 0:
            raise ChatomonConfigError(f"history_window must be >= 0, got {history_window!r}")
        # What was given, a client or a backend; `_client` is what the orbital calls.
        self.guard = guard
        self._client = client
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
        await _refresh_headers(self._client, self.headers)
        messages = (
            *_window(ctx.history, self.history_window),
            Message(role="user", content=ctx.message),
        )
        kwargs: dict[str, Any] = {"ai_service_description": self.description}
        if self.is_v2:
            kwargs["output_fields"] = list(V2_OUTPUT_FIELDS)
        result = await _call(self._client.validate, _conversation(messages), **kwargs)
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
            "guard": _describe_client(self._client),
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
            or `ClaimExtractor`), or an
            [ApiBackend][chatomon.orbitals.principled.ApiBackend], from which the
            orbital builds an `AsyncClaimExtractor` itself.
        description: Optional service description (string or `AIServiceDescription`).
        intents_only: Extract only intents.
        headers: Called before every request to refresh HTTP headers.
        history_window: Number of history messages sent with the last message.
        name: Unique name inside the atom; defaults to `"claim_extractor"`.
        reads: Earlier orbitals whose annotations this orbital needs.
        timeout: Seconds before the check is treated as failed.
        on_error: `"block"` (default) or `"allow"`.

    Raises:
        ChatomonConfigError: If a setting is invalid, or the environment variable an
            `ApiBackend` names is not set.

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

    calls_model = True
    """Every check is a call to the service."""

    def __init__(
        self,
        *,
        extractor: AsyncClaimExtractor | ClaimExtractor | ApiBackend,
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
        client: AsyncClaimExtractor | ClaimExtractor
        if isinstance(extractor, ApiBackend):
            url, key = extractor.api_url, _api_key(extractor)
            if extractor.model is None:
                # No model is given, so the library keeps its default.
                client = AsyncClaimExtractor(backend="api", api_url=url, api_key=key)
            else:
                client = AsyncClaimExtractor(
                    backend="api", api_url=url, api_key=key, model=extractor.model
                )
        elif isinstance(extractor, AsyncClaimExtractor | ClaimExtractor):
            client = extractor
        else:
            raise ChatomonConfigError(
                f"extractor must be a ClaimExtractor client from the orbitals library "
                f"or an ApiBackend, got {type(extractor).__name__}"
            )
        _check_headers(client, headers)
        if history_window < 0:
            raise ChatomonConfigError(f"history_window must be >= 0, got {history_window!r}")
        # What was given, a client or a backend; `_client` is what the orbital calls.
        self.extractor = extractor
        self._client = client
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
        await _refresh_headers(self._client, self.headers)
        role = "user" if ctx.side == "input" else "assistant"
        messages = (
            *_window(ctx.history, self.history_window),
            Message(role=role, content=ctx.message),
        )
        result = await _call(
            self._client.extract,
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
            "extractor": _describe_client(self._client),
            "description": description,
            "intents_only": self.intents_only,
            "history_window": self.history_window,
        }
