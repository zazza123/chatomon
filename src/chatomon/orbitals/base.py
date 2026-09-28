"""The orbital extension point: the base class and the `@orbital` decorator."""

from __future__ import annotations

import asyncio
import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, overload

from chatomon.errors import ChatomonConfigError
from chatomon.observability import sha256_text, stable_json
from chatomon.schemas import ErrorPolicy, OrbitalContext, Side, Verdict

if TYPE_CHECKING:
    from chatomon.connectors.base import Connector  # pyright: ignore[reportMissingImports]

__all__ = ["FunctionOrbital", "Orbital", "OrbitalFunction", "orbital"]

OrbitalFunction = Callable[[OrbitalContext], Verdict | Awaitable[Verdict]]
"""A sync or async function that checks a message and returns a verdict."""

_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.\-]+")


def _default_name(cls: type) -> str:
    base = cls.__name__
    if base.endswith("Orbital") and base != "Orbital":
        base = base[: -len("Orbital")]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()


class Orbital(ABC):
    """A check on a user message before the nucleus, or on a reply before the user.

    Subclasses implement [check][chatomon.orbitals.base.Orbital.check]. The engine
    runs orbitals in the order they appear in the atom's chains, applies `timeout`
    and turns failures into a verdict according to `on_error`.

    Args:
        name: Unique name inside an atom. Defaults to the class name in snake
            case without the `Orbital` suffix (`ScopeGuardOrbital` becomes
            `scope_guard`).
        reads: Names of earlier orbitals whose annotations this orbital needs.
        timeout: Seconds the engine waits for `check` before treating it as failed.
        on_error: What the engine does when `check` fails or times out: `"block"`
            (default, fail-closed) or `"allow"`.

    Raises:
        ChatomonConfigError: If the name, reads, timeout or error policy is invalid.

    Examples:
        ```python
        from chatomon import Orbital, Verdict
        from chatomon.schemas import OrbitalContext


        class NoShouting(Orbital):
            async def check(self, ctx: OrbitalContext) -> Verdict:
                if ctx.message.isupper():
                    return Verdict.allow(guidance="The user is shouting: stay calm.")
                return Verdict.allow()


        assert NoShouting().name == "no_shouting"
        ```
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"input", "output"})
    """Sides this orbital can run on. Subclasses that support only one side override it."""

    def __init__(
        self,
        *,
        name: str | None = None,
        reads: Sequence[str] = (),
        timeout: float = 30.0,
        on_error: ErrorPolicy = "block",
    ) -> None:
        resolved = name if name is not None else _default_name(type(self))
        if not _NAME_PATTERN.fullmatch(resolved):
            raise ChatomonConfigError(
                f"invalid orbital name {resolved!r}: use letters, digits, '_', '-' or '.'"
            )
        if isinstance(reads, str):
            raise ChatomonConfigError("reads must be a list of orbital names, not a single string")
        if timeout <= 0:
            raise ChatomonConfigError(f"timeout must be positive, got {timeout!r}")
        if on_error not in ("block", "allow"):
            raise ChatomonConfigError(f"on_error must be 'block' or 'allow', got {on_error!r}")
        self.name: str = resolved
        self.reads: tuple[str, ...] = tuple(reads)
        self.timeout: float = float(timeout)
        self.on_error: ErrorPolicy = on_error

    @abstractmethod
    async def check(self, ctx: OrbitalContext) -> Verdict:
        """Check one message and return a verdict.

        Raising an exception is allowed: the engine applies `on_error`.

        Args:
            ctx: The text under check, the history and earlier annotations.

        Returns:
            The verdict.
        """

    def bind(self, connector: Connector[Any]) -> None:
        """Receive the atom's connector; called once by `Atom.build()`.

        The default does nothing. Orbitals that call a model, such as the judge,
        use the connector to reach it.

        Args:
            connector: The connector of the atom being built.
        """
        del connector

    def config(self) -> dict[str, Any]:
        """Return the public configuration that identifies this orbital.

        Subclasses extend the dictionary with their own settings. It feeds
        [fingerprint][chatomon.orbitals.base.Orbital.fingerprint].

        Returns:
            A JSON-serializable dictionary.
        """
        return {
            "kind": type(self).__qualname__,
            "name": self.name,
            "reads": list(self.reads),
            "timeout": self.timeout,
            "on_error": self.on_error,
        }

    def fingerprint(self) -> str:
        """Return a stable hash of the configuration, recorded in audit reports.

        Returns:
            `"sha256:"` followed by the hex digest of `config()` as canonical JSON.
        """
        return sha256_text(stable_json(self.config()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class FunctionOrbital(Orbital):
    """An orbital backed by a function; create it with [orbital][chatomon.orbitals.base.orbital].

    Sync functions run in a worker thread so they never block the event loop.

    Args:
        func: Function taking an [OrbitalContext][chatomon.schemas.OrbitalContext]
            and returning a [Verdict][chatomon.schemas.Verdict].
        name: Defaults to the function name.
        reads: Names of earlier orbitals whose annotations this orbital needs.
        timeout: Seconds before the check is treated as failed.
        on_error: `"block"` (default) or `"allow"`.
    """

    def __init__(
        self,
        func: OrbitalFunction,
        *,
        name: str | None = None,
        reads: Sequence[str] = (),
        timeout: float = 30.0,
        on_error: ErrorPolicy = "block",
    ) -> None:
        super().__init__(
            name=name if name is not None else getattr(func, "__name__", "function"),
            reads=reads,
            timeout=timeout,
            on_error=on_error,
        )
        self.func = func

    async def check(self, ctx: OrbitalContext) -> Verdict:
        """Call the wrapped function.

        Args:
            ctx: The context passed to the function.

        Returns:
            The function's verdict.

        Raises:
            TypeError: If the function does not return a Verdict.
        """
        if inspect.iscoroutinefunction(self.func):
            result: object = self.func(ctx)
        else:
            result = await asyncio.to_thread(self.func, ctx)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Verdict):
            raise TypeError(
                f"orbital {self.name!r} returned {type(result).__name__}, expected Verdict"
            )
        return result

    def config(self) -> dict[str, Any]:
        """Return the configuration, including the wrapped function's path.

        Returns:
            A JSON-serializable dictionary.
        """
        module = getattr(self.func, "__module__", "?")
        qualname = getattr(self.func, "__qualname__", repr(self.func))
        return {**super().config(), "function": f"{module}.{qualname}"}


@overload
def orbital(func: OrbitalFunction, /) -> FunctionOrbital: ...


@overload
def orbital(
    *,
    name: str | None = None,
    reads: Sequence[str] = (),
    timeout: float = 30.0,
    on_error: ErrorPolicy = "block",
) -> Callable[[OrbitalFunction], FunctionOrbital]: ...


def orbital(
    func: OrbitalFunction | None = None,
    /,
    *,
    name: str | None = None,
    reads: Sequence[str] = (),
    timeout: float = 30.0,
    on_error: ErrorPolicy = "block",
) -> FunctionOrbital | Callable[[OrbitalFunction], FunctionOrbital]:
    """Turn a function into an orbital.

    Use it bare (`@orbital`) or with the same keyword arguments as
    [Orbital][chatomon.orbitals.base.Orbital].

    Args:
        func: The function, when used bare.
        name: Defaults to the function name.
        reads: Names of earlier orbitals whose annotations this orbital needs.
        timeout: Seconds before the check is treated as failed.
        on_error: `"block"` (default) or `"allow"`.

    Returns:
        A [FunctionOrbital][chatomon.orbitals.base.FunctionOrbital], or a decorator
        that creates one.

    Examples:
        ```python
        from chatomon import Verdict, orbital
        from chatomon.schemas import OrbitalContext


        @orbital
        async def no_competitors(ctx: OrbitalContext) -> Verdict:
            if "amazon" in ctx.message.lower():
                return Verdict.block(guidance="Do not mention competitors.")
            return Verdict.allow()


        @orbital(name="length_limit", on_error="allow")
        def length_limit(ctx: OrbitalContext) -> Verdict:
            return Verdict.block() if len(ctx.message) > 2000 else Verdict.allow()


        assert no_competitors.name == "no_competitors"
        assert length_limit.on_error == "allow"
        ```
    """

    def wrap(f: OrbitalFunction) -> FunctionOrbital:
        return FunctionOrbital(f, name=name, reads=reads, timeout=timeout, on_error=on_error)

    return wrap(func) if func is not None else wrap
