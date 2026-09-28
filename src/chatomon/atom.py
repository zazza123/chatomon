"""The atom: a chat agent declared once and built into a native agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic

from opentelemetry.trace import TracerProvider

from chatomon.connectors.base import AgentT, Connector
from chatomon.engine import Engine, TurnCallback
from chatomon.errors import ChatomonConfigError
from chatomon.knowledge import Knowledge
from chatomon.observability import ContentPolicy, get_tracer
from chatomon.orbitals.base import Orbital
from chatomon.schemas import (
    DEFAULT_BLOCKED_PLACEHOLDER,
    DEFAULT_REFUSAL,
    CaptureContent,
    Side,
)

__all__ = ["Atom"]


class Atom(Generic[AgentT]):
    """A chat agent: instructions, knowledge, a connector and its orbitals.

    The atom never runs anything itself: [build][chatomon.atom.Atom.build]
    returns a native agent of the connector's framework, with chatomon's hooks
    installed. Orbitals get their own definitions; the atom's instructions and
    knowledge are used only by the nucleus.

    Args:
        name: Name of the agent, used in traces and reports.
        instructions: Definition of the nucleus, free text.
        connector: The framework and model the atom is built into.
        knowledge: Knowledge appended to the nucleus prompt; a string is wrapped
            in [Knowledge][chatomon.knowledge.Knowledge].
        input_orbitals: Orbitals run in order on the user message.
        output_orbitals: Orbitals run in order on the nucleus reply.
        refusal: Reply used when a message is blocked without a specific reply.
        blocked_placeholder: Text that replaces a blocked user message in the history.
        max_retries: Regenerations allowed when an output orbital blocks with guidance.
        on_turn: Receives the [TurnReport][chatomon.schemas.TurnReport] of every turn.
        capture_content: `"none"` (default), `"hash"` or `"full"`; applies to logs,
            traces and reports.
        redactor: Applied to recorded text when `capture_content="full"`.
        tracer_provider: OpenTelemetry provider; the global one when `None`.

    Raises:
        ChatomonConfigError: If a setting is invalid.

    Examples:
        ```python test="skip"
        from chatomon import Atom, Knowledge
        from chatomon.connectors.pydantic_ai import PydanticAIConnector
        from chatomon.orbitals import JudgeOrbital

        faq = Knowledge("Opening hours: Monday to Friday, 9 to 18.")
        atom = Atom(
            name="speedypack",
            instructions="You are the customer assistant of SpeedyPack.",
            knowledge=faq,
            connector=PydanticAIConnector(model="openai:gpt-5"),
            output_orbitals=[
                JudgeOrbital(
                    model="openai:gpt-5-mini",
                    instructions="Block replies that state facts missing from the reference.",
                    context=faq,
                ),
            ],
        )
        agent = atom.build()
        ```
    """

    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        connector: Connector[AgentT],
        knowledge: Knowledge | str | None = None,
        input_orbitals: Sequence[Orbital] = (),
        output_orbitals: Sequence[Orbital] = (),
        refusal: str = DEFAULT_REFUSAL,
        blocked_placeholder: str = DEFAULT_BLOCKED_PLACEHOLDER,
        max_retries: int = 2,
        on_turn: TurnCallback | None = None,
        capture_content: CaptureContent = "none",
        redactor: Callable[[str], str] | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        if not name.strip():
            raise ChatomonConfigError("name must not be empty")
        if not instructions.strip():
            raise ChatomonConfigError("instructions must not be empty")
        if not isinstance(connector, Connector):
            raise ChatomonConfigError(
                f"connector must be a Connector, got {type(connector).__name__}"
            )
        for orb in (*input_orbitals, *output_orbitals):
            if not isinstance(orb, Orbital):
                raise ChatomonConfigError(
                    f"orbitals must be Orbital instances, got {type(orb).__name__}"
                )
        if max_retries < 0:
            raise ChatomonConfigError(f"max_retries must be >= 0, got {max_retries!r}")
        if not refusal.strip():
            raise ChatomonConfigError("refusal must not be empty")
        if not blocked_placeholder.strip():
            raise ChatomonConfigError("blocked_placeholder must not be empty")
        self.name = name
        self.instructions = instructions
        self.connector = connector
        self.knowledge = Knowledge(knowledge) if isinstance(knowledge, str) else knowledge
        self.input_orbitals: tuple[Orbital, ...] = tuple(input_orbitals)
        self.output_orbitals: tuple[Orbital, ...] = tuple(output_orbitals)
        self.refusal = refusal
        self.blocked_placeholder = blocked_placeholder
        self.max_retries = max_retries
        self.on_turn = on_turn
        self.content_policy = ContentPolicy(capture_content, redactor)
        self.tracer_provider = tracer_provider

    @property
    def system_prompt(self) -> str:
        """The nucleus prompt: the instructions followed by the knowledge section."""
        if self.knowledge is None:
            return self.instructions
        return f"{self.instructions}\n\n{self.knowledge.render()}"

    def validate(self) -> None:
        """Check that the orbital chains are consistent.

        A name identifies one orbital instance: the same instance may appear once
        in each chain, two different instances may not share a name. Every name in
        `reads` must be an orbital placed earlier in the same chain or, for the
        output chain, an input orbital. Every orbital must support the side of the
        chain it is in.

        Raises:
            ChatomonConfigError: If a rule is violated; the message says what to fix.
        """
        by_name: dict[str, Orbital] = {}
        for orb in (*self.input_orbitals, *self.output_orbitals):
            if by_name.setdefault(orb.name, orb) is not orb:
                raise ChatomonConfigError(
                    f"two different orbitals are named {orb.name!r}; give each a unique name="
                )
        input_names = {orb.name for orb in self.input_orbitals}
        chains: tuple[tuple[str, Side, tuple[Orbital, ...]], ...] = (
            ("input_orbitals", "input", self.input_orbitals),
            ("output_orbitals", "output", self.output_orbitals),
        )
        for chain_name, side, chain in chains:
            seen: set[str] = set()
            for orb in chain:
                if orb.name in seen:
                    raise ChatomonConfigError(f"orbital {orb.name!r} appears twice in {chain_name}")
                if side not in orb.sides:
                    raise ChatomonConfigError(
                        f"{type(orb).__name__} {orb.name!r} cannot run on the {side} side; "
                        f"it supports: {sorted(orb.sides)}"
                    )
                available = seen | input_names if side == "output" else seen
                for read in orb.reads:
                    if read == orb.name:
                        raise ChatomonConfigError(
                            f"orbital {orb.name!r} cannot read its own annotations"
                        )
                    if read not in available:
                        where = f"before it in {chain_name}"
                        if side == "output":
                            where += " or in input_orbitals"
                        raise ChatomonConfigError(
                            f"orbital {orb.name!r} reads {read!r}, which must be an orbital "
                            f"placed {where}"
                        )
                seen.add(orb.name)

    def create_engine(self) -> Engine:
        """Create the engine that runs this atom's orbitals.

        Returns:
            A new engine configured from the atom.
        """
        return Engine(
            atom_name=self.name,
            input_orbitals=self.input_orbitals,
            output_orbitals=self.output_orbitals,
            refusal=self.refusal,
            max_retries=self.max_retries,
            on_turn=self.on_turn,
            content=self.content_policy,
            tracer=get_tracer(self.tracer_provider),
        )

    def build(self) -> AgentT:
        """Validate the atom, bind its orbitals and build the native agent.

        Returns:
            The native agent produced by the connector.

        Raises:
            ChatomonConfigError: If the configuration is invalid.
        """  # noqa: DOC502 - raised by validate()
        self.validate()
        bound: set[int] = set()
        for orb in (*self.input_orbitals, *self.output_orbitals):
            if id(orb) not in bound:
                orb.bind(self.connector)
                bound.add(id(orb))
        return self.connector.build(self, self.create_engine())
