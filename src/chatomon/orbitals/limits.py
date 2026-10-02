"""Ready-made orbitals that cap the length of messages and the number of turns."""

from __future__ import annotations

from typing import Any

from chatomon.errors import ChatomonConfigError
from chatomon.orbitals.base import Orbital
from chatomon.schemas import ErrorPolicy, OrbitalContext, Verdict

__all__ = ["MessageLengthOrbital"]


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ChatomonConfigError(f"{name} must be a positive integer, got {value!r}")
    return value


def _not_blank(name: str, value: str) -> str:
    if not value.strip():
        raise ChatomonConfigError(f"{name} must not be empty")
    return value


class MessageLengthOrbital(Orbital):
    """Blocks a user message or a nucleus reply that is too long.

    Words are counted by splitting the text on whitespace, characters with
    `len()`; the text is too long when it exceeds any limit that is set. On the
    input side the user receives `reply`, the nucleus is not called and the
    message is replaced by the placeholder in the history. On the output side the
    nucleus rewrites its reply following the guidance, up to the atom's
    `max_retries`; when the retries run out the user receives the atom's refusal.

    Every verdict carries the annotations `{"words": ..., "chars": ...}`; the
    audit reasons are `message_too_long` (input) and `reply_too_long` (output).

    Args:
        max_words: Maximum number of words.
        max_chars: Maximum number of characters.
        reply: Sent to the user when their message is too long.
        guidance: Instruction for rewriting a reply that is too long; by default
            it states the limits that are set.
        name: Defaults to `message_length`.
        timeout: Seconds the engine waits for the check.
        on_error: `"block"` (default) or `"allow"`.

    Raises:
        ChatomonConfigError: If no limit is set, a limit is not a positive
            integer, or a text is empty.

    Examples:
        ```python
        import asyncio

        from chatomon.orbitals import MessageLengthOrbital
        from chatomon.schemas import OrbitalContext

        limit = MessageLengthOrbital(max_words=5)
        message = OrbitalContext(
            side="input", message="one two three four five six", turn_id="t1"
        )
        verdict = asyncio.run(limit.check(message))
        assert verdict.action == "block"
        assert verdict.reason == "message_too_long"
        assert verdict.annotations == {"words": 6, "chars": 27}
        ```
    """

    def __init__(
        self,
        *,
        max_words: int | None = None,
        max_chars: int | None = None,
        reply: str = "Your message is too long. Please shorten it and try again.",
        guidance: str | None = None,
        name: str | None = None,
        timeout: float = 30.0,
        on_error: ErrorPolicy = "block",
    ) -> None:
        super().__init__(name=name, timeout=timeout, on_error=on_error)
        if max_words is None and max_chars is None:
            raise ChatomonConfigError("set max_words, max_chars or both")
        self.max_words = None if max_words is None else _positive_int("max_words", max_words)
        self.max_chars = None if max_chars is None else _positive_int("max_chars", max_chars)
        self.reply = _not_blank("reply", reply)
        self.guidance = None if guidance is None else _not_blank("guidance", guidance)

    @property
    def output_guidance(self) -> str:
        """The guidance for a reply that is too long: `guidance`, or one stating the limits."""
        if self.guidance is not None:
            return self.guidance
        limits: list[str] = []
        if self.max_words is not None:
            limits.append(f"{self.max_words} words")
        if self.max_chars is not None:
            limits.append(f"{self.max_chars} characters")
        return f"Your previous reply was too long. Keep it to at most {' and '.join(limits)}."

    async def check(self, ctx: OrbitalContext) -> Verdict:
        """Measure the message and block it when it exceeds a limit.

        Args:
            ctx: The text under check; the history is not used.

        Returns:
            `allow` within the limits; otherwise `block` with `reply` on the input
            side, or with the guidance on the output side.
        """
        words = len(ctx.message.split())
        chars = len(ctx.message)
        measures = {"words": words, "chars": chars}
        too_long = (self.max_words is not None and words > self.max_words) or (
            self.max_chars is not None and chars > self.max_chars
        )
        if not too_long:
            return Verdict.allow(annotations=measures)
        if ctx.side == "input":
            return Verdict.block(self.reply, reason="message_too_long", annotations=measures)
        return Verdict.block(
            guidance=self.output_guidance, reason="reply_too_long", annotations=measures
        )

    def config(self) -> dict[str, Any]:
        """Return the configuration, including the limits and the texts.

        Returns:
            A JSON-serializable dictionary.
        """
        return {
            **super().config(),
            "max_words": self.max_words,
            "max_chars": self.max_chars,
            "reply": self.reply,
            "guidance": self.guidance,
        }
