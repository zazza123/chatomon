from typing import Any

import pytest

from chatomon import ChatomonConfigError
from chatomon.orbitals import MessageLengthOrbital, TurnLimitOrbital
from chatomon.schemas import Message, OrbitalContext, Side

LENGTH_REPLY = "Your message is too long. Please shorten it and try again."


def ctx(message: str = "hello", *, side: Side = "input", history=()) -> OrbitalContext:
    return OrbitalContext(side=side, message=message, history=tuple(history), turn_id="t1")


# --- MessageLengthOrbital: construction -------------------------------------------------


def test_length_default_name():
    assert MessageLengthOrbital(max_words=5).name == "message_length"


def test_length_needs_at_least_one_limit():
    with pytest.raises(ChatomonConfigError, match="max_words, max_chars or both"):
        MessageLengthOrbital()


@pytest.mark.parametrize("field", ["max_words", "max_chars"])
@pytest.mark.parametrize("value", [0, -1, 2.5, True, "10"])
def test_length_limits_must_be_positive_integers(field, value):
    with pytest.raises(ChatomonConfigError, match=f"{field} must be a positive integer"):
        MessageLengthOrbital(**{field: value})


def test_length_reply_must_not_be_blank():
    with pytest.raises(ChatomonConfigError, match="reply must not be empty"):
        MessageLengthOrbital(max_words=5, reply="  ")


@pytest.mark.parametrize("value", [None, 42])
def test_length_reply_must_be_text(value):
    with pytest.raises(ChatomonConfigError, match="reply must not be empty"):
        MessageLengthOrbital(max_words=5, reply=value)


def test_length_guidance_must_not_be_blank_when_given():
    with pytest.raises(ChatomonConfigError, match="guidance must not be empty"):
        MessageLengthOrbital(max_words=5, guidance="")


def test_length_keeps_the_base_settings():
    orb = MessageLengthOrbital(max_chars=10, name="short", timeout=2, on_error="allow")
    assert (orb.name, orb.timeout, orb.on_error) == ("short", 2.0, "allow")


# --- MessageLengthOrbital: checks -------------------------------------------------------


async def test_length_at_the_word_limit_is_allowed():
    verdict = await MessageLengthOrbital(max_words=3).check(ctx("one two three"))
    assert verdict.action == "allow"
    assert verdict.annotations == {"words": 3, "chars": 13}


async def test_length_over_the_word_limit_blocks_the_input_with_the_reply():
    verdict = await MessageLengthOrbital(max_words=3).check(ctx("one two three four"))
    assert (verdict.action, verdict.reply, verdict.reason) == (
        "block",
        LENGTH_REPLY,
        "message_too_long",
    )
    assert verdict.guidance is None
    assert verdict.annotations == {"words": 4, "chars": 18}


async def test_length_at_and_over_the_char_limit():
    orb = MessageLengthOrbital(max_chars=5)
    assert (await orb.check(ctx("abcde"))).action == "allow"
    assert (await orb.check(ctx("abcdef"))).action == "block"


async def test_length_any_exceeded_limit_blocks():
    orb = MessageLengthOrbital(max_words=10, max_chars=5)
    assert (await orb.check(ctx("abcdefgh"))).action == "block"
    orb = MessageLengthOrbital(max_words=1, max_chars=100)
    assert (await orb.check(ctx("two words"))).action == "block"


async def test_length_custom_reply():
    verdict = await MessageLengthOrbital(max_words=1, reply="Shorter, please.").check(ctx("a b"))
    assert verdict.reply == "Shorter, please."


async def test_length_counts_any_whitespace_as_a_separator():
    verdict = await MessageLengthOrbital(max_words=3).check(ctx("one\ntwo\t\tthree   four"))
    assert verdict.action == "block"
    assert verdict.annotations["words"] == 4


@pytest.mark.parametrize("message", ["", "   \n\t "])
async def test_length_empty_or_blank_message_has_no_words(message):
    verdict = await MessageLengthOrbital(max_words=1).check(ctx(message))
    assert verdict.action == "allow"
    assert verdict.annotations == {"words": 0, "chars": len(message)}


async def test_length_counts_characters_as_code_points():
    verdict = await MessageLengthOrbital(max_chars=4).check(ctx("città"))
    assert verdict.action == "block"
    assert verdict.annotations["chars"] == 5


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({"max_words": 150}, "Keep it to at most 150 words."),
        ({"max_chars": 800}, "Keep it to at most 800 characters."),
        ({"max_words": 150, "max_chars": 800}, "Keep it to at most 150 words and 800 characters."),
    ],
)
async def test_length_over_the_limit_on_output_asks_for_a_shorter_reply(limits, expected):
    verdict = await MessageLengthOrbital(**limits).check(ctx("x" * 801 + " w" * 150, side="output"))
    assert (verdict.action, verdict.reason, verdict.reply) == ("block", "reply_too_long", None)
    assert verdict.guidance == f"Your previous reply was too long. {expected}"


async def test_length_custom_guidance_is_used_verbatim():
    orb = MessageLengthOrbital(max_words=1, guidance="Answer in one word.")
    verdict = await orb.check(ctx("two words", side="output"))
    assert verdict.guidance == "Answer in one word."


async def test_length_within_the_limit_on_output_is_allowed():
    verdict = await MessageLengthOrbital(max_words=5).check(ctx("fine", side="output"))
    assert verdict.action == "allow"
    assert verdict.guidance is None


async def test_length_ignores_the_history():
    history = [Message(role="user", content="word " * 100)]
    verdict = await MessageLengthOrbital(max_words=5).check(ctx("short", history=history))
    assert verdict.action == "allow"


# --- MessageLengthOrbital: configuration ------------------------------------------------


def test_length_config_lists_every_setting():
    config = MessageLengthOrbital(max_words=5).config()
    assert config["kind"] == "MessageLengthOrbital"
    assert config["max_words"] == 5
    assert config["max_chars"] is None
    assert config["reply"] == LENGTH_REPLY
    assert config["guidance"] is None


def test_length_fingerprint_follows_the_settings():
    base = MessageLengthOrbital(max_words=5).fingerprint()
    assert MessageLengthOrbital(max_words=5).fingerprint() == base
    assert MessageLengthOrbital(max_words=6).fingerprint() != base
    assert MessageLengthOrbital(max_words=5, reply="Too long.").fingerprint() != base


# --- TurnLimitOrbital -------------------------------------------------------------------

TURN_REPLY = "This conversation has reached its limit. Please start a new one."
TURN_WARNING = "The conversation will end soon: help the user wrap up."
PLACEHOLDER = "[message removed by chatomon]"


def turns(count: int) -> list[Message]:
    """A history with `count` completed turns (user + assistant)."""
    history: list[Message] = []
    for index in range(count):
        history.append(Message(role="user", content=f"question {index}"))
        history.append(Message(role="assistant", content=f"answer {index}"))
    return history


def test_turn_default_name_and_side():
    orb = TurnLimitOrbital(max_turns=3)
    assert orb.name == "turn_limit"
    assert orb.sides == frozenset({"input"})


@pytest.mark.parametrize("value", [0, -2, 1.5, True])
def test_turn_max_turns_must_be_a_positive_integer(value):
    with pytest.raises(ChatomonConfigError, match="max_turns must be a positive integer"):
        TurnLimitOrbital(max_turns=value)


@pytest.mark.parametrize("value", [0, -1, False])
def test_turn_warn_before_must_be_a_positive_integer(value):
    with pytest.raises(ChatomonConfigError, match="warn_before must be a positive integer"):
        TurnLimitOrbital(max_turns=5, warn_before=value)


@pytest.mark.parametrize("value", [5, 6])
def test_turn_warn_before_must_be_smaller_than_max_turns(value):
    with pytest.raises(ChatomonConfigError, match="warn_before must be smaller than max_turns"):
        TurnLimitOrbital(max_turns=5, warn_before=value)


@pytest.mark.parametrize("field", ["reply", "warning"])
def test_turn_texts_must_not_be_blank(field):
    with pytest.raises(ChatomonConfigError, match=f"{field} must not be empty"):
        settings: dict[str, Any] = {field: " "}
        TurnLimitOrbital(max_turns=5, **settings)


@pytest.mark.parametrize("field", ["reply", "warning"])
def test_turn_texts_must_be_text(field):
    settings: dict[str, Any] = {field: None}
    with pytest.raises(ChatomonConfigError, match=f"{field} must not be empty"):
        TurnLimitOrbital(max_turns=5, **settings)


async def test_turn_within_the_limit_is_allowed():
    verdict = await TurnLimitOrbital(max_turns=3).check(ctx(history=turns(2)))
    assert verdict.action == "allow"
    assert verdict.guidance is None
    assert verdict.annotations == {"turn": 3, "max_turns": 3}


async def test_turn_past_the_limit_answers_with_the_reply():
    verdict = await TurnLimitOrbital(max_turns=3).check(ctx(history=turns(3)))
    assert (verdict.action, verdict.reply, verdict.reason) == ("respond", TURN_REPLY, "turn_limit")
    assert verdict.annotations == {"turn": 4, "max_turns": 3}


async def test_turn_keeps_answering_after_the_limit():
    orb = TurnLimitOrbital(max_turns=3, reply="Goodbye.")
    for done in (3, 4, 10):
        verdict = await orb.check(ctx(history=turns(done)))
        assert (verdict.action, verdict.reply) == ("respond", "Goodbye.")


async def test_turn_counts_only_user_messages():
    history = [Message(role="assistant", content="Welcome!"), *turns(1)]
    verdict = await TurnLimitOrbital(max_turns=5).check(ctx(history=history))
    assert verdict.annotations["turn"] == 2


async def test_turn_counts_blocked_messages():
    history = [Message(role="user", content=PLACEHOLDER), *turns(1)]
    verdict = await TurnLimitOrbital(max_turns=2).check(ctx(history=history))
    assert verdict.action == "respond"
    assert verdict.annotations["turn"] == 3


@pytest.mark.parametrize(
    ("done", "action", "reason"),
    [
        (7, "allow", None),
        (8, "allow", "turn_limit_warning"),
        (9, "allow", "turn_limit_warning"),
        (10, "respond", "turn_limit"),
    ],
)
async def test_turn_warning_window(done, action, reason):
    orb = TurnLimitOrbital(max_turns=10, warn_before=2)
    verdict = await orb.check(ctx(history=turns(done)))
    assert (verdict.action, verdict.reason) == (action, reason)
    if reason == "turn_limit_warning":
        assert verdict.guidance == TURN_WARNING


async def test_turn_without_warn_before_never_warns():
    verdict = await TurnLimitOrbital(max_turns=10).check(ctx(history=turns(9)))
    assert (verdict.action, verdict.guidance, verdict.reason) == ("allow", None, None)


async def test_turn_custom_warning():
    orb = TurnLimitOrbital(max_turns=2, warn_before=1, warning="Offer to email a summary.")
    verdict = await orb.check(ctx(history=turns(1)))
    assert verdict.guidance == "Offer to email a summary."


async def test_turn_one_instance_serves_many_conversations():
    orb = TurnLimitOrbital(max_turns=2)
    assert (await orb.check(ctx(history=turns(5)))).action == "respond"
    assert (await orb.check(ctx(history=()))).action == "allow"
    assert (await orb.check(ctx(history=turns(1)))).action == "allow"


def test_turn_config_lists_every_setting():
    config = TurnLimitOrbital(max_turns=10, warn_before=2).config()
    assert config["kind"] == "TurnLimitOrbital"
    assert (config["max_turns"], config["warn_before"]) == (10, 2)
    assert (config["reply"], config["warning"]) == (TURN_REPLY, TURN_WARNING)


def test_turn_fingerprint_follows_the_settings():
    base = TurnLimitOrbital(max_turns=10).fingerprint()
    assert TurnLimitOrbital(max_turns=10).fingerprint() == base
    assert TurnLimitOrbital(max_turns=11).fingerprint() != base
    assert TurnLimitOrbital(max_turns=10, warn_before=1).fingerprint() != base
