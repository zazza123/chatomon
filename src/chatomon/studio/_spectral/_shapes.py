"""Requests and replies in the two shapes of the OpenAI API a request can have.

Pure functions: the body of a request, already parsed from JSON, goes in and a
conversation comes out; a reply goes in and the body of the answer comes out.
"""

from __future__ import annotations

import time
from typing import Any, Literal
from uuid import uuid4

CHAT = "/v1/chat/completions"
RESPONSES = "/v1/responses"
ROUTES = (RESPONSES, CHAT)
"""The routes the adapter answers, first the one a Spectral target uses unless told otherwise."""

Said = tuple[Literal["user", "assistant"], str]
"""Who spoke and what was said."""

_SPEAKERS = ("user", "assistant")
_CHAT_PARTS = frozenset({"text"})
_RESPONSES_PARTS = frozenset({"input_text", "output_text", "text"})


class ShapeError(Exception):
    """A request the adapter does not read.

    Args:
        status: 400 for a request that asks for what the adapter does not do,
            422 for a body it cannot read.
        message: What is wrong, in the studio's words: it quotes nothing of the request.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _text(content: object, kinds: frozenset[str]) -> str:
    """Read the text of a message from a string or from a list of text parts.

    Returns:
        The text, its parts joined by line breaks; empty for no content.

    Raises:
        ShapeError: For a part that is not text, or content of another form.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ShapeError(422, "the content of a message must be a text or a list of parts")
    texts: list[str] = []
    for part in content:
        text = part.get("text") if isinstance(part, dict) else None
        if not isinstance(part, dict) or part.get("type") not in kinds or not isinstance(text, str):
            raise ShapeError(
                422, "the atom answers text only: the request has a part that is not text"
            )
        texts.append(text)
    return "\n".join(texts)


def _spoken(items: list[Any], kinds: frozenset[str], what: str) -> list[Said]:
    """Keep what the user and the assistant said, in order.

    System and developer messages, tool calls and their results are left out:
    the atom has its own instructions and reports no tools.

    Returns:
        The texts that are not empty, each with who said it.

    Raises:
        ShapeError: For an item that is not an object, or a text that cannot be read.
    """
    said: list[Said] = []
    for item in items:
        if not isinstance(item, dict):
            raise ShapeError(422, f"every {what} must be an object")
        role = item.get("role")
        if item.get("type", "message") != "message" or role not in _SPEAKERS:
            continue
        text = _text(item.get("content"), kinds)
        if text.strip():
            said.append((role, text))
    return said


def read(route: str, body: object) -> tuple[list[Said], str]:
    """Read a request into the conversation it carries.

    Args:
        route: `CHAT` or `RESPONSES`.
        body: The body of the request, parsed from JSON.

    Returns:
        What was said before, oldest first, and the user message to answer.

    Raises:
        ShapeError: If the request asks for streaming or for a conversation kept
            by the server, or its body cannot be read.
    """
    if not isinstance(body, dict):
        raise ShapeError(422, "the body must be a JSON object")
    if body.get("stream") is True:
        raise ShapeError(400, "streaming is not supported: send the request without stream")
    said: list[Said]
    if route == CHAT:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ShapeError(422, "messages must be a list with at least one message")
        said = _spoken(messages, _CHAT_PARTS, "message")
    else:
        if body.get("previous_response_id") is not None:
            # Ignoring it would lose the conversation without a word.
            raise ShapeError(400, "no conversation is kept here: send all of it in input")
        given = body.get("input")
        if isinstance(given, str):
            said = [("user", given)] if given.strip() else []
        elif isinstance(given, list) and given:
            said = _spoken(given, _RESPONSES_PARTS, "item of input")
        else:
            raise ShapeError(422, "input must be a text or a list with at least one item")
    if not said or said[-1][0] != "user":
        raise ShapeError(422, "the conversation must end with a message of the user")
    return said[:-1], said[-1][1]


def reply(route: str, text: str) -> dict[str, Any]:
    """Write the body that answers a request with a reply.

    The model is always named `chatomon`: the name of the atom does not leave.

    Args:
        route: `CHAT` or `RESPONSES`.
        text: The reply.

    Returns:
        A chat completion or a response, as the route wants it.
    """
    now = int(time.time())
    if route == CHAT:
        message = {"role": "assistant", "content": text}
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": "chatomon",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {
        "type": "message",
        "id": f"msg_{uuid4().hex}",
        "role": "assistant",
        "status": "completed",
        "content": [part],
    }
    return {
        "id": f"resp_{uuid4().hex}",
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": "chatomon",
        "output": [item],
    }


def error(kind: str, message: str) -> dict[str, Any]:
    """Write the body of an answer that is an error.

    Returns:
        `{"error": {"type": ..., "message": ...}}`.
    """
    return {"error": {"type": kind, "message": message}}
