"""What the adapter answers, to whom, and what it refuses."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from starlette.types import Message

from chatomon.studio._spectral import _adapter as adapter
from chatomon.studio._spectral._adapter import Unanswered, create_adapter

SECRET = "s3cret-of-this-connection"
PORT = 49231
BASE = f"http://127.0.0.1:{PORT}"
RESPONSES = f"/{SECRET}/v1/responses"
CHAT = f"/{SECRET}/v1/chat/completions"
HELLO = {"input": "hello"}


class Atom:
    """Stands in for the atom: records what it is asked and answers as it is told."""

    def __init__(self) -> None:
        self.asked: list[tuple[list[tuple[str, str]], str]] = []
        self.rejected = 0
        self.reply: str | Exception = "Hi"
        self.gate: asyncio.Event | None = None
        self.running = 0
        self.most = 0

    async def answer(self, history: list[Any], message: str) -> str:
        self.asked.append((list(history), message))
        self.running += 1
        self.most = max(self.most, self.running)
        try:
            if self.gate is not None:
                await self.gate.wait()
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply
        finally:
            self.running -= 1

    def count_rejected(self) -> None:
        self.rejected += 1


@pytest.fixture
def atom() -> Atom:
    return Atom()


@pytest.fixture
async def client(atom: Atom):
    app = create_adapter(secret=SECRET, port=PORT, answer=atom.answer, rejected=atom.count_rejected)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as http:
        yield http


async def test_a_response_request_is_answered_with_the_reply(client, atom):
    answer = await client.post(RESPONSES, json=HELLO)
    assert answer.status_code == 200
    assert answer.json()["output"][0]["content"][0]["text"] == "Hi"
    assert answer.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in answer.headers
    assert atom.asked == [([], "hello")]


async def test_a_chat_request_is_answered_with_the_reply(client, atom):
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "again"},
    ]
    answer = await client.post(CHAT, json={"messages": messages})
    assert answer.status_code == 200
    assert answer.json()["choices"][0]["message"] == {"role": "assistant", "content": "Hi"}
    assert atom.asked == [([("user", "hello"), ("assistant", "one")], "again")]


@pytest.mark.parametrize(
    "path",
    [
        "/wrong-secret/v1/responses",
        "/v1/responses",
        f"/{SECRET}",
        f"/{SECRET}/v1/models",
        f"/{SECRET}/v1/responses/extra",
        f"/{SECRET[:-1]}/v1/responses",
        "/",
    ],
)
async def test_a_path_without_the_key_or_with_another_route_is_not_found(client, atom, path):
    answer = await client.post(path, json=HELLO)
    assert answer.status_code == 404
    assert answer.json() == {"error": {"type": "NotFound", "message": "Nothing answers here."}}
    assert atom.asked == []


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "OPTIONS"])
async def test_only_post_is_answered(client, atom, method):
    answer = await client.request(method, RESPONSES)
    assert answer.status_code == 404
    assert atom.asked == []


@pytest.mark.parametrize("host", ["localhost:49231", "127.0.0.1", "evil.example", "127.0.0.1:1"])
async def test_another_host_is_refused(client, atom, host):
    answer = await client.post(RESPONSES, json=HELLO, headers={"host": host})
    assert (answer.status_code, answer.json()["error"]["type"]) == (400, "BadHost")
    assert atom.asked == []


@pytest.mark.parametrize("origin", [BASE, "https://evil.example", "null"])
async def test_a_request_that_names_an_origin_is_refused(client, atom, origin):
    answer = await client.post(RESPONSES, json=HELLO, headers={"origin": origin})
    assert (answer.status_code, answer.json()["error"]["type"]) == (403, "BadOrigin")
    assert atom.asked == []


async def test_a_body_that_is_not_sent_as_json_is_refused(client, atom):
    answer = await client.post(
        RESPONSES, content=json.dumps(HELLO), headers={"content-type": "text/plain"}
    )
    assert (answer.status_code, answer.json()["error"]["type"]) == (415, "UnsupportedMediaType")
    assert atom.asked == []


async def test_json_with_a_charset_is_read(client):
    headers = {"content-type": "application/json; charset=utf-8"}
    answer = await client.post(RESPONSES, content=json.dumps(HELLO), headers=headers)
    assert answer.status_code == 200


async def test_a_body_that_is_too_large_is_refused(client, atom, monkeypatch):
    monkeypatch.setattr(adapter, "MAX_BODY", 64)
    answer = await client.post(RESPONSES, json={"input": "x" * 100})
    assert (answer.status_code, answer.json()["error"]["type"]) == (413, "TooLarge")
    assert atom.asked == []


@pytest.mark.parametrize("raw", [b"{not json", b"\xff\xfe", b""])
async def test_a_body_that_is_not_json_is_refused(client, atom, raw):
    headers = {"content-type": "application/json"}
    answer = await client.post(RESPONSES, content=raw, headers=headers)
    assert (answer.status_code, answer.json()["error"]["type"]) == (400, "BadJSON")
    assert (atom.asked, atom.rejected) == ([], 0)


async def test_a_request_the_adapter_does_not_read_is_refused_and_counted(client, atom):
    answer = await client.post(RESPONSES, json={"input": "hello", "stream": True})
    assert answer.status_code == 400
    assert answer.json()["error"] == {
        "type": "BadRequest",
        "message": "streaming is not supported: send the request without stream",
    }
    answer = await client.post(CHAT, json={"messages": []})
    assert answer.status_code == 422
    assert (atom.asked, atom.rejected) == ([], 2)


async def test_a_turn_with_no_reply_is_answered_with_its_error(client, atom):
    atom.reply = Unanswered(502, "AtomFailed", "The atom failed with RuntimeError.")
    answer = await client.post(RESPONSES, json=HELLO)
    assert answer.status_code == 502
    assert answer.json() == {
        "error": {"type": "AtomFailed", "message": "The atom failed with RuntimeError."}
    }
    assert answer.headers["cache-control"] == "no-store"


async def test_an_error_nobody_expected_says_nothing_of_itself(client, atom):
    atom.reply = RuntimeError("the key is sk-123")
    answer = await client.post(RESPONSES, json=HELLO)
    assert answer.status_code == 500
    assert answer.json() == {"error": {"type": "AdapterFailed", "message": "The adapter failed."}}
    assert "sk-123" not in answer.text


async def test_four_requests_run_at_a_time_and_the_others_wait(client, atom):
    atom.gate = asyncio.Event()
    sent = [asyncio.ensure_future(client.post(RESPONSES, json=HELLO)) for _ in range(7)]
    await asyncio.sleep(0.1)
    assert (atom.running, len(atom.asked)) == (4, 4)
    atom.gate.set()
    answers = await asyncio.gather(*sent)
    assert [answer.status_code for answer in answers] == [200] * 7
    assert (atom.most, len(atom.asked)) == (4, 7)


def a_post() -> dict[str, Any]:
    """The scope of a request the adapter answers, as the server gives it."""
    return {
        "type": "http",
        "method": "POST",
        "path": RESPONSES,
        "raw_path": RESPONSES.encode(),
        "query_string": b"",
        "headers": [
            (b"host", f"127.0.0.1:{PORT}".encode()),
            (b"content-type", b"application/json"),
        ],
        "scheme": "http",
        "server": ("127.0.0.1", PORT),
        "client": ("127.0.0.1", 50000),
        "http_version": "1.1",
    }


async def test_a_caller_that_goes_away_stops_its_turn(atom):
    atom.gate = asyncio.Event()
    app = create_adapter(secret=SECRET, port=PORT, answer=atom.answer, rejected=atom.count_rejected)
    body = json.dumps(HELLO).encode()
    inbox: asyncio.Queue[Message] = asyncio.Queue()
    await inbox.put({"type": "http.request", "body": body, "more_body": False})
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    call = asyncio.ensure_future(app(a_post(), inbox.get, send))
    await asyncio.sleep(0.05)
    assert atom.running == 1
    await inbox.put({"type": "http.disconnect"})
    await asyncio.wait_for(call, timeout=2)
    assert (atom.running, sent) == (0, [])


async def test_a_caller_that_goes_away_before_its_body_is_no_error(atom):
    app = create_adapter(secret=SECRET, port=PORT, answer=atom.answer, rejected=atom.count_rejected)
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(a_post(), receive, send)
    assert (atom.asked, atom.rejected, sent) == ([], 0, [])


async def test_a_websocket_is_closed(atom):
    app = create_adapter(secret=SECRET, port=PORT, answer=atom.answer, rejected=atom.count_rejected)
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "websocket.connect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app({"type": "websocket", "path": RESPONSES, "headers": []}, receive, send)
    assert sent == [{"type": "websocket.close", "code": 1008}]
