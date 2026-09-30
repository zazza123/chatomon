"""Contract harness for Microsoft Agent Framework using a scripted chat client."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agent_framework import (
    AgentSession,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
)

from chatomon.connectors.maf import MAFConnector
from tests.contract.harness import Call, Harness, Step, ToolCall, lookup


def visible(messages: Sequence[Message]) -> list[tuple[str, str]]:
    return [
        (message.role, message.text)
        for message in messages
        if message.role in {"user", "assistant"}
        and message.text
        and not any(content.type == "function_call" for content in message.contents)
    ]


class ScriptedMAFClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """MAF-native client with the same middleware and tool layers as production clients."""

    def __init__(self, script: Sequence[Step], recorder: list[Call]) -> None:
        self.script = list(script)
        self.recorder = recorder
        self.options_seen: list[dict[str, Any]] = []
        self.kwargs_seen: list[dict[str, Any]] = []
        super().__init__()

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        self.recorder.append(
            Call(system=str(options.get("instructions", "")), messages=visible(messages))
        )
        self.options_seen.append(dict(options))
        self.kwargs_seen.append(dict(kwargs))
        step = self.script.pop(0)

        async def respond() -> ChatResponse:
            if isinstance(step, Exception):
                raise step
            if isinstance(step, ToolCall):
                return ChatResponse(
                    messages=[
                        Message(
                            "assistant",
                            [Content.from_function_call("call-1", "lookup", arguments={})],
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return ChatResponse(messages=[Message("assistant", [step])], finish_reason="stop")

        if not stream:
            return respond()

        async def updates():
            response = await respond()
            for message in response.messages:
                yield ChatResponseUpdate(
                    role="assistant",
                    contents=message.contents,
                    finish_reason=(
                        "tool_calls"
                        if any(content.type == "function_call" for content in message.contents)
                        else "stop"
                    ),
                )

        return self._build_response_stream(
            updates(), response_format=options.get("response_format")
        )


class MAFHarness(Harness):
    name = "maf"
    supports_sync = False

    def __init__(self) -> None:
        self._calls: list[Call] = []
        self._sessions: dict[str, AgentSession] = {}

    @property
    def calls(self) -> list[Call]:
        return self._calls

    def connector(
        self,
        script: Sequence[Step],
        *,
        tools: bool = False,
        judge_decisions: Sequence[dict[str, Any]] = (),
    ) -> MAFConnector:
        decisions = list(judge_decisions)
        kwargs: dict[str, Any] = {"tools": [lookup]} if tools else {}
        return MAFConnector(
            ScriptedMAFClient(script, self._calls),
            resolve_client=lambda _id: ScriptedMAFClient([json.dumps(decisions.pop(0))], []),
            **kwargs,
        )

    async def ask(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        session = self._sessions.setdefault(
            conversation, agent.create_session(session_id=conversation)
        )
        response = await agent.run(message, session=session)
        return response.text

    def ask_sync(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        raise NotImplementedError("MAF has no synchronous native run API")

    def history(self, agent: Any, *, conversation: str = "c1") -> list[tuple[str, str]]:
        session = self._sessions.get(conversation)
        if session is None:
            return []
        state = session.state.get("in_memory", {})
        return visible(state.get("messages", []))
