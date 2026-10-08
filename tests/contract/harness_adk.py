"""Contract harness for Google ADK, with a scripted model as the nucleus."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Any

from google.adk.apps import App
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

from chatomon.connectors.adk import ADKConnector
from tests.contract.harness import Call, Harness, Step, ToolCall, lookup

USER = "user-1"


def text_of(content: types.Content | None) -> str:
    """The text of a content, without thoughts."""
    parts = content.parts if content is not None and content.parts else []
    return "".join(part.text for part in parts if part.text and not part.thought)


def _calls_a_tool(content: types.Content | None) -> bool:
    parts = content.parts if content is not None and content.parts else []
    return any(part.function_call or part.function_response for part in parts)


def visible(events: Iterable[Event]) -> list[tuple[str, str]]:
    """User texts and assistant texts (no tool calls), in order."""
    seen: list[tuple[str, str]] = []
    for event in events:
        text = text_of(event.content)
        if text and not _calls_a_tool(event.content):
            seen.append(("user" if event.author == "user" else "assistant", text))
    return seen


def received(request: LlmRequest) -> list[tuple[str, str]]:
    """What a model was sent: user texts and assistant texts, in order."""
    seen: list[tuple[str, str]] = []
    for content in request.contents:
        text = text_of(content)
        if text and not _calls_a_tool(content):
            seen.append(("user" if content.role == "user" else "assistant", text))
    return seen


def reply(events: Iterable[Event]) -> str:
    """The text of the last event that has one."""
    texts = [text_of(event.content) for event in events if not event.partial]
    return next((text for text in reversed(texts) if text), "")


def said(text: str, *, thought: str | None = None) -> LlmResponse:
    """A model response with a text, after a thought when one is given."""
    parts = [types.Part(text=thought, thought=True)] if thought else []
    return LlmResponse(content=types.Content(role="model", parts=[*parts, types.Part(text=text)]))


class ScriptedLlm(BaseLlm):
    """Replies with the scripted steps and records what it received.

    A step is a text, an error to raise, a call to `lookup` or a response to
    give as it is. Asked to stream, it gives a text word by word before the
    whole of it.
    """

    model: str = "scripted"
    script: list[Any] = Field(default_factory=list)
    recorder: list[Any] = Field(default_factory=list)
    streamed: list[bool] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        system = llm_request.config.system_instruction
        self.recorder.append(Call(system=str(system or ""), messages=received(llm_request)))
        self.streamed.append(stream)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, LlmResponse):
            yield step
            return
        if isinstance(step, ToolCall):
            call = types.FunctionCall(name="lookup", args={})
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(function_call=call)])
            )
            return
        if stream:
            for word in step.split(" "):
                chunk = types.Content(role="model", parts=[types.Part(text=f"{word} ")])
                yield LlmResponse(content=chunk, partial=True)
        yield said(step)


class ADKHarness(Harness):
    name = "adk"

    def __init__(self) -> None:
        self._models: list[ScriptedLlm] = []
        self._runners: list[tuple[App, Runner]] = []

    @property
    def calls(self) -> list[Call]:
        return [call for model in self._models for call in model.recorder]

    def connector(
        self,
        script: Sequence[Step],
        *,
        tools: bool = False,
        judge_decisions: Sequence[dict[str, Any]] = (),
    ) -> ADKConnector:
        model = ScriptedLlm(script=list(script))
        self._models.append(model)
        decisions = list(judge_decisions)
        kwargs: dict[str, Any] = {"tools": [lookup]} if tools else {}
        return ADKConnector(
            model,
            resolve_model=lambda _id: ScriptedLlm(script=[json.dumps(decisions.pop(0))]),
            **kwargs,
        )

    def runner(self, app: App) -> Runner:
        """The runner of an app, with its sessions in memory; one per app."""
        for known, runner in self._runners:
            if known is app:
                return runner
        runner = Runner(app=app, session_service=InMemorySessionService(), auto_create_session=True)
        self._runners.append((app, runner))
        return runner

    @staticmethod
    def _message(text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part(text=text)])

    async def ask(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        events = [
            event
            async for event in self.runner(agent).run_async(
                user_id=USER, session_id=conversation, new_message=self._message(message)
            )
        ]
        return reply(events)

    def ask_sync(self, agent: Any, message: str, *, conversation: str = "c1") -> str:
        events = self.runner(agent).run(
            user_id=USER, session_id=conversation, new_message=self._message(message)
        )
        return reply(list(events))

    def history(self, agent: Any, *, conversation: str = "c1") -> list[tuple[str, str]]:
        runner = self.runner(agent)
        service = runner.session_service
        assert isinstance(service, InMemorySessionService)
        session = service.sessions.get(runner.app_name, {}).get(USER, {}).get(conversation)
        return visible(session.events) if session is not None else []
