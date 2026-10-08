# Google ADK

## Overview

`ADKConnector` (`src/chatomon/connectors/adk.py`) builds an atom into a `google.adk.apps.App`. The nucleus is the app's root `LlmAgent`; chatomon's checks are the app's first plugin and a wrapper around the nucleus model. It needs the `adk` extra (`pip install 'chatomon[adk]'`), which installs Google ADK 2.10 or later.

## Building the App

In ADK an agent does not run on its own: a `Runner` runs it, and plugins belong to the `App`. `Atom.build()` therefore returns the `App`, the object that carries chatomon's checks wherever ADK loads it: `Runner(app=app, ...)`, `InMemoryRunner(app=app)`, `adk web` and `adk run`. It is also what `adk deploy` loads from an agent module that exports `app`.

```{.python test="skip"}
import asyncio

from google.adk.runners import InMemoryRunner

from chatomon import Atom, TurnReport
from chatomon.connectors.adk import ADKConnector

reports: list[TurnReport] = []
app = Atom(
    name="speedypack",
    instructions="You are the customer assistant of SpeedyPack.",
    connector=ADKConnector("gemini-3.5-flash"),
    on_turn=reports.append,
).build()


async def main() -> None:
    runner = InMemoryRunner(app=app)
    events = await runner.run_debug("Where is my parcel?", quiet=True)
    print(events[-1].content.parts[0].text)
    print(reports[-1].final_action, reports[-1].conversation_id)


asyncio.run(main())
```

Every extra keyword argument of the connector goes to `LlmAgent` (`tools`, `generate_content_config`, callbacks, `output_key`, ...). What the `App` takes goes in `app_kwargs` (`plugins`, `context_cache_config`, `resumability_config`, ...). Your own plugins run after chatomon's.

The agent runs only inside the app that `build()` returned. A `Runner` given the agent alone has no plugin of chatomon's, and the nucleus then raises `ChatomonConfigError` instead of calling the model unchecked. For the same reason the nucleus answers only the model calls ADK makes for a turn: code of yours that calls the agent's model itself gets that error, and should call your own model object.

The atom's instructions and knowledge reach the model as written. ADK reads `{name}` in an instruction string as a variable of the session state; chatomon gives the text in a way ADK does not interpret, so braces in a knowledge base stay braces.

### Arguments chatomon Refuses

Each of these raises `ChatomonConfigError` with a message that says what to do instead.

| Argument | Why |
|----------|-----|
| `name`, `instruction`; `name` and `root_agent` in `app_kwargs` | chatomon sets them from the atom |
| `static_instruction`, `global_instruction` | The instructions come from the atom, and `static_instruction` would move them out of the system instruction |
| `output_schema` | The nucleus replies with text, which is what the orbitals check |
| `sub_agents` | After a transfer a sub-agent answers with a model of its own, which the output orbitals do not see. Give the other agents to the nucleus as tools (`AgentTool`) |
| `mode` other than `chat` | An atom is a chat agent, the mode a root agent has by default; in the others the result leaves through a tool call |
| `events_compaction_config` without a `summarizer` | ADK would summarise with the nucleus, outside a turn. Give the configuration a summarizer with a model of its own |

Two more checks happen when the atom is built. The atom's name must be a name ADK accepts for an agent: a Python identifier other than `user`. A model given by name must be one ADK's registry resolves; pass a `BaseLlm` object for any other model.

## Hooks

| Hook | What chatomon does there |
|------|--------------------------|
| Plugin, `on_user_message_callback` | Runs the input chain; replaces a blocked message before ADK stores it |
| Plugin, `before_run_callback` | Ends the run with the reply when an input orbital has it: the agent, its callbacks and the model are not reached |
| The nucleus model | Adds the guidance, calls your model, runs the output chain and the regenerations |
| Plugin, `after_run_callback` and `on_run_error_callback` | Finishes the turn, or closes it as `aborted` |

Both `Runner.run_async` and the synchronous `Runner.run` are supported.

## Conversations

The ADK session keeps the conversation, and its id is the conversation id of every report. The input orbitals check every text part of a message, each on a line of its own and whatever the part is flagged as; what else a message carries, such as an image, is not checked. A blocked message is replaced by the placeholder before ADK appends it to the session, so the original text is never stored, whatever session service the runner has. The plugins that come after chatomon's do not receive it either. For a turn an input orbital answers they are not asked `before_run_callback`, which chatomon answers first; they still receive `after_run_callback`.

The history the orbitals receive is read from the session. It leaves out the turns a rewind took back, as ADK does for the model. It keeps the full text of the turns ADK compacted, which the model sees as a summary.

A reply written by an input orbital is stored as a turn of the agent. ADK signs the content that ends a run early as `model`, and in the turns that follow it would quote that content to the agent as the words of another agent; chatomon signs it with the agent's name.

## Streaming

When the atom has output orbitals the nucleus is called without streaming, whatever the run asks for. A caller that sets `StreamingMode.SSE` receives each reply whole, in one final event, once it is checked. With input orbitals only, partial events pass as the model produces them.

Live runs (`run_live`, bidirectional audio and video) are refused with `ChatomonConfigError`: they carry no text turn the orbitals can check.

## Tools and Other Agents

Tools are declared on the agent as usual. An agent used as a tool (`AgentTool`) is a tool like any other: its inner run is not a turn of the atom, and the orbitals do not check it. The agent of another atom cannot be such a tool: it runs only inside its own app.

With a planner (`planner`), the output orbitals check the whole text the model writes, the plan with the answer; ADK separates the plan afterwards. With a code executor (`code_executor`), a step in which the model writes code is text, so the output orbitals check it like a reply, and one they block ends the turn before the code runs. The result of the code is not checked, as the result of a tool is not.

## Judges and Model Resolution

A `JudgeOrbital` without its own connector makes one direct call to an ADK model, with no agent and no session. By default its model id is a name ADK's registry resolves, such as the name of a Gemini model; `resolve_model` maps the id to a `BaseLlm` of your choice or to another name. The judge never goes through the nucleus.

## Observability

chatomon's spans join ADK's trace with no configuration: `chatomon.input` nests under ADK's `invocation` span and `chatomon.output` under its `call_llm` span.

ADK also traces and logs by itself, under its own rules:

- **ADK's spans hold content by default.** The system instruction, the messages and the accepted reply are attributes of ADK's spans unless `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false`. The atom's `capture_content` governs chatomon's reports, not ADK's spans.
- **What chatomon stopped does not reach ADK's spans.** A blocked message is replaced before ADK sees it. ADK sees one model call, with the reply that passed, so a rejected reply is not there. The request holds the guidance only while your model works, and ADK records it when a response is back: in ADK's trace the request appears without the `<guidance>` block, and the turn report has the guidance when `capture_content` allows it.
- **A model class logs what it handles.** With its logger at `DEBUG`, ADK's `Gemini` class logs every request and response, those of a rejected attempt and the guidance included. An instrumentation of the client under your model records the same calls.
- **ADK counts the tokens of the reply that passed.** ADK sees one model call for each step, so the usage on its events and in its metrics is that of the last attempt. The attempts the output orbitals rejected are not counted there.
- **ADK logs a run that fails.** When your model or a tool raises, ADK logs the error with its traceback through its own loggers and puts its message on its spans, before the exception reaches your code.

## Failures, Cancellation and Paused Runs

When the model or a tool raises, the turn is closed as `aborted` and the exception reaches your code unchanged (see [Failures](index.md#failures)). ADK keeps the message of the failed turn in the session, so the next turn finds it in the history. A run that is cancelled is closed as `aborted` too, wherever the cancellation arrives. While the agent works (an orbital, the model call, a tool) the report is there before the cancellation reaches your code. While ADK itself works (storing an event, a plugin of yours) ADK tells no plugin: chatomon closes the turn when Python's garbage collector frees the run. That report comes after the cancellation reached your code, within the next turns of a process that keeps working, and it is lost if the process stops first.

A run that pauses for a long-running tool, a confirmation or credentials ends without a reply, and its turn is closed as `aborted`. The answer arrives as a message with no text and resumes the run. chatomon then opens a new turn and runs the input orbitals again on the message of that run, so the model is never called without the checks. One exchange with a pause leaves two reports, and the input orbitals run twice on the same message. That message is already stored, so a second check that asks to hide it cannot take it out of the session; and what the answer itself carries is not checked, as the result of a tool is not. A run that ADK starts again by its id, with no new message, is treated the same way.

## Limits

- **Replies your callbacks write are not checked.** chatomon checks what the nucleus model writes. A callback that returns the reply itself (`before_model_callback`, `on_model_error_callback`, `before_agent_callback`, `after_agent_callback`) or rewrites it afterwards (`after_model_callback`, the `on_event_callback` of a plugin) is your code speaking: the output orbitals do not see that text. A turn whose reply never came from the nucleus is reported as `aborted`. In the same way, a plugin of yours that returns a new user message from `on_user_message_callback` replaces the message after the input orbitals checked it.
- **Thoughts are not checked.** What a thinking model returns as a thought is not part of the reply: the output orbitals do not see it, and it travels with a reply they accepted.
- **ADK does not see the class of your model.** The agent's model is chatomon's wrapper, which forwards the name and the declared capabilities of yours. Where ADK looks at the class instead, it does not find it: Gemini's Interactions API is not chained from one call to the next, and for Anthropic, LiteLLM and OpenAI Responses models ADK does not keep its own fallback ids of tool calls, which matters only for a provider that returns calls without an id.
- **One agent.** An atom is one `LlmAgent` with its tools. Sub-agents and ADK's workflow agents are not supported.
- **ADK 2.10 or later.** Earlier 2.x versions call the model even when a plugin has already answered, or store that answer as the words of another agent. The 1.x line is not supported.

## Related Documentation

- [Connectors](index.md) — shared connector behavior
- [PydanticAI](pydantic_ai.md) — the PydanticAI connector
- [LangChain / LangGraph](langchain.md) — the LangChain connector
- [Microsoft Agent Framework](maf.md) — the MAF connectors
- [Observability and Audit](../observability.md) — the spans and reports of a turn
- [Frameworks in the Studio](../studio/frameworks.md) — trying an ADK atom in the local studio
- [API Reference: `chatomon.connectors.adk`](../reference/connectors_adk.md) — every parameter and an example

---
**Last Updated:** October 2026
**Status:** Implemented
