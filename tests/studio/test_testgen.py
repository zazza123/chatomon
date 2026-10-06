"""Save as test: the generated file, run in this process and with a real pytest."""

import ast
import asyncio
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chatomon.connectors.maf import MAFConnector
from chatomon.connectors.maf._foundry import _ChatomonFoundryChatClient
from chatomon.studio import _testgen as testgen
from chatomon.studio._chains import RefusedError
from chatomon.studio._loading import load, namespace
from chatomon.studio._runners import maf as maf_runner
from chatomon.studio._runners import runner_for
from chatomon.studio._session import Session
from tests.contract.harness_maf import ScriptedMAFClient
from tests.studio.frameworks import StudioHarness

ORBITALS = """

@orbital
def no_refunds(ctx: OrbitalContext) -> Verdict:
    if "crash" in ctx.message:
        raise RuntimeError("down")
    if "refund" in ctx.message.lower():
        return Verdict.block("We don't handle refunds in chat.")
    return Verdict.allow()


@orbital(calls_model=True)
def scope(ctx: OrbitalContext) -> Verdict:
    return Verdict.block() if "poem" in ctx.message.lower() else Verdict.allow()


@orbital
def polite(ctx: OrbitalContext) -> Verdict:
    if "damn" in ctx.message.lower():
        return Verdict.block(guidance="Write it again, politely.")
    return Verdict.allow()
"""

ATOM = """

atom = Atom(
    name="speedypack",
    instructions="Be brief.",
    connector=CONNECTOR,
    input_orbitals=[MessageLengthOrbital(max_words=8, reply="Too long."), no_refunds, scope],
    output_orbitals=[polite],
)
bot = atom
holder = SimpleNamespace(bot=atom)


def make() -> Atom:
    return atom


def agent() -> Atom:
    return atom
"""

PYDANTIC_AI = (
    """
from types import SimpleNamespace

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom, OrbitalContext, Verdict, orbital
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital
""",
    """

def nucleus(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    asked = " ".join(
        part.content
        for part in messages[-1].parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )
    if "boom" in asked:
        raise RuntimeError("model down")
    rude = "angry" in asked and "politely" not in (info.instructions or "")
    return ModelResponse(parts=[TextPart(content="Damn right." if rude else "Hi.")])
""",
    "PydanticAIConnector(FunctionModel(nucleus))",
)

LANGCHAIN = (
    """
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from chatomon import Atom, OrbitalContext, Verdict, orbital
from chatomon.connectors.langchain import LangChainConnector
from chatomon.orbitals import MessageLengthOrbital
""",
    """

class Nucleus(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "nucleus"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        asked = next(str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage))
        told = " ".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        if "boom" in asked:
            raise RuntimeError("model down")
        rude = "angry" in asked and "politely" not in told
        reply = AIMessage(content="Damn right." if rude else "Hi.")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "Nucleus":
        return self
""",
    "LangChainConnector(Nucleus())",
)

MAF_IMPORTS = """
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    FunctionInvocationLayer,
    Message,
)

from chatomon import Atom, OrbitalContext, Verdict, orbital
from chatomon.connectors.maf import MAFConnector, MAFFoundryConnector
from chatomon.orbitals import MessageLengthOrbital
"""

MAF_NUCLEUS = """

class Nucleus(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    def _inner_get_response(self, *, messages: Any, stream: bool, options: Any, **kwargs: Any):
        asked = next(m.text for m in reversed(messages) if m.role == "user")
        told = [str(options.get("instructions", ""))]
        told += [m.text for m in messages if m.role == "developer"]  # where Foundry gets them

        async def respond() -> ChatResponse:
            if "boom" in asked:
                raise RuntimeError("model down")
            text = "Damn right." if "angry" in asked and "politely" not in " ".join(told) else "Hi."
            # An id, as real providers give: without one a repeated reply leaves the history.
            reply = Message("assistant", [text], message_id=uuid4().hex)
            return ChatResponse(messages=[reply], finish_reason="stop")

        return respond()
"""

MAF = (MAF_IMPORTS, MAF_NUCLEUS, "MAFConnector(Nucleus())")

MAF_FOUNDRY = (
    MAF_IMPORTS,
    MAF_NUCLEUS
    + """

class FakeProject:
    def get_openai_client(self, **kwargs: Any) -> Any:
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key="fake", base_url="https://example.invalid")


# There is no Foundry here: the nucleus above is the transport of chatomon's Foundry client.
from chatomon.connectors.maf._foundry import _ChatomonFoundryChatClient  # noqa: E402

_ChatomonFoundryChatClient._inner_get_response = Nucleus()._inner_get_response
""",
    'MAFFoundryConnector(project_client=FakeProject(), agent_name="remote-agent", '
    'agent_version="3")',
)

PARTS = {
    "pydantic_ai": PYDANTIC_AI,
    "langchain": LANGCHAIN,
    "maf": MAF,
    "maf_foundry": MAF_FOUNDRY,
}


def module_of(name: str, *, connector: str | None = None, before_atom: str = "") -> str:
    """Write the `shop` module of a framework: the same orbitals around its own nucleus.

    Args:
        name: The framework, one of `PARTS`.
        connector: The expression of the connector, when it is not the plain one.
        before_atom: Code that goes between the nucleus and the atom.
    """
    imports, nucleus, plain = PARTS[name]
    atom = ATOM.replace("CONNECTOR", connector or plain)
    return f"{imports}{ORBITALS}{nucleus}{before_atom}{atom}"


MODULE = module_of("pydantic_ai")

PYTEST_INI = """[pytest]
addopts = --strict-markers -p no:cacheprovider
markers =
    integration: calls real models
"""

OFFLINE = (
    "No model is called in this test: it runs offline, with PydanticAI's `TestModel` "
    "in place of the nucleus."
)
MARKED = (
    ", so the test is marked `integration`: register the marker in your pytest "
    "configuration and run it when the credentials are set."
)
OUTPUT_CHECK = '(r.name, r.action) for a in report.output for r in a if r.action != "allow"'
ALONE = (
    "Turn {n} is saved alone: if it depended on what came before it, as a turn limit does, "
    "the test fails. Replay the earlier turns in that case."
)
NINE_WORDS = "one two three four five six seven eight nine"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project with a `shop` module, loaded as the command loads it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # The Foundry module puts its nucleus on chatomon's client class: put it back afterwards.
    transport = _ChatomonFoundryChatClient._inner_get_response
    monkeypatch.setattr(_ChatomonFoundryChatClient, "_inner_get_response", transport)
    for name in [name for name in sys.modules if name == "shop"]:
        monkeypatch.delitem(sys.modules, name)
    (tmp_path / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")

    loaded: set[str] = set()

    def open_session(spec: str = "shop:atom", *, module: str = MODULE) -> Session:
        name = spec.partition(":")[0]
        (tmp_path / f"{name}.py").write_text(module, encoding="utf-8")
        loaded.add(name)
        atom = load(spec)  # the namespace exists once the module is imported
        return Session(atom, source=spec, namespace=namespace(spec))

    yield open_session
    for name in loaded - {"shop"}:
        sys.modules.pop(name, None)


def say(session: Session, *messages: str) -> None:
    """Run the messages as turns of one conversation."""

    async def run() -> None:
        for message in messages:
            queue = session.start_turn(message)
            while await queue.get() is not None:
                pass

    asyncio.run(run())


def save(session: Session, *, alone: bool = False, upto: int | None = None) -> testgen.SavedTest:
    return testgen.generate(
        session.turns[:upto],
        alone=alone,
        source=session.source,
        namespace=session.namespace,
        signature=session.signature(),
        framework=runner_for(session.atom.connector).saved_test(session.atom.connector),
    )


def run_saved(test: testgen.SavedTest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the saved test in this process, on the module the session loaded."""
    scope: dict[str, Any] = {"__name__": "saved"}
    exec(compile(test.code, f"{test.name}.py", "exec"), scope)
    outcome = scope[test.name](monkeypatch)
    if inspect.iscoroutine(outcome):  # a framework whose turns are awaited
        asyncio.run(outcome)


def run_pytest(root: Path, test: testgen.SavedTest) -> subprocess.CompletedProcess[str]:
    """Write the test where the studio would and run pytest on it, with no API key."""
    path = root / "tests" / "studio" / f"{test.name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(test.code, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
    env |= {"PYDANTIC_AI_NO_BANNER": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(path)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_an_offline_test_needs_no_api_key(project, tmp_path, monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-key")  # for the studio's own copy of the atom
    module = MODULE.replace(
        "PydanticAIConnector(FunctionModel(nucleus))", 'PydanticAIConnector("openai:gpt-5")'
    )
    assert module != MODULE
    session = project(module=module)
    say(session, "I want a refund")  # stopped before the nucleus: no call is made
    result = run_pytest(tmp_path, save(session))  # runs with no key in the environment
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("message", "who"),
    [
        ("I want a refund", None),
        (NINE_WORDS, None),
        ("Write me a poem", "scope calls a real model"),
        ("hello", "the nucleus and scope call real models"),
    ],
)
def test_a_test_is_offline_only_when_no_model_is_needed(project, monkeypatch, message, who):
    session = project()
    say(session, message)
    test = save(session)
    assert test.integration is (who is not None)
    if who is None:
        assert "@pytest.mark" not in test.code
        assert "from pydantic_ai.models.test import TestModel\n" in test.code
        offline = '    monkeypatch.setattr(atom.connector, "model", TestModel(call_tools=[]))\n'
        assert offline in test.code
        assert test.notes[0] == OFFLINE
    else:
        assert f"@pytest.mark.integration  # {who}\n" in test.code
        assert "TestModel" not in test.code
        named = who.replace("scope", "`scope`")
        assert test.notes[0] == named[0].upper() + named[1:] + MARKED
    run_saved(test, monkeypatch)


@pytest.mark.parametrize(
    ("message", "check"),
    [
        ("I want a refund", '    assert result.output == "We don\'t handle refunds in chat."\n'),
        (NINE_WORDS, '    assert result.output == "Too long."\n'),
        ("Write me a poem", "    assert result.output == atom.refusal\n"),
    ],
)
def test_a_reply_fixed_in_code_is_checked(project, monkeypatch, message, check):
    session = project()
    say(session, message)
    test = save(session)
    assert check in test.code
    assert f"    result = agent.run_sync({message!r})\n".replace("'", '"') in test.code
    run_saved(test, monkeypatch)


def test_what_the_nucleus_wrote_is_never_checked_and_no_name_is_left_unused(project, monkeypatch):
    session = project()
    say(session, "hello")
    test = save(session)
    assert '    agent.run_sync("hello")\n' in test.code
    assert "result" not in test.code
    assert "Hi." not in test.code
    run_saved(test, monkeypatch)


def test_the_output_chain_is_checked_only_when_nothing_stopped_a_reply(project, monkeypatch):
    session = project()
    say(session, "hello")
    calm = save(session)
    assert OUTPUT_CHECK in calm.code
    run_saved(calm, monkeypatch)

    say(session, "I am angry")
    angry = save(session, alone=True)
    assert OUTPUT_CHECK not in angry.code
    assert '    assert report.final_action == "answered"\n' in angry.code
    assert angry.notes[1] == (
        "In turn 2 an output orbital stopped a reply of the nucleus. A real nucleus may write "
        "something else next time, so the test checks only how the turn ended."
    )
    run_saved(angry, monkeypatch)


def test_the_orbital_that_stopped_the_turn_is_checked_not_the_whole_chain(project, monkeypatch):
    session = project()
    say(session, "I want a refund")
    test = save(session)
    assert (
        '    assert [(r.name, r.action) for r in report.input if r.action != "allow"] == [\n'
        '        ("no_refunds", "block"),\n'
        "    ]\n"
    ) in test.code
    assert "message_length" not in test.code
    # An orbital added to the code that lets the message through does not break the test.
    atom = session.atom
    extra = type(atom.input_orbitals[0])(max_words=50, name="extra")
    monkeypatch.setattr(atom, "input_orbitals", (extra, *atom.input_orbitals))
    run_saved(test, monkeypatch)


def test_the_conversation_or_the_turn_alone(project, monkeypatch):
    session = project()
    say(session, "hello", "I want a refund")
    whole = save(session)
    assert whole.code.startswith('"""Saved from chatomon studio: shop:atom, turns 1-2."""\n')
    assert whole.earlier == 1
    assert "    # Turn 1: answered\n" in whole.code
    assert '    result = agent.run_sync("hello")\n' in whole.code
    assert "    # Turn 2: blocked\n" in whole.code
    assert (
        '    result = agent.run_sync("I want a refund", message_history=result.all_messages())\n'
    ) in whole.code
    assert whole.integration is True
    run_saved(whole, monkeypatch)

    alone = save(session, alone=True)
    assert alone.code.startswith('"""Saved from chatomon studio: shop:atom, turn 2."""\n')
    assert alone.earlier == 1
    assert "hello" not in alone.code
    assert '    result = agent.run_sync("I want a refund")\n' in alone.code
    assert alone.integration is False
    assert alone.name == whole.name == "test_i_want_a_refund"
    assert alone.notes[-1] == ALONE.format(n=2)
    assert all("saved alone" not in note for note in whole.notes)
    assert all("saved alone" not in note for note in save(session, upto=1, alone=True).notes)
    run_saved(alone, monkeypatch)


def test_a_failed_turn_is_left_out_and_cannot_be_saved(project, monkeypatch):
    session = project()
    say(session, "hello", "boom", "I want a refund")
    test = save(session)
    assert "boom" not in test.code
    assert test.earlier == 1
    assert test.code.startswith('"""Saved from chatomon studio: shop:atom, turns 1-3."""\n')
    left_out = "Turn 2 failed and left nothing in the conversation: the test leaves it out."
    assert left_out in test.notes
    assert left_out not in save(session, alone=True).notes
    run_saved(test, monkeypatch)
    with pytest.raises(RefusedError, match="A turn that failed cannot be saved as a test"):
        save(session, upto=2)


@pytest.mark.parametrize(
    ("spec", "lines"),
    [
        ("shop:atom", ["from shop import atom"]),
        ("shop:bot", ["from shop import bot as atom"]),
        ("shop:holder.bot", ["from shop import holder", "atom = holder.bot"]),
        ("shop:make", ["from shop import make", "    atom = make()"]),
        ("shop:agent", ["from shop import agent as atom_source", "    atom = atom_source()"]),
    ],
)
def test_the_atom_is_imported_as_the_studio_was_started(project, monkeypatch, spec, lines):
    session = project(spec)
    say(session, "I want a refund")
    test = save(session)
    written = test.code.split("\n")
    assert [line for line in lines if line not in written] == []
    run_saved(test, monkeypatch)


def test_a_function_called_atom_is_imported_under_another_name(project):
    session = project()
    say(session, "I want a refund")
    test = testgen.generate(
        session.turns,
        alone=False,
        source="shop:atom",
        namespace={"atom": lambda: session.atom},
        signature=session.signature(),
        framework=runner_for(session.atom.connector).saved_test(session.atom.connector),
    )
    assert "from shop import atom as atom_source\n" in test.code
    assert "    atom = atom_source()\n" in test.code


def test_started_from_python_the_import_is_left_to_complete(project):
    session = project()
    say(session, "I want a refund")
    test = testgen.generate(
        session.turns,
        alone=False,
        source="serve()",
        namespace=None,
        signature=session.signature(),
        framework=runner_for(session.atom.connector).saved_test(session.atom.connector),
    )
    written = test.code.split("\n")
    assert written[0] == '"""Saved from chatomon studio: serve(), turn 1."""'
    assert "# Import it here, for example: from your_package.app import atom" in written
    assert not any(line.startswith("from shop") for line in written)
    compile(test.code, "saved.py", "exec")
    assert test.notes[-1] == (
        "The studio was started from Python, so it does not know where the atom lives: "
        "complete the import and save the file yourself. To let the studio write it, start "
        "it with `chatomon studio module:attribute`."
    )


MESSAGES = [
    'He said "hi" \\ bye',
    "line one\nline two",
    'three """ quotes',
    "naïve café 📦",
    "ends with a break\n",
    f"a{chr(0x2028)}b",
    "'single' and \"double\"",
]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I want a refund, NOW!", "i_want_a_refund_now"),
        ("Perché no?", "perche_no"),
        ("one two three four five six seven", "one_two_three_four_five_six"),
        (chr(0x3053) + chr(0x3093) + " " + chr(0x1F4E6), "turn_4"),
        ("   ", "turn_4"),
    ],
)
def test_the_name_comes_from_the_message(message, expected):
    assert testgen.slug(message, 4) == expected


def test_a_very_long_word_gives_a_name_the_system_accepts():
    name = testgen.slug("x" * 300 + " y", 1)
    assert len(name) <= 60
    assert name.isidentifier()


def test_a_long_message_goes_on_a_line_of_its_own(project):
    message = " ".join(["word"] * 15)
    session = project()
    say(session, message)  # more than eight words: stopped by message_length
    code = save(session).code
    assert f'    result = agent.run_sync(\n        "{message}",\n    )\n' in code


def test_a_turn_tried_with_studio_changes_is_warned_about(project):
    session = project()
    say(session, "hello")
    assert save(session).warnings == []
    session.edit("input", "message_length", {"max_words": 9})
    session.switch("input", "scope", False)
    say(session, "hello again")
    assert save(session).warnings == [
        "Turn 2 ran with 2 studio changes (message_length.max_words changed, scope turned "
        "off). The test builds the atom from your code, so it checks your code, not the "
        "studio: copy the configuration into your code first."
    ]
    assert save(session, upto=1).warnings == []
    session.switch("input", "scope", True)
    assert save(session).warnings[1:] == [
        "The orbitals have changed since: Copy as code writes them as they are now."
    ]
    session.reset_orbitals()
    session.switch("input", "scope", False)
    say(session, "once more")
    assert (
        save(session, alone=True)
        .warnings[0]
        .startswith("Turn 3 ran with 1 studio change (scope turned off).")
    )


def test_a_function_taken_as_calling_no_model_is_named(project):
    session = project()
    say(session, "I want a refund")
    assert save(session).notes == [
        OFFLINE,
        "`no_refunds` is a function of yours: the studio assumes it calls no model. If it "
        "does, declare it with `@orbital(calls_model=True)`.",
    ]
    say(session, NINE_WORDS)  # stopped by a class of the library before any function ran
    assert save(session, alone=True).notes == [OFFLINE, ALONE.format(n=2)]
    say(session, "hello")  # functions ran, but nothing in the test rests on them
    assert all("function of yours" not in note for note in save(session, alone=True).notes)


def test_an_orbital_that_failed_is_named(project, monkeypatch):
    session = project()
    say(session, "crash it")
    test = save(session)
    assert (
        "`no_refunds` failed in turn 1 (`RuntimeError`) and its `on_error` policy decided "
        "the result: the test expects the same."
    ) in test.notes
    assert "    assert result.output == atom.refusal\n" in test.code
    run_saved(test, monkeypatch)


def test_turns_tried_with_the_same_changes_share_one_warning(project):
    session = project()
    session.switch("input", "scope", False)
    say(session, "hello", "hello again", "and again")
    session.edit("input", "message_length", {"max_words": 9})
    say(session, "once more")
    rest = (
        " The test builds the atom from your code, so it checks your code, not the studio: "
        "copy the configuration into your code first."
    )
    assert save(session).warnings == [
        "Turns 1, 2 and 3 ran with 1 studio change (scope turned off)." + rest,
        "Turn 4 ran with 2 studio changes (message_length.max_words changed, scope turned off)."
        + rest,
        "The orbitals have changed since: Copy as code writes them as they are now.",
    ]
    assert save(session, upto=2).warnings[0].startswith("Turns 1 and 2 ran with 1 studio change")


def test_a_setting_changed_again_since_the_turn_is_said(project):
    session = project()
    session.edit("input", "message_length", {"max_words": 3})
    say(session, "one two three four")
    assert len(save(session).warnings) == 1
    session.edit("input", "message_length", {"max_words": 6})  # the same sentence, another value
    assert save(session).warnings == [
        "Turn 1 ran with 1 studio change (message_length.max_words changed). The test builds "
        "the atom from your code, so it checks your code, not the studio: copy the "
        "configuration into your code first.",
        "The orbitals have changed since: Copy as code writes them as they are now.",
    ]


def test_a_module_whose_name_is_not_a_python_name_leaves_the_import_to_complete(project):
    session = project("my-shop:atom")
    say(session, "I want a refund")
    test = save(session)
    assert test.importable is False
    written = test.code.split("\n")
    assert not any(line.startswith("from my-shop") for line in written)
    assert "# Import it here, for example: from your_package.app import atom" in written
    compile(test.code, "saved.py", "exec")
    assert test.notes[-1] == (
        "The atom's module cannot be written as an import, because its name is not a Python "
        "name: complete the import and save the file yourself."
    )
    assert session.saved_test(1)["writable"] is False
    with pytest.raises(RefusedError, match="the import of this test has to be completed by hand"):
        session.write_test(1, "tests/studio/test_x.py")
    plain = project()
    say(plain, "I want a refund")
    assert save(plain).importable is True


MODEL_WORDING = MODULE.replace(
    "@orbital\ndef polite(",
    """@orbital(calls_model=True)
def classify(ctx: OrbitalContext) -> Verdict:
    return Verdict.allow(annotations={"text": "wording a model chose"})


@orbital(reads=["classify"])
def answer(ctx: OrbitalContext) -> Verdict:
    if "advice" in ctx.message:
        return Verdict.block(ctx.annotations["classify"]["text"])
    return Verdict.allow()


@orbital
def polite(""",
).replace("no_refunds, scope]", "no_refunds, classify, answer, scope]")


def test_a_reply_that_may_carry_a_models_words_is_not_checked(project, monkeypatch):
    assert MODEL_WORDING.count("classify") == 4
    session = project(module=MODEL_WORDING)
    say(session, "some advice please")  # answered by a function, with text a model left behind
    test = save(session)
    assert session.turns[0].reply == "wording a model chose"
    assert "result.output" not in test.code
    assert "wording a model chose" not in test.code
    run_saved(test, monkeypatch)
    say(session, "I want a refund")  # no orbital that calls a model ran before this one
    assert "    assert result.output ==" in save(session, alone=True).code


LONG_NAMES = MODULE.replace(
    "@orbital(calls_model=True)\ndef scope",
    '@orbital(calls_model=True, name="a_very_long_name_for_the_orbital_that_classifies_scope")'
    "\ndef scope",
).replace(
    "@orbital\ndef polite",
    '@orbital(calls_model=True, name="another_long_name_for_the_orbital_that_checks_the_tone")'
    "\ndef polite",
)


def test_long_names_stay_within_88_columns(project, monkeypatch):
    assert LONG_NAMES.count("long_name_for_the_orbital") == 2
    session = project(module=LONG_NAMES)
    say(session, "Please explain your delivery policy today thanks")
    test = save(session)
    lines = test.code.split("\n")
    assert test.name == "test_please_explain_your_delivery_policy_today"
    assert [line for line in lines if len(line) > 88] == []
    assert "@pytest.mark.integration" in lines  # the comment went above it, on lines of its own
    assert f"def {test.name}(" in lines
    run_saved(test, monkeypatch)


def test_an_offline_turn_tried_with_studio_changes_says_what_offline_rests_on(project):
    module = MODULE.replace("no_refunds, scope]", "scope, no_refunds]")
    assert module != MODULE
    session = project(module=module)
    session.switch("input", "scope", False)  # in the code it calls a model before no_refunds
    say(session, "I want a refund")
    test = save(session)
    assert test.integration is False
    assert test.notes[:2] == [
        OFFLINE,
        "That is how the turn ran in the studio. Until its changes are in your code, your "
        "code's own orbitals run, and they may call a model.",
    ]


def test_a_name_that_is_not_an_orbitals_stays_out_of_the_comment(project):
    module = MODULE + 'scope.name = "scope\\nimport os"\n'
    session = project(module=module)
    say(session, "hello")
    test = save(session)
    lines = test.code.split("\n")
    assert "import os" not in lines
    assert "@pytest.mark.integration  # the nucleus and an orbital call real models" in lines
    compile(test.code, "saved.py", "exec")


def test_a_reply_over_several_lines_is_checked_as_written(project, monkeypatch):
    module = MODULE.replace(
        "We don't handle refunds in chat.", "We don't handle refunds in chat.\\nWrite to us."
    )
    assert module != MODULE
    session = project(module=module)
    say(session, "I want a refund")
    test = save(session)
    assert (
        "    assert result.output == (\n"
        '        "We don\'t handle refunds in chat.\\n"\n'
        '        "Write to us."\n'
        "    )\n"
    ) in test.code
    run_saved(test, monkeypatch)


# --- What depends on the runner, for every framework ---------------------------------

REPLY = {
    "pydantic_ai": "result.output",
    "langchain": 'result["messages"][-1].content',
    "maf": "result.text",
    "maf_foundry": "result.text",
}
MESSAGE_GOES_TO = {
    "pydantic_ai": "agent.run_sync",
    "langchain": "HumanMessage",
    "maf": "agent.run",
    "maf_foundry": "agent.run",
}
TOOL = '''

CALLED: list[str] = []


def refund_order(order_id: str) -> str:
    """Refund an order."""
    CALLED.append(order_id)
    return "refunded"
'''
KEEPS_OFFLINE = "    # No model is called in this test: the test model keeps the nucleus offline."
ASYNC = "Microsoft Agent Framework is asynchronous: the test needs `pytest-asyncio` 0.24 or later."


def sent(code: str, name: str) -> list[str]:
    """The messages a saved test sends, read back from its code."""
    calls = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == MESSAGE_GOES_TO[name]
    ]
    return [ast.literal_eval(call.args[0]) for call in calls]


def test_the_names_a_frameworks_lines_use_are_read_from_the_lines(project):
    session = project()
    names = runner_for(session.atom.connector).saved_test(session.atom.connector).names()
    assert {"agent", "result", "monkeypatch", "atom", "TestModel"} <= names
    # An attribute is not a name of the test, and neither is a word inside a string.
    assert not {"output", "connector", "all_messages", "setattr", "model"} & names


def test_every_framework_saves_a_test_that_passes_and_fails_without_the_orbital(
    project, tmp_path, framework: StudioHarness
):
    text = module_of(framework.name)
    session = project(module=text)
    say(session, "I want a refund")
    test = save(session)
    assert test.integration is False
    passed = run_pytest(tmp_path, test)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "1 passed" in passed.stdout

    without = text.replace("no_refunds, scope]", "scope]")
    assert without != text
    (tmp_path / "shop.py").write_text(without, encoding="utf-8")
    failed = run_pytest(tmp_path, test)
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "1 failed" in failed.stdout


def test_every_framework_saves_a_conversation_that_passes_with_its_markers(
    project, tmp_path, framework: StudioHarness
):
    session = project(module=module_of(framework.name))
    say(session, "hello", "I am angry", "I want a refund")
    test = save(session)
    assert test.integration is True
    assert [turn.report.final_action for turn in session.turns] == [
        "answered",
        "answered",
        "blocked",
    ]
    result = run_pytest(tmp_path, test)  # --strict-markers: every marker must be a known one
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_every_framework_checks_a_reply_fixed_in_code_with_its_own_expression(
    project, monkeypatch, framework: StudioHarness
):
    session = project(module=module_of(framework.name))
    say(session, "I want a refund")
    test = save(session)
    check = f'    assert {REPLY[framework.name]} == "We don\'t handle refunds in chat."\n'
    assert check in test.code
    run_saved(test, monkeypatch)


@pytest.mark.parametrize("message", MESSAGES)
def test_every_framework_sends_any_message_as_the_same_text(
    project, monkeypatch, framework: StudioHarness, message
):
    session = project(module=module_of(framework.name))
    say(session, message)
    test = save(session)
    assert sent(test.code, framework.name) == [message]
    assert chr(0x2028) not in test.code  # editors show it as a line break
    run_saved(test, monkeypatch)


def test_every_framework_keeps_its_code_to_88_columns(project, framework: StudioHarness):
    session = project(module=module_of(framework.name))
    say(session, "I want a refund")
    offline = save(session)
    say(session, "hello", "Write me a poem")
    for test in (offline, save(session)):
        assert [line for line in test.code.split("\n") if len(line) > 88] == []


def test_every_framework_keeps_the_atoms_tools_from_running_offline(
    project, monkeypatch, framework: StudioHarness
):
    plain = PARTS[framework.name][2]
    text = module_of(
        framework.name, connector=f"{plain[:-1]}, tools=[refund_order])", before_atom=TOOL
    )
    session = project(module=text)
    say(session, "I want a refund")
    test = save(session)
    assert test.integration is False
    # The code changes and lets the message through: the test fails, and no tool has run.
    atom = session.atom
    kept = tuple(orb for orb in atom.input_orbitals if orb.name != "no_refunds")
    monkeypatch.setattr(atom, "input_orbitals", kept)
    # LangChain's fake model cannot take the atom's tools: the card says so in a note.
    failure = NotImplementedError if framework.name == "langchain" else AssertionError
    with pytest.raises(failure):
        run_saved(test, monkeypatch)
    assert sys.modules["shop"].CALLED == []


def test_a_langchain_test_offline_needs_neither_the_providers_package_nor_its_key(
    project, tmp_path
):
    text = module_of("langchain")
    session = project(module=text)
    say(session, "I want a refund")
    test = save(session)
    assert (
        '        atom.connector, "model", FakeListChatModel(responses=["offline"])\n' in test.code
    )
    # Only now a model the atom cannot build here: the saved test replaces it before it builds.
    named = text.replace("LangChainConnector(Nucleus())", 'LangChainConnector("nowhere:model")')
    assert named != text
    (tmp_path / "shop.py").write_text(named, encoding="utf-8")
    result = run_pytest(tmp_path, test)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_langchain_atom_with_a_checkpointer_gives_every_call_a_thread(project, tmp_path):
    text = module_of(
        "langchain",
        connector="LangChainConnector(Nucleus(), checkpointer=InMemorySaver())",
        before_atom="\nfrom langgraph.checkpoint.memory import InMemorySaver  # noqa: E402\n",
    )
    session = project(module=text)
    say(session, "hello", "I want a refund")
    test = save(session)
    assert "from uuid import uuid4\n" in test.code
    assert test.code.count('        {"configurable": {"thread_id": uuid4().hex}},\n') == 2
    result = run_pytest(tmp_path, test)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_langchain_test_says_what_its_fake_model_cannot_do(project):
    session = project(module=module_of("langchain"))
    say(session, "I want a refund")
    assert save(session).notes[:2] == [
        "No model is called in this test: it runs offline, with LangChain's "
        "`FakeListChatModel` in place of the nucleus.",
        "If your code lets a message through, the test fails; on an atom with tools it fails "
        "with `NotImplementedError`, because `FakeListChatModel` cannot take them.",
    ]
    say(session, "hello")
    assert all("FakeListChatModel" not in note for note in save(session, alone=True).notes)


@pytest.mark.parametrize("name", ["maf", "maf_foundry"])
def test_a_microsoft_agent_framework_test_is_a_coroutine_with_a_stand_in_of_its_own(
    project, monkeypatch, name
):
    session = project(module=module_of(name))
    say(session, "I want a refund")
    offline = save(session)
    lines = offline.code.split("\n")
    marker = lines.index('@pytest.mark.asyncio(loop_scope="session")')
    assert lines[marker + 1] == (
        "async def test_i_want_a_refund(monkeypatch: pytest.MonkeyPatch) -> None:"
    )
    assert "class OfflineNucleus(ChatMiddlewareLayer, BaseChatClient):" in lines
    build = lines.index("    agent = atom.build()")
    assert lines[build + 1 : build + 4] == [
        KEEPS_OFFLINE,
        "    agent.client = OfflineNucleus()",
        "    session = agent.create_session()",
    ]
    assert '    result = await agent.run("I want a refund", session=session)' in lines
    assert offline.notes[:3] == [
        "No model is called in this test: it runs offline, with a client of its own, "
        "`OfflineNucleus`, in place of the nucleus.",
        "`OfflineNucleus` is written on Microsoft Agent Framework's client classes: "
        "if they change, save the test again.",
        ASYNC,
    ]
    run_saved(offline, monkeypatch)

    say(session, "hello")
    online = save(session, alone=True)
    lines = online.code.split("\n")
    marker = lines.index('@pytest.mark.asyncio(loop_scope="session")')
    assert lines[marker - 1].startswith("@pytest.mark.integration")
    assert lines[marker + 1].startswith("async def test_hello(")
    assert "OfflineNucleus" not in online.code
    assert "agent_framework" not in online.code
    assert online.notes[1] == ASYNC
    run_saved(online, monkeypatch)


def test_a_factory_named_like_something_the_runners_lines_use_gets_the_alias(project, monkeypatch):
    text = module_of("maf") + "\n\ndef session() -> Atom:\n    return atom\n"
    opened = project("shop:session", module=text)
    say(opened, "I want a refund")
    test = save(opened)
    assert "from shop import session as atom_source\n" in test.code
    assert "    atom = atom_source()\n" in test.code
    run_saved(test, monkeypatch)


def test_started_from_python_a_microsoft_agent_framework_test_still_compiles(project):
    session = project(module=module_of("maf"))
    say(session, "I want a refund")
    test = testgen.generate(
        session.turns,
        alone=False,
        source="serve()",
        namespace=None,
        signature=session.signature(),
        framework=runner_for(session.atom.connector).saved_test(session.atom.connector),
    )
    compile(test.code, "saved.py", "exec")
    # The import left to complete stays with the imports, above the class the test defines.
    assert test.code.index("# Import it here") < test.code.index("class OfflineNucleus")
    assert "async def test_i_want_a_refund(" in test.code


def test_the_names_a_runners_lines_use_are_read_from_the_lines():
    names = maf_runner.saved_test(MAFConnector(ScriptedMAFClient([], []))).names()
    assert {"session", "agent", "OfflineNucleus", "ChatResponse", "Message", "pytest"} <= names
    # An attribute is not a name of the test, and neither is a word inside a string.
    assert not {"client", "text", "create_session", "offline", "assistant", "Stands"} & names
