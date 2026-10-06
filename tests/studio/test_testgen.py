"""Save as test: the generated file, run in this process and with a real pytest."""

import ast
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chatomon.studio import _testgen as testgen
from chatomon.studio._chains import RefusedError
from chatomon.studio._loading import load, namespace
from chatomon.studio._runners.pydantic_ai import saved_test
from chatomon.studio._session import Session

MODULE = """
from types import SimpleNamespace

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom, OrbitalContext, Verdict, orbital
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital


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


atom = Atom(
    name="speedypack",
    instructions="Be brief.",
    connector=PydanticAIConnector(FunctionModel(nucleus)),
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
NINE_WORDS = "one two three four five six seven eight nine"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project with a `shop` module, loaded as the command loads it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [name for name in sys.modules if name == "shop"]:
        monkeypatch.delitem(sys.modules, name)
    (tmp_path / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")

    def open_session(spec: str = "shop:atom", *, module: str = MODULE) -> Session:
        (tmp_path / "shop.py").write_text(module, encoding="utf-8")
        atom = load(spec)  # the namespace exists once the module is imported
        return Session(atom, source=spec, namespace=namespace(spec))

    return open_session


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
        changes=[change.text for change in session.chains.changes()],
        framework=saved_test(),
    )


def run_saved(test: testgen.SavedTest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the saved test in this process, on the module the session loaded."""
    scope: dict[str, Any] = {"__name__": "saved"}
    exec(compile(test.code, f"{test.name}.py", "exec"), scope)
    scope[test.name](monkeypatch)


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


def test_a_saved_test_passes_on_the_code_and_fails_without_the_orbital(project, tmp_path):
    session = project()
    say(session, "I want a refund")
    test = save(session)
    assert test.integration is False
    passed = run_pytest(tmp_path, test)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "1 passed" in passed.stdout

    without = MODULE.replace("no_refunds, scope]", "scope]")
    assert without != MODULE
    (tmp_path / "shop.py").write_text(without, encoding="utf-8")
    failed = run_pytest(tmp_path, test)
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "1 failed" in failed.stdout


def test_a_conversation_passes_with_its_marker_registered(project, tmp_path):
    session = project()
    say(session, "hello", "I want a refund")
    test = save(session)
    assert test.integration is True
    result = run_pytest(tmp_path, test)  # --strict-markers: the marker must be the registered one
    assert result.returncode == 0, result.stdout + result.stderr


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
        assert '    monkeypatch.setattr(atom.connector, "model", TestModel())\n' in test.code
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
        changes=[],
        framework=saved_test(),
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
        changes=[],
        framework=saved_test(),
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


@pytest.mark.parametrize("message", MESSAGES)
def test_any_message_comes_back_as_the_same_text(project, monkeypatch, message):
    session = project()
    say(session, message)
    test = save(session)
    calls = [
        node
        for node in ast.walk(ast.parse(test.code))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "agent.run_sync"
    ]
    assert [ast.literal_eval(call.args[0]) for call in calls] == [message]
    assert chr(0x2028) not in test.code  # editors show it as a line break
    run_saved(test, monkeypatch)


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


def test_the_code_keeps_to_88_columns(project):
    session = project()
    say(session, "hello", "I want a refund", "Write me a poem")
    assert max(len(line) for line in save(session).code.split("\n")) <= 88


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
    assert save(session, alone=True).notes == [OFFLINE]
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
