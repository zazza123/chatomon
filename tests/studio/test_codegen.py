import ast
import math
import sys
import textwrap
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

from chatomon import Atom, Knowledge
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import JudgeOrbital
from chatomon.studio import _codegen as codegen
from chatomon.studio._codegen import Code
from chatomon.studio._loading import load, namespace
from chatomon.studio._session import Session

MODULE = """
from pydantic_ai.models.test import TestModel

from chatomon import Atom, Knowledge, OrbitalContext, Verdict, orbital
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import JudgeOrbital, MessageLengthOrbital, TurnLimitOrbital

faq = Knowledge("Open 9-18.")
connector = PydanticAIConnector(TestModel(custom_output_text="Hi"))


@orbital(name="scope")
def scope(ctx: OrbitalContext) -> Verdict:
    return Verdict.allow()


@orbital
def no_refunds(ctx: OrbitalContext) -> Verdict:
    return Verdict.allow()


faithfulness = JudgeOrbital(
    name="faithfulness", model="test:judge", instructions="Check facts.", context=faq
)

atom = Atom(
    name="speedypack",
    instructions="Be brief.",
    knowledge=faq,
    connector=connector,
    input_orbitals=[
        MessageLengthOrbital(max_words=60),
        TurnLimitOrbital(max_turns=8, warn_before=2),
        scope,
        no_refunds,
    ],
    output_orbitals=[faithfulness],
)


def make():
    return Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=connector,
        input_orbitals=[orbital(name="inline")(lambda ctx: Verdict.allow())],
    )
"""

SHARED = """
from pydantic_ai.models.test import TestModel

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import MessageLengthOrbital

connector = PydanticAIConnector(TestModel(custom_output_text="Hi"))
limit = MessageLengthOrbital(max_words=60)

atom = Atom(
    name="speedypack",
    instructions="Be brief.",
    connector=connector,
    input_orbitals=[limit],
    output_orbitals=[limit],
)


def inline():
    both = MessageLengthOrbital(max_words=60, name="limit.v2")
    return Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=connector,
        input_orbitals=[both],
        output_orbitals=[both],
    )
"""

TONE = {
    "name": "tone",
    "model": "test:judge",
    "instructions": "Block replies that are rude or off-brand.",
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project with an `app` module, loaded as the command loads it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name)

    def open_session(
        spec: str = "app:atom", *, module: str = MODULE
    ) -> tuple[Session, dict[str, object]]:
        (tmp_path / "app.py").write_text(module, encoding="utf-8")
        atom = load(spec)  # the namespace exists once the module is imported
        names = namespace(spec)
        return Session(atom, source=spec, namespace=names), names

    return open_session


def lists(code: Code, names: dict[str, object]) -> dict[str, list[Any]]:
    """Run the generated code in the module's namespace and return the two lists."""
    scope = dict(names)
    exec(code.imports, scope)
    exec(code.setup, scope)
    return eval(f"dict(\n{code.orbitals}\n)", scope)


def fingerprints(orbitals) -> list[str]:
    return [orbital.fingerprint() for orbital in orbitals]


def test_an_atom_built_with_the_code_has_the_sessions_fingerprints(project):
    session, names = project()
    session.edit("input", "message_length", {"max_words": 80, "reply": 'Say "less".\nPlease.'})
    session.switch("input", "scope", False)
    session.move("input", "no_refunds", 0)
    session.add("output", "JudgeOrbital", {**TONE, "reads": ["faithfulness"]})
    session.edit("output", "faithfulness", {"history_window": 2})
    code = codegen.generate(session.chains, names)
    assert code.placeholders == []
    atom = session.atom
    built = lists(code, names)
    rebuilt = Atom(
        name=atom.name,
        instructions=atom.instructions,
        connector=atom.connector,
        knowledge=atom.knowledge,
        input_orbitals=built["input_orbitals"],
        output_orbitals=built["output_orbitals"],
    )
    rebuilt.validate()
    assert fingerprints(rebuilt.input_orbitals) == fingerprints(session.chains.enabled("input"))
    assert fingerprints(rebuilt.output_orbitals) == fingerprints(session.chains.enabled("output"))
    assert [orbital.name for orbital in rebuilt.input_orbitals] == [
        "no_refunds",
        "message_length",
        "turn_limit",
    ]


def test_an_untouched_session_writes_names_and_rebuilds_what_is_inline(project):
    session, names = project()
    assert codegen.generate(session.chains, names).code == textwrap.dedent("""\
        # Imports
        from chatomon.orbitals import MessageLengthOrbital, TurnLimitOrbital

        # Inside Atom(...)
        input_orbitals=[
            MessageLengthOrbital(max_words=60),
            TurnLimitOrbital(max_turns=8, warn_before=2),
            scope,
            no_refunds,
        ],
        output_orbitals=[faithfulness],""")


def test_the_code_after_some_changes(project):
    session, names = project()
    session.edit("input", "message_length", {"max_words": 80})
    session.switch("input", "scope", False)
    session.add("output", "JudgeOrbital", TONE)
    assert codegen.generate(session.chains, names).code == textwrap.dedent("""\
        # Imports
        from chatomon.orbitals import JudgeOrbital, MessageLengthOrbital, TurnLimitOrbital

        # Inside Atom(...)
        input_orbitals=[
            MessageLengthOrbital(max_words=80),
            TurnLimitOrbital(max_turns=8, warn_before=2),
            # Off in the studio:
            # scope,
            no_refunds,
        ],
        output_orbitals=[
            faithfulness,
            JudgeOrbital(
                name="tone",
                model="test:judge",
                instructions="Block replies that are rude or off-brand.",
            ),
        ],""")


def test_imports_cover_only_the_active_lines(project):
    session, names = project()
    session.switch("input", "message_length", False)
    session.switch("input", "turn_limit", False)
    code = codegen.generate(session.chains, names)
    assert code.imports == ""
    assert code.code.startswith("# Inside Atom(...)\ninput_orbitals=[\n    # Off in the studio:\n")
    assert "    # MessageLengthOrbital(max_words=60),\n" in code.code


def test_text_with_line_breaks_is_written_one_line_per_literal(project):
    session, names = project()
    text = 'First line.\nSecond "line".\n'
    session.add("output", "JudgeOrbital", {**TONE, "instructions": text})
    code = codegen.generate(session.chains, names)
    assert (
        "        instructions=(\n"
        '            "First line.\\n"\n'
        '            "Second \\"line\\".\\n"\n'
        "        ),\n"
    ) in code.orbitals
    tone = lists(code, names)["output_orbitals"][1]
    assert tone.instructions == text


@pytest.mark.parametrize(
    "value",
    [
        'say "hi"',
        "back\\slash",
        "tab\there",
        "emoji 📦 è",
        "a\u2028b",
        "nul\x00",
        "\ud800",
        "",
        3,
        2.5,
        True,
        None,
        ["a", "b"],
    ],
)
def test_a_literal_reads_back_to_the_same_value(value):
    assert ast.literal_eval(codegen.literal(value)) == value


def test_a_number_that_is_not_finite_is_still_python():
    assert codegen.literal(math.inf) == 'float("inf")'
    assert codegen.literal(("a", "b")) == '["a", "b"]'


def test_what_cannot_be_written_is_marked_and_listed(project):
    session, names = project("app:make")
    code = codegen.generate(session.chains, names)
    assert code.orbitals == (
        "input_orbitals=[\n    ...,  # keep 'inline' (FunctionOrbital) from your code\n],"
    )
    assert code.placeholders == ["inline: a FunctionOrbital from your code"]
    assert code.notes == [codegen.NO_NAME]


def test_started_from_python_there_are_no_names(project):
    session, _ = project()
    code = codegen.generate(session.chains, None)
    assert "    MessageLengthOrbital(max_words=60),\n" in code.orbitals
    assert 'context="Open 9-18."' in code.orbitals  # a knowledge without a title is its text
    assert "    ...,  # keep 'scope' (FunctionOrbital) from your code\n" in code.orbitals
    assert code.placeholders == [
        "scope: a FunctionOrbital from your code",
        "no_refunds: a FunctionOrbital from your code",
    ]
    assert code.notes == [codegen.NO_MODULE]


def make_atom(**kwargs: Any) -> Atom:
    kwargs.setdefault("connector", PydanticAIConnector(TestModel(custom_output_text="Hi")))
    return Atom(name="speedypack", instructions="Be brief.", **kwargs)


def test_an_object_argument_is_written_with_its_name_or_marked():
    own = PydanticAIConnector(TestModel())
    judge = JudgeOrbital(model="test:judge", instructions="Check facts.", connector=own)
    session = Session(make_atom(output_orbitals=[judge]), source="app:atom")
    unnamed = codegen.generate(session.chains, {"other": 1})
    assert "    connector=...,  # your PydanticAIConnector\n" in unnamed.orbitals
    assert unnamed.placeholders == ["judge: `connector` is a PydanticAIConnector from your code"]
    named = codegen.generate(session.chains, {"judge_connector": own})
    assert "connector=judge_connector" in named.orbitals
    assert named.placeholders == []


def test_a_knowledge_with_a_title_is_written_as_a_knowledge():
    judge = JudgeOrbital(
        model="test:judge", instructions="Check.", context=Knowledge("Open 9-18.", title="FAQ")
    )
    session = Session(make_atom(output_orbitals=[judge]), source="app:atom")
    code = codegen.generate(session.chains, {})
    assert 'context=Knowledge("Open 9-18.", title="FAQ")' in code.orbitals
    assert code.imports == (
        "from chatomon import Knowledge\nfrom chatomon.orbitals import JudgeOrbital"
    )


def test_an_atom_without_orbitals_has_no_code():
    code = codegen.generate(Session(make_atom(), source="app:atom").chains, {})
    assert (code.code, code.notes) == ("", [codegen.EMPTY])


def test_plain_values_are_never_looked_up_by_name():
    assert codegen.names({"limit": 60, "text": "hi", "__doc__": object(), "flag": True}) == {}
    marker = object()
    assert codegen.names({"first": marker, "second": marker}) == {id(marker): "first"}
    assert codegen.names(None) == {}


def rebuilt(session: Session, code: Code, names: dict[str, object]) -> Atom:
    """Build the atom from code with the generated lists, as a developer who pasted them."""
    atom, built = session.atom, lists(code, names)
    again = Atom(
        name=atom.name,
        instructions=atom.instructions,
        connector=atom.connector,
        input_orbitals=built["input_orbitals"],
        output_orbitals=built["output_orbitals"],
    )
    again.validate()
    return again


def test_an_orbital_in_both_chains_is_written_once_and_used_twice(project):
    session, names = project(module=SHARED)
    session.edit("input", "message_length", {"max_words": 80})
    code = codegen.generate(session.chains, names)
    assert code.code == textwrap.dedent("""\
        # Imports
        from chatomon.orbitals import MessageLengthOrbital

        # Before the atom
        message_length = MessageLengthOrbital(max_words=80)

        # Inside Atom(...)
        input_orbitals=[message_length],
        output_orbitals=[message_length],""")
    atom = rebuilt(session, code, names)
    assert atom.input_orbitals[0] is atom.output_orbitals[0]
    assert fingerprints(atom.input_orbitals) == fingerprints(session.chains.enabled("input"))
    assert fingerprints(atom.output_orbitals) == fingerprints(session.chains.enabled("output"))


def test_an_untouched_orbital_in_both_chains_keeps_its_name(project):
    session, names = project(module=SHARED)
    code = codegen.generate(session.chains, names)
    assert code.setup == ""
    assert code.orbitals == "input_orbitals=[limit],\noutput_orbitals=[limit],"


def test_a_shared_orbital_without_a_name_gets_a_variable_nothing_else_uses(project):
    session, names = project("app:inline", module=SHARED)
    session.switch("output", "limit.v2", False)
    code = codegen.generate(session.chains, {**names, "limit_v2": "taken"})
    assert code.setup == 'limit_v2_2 = MessageLengthOrbital(name="limit.v2", max_words=60)'
    assert code.orbitals == (
        "input_orbitals=[limit_v2_2],\n"
        "output_orbitals=[\n    # Off in the studio:\n    # limit_v2_2,\n],"
    )
    assert code.imports == "from chatomon.orbitals import MessageLengthOrbital"
    atom = rebuilt(session, code, names)
    assert fingerprints(atom.input_orbitals) == fingerprints(session.chains.enabled("input"))
    assert atom.output_orbitals == ()
