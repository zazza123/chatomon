import sys
import textwrap

import pytest
from pydantic_ai.models.test import TestModel

from chatomon import Atom, ChatomonConfigError
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._loading import load, resolve


def make_atom() -> Atom:
    return Atom(name="t", instructions="Be brief.", connector=PydanticAIConnector(TestModel()))


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project directory with an `app` module, as the current directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name)

    def write(source: str) -> None:
        (tmp_path / "app.py").write_text(textwrap.dedent(source), encoding="utf-8")

    return write


HEADER = textwrap.dedent("""
    from pydantic_ai.models.test import TestModel
    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector

    def make():
        return Atom(name="t", instructions="Be brief.", connector=PydanticAIConnector(TestModel()))
""")


def test_an_atom_instance_is_loaded_from_the_current_directory(project):
    project(HEADER + "atom = make()\n")
    assert load("app:atom").name == "t"


def test_a_factory_is_called(project):
    project(HEADER)
    assert load("app:make").name == "t"


def test_a_dotted_attribute_is_followed(project):
    project(HEADER + "class Holder:\n    atom = make()\n")
    assert load("app:Holder.atom").name == "t"


@pytest.mark.parametrize("spec", ["app", "app:", ":atom"])
def test_a_spec_without_module_and_attribute_is_refused(spec):
    with pytest.raises(ChatomonConfigError, match="<module>:<attribute>"):
        load(spec)


def test_a_module_that_cannot_be_imported_chains_the_error(project):
    project("raise RuntimeError('boom')\n")
    with pytest.raises(ChatomonConfigError, match="could not import 'app'") as error:
        load("app:atom")
    assert isinstance(error.value.__cause__, RuntimeError)


def test_a_missing_attribute_is_reported(project):
    project(HEADER)
    with pytest.raises(ChatomonConfigError, match="has no attribute 'atom'") as error:
        load("app:atom")
    assert error.value.__cause__ is None


def test_a_value_that_is_not_an_atom_is_refused():
    with pytest.raises(ChatomonConfigError, match="is a int, not an Atom"):
        resolve(3, label="app:x")


def test_a_failing_factory_chains_the_error():
    def factory() -> Atom:
        raise KeyError("OPENAI_API_KEY")

    with pytest.raises(ChatomonConfigError, match=r"app:make\(\) raised KeyError") as error:
        resolve(factory, label="app:make")
    assert isinstance(error.value.__cause__, KeyError)


def test_a_factory_must_return_an_atom():
    with pytest.raises(ChatomonConfigError, match=r"returned a str, not an Atom"):
        resolve(lambda: "atom", label="app:make")


def test_an_instance_is_returned_unchanged():
    atom = make_atom()
    assert resolve(atom, label="x") is atom


def test_the_current_directory_wins_over_a_module_elsewhere_on_the_path(
    project, tmp_path, monkeypatch
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "app.py").write_text("atom = 'not the atom'\n", encoding="utf-8")
    project(HEADER + "atom = make()\n")
    monkeypatch.setattr(sys, "path", [str(elsewhere), str(tmp_path), *sys.path])
    assert load("app:atom").name == "t"
