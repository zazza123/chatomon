import sys
import textwrap
import types
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

import chatomon
from chatomon import Atom, ChatomonConfigError
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._loading import load, namespace, project_modules, reimporting, resolve


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


def test_the_namespace_of_the_loaded_module(project):
    project(HEADER + "atom = make()\nlimit = 3\n")
    atom = load("app:atom")
    found = namespace("app:atom")
    assert found["atom"] is atom
    assert found["limit"] == 3
    assert namespace("not_imported_anywhere:atom") == {}


NEEDS_ARGUMENTS = (
    "app:judge needs arguments (messages, info): the studio calls it with none. "
    "Point at an Atom, or at a function with no arguments that returns one"
)


def test_a_function_that_needs_arguments_is_explained_and_not_called(project):
    project(
        HEADER
        + "def judge(messages, info, *, strict=False):\n"
        + "    raise RuntimeError('this must not run')\n"
    )
    with pytest.raises(ChatomonConfigError) as caught:
        load("app:judge")
    assert str(caught.value) == NEEDS_ARGUMENTS
    assert caught.value.__cause__ is None  # nothing was called: there is no traceback to show


def test_a_function_whose_arguments_all_have_defaults_is_still_called(project):
    project(HEADER + "def build(name='t', *extra, **options):\n    return make()\n")
    assert load("app:build").name == "t"


def test_a_class_that_needs_arguments_is_explained_too():
    class Builder:
        def __init__(self, model: str) -> None:
            self.model = model

    with pytest.raises(ChatomonConfigError, match=r"app:Builder needs arguments \(model\)"):
        resolve(Builder, label="app:Builder")


APP = """
    from pydantic_ai.models.test import TestModel
    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector
    from helpers import NAME

    made = []

    def make():
        made.append(NAME)
        return Atom(name=NAME, instructions="Be brief.", connector=PydanticAIConnector(TestModel()))

    atom = make()
"""


def again(spec: str, root: Path) -> Atom:
    """Load a spec again as a reload does."""
    with reimporting(root):
        return load(spec)


def test_a_change_in_a_module_the_atoms_module_imports_is_seen(sources, tmp_path):
    sources("app.py", APP)
    sources("helpers.py", 'NAME = "one"\n')
    first = load("app:atom")
    assert first.name == "one"
    sources("helpers.py", 'NAME = "two"\n')
    second = again("app:atom", tmp_path)
    assert second.name == "two"
    assert second is not first
    assert sys.modules["app"].atom is second


def test_a_file_created_after_the_start_is_found(sources, tmp_path):
    sources("app.py", APP)
    sources("helpers.py", 'NAME = "one"\n')
    load("app:atom")
    sources("helpers.py", "from later import NAME\n")
    sources("later.py", 'NAME = "later"\n')
    assert again("app:atom", tmp_path).name == "later"


@pytest.mark.parametrize("init", [True, False], ids=["package", "namespace package"])
def test_the_modules_of_a_package_are_imported_again(sources, tmp_path, init):
    if init:
        sources("shop/__init__.py", "")
    sources("shop/app.py", APP.replace("from helpers import", "from shop.helpers import"))
    sources("shop/helpers.py", 'NAME = "one"\n')
    assert load("shop.app:atom").name == "one"
    sources("shop/helpers.py", 'NAME = "two"\n')
    assert again("shop.app:atom", tmp_path).name == "two"
    assert sys.modules["shop"].helpers is sys.modules["shop.helpers"]


def test_a_factory_is_called_again_on_the_new_code(sources, tmp_path):
    sources("app.py", APP)
    sources("helpers.py", 'NAME = "one"\n')
    assert load("app:make").name == "one"
    sources("helpers.py", 'NAME = "two"\n')
    assert again("app:make", tmp_path).name == "two"
    assert sys.modules["app"].made == ["two", "two"]  # the module's own atom, then the factory


def test_a_module_outside_the_directory_is_left_alone(sources, tmp_path, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (elsewhere / "outside_lib.py").write_text('NAME = "one"\n', encoding="utf-8")
    sys.path.append(str(elsewhere))
    sources("app.py", APP.replace("from helpers import", "from outside_lib import"))
    load("app:atom")
    library = sys.modules["outside_lib"]
    try:
        assert "outside_lib" not in project_modules(tmp_path)
        (elsewhere / "outside_lib.py").write_text('NAME = "another"\n', encoding="utf-8")
        assert again("app:atom", tmp_path).name == "one"
        assert sys.modules["outside_lib"] is library
    finally:
        del sys.modules["outside_lib"]


def test_an_environment_inside_the_directory_is_left_alone(sources, tmp_path, monkeypatch):
    packages = tmp_path / ".venv" / "lib" / "python3" / "site-packages"
    sources(".venv/lib/python3/site-packages/installed_lib.py", "VALUE = 1\n")
    sources(".venv/tools/env_tool.py", "VALUE = 2\n")
    sources("vendor/site-packages/vendored_lib.py", "VALUE = 3\n")
    sys.path.extend([str(packages), str(tmp_path / ".venv" / "tools")])
    sys.path.append(str(tmp_path / "vendor" / "site-packages"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv"))
    sources("app.py", "import installed_lib, env_tool, vendored_lib\n")
    __import__("app")
    assert sorted(project_modules(tmp_path)) == ["app"]


def test_a_project_inside_the_interpreters_prefix_is_still_the_project(
    sources, tmp_path, monkeypatch
):
    """A system interpreter in /usr and a project in /usr/src/app."""
    monkeypatch.setattr(sys, "prefix", str(tmp_path.parent))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path.parent))
    sources("app.py", "VALUE = 1\n")
    __import__("app")
    assert sorted(project_modules(tmp_path)) == ["app"]


def test_chatomon_itself_and_the_environment_are_never_the_project():
    """From this repository's root chatomon's own files are under the directory."""
    repository = Path(chatomon.__file__).resolve().parents[2]
    names = project_modules(repository)
    assert not [name for name in names if name.split(".")[0] == "chatomon"]
    assert not {"pytest", "starlette", "pydantic_ai"} & set(names)


def test_the_script_that_was_run_is_not_imported_again(sources, tmp_path, monkeypatch):
    sources("script.py", "VALUE = 1\n")
    main = types.ModuleType("__main__")
    main.__file__ = str(tmp_path / "script.py")
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert "__main__" not in project_modules(tmp_path)
    with reimporting(tmp_path):
        pass
    assert sys.modules["__main__"] is main


@pytest.mark.parametrize("init", [True, False], ids=["package", "namespace package"])
def test_a_failed_import_leaves_the_modules_that_were_there(sources, tmp_path, init):
    if init:
        sources("shop/__init__.py", "")
    sources("shop/app.py", APP.replace("from helpers import", "from shop.helpers import"))
    sources("shop/helpers.py", 'NAME = "one"\n')
    load("shop.app:atom")
    before = {name: sys.modules[name] for name in ("shop", "shop.app", "shop.helpers")}
    sources("shop/helpers.py", 'import shop.fresh\nimport shop.broken\nNAME = "two"\n')
    sources("shop/fresh.py", "VALUE = 1\n")
    sources("shop/broken.py", "def broken(:\n")
    with pytest.raises(ChatomonConfigError, match=r"could not import 'shop\.app'") as error:
        again("shop.app:atom", tmp_path)
    assert isinstance(error.value.__cause__, SyntaxError)
    assert {name: sys.modules[name] for name in before} == before
    assert "shop.fresh" not in sys.modules and "shop.broken" not in sys.modules
    assert sys.modules["shop"].helpers is before["shop.helpers"]
    assert sys.modules["shop.app"].atom.name == "one"


def test_an_error_of_the_block_itself_also_puts_the_modules_back(sources, tmp_path):
    sources("app.py", APP)
    sources("helpers.py", 'NAME = "one"\n')
    load("app:atom")
    before = sys.modules["app"]
    sources("helpers.py", 'NAME = "two"\n')
    with pytest.raises(RuntimeError, match="later in the reload"), reimporting(tmp_path):
        assert load("app:atom").name == "two"
        raise RuntimeError("later in the reload")
    assert sys.modules["app"] is before
    assert sys.modules["helpers"].NAME == "one"


def test_a_module_whose_file_is_not_a_full_path_is_never_taken_for_the_project(
    sources, tmp_path, monkeypatch
):
    """A library can build a module at run time and give it a bare file name."""
    sources("_ops.py", "VALUE = 1\n")
    built = types.ModuleType("library._ops")
    built.__file__ = "_ops.py"  # relative: it would be read from the current directory
    monkeypatch.setitem(sys.modules, "library._ops", built)
    assert "library._ops" not in project_modules(tmp_path)
    with reimporting(tmp_path):
        pass
    assert sys.modules["library._ops"] is built


def test_a_folder_of_the_project_that_is_a_link_is_still_the_project(
    sources, tmp_path, tmp_path_factory
):
    """Followed to where it points, the folder is outside; the project is where it was imported."""
    real = tmp_path_factory.mktemp("elsewhere") / "linked"
    real.mkdir()
    (real / "__init__.py").write_text("", encoding="utf-8")
    (real / "helpers.py").write_text('NAME = "one"\n', encoding="utf-8")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    sources("app.py", APP.replace("from helpers import", "from linked.helpers import"))
    assert load("app:atom").name == "one"
    try:
        assert sorted(project_modules(tmp_path)) == ["app", "linked", "linked.helpers"]
        (real / "helpers.py").write_text('NAME = "another"\n', encoding="utf-8")
        assert again("app:atom", tmp_path).name == "another"
    finally:
        for name in ("linked", "linked.helpers"):
            sys.modules.pop(name, None)


def test_reimporting_says_which_modules_it_took_out(sources, tmp_path):
    sources("app.py", APP)
    sources("helpers.py", 'NAME = "one"\n')
    load("app:atom")
    with reimporting(tmp_path) as removed:
        assert removed == {"app", "helpers"}
        assert "app" not in sys.modules
        load("app:atom")
