import importlib
import importlib.util
import sys

import pytest

from chatomon import MissingExtraError


@pytest.mark.parametrize(
    ("module", "package"),
    [
        ("starlette", "starlette"),
        ("uvicorn", "uvicorn"),
        ("opentelemetry.sdk", "opentelemetry-sdk"),
    ],
)
def test_a_missing_studio_dependency_names_the_extra(monkeypatch, module, package):
    for name in [name for name in sys.modules if name.startswith("chatomon.studio")]:
        monkeypatch.delitem(sys.modules, name)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == module else real_find_spec(name, *args),
    )
    with pytest.raises(MissingExtraError, match=r"chatomon\[studio\]") as error:
        importlib.import_module("chatomon.studio")
    assert error.value.package == package


def test_the_studio_imports_with_its_extra():
    assert importlib.import_module("chatomon.studio").__name__ == "chatomon.studio"
