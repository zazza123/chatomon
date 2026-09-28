from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = list(find_examples(ROOT / "src" / "chatomon", ROOT / "docs" / "public"))


@pytest.mark.parametrize("example", EXAMPLES, ids=str)
def test_documentation_example_runs(example: CodeExample, eval_example: EvalExample):
    if example.prefix_settings().get("test") == "skip":
        pytest.skip("example marked test=skip")
    eval_example.run(example, module_globals={"__name__": "__main__"})
