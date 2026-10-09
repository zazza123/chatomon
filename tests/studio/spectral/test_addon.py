"""Without its extra the add-on is not there: the studio is as it was."""

import re
import subprocess
import sys
from pathlib import Path

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._app import create_app
from chatomon.studio._session import Session
from tests.studio.frameworks import scripted
from tests.studio.spectral.helpers import NAMED, ORIGIN, TOKEN, page


async def test_without_the_extra_the_state_and_the_routes_are_as_they_were(no_extra):
    connector = PydanticAIConnector(scripted("Hi"))
    atom = Atom(name="shop", instructions="Be brief.", connector=connector)
    app = create_app(Session(atom, source="app:atom"), token=TOKEN, host="127.0.0.1", port=8016)
    async with page(app) as client:
        assert "spectral" not in (await client.get("/api/state")).json()["studio"]
        assert (await client.get("/api/spectral")).status_code == 404
        gone = await client.post("/api/spectral/connect", json=NAMED, headers=ORIGIN)
        assert gone.status_code == 404
    paths = [getattr(route, "path", "") for route in app.routes]
    assert [path for path in paths if "spectral" in path] == []


STATIC = Path(__file__).resolve().parents[3] / "src" / "chatomon" / "studio" / "static"


def test_the_page_names_no_file_of_the_add_on():
    assert "spectral" not in (STATIC / "index.html").read_text(encoding="utf-8").lower()
    assert "spectral" not in (STATIC / "studio.css").read_text(encoding="utf-8").lower()


def test_the_page_asks_for_the_add_on_only_when_the_state_names_it():
    """Its files are under a folder of their own, and one line of `main.js` fetches them."""
    named = {
        path.name: re.findall(r"^.*spectral.*$", path.read_text(encoding="utf-8"), re.MULTILINE)
        for path in (STATIC / "js").glob("*.js")
    }
    lines = named.pop("main.js")
    assert named and all(not found for found in named.values()), named
    # No `import ... from` at the top: the file is asked for behind the entry in the state.
    assert not [line for line in lines if re.match(r"\s*import\b.*\bfrom\b", line)]
    source = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    guard = source.index("if (!state.studio.spectral) return;")
    assert guard < source.index('import("./spectral/index.js")')


def test_importing_the_studio_does_not_import_the_extra():
    code = (
        "import sys\n"
        "import chatomon.studio._app\n"
        "import chatomon.studio._spectral\n"
        "found = [name for name in sys.modules if name.split('.')[0] == 'spectral_bridge']\n"
        "assert not found, found\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
