"""Without its extra the add-on is not there: the studio is as it was."""

import subprocess
import sys

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
