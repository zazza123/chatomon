"""Ctrl+C stops a studio that is connected, as it stops one that is not."""

import json
import signal
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
from http.cookiejar import CookieJar

from tests.studio.spectral.fake_relay import RelayThread

STUDIO = """
    import sys

    from pydantic_ai.models.test import TestModel

    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector
    from chatomon.studio import serve
    from chatomon.studio._spectral import Bridge

    # The relay of the test is on loopback, without TLS.
    Bridge.insecure = True
    connector = PydanticAIConnector(TestModel(custom_output_text="Hi"))
    atom = Atom(name="shop", instructions="Be brief.", connector=connector)
    serve(atom, port=int(sys.argv[1]), open_browser=False)
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_ctrl_c_stops_a_studio_that_is_connected(tmp_path, monkeypatch):
    with RelayThread() as relay:
        monkeypatch.setenv("SPECTRAL_BRIDGE_API_KEY", relay.key)
        monkeypatch.setenv("SPECTRAL_BRIDGE_RELAY_URL", relay.url)
        command = [sys.executable, "-c", textwrap.dedent(STUDIO), str(free_port())]
        process = subprocess.Popen(command, cwd=tmp_path, stderr=subprocess.PIPE, text=True)
        try:
            assert process.stderr is not None
            # The banner comes first: the link is on the line scripts read.
            running = next(
                line for line in process.stderr if line.startswith("chatomon studio is running at")
            )
            link = running.split()[-1]
            base = link.split("/?")[0]
            browser = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
            deadline = time.monotonic() + 10
            while True:
                try:
                    browser.open(link).close()
                    break
                except OSError:
                    assert time.monotonic() < deadline, "the studio did not start"
                    time.sleep(0.1)
            asked = urllib.request.Request(
                f"{base}/api/spectral/connect",
                data=b'{"api_key_env": "SPECTRAL_BRIDGE_API_KEY"}',
                headers={"Content-Type": "application/json", "Origin": base},
            )
            browser.open(asked).close()
            while json.load(browser.open(f"{base}/api/spectral"))["status"] != "connected":
                assert time.monotonic() < deadline, "the studio did not connect"
                time.sleep(0.05)
            started = time.monotonic()
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=10) == 0
            assert time.monotonic() - started < 5
            printed = process.stderr.read()
            assert "◇ spectral  can no longer reach this atom" in printed
            assert printed.rstrip().endswith("chatomon studio stopped.")
            assert relay.key not in printed
        finally:
            if process.poll() is None:
                process.kill()
