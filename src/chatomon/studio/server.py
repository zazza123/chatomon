"""Start the studio for an atom."""

from __future__ import annotations

import errno
import os
import socket
import sys
import threading
import webbrowser
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import uvicorn

from chatomon.atom import Atom
from chatomon.errors import ChatomonConfigError
from chatomon.studio._app import create_app
from chatomon.studio._loading import resolve
from chatomon.studio._presets import InvalidPresetError
from chatomon.studio._security import LOOPBACK, is_bind_all, new_token
from chatomon.studio._session import Session

__all__ = ["DEFAULT_PORT", "serve"]

DEFAULT_PORT = 8016
"""The studio's port: oxygen-16, the most common oxygen atom (8 protons, 8 neutrons).

The "o" of chatomon is the letter the atom replaces in the wordmark, and
chatomon has 8 letters. It also avoids the ports of similar tools: Gradio 7860,
ADK 8000, Microsoft Agent Framework DevUI 8080, `clai web` 7932.
"""


def serve(
    atom: Atom[Any] | Callable[[], Atom[Any]],
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    preset: str | os.PathLike[str] | None = None,
) -> None:
    """Open the studio for an atom and block until interrupted.

    The studio runs a copy of the atom with `capture_content="full"` and its
    own turn callback and tracer; the atom you pass is not changed. It prints
    the link of this run, which carries its access token, to stderr.

    Args:
        atom: The atom, or a function with no arguments that builds it.
        host: The address to listen on; another machine's address only with care.
        port: The port to listen on.
        open_browser: Whether to open the link in the browser.
        preset: The path of a preset exported from the studio, applied to the atom
            before the page opens. What differs from the exported session is
            printed as notes.

    Raises:
        ChatomonConfigError: If the atom is invalid or unsupported, the host is a
            bind-all address, the port is in use, or the preset cannot be read or
            applied; the message has one line for each problem of the preset.

    Examples:
        ```{.python test="skip"}
        from chatomon.studio import serve

        from speedypack.app import atom

        serve(atom)  # open the printed link, then chat with the atom
        ```

        Start with the orbital changes of an earlier session:

        ```{.python test="skip"}
        from chatomon.studio import serve

        from speedypack.app import atom

        serve(atom, preset="speedypack.preset.json")
        ```
    """  # noqa: DOC502 - raised by resolve() and _start()
    _start(
        resolve(atom, label="the atom given to serve()"),
        source="serve()",
        host=host,
        port=port,
        open_browser=open_browser,
        preset=preset,
    )


def _start(
    atom: Atom[Any],
    *,
    source: str,
    host: str,
    port: int,
    open_browser: bool,
    namespace: Mapping[str, object] | None = None,
    preset: str | os.PathLike[str] | None = None,
) -> None:
    if is_bind_all(host):
        raise ChatomonConfigError(
            f"the studio listens on one address; pass the address to use instead of {host!r}"
        )
    session = _session(atom, source, namespace, preset)
    _check_port(host, port)
    for note in session.notes:
        print(f"note: {note}", file=sys.stderr)
    token = new_token()
    shown = f"[{host}]" if ":" in host else host
    url = f"http://{shown}:{port}/?token={token}"
    print(f"chatomon studio is running at {url}", file=sys.stderr)
    if host not in LOOPBACK:
        print(
            "warning: the studio listens on a network address. Anyone who reaches it with "
            "this link can read the conversation and call the atom's models.",
            file=sys.stderr,
        )
    if open_browser:
        timer = threading.Timer(0.8, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    app = create_app(session, token=token, host=host, port=port)
    # Three seconds for open requests on Ctrl+C: a turn waiting on its model must not hold the exit.
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=3,
    )
    print("chatomon studio stopped.", file=sys.stderr)


def _session(
    atom: Atom[Any],
    source: str,
    namespace: Mapping[str, object] | None,
    preset: str | os.PathLike[str] | None,
) -> Session:
    """Create the session, with the changes of a preset file when one is given.

    Returns:
        The session the page will drive.

    Raises:
        ChatomonConfigError: If the file cannot be read, or the preset cannot be
            applied: the message has a heading and one line for each problem.
    """
    if preset is None:
        return Session(atom, source=source, namespace=namespace)
    shown = os.fspath(preset)
    if not shown:
        # An empty path would be read as the current directory.
        raise ChatomonConfigError("cannot read the preset: the path is empty")
    try:
        # Some editors put a byte order mark in front of the text: it is not part of it.
        text = Path(preset).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ChatomonConfigError(f"cannot read the preset {shown}: it is not UTF-8 text") from exc
    except OSError as exc:
        raise ChatomonConfigError(f"cannot read the preset {shown}: {exc.strerror or exc}") from exc
    try:
        return Session(atom, source=source, namespace=namespace, preset=text)
    except InvalidPresetError as exc:
        problems = "".join(f"\n  - {_plain(problem)}" for problem in exc.problems)
        raise ChatomonConfigError(f"the preset {shown} was not applied:{problems}") from exc


def _plain(text: str) -> str:
    """Write as escapes the characters a terminal would act on.

    A problem can quote the preset file, which may come from someone else: an
    escape sequence or a line break in it must not rewrite what the terminal shows.

    Returns:
        The text with every character that is not printable as its escape;
        letters of any alphabet stay as they are.
    """
    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode("ascii")
        for char in text
    )


def _check_port(host: str, port: int) -> None:
    if not 0 < port < 65536:
        raise ChatomonConfigError(f"the port must be between 1 and 65535, got {port}")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        if os.name == "posix":
            # Like the server itself: a port still in TIME_WAIT after a restart is free.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise ChatomonConfigError(
                    f"port {port} is in use; choose another one with port= or --port"
                ) from exc
            raise ChatomonConfigError(
                f"cannot listen on {host}:{port}: {exc.strerror or exc}"
            ) from exc
