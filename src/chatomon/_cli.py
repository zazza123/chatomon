"""The `chatomon` command. Standard library only: each subcommand imports its extra."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence

from chatomon.errors import ChatomonConfigError, MissingExtraError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chatomon", description="Tools for chatomon atoms.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    studio = commands.add_parser(
        "studio",
        help="chat with an atom in a local page and watch its orbitals decide",
        description="Open the chatomon studio for an atom.",
    )
    studio.add_argument(
        "atom",
        metavar="MODULE:ATTRIBUTE",
        help="where the atom is, e.g. speedypack.app:atom; it may be a function that builds it",
    )
    studio.add_argument(
        "--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)"
    )
    studio.add_argument("--port", type=int, default=None, help="port to listen on (default: 8016)")
    studio.add_argument("--no-browser", action="store_true", help="print the link, do not open it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `chatomon` command.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when `None`.

    Returns:
        The exit code: 0 on success, 1 on an error, 2 for wrong arguments.
    """
    args = _parser().parse_args(argv)
    try:
        from chatomon.studio import server
        from chatomon.studio._loading import load, namespace
    except MissingExtraError as exc:
        print(f"chatomon studio: {exc}", file=sys.stderr)
        return 1
    try:
        atom = load(args.atom)
    except ChatomonConfigError as exc:
        if exc.__cause__ is not None:
            traceback.print_exception(exc.__cause__, file=sys.stderr)
        print(f"chatomon studio: {exc}", file=sys.stderr)
        return 1
    port = args.port if args.port is not None else server.DEFAULT_PORT
    try:
        server._start(
            atom,
            source=args.atom,
            namespace=namespace(args.atom),
            host=args.host,
            port=port,
            open_browser=not args.no_browser,
        )
    except ChatomonConfigError as exc:
        print(f"chatomon studio: {exc}", file=sys.stderr)
        return 1
    return 0
