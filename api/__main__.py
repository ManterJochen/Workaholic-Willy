"""Start the console: ``python -m api``.

Bound to 127.0.0.1 and not configurable to anything else from this entry point. There is no
authentication in front of these endpoints, and the machine running them is next to a robot arm.
Making the bind address a flag would turn "expose the cell to the network" into a typo. Remote access
means a reverse proxy with real authentication in front of this server, decided deliberately rather
than through a convenience flag.

The port is checked, then claimed, and only then announced. ``--port 99999`` used to print
"serving http://127.0.0.1:99999" and die eleven asyncio frames later in
``OverflowError: bind(): port must be 0-65535``, and a port another console already held printed the
same "serving" line before uvicorn found out. The banner is the operator's evidence that the thing
they started is the thing in their browser, so it is printed last of the three.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

from api.cell import Console, set_console
from src.config.loader import active_profile, set_active_profile

_HOST = "127.0.0.1"

#: What a TCP port is. 0 is excluded although the kernel accepts it: it means "any free port", and a
#: console whose URL the operator cannot predict is not a console they can open.
_PORT_RANGE = (1, 65535)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m api",
        description="The optional operator console. Localhost only, no authentication.",
    )
    parser.add_argument("--port", type=int, default=8000, help="port on 127.0.0.1 (default: 8000)")
    parser.add_argument(
        "--profile",
        default=None,
        help="profile chain the console configures, e.g. 'ur3e,tiltcam'. Default: the shipped tree.",
    )
    parser.add_argument("--data", default=None, help="config data directory (default: the repo's)")
    parser.add_argument(
        "--reload", action="store_true", help="restart on source changes (development only)"
    )
    args = parser.parse_args(argv)

    low, high = _PORT_RANGE
    if not low <= args.port <= high:
        # Measured: `python -m api --port 99999` printed "serving http://127.0.0.1:99999" and then
        # died eleven asyncio frames deep in `OverflowError: bind(): port must be 0-65535`. Not an
        # `OSError`, so uvicorn's own bind-failure handler (the thing that turns a bad port into
        # one readable line) never saw it either. A typo in a number deserves a sentence.
        print(
            f"--port {args.port} is not a port: a TCP port is {low}-{high}.",
            file=sys.stderr,
        )
        return 2

    try:
        import uvicorn
    except ImportError:
        print(
            "the console's dependencies are not installed.\n"
            "  pip install -r requirements.txt\n"
            "They are an optional extra on purpose: the library and every CLI run without them.",
            file=sys.stderr,
        )
        return 2

    if args.reload and args.data:
        # --reload runs the app in a worker subprocess that re-imports the module and never sees the
        # Console built here. The profile survives because it travels in WILLY_PROFILE; a data root
        # has no such channel, so accepting both would serve the shipped tree while claiming otherwise.
        print("--data cannot be combined with --reload.", file=sys.stderr)
        return 2

    if args.profile is not None:
        set_active_profile(args.profile)
    cell = Console(profile=args.profile if args.profile is not None else active_profile())
    if args.data:
        cell.root = Path(args.data).resolve()
    set_console(cell)

    try:
        report = cell.preflight()
    except Exception as exc:  # noqa: BLE001 (a broken tree is a message, not a traceback)
        print(f"config error:\n{exc}", file=sys.stderr)
        return 1

    # Printed before serving: the checklist is the first thing the console is for, and an operator
    # who can see it in the terminal knows the browser is showing them the real thing.
    blocking = len([c for c in report.checks if str(c.status) == "block"])
    chain = cell.profile or "(no profile)"
    print("Willy the Workaholic: operator console")
    print(f"  config    {cell.root}   profile: {chain}")
    print(f"  preflight {'0 blocking' if not blocking else f'{blocking} BLOCKING'}"
          f"  ({len(report.checks)} checks)")
    # The port is claimed before it is announced. This line used to print several frames before
    # anything bound, so starting a second console on the port the first one holds still printed
    # "serving", and the operator's evidence that the thing they started is the thing in their
    # browser was printed whether or not it was true.
    if (why_not := _why_the_port_will_not_bind(args.port)) is not None:
        print(
            f"cannot serve on {_HOST}:{args.port}: {why_not}\n"
            f"Another console is probably already running here. Stop it, or start this one on a "
            f"different port with --port.",
            file=sys.stderr,
        )
        return 1
    print(f"  serving   http://{_HOST}:{args.port}   (localhost only, no authentication)")
    # Flushed explicitly: uvicorn logs through the logging module to stderr, which is unbuffered,
    # while these go to a block-buffered stdout. Without this the banner is written into a buffer
    # that a Ctrl-C or a killed process discards, so the operator sees uvicorn's lines and never the
    # checklist summary that is the reason for starting the thing.
    sys.stdout.flush()

    if args.reload:
        uvicorn.run("api.app:app", host=_HOST, port=args.port, reload=True)
    else:
        from api.app import app

        uvicorn.run(app, host=_HOST, port=args.port, log_level="info")
    return 0


def _why_the_port_will_not_bind(port: int) -> str | None:
    """The reason this port cannot be served, or None. Binds it briefly and hands it straight back.

    SO_REUSEADDR is deliberately not set. On Windows it lets a second socket take a port another
    process is already listening on, which would turn this check into the opposite of an answer.

    And there is a window between this bind and uvicorn's, in which something else could take the
    port. It is microseconds wide and it costs a wrong banner, not a wrong server: uvicorn refuses
    the bind itself, logs the `OSError` and exits 1. What this removes is the ordinary case, a
    console already running, which was reported as success.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((_HOST, port))
    except OSError as error:
        return str(error)
    finally:
        probe.close()
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
