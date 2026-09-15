"""``python -m src.config``: validate the configuration tree, and ask it about itself.

Exit codes:
* 0: the command succeeded (the config validates; a query ran)
* 1: :class:`src.config.ConfigError` raised (file, parse or schema error, or a gripper registry that
  cannot be read or does not hold the hand the cell names)
* 2: bad CLI arguments

A query that finds nothing still exits 0: ``explain`` reports the key as unknown and offers near
matches, ``where`` prints "no config key matches".

Examples::

    python -m src.config                             # validate the default tree
    python -m src.config --data ./data                # validate a custom tree
    python -m src.config --print                      # also print the parsed config

    # ...and the questions reading the YAML cannot answer:
    python -m src.config explain robot.safety.self_collision.planner_margin_mm --profile sim,ur3e
    python -m src.config where gripper

``explain`` reports a key's type, constraints, default, which file and layer set the winning value,
the whole override chain, and the comment written above that line, which ``yaml.safe_load`` discards.
``where`` searches the schema rather than the files, so it finds the fields no YAML mentions
(measured: 107 of them, including ``robot.ur.model``).
"""

from __future__ import annotations

import argparse
import sys

from src.contracts import UNSET

from .tree import ConfigTree


def _emit(text: str) -> None:
    """Print without failing on the console's encoding.

    The config's own comments contain box-drawing characters and typographic dashes, and a stock
    Windows console is cp1252, where printing one raises UnicodeEncodeError. Unrepresentable
    characters are replaced so that the answer still reaches the operator.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
        return
    except UnicodeEncodeError:
        pass
    try:
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    except UnicodeEncodeError:  # a console that cannot even take its own replacement character
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _shared_options() -> argparse.ArgumentParser:
    """The flags that work on both sides of the subcommand, a fresh parser on every call.

    Two argparse behaviours interact here, and neither announces itself:

    1. ``default=SUPPRESS`` is what makes ``config --profile sim explain KEY`` work at all. With a
       normal default the subparser's copy of the flag writes that default into the namespace after
       the top-level parser stored the real value, so a flag given before the subcommand is lost and
       the tool answers about the base tree.
    2. :meth:`ArgumentParser.set_defaults` mutates the Action objects it matches, and ``parents=``
       shares Action instances rather than copying them, so with one shared parent the top level's
       ``set_defaults`` rewrites the subparser's ``SUPPRESS`` back to ``None`` and reinstates (1).

    A fresh parser per caller keeps ``set_defaults`` local to the top-level parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--data", metavar="DIR", default=argparse.SUPPRESS,
        help="path to a custom data/ directory (default: config)",
    )
    # `--print` stays on this shared parent although only the default command reads it. Without it,
    # `config --print explain KEY` fails with argparse's generic "unrecognized arguments"; with it,
    # the parse succeeds and `main` can refuse by naming the command that does accept the flag.
    parser.add_argument(
        "--print", action="store_true", default=argparse.SUPPRESS,
        help="dump the validated AppConfig as JSON",
    )
    parser.add_argument(
        "--profile", metavar="CHAIN", default=argparse.SUPPRESS,
        help="profile layer chain to load, e.g. 'sim,ur3e' (default: whatever WILLY_PROFILE says)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Shared options live on a parent parser so they work on both sides of the subcommand:
    # `config --profile X explain KEY` and `config explain KEY --profile X` both parse.
    parser = argparse.ArgumentParser(
        prog="python -m src.config", parents=[_shared_options()],
        description="Validate the YAML configuration tree, or ask it where a value came from.",
    )
    parser.set_defaults(data=None, profile=None, **{"print": False})
    sub = parser.add_subparsers(dest="command")
    explain_cmd = sub.add_parser(
        "explain", parents=[_shared_options()],
        help="what a key means, where it was set, and why that value",
    )
    explain_cmd.add_argument("key", help="dotted key, e.g. robot.sim.robot_model")
    where_cmd = sub.add_parser(
        "where", parents=[_shared_options()],
        help="find config keys by substring: searches the schema, so it finds unwritten ones",
    )
    where_cmd.add_argument("needle", help="substring to look for, e.g. gripper")
    where_cmd.add_argument(
        "--tier", default=None,
        help="show only this tier (safety | site | tuned | advanced), or 'all'. A tier is a display "
             "filter: every field stays settable whether it is shown or not.",
    )
    where_cmd.add_argument(
        "--limit", type=int, default=40, metavar="N",
        help="max keys listed per tier (default 40). `where grasping` matches 188 keys, so the default "
             "elides most of them; raise it when you are hunting rather than browsing.",
    )
    dec_cmd = sub.add_parser(
        "decisions", parents=[_shared_options()],
        help="only the values that differ from their schema default: what this cell actually decides",
    )
    dec_cmd.add_argument(
        "--section", default=None, metavar="PREFIX",
        help="limit to a dotted prefix, e.g. robot.safety",
    )
    dec_cmd.add_argument(
        "--tier", default=None,
        help="show only this tier (safety | site | tuned | advanced), or 'all'",
    )
    args = parser.parse_args(argv)

    # A flag that cannot apply is rejected rather than accepted and ignored. Both flags below ride
    # the shared parent parser onto the subcommands, where nothing reads them: `--print` dumps the
    # validated tree and is read only after every subcommand has returned, and `find_keys` behind
    # `where` takes no root at all, so `--data` and `--profile` cannot change an answer that comes
    # from the schema compiled into this checkout. Exit 2 is argparse's own bad-arguments code,
    # documented at the top of this file, and each message names the command that accepts the flag.
    if args.command is not None and args.print:
        parser.error(
            f"--print dumps the whole validated config, which is what the default command does; "
            f"'{args.command}' answers a narrower question. Run 'python -m src.config --print'."
        )
    if args.command == "where":
        for flag, value in (("--data", args.data), ("--profile", args.profile)):
            if value is not None:
                parser.error(
                    f"'where' searches the schema compiled into this checkout, not a YAML tree, so "
                    f"{flag} cannot change its answer. Drop {flag}, or use 'explain' / 'decisions', "
                    f"which do read the tree."
                )

    if args.command == "where":  # schema-only: needs no tree and cannot fail on a broken one
        from .explain import find_keys

        _emit(find_keys(args.needle, limit=args.limit, tier=args.tier))
        return 0

    # `ConfigTree` derives root, chain and layers once and its ask methods take only the key, so a
    # value and its provenance cannot come from two different expressions. The profile is passed as
    # an argument rather than through `WILLY_PROFILE`, so the `source` that `_validated_chain`
    # carries names what the operator typed and not an environment variable nobody set.
    loaded = ConfigTree.from_directory(
        root=args.data if args.data is not None else UNSET,
        profile=args.profile if args.profile is not None else UNSET,
    ).load()
    if not loaded.ok:
        print(loaded.render(), file=sys.stderr)
        return loaded.exit_code

    if args.command == "decisions":
        _emit(loaded.decisions(section=args.section, tier=args.tier))
        return 0

    if args.command == "explain":
        _emit(loaded.explain(args.key).render())
        return 0

    if args.print:
        _emit(loaded.config.model_dump_json(indent=2))
    else:
        _emit(loaded.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
