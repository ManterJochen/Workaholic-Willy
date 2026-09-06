"""`python -m datagen.assets`: fetch the real meshes, or check whether this machine has them.

    --check                    what is present, per source. Exit 0 only when every source is ready.
    --fetch --from DIR         import meshes from a local directory into the gitignored library.
    --attribution              print the attribution text the licences oblige.

The meshes are not in the repository, for the same reason the MediaPipe bundles are not: they are
hundreds of megabytes of third-party binaries, and an operator may hold a different revision from
the one measured against. `--check` is what turns "it should be there" into an exit code.

Exit codes: 0 ready, 1 something is missing, 2 the request itself was wrong (bad source, absent
origin). A missing mesh library and a mistyped flag must not look alike to a script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datagen.assets.library import (
    ATTRIBUTIONS,
    LICENSES,
    MESH_LIBRARY_DIR,
    OPTIONAL_SOURCES,
    SUPPORTED_SOURCES,
    MeshLibrary,
    declared_license,
    import_from_directory,
    write_sha_index,
)

_READY, _MISSING, _BAD_REQUEST = 0, 1, 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m datagen.assets",
        description="Real-mesh library for the scene generator: fetch, check, attribute.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report what is present and exit")
    mode.add_argument("--fetch", action="store_true", help="import meshes from --from")
    mode.add_argument("--attribution", action="store_true", help="print the ATTRIBUTION text")
    parser.add_argument(
        "--source", choices=sorted(SUPPORTED_SOURCES), action="append",
        help="restrict to one source (repeatable); default is all",
    )
    parser.add_argument("--from", dest="origin", metavar="DIR", help="directory to import from")
    parser.add_argument(
        "--limit", type=int, metavar="N", help="import at most N meshes per source (for a trial)",
    )
    parser.add_argument(
        "--library", default=str(MESH_LIBRARY_DIR), metavar="DIR", help="library root",
    )
    # Required for `--source custom`, and refused rather than defaulted. See `import_from_directory`.
    parser.add_argument(
        "--license", metavar="SPDX",
        help="licence your own meshes are under, e.g. `own` or `CC0-1.0` (required for --source custom)",
    )
    parser.add_argument(
        "--attribution-text", metavar="TEXT", default="",
        help="who to credit for your own meshes; recorded beside them and in the manifest",
    )
    return parser


def _sources(args: argparse.Namespace) -> list[str]:
    """Which sources this invocation acts on.

    The default excludes the optional ones. `custom` is a customer's own parts: almost no machine
    has any, so sweeping it into the default would turn `--check` red for every operator who uses
    only the research datasets, and would make a bare `--fetch` refuse for want of a licence nobody
    was being asked for. Naming it explicitly selects it.
    """
    if args.source:
        return sorted(args.source)
    return sorted(set(SUPPORTED_SOURCES) - OPTIONAL_SOURCES)


def _check(args: argparse.Namespace) -> int:
    library = MeshLibrary(args.library)
    counts = library.counts()
    ready = True
    print(f"mesh library: {Path(args.library).resolve()}\n")
    for source in _sources(args):
        n = counts.get(source, 0)
        ready = ready and n > 0
        licence = LICENSES.get(source) or declared_license(Path(args.library) / source)[0]
        print(f"  {source:10} {n:5d} mesh(es)   licence {licence or 'UNDECLARED'}")
        if n == 0:
            print(f"             MISSING; python -m datagen.assets --fetch --from <dir> "
                  f"--source {source}")
    print(f"\nready: {'yes' if ready else 'NO'}")
    return _READY if ready else _MISSING


def _fetch(args: argparse.Namespace) -> int:
    if not args.origin:
        print("--fetch needs --from DIR", file=sys.stderr)
        return _BAD_REQUEST

    origin_root = Path(args.origin)
    imported = []
    for source in _sources(args):
        # Accept either `<origin>/<source>/` or an origin that is the source directory.
        candidate = origin_root / source
        directory = candidate if candidate.is_dir() else origin_root
        try:
            entries = import_from_directory(
                source, directory, destination=args.library, limit=args.limit,
                license=args.license, attribution=args.attribution_text,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"{source}: {exc}", file=sys.stderr)
            return _BAD_REQUEST
        imported.extend(entries)
        print(f"  {source:10} {len(entries):5d} mesh(es) imported")

    index = write_sha_index(imported, Path(args.library) / "sha256.csv")
    print(f"\nSHA index: {index}")
    # Not "every mesh above is CC-BY": `custom` carries whatever licence the operator declared, and
    # a closing line that asserts a licence over meshes it never looked at is exactly the kind of
    # confident-and-wrong statement the licence gate exists to prevent.
    print("Each mesh above carries the licence its source declares, recorded in the manifest; run "
          "`--attribution` for the text that must ship with any dataset built from them.")
    return _READY


def _attribution(args: argparse.Namespace) -> int:
    library = MeshLibrary(args.library)
    print("Third-party 3-D assets rendered into this dataset\n")
    for source in _sources(args):
        n = len(library.available(source))
        text = ATTRIBUTIONS.get(source)
        if text is None:
            declared, credit = declared_license(Path(args.library) / source)
            text = (f"Custom meshes supplied by the operator ({credit or 'no attribution'}), "
                    f"{declared or 'NO LICENCE DECLARED'}.")
        print(f"{text}")
        print(f"    {n} mesh(es) in this library\n")
    return _READY


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.check:
        return _check(args)
    if args.fetch:
        return _fetch(args)
    return _attribution(args)


if __name__ == "__main__":  # pragma: no cover (CLI entry)
    raise SystemExit(main())
