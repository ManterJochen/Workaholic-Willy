"""Fetch the UR arm meshes a cuRobo descriptor builds from, pinned to one commit and held to a committed sha256.

    ext_deps/curobo_env/python.exe scripts/curobo/fetch_ur_meshes.py            # fetch what is missing, verify all
    ext_deps/curobo_env/python.exe scripts/curobo/fetch_ur_meshes.py --check    # verify only, no network

Isaac's UR descriptions reference ``package://ur_description/meshes/<arm>/{collision,visual}/...``, and
``build_ur_config.py`` resolves those under the cuRobo asset root. cuRobo ships that root with ur5e and ur10e meshes
only, so without this ur3, ur3e and ur5 cannot be built at all, ur16e builds from 10 of its 14 meshes, and a fresh
``install.ps1`` stops at its ur3e descriptor.

The files come from Universal Robots' ROS 2 description (BSD-3-Clause), the source cuRobo's own
``ur_description/LICENSE`` names, at the commit this file and ``ur_meshes.sha256`` pin. At that commit the ur5e and
ur10e collision meshes are byte identical to the ones cuRobo ships. Every file that has to be written is verified before
the first one is, a file already installed with other bytes is refused and left as it is, and ``--check`` needs no
network.

Each file is written as the commit holds it, read from git's object store rather than checked out. A checkout converts
line endings on a machine with ``core.autocrlf=true``, and a pin hashed from a converted file matches no other machine.

Standard library only, because it runs in the cuRobo environment beside the build and is also loaded by path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

__all__ = [
    "COMMIT", "DESCRIPTION", "DESCRIPTION_MANIFEST", "MANIFEST", "REPO_URL", "MeshPinError", "check", "fetch_source",
    "install", "main", "mesh_owner", "read_manifest",
]

REPO_URL = "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git"
#: The pinned commit. The header of ``ur_meshes.sha256`` names the same one, and the two are held together.
COMMIT = "89bbe795f38a7ab00fb66fe8831dfff79dc99edf"
MANIFEST = Path(__file__).resolve().with_name("ur_meshes.sha256")
#: The same commit's config and xacro, vendored in this repository rather than installed into the asset root: the
#: kinematics, joint limits, physical and visual parameters per model that the URDF writer reads. Written once with
#: ``--dest <DESCRIPTION> --manifest <DESCRIPTION_MANIFEST>``, and a pin bump reruns that command.
DESCRIPTION = Path(__file__).resolve().with_name("ur_ros2_description")
DESCRIPTION_MANIFEST = Path(__file__).resolve().with_name("ur_ros2_description.sha256")
REPO = Path(__file__).resolve().parents[2]

#: UR16e takes its base, shoulder and wrists from UR10e: Universal Robots' own config/ur16e/visual_parameters.yaml and
#: Isaac's ur16e.urdf both reference them there. Only its upper arm and forearm are its own.
_BORROWED = {"ur16e": ("ur10e", frozenset({"base", "shoulder", "wrist1", "wrist2", "wrist3"}))}


class MeshPinError(RuntimeError):
    """A mesh that does not match its pin, or one installed with other bytes. Nothing was written."""


def mesh_owner(model: str, link: str) -> str:
    """The arm whose mesh folder holds ``link`` for ``model``."""
    borrowed = _BORROWED.get(model)
    return borrowed[0] if borrowed is not None and link in borrowed[1] else model


def read_manifest(path: Path) -> dict[str, str]:
    """``{path relative to ur_description: sha256}``. Lines starting with ``#`` are the header."""
    pinned: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        digest, _, relative = line.partition("  ")
        if not relative.strip():
            raise MeshPinError(f"{path}:{number} is not '<sha256>  <path>': {line!r}")
        pinned[relative.strip()] = digest.strip().lower()
    return pinned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(dest: Path, *, manifest: Path = MANIFEST) -> list[str]:
    """What is wrong with the meshes under ``dest``, one sentence per file; empty when every pinned mesh is in place."""
    problems: list[str] = []
    for relative, digest in sorted(read_manifest(manifest).items()):
        path = Path(dest) / relative
        if not path.is_file():
            problems.append(f"{relative} is missing under {dest}")
        elif _sha256(path) != digest:
            problems.append(f"{relative} under {dest} has other bytes than its pin {digest[:12]}")
    return problems


def install(*, source: Path, dest: Path, manifest: Path = MANIFEST) -> list[str]:
    """Copy every pinned mesh ``dest`` lacks from ``source``, all or nothing, and return the paths written.

    A mesh already in place with its pinned bytes is left alone and needs no source. Everything else is decided before
    the first copy: a source file that does not match its pin, or an installed file with other bytes, refuses the
    whole install with nothing written.
    """
    to_write: list[str] = []
    problems: list[str] = []
    for relative, digest in sorted(read_manifest(manifest).items()):
        target = Path(dest) / relative
        if target.is_file():
            if _sha256(target) != digest:
                problems.append(f"{relative} is already installed under {dest} with other bytes, and is left as it is")
            continue
        origin = Path(source) / relative
        if not origin.is_file():
            problems.append(f"{relative} is not in the source {source}")
        elif _sha256(origin) != digest:
            problems.append(f"{relative} in the source does not match its pin {digest[:12]}")
        else:
            to_write.append(relative)
    if problems:
        raise MeshPinError("nothing was written: " + "; ".join(problems))
    for relative in to_write:
        target = Path(dest) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / relative, target)
    return to_write


def fetch_source(workdir: Path, relatives: list[str]) -> Path:
    """The files ``relatives`` at the pinned commit, written under ``workdir`` as the commit holds them.

    The fetch is shallow and blobless, and each file is read with ``git cat-file``, which fetches its blob on demand.
    Nothing is checked out, so no line ending setting, attribute or filter of this machine touches a byte.
    """
    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=workdir, check=True, capture_output=True).stdout

    git("init", "-q")
    git("remote", "add", "origin", REPO_URL)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    for relative in relatives:
        target = Path(workdir) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("cat-file", "blob", f"{COMMIT}:{relative}"))
    return Path(workdir)


def _default_dest() -> Path:
    """The ``ur_description`` folder under the cuRobo asset root: the override, cuRobo's own answer, or this repo's."""
    override = os.environ.get("WILLY_CUROBO_CONTENT")
    if override:
        return Path(override) / "assets" / "robot" / "ur_description"
    try:
        from curobo.content import get_content_root  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 (outside the cuRobo environment the repository's install root answers)
        return REPO / "ext_deps" / "curobo" / "curobo" / "content" / "assets" / "robot" / "ur_description"
    return Path(str(get_content_root())) / "assets" / "robot" / "ur_description"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify the pinned UR arm meshes for the cuRobo descriptors.")
    parser.add_argument("--dest", type=Path, default=None, help="the ur_description folder under the cuRobo asset root")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--source", type=Path, default=None,
                        help="a folder holding the pinned files at their relative paths, used instead of fetching")
    parser.add_argument("--check", action="store_true", help="verify what is installed and fetch nothing")
    args = parser.parse_args(argv)
    dest = args.dest if args.dest is not None else _default_dest()
    total = len(read_manifest(args.manifest))

    problems = check(dest, manifest=args.manifest)
    if args.check or not problems:
        for problem in problems:
            print(f"[bad ] {problem}")
        print(f"{total - len(problems)} of {total} pinned UR files in place under {dest}")
        return 1 if problems else 0

    wanted = [problem.split(" ", 1)[0] for problem in problems]
    try:
        if args.source is not None:
            written = install(source=args.source, dest=dest, manifest=args.manifest)
        else:
            with tempfile.TemporaryDirectory(prefix="willy_ur_meshes_", ignore_cleanup_errors=True) as workdir:
                source = fetch_source(Path(workdir), wanted)
                written = install(source=source, dest=dest, manifest=args.manifest)
    except MeshPinError as exc:
        print(f"refused: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        said = (exc.stderr or b"").decode("utf-8", "replace").strip()
        print(f"could not fetch {REPO_URL} at {COMMIT}: git {' '.join(exc.cmd[1:3])} exited {exc.returncode}: {said}")
        return 1
    except OSError as exc:
        print(f"could not fetch {REPO_URL} at {COMMIT}: {type(exc).__name__}: {exc}")
        return 1

    remaining = check(dest, manifest=args.manifest)
    for problem in remaining:
        print(f"[bad ] {problem}")
    print(f"wrote {len(written)} pinned file(s) from {REPO_URL} at {COMMIT[:12]}; "
          f"{total - len(remaining)} of {total} pinned UR files in place under {dest}")
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
