"""A copy that provably is the tree, an original that provably did not move, and the tests the chain touches.

    .venv/Scripts/python.exe scripts/trial/tree_copy.py copy --original D:/dev/Workaholic-Willy --copy D:/willy_trial/customer_hand
    .venv/Scripts/python.exe scripts/trial/tree_copy.py complete --original D:/dev/Workaholic-Willy --copy D:/willy_trial/customer_hand
    .venv/Scripts/python.exe scripts/trial/tree_copy.py snapshot --tree D:/dev/Workaholic-Willy --curobo-content DIR --out before.json
    .venv/Scripts/python.exe scripts/trial/tree_copy.py compare --tree D:/dev/Workaholic-Willy --before before.json
    .venv/Scripts/python.exe scripts/trial/tree_copy.py where --expect-root D:/willy_trial/customer_hand --expect-engine coal
    .venv/Scripts/python.exe scripts/trial/tree_copy.py touched-tests --original D:/dev/Workaholic-Willy [--run]

A trial instrument for the customer chain, not product code. The trial runs the customer runbook in a copy of the
tree, so three things have to be proven rather than assumed: the copy holds every file git sees in the original byte
for byte (a CRLF rewrite is a different file), the chain's scripts resolve every artefact directory inside the copy and
not back in the original, and the original did not move while the trial ran.

Exit codes: 0 the answer is yes, 1 it is no and every difference is named, 2 the question cannot be asked (no git, no
tree). Standard library and git only, except ``where``, which imports the package it is asked about.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

#: The directories the chain's file steps write, and so the ones a trial must leave untouched in the original.
WATCHED_DIRS = (
    "config",
    "src/robot/safety/data",
    "src/robot/safety/planning/robot",
    "tests",
    "docs/runbooks",
)
#: What a test's source names when it reads an artefact family the chain writes.
ARTIFACT_PATTERNS = (
    "available_grippers(", "_hand_meshes", "_gripper_spheres", "ur_retract", "EVIDENCE_DIR", "bundles.json",
    "grippers/", "real_cell", "preflight", "planner_hand", "robot.*.yaml",
)
_SKIPPED_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


class CannotAsk(Exception):
    """The question cannot be asked here: exit 2."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(tree: Path, *args: str) -> bytes:
    if shutil.which("git") is None:
        raise CannotAsk("git is not on PATH")
    if not tree.is_dir():
        raise CannotAsk(f"no tree at {tree}")
    done = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, check=False)
    if done.returncode != 0:
        raise CannotAsk(f"git {' '.join(args)} in {tree} failed: {done.stderr.decode(errors='replace').strip()}")
    return done.stdout


def git_files(tree: Path) -> list[str]:
    """Every file git sees in ``tree``: tracked and untracked, ignored ones left out, deleted ones left out."""
    names = [name for name in _git(tree, "ls-files", "-co", "--exclude-standard", "-z").decode("utf-8").split("\0")
             if name]
    return sorted({name for name in names if (tree / name).is_file()})


# --- complete ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CopyReport:
    original: str
    copy: str
    checked: int
    missing: tuple[str, ...] = ()
    differing: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return 0 if not (self.missing or self.differing) else 1

    def render(self) -> str:
        head = f"copy {self.copy} of {self.original}: {self.checked} files checked"
        if not self.exit_code:
            return f"{head}, every one present with identical bytes"
        lines = [f"{head}, {len(self.missing)} missing and {len(self.differing)} with other bytes"]
        lines += [f"  missing   {name}" for name in self.missing]
        lines += [f"  differs   {name}" for name in self.differing]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"original": self.original, "copy": self.copy, "checked": self.checked,
                "missing": list(self.missing), "differing": list(self.differing), "exit_code": self.exit_code}


def complete(original: Path, copy: Path, *, exclude: tuple[str, ...] = ()) -> CopyReport:
    """Every file git sees in ``original`` exists in ``copy`` with identical sha256, except under ``exclude``."""
    if not copy.is_dir():
        raise CannotAsk(f"no copy at {copy}")
    wanted = [name for name in git_files(original)
              if not any(name.startswith(prefix.rstrip("/") + "/") or name == prefix for prefix in exclude)]
    missing, differing = [], []
    for name in wanted:
        theirs = copy / name
        if not theirs.is_file():
            missing.append(name)
        elif _sha256(theirs) != _sha256(original / name):
            differing.append(name)
    return CopyReport(original=str(original), copy=str(copy), checked=len(wanted), missing=tuple(missing),
                      differing=tuple(differing))


def copy_tree(original: Path, copy: Path, *, also: tuple[str, ...] = ()) -> int:
    """Copy every file git sees in ``original`` into ``copy``, and the directories ``also`` names whole.

    What git sees is the tree; ``also`` is for ignored directories a test reads (meshes, caches), never for the
    interpreters, which the trial reaches in the original through environment variables. Refuses a copy that exists.
    """
    if copy.exists():
        raise CannotAsk(f"{copy} exists: a trial copies into a fresh directory, so nothing stale is in it")
    if _inside(copy, original):
        # A copy named D:willy_trial is relative to the drive's current directory and can land inside the original
        # tree, and a trial that writes into the tree it must leave unchanged proves nothing.
        raise CannotAsk(f"{copy} resolves to {copy.resolve()}, inside the original {original.resolve()}")
    names = git_files(original)
    for name in names:
        target = copy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / name, target)
    count = len(names)
    for relative in also:
        source = original / relative
        if not source.is_dir():
            raise CannotAsk(f"--also {relative}: no directory at {source}")
        for path in source.rglob("*"):
            if path.is_file() and not _SKIPPED_PARTS & set(path.parts):
                target = copy / path.relative_to(original)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    count += 1
    return count


# --- snapshot and compare ---------------------------------------------------------------------------------------------


def _watched(tree: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in WATCHED_DIRS:
        root = tree / relative
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not _SKIPPED_PARTS & set(path.parts):
                hashes[path.relative_to(tree).as_posix()] = _sha256(path)
    return hashes


def _content(curobo_content: "Path | None") -> dict[str, Any]:
    if curobo_content is None:
        return {"dir": None, "listing": [], "descriptors": {}}
    robots = curobo_content / "configs" / "robot"
    listing = sorted(path.name for path in robots.iterdir()) if robots.is_dir() else []
    descriptors = {path.name: _sha256(path) for path in sorted(robots.glob("willy_*.yml"))} if robots.is_dir() else {}
    return {"dir": str(curobo_content), "listing": listing, "descriptors": descriptors}


def snapshot(tree: Path, *, curobo_content: "Path | None" = None) -> dict[str, Any]:
    """What a trial must leave unchanged in the original: git's status, the watched files, the cuRobo descriptors."""
    return {
        "tree": str(tree),
        "git_status": _git(tree, "status", "--porcelain=v1", "-uall").decode("utf-8", errors="replace"),
        "watched": _watched(tree),
        "curobo_content": _content(curobo_content),
    }


@dataclass(frozen=True)
class CompareReport:
    tree: str
    status_changed: bool
    changed: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    content: tuple[str, ...] = field(default=())

    @property
    def exit_code(self) -> int:
        return 0 if not (self.status_changed or self.changed or self.added or self.removed or self.content) else 1

    def render(self) -> str:
        if not self.exit_code:
            return f"{self.tree}: unchanged since the snapshot (git status, watched files, cuRobo descriptors)"
        lines = [f"{self.tree}: CHANGED since the snapshot"]
        if self.status_changed:
            lines.append("  git status differs")
        lines += [f"  changed   {name}" for name in self.changed]
        lines += [f"  added     {name}" for name in self.added]
        lines += [f"  removed   {name}" for name in self.removed]
        lines += [f"  content   {name}" for name in self.content]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"tree": self.tree, "status_changed": self.status_changed, "changed": list(self.changed),
                "added": list(self.added), "removed": list(self.removed), "content": list(self.content),
                "exit_code": self.exit_code}


def compare(tree: Path, before: dict[str, Any]) -> CompareReport:
    content_dir = before.get("curobo_content", {}).get("dir")
    now = snapshot(tree, curobo_content=Path(content_dir) if content_dir else None)
    old, new = before["watched"], now["watched"]
    was, is_ = before["curobo_content"], now["curobo_content"]
    content = sorted(
        {name for name in set(was["listing"]) ^ set(is_["listing"])}
        | {name for name in set(was["descriptors"]) & set(is_["descriptors"])
           if was["descriptors"][name] != is_["descriptors"][name]}
    )
    return CompareReport(
        tree=str(tree), status_changed=before["git_status"] != now["git_status"],
        changed=tuple(sorted(name for name in set(old) & set(new) if old[name] != new[name])),
        added=tuple(sorted(set(new) - set(old))), removed=tuple(sorted(set(old) - set(new))),
        content=tuple(content),
    )


# --- where ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WhereReport:
    expected_root: str
    paths: tuple[tuple[str, str], ...]
    outside: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return 0 if not (self.outside or self.notes) else 1

    def render(self) -> str:
        lines = [f"where the chain resolves, against {self.expected_root}"]
        lines += [f"  {name:<24} {path}" for name, path in self.paths]
        lines += [f"  OUTSIDE   {name}" for name in self.outside]
        lines += [f"  WRONG     {note}" for note in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"expected_root": self.expected_root, "paths": dict(self.paths), "outside": list(self.outside),
                "notes": list(self.notes), "exit_code": self.exit_code}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def where(*, expect_root: Path, expect_curobo_python: "str | None" = None,
          expect_engine: "str | None" = None) -> WhereReport:
    """Where the chain's artefact directories resolve for the package this script imports, and which fall outside.

    ``src`` is a namespace package with no ``__file__``, so where it was imported from is read off ``src.config``.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import src.config
    from src.config.loader import _DEFAULT_DATA_DIR
    from src.robot.safety.planning import environment
    from src.robot.safety.planning.evidence import EVIDENCE_DIR
    from src.robot.safety.planning.hand import sphere_map_path
    from src.robot.safety.planning.robot.retract_table import TABLE_PATH
    from src.utility.paths import logs_dir, project_root

    rooted = (
        ("src", Path(src.config.__file__).resolve()),
        ("registry", (_DEFAULT_DATA_DIR / "grippers").resolve()),
        ("collision meshes", Path(environment.COLLISION_MESH_DIR).resolve()),
        ("evidence", Path(EVIDENCE_DIR).resolve()),
        ("retract table", Path(TABLE_PATH).resolve()),
        ("sphere maps", sphere_map_path("robotiq_2f85").parent.resolve()),
        ("project root", Path(project_root()).resolve()),
        ("logs", Path(logs_dir()).resolve()),
    )
    paths = [(name, str(path)) for name, path in rooted]
    outside = [f"{name} {path}" for name, path in rooted if not _inside(path, expect_root)]
    notes: list[str] = []
    curobo_python = environment.curobo_python_path()
    paths.append(("cuRobo python", curobo_python))
    if expect_curobo_python is not None and Path(curobo_python).resolve() != Path(expect_curobo_python).resolve():
        notes.append(f"cuRobo python is {curobo_python}, expected {expect_curobo_python}")
    engine = environment.import_collision_engine()[1]
    paths.append(("collision engine", str(engine)))
    if expect_engine is not None and engine != expect_engine:
        notes.append(f"the collision engine is {engine!r}, expected {expect_engine!r}")
    return WhereReport(expected_root=str(expect_root), paths=tuple(paths), outside=tuple(outside), notes=tuple(notes))


# --- touched-tests ----------------------------------------------------------------------------------------------------


def touched_tests(original: Path, *, patterns: "tuple[str, ...]" = ARTIFACT_PATTERNS) -> list[str]:
    """The test files git status names under tests/, and every test whose source names an artefact the chain writes."""
    status = _git(original, "status", "--porcelain=v1", "-uall", "--", "tests/").decode("utf-8", errors="replace")
    named = set()
    for line in status.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        name = Path(path).name
        if name.startswith("test_") and name.endswith(".py") and (original / path).is_file():
            named.add(Path(path).as_posix())
    for test_file in sorted((original / "tests").glob("test_*.py")):
        text = test_file.read_text(encoding="utf-8", errors="replace")
        if any(pattern in text for pattern in patterns):
            named.add(test_file.relative_to(original).as_posix())
    return sorted(named)


# --- the command line -------------------------------------------------------------------------------------------------


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Prove a trial copy is the tree, and that the tree did not move.")
    verbs = parser.add_subparsers(dest="verb", required=True)
    zero = verbs.add_parser("copy", help="copy every file git sees, and whole directories named by --also")
    zero.add_argument("--original", required=True)
    zero.add_argument("--copy", required=True)
    zero.add_argument("--also", action="append", default=[], help="a relative directory copied whole")
    one = verbs.add_parser("complete", help="every file git sees in the original is in the copy, byte for byte")
    one.add_argument("--original", required=True)
    one.add_argument("--copy", required=True)
    one.add_argument("--exclude", action="append", default=[], help="a relative path prefix to leave out")
    two = verbs.add_parser("snapshot", help="record what the trial must leave unchanged in the original")
    two.add_argument("--tree", required=True)
    two.add_argument("--curobo-content", default=None)
    two.add_argument("--out", required=True)
    three = verbs.add_parser("compare", help="the original against a snapshot")
    three.add_argument("--tree", required=True)
    three.add_argument("--before", required=True)
    four = verbs.add_parser("where", help="where the chain's artefact directories resolve")
    four.add_argument("--expect-root", required=True)
    four.add_argument("--expect-curobo-python", default=None)
    four.add_argument("--expect-engine", default=None)
    five = verbs.add_parser("touched-tests", help="the tests the chain touches, derived")
    five.add_argument("--original", required=True)
    five.add_argument("--run", action="store_true", help="run them with this interpreter, in the current directory")
    args = parser.parse_args(argv)

    try:
        if args.verb == "copy":
            count = copy_tree(Path(args.original), Path(args.copy), also=tuple(args.also))
            print(f"copied {count} files from {args.original} into {args.copy}")
            return 0
        if args.verb == "complete":
            report: Any = complete(Path(args.original), Path(args.copy), exclude=tuple(args.exclude))
        elif args.verb == "snapshot":
            taken = snapshot(Path(args.tree), curobo_content=Path(args.curobo_content) if args.curobo_content else None)
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(taken, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            print(f"snapshot of {args.tree}: {len(taken['watched'])} watched files -> {out}")
            return 0
        elif args.verb == "compare":
            report = compare(Path(args.tree), json.loads(Path(args.before).read_text(encoding="utf-8")))
        elif args.verb == "where":
            report = where(expect_root=Path(args.expect_root), expect_curobo_python=args.expect_curobo_python,
                           expect_engine=args.expect_engine)
        else:
            tests = touched_tests(Path(args.original))
            print("\n".join(tests))
            if not args.run:
                return 0
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            return subprocess.run([sys.executable, "-m", "pytest", *tests, "-p", "no:cacheprovider"], env=env,
                                  check=False).returncode
    except CannotAsk as exc:
        print(f"cannot ask: {exc}", file=sys.stderr)
        return 2
    print(report.render())
    return int(report.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
