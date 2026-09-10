"""Gap D2 - markdown link-resolution test.

The past quickstart drift (broken Phase-U link, etc.) landed because no test resolved doc links. This
scans EVERY markdown document in the repository, extracts every relative markdown link, strips the
anchor/query, and asserts the target file exists. http(s)/mailto and pure-anchor links are skipped.
Runs in the mock/CI suite (pure filesystem).

The set used to be enumerated by hand: `QUICKSTART.md`, `README.md`, `docs/*.md`, `docs/runbooks/*.md`,
`docs/guide/*.md` and every `backend/src/**/*_README.md`. Two of those roots stopped existing when the
library moved to `src/` and every `<pkg>_README.md` became `README.md`, which left a hand-written list
silently scanning nothing -- the exact failure mode of a hand-written list. Enumerating the tree
instead cannot rot that way, and it covers the documents the old list never reached (`api/`,
`datagen/`, `scripts/`, `frontend/`, `config/`) as well as the ones it did.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")

#: Fenced blocks and inline code spans, stripped before links are looked for.
#:
#: ⛔ MATHS READS AS A LINK. `T_dh[frame](q)` in `safety-math.md` matches `_LINK` with a target of
#: "q", which is not a path and never was. Widening this test to the top level of `docs/` surfaced it
#: on the first run, and the honest fix is to stop scanning code rather than to special-case one
#: filename. It also removes a whole class of future false alarm: every one of these pages carries
#: formulas, and a formula is not a link.
_CODE = re.compile(r"```.*?```|`[^`]*`", re.S)


#: Directories that hold no documentation of ours: third-party trees, build output and run logs.
#: Dot-directories go with them (`.git`, `.venv`, and the `.commits/` message archive), which is why
#: this is a name test rather than a fixed list of four.
_SKIP_DIRS = frozenset({"node_modules", "ext_deps", "logs"})


def _docs() -> list[Path]:
    """Every markdown document in the repository, third-party and generated trees aside."""
    return sorted(
        path
        for path in _ROOT.rglob("*.md")
        if path.is_file()
        and not any(
            part in _SKIP_DIRS or part.startswith(".")
            for part in path.relative_to(_ROOT).parts[:-1]
        )
    )


@pytest.mark.parametrize("doc", _docs(), ids=lambda f: str(f.relative_to(_ROOT)).replace("\\", "/"))
def test_relative_markdown_links_resolve(doc: Path) -> None:
    text = _CODE.sub(" ", doc.read_text(encoding="utf-8"))
    broken: list[str] = []
    for match in _LINK.finditer(text):
        target = match.group(1).strip()
        if not target or target.startswith(_SKIP_PREFIXES):
            continue
        path_part = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not path_part:  # pure in-page anchor
            continue
        resolved = (doc.parent / path_part).resolve()
        if not resolved.exists():
            broken.append(target)
    assert not broken, f"{doc.relative_to(_ROOT)} has unresolved relative links: {broken}"


def _canon_docs() -> list[Path]:
    """The same documents, judged by the weaker of the two link conventions.

    This case used to hold `CLAUDE.md` plus `.ai-memory/*.md`, a session-facing canon that linked into
    the source with the repo-root convention (`](backend/...)`) rather than relative to itself. Neither
    file travelled into this repository, so the case was scanning an empty list and asserting over
    nothing. Pointing it at the same tree as the strict case keeps what it was for: a link that
    resolves NEITHER from the repository root NOR from the document's own directory is a genuinely
    dead target -- a moved or deleted file -- whichever convention its author had in mind.
    """
    return _docs()


@pytest.mark.parametrize("doc", _canon_docs(), ids=lambda f: str(f.relative_to(_ROOT)).replace("\\", "/"))
def test_canon_source_links_point_at_real_files(doc: Path) -> None:
    """A canon link must resolve -- from the repo root OR the file's own directory.

    A document may legitimately link with either convention, so this accepts EITHER resolution and
    flags only links that resolve NEITHER way -- i.e. a genuinely dead target (a moved or deleted file),
    which is the rot this guards against, not a convention choice.
    """
    text = _CODE.sub(" ", doc.read_text(encoding="utf-8"))
    dead: list[str] = []
    for match in _LINK.finditer(text):
        target = match.group(1).strip()
        if not target or target.startswith(_SKIP_PREFIXES):
            continue
        path_part = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not path_part:
            continue
        if (doc.parent / path_part).exists() or (_ROOT / path_part).exists():
            continue
        dead.append(target)
    assert not dead, f"{doc.relative_to(_ROOT)} links at files that do not exist (moved/deleted?): {dead}"


# ---------------------------------------------------------------------------
# `python -m ...` commands, which rot the same way a link does and were unguarded
# ---------------------------------------------------------------------------

#: `python -m a.b.c`, as an operator would copy it out of a fenced block.
#:
#: ⛔ MEASURED 2026-09-10 in the sibling tree: six commands across four operator documents named
#: modules the interpreter cannot find, because the old top-level `examples/` package had been
#: deleted and replaced by `scripts/examples/` while the documents kept naming it. Nothing resolved
#: a command the way `test_relative_markdown_links_resolve` resolves a link. A dead link is visibly
#: dead in a browser; a dead command looks like a broken install to the operator who runs it, which
#: is the more expensive failure.
_RUN_MODULE = re.compile(r"python\s+-m\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")

#: Modules that come from the interpreter or from the pinned requirements, not from this checkout.
_NOT_OURS = frozenset({
    "pytest", "unittest", "venv", "pip", "mypy", "ruff", "coverage", "build", "twine",
    "http", "json", "compileall", "site", "ensurepip", "IPython", "jupyter", "torch",
})


def _repo_packages() -> frozenset[str]:
    """Top-level importable directories of this checkout, measured rather than listed.

    A hard-coded list would stop seeing a package the day one is added, which is the same defect
    this whole module exists to catch one level up.
    """
    return frozenset(
        entry.name
        for entry in _ROOT.iterdir()
        if entry.is_dir()
        and not entry.name.startswith((".", "_"))
        and entry.name not in _SKIP_DIRS
    )


def _module_is_runnable(dotted: str) -> bool:
    """What `python -m dotted` needs on disk: a module file, or a package with a `__main__`."""
    parts = dotted.split(".")
    as_module = _ROOT.joinpath(*parts).with_suffix(".py")
    as_package_main = _ROOT.joinpath(*parts) / "__main__.py"
    return as_module.is_file() or as_package_main.is_file()


@pytest.mark.parametrize("doc", _docs(), ids=lambda f: str(f.relative_to(_ROOT)).replace("\\", "/"))
def test_documented_run_module_commands_name_real_modules(doc: Path) -> None:
    """Every `python -m <ours>` an operator can copy out of these docs must exist on disk.

    Scoped to modules whose first segment is a directory of this checkout, so `python -m pytest` and
    `python -m venv` are not asserted about. The raw text is scanned rather than the code-stripped
    text the link tests use, because a command lives inside a fenced block by definition.
    """
    text = doc.read_text(encoding="utf-8")
    packages = _repo_packages()
    missing: list[str] = []
    for match in _RUN_MODULE.finditer(text):
        dotted = match.group(1)
        head = dotted.split(".", 1)[0]
        if head in _NOT_OURS or head not in packages:
            continue
        if not _module_is_runnable(dotted):
            missing.append(dotted)
    assert not missing, (
        f"{doc.relative_to(_ROOT)} documents `python -m` commands that exit 1 with "
        f"'No module named': {sorted(set(missing))}"
    )


#: `python some/path/to/file.py`, the other half of the same rot.
#:
#: The dead `python -m examples.*` commands were repointed at `scripts/examples/` and
#: `scripts/checks/`, which are run by PATH rather than by module name. Guarding only the `-m` form
#: would have moved the rot rather than caught it.
_RUN_SCRIPT = re.compile(r"python(?:\.bat)?\s+((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py)\b")


@pytest.mark.parametrize("doc", _docs(), ids=lambda f: str(f.relative_to(_ROOT)).replace("\\", "/"))
def test_documented_script_paths_exist(doc: Path) -> None:
    """`python scripts/checks/cell_bringup.py` must name a file, resolved from the repo root.

    Every such command in these docs is written for an operator standing in the checkout root, which
    is the only place the repository is importable from, so that is the one resolution asserted.
    """
    text = doc.read_text(encoding="utf-8")
    missing = [
        script
        for script in (m.group(1) for m in _RUN_SCRIPT.finditer(text))
        if not (_ROOT / script).is_file()
    ]
    assert not missing, (
        f"{doc.relative_to(_ROOT)} documents `python <path>` commands whose file is gone: "
        f"{sorted(set(missing))}"
    )
