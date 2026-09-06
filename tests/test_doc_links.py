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
