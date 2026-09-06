"""Gap D2 — markdown link-resolution test.

The past QUICKSTART drift (broken Phase-U link, etc.) landed because no test resolved doc links. This
scans the operator-facing docs (QUICKSTART + runbooks + every `backend/src` package README, incl.
willy_sim/models/calibration/camera/geometry — extended in R10.3a), extracts every relative markdown
link, strips the anchor/query, and asserts the target file exists. http(s)/mailto and pure-anchor links
are skipped. Runs in the mock/CI suite (pure filesystem).
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


def _docs() -> list[Path]:
    files: list[Path] = [_ROOT / "QUICKSTART.md", _ROOT / "README.md"]
    files += sorted((_ROOT / "docs" / "runbooks").glob("*.md"))
    # ⛔ THE TOP LEVEL OF docs/ WAS OUTSIDE THIS SET, AND THAT WAS FOUND BY A READER RATHER THAN BY
    # CI. `calibration-setup.md`, `grasping-math.md`, `safety-math.md`, `code-integrity.md` and
    # `isaac-ready.md` all link into the source with `../`, the same shape that rotted in QUICKSTART
    # before this test existed, and nothing checked them. README.md is in for the same reason: it is
    # the single most-read file in the repository and carried four links that no test resolved.
    files += sorted((_ROOT / "docs").glob("*.md"))
    # The docs/guide/ set is the end-to-end walkthrough (config -> models -> calibration -> robot+safety
    # -> pick loop). It links into the source with `../../` on nearly every page, which is exactly the
    # link shape that rotted in QUICKSTART before this test existed.
    files += sorted((_ROOT / "docs" / "guide").glob("*.md"))
    files += sorted((_ROOT / "backend" / "src").rglob("*[Rr][Ee][Aa][Dd][Mm][Ee]*.md"))
    return [f for f in files if f.exists()]


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
    """The session-facing canon: CLAUDE.md + every `.ai-memory/*.md`.

    CLAUDE.md's first line orders every session to read these INSTEAD of re-scanning the code, so a link
    here pointing at a file the code no longer has is a load-bearing lie -- exactly how a session ends up
    citing `grasping/pick_loop.py` after the fcbf002 reorg moved it. This guard makes that an invariant.
    """
    files = [_ROOT / "CLAUDE.md"]
    files += sorted((_ROOT / ".ai-memory").glob("*.md"))
    return [f for f in files if f.exists()]


@pytest.mark.parametrize("doc", _canon_docs(), ids=lambda f: str(f.relative_to(_ROOT)).replace("\\", "/"))
def test_canon_source_links_point_at_real_files(doc: Path) -> None:
    """A canon link must resolve -- from the repo root OR the file's own directory.

    The `.ai-memory/*.md` files link into the source with the repo-root convention (`](backend/...)`),
    which is legitimate: a session reads them from the repo root. So this accepts EITHER resolution and
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
