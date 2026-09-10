"""Cross-platform path helpers.

Every filesystem-layout decision lives here, so the code behaves identically on
Windows and Linux. All paths are ``pathlib.Path``, never raw strings with
``\\`` separators.

Three environment variables override the defaults:

* ``WILLY_PROJECT_ROOT`` the project root, otherwise the folder that holds
  ``src/``.
* ``WILLY_LOG_DIR``      the logs root, otherwise ``<root>/logs``.
* ``WILLY_DEBUG_DIR``    the debug image root, otherwise ``<logs>/debug``.

Two events reach ``logs/utility/paths.log``: a project root that had to be
guessed, and the bounded-bucket rotation in :func:`rotate_files`, the only place
in this package that deletes a file.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from src.utility.constants import PATHS_LOG_FILE, utility_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from logging import Logger

__all__ = [
    "debug_dir",
    "ensure_dir",
    "fence_model_downloads",
    "logs_dir",
    "project_root",
    "rotate_files",
    "weights_root",
]


def _log() -> Logger:
    """Logger for this module, built on first use. See :func:`utility_logger`.

    Call it only where a line is actually going to be emitted. ``create_logger``
    opens the rotating file eagerly, and ``project_root`` runs on every path
    lookup here, so touching this accessor on a hot happy path leaves a
    permanently empty ``paths.log`` behind even at debug level.
    """
    return utility_logger("UtilityPaths", PATHS_LOG_FILE)


#: True while :func:`project_root` is inside its own fallback branch. Read the comment
#: there: the warning that branch emits is built by machinery that asks for the root again,
#: so without this the one situation the warning exists for is the one where it recurses to
#: death instead of printing.
_GUESSING_THE_ROOT = False


def _from_env(var: str) -> Path | None:
    raw = os.environ.get(var)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def project_root() -> Path:
    """Return the project root as a ``Path``.

    Resolution order:

    1. The ``WILLY_PROJECT_ROOT`` environment variable, if set.
    2. The nearest ancestor directory that holds ``src`` together with either
       ``requirements.txt`` or ``pyproject.toml``.
    3. Two levels up from this file, which sits at ``src/utility/paths.py``.
    """
    env = _from_env("WILLY_PROJECT_ROOT")
    if env is not None:
        return env

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "src").is_dir() and (
            (ancestor / "requirements.txt").is_file()
            or (ancestor / "pyproject.toml").is_file()
        ):
            return ancestor

    # utility/paths.py -> utility -> src -> <root>
    #
    # Nothing matched, so this is a guess from the depth of this file alone. It
    # holds in a normal checkout and breaks once the package is vendored or
    # installed elsewhere, where every caller building a path off the root
    # inherits the mistake silently. The warning below names the fix.
    fallback = here.parents[2]

    # The warning below needed the root it is warning about, and it took the process
    # down. Measured 2026-09-10 on a copy of this package under a directory holding
    # `src/` and no `pyproject.toml`, which is what a vendored or site-packages install
    # looks like: building the line runs `_log` to `utility_logger` to `create_logger`
    # to `resolve_log_dir` to `logs_dir` to `project_root`, which lands back here, and
    # the operator got `RecursionError: maximum recursion depth exceeded` instead of the
    # one sentence naming WILLY_PROJECT_ROOT. The diagnostic could not run in exactly the
    # situation it was written for.
    #
    # A re-entry flag rather than a plain `print`: the loop is not a property of the
    # logger, it is the property that anything reached while this branch is open may ask
    # for the root again. Whatever `logs_dir` comes to depend on later, the second entry
    # answers from the same guess instead of recursing, and the sentence still reaches the
    # log file it belongs in.
    global _GUESSING_THE_ROOT
    if _GUESSING_THE_ROOT:
        return fallback
    _GUESSING_THE_ROOT = True
    try:
        _log().warning(
            "project_root: no ancestor of %s holds src/ plus requirements.txt or "
            "pyproject.toml; guessing %s from path depth. Set WILLY_PROJECT_ROOT to be sure.",
            here,
            fallback,
        )
    finally:
        _GUESSING_THE_ROOT = False
    return fallback


def logs_dir() -> Path:
    """Return the directory where application logs live, created on demand."""
    env = _from_env("WILLY_LOG_DIR")
    base = env if env is not None else project_root() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create ``path`` and its parents if needed, and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rotate_files(
    directory: os.PathLike[str] | str,
    *,
    max_files: int,
    patterns: Iterable[str] = ("*",),
) -> int:
    """Delete the oldest files in ``directory`` so that at most ``max_files`` remain.

    Only files matching one of the ``patterns`` globs count towards the cap.
    Returns the number of files removed. A missing directory is ignored, so this
    can be called before anything has written to the directory.
    """
    d = Path(directory)
    if not d.is_dir() or max_files <= 0:
        return 0

    found: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for f in d.glob(pat):
            if f.is_file() and f not in seen:
                seen.add(f)
                found.append(f)

    if len(found) <= max_files:
        return 0

    found.sort(key=lambda p: p.stat().st_mtime)
    to_remove = found[: len(found) - max_files]
    removed = 0
    failed = 0
    for f in to_remove:
        try:
            f.unlink()
            removed += 1
        except OSError:
            # Best effort: another process may hold the file.
            failed += 1
            continue

    # One line per call rather than per file, at info because this is the only
    # helper in the package that deletes data. It runs on every debug_dir()
    # call, and the answerable question afterwards is how many files left which
    # bucket, not which artefact was oldest.
    if removed:
        _log().info(
            "rotate_files: removed %d of %d matching file(s) from %s (cap %d)",
            removed,
            len(found),
            d,
            max_files,
        )
    if failed:
        # A locked file is the one way the cap quietly stops holding: the bucket
        # keeps growing and nothing else reports it.
        _log().warning(
            "rotate_files: %d file(s) in %s could not be deleted (held by another "
            "process?); the directory stays above its %d-file cap",
            failed,
            d,
            max_files,
        )
    return removed


def debug_dir(
    subdir: str,
    *,
    max_files: int = 200,
    rotate_patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg"),
) -> Path:
    """Return a per-module debug directory under ``logs/debug/<subdir>``.

    ``subdir`` is a single path segment: a separator or ``..`` raises
    ``ValueError``. The directory is created and then rotated through
    :func:`rotate_files`, so the bucket never grows unbounded. Write debug
    artefacts here rather than into the current working directory.
    """
    if not subdir or any(ch in subdir for ch in ("/", "\\", "..")):
        raise ValueError(f"debug_dir: invalid subdir {subdir!r}")

    env = _from_env("WILLY_DEBUG_DIR")
    base = env if env is not None else logs_dir() / "debug"
    target = base / subdir
    target.mkdir(parents=True, exist_ok=True)
    # No log line here: this runs once per saved debug image, and rotate_files
    # names the bucket whenever it deletes anything.
    rotate_files(target, max_files=max_files, patterns=rotate_patterns)
    return target


def weights_root() -> Path:
    """Where every downloaded model file lives: ``assets/models/hf`` under the project root.

    One root, sub-categorised by whoever writes into it. Hugging Face lays out ``hub/`` and ``xet/``
    itself, the fetch script writes a directory per model under a category such as ``detection`` or
    ``vlm``, and the MediaPipe bundles sit in ``mediapipe/`` beside them.

    Deleting the repository takes the weights with it, which is the reason for the location. A
    user level cache does not have that property: a download that lands in the profile outlives every
    checkout, is shared silently between them, and cannot be reasoned about from inside the tree.

    Trained artifacts this repository produces, ``assets/models/success_probability`` and
    ``assets/models/grasp_ranker``, are not under here. They are committed or generated rather than
    fetched, and that difference is what this directory is ignored for.
    """
    return project_root() / "assets" / "models" / "hf"


def fence_model_downloads() -> Path:
    """Point every download library at :func:`weights_root` and return it.

    Call this before importing ``huggingface_hub``, ``transformers`` or ``torch.hub``. The hub reads
    its cache locations into module level constants at import time, so an assignment afterwards is
    accepted by ``os.environ`` and changes nothing. The fence therefore belongs at an entry point
    rather than in a runner that has already imported the stack.

    ``HF_HOME`` rather than ``HF_HUB_CACHE``: the chunk store, the downloaded ``modules/`` and the
    token file derive from ``HF_HOME``, so setting only the narrower variable leaves pieces in the
    user profile.

    ``TORCH_HOME`` is set for the same reason. The token variables are removed rather than ignored,
    because an account makes the download path depend on who is logged in and this repository needs a
    fetch anyone can reproduce.
    """
    import os

    root = weights_root()
    os.environ["HF_HOME"] = str(root)
    os.environ["TORCH_HOME"] = str(root / "torch")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    # Windows has no symlinks here, so the cache cannot deduplicate. Saying so once beats a warning
    # per file.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        os.environ.pop(variable, None)
    return root
