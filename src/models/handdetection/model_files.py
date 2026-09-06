"""Find the MediaPipe `.task` bundle, or refuse with something an operator can act on.

MediaPipe is an optional extra and its model files are not in this repository; the operator installs
both. Left to the libraries, either absence reports poorly:

* mediapipe missing raises `ModuleNotFoundError: No module named 'mediapipe'` at import time, from a
  module the caller never imported by name;
* the `.task` file missing raises from inside MediaPipe's C++ graph, naming neither the config key
  that supplied the path nor where the file comes from.

Both are therefore checked here, up front, and the message names the config key, the resolved
absolute path and the download URL. The check is fail-closed: a missing hand reads exactly like an
absent hand, so a detector that silently does not run is worse than one that refuses to start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "MEDIAPIPE_AVAILABLE",
    "MEDIAPIPE_INSTALL_HINT",
    "require_mediapipe",
    "resolve_model_file",
]

try:  # pragma: no cover (the branch taken depends on the install, not on the tests)
    import mediapipe as _mp  # noqa: F401

    _available = True
except ImportError:  # pragma: no cover (only on installs without the optional extra)
    _available = False

#: Whether the optional extra is importable. Probed once, at import, so a module-level `from ...
#: import MEDIAPIPE_AVAILABLE` is cheap and every class can guard its constructor the same way.
#: Assigned from a single branch result rather than inside both branches, because a `Final` written
#: twice is not final and mypy rejects it.
MEDIAPIPE_AVAILABLE: Final[bool] = _available

MEDIAPIPE_INSTALL_HINT: Final[str] = (
    "mediapipe is pinned in requirements.txt: pip install -r requirements.txt"
)


def require_mediapipe(what: str) -> None:
    """Raise a named `ImportError` when the optional extra is absent.

    `what` names the thing the caller was building, because "mediapipe is not installed" on its own
    does not tell an operator which feature is unavailable.
    """
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError(f"mediapipe is required for {what} but is not installed. {MEDIAPIPE_INSTALL_HINT}")


def resolve_model_file(path: str, *, config_key: str, download_url: str) -> str:
    """Return `path` as an absolute string, or raise naming the key, the path and the download.

    Relative paths resolve against the process working directory, the convention `create_logger`
    uses for `logs/`, so one config resolves the same way across the stack.
    """
    if not path or not path.strip():
        raise FileNotFoundError(
            f"{config_key} is empty, so there is no MediaPipe model to load.\n"
            f"  Set it to the path of the downloaded bundle.\n"
            f"  Download: {download_url}"
        )

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved

    if not resolved.is_file():
        raise FileNotFoundError(
            f"MediaPipe model file not found: {resolved}\n"
            f"  Configured by: {config_key} = {path!r}\n"
            f"  Download it and put it there:\n"
            f"    {download_url}\n"
            f"  The bundles are NOT committed to this repository (they are binaries, and the\n"
            f"  operator chooses which revision to run)."
        )
    return str(resolved)
