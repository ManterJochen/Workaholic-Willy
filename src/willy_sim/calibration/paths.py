"""Per-robot calibration artifact directories.

Calibration artifacts are gitignored and share a filename across robots, so a flat
``logs/calibration/`` lets one robot's calibration run overwrite another's in place, silently.
Namespacing by ``robot_model`` keeps each robot's artifacts separate
(``logs/calibration/ur5e/...`` against ``.../ur3e/...``), so a two-robot rehearsal cannot lose one
robot's calibration by running the other's.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["CALIBRATION_ROOT", "calibration_dir"]

#: ``<repo>/logs/calibration``: ``parents[3]`` reaches the repository root from this file.
CALIBRATION_ROOT = Path(__file__).resolve().parents[3] / "logs" / "calibration"


def calibration_dir(robot_model: str | None) -> Path:
    """The calibration artifact directory for ``robot_model``, e.g. ``logs/calibration/ur3e``.

    ``None`` or empty falls back to the shared root, so a mis-wired cell degrades to the flat
    layout rather than writing to a directory literally named ``None``. Callers pass
    ``cell.sim.robot_model``.
    """
    model = (robot_model or "").strip()
    return CALIBRATION_ROOT / model if model else CALIBRATION_ROOT
