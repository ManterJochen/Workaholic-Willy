"""Per-robot calibration artifact directories, and the stations a sim sweep runs.

Calibration artifacts are gitignored and share a filename across robots, so a flat
``logs/calibration/`` lets one robot's calibration run overwrite another's in place, silently.
Namespacing by ``robot_model`` keeps each robot's artifacts separate
(``logs/calibration/ur5e/...`` against ``.../ur3e/...``), so a two-robot rehearsal cannot lose one
robot's calibration by running the other's.

The stations are declared, one file per mounting and robot model under ``stations/`` beside this module
(:func:`declared_stations`), never generated.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["CALIBRATION_ROOT", "STATIONS_DIR", "calibration_dir", "declared_stations"]

#: ``<repo>/logs/calibration``: ``parents[3]`` reaches the repository root from this file.
CALIBRATION_ROOT = Path(__file__).resolve().parents[3] / "logs" / "calibration"
#: Where the declared sim stations live: ``eth_<robot_model>.json`` and ``eih_<robot_model>.json``.
STATIONS_DIR = Path(__file__).resolve().parent / "stations"


def calibration_dir(robot_model: str | None) -> Path:
    """The calibration artifact directory for ``robot_model``, e.g. ``logs/calibration/ur3e``.

    ``None`` or empty falls back to the shared root, so a mis-wired cell degrades to the flat
    layout rather than writing to a directory literally named ``None``. Callers pass
    ``cell.sim.robot_model``.
    """
    model = (robot_model or "").strip()
    return CALIBRATION_ROOT / model if model else CALIBRATION_ROOT


def declared_stations(mounting: str, robot_model: str, path: "str | Path | None" = None) -> Path:
    """The stations file a sim sweep runs: ``path`` when given, else the one declared for ``robot_model``.

    ``mounting`` is ``"eth"`` or ``"eih"``. Each shipped file is a list a person can read and edit, in the format
    the real cell's ``--fixed-poses`` reads (``pose_provider.load_stations``). The eye-in-hand ones hold TCP poses
    whose wrist camera looks at the scene marker from a ring of distances, elevations and azimuths, the eye-to-hand
    ones tool-down poses tilted up to 30 degrees in a column under the overhead camera. They were frozen from the
    generators this repository used to carry (the ring through the wrist mount the sim authors, in place of the
    runtime oracle), and an Isaac run has not been repeated on them since. A model with no declared file is refused
    with the sentence that says how to give one.
    """
    if path is not None:
        return Path(path)
    declared = STATIONS_DIR / f"{mounting}_{robot_model}.json"
    if not declared.is_file():
        raise SystemExit(f"no declared {mounting} calibration stations for robot_model {robot_model!r} in "
                         f"{STATIONS_DIR}; pass --stations PATH, a JSON list of poses or joint stations "
                         "(see pose_provider)")
    return declared
