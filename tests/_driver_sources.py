"""Where the UR arm's code lives, asked of the class rather than written down as a path (P0e).

Source scans that name `drivers/ur/arm.py` stop seeing the code the moment P0e moves the planning half into
`PlannedArm` in `drivers/planned/`: the scan stays green over a file that no longer holds what it checks. These
helpers answer from `URRobotArm`'s own method resolution order, so a scan follows the code wherever the extraction puts
it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DRIVERS = _REPO / "src" / "robot" / "drivers"


def ur_arm_source_files() -> list[Path]:
    """The file of ``URRobotArm`` and of every driver-layer class it inherits from, in MRO order."""
    from src.robot.drivers.ur.arm import URRobotArm

    files: list[Path] = []
    for cls in inspect.getmro(URRobotArm):
        if not cls.__module__.startswith("src.robot.drivers."):
            continue
        path = Path(inspect.getsourcefile(cls) or "").resolve()
        if path not in files:
            files.append(path)
    return files


def ur_arm_source() -> str:
    """The text of :func:`ur_arm_source_files`, joined."""
    return "\n".join(path.read_text(encoding="utf-8") for path in ur_arm_source_files())


def driver_source_files() -> list[Path]:
    """Every Python file under ``src/robot/drivers``."""
    return sorted(_DRIVERS.rglob("*.py"))
