"""The numbers the guards judge an arm by that no cell configures: a real arm's are the owner's cell's.

Until 2026-10-17 each was a key under ``robot.safety`` (the owner's cleanup took them; ``schema/_removed.py`` says what
became of each). Every real arm runs :data:`CELL`, the values the owner's cell ran. An arm whose physics are not a
real arm's states its own in code, explained number by number, and its harness hands them to the preflight
(``SafetyPreflight.from_safety_config(numbers=...)``): the Isaac arm's are ``src/willy_sim/guard_numbers.py``. Nothing
reads them from a tree, so no tree can loosen a real arm's.

The numbers every arm shares are their guards' own: the workspace box shrunk by 20 mm on every face
(``workspace.WORKSPACE_MARGIN_MM``), 5 degrees kept inside each end of a joint's range
(``joint_limits.JOINT_LIMIT_MARGIN_DEG``), a solution refused within 2 degrees of one (``ik_quality.LIMIT_PROXIMITY_DEG``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = ["CELL", "GuardNumbers"]


@dataclass(frozen=True, slots=True)
class GuardNumbers:
    """What the guards hold one arm to beyond the cell's own config; :data:`CELL` for every real arm."""

    #: How far the arm's DH base is turned about z from the frame poses and fixtures are written in, degrees. 0 on a
    #: real arm: a UR's controller reports poses in its DH base (``config/arms/<model>.yaml``, ``base``), so the guard
    #: places the arm's links where the controller's poses and the cell's fixtures stand without a turn.
    base_yaw_deg: float = 0.0
    #: Whether every joint that turns a full turn is kept within half a turn of the arm's home, less the guard's
    #: margin (``joint_limits``): a cable run along a real arm is then never wound more than half a turn from where it
    #: lay at home, and nobody types a joint range. It costs no reach, since every UR joint is 2*pi-periodic; what it
    #: refuses is the move across the seam behind home.
    half_turn_about_home: bool = True
    #: The largest step any joint of an IK solution may take from where the arm stands, radians: past it the solver
    #: has most likely wrapped onto another branch of the same pose.
    max_jump_rad: float = 1.0
    #: The smallest singular value of the Jacobian a solution may have, and the largest condition number: a
    #: near-singular configuration turns a small Cartesian step into a fast joint one on a servo drive.
    min_singular_value: float = 0.005
    max_condition_number: float = 250.0
    #: How far two commanded targets in a row may lie apart: the largest joint step, degrees, the turn of the TCP,
    #: degrees, and its travel, millimetres. A stray command that would slam the arm across the cell is refused.
    max_joint_step_deg: float = 45.0
    max_orientation_step_deg: float = 30.0
    max_tcp_step_mm: float = 250.0


#: The owner's cell's numbers, every real arm's.
CELL: Final = GuardNumbers()
