"""What the Isaac arm states for itself in place of a real arm's guard numbers: one table, explained.

Every real arm runs the owner's cell's numbers (``src/robot/safety/guard_numbers.CELL``), and no tree configures them
since 2026-10-17 (the owner's cleanup). The Isaac arm is not a real arm, and this harness hands :data:`ISAAC` to its
preflight (:func:`src.willy_sim.config.sim_safety_preflight`, and the runners that rebuild one). Each number says why
it differs; they are the numbers the sim profile (``config/robot/robot.sim.yaml``) wrote until then.
"""

from __future__ import annotations

from typing import Final

from src.robot.safety.guard_numbers import GuardNumbers

__all__ = ["ISAAC"]

#: The Isaac arm's numbers.
ISAAC: Final = GuardNumbers(
    # The Isaac UR asset's base_link stands half a turn about z from UR's DH base, where a real controller reports
    # (``ur_link_origins_mm`` is ``[-x, -y, z]`` against the Lula FK ground truth at every pose), and the sim's poses and
    # fixtures are written in base_link: the guard turns the DH chain by this to place the links among them.
    base_yaw_deg=180.0,
    # Every joint over the asset's whole +-360 degree envelope, its description's: the Isaac arm winds no cable, and
    # its park pose turns the base half a turn from home, which a window about home less its margin would refuse.
    half_turn_about_home=False,
    # A real arm's IK and continuity numbers are sized for a servo drive, where an IK jump is an unfollowable
    # discontinuity and a near-singular configuration amplifies joint velocity. The Isaac drive has neither failure
    # mode: ``move_joint`` is position-controlled and walks a move in 0.03 rad waypoints, up to 250 of them, so about
    # 7.5 rad in one command, and ``move()`` verifies the achieved TCP by FK. These pass a large single-move
    # reconfiguration and an awkward but valid near-singular grasp, and still refuse IK garbage: a condition number
    # above 1000, a smallest singular value under 1e-4, a wildly wrong branch.
    max_jump_rad=3.5,
    min_singular_value=0.0001,
    max_condition_number=1000.0,
    max_joint_step_deg=180.0,
    max_orientation_step_deg=180.0,
    max_tcp_step_mm=1000.0,
)
