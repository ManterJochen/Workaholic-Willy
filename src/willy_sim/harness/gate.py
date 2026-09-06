"""Shared run-gate primitives for the willy_sim pick runners.

Every runner's ``run_gate`` has the same skeleton around its own move and scoring: re-place the
object at its home pose, zero its velocity, settle, read the world-Z baseline, pick, read the new Z,
score the lift, then roll the per-run results up into a gate verdict. This module holds the shared
fragments of that skeleton, the object reset and measure and the gate-fraction math, plus the typed
:class:`GateResult` a runner returns.

Two parts stay per-runner. The reset move stays, because ``move_joint``, ``move_to_joints`` and
``move`` carry different preflight and continuity semantics. The pass criterion stays, because it
ranges from lift alone through succeeded plus lift to target lifted with no distractor lifted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.drivers.sim._isaac_protocols import IsaacRigidPrim
    from src.robot.drivers.sim.session import IsaacSimSession

__all__ = ["GateResult", "gate_passed", "lift_mm_since", "reset_object_to_home_z0"]


@dataclass(frozen=True, slots=True)
class GateResult:
    """Typed result of a runner's ``run_gate``.

    ``runs``, ``passed`` and ``results`` are always present. ``gate_passed``, ``target`` (the dense
    runner's prompted label) and ``distractor_lifts`` (the fused clutter gate) are per-runner extras
    emitted only when set; the vision runner scores on lift alone and omits ``gate_passed``.
    :meth:`to_dict` emits exactly the key set its runner expects, and ``--result-json`` is written
    with ``sort_keys=True``.
    """

    runs: int
    passed: int
    results: list[dict[str, Any]]
    gate_passed: bool | None = None
    target: str | None = None
    distractor_lifts: int | None = None
    #: Set only when the cell was configured for a planner it could not start and ran on the fallback
    #: path instead. A pass rate measured that way does not describe the configured cell, and the
    #: number alone cannot say so. Emitted only when set, so a healthy run's JSON does not carry it.
    planner_degraded: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"runs": self.runs, "passed": self.passed}
        if self.gate_passed is not None:
            out["gate_passed"] = self.gate_passed
        if self.target is not None:
            out["target"] = self.target
        if self.distractor_lifts is not None:
            out["distractor_lifts"] = self.distractor_lifts
        if self.planner_degraded is not None:
            out["planner_degraded"] = self.planner_degraded
        out["results"] = self.results
        return out


def reset_object_to_home_z0(
    obj: IsaacRigidPrim,
    home_pos: Any,
    home_quat: Any,
    session: IsaacSimSession,
    *,
    settle_steps: int,
) -> float:
    """Re-place ``obj`` at its home pose, zero its velocity, settle, and return its world-Z in metres.

    The per-run reset tail the single-object runners share, run after the runner's own
    ``gripper.open`` and arm move: teleport to the home pose, zero linear and angular velocity where
    the prim wrapper allows it, step the sim ``settle_steps`` times, then read the baseline Z.
    """
    obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
    try:
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))
    except Exception:  # noqa: BLE001 (velocity-zero is best-effort; some prim wrappers lack it)
        pass
    session.step_n(settle_steps)
    return float(np.asarray(obj.get_world_pose()[0])[2])


def lift_mm_since(obj: IsaacRigidPrim, z0: float) -> float:
    """Return the world-Z lift in mm of ``obj`` since baseline ``z0`` (metres)."""
    z1 = float(np.asarray(obj.get_world_pose()[0])[2])
    return (z1 - z0) * 1000.0


def gate_passed(n_pass: int, runs: int, pass_fraction: float) -> bool:
    """The shared gate verdict: ``n_pass`` reaches ``int(pass_fraction * runs)``, and at least 1."""
    return n_pass >= max(1, int(pass_fraction * runs))
