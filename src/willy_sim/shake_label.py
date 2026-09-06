"""Native physics shake-test labeller, with no dependency on the external aletheia data engine.

Labels a grasped object ``held`` or ``dropped`` by perturbing the grasp and watching whether the
object tracks the gripper: settle, then six-axis linear kicks, then a gravity overload, then a final
still check. It is the in-repo equivalent of aletheia's physics-shake label, so this stack can
generate physics-labelled grasps whose labels are consistent with its own scenes and scorer instead
of importing a corpus whose physics that scorer only weakly predicts.

Isaac is not imported here. The Isaac coupling is a set of injected callables (``step``, the pose
getters, ``kick``), so both the shake orchestration and the held decision run off-box; a thin runner
wires the real Isaac cell in. The held decision: the object's pose expressed in the gripper frame
must stay within ``held_displacement_threshold_mm`` of its initial grasped pose across every phase,
which is to say it moved with the gripper and not out of it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

#: A kick = (label, linear-velocity direction). Applied as a brief impulse on the gripper/arm.
_UNIT_KICKS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("+x", (1.0, 0.0, 0.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)),
    ("-z", (0.0, 0.0, -1.0)),
)


@dataclass(frozen=True)
class ShakeConfig:
    """Shake-test parameters (all default-off-safe; tune per gripper on-box)."""

    settle_steps: int = 60
    kick_steps: int = 15
    kick_speed_mps: float = 0.6
    overload_steps: int = 30
    overload_gravity_scale: float = 3.0
    final_still_steps: int = 20
    #: The object's gripper-frame pose may drift at most this far (mm) before it counts as dropped.
    held_displacement_threshold_mm: float = 25.0
    kicks: tuple[tuple[str, tuple[float, float, float]], ...] = _UNIT_KICKS


@dataclass(frozen=True)
class ShakeResult:
    """Outcome of one shake test."""

    held: bool
    max_displacement_mm: float
    failed_phase: str = ""  # "" when held
    per_phase_max_mm: dict[str, float] = field(default_factory=dict)


def object_in_gripper_frame(
    gripper_pos_mm: np.ndarray, gripper_rot: np.ndarray, object_pos_mm: np.ndarray
) -> np.ndarray:
    """Express the object position in the gripper frame: ``R_grip^T @ (obj - grip)`` (mm)."""

    delta = np.asarray(object_pos_mm, dtype=np.float64) - np.asarray(gripper_pos_mm, dtype=np.float64)
    return np.asarray(gripper_rot, dtype=np.float64).T @ delta


def evaluate_held(
    gripper_frame_positions_mm: Sequence[np.ndarray], threshold_mm: float
) -> tuple[bool, float]:
    """Report whether the object stayed held, with the largest drift observed.

    Held iff the gripper-frame position never drifts more than ``threshold_mm`` from its initial
    (grasped) value. Returns ``(held, max_displacement_mm)``. An empty sequence is not held.
    """

    if not gripper_frame_positions_mm:
        return False, float("inf")
    ref = np.asarray(gripper_frame_positions_mm[0], dtype=np.float64)
    max_disp = 0.0
    for rel in gripper_frame_positions_mm:
        disp = float(np.linalg.norm(np.asarray(rel, dtype=np.float64) - ref))
        max_disp = max(max_disp, disp)
    return max_disp <= float(threshold_mm), max_disp


def run_shake_test(
    *,
    step: Callable[[int], None],
    gripper_pose: Callable[[], tuple[np.ndarray, np.ndarray]],
    object_position: Callable[[], np.ndarray],
    kick: Callable[[np.ndarray], None],
    set_gravity_scale: Callable[[float], None] | None = None,
    config: ShakeConfig | None = None,
) -> ShakeResult:
    """Run the shake test over injected Isaac callables and return a :class:`ShakeResult`.

    ``step(n)`` advances physics n frames. ``gripper_pose()`` returns ``(pos_mm, rot3x3)`` and
    ``object_position()`` returns ``pos_mm``. ``kick(vel_mm_s)`` applies a brief velocity impulse to
    the gripper or arm. The optional ``set_gravity_scale(s)`` scales gravity for the overload phase;
    gravity is restored to 1.0 afterwards.
    """

    cfg = config or ShakeConfig()

    def _rel() -> np.ndarray:
        gp, gr = gripper_pose()
        return object_in_gripper_frame(gp, gr, object_position())

    ref = _rel()
    per_phase: dict[str, float] = {}
    all_positions: list[np.ndarray] = [ref]

    def _phase(name: str, n_steps: int) -> None:
        # sample within the phase (~5x) so a transient peak (object springs out then back on an elastic
        # grasp) is caught, not just the post-settle value.
        sub = max(1, n_steps // 5)
        phase_max = 0.0
        done = 0
        while done < n_steps:
            s = min(sub, n_steps - done)
            step(s)
            done += s
            rel = _rel()
            all_positions.append(rel)
            phase_max = max(phase_max, float(np.linalg.norm(rel - ref)))
        per_phase[name] = phase_max

    _phase("settle", cfg.settle_steps)
    speed = cfg.kick_speed_mps * 1000.0  # mm/s
    for name, direction in cfg.kicks:
        kick(np.asarray(direction, dtype=np.float64) * speed)
        _phase(f"kick_{name}", cfg.kick_steps)
    if set_gravity_scale is not None:
        set_gravity_scale(cfg.overload_gravity_scale)
        _phase("overload", cfg.overload_steps)
        set_gravity_scale(1.0)
    _phase("final_still", cfg.final_still_steps)

    held, max_disp = evaluate_held(all_positions, cfg.held_displacement_threshold_mm)
    failed = ""
    if not held:
        failed = max(per_phase, key=lambda k: per_phase[k]) if per_phase else "initial"
    return ShakeResult(
        held=held, max_displacement_mm=max_disp, failed_phase=failed, per_phase_max_mm=per_phase
    )


__all__ = (
    "ShakeConfig",
    "ShakeResult",
    "evaluate_held",
    "object_in_gripper_frame",
    "run_shake_test",
)
