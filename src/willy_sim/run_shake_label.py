"""On-box native physics-shake labeller: grasp the known-pose object, then label it held or dropped.

Boots the sim cell and the known-pose pick (``run_m1_pick.build_service``), grasps and lifts the
object, then runs :func:`shake_label.run_shake_test`, which perturbs the grasped object with six-axis
velocity impulses and a gravity overload and watches whether it tracks the gripper. Prints the
``held`` label per run. The label is produced in this stack, so a generated corpus carries labels
consistent with the scenes and the scorer that made them.

Run on-box:
    <isaac-sim>\\python.bat -m src.willy_sim.run_shake_label --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.utility.log_cfg import create_logger
from src.willy_sim.constants import SHAKE_LABEL_LOG_FILE, WILLY_SIM_LOG_DIR
from src.willy_sim.shake_label import ShakeConfig, run_shake_test

#: Results are written here (flush + fsync per run) so an Isaac teardown crash cannot swallow them.
_DEFAULT_OUT = "logs/shake/shake_results.jsonl"

#: This runner produces labels, and a label file outlives every shell that could explain it. The
#: perturbation config goes into this log rather than into the JSONL, because without it two corpora
#: shaken with different gravity overloads are indistinguishable after the fact.
_LOG = create_logger("ShakeLabeller", log_file=SHAKE_LABEL_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)


def _write_result(out_path: Path, row: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _quat_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def run_shake_gate(
    runs: int = 3,
    *,
    headless: bool = True,
    data_dir: str | None = None,
    mode: str = "easy",
    out: str = _DEFAULT_OUT,
) -> None:
    from src.robot.core import JointPositions
    from src.willy_sim.harness.gate import reset_object_to_home_z0
    from src.willy_sim.config import require_robot
    from src.willy_sim.run_m1_pick import build_service

    # build_service boots the SimulationApp; isaacsim.* imports must come after it (Kit requirement).
    service, arm, gripper, handles, cfg, _cell = build_service(
        headless=headless, data_dir=data_dir, mode=mode
    )
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]  # noqa: E402

    robot = require_robot(cfg)
    home_q = np.asarray(robot.sim.home_joint_positions, dtype=np.float64)
    obj = SingleRigidPrim(handles.object_prim_path)
    home_pos, home_quat = obj.get_world_pose()
    # The parallel-jaw sim grasp is a friction grip (a finger_joint position drive), not a fixed-joint
    # weld, so a single-frame velocity impulse on the held body is immediately damped by the contact
    # solver and leaves no relative drift. The physically meaningful perturbations for a friction grip
    # are a base or arm jerk and a gravity overload ("can the grip hold N x weight"). Gravity overload
    # fits the injected-callable model cleanly, being a global scene setting with no stepping conflict,
    # so it is the channel driven here; the velocity kick stays a secondary.
    cfg_shake = ShakeConfig(kick_speed_mps=3.0, overload_gravity_scale=8.0, overload_steps=60)

    def _step(n: int) -> None:
        for _ in range(n):
            arm.session.step()

    def _gripper_pose():
        p = arm.get_tcp_pose()
        return np.asarray(p.position_mm, dtype=np.float64), _quat_to_mat(p.quaternion_xyzw)

    def _object_position():
        return np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0  # m -> mm

    def _kick(vel_mm_s: np.ndarray) -> None:
        obj.set_linear_velocity(np.asarray(vel_mm_s, dtype=np.float64) * 0.001)  # mm/s -> m/s

    def _set_gravity_scale(scale: float) -> None:
        """Scale scene gravity (magnitude, along -Z) for the overload phase.

        Tries the scalar-magnitude signature first, then the vector form, so it works across the
        Isaac 5.1 gravity API variants.
        """
        world = arm.session.world
        if world is None:
            return
        ctx = world.get_physics_context()
        try:
            ctx.set_gravity(-9.81 * float(scale))  # scalar magnitude along -Z
        except (TypeError, ValueError):
            ctx.set_gravity([0.0, 0.0, -9.81 * float(scale)])  # vector fallback

    out_path = Path(out)
    _LOG.info(
        "shake labelling %d run(s) on the M1 cell (mode=%s) -> %s; perturbation: kick=%.1f m/s, "
        "gravity overload x%.1f for %d steps",
        runs, mode, out_path, cfg_shake.kick_speed_mps, cfg_shake.overload_gravity_scale,
        cfg_shake.overload_steps,
    )
    held_count = 0
    grasped_count = 0
    for i in range(runs):
        gripper.open()
        arm.move_joint(JointPositions(home_q))
        reset_object_to_home_z0(obj, home_pos, home_quat, arm.session, settle_steps=30)
        report = service.pick()
        outcome = getattr(report, "outcome", None)
        succeeded = getattr(outcome, "value", outcome) == "succeeded"
        if not succeeded:
            row: dict[str, object] = {"run": i, "pick_succeeded": False}
            print(f"RUN {i}: pick FAILED (nothing to shake)", flush=True)
            _write_result(out_path, row)
            continue
        grasped_count += 1
        # Wake the held body: a settled rigid body can sleep and then ignore gravity and velocity
        # changes, which looks identical to a rock-solid grip. A tiny velocity nudge wakes it first.
        obj.set_linear_velocity(np.array([0.0, 0.0, 1e-3], dtype=np.float64))
        abs_before = _object_position()
        result = run_shake_test(
            step=_step,
            gripper_pose=_gripper_pose,
            object_position=_object_position,
            kick=_kick,
            set_gravity_scale=_set_gravity_scale,
            config=cfg_shake,
        )
        abs_after = _object_position()
        abs_moved = float(np.linalg.norm(abs_after - abs_before))
        # Physics-alive probe: open the jaws and see whether the object falls. The gripper is
        # stationary during the shake, so a held object and a frozen body both read ~0 relative drift;
        # only releasing the grip separates them. A large downward drop means a real friction grip, and
        # the shake then needs a stronger or arm-jerk perturbation. A drop of ~0 means the object is
        # effectively welded or kinematic, and native shake-labelling is impossible in this sim
        # without un-welding it.
        z_before_release = _object_position()[2]
        gripper.open()
        _step(60)
        z_after_release = _object_position()[2]
        released_fell_mm = round(float(z_before_release - z_after_release), 2)
        held_count += int(result.held)
        row = {
            "run": i,
            "pick_succeeded": True,
            "shake_held": bool(result.held),
            "max_disp_mm": round(result.max_displacement_mm, 2),
            "abs_world_moved_mm": round(abs_moved, 2),
            "released_fell_mm": released_fell_mm,
            "failed_phase": result.failed_phase,
            "per_phase_max_mm": {k: round(v, 2) for k, v in result.per_phase_max_mm.items()},
        }
        print(
            f"RUN {i}: pick_succeeded=True  shake_held={result.held}  "
            f"max_disp={result.max_displacement_mm:.1f}mm  failed_phase={result.failed_phase or '-'}",
            flush=True,
        )
        _write_result(out_path, row)
    _write_result(
        out_path,
        {"summary": True, "grasped": grasped_count, "held_after_shake": held_count, "runs": runs},
    )
    # The yield, not the per-run detail; that is on stdout and in the JSONL. ``runs - grasped`` is the
    # share of attempts that produced no label at all, a corpus this run silently did not grow.
    _LOG.info(
        "shake labelling done: %d/%d grasped objects held, %d run(s) produced no label (pick failed); "
        "%d row(s) + summary appended to %s",
        held_count, grasped_count, runs - grasped_count, runs, out_path,
    )
    print(f"\nSHAKE-LABEL SUMMARY: {held_count}/{grasped_count} grasped objects survived the shake", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--mode", default="easy")
    ap.add_argument("--out", default=_DEFAULT_OUT)
    args = ap.parse_args()
    t0 = time.time()
    run_shake_gate(
        runs=args.runs, headless=not args.gui, data_dir=args.data_dir, mode=args.mode, out=args.out
    )
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
