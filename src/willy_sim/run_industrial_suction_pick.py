"""Industrial suction pick: a suction cup seals the flat top of a real GSO package the jaw cannot span,
and lifts it. This is the complementary-modality half of the industrial demo, where the jaw picks small
parts and suction picks the wide flat packages.

The pick reuses the suction stack: mount the surface-gripper cup on the wrist with the jaw hidden,
perceive the object's real top geometry (ground-truth instance mask plus rendered depth), run the
analytical ``synthesize_suction_grasps`` seal and wrench model to place the seal on the top surface,
drive the cup down its approach, then lift with a render-safe kinematic carry. The Isaac surface-gripper
binary bond will not weld a non-root articulation link, so the analytical seal carries the value and the
carry is the visible-lift surrogate, as in the KLT demo. Scored on the object's real world-Z lift.

The target is a flat-topped GSO package too wide for the 2F-85 to span, so it is the suction cup's job;
``SUCTION_TARGET_LABELS`` names the ones flagged as flat-top suction targets and ``--target`` defaults to
one of them. Run ``python -m src.willy_sim.gso_assets --convert`` first.

On-box only (needs Isaac). Run with Isaac's bundled python from the repo root:

    <isaac-sim>\\python.bat -m src.willy_sim.run_industrial_suction_pick --runs 3 --target "ink cartridge box"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.willy_sim.gso_assets import GSO_BY_LABEL, SUCTION_TARGET_LABELS, gso_object_spec
from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.harness.gate import GateResult, gate_passed

# A generous "the object rose with the cup" bound: a real seal lifts it to ~the retreat height; a much larger
# jump is a physics fling (the descent rammed it). Separates a genuine carry from a ram.
_MAX_LIFT_MM = 200.0


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Nearest proper rotation, by SVD: ``U @ Vt`` with a sign fix so ``det == +1``.

    Composing near-orthonormal axes, whether a Gram-Schmidt closing axis with a cross-product binormal or
    a product of measured wrist and object rotations, drifts the determinant about 1e-6 off 1.0, enough
    that ``from_rotation_matrix``'s strict ``det == +1`` validation aborts the run. Snapping to the
    nearest orthonormal matrix absorbs that drift, the same fix the planning stack uses on the
    real-hardware IK path.
    """
    u, _s, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rot = u @ vt
    if float(np.linalg.det(rot)) < 0.0:
        u[:, -1] = -u[:, -1]
        rot = u @ vt
    return rot


def run_suction_gate(
    runs: int = 3,
    *,
    target_label: str = "ink cartridge box",
    headless: bool = True,
    data_dir: str | None = None,
    lift_mm: float = 120.0,
    seal_threshold: float = 0.30,
    settle_steps: int = 40,
    out: str | None = None,
    dump_frame: str | None = None,
) -> GateResult:
    """Perceive, take the analytical suction seal, descend, carry kinematically and lift.

    Each run is scored on the lift the object achieves.
    """
    # Labels carry spaces, because they are prompt strings, and spaces do not survive argument splitting
    # through a shell. An underscore alias (``chocolate_box`` for ``chocolate box``) is accepted instead.
    target_label = target_label.replace("_", " ")
    if target_label not in GSO_BY_LABEL:
        raise SystemExit(f"unknown GSO target {target_label!r}; known: {sorted(GSO_BY_LABEL)}")
    if target_label not in SUCTION_TARGET_LABELS:
        print(f"note: {target_label!r} is not a flagged flat-top suction target {SUCTION_TARGET_LABELS}", flush=True)
    part = GSO_BY_LABEL[target_label]
    spec = gso_object_spec(part, position_mm=(450.0, 0.0, 60.0))  # dropped over the tray; settles flat
    print(f"=== INDUSTRIAL SUCTION pick (target={target_label!r}, GSO package the jaw can't span) ===", flush=True)

    from src.willy_sim.scene import ARM_PRIM, WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import SUCTION_CUP, mount_visible_suction_cup

    def _mount_cup(stage: Any) -> None:
        mount_visible_suction_cup(stage, ARM_PRIM, SUCTION_CUP, hide_jaw=True)

    cell = bootstrap_sim_cell(data_dir, headless=headless, scene_kwargs={"objects_override": [spec]},
                              post_scene_hook=_mount_cup)
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    session = arm.session
    gate = sim.scene_setup.gate
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    park_q = np.asarray(getattr(sim, "park_joint_positions", None) or sim.home_joint_positions, dtype=np.float64)

    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import SuctionConfig, synthesize_suction_grasps

    from src.willy_sim.perception import GroundTruthPerceptionSource

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    obj = SingleRigidPrim(handles.object_prim_path)
    wrist = SingleRigidPrim(WRIST_LINK_PRIM)
    home_pose = obj.get_world_pose()
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass

    # Real-top perception: the GT instance mask keeps the object clean, the real rendered depth (not a flat
    # GT patch) lets the seal model read the true top-surface geometry.
    perception = GroundTruthPerceptionSource(
        camera=handles.camera, target_prim_path=handles.object_prim_path, session=session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps), ground_truth_depth=False,
    )

    # Kinematic carry: while state["carry"], the object rides the wrist at the pose captured on contact.
    state: dict[str, Any] = {"carry": False, "p_rel": None, "r_rel": None}
    orig_step = session.step

    def _carry_step(dt: Any = None, *, render: bool = False) -> None:  # noqa: ARG001
        orig_step(dt, render=render)
        if state["carry"] and state["p_rel"] is not None:
            wp, wq = wrist.get_world_pose()
            r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
            pos = np.asarray(wp, dtype=np.float64) + r_w @ state["p_rel"]
            q = from_rotation_matrix(_orthonormalize(r_w @ state["r_rel"]))
            obj.set_world_pose(position=pos, orientation=np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
            try:
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass

    session.step = _carry_step  # type: ignore[assignment]

    results: list[dict[str, Any]] = []
    for i in range(runs):
        try:
            state["carry"] = False
            arm.move_joint(JointPositions(park_q))  # park clear so the overhead sees the object
            obj.set_world_pose(position=np.asarray(home_pose[0]), orientation=np.asarray(home_pose[1]))
            try:
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
            session.step_n(settle_steps)
            z0 = float(np.asarray(obj.get_world_pose()[0])[2]) * 1000.0

            frame = perception.acquire()
            mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
            tries = 0
            while (mask.ndim != 2 or not mask.any()) and tries < 8:
                session.step_n(5)
                frame = perception.acquire()
                mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
                tries += 1
            depth = np.asarray(frame.depth_map, dtype=np.float64)
            px = int(mask.sum())
            dvalid = int(np.isfinite(depth[mask]).sum()) if mask.any() else 0
            # Diagnostic: dump run-0's perception frame so the seal model can be iterated off-box (no Isaac
            # boot). Saves everything synthesize_suction_grasps consumes + the object/camera poses for context.
            if dump_frame and i == 0 and mask.any():
                dp = Path(dump_frame)
                dp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dp, mask=mask, depth_mm=depth, intrinsics=np.asarray(frame.intrinsics, dtype=np.float64),
                    camera_to_base=np.asarray(handles.camera_to_base.to_matrix(), dtype=np.float64),
                    obj_pose_pos_m=np.asarray(obj.get_world_pose()[0], dtype=np.float64),
                    cam_pose_pos_m=np.asarray(handles.camera.get_world_pose()[0], dtype=np.float64),
                )
                print(f"[dump] wrote perception frame -> {dp}", flush=True)
            grasps = synthesize_suction_grasps(
                mask, frame.depth_map, frame.intrinsics, camera_to_base=handles.camera_to_base,
                payload_mass_g=200.0,
                config=SuctionConfig(scorer=AnalyticalSuctionScorer(), max_results=8, min_quality=0.0),
            )
            _bm = grasps[0].metadata if grasps else {}
            print(f"RUN {i}: mask_px={px} depth_valid={dvalid} raw_candidates={len(grasps)} "
                  f"best_seal={(grasps[0].seal_score if grasps else 0.0):.3f} "
                  f"max_def_mm={_bm.get('max_deformation_mm', float('nan')):.2f} "
                  f"perim_support={_bm.get('perimeter_support', float('nan')):.2f} "
                  f"align={_bm.get('normal_alignment', float('nan')):.2f}", flush=True)
            if not grasps or grasps[0].seal_score < seal_threshold:
                seal = grasps[0].seal_score if grasps else 0.0
                print(f"RUN {i}: no sealable candidate (best seal={seal:.3f} < {seal_threshold}) -> fail", flush=True)
                results.append({"run": i, "success": False, "lift_mm": 0.0, "passed": False, "seal": round(seal, 3)})
                continue

            best = grasps[0]
            pos = np.asarray(best.position_mm, dtype=np.float64)
            approach = np.asarray(best.approach, dtype=np.float64)
            approach = approach / (float(np.linalg.norm(approach)) or 1.0)
            seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            closing = seed - approach * float(np.dot(seed, approach))
            closing = closing / (float(np.linalg.norm(closing)) or 1.0)
            binormal = np.cross(approach, closing)
            binormal = binormal / (float(np.linalg.norm(binormal)) or 1.0)
            quat = from_rotation_matrix(_orthonormalize(np.column_stack([closing, binormal, approach])))

            arm.move_joint(JointPositions(home_q))
            session.step_n(8)
            for _label, gap in (("standoff", 60.0), ("contact", 22.0)):
                arm.move(Pose(position_mm=pos - approach * gap, quaternion_xyzw=quat, frame=Frame.BASE,
                              label=f"suction-{_label}"))
            session.step_n(6)

            wp, wq = wrist.get_world_pose()
            op, oq = obj.get_world_pose()
            r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
            r_o = to_rotation_matrix(np.array([oq[1], oq[2], oq[3], oq[0]], dtype=np.float64))
            state["p_rel"] = r_w.T @ (np.asarray(op, dtype=np.float64) - np.asarray(wp, dtype=np.float64))
            state["r_rel"] = r_w.T @ r_o
            state["carry"] = True
            session.step_n(6)
            arm.move(Pose(position_mm=pos - approach * 22.0 + (-approach) * float(lift_mm),
                          quaternion_xyzw=quat, frame=Frame.BASE, label="suction-lift"))
            session.step_n(12)

            z1 = float(np.asarray(obj.get_world_pose()[0])[2]) * 1000.0
            lift = z1 - z0
            passed = bool(gate.lift_threshold_mm <= lift <= _MAX_LIFT_MM)
            results.append({"run": i, "success": passed, "lift_mm": round(lift, 1), "passed": passed,
                            "seal": round(float(best.seal_score), 3), "quality": round(float(best.quality), 3)})
            print(f"RUN {i}: seal={best.seal_score:.3f} quality={best.quality:.3f} lift={lift:.1f}mm passed={passed}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 (a motion/seal failure = a failed run, not a crash)
            print(f"RUN {i}: ABORTED ({type(exc).__name__}: {exc})", flush=True)
            results.append({"run": i, "success": False, "lift_mm": 0.0, "passed": False, "aborted": str(exc)})
        finally:
            state["carry"] = False

    session.step = orig_step  # type: ignore[assignment]
    n_pass = sum(1 for r in results if r["passed"])
    gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
    print(f"\n=== INDUSTRIAL SUCTION [{target_label}]: {n_pass}/{runs} lifted (>= {gate.lift_threshold_mm:.0f} mm) "
          f"-> gate_passed={gate_ok} ===", flush=True)

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "target": target_label, "runs": runs, "n_success": n_pass,
            "pick_rate": round(n_pass / runs, 3) if runs else 0.0, "generated_unix": time.time(),
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"[report] wrote {out_path}", flush=True)
    return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, target=target_label)


def main() -> int:
    ap = argparse.ArgumentParser(description="Industrial suction pick: seal + lift a flat-top GSO package.")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--target", default="ink cartridge box",
                    help=f"GSO suction target label (underscores ok); validated: {SUCTION_TARGET_LABELS}")
    ap.add_argument("--lift-mm", type=float, default=120.0)
    ap.add_argument("--seal-threshold", type=float, default=0.30)
    ap.add_argument("--data", default=None)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dump-frame", type=str, default=None,
                    help="save run-0's perception frame (mask+depth+intrinsics) to this .npz for off-box seal analysis")
    args = ap.parse_args()
    result = run_suction_gate(
        runs=args.runs, target_label=args.target, headless=not args.gui, data_dir=args.data,
        lift_mm=args.lift_mm, seal_threshold=args.seal_threshold, out=args.out, dump_frame=args.dump_frame,
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
