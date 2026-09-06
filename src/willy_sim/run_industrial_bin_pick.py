"""Industrial bin-pick: two fixed side cameras localize the prompted part in a KLT tray of mixed parts,
the full grasp pipeline picks it, and every attempt is scored on lift plus an attributable failure
taxonomy.

The industrial-cell showcase, built from the existing pieces:

* Sensing is two fixed eye-to-hand cameras, one left and one right, with no fixed overhead
  (``oblique_L`` and ``oblique_R`` from the sim config). The reaching arm is the dominant top-down
  occluder, so two symmetric side views see the target past it and fuse into one BASE-frame
  localization. The wrist eye-in-hand camera then does the close-up grasp, arm-mounted, not overhead.
* The scene is a real small_KLT bin: a visible mesh plus collidable walls registered into the
  SafetyPreflight and cuRobo's world, so a plan routes around them and an approach into a wall is
  fail-closed rejected. It holds a settled layout of varied jaw-graspable parts. ``tray``, a shallow
  Z-squashed KLT the 2F-85 reaches, is the jaw's base; the full-depth ``klt`` is the suction cup's
  domain, because the jaw wrist cannot descend the 146 mm-deep narrow bin without grazing a wall and
  the Coal guard rejects it.
* The pick is the full grasp pipeline. The two side cameras localize the prompted target among the
  clutter, the wrist is driven over its fused centroid, and ``AutonomousGraspService.pick()`` runs the
  same path the other runners use: perceive, GraspCalculator, SafetyPreflight, cuRobo and Coal motion,
  then the gripper.

Each attempt re-randomizes the layout and is classified (``no_localize``, ``collision_fling``,
``slip_or_ram``, ``execution_failed``), so the pick rate carries an attributable failure breakdown.
Perception is ground-truth here, which measures the generation ceiling for the two-side rig; a
real-detection variant is the next pass. The prompted part is fixed in code as ``TARGET_PART``, not
chosen on the command line.

On-box only (needs Isaac). Run with Isaac's bundled python from the repo root (cuRobo + Coal env set):

    <isaac-sim>\\python.bat -m src.willy_sim.run_industrial_bin_pick --runs 10
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.config.schema.robot import SimObjectConfig
from src.willy_sim.config import require_robot
from src.willy_sim.harness.gate import GateResult, gate_passed
from src.willy_sim.harness.randomizer import DomainRandomizer, RandomizationConfig
from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource
from src.willy_sim.run_eih_pick import _load_calibrated_eih, build_service, topdown_view_pose
from src.willy_sim.run_multiview_pick import BIN_PRESETS, _bin_preset_walls, multi_view_localize
from src.willy_sim.scene import OBJECT_PRIM, author_fixed_camera, camera_to_base_ground_truth

# The two fixed side cameras that make up the rig (config poses come from sim.cameras[...]). No overhead.
SIDE_CAMERAS = ("oblique_L", "oblique_R")

# A real parallel-jaw pick lifts to about the retreat height (~100 mm) and holds. A larger apparent
# "lift" is a physics fling: the gripper rammed a part and PhysX launched it. That is a failure, not a
# pick, so this bound separates a genuine hold from a fling (a negative lift = the part was rammed down).
RETREAT_MAX_MM = 160.0

# The prompted target part (placed at the tray centre so it is reliably reachable) plus varied clutter
# parts around it (the "bin" of mixed parts). Every footprint fits the 2F-85 span; sizes and colours
# differ. Each entry is (name, RGB, size_mm). Cubes and boxes only: a cylinder rolls, a separate problem.
TARGET_PART: tuple[str, tuple[float, float, float], tuple[float, float, float]] = (
    "green block", (0.15, 0.62, 0.22), (30.0, 30.0, 30.0),
)
CLUTTER_PARTS: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
    ("red block", (0.82, 0.16, 0.16), (35.0, 35.0, 35.0)),
    ("blue block", (0.16, 0.28, 0.85), (30.0, 42.0, 30.0)),
    ("yellow block", (0.88, 0.76, 0.14), (34.0, 30.0, 34.0)),
    ("magenta block", (0.78, 0.16, 0.55), (42.0, 30.0, 28.0)),
    ("cyan block", (0.14, 0.68, 0.72), (36.0, 34.0, 30.0)),
]
# Clutter placed at the tray periphery (inside +/-146 x, +/-94 y but clear of the central target), so the
# target's top-down grasp corridor stays open. Denser layouts, where the descent must thread between
# touching parts, need neighbour-aware grasp rejection: the fused-scene collision seam, a later step.
_CLUTTER_XY_MM = [(-95.0, -55.0), (95.0, -55.0), (-95.0, 55.0), (95.0, 55.0), (0.0, 65.0)]


def _scene_specs(clutter_count: int, *, gso_clutter: bool = False) -> list[SimObjectConfig]:
    """The target cube at the tray centre + ``clutter_count`` clutter parts at the periphery.

    ``gso_clutter`` swaps the primitive blocks for real Aletheia GSO objects (pre-rigged converted USD) as
    dynamic bodies dropped around the target. The target stays a graspable cube; the GSO objects are
    realistic distractors around it.
    """
    tname, tcolor, tsize = TARGET_PART
    specs = [SimObjectConfig(name=tname, shape="cube", size_mm=tsize, position_mm=(450.0, 0.0, 25.0),
                             mass_kg=0.15, color=tcolor)]
    if gso_clutter:
        from src.willy_sim.gso_assets import GSO_PARTS, gso_object_spec

        for i in range(min(clutter_count, len(GSO_PARTS), len(_CLUTTER_XY_MM))):
            cx, cy = _CLUTTER_XY_MM[i]
            specs.append(gso_object_spec(GSO_PARTS[i], position_mm=(450.0 + cx, cy, 70.0)))  # dropped, settles
        return specs
    for i in range(min(clutter_count, len(CLUTTER_PARTS))):
        name, color, size = CLUTTER_PARTS[i]
        cx, cy = _CLUTTER_XY_MM[i]
        specs.append(SimObjectConfig(name=name, shape="cube", size_mm=size,
                                     position_mm=(450.0 + cx, cy, 25.0 + 20.0 * (i % 3)), mass_kg=0.15, color=color))
    return specs


def run_industrial_gate(
    runs: int = 10,
    *,
    scene: str = "tray",
    clutter_count: int = 4,
    gso_clutter: bool = False,
    headless: bool = True,
    data_dir: str | None = None,
    marker: str = "ground_truth",
    view_height_mm: float = 275.0,
    seed: int = 0,
    jitter_mm: float = 8.0,
    settle_steps: int = 150,
    out: str | None = None,
) -> GateResult:
    """Two-side localize, then wrist refine, then the full-pipeline pick, scored on lift and failure class."""
    bin_walls, real_klt = _bin_preset_walls(scene)
    specs = _scene_specs(clutter_count, gso_clutter=gso_clutter)
    target_label = TARGET_PART[0]
    target_idx = 0  # the target is authored first, so it is Object_0
    target_prim = f"{OBJECT_PRIM}_{target_idx}"
    print(f"=== INDUSTRIAL BIN PICK (scene={scene!r}: two side cameras + KLT bin, target={target_label!r} "
          f"among {len(specs) - 1} clutter parts) ===", flush=True)

    # The full pick service in the industrial bin: wrist eye-in-hand build_service with no fixed overhead,
    # targeting the prompted part, plus the bin walls (SafetyPreflight fixtures and cuRobo world) and the
    # real KLT visual. Any GSO clutter rides in ``specs`` as pre-rigged dynamic bodies, authored with the
    # scene, before perception.
    service, arm, gripper, handles, cfg, _view_pose, _cell = build_service(
        headless=headless, data_dir=data_dir, marker=marker, objects_override=specs,
        wrist_target_prim=target_prim, bin_walls=bin_walls, real_klt_bin=real_klt, curobo_bin_world=True,
    )
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    robot = require_robot(cfg)
    sim = robot.sim
    gate = sim.scene_setup.gate
    session = arm.session
    park_q = np.asarray(sim.park_joint_positions, dtype=np.float64)
    object_specs = list(handles.object_specs)  # [(prim_path, label), ...] in spec order
    objs = [SingleRigidPrim(p) for (p, _l) in object_specs]
    homes = [o.get_world_pose() for o in objs]

    cal = _load_calibrated_eih(marker)
    if cal is None:
        raise SystemExit(f"industrial bin pick needs a calibrated EIH artifact (marker={marker})")
    p_cam_tool = np.asarray(cal.to_matrix(), dtype=np.float64)[:3, 3]

    # Author the two fixed side cameras (config poses) + fit their CAMERA->BASE ground-truth extrinsics.
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001 (near-clip best-effort)
        pass
    cameras: dict[str, tuple[MultiObjectGroundTruthPerceptionSource, Any]] = {}
    for name in SIDE_CAMERAS:
        ccfg = sim.cameras.get(name)
        if ccfg is None or ccfg.position_mm is None or ccfg.mount_aim_mm is None:
            raise SystemExit(f"config sim.cameras[{name!r}] must define position_mm + mount_aim_mm")
        cam = author_fixed_camera(
            session, ccfg.prim_path or f"/World/Cameras/{name}",
            position_mm=ccfg.position_mm, aim_mm=ccfg.mount_aim_mm,
            near_clip_m=ccfg.near_clip_m or 0.05,
            resolution=ccfg.resolution if ccfg.resolution else (640, 480),
        )
        c2b, rmse = camera_to_base_ground_truth(cam)
        print(f"{name} @ {tuple(ccfg.position_mm)} -> aim {tuple(ccfg.mount_aim_mm)} (cam->base fit rmse={rmse:.1f} mm)",
              flush=True)
        cameras[name] = (
            MultiObjectGroundTruthPerceptionSource(camera=cam, targets=object_specs, session=session, warmup_steps=6),
            c2b,
        )

    # Position jitter only (a fresh layout each attempt); no yaw jitter, because a rotated cube changes
    # the top-down grasp geometry and needs a separate accommodation (adaptive close, oriented grasp).
    randomizer = DomainRandomizer(RandomizationConfig(
        enabled=True, seed=int(seed), position_jitter_mm=float(jitter_mm), orientation_jitter_deg=0.0,
    ))
    home_pos_mm = [tuple(np.asarray(hp, dtype=np.float64) * 1000.0) for (hp, _hq) in homes]
    home_quat = [tuple(np.asarray(hq, dtype=np.float64)) for (_hp, hq) in homes]

    results: list[dict[str, Any]] = []
    for i in range(runs):
        try:
            # Reset: open, park the arm clear of the bin, re-randomize the layout, settle.
            gripper.open()
            arm.move_to_joints(JointPositions(park_q))
            ep = randomizer.sample_pose(i, home_pos_mm, home_quat)
            for o, pos_mm, quat in zip(objs, ep.positions_mm, ep.orientations_wxyz):
                o.set_world_pose(position=np.asarray(pos_mm, dtype=np.float64) / 1000.0,
                                 orientation=np.asarray(quat, dtype=np.float64))
                try:
                    o.set_linear_velocity(np.zeros(3))
                    o.set_angular_velocity(np.zeros(3))
                except Exception:  # noqa: BLE001 (velocity-zero is best-effort)
                    pass
            session.step_n(settle_steps)

            z0 = float(np.asarray(objs[target_idx].get_world_pose()[0])[2]) * 1000.0
            fused_c, per = multi_view_localize(cameras, target_idx)
            true_c = np.asarray(objs[target_idx].get_world_pose()[0], dtype=np.float64) * 1000.0
            if fused_c is None:
                print(f"RUN {i}: NO side camera saw {target_label!r} -> fail  per={per}", flush=True)
                results.append({"run": i, "success": False, "lift_mm": 0.0, "passed": False,
                                "fail_class": "no_localize"})
                continue
            err = float(np.linalg.norm((fused_c - true_c)[:2]))
            print(f"RUN {i}: localized {target_label!r} XY={np.round(fused_c[:2], 1)} (true {np.round(true_c[:2], 1)}, "
                  f"err {err:.1f} mm) from {[k for k, v in per.items() if v['target_px'] > 0]}", flush=True)

            # Drive the wrist over the fused centroid, then the full-pipeline pick.
            arm.move(topdown_view_pose(fused_c, p_cam_tool, view_height_mm))
            report = service.pick()
            z1 = float(np.asarray(objs[target_idx].get_world_pose()[0])[2]) * 1000.0
            lift = z1 - z0
            outcome_str = str(getattr(getattr(report, "outcome", None), "value", "?"))
            other_lifts = [
                (float(np.asarray(o.get_world_pose()[0])[2]) * 1000.0
                 - float(np.asarray(h[0])[2]) * 1000.0)
                for o, h in zip(objs, homes) if o is not objs[target_idx]
            ]
            flung = lift > RETREAT_MAX_MM or any(lf > RETREAT_MAX_MM for lf in other_lifts)
            holds = gate.lift_threshold_mm <= lift <= RETREAT_MAX_MM
            succeeded = outcome_str == "succeeded"
            success = bool(succeeded and holds and not flung)
            fail_class: str | None
            if success:
                fail_class = None
            elif flung:
                fail_class = "collision_fling"       # the gripper rammed a part and PhysX launched it
            elif succeeded and not holds:
                fail_class = "slip_or_ram"           # closed but did not hold (rammed down / slipped)
            else:
                fail_class = "execution_failed"      # a plan/motion failed (e.g. a fail-closed reject)
            results.append({"run": i, "target": target_label, "success": success, "lift_mm": round(lift, 1),
                            "passed": success, "localize_err_mm": round(err, 1), "outcome": outcome_str,
                            "fail_class": fail_class})
            print(f"RUN {i}: outcome={outcome_str} lift={lift:.1f}mm success={success} fail={fail_class}", flush=True)
        except Exception as exc:  # noqa: BLE001 (a safety abort / motion failure = a failed run, not a crash)
            print(f"RUN {i}: ABORTED ({type(exc).__name__}: {exc})", flush=True)
            results.append({"run": i, "success": False, "lift_mm": 0.0, "passed": False,
                            "fail_class": "pick_error", "aborted": str(exc)})

    n_pass = sum(1 for r in results if r["passed"])
    gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
    fail_counts = Counter(r["fail_class"] for r in results if r.get("fail_class") is not None)
    print(f"\n=== INDUSTRIAL BIN PICK [{scene}]: {n_pass}/{runs} lifted (>= {gate.lift_threshold_mm:.0f} mm) "
          f"-> gate_passed={gate_ok} | failures={dict(fail_counts)} ===", flush=True)

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "scene": scene, "target": target_label, "runs": runs, "clutter_count": clutter_count,
            "n_success": n_pass, "pick_rate": round(n_pass / runs, 3) if runs else 0.0,
            "lift_threshold_mm": float(gate.lift_threshold_mm), "failure_breakdown": dict(fail_counts),
            "generated_unix": time.time(), "results": results,
        }, indent=2), encoding="utf-8")
        print(f"[report] wrote {out_path}", flush=True)
    return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, target=target_label)


def main() -> int:
    ap = argparse.ArgumentParser(description="Industrial bin pick: two side cameras + KLT bin + full grasp pipeline.")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--scene", default="tray", choices=sorted(BIN_PRESETS),
                    help="'tray' (shallow KLT the jaw reaches, the base) | 'klt' (deep bin, suction's domain)")
    ap.add_argument("--clutter-count", type=int, default=4, help="number of clutter parts around the target (0-5)")
    ap.add_argument("--gso-clutter", action="store_true",
                    help="use real Aletheia GSO objects (converted USD meshes) as clutter around the target "
                         "(run 'gso_assets --convert' first). Validated on-box: cube target picks 3/3 among "
                         "3 GSO clutter objects.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter-mm", type=float, default=12.0)
    ap.add_argument("--settle-steps", type=int, default=150)
    ap.add_argument("--data", default=None)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", type=str, default=None, help="write the JSON pick-rate + failure-breakdown report here")
    args = ap.parse_args()
    result = run_industrial_gate(
        runs=args.runs, scene=args.scene, clutter_count=args.clutter_count, gso_clutter=args.gso_clutter,
        headless=not args.gui, data_dir=args.data, seed=args.seed, jitter_mm=args.jitter_mm,
        settle_steps=args.settle_steps, out=args.out,
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
