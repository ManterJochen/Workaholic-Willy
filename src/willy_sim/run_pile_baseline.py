"""Dense-clutter pile baseline: the current default open-loop pick, measured on a piled scene.

The floor a later lever is quoted against. It runs the default open-loop pick on a genuinely piled,
touching, occluded scene of varied primitives rather than the well-separated cubes the stack was
validated on, and reports the pick rate with a failure-mode breakdown, so a fusion, perception or
generation change reads as a delta against this floor.

Two perception modes, one per invocation, because SimulationApp is a singleton and a process boots it
once:
  * ``--perception vision`` is the end-to-end floor: GroundingDINO detect_all plus SAM2.
  * ``--perception gt``     is the perception ceiling: Isaac instance-id masks plus true depth.
Running both and diffing the two reports separates perception-limited from generation-limited failure.
A large gap between vision and gt means perception is the constraint; a small gap means the geometric
generator is.

Each attempt parks the arm, randomizes and settles a fresh pile, targets the top settled object, calls
``service.pick()``, measures that object's real lift against the sim gate threshold, and classifies the
result as ``no_grasp_found``, ``collision_fling``, ``slip_or_ram``, ``execution_failed`` or
``pick_error``. It reuses ``run_dense_pick.build_service(pile=True, ...)``, so the pile scene, the
shallow bin, GT or vision perception, cuRobo with Coal and ``SafetyPreflight`` are all the standard
wiring: nothing in the pick path is changed, and this measures the real stack.

On-box only (needs Isaac). Imports stay macOS and CI safe: Isaac is lazy inside build_service and the
attempt loop.

    <isaac-sim>\\python.bat -m src.willy_sim.run_pile_baseline \\
        --runs 20 --perception vision --pile-count 8 --out logs/baselines/pile_vision.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.utility.log_cfg import create_logger
from src.willy_sim.config import require_robot
from src.willy_sim.constants import PILE_BASELINE_LOG_FILE, WILLY_SIM_LOG_DIR
from src.willy_sim.harness.randomizer import DomainRandomizer, RandomizationConfig
from src.willy_sim.run_dense_pick import build_service

#: This runner's number is a floor that later levers are quoted against, so the conditions behind it
#: are part of the result: perception mode, pile density, and which opt-in blocks were on. The report
#: JSON carries them, and the CLI always writes one: ``--out`` only moves it off the default
#: ``logs/baselines/pile_<perception>.json``. A baseline is still quoted far more often than the file
#: behind it is opened.
_LOG = create_logger("PileBaseline", log_file=PILE_BASELINE_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)

# A real parallel-jaw grasp lifts the object to about the retreat height (~100 mm) and holds it. A
# larger apparent lift is a sim-physics collision fling: the gripper rammed the tight pile and PhysX
# launched an object. That is a failure, not a pick, and this bound separates a genuine hold from a
# fling. A negative lift means the object was rammed down.
RETREAT_MAX_MM = 160.0


def _top_object_index(
    objs: list, z_settled: list[float], center_xy_mm: tuple[float, float], reach_half_mm: float
) -> int:
    """Index of the highest settled object still inside the tray footprint.

    That is the object a picker goes for on a pile. Restricting to the footprint ignores an object that
    bounced out of the tray on settle; falls back to the global argmax(z) when none qualify.
    """
    cx, cy = center_xy_mm
    in_tray: list[int] = []
    for k, o in enumerate(objs):
        p = np.asarray(o.get_world_pose()[0], dtype=np.float64) * 1000.0
        if abs(float(p[0]) - cx) <= reach_half_mm and abs(float(p[1]) - cy) <= reach_half_mm and z_settled[k] > 5.0:
            in_tray.append(k)
    pool = in_tray if in_tray else list(range(len(objs)))
    return max(pool, key=lambda k: z_settled[k])


def run_baseline(
    runs: int = 20,
    *,
    perception: str = "vision",
    headless: bool = True,
    data_dir: str | None = None,
    pile_count: int = 8,
    pile_spread_mm: float = 50.0,
    pile_shapes: str = "mixed",
    enable_bin: bool = True,
    curobo_bin_world: bool = False,
    adaptive_close: bool = False,
    top_ref: bool = False,
    enable_g12: bool = False,          # swept-approach validation: reject candidates whose approach
    #                                    sweep hits a neighbour, not just the final pose
    enable_g4: bool = False,           # per-candidate corridor-risk producer (approach clearance)
    g4_rerank: bool = False,           # G4 uncertainty rerank consumer: demote blocked-corridor candidates
    g4_rerank_weight: float = 0.3,
    oblique: bool = False,             # multi-angle approach generation (top-down plus tilts)
    oblique_tilt_deg: float = 30.0,
    oblique_azimuths: int = 4,
    multiview: bool = False,           # two fixed oblique cameras, fused into a BASE target cloud of
    #                                    side faces, fed to the generator for grasps in clutter
    bin_half_width_mm: float = 110.0,
    bin_height_mm: float = 50.0,
    seed: int = 0,
    jitter_mm: float = 25.0,
    settle_steps: int = 150,
    out: str | None = None,
) -> dict:
    """Run ``runs`` pile picks and return the aggregated baseline report (also written to ``out``)."""
    vision = perception == "vision"
    center_xy = (450.0, 0.0)
    # The conditions are logged before the picks that follow, so a run killed halfway still leaves a
    # record of what it was measuring. Only the non-default blocks are named: a floor taken with G12
    # or multiview on is not the same floor.
    _LOG.info(
        "pile baseline starting: %d run(s), perception=%s, pile=%d objects spread=%.0f mm shapes=%s, "
        "bin=%s, seed=%d, opt-in=%s",
        runs, perception, pile_count, pile_spread_mm, pile_shapes, enable_bin, seed,
        ", ".join(n for n, on in (
            ("curobo_bin_world", curobo_bin_world), ("adaptive_close", adaptive_close),
            ("top_ref", top_ref), ("g12", enable_g12), ("g4", enable_g4), ("g4_rerank", g4_rerank),
            ("oblique", oblique), ("multiview", multiview),
        ) if on) or "none (the anchored default config)",
    )
    service, arm, gripper, handles, cfg, _target_idx, _target_label = build_service(
        prompt="the top object", headless=headless, data_dir=data_dir, mode="dense_clutter",
        pile=True, pile_count=pile_count, pile_seed=seed, pile_spread_mm=pile_spread_mm,
        pile_shapes=pile_shapes, pile_top_ref=top_ref,
        vision=vision, enable_bin=enable_bin, curobo_bin_world=curobo_bin_world,
        enable_g12=enable_g12, enable_g4=enable_g4, g4_rerank=g4_rerank, g4_rerank_weight=g4_rerank_weight,
        oblique_approach=oblique, oblique_tilt_deg=oblique_tilt_deg, oblique_azimuths=oblique_azimuths,
        bin_half_width_mm=bin_half_width_mm, bin_height_mm=bin_height_mm,
        # adaptive_close and top_ref are opt-in. Default off is the run_dense_pick configuration: fixed
        # close_width, centre-referenced depth. Each varied-object accommodation is enabled one at a
        # time, so its effect on the floor stays separable.
        adaptive_close=adaptive_close,
    )

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    robot = require_robot(cfg)
    gate = robot.sim.scene_setup.gate
    park_q = np.asarray(robot.sim.park_joint_positions, dtype=np.float64)
    labels = [lbl for (_p, lbl) in handles.object_specs]
    prims = [p for (p, _lbl) in handles.object_specs]
    objs = [SingleRigidPrim(p) for p in prims]
    homes = [o.get_world_pose() for o in objs]
    # Per-attempt pile variety: position and yaw jitter give a fresh arrangement each run. Pose only,
    # no colour or lighting, so the object identities and labels stay stable for the target match.
    rand_cfg = RandomizationConfig(
        enabled=True, seed=int(seed), position_jitter_mm=float(jitter_mm), orientation_jitter_deg=20.0,
    )
    randomizer = DomainRandomizer(rand_cfg)
    reach_half_mm = bin_half_width_mm + 20.0

    # Optional multi-view rig: two fixed oblique cameras, which see object sides the overhead cannot,
    # with per-camera ground-truth perception and ground-truth-fit extrinsics. Each attempt then fuses a
    # BASE-frame target cloud of side-face points into orch.external_target_geometry_base_mm.
    orch = getattr(getattr(service, "runtime", None), "orchestrator", None)
    mv_cameras: "dict[str, tuple[Any, Any]] | None" = None
    _build_fused_target: Any = None
    _build_fused_scene: Any = None
    if multiview:
        from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource
        from src.willy_sim.run_multiview_pick import (
            build_fused_scene_cloud,
            build_fused_target_cloud,
        )
        from src.willy_sim.run_occlusion_probe import _author_oblique_camera
        from src.willy_sim.scene import camera_to_base_ground_truth

        _build_fused_target = build_fused_target_cloud
        _build_fused_scene = build_fused_scene_cloud
        _specs = list(handles.object_specs)
        try:
            handles.camera.set_clipping_range(0.05, 1.0e6)
        except Exception:  # noqa: BLE001 (near-clip best-effort)
            pass
        _t_over = camera_to_base_ground_truth(handles.camera)[0]
        mv_cameras = {"overhead": (
            MultiObjectGroundTruthPerceptionSource(
                camera=handles.camera, targets=_specs, session=arm.session, warmup_steps=6),
            _t_over,
        )}
        for _cn in ("oblique_L", "oblique_R"):
            _cc = robot.sim.cameras.get(_cn)
            if _cc is None or _cc.position_mm is None or _cc.mount_aim_mm is None:
                raise SystemExit(f"--multiview needs sim.cameras[{_cn!r}] position_mm + mount_aim_mm")
            _cam = _author_oblique_camera(
                arm.session, _cc.prim_path or f"/World/Cameras/{_cn}",
                position_mm=_cc.position_mm, aim_mm=_cc.mount_aim_mm,
                near_clip_m=_cc.near_clip_m or 0.05,
                resolution=_cc.resolution if _cc.resolution else (640, 480),
            )
            _c2b, _rmse = camera_to_base_ground_truth(_cam)
            print(f"[multiview] {_cn} cam->base GT-fit rmse={_rmse:.1f}mm", flush=True)
            mv_cameras[_cn] = (
                MultiObjectGroundTruthPerceptionSource(
                    camera=_cam, targets=_specs, session=arm.session, warmup_steps=6),
                _c2b,
            )
        print(f"[multiview] rig ready: {list(mv_cameras)}", flush=True)

    print(f"=== PILE BASELINE: perception={perception} objects={pile_count} spread={pile_spread_mm}mm "
          f"tray=+/-{bin_half_width_mm}mm h={bin_height_mm}mm runs={runs} ===", flush=True)

    results: list[dict] = []
    for i in range(runs):
        gripper.open()
        arm.move_to_joints(JointPositions(park_q))
        _home_pos_mm = [tuple(np.asarray(hp, dtype=np.float64) * 1000.0) for (hp, _hq) in homes]
        _home_quat = [tuple(np.asarray(hq, dtype=np.float64)) for (_hp, hq) in homes]
        ep = randomizer.sample_pose(i, _home_pos_mm, _home_quat)
        for o, pos_mm, quat in zip(objs, ep.positions_mm, ep.orientations_wxyz):
            o.set_world_pose(position=np.asarray(pos_mm, dtype=np.float64) / 1000.0,
                             orientation=np.asarray(quat, dtype=np.float64))
            try:
                o.set_linear_velocity(np.zeros(3))
                o.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001 (velocity zero is best-effort)
                pass
        arm.session.step_n(settle_steps)  # let the pile drop + settle (touching/overlapping)
        z0 = [float(np.asarray(o.get_world_pose()[0])[2]) * 1000.0 for o in objs]
        pile_height_mm = round(max(z0) - min(z0), 1)  # occlusion/stacking proxy for the pile
        # Target the top object: the clear-the-bin proxy a real bin picker uses. The target is always a
        # labelled object, never set_target_label(None); a pick-any target misses the grasp on
        # separated cubes.
        tgt = _top_object_index(objs, z0, center_xy, reach_half_mm)
        target_label = labels[tgt]
        service.set_target_label(target_label)

        # Fuse a BASE-frame target cloud of side-face points from the obliques, plus the neighbour
        # cloud, and hand both to the orchestrator, so the generator searches real 3D contacts and the
        # collision filter sees fused neighbours. With mv_cameras None the seam keys stay absent and the
        # run is byte-identical to the default.
        if mv_cameras is not None and orch is not None and _build_fused_target is not None:
            _per = _build_fused_target(mv_cameras, target_idx=tgt)
            orch.external_target_geometry_base_mm = np.vstack(list(_per.values())) if _per else None
            if _build_fused_scene is not None:
                orch.external_scene_points_base_mm = _build_fused_scene(mv_cameras, target_idx=tgt)

        try:
            report = service.pick()
            pick_error = None
        except Exception as exc:  # noqa: BLE001 (keep the gate running; record the failure honestly)
            report = None
            pick_error = f"{type(exc).__name__}: {exc}"

        z1 = [float(np.asarray(o.get_world_pose()[0])[2]) * 1000.0 for o in objs]
        lifts_all = [z1[k] - z0[k] for k in range(len(objs))]
        outcome_value = getattr(getattr(report, "outcome", None), "value", None)
        pr = getattr(report, "pick_report", None)
        pick_outcome = getattr(getattr(pr, "outcome", None), "value", None)
        # Attribution: split an opaque `no_grasp_found` into perception and generation versus the
        # downstream filters. `reasons` is the last attempt's GraspFailureReason set (MASK_TOO_SMALL or
        # NO_CANDIDATES against ALL_COLLIDED or IK_FAILED); the calculator telemetry counts what
        # generation produced and what the collision and IK filters discarded. Both are nullable.
        _attempts = getattr(pr, "attempts", None) or ()
        reasons = [str(getattr(r, "value", r)) for r in (_attempts[-1].reasons if _attempts else ())]
        _calc_tel = dict(getattr(getattr(orch, "calculator", None), "last_telemetry", None) or {})
        executed = outcome_value == "succeeded"
        flung = any(lf > RETREAT_MAX_MM for lf in lifts_all)  # any object launched by a ramming collision
        clean = [lf for lf in lifts_all if gate.lift_threshold_mm <= lf <= RETREAT_MAX_MM]
        picked_lift = lifts_all[tgt]  # the targeted (top) object's lift
        holds = gate.lift_threshold_mm <= picked_lift <= RETREAT_MAX_MM
        # Honest success: executed a grasp and a clean hold to ~retreat height and no violent fling of any object.
        success = bool(executed and holds and not flung and pick_error is None)
        max_other = max((lifts_all[k] for k in range(len(objs)) if k != tgt), default=0.0)
        fail_class: str | None
        if success:
            fail_class = None
        elif pick_error is not None:
            fail_class = "pick_error"
        elif outcome_value == "no_valid_grasp" or pick_outcome in ("no_perception", "rescanned_exhausted"):
            fail_class = "no_grasp_found"          # perception/generation produced nothing to try
        elif flung:
            fail_class = "collision_fling"         # gripper rammed the pile and PhysX launched an object
        elif executed and not clean:
            fail_class = "slip_or_ram"             # descended/closed but did not hold (rammed down / slipped)
        else:
            fail_class = "execution_failed"        # motion/plan failed with no large disturbance
        row = {
            "run": i, "target": target_label, "top_idx": tgt,
            "picked_lift_mm": round(picked_lift, 1), "max_other_lift_mm": round(max_other, 1),
            "success": success, "pile_height_mm": pile_height_mm,
            "outcome": outcome_value, "pick_outcome": pick_outcome,
            "fail_class": fail_class, "pick_error": pick_error,
            "reasons": reasons,
            "candidates_silhouette": _calc_tel.get("candidates_silhouette"),
            "candidates_geometry": _calc_tel.get("candidates_geometry"),
            "rejected_collision": _calc_tel.get("rejected_collision"),
            "rejected_ik": _calc_tel.get("rejected_ik"),
        }
        results.append(row)
        print(f"RUN {i}: target={target_label!r} picked_lift={picked_lift:.1f} success={success} "
              f"outcome={outcome_value} pick_outcome={pick_outcome} fail={fail_class} "
              f"pile_h={pile_height_mm} flung={flung}", flush=True)

    n_success = sum(1 for r in results if r["success"])
    fail_counts = Counter(r["fail_class"] for r in results if r["fail_class"] is not None)
    outcome_counts = Counter(str(r["outcome"]) for r in results)
    lifts = [r["picked_lift_mm"] for r in results if r["success"]]
    report_dict = {
        "iteration": "1-pile-baseline",
        "opportunity": "GraspingRevisit Opp 0 (real dense-clutter floor)",
        "perception": perception,
        "target_mode": "top_object",
        "config": {"enable_bin": enable_bin, "curobo_bin_world": curobo_bin_world,
                   "adaptive_close": adaptive_close, "top_ref": top_ref, "pile_shapes": pile_shapes,
                   "enable_g12": enable_g12, "enable_g4": enable_g4, "g4_rerank": g4_rerank,
                   "oblique": oblique, "oblique_tilt_deg": oblique_tilt_deg, "oblique_azimuths": oblique_azimuths,
                   "multiview": multiview},
        "runs": runs,
        "pile_count": pile_count,
        "pile_spread_mm": pile_spread_mm,
        "bin_half_width_mm": bin_half_width_mm,
        "bin_height_mm": bin_height_mm,
        "seed": seed,
        "lift_threshold_mm": float(gate.lift_threshold_mm),
        "pick_rate": round(n_success / runs, 3) if runs else 0.0,
        "n_success": n_success,
        "failure_breakdown": dict(fail_counts),
        "outcome_breakdown": dict(outcome_counts),
        "mean_picked_lift_mm": round(float(np.mean(lifts)), 1) if lifts else 0.0,
        "mean_pile_height_mm": round(float(np.mean([r["pile_height_mm"] for r in results])), 1) if results else 0.0,
        "generated_unix": time.time(),
        "results": results,
    }
    print(f"\n=== PILE BASELINE [{perception}/{report_dict['target_mode']}]: "
          f"pick_rate={report_dict['pick_rate']} ({n_success}/{runs}) | failures={dict(fail_counts)} ===",
          flush=True)

    # The floor itself, with the failure breakdown beside it: pick_rate alone cannot say whether a low
    # number is perception-limited or generation-limited.
    _LOG.info(
        "pile baseline [%s]: pick_rate=%.3f (%d/%d), failures=%s, outcomes=%s, mean lift=%.1f mm, "
        "mean pile height=%.1f mm",
        perception, report_dict["pick_rate"], n_success, runs, dict(fail_counts), dict(outcome_counts),
        report_dict["mean_picked_lift_mm"], report_dict["mean_pile_height_mm"],
    )
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _text = json.dumps(report_dict, indent=2)
        out_path.write_text(_text, encoding="utf-8")
        _LOG.info("baseline report written -> %s (%d bytes)", out_path, len(_text.encode("utf-8")))
        print(f"[report] wrote {out_path}", flush=True)
    else:
        # Reached only when a caller passes no path: main() always supplies a default, so a CLI run
        # never lands here. The number then exists only in this shell, and a floor nobody can reopen
        # cannot be diffed against later.
        _LOG.warning("no --out given: this baseline was NOT persisted, only printed")
    return report_dict


def main() -> None:
    ap = argparse.ArgumentParser(description="Iteration-1 dense-clutter PILE baseline (GraspingRevisit Opp 0).")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--perception", choices=("vision", "gt"), default="vision",
                    help="vision = end-to-end FLOOR (GDINO+SAM2); gt = perception CEILING (Isaac GT masks).")
    ap.add_argument("--pile-count", type=int, default=8)
    ap.add_argument("--pile-spread-mm", type=float, default=50.0, help="pile-density dial (smaller = tighter).")
    ap.add_argument("--pile-shapes", choices=("mixed", "cube", "cylinder"), default="mixed",
                    help="'cube' = the jaw-friendly ANCHOR scene (cylinders roll -> a separate H4.2 problem).")
    ap.add_argument("--no-bin", action="store_true", help="disable the shallow tray walls (open-table anchor).")
    ap.add_argument("--curobo-bin-world", action="store_true",
                    help="register the tray walls in cuRobo's world so it plans AROUND them (bin correctness).")
    ap.add_argument("--g12", action="store_true", help="It.2a: swept-approach validation (reject colliding approaches).")
    ap.add_argument("--g4", action="store_true", help="It.2a: per-candidate corridor-risk producer.")
    ap.add_argument("--g4-rerank", action="store_true", help="It.2a: G4 corridor rerank (demote blocked corridors).")
    ap.add_argument("--g4-rerank-weight", type=float, default=0.3)
    ap.add_argument("--oblique", action="store_true",
                    help="It.3a: generate grasps at multiple approach angles (top-down + tilts) -> gap-threading.")
    ap.add_argument("--oblique-tilt-deg", type=float, default=30.0)
    ap.add_argument("--oblique-azimuths", type=int, default=4)
    ap.add_argument("--multiview", action="store_true",
                    help="It.3b: 2 fixed oblique cameras -> fused BASE target cloud (side faces) into the generator.")
    ap.add_argument("--adaptive-close", action="store_true",
                    help="opt-in adaptive close (grip_w - squeeze) for varied sizes; default = fixed close_width.")
    ap.add_argument("--top-ref", action="store_true",
                    help="opt-in top-referenced grasp depth for tall objects; default = centre-ref (run_dense_pick).")
    ap.add_argument("--bin-half-width-mm", type=float, default=110.0)
    ap.add_argument("--bin-height-mm", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter-mm", type=float, default=25.0)
    ap.add_argument("--settle-steps", type=int, default=150)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--gui", action="store_true", help="run with the Isaac GUI (default headless).")
    ap.add_argument("--out", type=str, default=None, help="write the JSON baseline report here.")
    args = ap.parse_args()
    default_out = args.out or f"logs/baselines/pile_{args.perception}.json"
    run_baseline(
        runs=args.runs, perception=args.perception,
        headless=not args.gui, data_dir=args.data_dir,
        pile_count=args.pile_count, pile_spread_mm=args.pile_spread_mm, pile_shapes=args.pile_shapes,
        enable_bin=not args.no_bin, curobo_bin_world=args.curobo_bin_world,
        adaptive_close=args.adaptive_close, top_ref=args.top_ref,
        enable_g12=args.g12, enable_g4=args.g4, g4_rerank=args.g4_rerank, g4_rerank_weight=args.g4_rerank_weight,
        oblique=args.oblique, oblique_tilt_deg=args.oblique_tilt_deg, oblique_azimuths=args.oblique_azimuths,
        multiview=args.multiview,
        bin_half_width_mm=args.bin_half_width_mm, bin_height_mm=args.bin_height_mm,
        seed=args.seed, jitter_mm=args.jitter_mm, settle_steps=args.settle_steps, out=default_out,
    )


if __name__ == "__main__":
    main()
