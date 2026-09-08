"""Dual-cam fused pick (Isaac): overhead coarse scan, then wrist eye-in-hand refine and grasp.

A two-phase coordinator that reuses the existing machinery, with no new fuser or resolver:
  1. Coarse: the fixed overhead (eye-to-hand) camera localizes the object. Perceive on
     ``handles.camera`` and back-project the object-mask centroid through the overhead's CAMERA->BASE
     transform to a BASE-frame point (``coarse_centroid_base``).
  2. Handoff: that BASE centroid becomes the target for the closed-form wrist view pose
     (``topdown_view_pose``). The overhead localizes, the wrist aims, and this handoff is the fusion.
  3. Refine and grasp: the existing wrist eye-in-hand pick service (``build_service``, the calibrated
     ``EyeInHandFrameResolver``, the wrist GroundTruthPerceptionSource, ``service.pick()``).

This sidesteps two structural blockers: the orchestrator binds one resolver, and ``SceneFusion``
refuses a second camera's intrinsics. The overhead uses ground-truth perception, which is
deterministic; swapping it for the GroundingDINO+SAM2 stack is a later step.

On-box only (needs Isaac). Run with Isaac's bundled python:
    <isaac-sim>\\python.bat -m src.willy_sim.run_fused_pick --runs 10 [--gui]
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np

from src.robot.safety.planning.world import planner_cuboid

from src.config.schema.robot import SimObjectConfig
from src.willy_sim.harness.gate import GateResult, gate_passed
from src.willy_sim.perception import GroundTruthPerceptionSource
from src.willy_sim.run_eih_pick import (
    _load_calibrated_eih,
    build_service,
    coarse_centroid_base,
    topdown_view_pose,
)
from src.willy_sim.scene import OBJECT_PRIM

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.types.perception import PerceptionSource

# Clutter scene: 3 distinct-coloured cubes, well separated in Y, all reachable around (450, 0).
# Distinct colours and labels are the vision-language disambiguation handle; same-class
# disambiguation is a later step.
CLUTTER_OBJECTS_P0 = [
    ("red cube", (0.90, 0.12, 0.12), (450.0, -95.0, 25.0)),
    ("green cube", (0.10, 0.65, 0.18), (450.0, 5.0, 25.0)),
    ("blue cube", (0.12, 0.22, 0.90), (450.0, 105.0, 25.0)),
]


def _clutter_specs() -> list:
    return [SimObjectConfig(name=n, color=c, position_mm=p) for (n, c, p) in CLUTTER_OBJECTS_P0]


def _select_by_prompt(objects: list, prompt: str) -> int:
    """Pick the object whose label words best overlap the text prompt: the prompt-to-target matcher.

    Maps the prompt to exactly one target object. Distinct labels keep this unambiguous; same-class
    disambiguation is out of scope here.
    """
    p = prompt.lower()
    scores = [sum(1 for w in obj.name.lower().split() if w in p) for obj in objects]
    return int(np.argmax(scores))


def run_fused_gate(runs: int = 10, *, headless: bool = True, data_dir: str | None = None,
                   marker: str = "ground_truth", coarse: str = "gt", prompt: str = "a red cube",
                   view_height_mm: float = 275.0) -> GateResult:
    """Overhead coarse-localize, then wrist refine and grasp, scored on the lift gate and coarse error.

    ``coarse='gt'`` localizes with ground-truth perception, which is deterministic; ``coarse='vision'``
    uses the real GroundingDINO+SAM2 stack with the text ``prompt`` for the overhead scan, which is
    the full vision-language dual-cam pick.
    """
    print("=== DUAL-CAM FUSED pick (overhead ETH coarse-scan -> wrist EIH refine -> grasp) ===", flush=True)
    service, arm, gripper, handles, cfg, _view_pose, _cell = build_service(
        headless=headless, data_dir=data_dir, marker=marker,
        continuous_guard=True,  # eye-in-hand closed-form path; natural_aim does not apply here
    )
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    # Give the planner this cell's world before asking it to plan in it. With planned joint moves on
    # and no world set, cuRobo still holds its boot world, a table plus placeholder cuboids reserved
    # for `set_world` to fill, and refuses every target against it, park pose and home alike. That is
    # a stale world, not a planner agreeing with the continuous guard.
    #
    # The low floor is the same obstacle set run_bin_clearing_demo uses: a 1 m slab whose top sits at
    # z = -50 mm keeps the arm from driving under the table without intruding into the grasp column.
    # Walls are deliberately omitted, because they make cuRobo refuse low top-down reaches.
    _floor = planner_cuboid("table_floor", (0.0, 0.0, -550.0), (3000.0, 3000.0, 1000.0))
    n_world = int(arm.set_curobo_world([_floor]))
    print(f"[curobo-world] registered {n_world} obstacle(s) (low floor top=-50mm)", flush=True)

    # Now planning is meaningful, so opt in. This is the only pick gate that asks for the continuous
    # guard, and with a mesh backend (Coal, in ext_deps) the guard refuses the joint-space reset move:
    # forearm and lfinger come closer than the 8 mm margin. A joint-space straight line is not
    # collision-aware; a planned path is.
    arm._plan_joint_moves = True  # noqa: SLF001 (the runner-sets-it idiom already used for natural_aim)

    sim = cfg.robot.sim
    gate = sim.scene_setup.gate

    # The fused pick consumes both calibrations: the wrist EIH transform (inside build_service's
    # resolver + here for the view-pose p_cam_tool) and the overhead ETH camera_to_base (handles).
    from src.willy_sim.calibration.paths import calibration_dir

    cal = _load_calibrated_eih(marker, cal_dir=calibration_dir(sim.robot_model))
    if cal is None:
        raise SystemExit(f"fused pick requires a calibrated EIH artifact "
                         f"(logs/calibration/{sim.robot_model}/eih_wrist_cam_{marker}.json, quality good/excellent)")
    p_cam_tool = np.asarray(cal.to_matrix(), dtype=np.float64)[:3, 3]

    # Overhead (eye-to-hand) coarse perception. The ground-truth path keeps near clip 1.0, which hides
    # the home-pose arm, so it scans from home; the real-vision stack sets near clip 0.05, sees the
    # arm, and so scans from the park pose. Either way the wrist view pose is approached via home,
    # because the extreme -X park does not swing cleanly to it.
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    overhead: PerceptionSource
    if coarse == "vision":
        # Default to HF offline before transformers imports huggingface_hub, which bakes
        # HF_HUB_OFFLINE at module load. Otherwise the detector pings the hub at boot, Isaac's Kit has
        # already closed the httpx client, and the boot fails with "client has been closed". The five
        # other vision runners do the same. Override with HF_HUB_OFFLINE=0 to force online.
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from src.willy_sim.perception import IsaacVisionPerceptionSource
        from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector
        from src.models.segmentation.realtime.segmenter import Sam2Segmenter

        overhead = IsaacVisionPerceptionSource(
            camera=handles.camera, detector=GroundingDinoObjectDetector(cfg.models.objectdetector),
            segmenter=Sam2Segmenter(cfg.models.segmenter), prompt=prompt, session=arm.session,
            warmup_steps=max(25, sim.scene_setup.render_warmup_steps),
        )
        scan_q = np.asarray(sim.park_joint_positions, dtype=np.float64)
        print(f"overhead COARSE = real-vision (GroundingDINO+SAM2), prompt={prompt!r}", flush=True)
    else:
        overhead = GroundTruthPerceptionSource(
            camera=handles.camera, target_prim_path=handles.object_prim_path, session=arm.session,
            warmup_steps=sim.scene_setup.render_warmup_steps, ground_truth_depth=True, grasp_lift_mm=0.0,
        )
        scan_q = home_q
        print("overhead COARSE = ground-truth perception", flush=True)
    obj = SingleRigidPrim(handles.object_prim_path)
    home_pos, home_quat = obj.get_world_pose()
    results = []
    for i in range(runs):
        # Reset: release jaw, move to the scan pose (home for GT / park for vision), re-place + settle.
        gripper.open()
        arm.move_joint(JointPositions(scan_q))
        obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
        arm.session.step_n(30)

        # Phase 1: overhead coarse localize.
        frame = overhead.acquire()
        coarse_c = coarse_centroid_base(frame, handles.camera_to_base)
        true_c = np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0
        if coarse_c is None:
            print(f"RUN {i}: overhead coarse FOUND NO object -> skip", flush=True)
            results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False, "coarse_err_mm": None})
            continue
        coarse_err = float(np.linalg.norm((coarse_c - true_c)[:2]))
        print(f"RUN {i}: overhead localized XY={np.round(coarse_c[:2], 1)} mm "
              f"(true {np.round(true_c[:2], 1)}, coarse XY err {coarse_err:.1f} mm)", flush=True)

        # Phases 2 and 3: handoff to the wrist view pose, then the existing EIH refine and grasp.
        if coarse == "vision":
            arm.move_joint(JointPositions(home_q))  # via home: reachable approach to the view pose
        arm.move(topdown_view_pose(coarse_c, p_cam_tool, view_height_mm))
        z0 = float(np.asarray(obj.get_world_pose()[0])[2])
        report = service.pick()
        z1 = float(np.asarray(obj.get_world_pose()[0])[2])
        outcome = getattr(report, "outcome", None)
        succeeded = getattr(outcome, "value", outcome) == "succeeded"
        lift_mm = (z1 - z0) * 1000.0
        passed = bool(lift_mm >= gate.lift_threshold_mm)
        results.append({"run": i, "succeeded": succeeded, "lift_mm": round(lift_mm, 1),
                        "passed": passed, "coarse_err_mm": round(coarse_err, 1)})
        print(f"RUN {i}: succeeded={succeeded} lift_mm={lift_mm:.1f} passed(lifted)={passed}", flush=True)

    n_pass = sum(1 for r in results if r["passed"])
    gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
    errs = [r["coarse_err_mm"] for r in results if r["coarse_err_mm"] is not None]
    coarse_med = float(np.median(errs)) if errs else float("nan")
    print(f"FUSED PICK GATE: {n_pass}/{runs} passed -> gate_passed={gate_ok} "
          f"(overhead coarse XY err median {coarse_med:.1f} mm)", flush=True)
    return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok)


def run_clutter_gate(runs: int = 10, *, headless: bool = True, data_dir: str | None = None,
                     marker: str = "ground_truth", prompt: str = "the green cube",
                     coarse: str = "gt", view_height_mm: float = 275.0,
                     overhead_source: str = "calibrated") -> GateResult:
    """Multi-object pick: select the prompted object among 3 distinct cubes, grasp it, never a distractor.

    Reuses the dual-cam fused flow, overhead coarse-localize then wrist eye-in-hand refine, in a
    clutter scene. Ground-truth coarse perception carries no perception risk: the prompt-to-target
    match picks the chosen object, and the wrist ground-truth perception is bound to exactly that
    prim, which is the correctness invariant, so the grasp stack can only grasp the prompted object.
    Scored on the prompted lift and a 'no distractor lifted' check.
    """
    objects = _clutter_specs()
    chosen_idx = _select_by_prompt(objects, prompt)
    chosen = objects[chosen_idx]
    chosen_prim = f"{OBJECT_PRIM}_{chosen_idx}"
    print(f"=== CLUTTER pick: prompt={prompt!r} -> chosen '{chosen.name}' ({chosen_prim}) "
          f"among {len(objects)} objects (coarse={coarse}) ===", flush=True)

    service, arm, gripper, handles, cfg, _vp, _cell = build_service(
        headless=headless, data_dir=data_dir, marker=marker,
        objects_override=objects, wrist_target_prim=chosen_prim,
        continuous_guard=True,  # eye-in-hand closed-form path; natural_aim does not apply here
    )
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    sim = cfg.robot.sim
    gate = sim.scene_setup.gate
    cal = _load_calibrated_eih(marker)
    if cal is None:
        raise SystemExit(f"clutter pick requires a calibrated EIH artifact (eih_wrist_cam_{marker}.json)")
    p_cam_tool = np.asarray(cal.to_matrix(), dtype=np.float64)[:3, 3]

    # Coarse localize from the persisted calibration (eth_overhead_<marker>.json), the real hand-eye
    # result from run_eth_calibrate, not a live ground-truth Umeyama fit. marker='aruco' is the
    # vision-only ArUco calib: marginal on its own, but the wrist eye-in-hand refines the few-mm
    # overhead error into a fully real-calibrated dual-cam pick. Falls back to the live fit if there is
    # no artifact, or by request. Never uses handles.camera_to_base, the hand-built M1 transform,
    # which is 180 deg in-plane from true and so flips off-centre positions about the camera centre;
    # the ArUco and ground-truth calibs are convention-reconciled to true. Needs the depth annotator
    # and near clip 0.05.
    from src.willy_sim.run_eih_pick import _load_calibrated_eth
    from src.willy_sim.scene import camera_to_base_ground_truth

    overhead_cam = handles.camera
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    park_q = np.asarray(sim.park_joint_positions, dtype=np.float64)
    for fn in (lambda: overhead_cam.add_distance_to_image_plane_to_frame(),
               lambda: overhead_cam.set_clipping_range(0.05, 1.0e6)):
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass
    arm.move_joint(JointPositions(park_q))  # clean overhead view for the calib load / live fit + coarse scans
    app = getattr(arm.session, "app", None)
    for _ in range(sim.scene_setup.render_warmup_steps):  # warm the overhead RGB+depth (perception needs it)
        arm.session.step(render=True)
        if app is not None:
            app.update()
    from src.willy_sim.calibration.paths import calibration_dir

    t_overhead_true = (
        _load_calibrated_eth(marker, cal_dir=calibration_dir(sim.robot_model))
        if overhead_source == "calibrated" else None
    )
    src_kind = "CALIBRATED" if t_overhead_true is not None else "LIVE GT-fit"
    if t_overhead_true is None:  # live ground-truth Umeyama fit fallback (the M1-style oracle)
        for attempt in range(6):
            for _ in range(12):
                arm.session.step(render=True)
                if app is not None:
                    app.update()
            try:
                t_overhead_true, _ = camera_to_base_ground_truth(overhead_cam)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"overhead live-extrinsic retry {attempt + 1}/5 ({exc})", flush=True)
    if t_overhead_true is None:  # the calibrated load or live-fit loop sets it (loop re-raises on failure)
        raise RuntimeError("overhead extrinsic could not be resolved (no calibration + live fit failed).")
    print(f"overhead {src_kind} extrinsic T_cam_to_BASE mm: "
          f"{np.round(np.asarray(t_overhead_true.to_matrix())[:3, 3], 1)}", flush=True)

    # Overhead localize: a per-run _localize() returning the chosen object's BASE centroid through the
    # true extrinsic. With coarse='gt' this is ground-truth segmentation of the chosen prim. With
    # coarse='vision', GroundingDINO detect_all finds all the cubes, an RGB colour match selects the
    # prompted one (GroundingDINO does not colour-discriminate by label; the mean RGB per box does),
    # and SAM2 then segments it.
    if coarse == "vision":
        from types import SimpleNamespace

        from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector
        from src.models.segmentation.realtime.segmenter import Sam2Segmenter

        detector = GroundingDinoObjectDetector(cfg.models.objectdetector)
        segmenter = Sam2Segmenter(cfg.models.segmenter)
        color_idx = {"red": 0, "green": 1, "blue": 2}.get(
            next((w for w in prompt.lower().split() if w in ("red", "green", "blue")), "")
        )
        K = np.asarray(overhead_cam.get_intrinsics_matrix(), dtype=np.float64)

        def _localize():
            for _ in range(8):
                arm.session.step(render=True)
                if app is not None:
                    app.update()
            fr = overhead_cam.get_current_frame()
            rgb = np.asarray(fr["rgb"])[..., :3] if isinstance(fr, dict) and fr.get("rgb") is not None else None
            if rgb is None:
                return None
            bgr = rgb[..., ::-1].copy()
            depth_mm = np.where(np.isfinite(np.asarray(overhead_cam.get_depth(), dtype=np.float64)),
                                np.asarray(overhead_cam.get_depth(), dtype=np.float64) * 1000.0, 0.0)
            dets = detector.detect_all(bgr, prompt)
            if not dets:
                return None

            def _mean_rgb(d):
                x0, y0, x1, y1 = (int(max(0, c)) for c in d.box)
                patch = rgb[y0:y1, x0:x1].reshape(-1, 3)
                return patch.mean(axis=0) if patch.size else np.zeros(3)

            if color_idx is not None:
                matches = [d for d in dets if int(np.argmax(_mean_rgb(d))) == color_idx]
                chosen_det = max(matches or dets, key=lambda d: d.score)
            else:
                chosen_det = max(dets, key=lambda d: d.score)
            seg = segmenter.segment_detection(bgr, chosen_det)
            frame_like = SimpleNamespace(segmentations=[seg], depth_map=depth_mm, intrinsics=K)
            return coarse_centroid_base(frame_like, t_overhead_true)
    else:
        overhead = GroundTruthPerceptionSource(
            camera=overhead_cam, target_prim_path=chosen_prim, session=arm.session,
            warmup_steps=sim.scene_setup.render_warmup_steps, ground_truth_depth=True, grasp_lift_mm=0.0,
        )

        def _localize():
            return coarse_centroid_base(overhead.acquire(), t_overhead_true)

    prims = [SingleRigidPrim(spec[0]) for spec in handles.object_specs]
    # Store home pose (position + orientation) per object so each run resets to an upright cube. A slip
    # otherwise tilts a cube, the next grasp on the tilted cube slips too, and the error compounds
    # across runs.
    homes = [(np.asarray(p.get_world_pose()[0], dtype=np.float64),
              np.asarray(p.get_world_pose()[1], dtype=np.float64)) for p in prims]

    results: list[dict[str, float]] = []  # bool/int values widen to float; keeps the mixed dict typed
    for i in range(runs):
        gripper.open()
        arm.move_joint(JointPositions(park_q))  # park out of the overhead view for a clean coarse scan
        for p, (hp, hq) in zip(prims, homes):
            p.set_world_pose(position=hp, orientation=hq)
            try:
                p.set_linear_velocity(np.zeros(3))
                p.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
        arm.session.step_n(30)

        coarse_c = _localize()
        true_chosen = np.asarray(prims[chosen_idx].get_world_pose()[0], dtype=np.float64) * 1000.0
        if coarse_c is None:
            print(f"RUN {i}: overhead coarse FOUND NO '{chosen.name}' -> skip", flush=True)
            results.append({"run": i, "passed": False, "lifted_distractor": False})
            continue
        coarse_err = float(np.linalg.norm((coarse_c - true_chosen)[:2]))
        z0 = float(np.asarray(prims[chosen_idx].get_world_pose()[0])[2])

        arm.move_joint(JointPositions(home_q))  # via home: reachable approach to the wrist view pose
        arm.move(topdown_view_pose(coarse_c, p_cam_tool, view_height_mm))
        report = service.pick()
        outcome = getattr(getattr(report, "outcome", None), "value", getattr(report, "outcome", None))

        z1 = float(np.asarray(prims[chosen_idx].get_world_pose()[0])[2])
        lift_mm = (z1 - z0) * 1000.0
        passed = bool(lift_mm >= gate.lift_threshold_mm)
        # Distractor check: did any non-chosen object rise by >= the lift threshold, i.e. get grasped?
        distractor_lifts = [
            (float(np.asarray(prims[j].get_world_pose()[0])[2]) - float(homes[j][0][2])) * 1000.0
            for j in range(len(prims)) if j != chosen_idx
        ]
        max_distractor = max(distractor_lifts) if distractor_lifts else 0.0
        lifted_distractor = bool(max_distractor >= gate.lift_threshold_mm)
        run_ok = passed and not lifted_distractor
        results.append({"run": i, "passed": run_ok, "lift_mm": round(lift_mm, 1),
                        "max_distractor_mm": round(max_distractor, 1), "lifted_distractor": lifted_distractor})
        print(f"RUN {i}: coarse err {coarse_err:.1f} mm | outcome={outcome} | chosen lift={lift_mm:.1f} mm "
              f"(passed={passed}) | max distractor rise={max_distractor:.1f} mm -> ok={run_ok}", flush=True)

    n_pass = sum(1 for r in results if r["passed"])
    n_distractor = sum(1 for r in results if r.get("lifted_distractor", False))
    gate_ok = gate_passed(n_pass, runs, gate.pass_fraction) and n_distractor == 0
    print(f"CLUTTER PICK GATE: {n_pass}/{runs} grasped '{chosen.name}' | "
          f"{n_distractor} distractor(s) wrongly lifted -> gate_passed={gate_ok}", flush=True)
    return GateResult(
        runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, distractor_lifts=n_distractor,
    )


def run_vision_probe(*, headless: bool = True, data_dir: str | None = None,
                     prompt: str = "a cube") -> None:
    """Measure what GroundingDINO detects on the 3-cube clutter scene.

    Prints every detection (box, label, score) and the mean RGB inside each box, which shows whether
    GroundingDINO finds all the cubes and whether colour can be read per detection. An RGB colour
    match is robust to GroundingDINO's weak colour grounding.
    """
    objects = _clutter_specs()
    service, arm, gripper, handles, cfg, _vp, _cell = build_service(  # noqa: F841
        headless=headless, data_dir=data_dir, objects_override=objects,
        continuous_guard=True,  # eye-in-hand closed-form path; natural_aim does not apply here
    )
    from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector

    from src.robot.core import JointPositions

    sim = cfg.robot.sim
    overhead_cam = handles.camera
    for fn in (lambda: overhead_cam.set_clipping_range(0.05, 1.0e6),):
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass
    arm.move_joint(JointPositions(np.asarray(sim.park_joint_positions, dtype=np.float64)))
    app = getattr(arm.session, "app", None)
    for _ in range(max(30, sim.scene_setup.render_warmup_steps)):
        arm.session.step(render=True)
        if app is not None:
            app.update()
    frame = overhead_cam.get_current_frame()
    rgb = np.asarray(frame["rgb"])[..., :3] if isinstance(frame, dict) and frame.get("rgb") is not None else None
    if rgb is None:
        raise SystemExit("overhead RGB not available")
    bgr = rgb[..., ::-1].copy()
    detector = GroundingDinoObjectDetector(cfg.models.objectdetector)
    for probe_prompt in (prompt, "a cube", "the green cube", "red cube . green cube . blue cube"):
        dets = detector.detect_all(bgr, probe_prompt)
        print(f"\n--- detect_all(prompt={probe_prompt!r}) -> {len(dets)} detections ---", flush=True)
        for d in dets:
            x0, y0, x1, y1 = (int(max(0, c)) for c in d.box)
            patch = rgb[y0:y1, x0:x1]
            mean_rgb = np.round(patch.reshape(-1, 3).mean(axis=0), 0) if patch.size else np.array([-1, -1, -1])
            print(f"  label={d.label!r} score={d.score:.2f} center=({d.x_center:.0f},{d.y_center:.0f}) "
                  f"mean_rgb={mean_rgb.tolist()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy dual-cam fused pick (Isaac).")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-headless", action="store_true", help="alias for --gui")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--marker", type=str, default="ground_truth", choices=["ground_truth", "aruco"])
    ap.add_argument("--coarse", type=str, default="gt", choices=["gt", "vision"],
                    help="overhead coarse-localize perception: gt (deterministic) or vision (GroundingDINO+SAM2)")
    ap.add_argument("--prompt", type=str, default="a red cube", help="text prompt (clutter selection / coarse vision)")
    ap.add_argument("--clutter", action="store_true",
                    help="multi-object scene: select + grasp the prompted object among 3 distinct cubes (P0)")
    ap.add_argument("--probe-vision", action="store_true",
                    help="measure GroundingDINO detect_all on the 3-cube scene (P1 design input), then exit")
    ap.add_argument("--overhead-source", type=str, default="calibrated", choices=["calibrated", "live"],
                    help="P5.4: overhead extrinsic from the persisted calibration (eth_overhead_<marker>.json) "
                         "or a live ground-truth Umeyama fit (the GT oracle)")
    args = ap.parse_args()
    if args.probe_vision:
        run_vision_probe(headless=not (args.gui or args.no_headless), data_dir=args.data_dir, prompt=args.prompt)
    elif args.clutter:
        run_clutter_gate(runs=args.runs, headless=not (args.gui or args.no_headless),
                         data_dir=args.data_dir, marker=args.marker, prompt=args.prompt, coarse=args.coarse,
                         overhead_source=args.overhead_source)
    else:
        run_fused_gate(runs=args.runs, headless=not (args.gui or args.no_headless),
                       data_dir=args.data_dir, marker=args.marker, coarse=args.coarse, prompt=args.prompt)


if __name__ == "__main__":
    main()
