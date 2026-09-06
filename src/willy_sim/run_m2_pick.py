"""M2 real-vision pick runner: perception in the loop (config-driven).

Boots the combined UR5e+2F-85 scene from the sim config tree, wires the Isaac arm, gripper and
``IsaacVisionPerceptionSource`` (overhead RGB, then GroundingDINO, then SAM2) into
``AutonomousGraspService.from_components``, parks the arm out of the overhead camera's view, and
drives ``pick()`` from a text prompt. Detector and segmenter model ids come from ``cfg.models.*``;
the park pose and gate thresholds from ``cfg.robot.sim``.

On-box only (needs Isaac + the GroundingDINO/SAM2 weights). Run with Isaac's bundled python:

    <isaac-sim>\\python.bat -m src.willy_sim.run_m2_pick --runs 10 --prompt "a red cube"
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.harness.cli import add_cell_arguments, cell_profile_kwargs
from src.willy_sim.harness.gate import (
    GateResult,
    lift_mm_since,
    reset_object_to_home_z0,
)
from src.willy_sim.config import require_robot
from src.willy_sim.harness.instrumentation import (
    cell_identity,
    write_pick_artifacts,
    write_run_result,
)
from src.willy_sim.harness.modes import mode_service_kwargs, resolve_demo_mode


def build_service(
    *,
    prompt: str = "a red cube",
    headless: bool = True,
    data_dir: str | None = None,
    mode: str = "easy",   # "easy" | "auto" | "closed_loop" (see willy_sim/harness/modes.py)
    # Pick-policy tunings: not paths, so they stay as defaults rather than config.
    pre_open_width_mm: float = 80.0,
    close_width_mm: float = 25.0,
    standoff_mm: float = 150.0,   # high pre-grasp, so the approach out of park comes from above
    retreat_mm: float = 100.0,
    natural_aim: bool = False,           # IK prefers the natural (non-self-colliding) branch
    continuous_guard: bool = False,      # per-waypoint collision-avoidance guard (needs WILLY_COAL_PREFIX)
    continuous_guard_margin_mm: float = 8.0,
    cell_kwargs: Mapping[str, Any] | None = None,  # which cell to boot (--robot-model / --profile)
    radial_closing: bool = False,  # spend a symmetric object's free yaw on the reachable axis
    detector_model_id: str | None = None,  # override the grounding model, for A/B measurement only
):
    """Boot the scene + wire the full real-vision pick service from config."""
    # Vision runners load cached GroundingDINO/SAM2; default to HF offline so transformers does not ping
    # the hub at boot. Isaac's Kit closes the httpx client, so a ping fails with "client has been closed"
    # and the boot crashes. Export HF_HUB_OFFLINE=0 before the run to force online.
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # The byte-identical boot prefix (fail-closed arm + session + scene + gripper) is shared.
    cell = bootstrap_sim_cell(data_dir, headless=headless, **(cell_kwargs or {}))
    arm, gripper, handles, cfg, sim = cell.arm, cell.gripper, cell.handles, cell.cfg, cell.sim
    _dwell = cell.dwell  # the steady-state dwell gate
    from src.willy_sim.run_dense_pick import wire_safety_guards  # opt-in guards, default-off
    wire_safety_guards(arm, natural_aim=natural_aim, continuous_guard=continuous_guard,
                       continuous_guard_margin_mm=continuous_guard_margin_mm, fixtures=())

    # Heavy vision stack imported + constructed after the scene/camera exist (model ids from config).
    from src.willy_sim.perception import IsaacVisionPerceptionSource
    from src.models.factory import build_perception
    from src.robot.execution.autonomous_grasp import AutonomousGraspService
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
    # Build through the factory, never by name. Naming `GraspCalculator` here makes
    # `robot.grasping.calculator` inert on the Isaac box: a cell that asks for `deep` silently gets
    # the analytic stack, and every on-box pick rate then grades the analytic generator by
    # construction. The selector defaults to `geometric`, and that branch is
    # `GraspCalculator(**kwargs)` verbatim.
    from src.robot.grasping.calculator_factory import build_calculator

    # Building the perception backend from the factory replaces "construct a detector, construct a
    # segmenter", and makes the whole models.pipeline block reachable from this gate: the VLM route
    # and the prompt router are a config edit away rather than unreachable.
    #
    # This is not a pure refactor. A backend runs `detect_all` and this source takes the first
    # detection, which can differ from the single best match `detect` returns, so a change here is
    # accepted on the M2 gate itself, never on inspection.
    perception = IsaacVisionPerceptionSource(
        camera=handles.camera, backend=build_perception(_models_for(cfg, detector_model_id)),
        prompt=prompt,
        session=arm.session, warmup_steps=max(25, sim.scene_setup.render_warmup_steps),
    )
    # The cell declares whether this arm needs a radial close (robot.ur3e.yaml declares it for the
    # UR3e); --radial-closing still forces it on for a cell that has not declared it.
    _radial = bool(radial_closing or getattr(cell.robot.grasping, "isotropic_radial_closing", False))
    # Analytic-only lever, and the banner says so. The learned generator decodes its own closing
    # axis, so a bare `True` under `calculator: deep` would describe the other path.
    print(f"[grasp] isotropic_radial_closing={_radial} "
          f"(cell={getattr(cell.robot.grasping, 'isotropic_radial_closing', False)}, "
          f"flag={bool(radial_closing)}, analytic generator only)",
          flush=True)
    calculator = build_calculator(
        cell.robot,
        camera_matrix=np.asarray(handles.camera.get_intrinsics_matrix(), dtype=np.float64),
        max_grip_width_mm=cell.robot.gripper.max_width_mm,
        min_grip_width_mm=cell.robot.gripper.min_width_mm,
        isotropic_radial_closing=_radial,
    )
    resolver = StaticCameraToBaseResolver(transform=handles.camera_to_base)
    policy = GraspExecutionPolicy(
        arm=arm, gripper=gripper, standoff_mm=standoff_mm, retreat_mm=retreat_mm,
        pre_open_width_mm=pre_open_width_mm, close_width_mm=close_width_mm,
        require_base_frame_grasp=True,
        # Gate every commanded move on the dwell steady-state config (the sim arm settles fast).
        require_steady_before_motion=(_dwell.require_steady_before_motion if _dwell else False),
        steady_timeout_s=(_dwell.steady_timeout_s if _dwell else 5.0),
    )
    # One mode knob via willy_sim/harness/modes.py. m2 uses a fixed overhead camera, so the two
    # closed_loop refine scans share the same view and the image-space IoU tracker can match
    # (target_match_iou_threshold 0.1). fail_closed=False keeps an inconclusive width verification
    # from failing an otherwise-good pick.
    grasp_mode = resolve_demo_mode(mode)
    service = AutonomousGraspService.from_components(
        arm=arm, calculator=calculator, perception=perception, gripper=gripper,
        mode=grasp_mode, frame_resolver=resolver, policy=policy, max_attempts=2,
        **mode_service_kwargs(grasp_mode, refinement_kwargs={"target_match_iou_threshold": 0.1}),
    )
    return service, arm, gripper, handles, cfg, cell


def _models_for(cfg: Any, detector_model_id: str | None) -> Any:
    """The models config, optionally with the grounding model swapped, for A/B measurement.

    ``None`` returns the config untouched, so the gate's path is byte-identical. A value swaps only
    ``objectdetector.model_id``, the same one-field override ``run_dense_pick`` uses for its larger
    detector.

    It stays a flag rather than a config default: which detector is better on the gated scene is a
    question this runner measures, not a settled answer.
    """
    if not detector_model_id:
        return cfg.models
    swapped = cfg.models.objectdetector.model_copy(update={"model_id": detector_model_id})
    return cfg.models.model_copy(update={"objectdetector": swapped})


def run_gate(runs: int = 10, *, prompt: str = "a red cube", headless: bool = True,
             data_dir: str | None = None, mode: str = "easy",
             record_log: str | None = None, debug_frames: str | None = None,
             cell_kwargs: Mapping[str, Any] | None = None,
             standoff_mm: float | None = None,
             radial_closing: bool = False,
             detector_model_id: str | None = None) -> GateResult:
    """Run ``runs`` real-vision picks (re-place object, park arm, perceive, grasp) and score lift.

    ``mode`` is "easy", "auto" (a real ``DecisionEngine``) or "closed_loop", where the fixed overhead
    camera lets the two-scan refine IoU-match. closed_loop also prints the per-run refinement
    telemetry, so a reader can see whether the refine changes the grasp and still holds the gate.

    Both instrumentation hooks are opt-in and default-off: ``record_log`` appends one
    ``GraspAttemptRecord`` JSONL line per pick, stamped with the ground-truth ``sim_lift_mm`` and
    ``sim_lifted``, and ``debug_frames`` dumps the grasp-point overlay PNG per pick (the real-vision
    frame: SAM2 mask plus grasp). With neither set the gate is byte-identical.
    """
    cl = str(mode).lower().replace("-", "_") == "closed_loop"
    service, arm, gripper, handles, cfg, cell = build_service(
        prompt=prompt, headless=headless, data_dir=data_dir, mode=mode, cell_kwargs=cell_kwargs,
        # None keeps build_service's 150 mm default, so the call is byte-identical.
        standoff_mm=150.0 if standoff_mm is None else float(standoff_mm),
        radial_closing=radial_closing,
        detector_model_id=detector_model_id,
    )
    if debug_frames:  # render the grasp-point overlay per pick (M2 has real overhead rgb + SAM2 mask)
        service.enable_debug_image_rendering()
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    robot = require_robot(cfg)
    gate = robot.sim.scene_setup.gate
    park_q = np.asarray(robot.sim.park_joint_positions, dtype=np.float64)
    obj = SingleRigidPrim(handles.object_prim_path)
    home_pos, home_quat = obj.get_world_pose()
    run_id = f"m2-{int(time.time())}"
    results = []
    for i in range(runs):
        # Clean reset: release jaw, park the arm clear of the camera, re-place the object at rest.
        gripper.open()
        # A typed joint move: it gates the park destination through the preflight (joint limit,
        # self-collision, payload, static IK quality) and resets the continuity reference around this
        # deliberate discontinuity, so the next grasp move() is not diffed across the park.
        arm.move_to_joints(JointPositions(park_q))
        z0 = reset_object_to_home_z0(obj, home_pos, home_quat, arm.session, settle_steps=25)
        report = service.pick()
        if cl:
            tele = getattr(report, "telemetry", None) or {}
            print(f"  [closed_loop] outcome={getattr(report, 'outcome', None)} "
                  f"refinement_mode={tele.get('refinement_mode')} "
                  f"refinement_telemetry={tele.get('refinement_telemetry')}", flush=True)
        lift_mm = lift_mm_since(obj, z0)
        outcome = getattr(report, "outcome", None)
        succeeded = getattr(outcome, "value", outcome) == "succeeded"
        # M2 scores on the lift: the retreat move occasionally reports execution_failed even when the
        # object is lifted (matters for placing, not picking). `succeeded` is still printed.
        passed = bool(lift_mm >= gate.lift_threshold_mm)
        results.append({"run": i, "succeeded": succeeded, "lift_mm": round(lift_mm, 1), "passed": passed})
        print(f"RUN {i}: succeeded={succeeded} lift_mm={lift_mm:.1f} passed(lifted)={passed}", flush=True)
        write_pick_artifacts(
            report, attempt_id=f"{run_id}-{i:04d}",
            record_log=record_log, debug_dir=debug_frames,
            debug_png=service.last_debug_image_png,
            extra={**cell_identity(cell), "sim_runner": "m2", "sim_mode": mode, "sim_prompt": prompt,
                   "sim_lift_mm": round(lift_mm, 2), "sim_lifted": bool(passed), "sim_gate_passed": passed},
        )
    n_pass = sum(1 for r in results if r["passed"])
    # A rate measured on the fallback path is not a measurement of the configured cell, and the number
    # cannot say so on its own. The cause is reported next to the number rather than left as one
    # warning among thousands of log lines.
    degraded = getattr(arm, "curobo_degraded", False)
    reason = str(getattr(arm, "curobo_degraded_reason", "") or "") if degraded else None
    if degraded:
        print("", flush=True)
        print("!" * 78, flush=True)
        print("!! THIS RUN DID NOT USE THE CONFIGURED PLANNER.", flush=True)
        print("!! motion_planner='curobo' was configured; the sidecar could not start, so every", flush=True)
        print("!! motion was planned by the blind IK path. Expect the self-collision guard to", flush=True)
        print("!! refuse poses it would never have been offered; the pass rate below measures", flush=True)
        print("!! the FALLBACK, not this cell.", flush=True)
        print(f"!! reason: {reason}", flush=True)
        print("!! see ext_deps/README.md section 5.", flush=True)
        print("!" * 78, flush=True)
    print(f"M2 GATE: {n_pass}/{runs} passed", flush=True)
    return GateResult(runs=runs, passed=n_pass, results=results, planner_degraded=reason)


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy M2 real-vision pick (Isaac).")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--prompt", type=str, default="a red cube")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-headless", action="store_true", help="alias for --gui")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--mode", type=str, default="easy", choices=["easy", "auto", "closed_loop"],
                    help="P1: grasp mode (easy=trust path; auto=real DecisionEngine; closed_loop=S3+S4)")
    ap.add_argument("--closed-loop", action="store_true", help="deprecated alias for --mode closed_loop")
    ap.add_argument("--record-log", type=str, default=None,
                    help="P0: append one GraspAttemptRecord JSONL line per pick to this path "
                         "(the soak/KPI/RL data source; default off -> byte-identical)")
    ap.add_argument("--debug-frames", type=str, default=None,
                    help="P0: dump the grasp-point overlay PNG per pick into this dir")
    ap.add_argument("--result-json", type=str, default=None,
                    help="P1: write the per-run result dict (for the mode-matrix harness) to this path")
    ap.add_argument("--standoff-mm", type=float, default=None,
                    help="pre-grasp standoff height above the grasp (default 150). Diagnostic knob for a "
                         "SHORTER arm: measured on a UR3e at x=350, a top-down grasp closing along base +Y "
                         "is reachable up to z=100 mm and UNREACHABLE from z=120 mm, while a base +X "
                         "closing is reachable throughout. Lowering the standoff alone does NOT fix that; "
                         "the pick still has to lift, so a base-+Y grasp fails at the retreat instead. "
                         "The real fix is not choosing a base-+Y closing in the first place.")
    ap.add_argument("--detector", type=str, default=None,
                    help="override the grounding model id (e.g. IDEA-Research/grounding-dino-base). "
                         "For A/B measurement against the gated baseline; unset uses the config.")
    add_cell_arguments(ap)
    args = ap.parse_args()
    mode = "closed_loop" if args.closed_loop else args.mode
    result = run_gate(runs=args.runs, prompt=args.prompt, headless=not (args.gui or args.no_headless),
                      data_dir=args.data_dir, mode=mode,
                      record_log=args.record_log, debug_frames=args.debug_frames,
                      cell_kwargs=cell_profile_kwargs(args), standoff_mm=args.standoff_mm,
                      radial_closing=args.radial_closing, detector_model_id=args.detector)
    write_run_result(args.result_json, result.to_dict(), scene="m2", mode=mode, prompt=args.prompt)


if __name__ == "__main__":
    main()
