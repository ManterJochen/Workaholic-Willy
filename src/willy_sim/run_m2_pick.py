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
from src.robot.core.camera_world import CameraWorldDecline
from src.willy_sim.harness.camera_world import SimCameraWorld
from src.contracts import UNSET, Maybe


def build_service(
    *,
    camera_world: "Maybe[CameraWorldDecline | SimCameraWorld]" = UNSET,
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
    cell = bootstrap_sim_cell(data_dir, headless=headless, **(cell_kwargs or {}),
                              camera_world=camera_world)
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
        # The overhead camera's name in the live world, so the pick loop offers its masks under it.
        camera_name="overhead",
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
        data_dir=data_dir,
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


def _declared_planner(arm: object) -> str | None:
    """The sentence a report carries when this run did not use a planner, else ``None``.

    Nothing degrades: a cuRobo arm that cannot reach its sidecar refuses its motions rather
    than running them blind, so the only way a run is unplanned is that somebody asked for
    it. That is a decision worth printing beside the rate, because a rate measured without
    a planner is not a measurement of a planned cell.
    """
    planner = str(getattr(arm, "_motion_planner", "") or "")
    if planner in ("", "curobo"):
        return None
    return f"this run was asked for motion_planner={planner!r}, so no planner planned its motions"


#: Physics steps after the jaw opens and before the park, so a released object has landed (a 100 mm
#: fall takes about 0.15 s; this is half a second at the sim's 60 Hz).
_RELEASE_SETTLE_STEPS = 30
#: Where a reset carries the object when the park is refused: 1.5 m along base +Y, outside every
#: world limit.
_OUT_OF_CELL_M = (0.0, 1.5, 0.0)

#: The profile layer that gives the reference cell a live camera world.
CAMERA_WORLD_LAYER = "sim_camera_world"


def camera_world_for(decline_camera_world: str | None, cell_kwargs: Mapping[str, Any] | None,
                     ) -> "tuple[CameraWorldDecline | SimCameraWorld, dict[str, Any]]":
    """What this run's cell stands on, and the cell kwargs it boots with.

    Wired by default: the overhead camera's live world, with the ``sim_camera_world`` layer
    appended. A reason declines it for a control run, and the layer is left as the caller gave it.
    """
    kwargs = dict(cell_kwargs or {})
    if decline_camera_world is not None:
        return CameraWorldDecline(decline_camera_world), kwargs
    profiles = list(kwargs.get("extra_profiles") or ())
    if CAMERA_WORLD_LAYER not in profiles:
        profiles.append(CAMERA_WORLD_LAYER)
    kwargs["extra_profiles"] = profiles
    return SimCameraWorld("overhead"), kwargs


def _failure_of(report: Any) -> "str | None":
    """Why a pick did not succeed, in one line, from the report's own summary.

    ``None`` when the report says nothing.
    """
    summary = getattr(report, "failure_summary", None)
    text = summary() if callable(summary) else None
    return " ".join(str(text).split())[:200] if text else None


def camera_world_line(results: "list[dict[str, Any]]") -> str:
    """The one line printed beside the rate.

    It counts the weakest camera world of each run and names the runs that did not pick.
    """
    counts: dict[str, int] = {}
    for row in results:
        use = str(row.get("camera_world") or "none")
        counts[use] = counts.get(use, 0) + 1
    raised = [row["run"] for row in results if row.get("camera_world_raised")]
    failed = [
        f"run {row['run']} {row.get('failure') or row.get('motion_status') or 'failed'}"
        + (f" ({row['motion_message']})" if row.get("motion_message") else "")
        for row in results if not row.get("succeeded") and not row.get("camera_world_raised")
    ]
    text = "camera world: " + (", ".join(f"{use} {n}" for use, n in sorted(counts.items())) or "no runs")
    text += f"; CameraWorldUnavailable raised in {len(raised)} run(s)"
    if raised:
        text += " (" + ", ".join(str(run) for run in raised) + ")"
    if any("planned_stamps" in row for row in results):
        # What each planned motion's world left out rides on its stamp, and the gate says on how many
        # it did.
        planned = sum(int(row.get("planned_stamps") or 0) for row in results)
        kept = sum(int(row.get("kept_out_stamps") or 0) for row in results)
        text += f"; kept out on {kept} of {planned} planned motion stamp(s)"
    if failed:
        text += "; not picked: " + ", ".join(failed)
    refused_parks = [f"{row['run']} ({row['park']})" for row in results if row.get("park")]
    if refused_parks:
        text += f"; reset park refused before {len(refused_parks)} run(s): " + ", ".join(refused_parks)
    return text


def run_gate(runs: int = 10, *, prompt: str = "a red cube", headless: bool = True,
             data_dir: str | None = None, mode: str = "easy",
             record_log: str | None = None, debug_frames: str | None = None,
             cell_kwargs: Mapping[str, Any] | None = None,
             standoff_mm: float | None = None,
             radial_closing: bool = False,
             detector_model_id: str | None = None,
             decline_camera_world: str | None = None) -> GateResult:
    """Run ``runs`` real-vision picks (re-place object, park arm, perceive, grasp) and score lift.

    ``mode`` is "easy", "auto" (a real ``DecisionEngine``) or "closed_loop", where the fixed overhead
    camera lets the two-scan refine IoU-match. closed_loop also prints the per-run refinement
    telemetry, so a reader can see whether the refine changes the grasp and still holds the gate.

    Both instrumentation hooks are opt-in and default-off: ``record_log`` appends one
    ``GraspAttemptRecord`` JSONL line per pick, stamped with the ground-truth ``sim_lift_mm`` and
    ``sim_lifted``, and ``debug_frames`` dumps the grasp-point overlay PNG per pick (the real-vision
    frame: SAM2 mask plus grasp). With neither set the gate is byte-identical.

    The cell plans against a live camera world from the overhead camera unless
    ``decline_camera_world`` gives a reason to decline it (see :func:`camera_world_for`). A pick whose
    camera could not vouch for the cell is counted beside the rate rather than ending the gate.
    """
    cl = str(mode).lower().replace("-", "_") == "closed_loop"
    camera_world, cell_kwargs = camera_world_for(decline_camera_world, cell_kwargs)
    service, arm, gripper, handles, cfg, cell = build_service(
        prompt=prompt, headless=headless, data_dir=data_dir, mode=mode, cell_kwargs=cell_kwargs,
        # None keeps build_service's 150 mm default, so the call is byte-identical.
        standoff_mm=150.0 if standoff_mm is None else float(standoff_mm),
        radial_closing=radial_closing,
        detector_model_id=detector_model_id,
        camera_world=camera_world,
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
    results: list[dict[str, Any]] = []
    #: Each run whose reset park was refused, and what the reset did about it.
    parks: dict[int, str] = {}
    for i in range(runs):
        # Clean reset: release jaw, park the arm clear of the camera, re-place the object at rest.
        gripper.open()
        # Let a released object fall before the park. With a live camera world the park is planned
        # against what the camera sees, and a cube still between the opened fingers is an obstacle the
        # park starts inside, so every later run would begin from the lifted pose with the arm in the
        # camera's view.
        arm.session.step_n(_RELEASE_SETTLE_STEPS)
        # A typed joint move: it gates the park destination through the preflight (joint limit,
        # self-collision, payload, static IK quality) and resets the continuity reference around this
        # deliberate discontinuity, so the next grasp move() is not diffed across the park.
        parked = arm.move_to_joints(JointPositions(park_q))
        if not getattr(parked, "ok", True):
            print(f"RUN {i}: park refused ({getattr(parked, 'status', None)}): {getattr(parked, 'message', '')}",
                  flush=True)
            # A reset, not a pick: the object is carried out of the cell, the frames of it are dropped
            # and the park is asked again, so one refused park does not leave the arm in the camera's
            # view for every later run. A second refusal is printed too, and the object then stood
            # somewhere the arm is not.
            obj.set_world_pose(position=np.asarray(home_pos) + np.array(_OUT_OF_CELL_M), orientation=home_quat)
            arm.session.step_n(_RELEASE_SETTLE_STEPS)
            world = getattr(arm, "live_planner_world", None)
            if world is not None:
                world.drop_cached_frames()
            again = arm.move_to_joints(JointPositions(park_q))
            print(f"RUN {i}: park with the object out of the cell: {getattr(again, 'status', None)}", flush=True)
            park = "refused"
            subset = getattr(arm, "_arm_subset", None)
            if not getattr(again, "ok", True) and subset is not None:
                # Refused with the object gone, so the obstacle is the robot's own body where the self
                # filter misses it. A reset, not a motion: the arm is set to the park pose in the
                # simulator, printed and counted, so the next run measures its own pick instead of
                # inheriting an arm left in the camera's view.
                subset.set_joint_positions(np.asarray(park_q, dtype=np.float64))
                subset.apply_action(joint_positions=np.asarray(park_q, dtype=np.float64))
                arm.session.step_n(_RELEASE_SETTLE_STEPS)
                if world is not None:
                    world.drop_cached_frames()
                park = "set in the sim"
                print(f"RUN {i}: the arm was set to the park pose in the simulator, not moved there", flush=True)
            parks[i] = park
        z0 = reset_object_to_home_z0(obj, home_pos, home_quat, arm.session, settle_steps=25)
        from src.robot.core import CameraWorldUnavailable

        report = service.pick()
        # `pick()` reports a fault of the cell instead of raising it. Any fault but a camera world
        # stops the gate, as its raise did.
        fault = getattr(report, "fault", None)
        if fault is not None and not isinstance(fault, CameraWorldUnavailable):
            raise fault
        if fault is not None:
            # A camera that could not vouch for the cell stops that pick, not the gate: counted
            # beside the rate.
            print(f"RUN {i}: CameraWorldUnavailable: {fault}", flush=True)
            results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False,
                            "camera_world": None, "camera_world_raised": True, "motion_status": None})
            continue
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
        from src.robot.grasping.motion.execution_policy import weakest_camera_world

        stamps = tuple(getattr(getattr(report, "pick_report", None), "camera_worlds", ()) or ())
        weakest = weakest_camera_world(stamps)
        motion = getattr(getattr(report, "pick_report", None), "motion_status", None)
        message = getattr(getattr(report, "pick_report", None), "motion_message", None)
        planned = [stamp for stamp in stamps if str(getattr(stamp, "use", "")) == "planned"]
        results.append({"run": i, "succeeded": succeeded, "lift_mm": round(lift_mm, 1), "passed": passed,
                        "camera_world": str(weakest.use) if weakest is not None else None,
                        "camera_world_raised": False,
                        "motion_status": str(getattr(motion, "value", motion)) if motion is not None else None,
                        "failure": None if succeeded else _failure_of(report), "park": parks.get(i),
                        "planned_stamps": len(planned),
                        "kept_out_stamps": sum(1 for stamp in planned if getattr(stamp, "keep_out", None) is not None),
                        "motion_message": None if succeeded or message is None else " ".join(str(message).split())[:200]})
        print(f"RUN {i}: succeeded={succeeded} lift_mm={lift_mm:.1f} passed(lifted)={passed}", flush=True)
        from src.willy_sim.harness.gate import last_line_motion

        print(f"RUN {i}: line_motion={last_line_motion(service)}", flush=True)
        write_pick_artifacts(
            report, attempt_id=f"{run_id}-{i:04d}",
            record_log=record_log, debug_dir=debug_frames,
            debug_png=service.last_debug_image_png,
            extra={**cell_identity(cell), "sim_runner": "m2", "sim_mode": mode, "sim_prompt": prompt,
                   "sim_lift_mm": round(lift_mm, 2), "sim_lifted": bool(passed), "sim_gate_passed": passed},
        )
    n_pass = sum(1 for r in results if r["passed"])
    # A rate measured without a planner is not a measurement of a planned cell, and the number cannot
    # say so on its own. The cause is reported next to the number rather than left as one warning
    # among thousands of log lines.
    reason = _declared_planner(arm)
    if reason is not None:
        print("", flush=True)
        print("!" * 78, flush=True)
        print("!! NO PLANNER PLANNED THIS RUN.", flush=True)
        print("!! Somebody asked for it: a cuRobo cell whose sidecar cannot start refuses its", flush=True)
        print("!! motions rather than driving them blind, so this is a decision and not a", flush=True)
        print("!! degradation. Expect the self-collision guard to refuse poses a planner would", flush=True)
        print("!! never have offered; the pass rate below measures the blind path, not this cell.", flush=True)
        print(f"!! reason: {reason}", flush=True)
        print("!! see ext_deps/README.md section 5.", flush=True)
        print("!" * 78, flush=True)
    print(camera_world_line(results), flush=True)
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
    ap.add_argument("--decline-camera-world", type=str, default=None, metavar="REASON",
                    help="boot without the live camera world, declining it for this reason (a control run); "
                         "by default the cell plans against the overhead camera's world")
    add_cell_arguments(ap)
    args = ap.parse_args()
    mode = "closed_loop" if args.closed_loop else args.mode
    result = run_gate(runs=args.runs, prompt=args.prompt, headless=not (args.gui or args.no_headless),
                      data_dir=args.data_dir, mode=mode,
                      record_log=args.record_log, debug_frames=args.debug_frames,
                      cell_kwargs=cell_profile_kwargs(args), standoff_mm=args.standoff_mm,
                      radial_closing=args.radial_closing, detector_model_id=args.detector,
                      decline_camera_world=args.decline_camera_world)
    write_run_result(args.result_json, result.to_dict(), scene="m2", mode=mode, prompt=args.prompt)


if __name__ == "__main__":
    main()
