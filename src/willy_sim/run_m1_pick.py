"""M1 known-pose pick runner: the Isaac composition entry point (config-driven).

Boots the combined UR5e+2F-85 scene from the sim config (the ``config`` tree under the ``sim``
profile), wires the Isaac arm, gripper and ground-truth perception into
``AutonomousGraspService.from_components`` (not ``from_robot_config``: ``IsaacGripper`` is not in the
gripper registry and needs the shared session), and drives ``pick()``. Scene, prim, home, tcp and
asset values all come from config; the pick policy tunings (standoff, retreat, widths, grasp lift)
stay as documented defaults, so the gate stays reproducible.

On-box only (needs Isaac); run with Isaac's bundled python from the repo root:

    <isaac-sim>\\python.bat -m src.willy_sim.run_m1_pick --runs 10
"""

from __future__ import annotations

import os

import argparse
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.harness.cli import add_cell_arguments, cell_profile_kwargs
from src.willy_sim.harness.gate import (
    GateResult,
    gate_passed,
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
from src.willy_sim.perception import GroundTruthPerceptionSource


#: Far clip (m) for the overhead camera on the rendered-depth path, swept without editing code:
#: ``WILLY_M1_FAR_CLIP_M=5.0``. 0 means do not touch the camera at all, because mutating a camera
#: whose annotators are already attached is itself a suspect. The sim config leaves this camera's
#: ``near_clip_m`` unset, so it keeps Isaac's 1.0 m default and the re-clip in ``build_service``
#: is the only thing that widens it. The far value is not the cause of the rendered-depth failure.
_RENDERED_FAR_CLIP_M = float(os.environ.get("WILLY_M1_FAR_CLIP_M", "1.0e6"))


def build_service(
    *,
    headless: bool = True,
    data_dir: str | None = None,
    mode: str = "easy",   # "easy" | "auto" | "closed_loop" (see willy_sim/harness/modes.py)
    # Pick-policy tunings: not paths, so they stay as defaults rather than config.
    pre_open_width_mm: float = 80.0,   # jaw clearance during the descent (object is 30 mm)
    close_width_mm: float = 25.0,      # grip a 30 mm object (5 mm compression)
    grasp_lift_mm: float = 12.0,       # raise the grasp so the fingertips clear the table
    # False: the camera's own rendered depth reaches the pick instead of a flat ground-truth sheet at
    # the object's centre, a sheet that is short by h/2 - grasp_lift. Rendered depth is what a real
    # RealSense delivers.
    ground_truth_depth: bool = True,
    standoff_mm: float = 50.0,
    retreat_mm: float = 100.0,
    natural_aim: bool = False,           # IK prefers the natural (non-self-colliding) branch
    continuous_guard: bool = False,      # per-waypoint collision-avoidance guard (needs WILLY_COAL_PREFIX)
    continuous_guard_margin_mm: float = 8.0,
    cell_kwargs: Mapping[str, Any] | None = None,  # which cell to boot (--robot-model / --profile)
    radial_closing: bool = False,  # spend a symmetric object's free yaw on the reachable axis
):
    """Boot the scene and wire the full pick service from config.

    Returns ``(service, arm, gripper, handles, cfg, cell)``.
    """
    from src.robot.execution.autonomous_grasp import AutonomousGraspService
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
    # Build through the factory, never by name. Naming `GraspCalculator` here makes
    # `robot.grasping.calculator` inert on the Isaac box: a cell that asks for `deep` silently gets
    # the analytic stack, every on-box pick rate then grades the analytic generator by construction,
    # and the factory's fail-closed refusal cannot fire. The selector defaults to `geometric`, appears
    # in no shipped YAML, and that branch is `GraspCalculator(**kwargs)` verbatim.
    from src.robot.grasping.calculator_factory import build_calculator

    # The byte-identical boot prefix (fail-closed arm + session + scene + gripper) is shared.
    cell = bootstrap_sim_cell(data_dir, headless=headless, **(cell_kwargs or {}))
    arm, gripper, handles, cfg, sim = cell.arm, cell.gripper, cell.handles, cell.cfg, cell.sim
    _dwell = cell.dwell  # the steady-state dwell gate
    from src.willy_sim.run_dense_pick import wire_safety_guards  # opt-in guards, default-off
    wire_safety_guards(arm, natural_aim=natural_aim, continuous_guard=continuous_guard,
                       continuous_guard_margin_mm=continuous_guard_margin_mm, fixtures=())

    if not ground_truth_depth:
        # A 1.0 m near clip on a camera ~1 m above its own workspace is wrong for anything that reads
        # the depth. robot.sim.yaml keeps the 1.0 m default for the M1 ground-truth path, where the
        # ground-truth override replaces the rendered depth so nothing looks at it; run_multiview_pick
        # already widens it for this same camera.
        #
        # Widening the clip is not the cause of m1's rendered-depth failure, and does not fix it: the
        # whole masked cloud still lands flat on the support plane. The re-clip is kept because the
        # clip is wrong on its own terms. The failure is open: the mask is non-empty, so candidates
        # get built, while the depth over that mask reads the table, which means the object is absent
        # from the depth buffer for some other reason.
        if _RENDERED_FAR_CLIP_M > 0.0:
            try:
                handles.camera.set_clipping_range(0.05, _RENDERED_FAR_CLIP_M)
            except Exception as exc:  # noqa: BLE001 (a camera that cannot be re-clipped is worth saying)
                print(f"WARN: could not widen the overhead clipping range ({exc}); rendered depth may "
                      "read the support plane through a clipped object", flush=True)
        else:
            print("[m1] runtime re-clip SKIPPED (WILLY_M1_FAR_CLIP_M=0); the camera keeps the "
                  "near_clip_m the config authored", flush=True)
    perception = GroundTruthPerceptionSource(
        camera=handles.camera, target_prim_path=handles.object_prim_path, session=arm.session,
        warmup_steps=sim.scene_setup.render_warmup_steps, ground_truth_depth=ground_truth_depth,
        grasp_lift_mm=grasp_lift_mm,
    )
    # The cell declares whether this arm needs a radial close (robot.ur3e.yaml declares it for the
    # UR3e); --radial-closing still forces it on for a cell that has not declared it.
    _radial = bool(radial_closing or getattr(cell.robot.grasping, "isotropic_radial_closing", False))
    # The banner says which generator the lever applies to. `isotropic_radial_closing` is an
    # analytic-only knob: the learned generator decodes its own closing axis and has nothing to act
    # on, and the factory logs that it is ignoring it. Printing a bare `True` under `calculator: deep`
    # would describe the other path.
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
    # Whether this cell has a CAMERA->BASE transform at all, said at build time. A resolver built
    # around a None transform reads as wired and is not: the calculator stamps no camera_base_z_mm,
    # a field it sets whenever it receives a transform, no table check runs so support_footprint_ready
    # stays False, and the pick simply behaves as a camera-frame pick with nothing downstream
    # saying so.
    _c2b_state = "SET" if handles.camera_to_base is not None else "NONE (no BASE frame downstream)"
    print(f"[m1] camera->base resolver: transform={_c2b_state}", flush=True)
    policy = GraspExecutionPolicy(
        arm=arm, gripper=gripper, standoff_mm=standoff_mm, retreat_mm=retreat_mm,
        pre_open_width_mm=pre_open_width_mm, close_width_mm=close_width_mm,
        require_base_frame_grasp=True,
        # Gate every commanded move on the dwell steady-state config (the sim arm settles fast).
        require_steady_before_motion=(_dwell.require_steady_before_motion if _dwell else False),
        steady_timeout_s=(_dwell.steady_timeout_s if _dwell else 5.0),
    )
    grasp_mode = resolve_demo_mode(mode)
    service = AutonomousGraspService.from_components(
        arm=arm, calculator=calculator, perception=perception, gripper=gripper,
        mode=grasp_mode, frame_resolver=resolver, policy=policy, max_attempts=3,
        **mode_service_kwargs(grasp_mode),
    )
    return service, arm, gripper, handles, cfg, cell


def run_gate(runs: int = 10, *, headless: bool = True, data_dir: str | None = None,
             mode: str = "easy", record_log: str | None = None,
             debug_frames: str | None = None,
             cell_kwargs: Mapping[str, Any] | None = None,
             radial_closing: bool = False,
             # False: the camera's own rendered depth reaches the pick. See build_service.
             ground_truth_depth: bool = True) -> GateResult:
    """Run ``runs`` known-pose picks, re-placing the object each time, and score the M1 gate.

    A run passes iff ``pick()`` reports succeeded and the object rose by >= the configured lift
    threshold (``robot.sim.scene_setup.gate``).

    Both instrumentation hooks are opt-in and default-off: ``record_log`` appends one
    ``GraspAttemptRecord`` JSONL line per pick, stamped with the ground-truth ``sim_lift_mm`` and
    ``sim_lifted``, and ``debug_frames`` dumps the grasp-point overlay PNG per pick. With neither
    set the gate is byte-identical.
    """
    service, arm, gripper, handles, cfg, cell = build_service(
        headless=headless, data_dir=data_dir, mode=mode, cell_kwargs=cell_kwargs,
        radial_closing=radial_closing, ground_truth_depth=ground_truth_depth,
    )
    if debug_frames:  # render the grasp-point overlay per pick. M1 ground-truth perception carries
        service.enable_debug_image_rendering()  # no usable rgb on Isaac 5.1, so no frame; M2 does.
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    robot = require_robot(cfg)
    gate = robot.sim.scene_setup.gate
    home_q = np.asarray(robot.sim.home_joint_positions, dtype=np.float64)
    obj = SingleRigidPrim(handles.object_prim_path)
    home_pos, home_quat = obj.get_world_pose()
    run_id = f"m1-{int(time.time())}"
    results = []
    for i in range(runs):
        # Clean reset before each run: release the jaw, home the arm, re-place + settle the object.
        gripper.open()
        arm.move_joint(JointPositions(home_q))
        z0 = reset_object_to_home_z0(obj, home_pos, home_quat, arm.session, settle_steps=30)
        report = service.pick()
        lift_mm = lift_mm_since(obj, z0)
        outcome = getattr(report, "outcome", None)
        succeeded = getattr(outcome, "value", outcome) == "succeeded"
        passed = bool(succeeded and lift_mm >= gate.lift_threshold_mm)
        results.append({"run": i, "succeeded": succeeded, "lift_mm": round(lift_mm, 1), "passed": passed})
        print(f"RUN {i}: succeeded={succeeded} lift_mm={lift_mm:.1f} passed={passed}", flush=True)
        # Telemetry on stdout, not via logs/robot/robot.log. That file is a RotatingFileHandler on
        # Windows: while anything else holds it the writes are dropped, so a line read back from it
        # cannot be attributed to any particular run. This run's stdout goes to its own redirect, so
        # a number printed here provably belongs to it.
        from src.willy_sim.run_multiview_pick import _trace_calc  # noqa: PLC0415

        if _trace_calc():
            _calc = getattr(getattr(getattr(service, "runtime", None), "orchestrator", None),
                            "calculator", None)
            _pos = np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0
            print(f"[calc] run={i} target_true_mm={np.round(_pos, 1).tolist()} "
                  f"telemetry={dict(getattr(_calc, 'last_telemetry', None) or {})}", flush=True)
        write_pick_artifacts(
            report, attempt_id=f"{run_id}-{i:04d}",
            record_log=record_log, debug_dir=debug_frames,
            debug_png=service.last_debug_image_png,
            extra={**cell_identity(cell), "sim_runner": "m1", "sim_mode": mode, "sim_lift_mm": round(lift_mm, 2),
                   "sim_lifted": bool(lift_mm >= gate.lift_threshold_mm), "sim_gate_passed": passed},
        )
    n_pass = sum(1 for r in results if r["passed"])
    gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
    print(f"GATE: {n_pass}/{runs} passed -> gate_passed={gate_ok}", flush=True)
    return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy M1 known-pose pick (Isaac).")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-headless", action="store_true", help="alias for --gui")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--mode", type=str, default="easy", choices=["easy", "auto", "closed_loop"],
                    help="P1: grasp mode (easy=trust path; auto=real DecisionEngine; closed_loop=S3+S4)")
    ap.add_argument("--record-log", type=str, default=None,
                    help="P0: append one GraspAttemptRecord JSONL line per pick to this path "
                         "(the soak/KPI/RL data source; default off -> byte-identical)")
    ap.add_argument("--debug-frames", type=str, default=None,
                    help="P0: dump the grasp-point overlay PNG per pick into this dir")
    ap.add_argument("--result-json", type=str, default=None,
                    help="P1: write the per-run result dict (for the mode-matrix harness) to this path")
    ap.add_argument("--rendered-depth", action="store_true",
                    help="let the camera's OWN rendered depth reach the pick instead of the flat "
                         "ground-truth sheet at the object's centre; the September-shaped path")
    add_cell_arguments(ap)
    args = ap.parse_args()
    result = run_gate(runs=args.runs, headless=not (args.gui or args.no_headless), data_dir=args.data_dir,
                      mode=args.mode, record_log=args.record_log, debug_frames=args.debug_frames,
                      cell_kwargs=cell_profile_kwargs(args), radial_closing=args.radial_closing,
                      ground_truth_depth=not args.rendered_depth)
    write_run_result(args.result_json, result.to_dict(), scene="m1", mode=args.mode)


if __name__ == "__main__":
    main()
