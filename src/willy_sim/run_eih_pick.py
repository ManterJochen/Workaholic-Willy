"""Eye-in-hand wrist-camera pick runner for Isaac.

The camera rides the wrist, unlike the fixed overhead cameras of ``run_m1_pick`` and
``run_m2_pick`` with their static CAMERA->BASE. Per pick the runner moves to a view pose that aims
the wrist camera at the object (reusing the calibration look-at geometry), perceives from the wrist
with ground-truth depth, resolves CAMERA->BASE live through :class:`EyeInHandFrameResolver` (the
calibrated ``T_cam_to_tool`` composed with the view-pose TCP), then grasps and executes. It scores
the lift gate from ``sim.scene_setup.gate``.

``T_cam_to_tool`` prefers the persisted calibrated hand-eye result (``_load_calibrated_eih``,
written by run_eih_calibrate from the AX=XB calibration); it falls back to the empirical
ground-truth oracle only when no good or excellent artifact exists. The dual-cam coordinator
(run_fused_pick) likewise drives its overhead coarse localize from the persisted eye-to-hand
calibration (``_load_calibrated_eth``) rather than from a live fit.

The view pose is derived in closed form from that transform, so the offset wrist camera looks
straight down and centred over the object: an oblique view biases the ground-truth-depth centroid
and the grasp misses. The executed pick runs through ``AutonomousGraspService`` in easy mode with
the SINGLE_OBJECT profile, and the surrounding autonomy (effective-config telemetry plus
orchestrator overlays) is wired and reported alongside it.

Requires Isaac, so it runs on the sim box only. Run with Isaac's bundled python:
    <isaac-sim>\\python.bat -m src.willy_sim.run_eih_pick --runs 10
    <isaac-sim>\\python.bat -m src.willy_sim.run_eih_pick --runs 3 --gui   # watch it
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
from src.willy_sim.harness.env import RunnerEnv
from src.willy_sim.harness.instrumentation import (
    cell_identity,
    write_pick_artifacts,
    write_run_result,
)
from src.willy_sim.harness.modes import mode_service_kwargs, resolve_demo_mode
from src.willy_sim.perception import (
    GroundTruthPerceptionSource,
    MultiObjectGroundTruthPerceptionSource,
)
from src.willy_sim.scene import (
    camera_to_tcp_ground_truth,
    mount_wrist_camera,
)
from src.geometry import Frame, Pose

# Down-looking pose to seed the empirical CAMERA->TOOL oracle; the camera must see the table here.
# Only reached on the ground-truth-oracle fallback (no calibrated artifact); a loaded calibration
# skips it entirely. Per-robot because the UR5e default (450,0,350) is r=492 mm, 98% of the UR3e's
# 500 mm sphere and outside its shipped workspace (x<=400, z<=320); (280,0,300) is reachable
# top-down and stays inside the ur3e workspace.
_INIT_VIEWPOSE_MM: dict[str, tuple[float, float, float]] = {
    "ur3e": (280.0, 0.0, 300.0),
}
_INIT_VIEWPOSE_DEFAULT = (450.0, 0.0, 350.0)


def init_viewpose(robot_model: str | None) -> Pose:
    """The oracle-seed down-looking pose for ``robot_model`` (reachable + table-in-view)."""
    x, y, z = _INIT_VIEWPOSE_MM.get((robot_model or "").strip(), _INIT_VIEWPOSE_DEFAULT)
    return Pose(
        position_mm=np.array([x, y, z]),
        quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
        frame=Frame.BASE,
        label="eih-pick-init",
    )
# Where run_eih_calibrate persists the calibrated hand-eye transforms.
DEFAULT_CAL_DIR = Path(__file__).resolve().parents[2] / "logs" / "calibration"


def _load_calibrated_eih(marker: str, cal_dir: "Path | None" = None):
    """Load the persisted calibrated ``CAMERA->TOOL`` transform from run_eih_calibrate, or None.

    Deserialized with ``transform_from_dict``: the artifact round-trips frame-tagged, so it drops
    straight into ``EyeInHandFrameResolver``. Returns None, which the caller reads as the
    ground-truth-oracle fallback, when the file is missing or its quality is not good or
    excellent; that is the same gate run_eih_calibrate applies on the writing side. ``cal_dir`` is
    the per-robot artifact directory; ``None`` reads the shared flat directory.
    """
    import json

    from src.geometry import transform_from_dict

    path = (cal_dir or DEFAULT_CAL_DIR) / f"eih_wrist_cam_{marker}.json"
    if not path.exists():
        print(f"no calibrated EIH artifact at {path} -> GT-oracle fallback", flush=True)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload.get("quality")
    if quality not in ("good", "excellent"):
        print(f"EIH calibration quality={quality!r} (not good/excellent) -> GT-oracle fallback", flush=True)
        return None
    t = transform_from_dict(payload["t_cam_to_tool"])
    print(f"loaded CALIBRATED EIH t_cam_to_tool (marker={marker}, quality={quality}, "
          f"err_vs_oracle={payload.get('error_vs_oracle_mm')} mm)", flush=True)
    return t


def _load_calibrated_eth(marker: str, *, allow_marginal: bool = True, cal_dir: "Path | None" = None):
    """Load the persisted eye-to-hand overhead CAMERA->BASE transform from run_eth_calibrate, or None.

    This lets the dual-cam pick drive its coarse localize from a persisted calibration instead of
    the live ground-truth Umeyama fit. The overhead only coarse-localizes and the wrist eye-in-hand
    view refines, so a 'marginal' calibration is accepted by default: a few mm of overhead error is
    absorbed by the refine step. Returns None, which the caller reads as the live-fit fallback, when
    the artifact is missing or its quality is outside the accepted set.
    """
    import json

    from src.geometry import transform_from_dict

    path = (cal_dir or DEFAULT_CAL_DIR) / f"eth_overhead_{marker}.json"
    if not path.exists():
        print(f"no calibrated ETH artifact at {path} -> live GT-fit fallback", flush=True)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload.get("quality")
    accepted = ("good", "excellent", "marginal") if allow_marginal else ("good", "excellent")
    if quality not in accepted:
        print(f"ETH calibration quality={quality!r} (not in {accepted}) -> live GT-fit fallback", flush=True)
        return None
    t = transform_from_dict(payload["transform"])
    print(f"loaded CALIBRATED ETH overhead cam->base (marker={marker}, quality={quality}, "
          f"rmse={payload.get('rmse_mm')} mm)", flush=True)
    return t


def topdown_view_pose(obj_pos_mm, p_cam_tool, view_height_mm: float) -> Pose:
    """Closed-form TCP pose that aims the offset wrist camera straight down at an object.

    The camera ends up centred over ``obj_pos_mm`` at ``view_height_mm`` above it:
    ``p_tool_base = (C + [0,0,H]) - R_down @ p_cam_tool``, with quat_xyzw [1,0,0,0] pointing down.
    Shared by the single-cam pick and the dual-cam coordinator.
    """
    r_down = np.diag([1.0, -1.0, -1.0])  # rotation of the down quat (180 deg about base X)
    centre = np.asarray(obj_pos_mm, dtype=np.float64)
    p_tool_base = (centre + np.array([0.0, 0.0, float(view_height_mm)])) - r_down @ np.asarray(p_cam_tool, dtype=np.float64)
    return Pose(position_mm=p_tool_base, quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                frame=Frame.BASE, label="eih-topdown-view")


def coarse_centroid_base(frame, t_cam_to_base, seg_index: int = 0):
    """Back-project a perception frame's object-mask centroid to a BASE-frame XYZ (mm), or None.

    Reuses the calculator's own back-projector (``masked_point_cloud``) so the optical convention
    matches the grasp pipeline exactly. ``t_cam_to_base`` is the camera's CAMERA->BASE Transform
    (the overhead StaticCameraToBaseResolver value). ``seg_index`` selects which segmentation to
    localize (the prompted object among clutter). This is the overhead coarse-localize for the
    dual-cam fused pick.
    """
    from src.robot.grasping.geometry import masked_point_cloud

    segs = getattr(frame, "segmentations", None)
    if not segs or seg_index >= len(segs):
        return None
    mask = np.asarray(segs[seg_index].mask, dtype=bool)
    cloud = masked_point_cloud(mask, frame.depth_map, frame.intrinsics, unit="mm")
    pts = np.asarray(cloud.points_mm, dtype=np.float64)
    if pts.shape[0] == 0:
        return None
    m = np.asarray(t_cam_to_base.to_matrix(), dtype=np.float64)
    base_pts = (m[:3, :3] @ pts.T).T + m[:3, 3]
    return base_pts.mean(axis=0)


def build_service(
    *,
    headless: bool = True,
    data_dir: str | None = None,
    mode: str = "easy",              # "easy" | "auto" | "closed_loop" (see harness/modes.py)
    marker: str = "ground_truth",    # which calibrated EIH artifact to use (else GT-oracle fallback)
    coarse_object_pos_mm=None,       # BASE-frame object centroid (dual-cam coarse handoff); else sim pos
    objects_override=None,           # list[SimObjectConfig] for a clutter scene; else the config object
    wrist_target_prim: str | None = None,  # prim the wrist GT perception targets (the chosen object); else first
    view_height_mm: float = 275.0,   # wrist-camera height above the object centroid (band 250-300)
    pre_open_width_mm: float = 80.0,
    close_width_mm: float = 25.0,
    grasp_lift_mm: float = 12.0,
    # Default True keeps the flat ground-truth depth sheet. See the call site for what it costs.
    ground_truth_depth: bool = True,
    standoff_mm: float = 50.0,
    retreat_mm: float = 100.0,
    natural_aim: bool = False,           # IK prefers the natural (non-self-colliding) branch
    continuous_guard: bool = False,      # per-waypoint collision-avoidance guard (needs WILLY_COAL_PREFIX)
    continuous_guard_margin_mm: float = 8.0,
    bin_walls: "list | None" = None,     # static bin/tray walls (FixtureBoxConfig list): collidable prims + a
    #                                      self-collision fixture per wall so an approach into a wall is rejected
    real_klt_bin: "tuple[float, ...] | None" = None,  # (cx,cy,z[,zscale]): reference the real small_KLT visual mesh
    curobo_bin_world: bool = False,      # register the bin walls in cuRobo's collision world so it plans around them
    post_scene_hook: "object | None" = None,  # callback(stage) run after the scene is built, before perception is
    #                                      created (so extra prims are in the GT instance-id map, e.g. GSO clutter)
    cell_kwargs: "dict | None" = None,   # bootstrap kwargs (robot_model / extra_profiles) for the cell
    service_from_config: bool = False,   # boot through from_robot_config (the path a real cell takes)
    # Let the config own the sub-policies instead of this runner. Default False leaves every run
    # byte-identical. It covers a measurement trap: `mode_service_kwargs` hands `from_robot_config`
    # a hand-built DecisionEngine in auto mode (and a refiner plus verifier in closed_loop), and
    # `build_decision_layer` takes the constructor argument in preference to the config block
    # (`resolved_decision_engine is None and ... decision.enabled`), as `build_subpolicies` does for
    # the refiner and the verifier. So `--boot config --grasp-mode auto` runs a decision engine that
    # came from this file, and toggling `grasping.decision.enabled` changes nothing: a null result
    # that reads as "the block is harmless" when it means "the block was never consulted". With this
    # True the runner passes nothing and the three config blocks (`decision`, `closed_loop`,
    # `verification`) are the only source.
    config_subpolicies: bool = False,
    # Opt-in reachability filter. Default False is byte-identical: no IK service reaches the
    # calculator, `rejected_ik` stays 0, and `grasping.feasibility`, which re-ranks on
    # per-candidate IK quality, cannot act even when its config arrives, because the re-rank sits
    # behind `if ik_service is not None`.
    #
    # It does two things at once, so measure them apart. Switching it on (a) adds a reachability
    # filter that discards candidates, and (b) makes the feasibility re-rank possible. A single run
    # that moves both attributes nothing to either. Run IK alone first and read `rejected_ik`, then
    # run IK together with the feasibility layer.
    ik_service: bool = False,
    # What the grasp path sees: the target's mask alone, or every authored object's mask.
    #
    # The single-object source emits one segmentation, so on a scene with neighbours the stack is
    # structurally blind to them: ``other_object_masks`` is empty, and that is the input the
    # DENSE_CLUTTER sampler, the corridor-risk analyzer and the approach validator all consume. The
    # corridor analyzer then scores the target's own body against nothing, so its blockage numbers
    # cannot be calibrated against a real obstacle.
    #
    # True, the default, gives every object its own labelled segmentation and tells the orchestrator
    # which label is the target. On a single-object scene both sources emit exactly one mask.
    #
    # Safe only with ``target_label`` set. With N segmentations and no label the loop treats every
    # one of them as an executable target (`_best_result_over_segmentations_inner` in
    # `pick_loop.py`), so the cell may lift a distractor and report success. The label is resolved
    # from the target prim below and the build refuses if it cannot resolve it, rather than running
    # unlabelled.
    multi_object_perception: bool = True,
    # Bin-clearing semantics: every authored object is a legitimate target, so no label is set.
    #
    # This is the only configuration in which an unlabelled multi-object frame is correct, and the
    # distinction is the task, not a tuning. With a prompt ("pick the green cube") an unlabelled
    # frame is a bug: the loop ranks across all segmentations and may lift a distractor while
    # reporting success. With "empty this bin" there is no distractor, whichever object the policy
    # takes first is a legitimate answer, and choosing which one is the job of `grasping.ordering`.
    #
    # It is therefore also the only setting in which ordering can be measured. With a target label
    # exactly one segmentation is executable, `select_target` is handed a single candidate, and the
    # block is inert: when the operator names the object there is no ordering decision left.
    #
    # Default False: every prompt-driven runner keeps the label and the distractor guarantee.
    clear_any_object: bool = False,
):
    """Boot the scene, the wrist camera and the eye-in-hand pick service.

    Returns (service, arm, gripper, handles, cfg, view_pose, cell).

    ``mode`` selects the grasp mode: "easy" is the deterministic open-loop path, "auto" wires the
    real ``DecisionEngine`` (perceive, rank, decide, grasp), and "closed_loop" adds the refiner and
    the verifier.
    """
    from src.robot.execution.autonomous_grasp import AutonomousGraspService
    from src.robot.execution.autonomous_grasp.config import GraspMode
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
    # Through the factory, not by name, so `robot.grasping.calculator: deep` reaches this runner
    # and the six modules that share its `build_service`. Default-off byte-identical: the selector
    # defaults to `geometric` and that branch is `GraspCalculator(**kwargs)` verbatim.
    #
    # The deep calculator cannot honour two of the kwargs below, and the factory refuses them when
    # they are switched on: `ik_service` is the reachability filter that discards unreachable
    # candidates, and `corridor_risk_per_candidate` produces the per-candidate corridor risk. At
    # their off positions, which is what every default run passes, the build proceeds unchanged.
    from src.robot.grasping.calculator_factory import build_calculator

    # The byte-identical boot prefix is shared; eih passes its single/clutter object override, plus (opt-in)
    # the industrial bin: collidable wall prims + the real small_KLT visual mesh. No bin leaves it unchanged.
    scene_kwargs: dict[str, object] = {"objects_override": objects_override}
    if bin_walls is not None:
        scene_kwargs["bin_walls"] = bin_walls
    if real_klt_bin is not None:
        scene_kwargs["real_klt_bin"] = real_klt_bin
    cell = bootstrap_sim_cell(
        data_dir, headless=headless, scene_kwargs=scene_kwargs,
        post_scene_hook=post_scene_hook,  # type: ignore[arg-type]
        **(cell_kwargs or {}),
    )
    arm, gripper, handles, cfg, sim = cell.arm, cell.gripper, cell.handles, cell.cfg, cell.sim
    _dwell = cell.dwell  # steady-state dwell gate
    # Opt-in natural-aim IK seed + continuous collision-avoidance guard (default-off byte-identical). When a
    # bin is wired, its walls become the guard fixtures so an in-motion approach into a wall is caught too.
    from src.willy_sim.run_dense_pick import wire_safety_guards
    _fixtures = list(bin_walls) if bin_walls else []
    wire_safety_guards(arm, natural_aim=natural_aim, continuous_guard=continuous_guard,
                       continuous_guard_margin_mm=continuous_guard_margin_mm, fixtures=_fixtures)
    if bin_walls:
        # Make the bin "really real" for the whole stack: (1) inject the walls into the SafetyPreflight so a
        # static approach into a wall is self-collision-rejected, and (2) register them in cuRobo's collision
        # world so a "curobo" plan routes around them (mirrors run_dense_pick). BASE frame, metres, WXYZ.
        from src.robot.safety.preflight import SafetyPreflight

        _bin_sc = cell.robot.safety.self_collision.model_copy(update={"fixtures": _fixtures})
        _safety = cell.robot.safety.model_copy(update={"self_collision": _bin_sc})
        arm._preflight = SafetyPreflight.from_safety_config(_safety, cell.robot.workspace_limits)
        if curobo_bin_world:
            _cuboids = [
                {"name": fx.name,
                 "dims_m": [2.0 * float(fx.half_extents_mm[0]) / 1000.0,
                            2.0 * float(fx.half_extents_mm[1]) / 1000.0,
                            2.0 * float(fx.half_extents_mm[2]) / 1000.0],
                 "pose": [float(fx.center_mm[0]) / 1000.0, float(fx.center_mm[1]) / 1000.0,
                          float(fx.center_mm[2]) / 1000.0, 1.0, 0.0, 0.0, 0.0]}
                for fx in bin_walls
            ]
            _cuboids.append({"name": "floor", "dims_m": [2.0, 2.0, 0.05],
                             "pose": [0.0, 0.0, -0.026, 1.0, 0.0, 0.0, 0.0]})
            _n = arm.set_curobo_world(_cuboids)
            print(f"[bin] {len(_fixtures)} walls -> SafetyPreflight + {_n} cuRobo obstacles (plans around the walls)",
                  flush=True)
        else:
            print(f"[bin] {len(_fixtures)} walls -> SafetyPreflight self-collision fixtures", flush=True)

    wcam = sim.cameras["wrist"]
    # The wrist camera's mount, resolution and near-clip fields are Optional in the schema but
    # required by this runner; narrow them here and raise if a sim tree omits them, which the sim
    # profile never does because it sets all five. The fields are already Pydantic-coerced tuples.
    if (
        wcam.mount_offset_mm is None or wcam.mount_aim_mm is None or wcam.mount_up_hint is None
        or wcam.resolution is None or wcam.near_clip_m is None
    ):
        raise ValueError(
            "sim wrist camera config requires mount_offset_mm / mount_aim_mm / mount_up_hint / "
            "resolution / near_clip_m."
        )
    wrist_cam = mount_wrist_camera(
        arm.session, prim_path=wcam.prim_path,
        offset_mm=wcam.mount_offset_mm, aim_target_mm=wcam.mount_aim_mm,
        up_hint=wcam.mount_up_hint, resolution=wcam.resolution,
        near_clip_m=wcam.near_clip_m,
    )

    # CAMERA->TOOL transform: prefer the persisted calibrated hand-eye result from run_eih_calibrate
    # (the real AX=XB calibration); fall back to the empirical GT oracle if it is missing/poor. Using
    # the calibration also skips the flaky oracle depth-seed entirely.
    from src.willy_sim.calibration.paths import calibration_dir

    t_cam_to_tool = _load_calibrated_eih(marker, cal_dir=calibration_dir(sim.robot_model))
    if t_cam_to_tool is not None:
        src_note = f"CALIBRATED({marker})"
    else:
        # Ground-truth-oracle fallback: the wrist cam must see the table here; flaky depth is retried.
        arm.move(init_viewpose(sim.robot_model))
        arm.session.step_n(15)
        app = getattr(arm.session, "app", None)

        def _pump(n: int) -> None:
            for _ in range(n):
                arm.session.step(render=True)
                if app is not None:
                    app.update()

        warmup = int(sim.scene_setup.render_warmup_steps)
        fit_rmse = None
        for attempt in range(6):
            _pump(warmup if attempt == 0 else 12)
            try:
                t_cam_to_tool, fit_rmse = camera_to_tcp_ground_truth(wrist_cam, arm)
                break
            except Exception as exc:  # noqa: BLE001 (depth not ready yet; pump + retry)
                if attempt == 5:
                    raise
                print(f"oracle seed retry {attempt + 1}/5 (wrist depth not ready: {exc})", flush=True)
        src_note = f"GT-ORACLE(fit rmse {fit_rmse:.3f} mm)"
    print(f"T_cam_to_TCP [{src_note}]; mount {np.round(np.asarray(t_cam_to_tool.to_matrix())[:3, 3], 1)}", flush=True)

    # Top-down view pose, closed form via topdown_view_pose. C is the coarse-localized object
    # centroid when the dual-cam coordinator passes coarse_object_pos_mm, else the sim object
    # position. The offset wrist camera then looks straight down over C, the unbiased overhead view
    # that run_m1_pick gets from its fixed camera; p_cam_tool comes from the calibrated or oracle
    # t_cam_to_tool.
    p_cam_tool = np.asarray(t_cam_to_tool.to_matrix(), dtype=np.float64)[:3, 3]
    obj_pos = np.asarray(coarse_object_pos_mm if coarse_object_pos_mm is not None
                         else sim.scene_setup.object.position_mm, dtype=np.float64)
    view_pose = topdown_view_pose(obj_pos, p_cam_tool, view_height_mm)
    print(f"top-down view TCP mm={np.round(np.asarray(view_pose.position_mm), 1)} -> wrist cam over "
          f"{np.round(obj_pos[:2], 1)} at +{view_height_mm:.0f} mm", flush=True)

    if clear_any_object and not multi_object_perception:
        raise SystemExit(
            "clear_any_object needs multi_object_perception: a single-mask frame carries one "
            "segmentation, so there is nothing to choose between and no bin to clear."
        )
    _target_prim = wrist_target_prim or handles.object_prim_path
    _target_label: str | None = None
    if multi_object_perception and not clear_any_object:
        _labels_by_prim = {prim: label for prim, label in handles.object_specs}
        if _target_prim not in _labels_by_prim:
            # Fail closed. An unlabelled multi-object frame is the one configuration that can lift
            # the wrong object and report success, so refuse to build rather than fall back to a
            # target-less frame or silently to the single-mask source.
            raise SystemExit(
                f"multi_object_perception: target prim {_target_prim!r} is not among the authored "
                f"objects {sorted(_labels_by_prim)}. Without its label the pick loop would treat every "
                "segmentation as an executable target and could lift a distractor."
            )
        _target_label = _labels_by_prim[_target_prim]
    perception: object
    if multi_object_perception:
        perception = MultiObjectGroundTruthPerceptionSource(
            camera=wrist_cam, targets=list(handles.object_specs),
            session=arm.session, warmup_steps=sim.scene_setup.render_warmup_steps,
            # The same two knobs with the same meaning, applied per object over disjoint masks. See
            # the single-object call below for what `ground_truth_depth` costs.
            ground_truth_depth=ground_truth_depth,
            grasp_lift_mm=grasp_lift_mm,
        )
    else:
        perception = GroundTruthPerceptionSource(
            camera=wrist_cam, target_prim_path=_target_prim,
            session=arm.session, warmup_steps=sim.scene_setup.render_warmup_steps,
            # Which depth the pick sees. True, the default, overwrites the rendered depth over the
            # object mask with a flat sheet at the object's centre, so the object is perceived as
            # shorter than it is and every geometric check downstream reasons about a body that is
            # not there. False uses the camera's own distance_to_image_plane, which is what an
            # RGB-D sensor delivers on real hardware.
            #
            # It is not a free switch: rendered depth over a mask can pick up support-plane pixels
            # on a short object and drag the grasp to its base. That is a perception problem the
            # flat sheet hides, so the default stays True until each gate has been re-measured
            # through the rendered path.
            ground_truth_depth=ground_truth_depth,
            grasp_lift_mm=grasp_lift_mm,
        )
    # The calculator's owner turns the per-candidate corridor-risk producer on from config
    # (uncertainty.rerank_enabled) at construction: no post-build mutation, and the camera_matrix
    # lives here rather than in from_robot_config. The producer is dense-only, so it is inert in the
    # easy and auto runners, where rerank_enabled defaults to False and the mode skips the re-rank.
    # A dense-clutter runner sets rerank_enabled=true to make it fire.
    _unc_cfg = getattr(getattr(cell.robot, "grasping", None), "uncertainty", None)
    _ik = None
    _ik_banner = ""
    if ik_service:
        from src.robot.execution.ik_service import RobotArmIKService

        # A live handle, not a config value, so the constructor is the right home for it here,
        # unlike the config-driven knobs that must arrive per call from the orchestrator.
        _ik = RobotArmIKService(arm=arm)
        # Printed only after the build succeeds. Under `calculator: deep` the factory refuses an
        # active `ik_service`, so announcing the filter here would be a line claiming a filter that
        # the very next statement declines to install.
        _ik_banner = ("[ik] reachability filter ON (RobotArmIKService on the live sim arm): "
                      "candidates that fail IK are now discarded, and feasibility can re-rank")
    calculator = build_calculator(
        cell.robot,
        camera_matrix=np.asarray(wrist_cam.get_intrinsics_matrix(), dtype=np.float64),
        max_grip_width_mm=cell.robot.gripper.max_width_mm,
        min_grip_width_mm=cell.robot.gripper.min_width_mm,
        corridor_risk_per_candidate=bool(getattr(_unc_cfg, "rerank_enabled", False)),
        ik_service=_ik,
    )
    if _ik_banner:
        print(_ik_banner, flush=True)
    resolver = EyeInHandFrameResolver(t_cam_to_tool=t_cam_to_tool)
    policy = GraspExecutionPolicy(
        arm=arm, gripper=gripper, standoff_mm=standoff_mm, retreat_mm=retreat_mm,
        pre_open_width_mm=pre_open_width_mm, close_width_mm=close_width_mm,
        require_base_frame_grasp=True,
        # Gate every commanded move on the dwell steady-state config; the sim arm settles fast.
        require_steady_before_motion=(_dwell.require_steady_before_motion if _dwell else False),
        steady_timeout_s=(_dwell.steady_timeout_s if _dwell else 5.0),
    )
    # One mode knob, resolved through harness/modes.py. Easy is the deterministic open-loop path.
    # Auto must be built in auto mode, because the per-call override guard refuses
    # easy_service.pick(mode='auto') when the sampling_mode differs; it wires the real
    # DecisionEngine as a single perceive-then-decide tick with no viewpoint_planner, so no
    # MOVE_CAMERA is issued and the grasp and motion path after GRASP_NOW is identical to easy.
    # CLOSED_LOOP wires the refiner and the verifier, both SafetyPreflight-gated, with a fail-open
    # advisory width.
    #   Closed loop is wired but not viable on this camera: the wrist camera moves between the two
    #   refine scans. The viewpoint-invariant WorldSpacePoseTracker
    #   (WILLY_C1_WORLD_SPACE_TRACKER=1) re-identifies the target by 3D base-frame pose where the
    #   image-space IoU tracker drops out, but at the default 275 mm refine standoff the object
    #   leaves the wrist-camera FOV entirely, the instance-id mask is empty, and no tracker has
    #   anything to match. A smaller standoff (WILLY_C1_STANDOFF_MM) keeps it in view
    #   intermittently, but eye-in-hand execution then fails. A viable eye-in-hand CLOSED_LOOP
    #   needs the world-space tracker, a re-perceive standoff that keeps the target visible, and an
    #   execution fix.
    grasp_mode = resolve_demo_mode(mode)
    is_auto = grasp_mode is GraspMode.AUTO
    # Opt into the viewpoint-invariant WorldSpacePoseTracker: the wrist camera moves between scans,
    # so the image-space IoU tracker drops below IoU 0.1 and reports TARGET_LOST. The refine
    # standoff defaults to the wrist view height; a smaller standoff keeps the object inside the
    # moving wrist camera's FOV, while a large standoff empties the instance-id mask and leaves
    # nothing to track.
    env = RunnerEnv.from_env(vision=False, view_height_mm=view_height_mm)  # WILLY_C1_* knobs
    _c1_world_space = env.c1_world_space_tracker
    _c1_standoff = env.c1_standoff_mm
    _mode_kwargs = mode_service_kwargs(grasp_mode, refinement_kwargs={
        "standoff_mm": _c1_standoff, "use_world_space_tracker": _c1_world_space,
    })
    if config_subpolicies:
        if not service_from_config:
            raise SystemExit(
                "config_subpolicies needs service_from_config=True (--boot config): from_components has "
                "no config to read the sub-policies from, so suppressing the runner's would leave none."
            )
        if _mode_kwargs:
            print(f"[config-subpolicies] runner sub-policies suppressed ({sorted(_mode_kwargs)}): "
                  f"grasping.decision/closed_loop/verification are the only source", flush=True)
        _mode_kwargs = {}
    if service_from_config:
        # A sim measurement is only a statement about a real cell if the sim boots the way a real
        # cell boots. The arm and the gripper must be passed in: `SimulationApp` is a process
        # singleton (see `IsaacSimSession` in `drivers/sim/session.py`) and `IsaacRobotArm` mints
        # its session in its own constructor, so a config-built arm would carry a second session on
        # no stage, and `create_arm(RobotVendor.SIM)` refuses every kwarg but `config`, so it could
        # not receive the fail-closed `safety_preflight` either. Both are live handles, not data.
        # Passing them lets every other decision come from the config tree.
        service = AutonomousGraspService.from_robot_config(
            cell.robot, calculator=calculator, perception=perception,
            mode=grasp_mode, frame_resolver=resolver, policy=policy, max_attempts=3,
            arm=arm, gripper=gripper, **_mode_kwargs,
        )
    else:
        service = AutonomousGraspService.from_components(
            arm=arm, calculator=calculator, perception=perception, gripper=gripper,
            mode=grasp_mode, frame_resolver=resolver, policy=policy, max_attempts=3,
            **_mode_kwargs,
        )

    # A bare from_components leaves effective_config=None, which silences the uncertainty, watchdog
    # and latency-SLO telemetry and the orchestrator overlays. Wire them on the executed pick
    # exactly as from_robot_config does, so that autonomy is active and reported rather than
    # bypassed. Guarded and lazy: it must never break the deterministic easy grasp or a mock-mode
    # boot.
    try:
        from src.robot.execution.autonomous_grasp.builders import (
            apply_orchestrator_overlays,
            build_effective_config,
        )

        grasping_cfg = getattr(cell.robot, "grasping", None)
        if service_from_config:
            # from_robot_config already did both, unconditionally. Re-running them here would apply
            # the overlays twice and drag the auto special case below along with it.
            pass
        elif grasping_cfg is not None:
            service.effective_config = build_effective_config(
                grasping_cfg, resolved_mode=grasp_mode, resolved_max_attempts=3,
            )
            # The orchestrator overlays wire multi-view fusion, whose BASE-frame voxel occupancy
            # accumulates across pick() calls on the shared orchestrator. That is correct for
            # multiple views of one scene and wrong across the independent reset-picks this runner
            # makes: in auto mode the grid fills up over successive picks and starves candidate
            # generation, which reports no_candidates. So apply the overlays only outside auto; the
            # auto decision gate does not need them, because it is wired through decision_engine.
            if not is_auto:
                apply_orchestrator_overlays(service.runtime, grasping_cfg, resolved_mode=grasp_mode)
            print(f"autonomy showcase: effective_config wired (mode={grasp_mode.value}); "
                  f"orchestrator overlays {'wired (U2/U4/U5/U6)' if not is_auto else 'SKIPPED in AUTO (fusion accumulates across reset-picks)'} "
                  "+ U8/U9/U10 telemetry",
                  flush=True)
    except Exception as exc:  # noqa: BLE001 (the showcase must never break the executed pick)
        print(f"autonomy showcase skipped ({exc})", flush=True)

    # The half that makes the multi-object frame safe. N segmentations with no label are N
    # executable targets; the loop would rank across all of them and could lift a distractor while
    # reporting a successful pick. With the label set, exactly one segmentation is executable and
    # the rest become ``other_object_masks``: neighbour clutter for the dense sampler, the corridor
    # analyzer and the approach validator. This is part of the switch above, not an addition to it.
    if _target_label is not None:
        _orch = getattr(getattr(service, "runtime", None), "orchestrator", None)
        if _orch is None:
            raise SystemExit(
                "multi_object_perception: the service exposes no orchestrator, so the target label "
                f"{_target_label!r} cannot be set. Refusing to run a multi-object frame in which every "
                "segmentation would be an executable target."
            )
        _orch.target_label = _target_label
        print(f"perception: MULTI-OBJECT ({len(handles.object_specs)} labelled masks) -> "
              f"target_label={_target_label!r}; the other {len(handles.object_specs) - 1} feed "
              "other_object_masks (clutter/corridor/approach)", flush=True)
    elif clear_any_object:
        print(f"perception: MULTI-OBJECT ({len(handles.object_specs)} labelled masks) -> NO "
              "target_label (bin-clearing: every object is a legitimate target, so the loop; and "
              "grasping.ordering; chooses which one goes first)", flush=True)
    else:
        print(f"perception: SINGLE-MASK ({_target_prim}) -> other_object_masks is empty; the corridor "
              "analyzer and the approach validator have no neighbour to see", flush=True)

    _print_boot_path(service, arm=arm, gripper=gripper, from_config=service_from_config)
    return service, arm, gripper, handles, cfg, view_pose, cell


def _print_boot_path(service, *, arm, gripper, from_config: bool) -> None:  # noqa: ANN001
    """State which path built this service and check the four ways it can look right and be wrong.

    Each of these produces a cell that runs, reports success and measures nothing:

    * no ``effective_config``: the config path did not engage, so every number still describes the
      hand-wired stack;
    * a substituted gripper: ``gripper.vendor: robotiq`` on a ``sim`` arm becomes a NullGripper
      whose jaws close on nothing while every pick reports success;
    * an arm that is not the one holding the live session: a second ``SimulationApp`` drives prims
      no camera is looking at, and nothing crashes;
    * ``mode_label`` outside ``_approach_path_modes``: the swept-volume validator reads as enabled
      and never runs, because the config default is ``(dense_clutter, dense_autonomous)`` while
      this runner is ``easy``.
    """

    orch = getattr(getattr(service, "runtime", None), "orchestrator", None)
    substitution = getattr(gripper, "substitution", None)
    approach_modes = getattr(orch, "_approach_path_modes", None)
    mode_label = getattr(orch, "mode_label", None)
    print(
        f"boot path: {'from_robot_config (CONFIG)' if from_config else 'from_components (RUNNER)'} | "
        f"effective_config={'set' if getattr(service, 'effective_config', None) is not None else 'NONE'} | "
        f"mode_label={mode_label!r} | gripper={type(gripper).__name__}"
        f"{' SUBSTITUTED=' + str(substitution.reason) if substitution is not None else ''} | "
        f"arm_is_live_handle={getattr(orch, 'arm', None) is arm} | "
        f"scene_fusion={'set' if getattr(orch, 'scene_fusion', None) is not None else 'None'}",
        flush=True,
    )
    if substitution is not None:
        print("  WARN: the gripper was SUBSTITUTED: this cell closes on nothing and still "
              "reports success. Every pick number below is meaningless.", flush=True)
    if from_config and getattr(service, "effective_config", None) is None:
        print("  WARN: booted from config but effective_config is None: the grasping block is "
              "absent from the tree, so the overlays did not run.", flush=True)
    if approach_modes and mode_label is not None and mode_label not in approach_modes:
        print(f"  WARN: approach validation is wired for {sorted(approach_modes)} but this run is "
              f"{mode_label!r}: it reads as ON and will never fire.", flush=True)


def _print_autonomy_telemetry(report: object) -> None:
    """Labelled dump of the AutonomousGrasp decision, uncertainty and latency-SLO telemetry."""
    mode = getattr(report, "mode", None)
    profile = getattr(report, "profile", None)
    prof_name = getattr(profile, "sampling_mode", getattr(profile, "name", profile))
    print(f"  [autonomy] mode={getattr(mode, 'value', mode)} profile={prof_name} outcome={getattr(report, 'outcome', None)}", flush=True)
    unc = getattr(report, "uncertainty", None)
    if unc is not None:
        print(f"  [autonomy] uncertainty={unc}", flush=True)
    decision = getattr(report, "decision", None)
    if decision is not None:
        to_dict = getattr(decision, "to_dict", None)
        print(f"  [autonomy] decision={to_dict() if callable(to_dict) else decision}", flush=True)
    tele = getattr(report, "telemetry", None)
    if isinstance(tele, dict) and tele:
        keys = sorted(tele)[:12]
        print(f"  [autonomy] telemetry keys={keys}", flush=True)
        for k in ("latency", "latency_spans", "slo", "stage_latency_ms",
                  "refinement_mode", "refinement_telemetry",
                  "verification_enabled", "verification_telemetry"):
            if k in tele:
                print(f"  [autonomy] {k}={tele[k]}", flush=True)


def run_gate(runs: int = 10, *, headless: bool = True, data_dir: str | None = None,
             mode: str = "easy", marker: str = "ground_truth",
             record_log: str | None = None, debug_frames: str | None = None,
             cell_kwargs: "dict | None" = None,
             # False lets the camera's own rendered depth reach the pick. See build_service.
             ground_truth_depth: bool = True) -> GateResult:
    """Run ``runs`` eye-in-hand picks and score the lift gate.

    Each pick perceives from the view pose, resolves CAMERA->BASE, then grasps.

    ``mode`` is "easy", "auto" (the real ``DecisionEngine``, with the typed ``DecisionReport``
    printed per run) or "closed_loop" (the refiner and the verifier, with per-run refine telemetry;
    wired but not viable on the moving wrist camera, see build_service).

    Both instruments are opt-in and default off: ``record_log`` appends one ``GraspAttemptRecord``
    JSONL line per pick, stamped with the ground-truth ``sim_lift_mm`` and ``sim_lifted``, and
    ``debug_frames`` dumps the grasp-point overlay PNG of the wrist-camera frame per pick. With
    neither set the run is byte-identical.
    """
    _m = str(mode).lower().replace("-", "_")
    is_auto, is_cl = _m == "auto", _m == "closed_loop"
    if is_auto:
        print("=== EIH AUTO decision-gate showcase (perceive -> rank -> DecisionEngine -> grasp) ===", flush=True)
        print("    NOTE: the Isaac arm reports is_simulated=True, so the gate is PERMISSIVE: it "
              "exercises the\n    decide control flow + typed DecisionReport, not a hardware fail-closure "
              "(that needs real hw).", flush=True)
    if is_cl:
        print("=== EIH CLOSED_LOOP showcase (perceive -> grasp -> STANDOFF -> re-perceive -> REFINE -> "
              "execute -> VERIFY) ===", flush=True)
        print("    Measuring whether the second-scan refine changes the executed grasp (position_delta) "
              "and holds the gate.", flush=True)
    service, arm, gripper, handles, cfg, view_pose, cell = build_service(
        headless=headless, data_dir=data_dir, mode=mode, marker=marker, cell_kwargs=cell_kwargs,
        ground_truth_depth=ground_truth_depth,
    )
    identity = cell_identity(cell)
    if debug_frames:  # render the grasp-point overlay per pick (eye-in-hand ground-truth perception
        service.enable_debug_image_rendering()  # has no usable rgb on Isaac 5.1, so no frame; vision does)
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    gate = require_robot(cfg).sim.scene_setup.gate
    obj = SingleRigidPrim(handles.object_prim_path)
    home_pos, home_quat = obj.get_world_pose()
    run_id = f"eih-{int(time.time())}"
    results = []
    for i in range(runs):
        # Reset: release jaw, move to the wrist-cam view pose, re-place + settle the object.
        gripper.open()
        arm.move(view_pose)
        z0 = reset_object_to_home_z0(obj, home_pos, home_quat, arm.session, settle_steps=30)
        report = service.pick()
        if is_auto:  # surface the real DecisionEngine verdict for this pick
            decision = getattr(report, "decision", None)
            if decision is not None and hasattr(decision, "to_dict"):
                d = decision.to_dict()
                print(f"  [AUTO] action={d.get('decision_action')} reason={d.get('decision_reason_code')} "
                      f"uncertainty={d.get('uncertainty_score')} top_score={d.get('top_score')} "
                      f"threshold={d.get('threshold_used')} reobs={d.get('reobservation_count')} "
                      f"src={d.get('uncertainty_source')}", flush=True)
            else:
                print(f"  [AUTO] no DecisionReport on report (outcome={getattr(report, 'outcome', None)})", flush=True)
        if i == 0:  # first run only: report where the resolved grasp landed vs the true position
            pr = getattr(report, "pick_report", None)
            eg = getattr(pr, "executed_grasp", None)
            if eg is not None and eg.candidates:
                ti = (pr.target_index or 0) if pr is not None else 0
                g = eg.candidates[ti] if ti < len(eg.candidates) else eg.candidates[0]
                gp = np.asarray(g.position, dtype=np.float64)
                obj_true = np.asarray(home_pos, dtype=np.float64) * 1000.0
                d_xy = float(np.linalg.norm((gp - obj_true)[:2]))
                print(f"  grasp pos_mm={np.round(gp, 1)} frame={g.frame} | object_true_mm={np.round(obj_true, 1)} "
                      f"| on-target XY delta={d_xy:.1f} mm", flush=True)
            _print_autonomy_telemetry(report)
        lift_mm = lift_mm_since(obj, z0)
        outcome = getattr(report, "outcome", None)
        succeeded = getattr(outcome, "value", outcome) == "succeeded"
        passed = bool(lift_mm >= gate.lift_threshold_mm)
        results.append({"run": i, "succeeded": succeeded, "lift_mm": round(lift_mm, 1), "passed": passed})
        print(f"RUN {i}: succeeded={succeeded} lift_mm={lift_mm:.1f} passed(lifted)={passed}", flush=True)
        write_pick_artifacts(
            report, attempt_id=f"{run_id}-{i:04d}",
            record_log=record_log, debug_dir=debug_frames,
            debug_png=service.last_debug_image_png,
            extra={**identity, "sim_runner": "eih", "sim_mode": mode, "sim_auto": is_auto,
                   "sim_closed_loop": is_cl, "sim_lift_mm": round(lift_mm, 2), "sim_lifted": bool(passed),
                   "sim_gate_passed": passed},
        )
    n_pass = sum(1 for r in results if r["passed"])
    gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
    print(f"EIH PICK GATE: {n_pass}/{runs} passed -> gate_passed={gate_ok}", flush=True)
    return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy EIH wrist-camera pick (Isaac).")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-headless", action="store_true", help="alias for --gui")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--mode", type=str, default="easy", choices=["easy", "auto", "closed_loop"],
                    help="P1: grasp mode (easy=10/10 trust path; auto=real DecisionEngine; closed_loop=S3+S4)")
    ap.add_argument("--demo-auto", action="store_true", help="deprecated alias for --mode auto")
    ap.add_argument("--closed-loop", action="store_true", help="deprecated alias for --mode closed_loop")
    ap.add_argument("--marker", type=str, default="ground_truth", choices=["ground_truth", "aruco"],
                    help="which CALIBRATED EIH artifact to drive the resolver (logs/calibration/eih_wrist_cam_<marker>.json); "
                         "falls back to the GT oracle if missing/poor")
    ap.add_argument("--record-log", type=str, default=None,
                    help="P0: append one GraspAttemptRecord JSONL line per pick to this path "
                         "(the soak/KPI/RL data source; default off -> byte-identical)")
    ap.add_argument("--debug-frames", type=str, default=None,
                    help="P0: dump the grasp-point overlay PNG per pick into this dir")
    ap.add_argument("--result-json", type=str, default=None,
                    help="P1: write the per-run result dict (for the mode-matrix harness) to this path")
    add_cell_arguments(ap)   # --robot-model / --profile so a UR3e cell runs from the same runner
    ap.add_argument("--rendered-depth", action="store_true",
                    help="let the camera's OWN rendered depth reach the pick instead of the flat "
                         "ground-truth sheet at the object's centre: the September-shaped path")
    args = ap.parse_args()
    mode = "auto" if args.demo_auto else "closed_loop" if args.closed_loop else args.mode
    result = run_gate(runs=args.runs, headless=not (args.gui or args.no_headless), data_dir=args.data_dir,
                      mode=mode, marker=args.marker,
                      record_log=args.record_log, debug_frames=args.debug_frames,
                      cell_kwargs=cell_profile_kwargs(args),
                      ground_truth_depth=not args.rendered_depth)
    write_run_result(args.result_json, result.to_dict(), scene="eih", mode=mode, marker=args.marker)


if __name__ == "__main__":
    main()
