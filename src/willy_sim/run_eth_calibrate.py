"""Eye-to-hand overhead-camera calibration runner for Isaac.

The fixed overhead camera is calibrated against a marker carried on the moving tool: the arm is
driven through diverse, workspace-gated poses (``PoseProvider`` via
``CalibrationRoutine.run_auto``), the overhead camera observes the tool marker at each pose, and
``EyeToHandCalibrator`` solves ``T_cam_to_base`` through AX=XB. The result is validated against the
empirical CV-optical overhead extrinsic fitted by ``camera_to_base_ground_truth``, not against the
hand-built ``handles.camera_to_base``, and saved as a schema-versioned ``Extrinsics`` artifact.

The marker source is swappable:
  * ``ground_truth``: the tool frame itself is the marker (T_cam_to_marker = inv(oracle) @ FK), so
    no physical board is needed; it exercises the collection and AX=XB machinery.
  * ``aruco``: a rendered ArUco board carried on the wrist and detected in the overhead RGB, which
    is genuine perception.

Run with Isaac's bundled python:
    <isaac-sim>\\python.bat -m src.willy_sim.run_eth_calibrate --marker ground_truth
    ...\\python.bat -m src.willy_sim.run_eth_calibrate --marker aruco --gui
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from src.utility.log_cfg import create_logger
from src.willy_sim.constants import CALIBRATION_RUNS_LOG_FILE, WILLY_SIM_LOG_DIR
from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.harness.cli import add_cell_arguments, cell_profile_kwargs
from src.willy_sim.scene import (
    camera_to_base_ground_truth,
    mount_tool_aruco_marker,
)
from src.geometry.matrix import invert_homogeneous

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.calibration.quality import QualityBandsMm
    from src.geometry import Transform

DEFAULT_SAVE_DIR = "logs/calibration"

#: The extrinsic this writes is trusted by every later pick on this cell, and the artifact records
#: the result without the conditions: which marker source, how many of the planned poses survived,
#: and whether the gate passed at all, since a CHECK is saved exactly like a PASS. Shares its file
#: with the eye-in-hand runner, because the two halves of a session are read together.
_LOG = create_logger("SimEthCalibrate", log_file=CALIBRATION_RUNS_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)


def _resolve_eth_camera(cell, sim, camera_id: str):
    """Resolve the Isaac camera to calibrate.

    ``overhead`` is the scene's built-in overhead camera. Any other id is authored as a fixed
    camera from ``robot.sim.cameras[id]`` (position and aim), so each camera of a multi-view rig
    can be calibrated individually under its own id.
    """
    if camera_id == "overhead":
        return cell.handles.camera
    if camera_id not in sim.cameras:
        raise SystemExit(f"--camera {camera_id!r} not in robot.sim.cameras {sorted(sim.cameras)}")
    ccfg = sim.cameras[camera_id]
    if ccfg.mounting_mode != "eye_to_hand":
        raise SystemExit(f"--camera {camera_id!r} is {ccfg.mounting_mode}, not eye_to_hand "
                         "(use run_eih_calibrate for a wrist camera)")
    from src.willy_sim.scene import author_fixed_camera

    return author_fixed_camera(
        cell.arm.session, ccfg.prim_path,
        position_mm=tuple(ccfg.position_mm), aim_mm=tuple(ccfg.mount_aim_mm),
        resolution=tuple(ccfg.resolution), near_clip_m=float(ccfg.near_clip_m),
    )


def calibrate(*, headless: bool = True, marker: str = "ground_truth", camera_id: str = "overhead",
              data_dir: str | None = None, n_poses: int = 22, save_dir: str = DEFAULT_SAVE_DIR,
              cell_kwargs: dict | None = None,
              # Pose-box levers. The defaults reproduce the shipped behaviour exactly, so they are
              # experiment handles for a cell whose arm cannot work the default box, not policy.
              z_frac: tuple[float, float] | None = None,    # z band as a fraction of z_max
              #   None means (0.55, 0.75) for aruco, which keeps the marker in the narrow overhead
              #   FOV, and the full range for ground_truth, which reads FK and never sees an image.
              xy_mm: float = 50.0,                          # aruco: +-half-extent under the camera
              reach_frac: float = 0.0):                     # >0: also clip the box to this share of reach
    """Run the eye-to-hand calibration for one named camera end-to-end via CalibrationRoutine.

    ``camera_id`` names the camera (``overhead`` = the built-in overhead; any other = a fixed camera
    from ``robot.sim.cameras``). The artifact is saved keyed by camera id (``eth_<camera>.json``,
    ``rig_id=<camera>``) so a multi-view rig's per-camera calibrations coexist + drop into the central
    ``grasping.fusion.cameras`` map.
    """
    import dataclasses

    from src.calibration.eye_hand.types import MountingMode
    from src.calibration.quality import classify_rmse
    from src.calibration.serialization import save_extrinsics
    from src.robot.execution import CalibrationRoutine

    # The shared boot prefix, with no fixed marker: the eye-to-hand marker rides the tool.
    cell = bootstrap_sim_cell(data_dir, headless=headless, **(cell_kwargs or {}))
    arm, gripper, sim = cell.arm, cell.gripper, cell.sim
    # Namespace the artifacts per robot_model so a UR3e run cannot overwrite the UR5e's.
    if save_dir == DEFAULT_SAVE_DIR:
        from src.willy_sim.calibration.paths import calibration_dir

        save_dir = str(calibration_dir(sim.robot_model))
    he = cell.cfg.camera.hand_eye.eye_to_hand     # ETH marker length + dict + sample thresholds
    cal = cell.robot.calibration
    gripper.open()

    overhead = _resolve_eth_camera(cell, sim, camera_id)
    app = getattr(arm.session, "app", None)
    warmup = int(sim.scene_setup.render_warmup_steps)

    def _pump(n: int) -> None:
        for _ in range(n):
            arm.session.step(render=True)
            if app is not None:
                app.update()

    # The reference convention is the CV-optical overhead extrinsic from the empirical Umeyama fit.
    # That is the convention cv2-ArUco's solvePnP produces and EyeToHandCalibrator recovers, and the
    # one the eye-in-hand runner validates against. Do not validate against the hand-built grasp
    # transform in handles.camera_to_base: it is deliberately yawed to dodge the wrist singularity,
    # so the ArUco result reads 180 deg off. The fit needs near-clip 0.05, because the overhead
    # default of 1.0 m clips the table and leaves no depth.
    arm.session.step_n(5)
    try:
        overhead.add_distance_to_image_plane_to_frame()  # build_combined_scene doesn't add it to the overhead
    except Exception:  # noqa: BLE001
        pass
    try:
        overhead.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    _LOG.info(
        "ETH calibration starting: camera=%s marker=%s n_poses=%d robot=%s save_dir=%s",
        camera_id, marker, n_poses, sim.robot_model, save_dir,
    )
    overhead_oracle: Transform | None = None
    for attempt in range(6):
        _pump(warmup if attempt == 0 else 12)
        try:
            overhead_oracle, _fit_rmse = camera_to_base_ground_truth(overhead)
            break
        except Exception as exc:  # noqa: BLE001 (depth not ready yet; pump + retry)
            if attempt == 5:
                raise
            # The reference the whole run is validated against needed extra render pumps. Harmless
            # once, but a run that took five is a rendering problem the result will not mention.
            _LOG.warning("overhead oracle fit retry %d/5: %s", attempt + 1, exc)
            print(f"overhead oracle retry {attempt + 1}/5 ({exc})", flush=True)
    if overhead_oracle is None:  # the loop sets it or re-raises on the final attempt; this narrows the type
        raise RuntimeError("overhead oracle fit did not converge")

    print("\n========== ETH OVERHEAD CALIBRATION (CalibrationRoutine) ==========", flush=True)
    print(f"overhead oracle (TRUE CV-optical) T_cam_to_BASE mm: {np.round(np.asarray(overhead_oracle.to_matrix())[:3, 3], 1)}", flush=True)

    if marker == "ground_truth":
        oracle_inv = invert_homogeneous(np.asarray(overhead_oracle.to_matrix(), dtype=np.float64))

        def marker_source():
            # The tool frame is the marker: T_cam_to_marker = inv(T_cam_to_base) @ FK(tool-in-base).
            base_t_tool = np.asarray(arm.get_tcp_pose().to_matrix(), dtype=np.float64)
            return oracle_inv @ base_t_tool
    elif marker == "aruco":
        wrist_link = sim.cameras["wrist"].prim_path.rsplit("/", 1)[0]  # /World/UR5e/wrist_3_link
        mount_tool_aruco_marker(
            arm.session, parent_prim=wrist_link,
            length_mm=he.marker_length_mm, dict_name=he.aruco_dict_name,
            marker_id=sim.scene_setup.marker.aruco_marker_id,
        )
        # (overhead near-clip 0.05 already set above for the oracle fit; the tool marker sits ~0.4 m away.)
        from src.willy_sim.calibration.hand_eye import ArucoMarkerPoseSource

        src = ArucoMarkerPoseSource(
            marker_id=sim.scene_setup.marker.aruco_marker_id,
            marker_length_mm=he.marker_length_mm, dict_name=he.aruco_dict_name,
        )

        def marker_source():
            for attempt in range(4):
                _pump(warmup if attempt == 0 else 10)
                t = src.marker_in_camera(overhead)
                if t is not None:
                    return np.asarray(t.to_matrix(), dtype=np.float64)
            return None
    else:
        raise SystemExit(f"unknown marker source {marker!r}")

    # ArUco needs the tool marker to stay inside the narrow overhead FOV, about 167 mm across at the
    # marker height, so tighten the pose box around the workspace centre. The ground-truth source
    # works over the full box.
    #
    # The box is derived from the cell, not hardcoded: a +-50 mm column directly under the camera in
    # the upper part of the reachable z, which on the UR5e evaluates to x 400-500, y +-50,
    # z 330-450. It has to be derived because those values lie wholly outside a UR3e's box
    # (x_max 400, z_max 320), where a hardcoded box would collect zero samples.
    ws = cell.robot.workspace_limits
    if marker == "aruco":
        ccfg = sim.cameras.get(camera_id) or sim.cameras.get("overhead")
        cam_pos = ccfg.position_mm if ccfg is not None else None   # Optional in the schema
        cx, cy = (float(cam_pos[0]), float(cam_pos[1])) if cam_pos is not None else (450.0, 0.0)
        ws = ws.model_copy(update={
            "x_min": max(ws.x_min, cx - xy_mm), "x_max": min(ws.x_max, cx + xy_mm),
            "y_min": max(ws.y_min, cy - xy_mm), "y_max": min(ws.y_max, cy + xy_mm),
        })
    # precedence: --z-frac (experiment) > robot.calibration.pose_box_z_frac (the cell states its own
    # arm property) > the shipped per-marker default.
    band = (z_frac or (cal.pose_box_z_frac or {}).get(marker)
            or ((0.55, 0.75) if marker == "aruco" else (0.0, 1.0)))
    if band != (0.0, 1.0):
        ws = ws.model_copy(update={"z_min": band[0] * ws.z_max, "z_max": band[1] * ws.z_max})
    if reach_frac > 0.0:
        # Clip the box to the arm's dexterous sphere. The ground-truth run samples the raw workspace
        # box, and on a UR3e, whose box corner sits at 96% of reach, poses come back status=timeout
        # on the near-singular failure this cell already documents for its park pose, leaving too
        # few samples to solve. Off by default (reach_frac=0), which leaves the UR5e path untouched.
        from src.willy_sim.harness.reach import shoulder_height_mm, ur_model_spec

        shoulder = shoulder_height_mm(sim.robot_model)
        limit = ur_model_spec(sim.robot_model).max_reach_mm * reach_frac
        # worst |z - shoulder| over the band decides how much x/y the sphere still allows
        dz = max(abs(ws.z_min - shoulder), abs(ws.z_max - shoulder))
        half = float(np.sqrt(max(0.0, limit * limit - dz * dz)))
        ws = ws.model_copy(update={
            "x_min": max(ws.x_min, -half), "x_max": min(ws.x_max, half),
            "y_min": max(ws.y_min, -half), "y_max": min(ws.y_max, half),
        })
    if marker == "aruco" or reach_frac > 0.0 or band != (0.0, 1.0):
        print(f"[eth] pose box (marker={marker}, derived from camera {camera_id!r} + cell box"
              f"{f', reach<={reach_frac:.0%}' if reach_frac > 0 else ''}): "
              f"x[{ws.x_min:.0f}..{ws.x_max:.0f}] y[{ws.y_min:.0f}..{ws.y_max:.0f}] "
              f"z[{ws.z_min:.0f}..{ws.z_max:.0f}]", flush=True)

    routine = CalibrationRoutine(
        arm=arm, marker_source=marker_source,
        workspace_limits=ws, eth_settings=he,
        marker_id=sim.scene_setup.marker.aruco_marker_id,
        settle_time_s=cal.settle_time_s,
        calibration_mode=MountingMode.EYE_TO_HAND,
        on_event=lambda evt, data: print(f"  [{evt}] {data.get('reason', data.get('accepted', ''))}", flush=True),
    )
    # ArUco needs oblique views to break the planar-marker IPPE flip ambiguity that ruins the AX=XB
    # rotation when the overhead sees the up-facing marker near-frontally. The ground-truth source
    # is exact, so a small spread is enough for it.
    spread = 30.0 if marker == "aruco" else cal.orientation_spread_deg
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    result = routine.run_auto(
        n_poses, base_orientation=[float(np.pi), 0.0, 0.0],  # tool down, so the marker faces the cam
        orientation_spread_deg=spread,
        max_attempts_per_pose=cal.max_attempts_per_pose, seed=0,
        dataset_save_path=f"{save_dir}/eth_{camera_id}_dataset_{marker}.json",
    )

    from src.willy_sim.calibration.hand_eye import transform_delta

    d_t, d_r = transform_delta(result.transform, overhead_oracle)
    # cal.quality_bands_mm is the config-schema RobotCalibrationQualityBandsMm, structurally
    # identical to QualityBandsMm: the excellent, good and marginal floats are all classify_rmse
    # reads, so the cast is a typing narrowing.
    quality = str(classify_rmse(result.rmse_mm, cast("QualityBandsMm", cal.quality_bands_mm)))
    gate_mm = 1.0 if marker == "ground_truth" else 8.0
    # Accuracy is bounded by d_t + d_r (vs the true CV-optical oracle). The AX=XB rmse band: GT must be
    # good+ (it is exact); a small ArUco tool marker viewed obliquely from the overhead is realistically
    # ~3 mm rmse (marginal), so accept marginal-but-not-poor for the fiducial path.
    ok_quality = quality in (("good", "excellent") if marker == "ground_truth"
                             else ("good", "excellent", "marginal"))
    ok = d_t < gate_mm and d_r < 1.5 and ok_quality
    print("\n--- RESULT ---", flush=True)
    print(f"  accepted samples: {result.num_samples}", flush=True)
    print(f"  calibrated T_cam_to_BASE mm: {np.round(np.asarray(result.transform.to_matrix())[:3, 3], 1)}", flush=True)
    print(f"  AX=XB rmse: {result.rmse_mm:.4f} mm | max_error: {result.max_error_mm:.4f} | quality: {quality}", flush=True)
    print(f"  ERROR vs overhead oracle: translation={d_t:.3f} mm, rotation={d_r:.3f} deg", flush=True)
    print(f"  GATE (<{gate_mm:.0f} mm, <1.5 deg, quality good+): {'PASS' if ok else 'CHECK'}", flush=True)
    # The solve itself, once: accepted-vs-planned samples (AX=XB degrades quietly when views were
    # refused), the residual, and the error against the true CV-optical oracle.
    _log_result = _LOG.info if ok else _LOG.warning
    _log_result(
        "ETH %s/%s solved: %d/%d samples accepted, rmse=%.4f mm max_error=%.4f mm quality=%s, "
        "vs oracle dt=%.3f mm dr=%.3f deg -> gate(<%.0f mm, <1.5 deg, good+)=%s",
        camera_id, marker, result.num_samples, n_poses, result.rmse_mm, result.max_error_mm, quality,
        d_t, d_r, gate_mm, "PASS" if ok else "CHECK: the artifact below is saved anyway",
    )

    # Persist keyed by camera id as a schema-versioned Extrinsics (eye-to-hand CAMERA->BASE). rig_id
    # is stamped to the camera id so the artifact and the central grasping.fusion.cameras map agree.
    if result.extrinsics is not None:
        ext = dataclasses.replace(result.extrinsics, rig_id=camera_id)
        ext_path = save_extrinsics(f"{save_dir}/eth_{camera_id}.json", ext)
        _LOG.info("saved extrinsics %s (rig_id=%s) + dataset %s", ext_path, camera_id, result.dataset_path)
        print(f"\n  saved {ext_path} (rig_id={camera_id}) and {result.dataset_path}", flush=True)
    else:
        # Without a carrier nothing lands in grasping.fusion.cameras, so the run looks done and the
        # cell keeps using whatever extrinsic it had. Silent everywhere else.
        _LOG.error(
            "no Extrinsics carrier on the result: eth_%s.json was not written; this camera keeps "
            "its previous calibration", camera_id,
        )
    print("========== END ETH OVERHEAD CALIBRATION ==========", flush=True)

    if not headless:
        arm.session.step_n(900)
    return {"ok": ok, "error_mm": d_t, "error_deg": d_r, "quality": quality}


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy eye-to-hand overhead-camera calibration (Isaac).")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-headless", action="store_true", help="alias for --gui")
    ap.add_argument("--marker", type=str, default="ground_truth", choices=["ground_truth", "aruco"])
    ap.add_argument("--camera", type=str, default="overhead",
                    help="camera id to calibrate: 'overhead' (built-in) or a robot.sim.cameras key "
                         "(authored fixed from its position+aim). Saved as eth_<camera>.json.")
    ap.add_argument("--n-poses", type=int, default=22)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--z-frac", type=str, default=None,
                    help="pose-box z band as LO,HI fractions of workspace z_max (default = shipped per marker)")
    ap.add_argument("--xy-mm", type=float, default=50.0,
                    help="aruco pose-box +-half-extent under the camera, mm (default = shipped)")
    ap.add_argument("--reach-frac", type=float, default=0.0,
                    help="clip the pose box to this share of the arm's reach (0 = off, = shipped)")
    add_cell_arguments(ap)   # --robot-model / --profile (artifacts namespace per robot)
    args = ap.parse_args()
    calibrate(headless=not (args.gui or args.no_headless), marker=args.marker, camera_id=args.camera,
              n_poses=args.n_poses, data_dir=args.data_dir, cell_kwargs=cell_profile_kwargs(args),
              z_frac=(tuple(float(v) for v in args.z_frac.split(",")) if args.z_frac else None),  # type: ignore[arg-type]
              xy_mm=args.xy_mm, reach_frac=args.reach_frac)


if __name__ == "__main__":
    main()
