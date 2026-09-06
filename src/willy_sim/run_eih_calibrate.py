"""Config-driven eye-in-hand wrist-camera calibration runner for Isaac.

Drives the vendor-neutral :class:`src.robot.execution.CalibrationRoutine`, the same loop real
hardware uses: move, settle, read FK, detect the marker, add the sample, solve AX=XB. The marker
source is injected by the sim (a ground-truth prim pose or a rendered-ArUco PnP solve) and the
look-at viewpoints aim the wrist camera at the calibration marker. Every scene, prim and
calibration value comes from the sim config, the ``config`` tree under the ``sim`` profile; nothing
is hardcoded here. The recovered ``T_cam_to_tool`` is validated against the empirical ground-truth
oracle and persisted alongside the schema-versioned dataset and a typed transform dict.

Run with Isaac's bundled python:
    <isaac-sim>\\python.bat -m src.willy_sim.run_eih_calibrate --marker aruco
    ...\\python.bat -m src.willy_sim.run_eih_calibrate --marker ground_truth --gui
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from src.utility.log_cfg import create_logger
from src.willy_sim.constants import CALIBRATION_RUNS_LOG_FILE, WILLY_SIM_LOG_DIR
from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.harness.cli import add_cell_arguments, cell_profile_kwargs
from src.willy_sim.config import load_sim_config
from src.willy_sim.scene import (
    camera_to_tcp_ground_truth,
    mount_wrist_camera,
)
from src.geometry import transform_to_dict

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.calibration.quality import QualityBandsMm

# Initial down-looking pose so the empirical oracle has table geometry to fit; TCP +Z points along
# base -Z. Per-robot, because the UR5e default is out of a UR3e's reach: the seed pose comes from
# the shared per-robot table in run_eih_pick, one source of truth for both the calibration seed and
# the pick's oracle fallback.
from src.willy_sim.run_eih_pick import init_viewpose  # noqa: E402
DEFAULT_SAVE_DIR = "logs/calibration"

#: Same reasoning, and the same file, as the eye-to-hand runner: the wrist transform this solves is
#: trusted by every later eye-in-hand pick, and neither the JSON nor the dataset records how many of
#: the planned viewpoints produced a sample, or that the gate returned CHECK.
_LOG = create_logger("SimEihCalibrate", log_file=CALIBRATION_RUNS_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)


def calibrate(*, headless: bool = True, marker: str = "ground_truth", camera_id: str = "wrist",
              data_dir: str | None = None, save_dir: str = DEFAULT_SAVE_DIR,
              cell_kwargs: dict | None = None):
    """Run the eye-in-hand calibration end-to-end via CalibrationRoutine. Returns a dict report."""
    from src.calibration.eye_hand.types import MountingMode
    from src.calibration.quality import classify_rmse
    from src.robot.execution import CalibrationRoutine

    cfg = load_sim_config(data_dir)
    he = cfg.camera.hand_eye.eye_in_hand          # marker length + dict + sample thresholds
    marker_kind = "aruco" if marker == "aruco" else "flat"
    # The shared boot prefix authors the scene with the hand-eye marker (kind + ArUco params).
    cell = bootstrap_sim_cell(
        data_dir, headless=headless, **(cell_kwargs or {}), scene_kwargs={
            "marker_kind": marker_kind, "aruco_length_mm": he.marker_length_mm,
            "aruco_dict_name": he.aruco_dict_name,
        },
    )
    arm, gripper, handles, sim = cell.arm, cell.gripper, cell.handles, cell.sim
    # Namespace the artifacts per robot_model so a UR3e run cannot overwrite the UR5e's.
    if save_dir == DEFAULT_SAVE_DIR:
        from src.willy_sim.calibration.paths import calibration_dir

        save_dir = str(calibration_dir(sim.robot_model))
    scene = sim.scene_setup
    gripper.open()

    wcam = sim.cameras["wrist"]
    # The wrist camera's mount, resolution and near-clip fields are Optional in the schema but
    # required here; narrow them and raise if a sim tree omits them, which the sim profile never
    # does because it sets all five. The fields are already tuples.
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

    from src.willy_sim.calibration.hand_eye import (
        ArucoMarkerPoseSource,
        GroundTruthMarkerPoseSource,
        MarkerPoseSource,
        generate_hemisphere_viewpoints,
        transform_delta,
    )

    print("\n========== EIH CALIBRATION (CalibrationRoutine) ==========", flush=True)
    # 1) initial empirical oracle (camera must see the table here).
    arm.move(init_viewpose(sim.robot_model))
    arm.session.step_n(15)
    app = getattr(arm.session, "app", None)
    warmup = int(scene.render_warmup_steps)

    def _pump(n: int) -> None:
        for _ in range(n):
            arm.session.step(render=True)
            if app is not None:
                app.update()

    _pump(warmup)
    oracle, fit_rmse = camera_to_tcp_ground_truth(wrist_cam, arm)
    print(f"oracle T_cam_to_TCP mm: {np.round(np.asarray(oracle.to_matrix())[:3, 3], 2)} (fit rmse {fit_rmse:.3f})", flush=True)
    # The oracle is the reference the gate below judges against, so its own fit residual bounds what
    # "error vs oracle" can mean: a 3 mm oracle cannot certify a 1 mm calibration.
    _LOG.info(
        "EIH calibration starting: camera=%s marker=%s robot=%s save_dir=%s, oracle fit rmse=%.3f mm",
        camera_id, marker, sim.robot_model, save_dir, fit_rmse,
    )

    # 2) look-at viewpoints around the marker (from config).
    vp = scene.eih_viewpoints
    marker_pos_mm = np.asarray(scene.marker.position_mm, dtype=np.float64)
    viewpoints = generate_hemisphere_viewpoints(
        marker_pos_mm, oracle,
        radii_mm=vp.radii_mm, elevations_deg=vp.elevations_deg, azimuths_deg=vp.azimuths_deg,
    )
    tcp_poses = [v.tcp_pose for v in viewpoints]
    print(f"planned {len(tcp_poses)} viewpoints around marker {np.round(marker_pos_mm, 1)}", flush=True)

    # 3) marker source (swappable perception seam) wrapped as a render-pumping MarkerPoseProvider.
    src: MarkerPoseSource
    if marker == "ground_truth":
        if handles.marker_prim_path is None:
            raise RuntimeError("ground-truth EIH calibration requires an authored marker prim.")
        src = GroundTruthMarkerPoseSource(marker_prim_path=handles.marker_prim_path)
    elif marker == "aruco":
        src = ArucoMarkerPoseSource(
            marker_id=scene.marker.aruco_marker_id,
            marker_length_mm=he.marker_length_mm, dict_name=he.aruco_dict_name,
        )
    else:
        raise SystemExit(f"unknown marker source {marker!r}")

    def marker_source():
        # Retry with extra render pumps: the RGB annotator is flaky to populate right after a long
        # no-render arm move, so a single warmup occasionally yields an empty frame (marker missed).
        # Depth-based (ground-truth) detection succeeds on the first try; this only helps ArUco.
        for attempt in range(4):
            _pump(warmup if attempt == 0 else 10)
            t = src.marker_in_camera(wrist_cam)
            if t is not None:
                return np.asarray(t.to_matrix(), dtype=np.float64)
        return None

    # 4) drive the real routine (auto-selects EyeInHandCalibrator from the EYE_IN_HAND mode).
    routine = CalibrationRoutine(
        arm=arm, marker_source=marker_source,
        workspace_limits=cell.robot.workspace_limits, eth_settings=he,
        marker_id=scene.marker.aruco_marker_id,
        settle_time_s=cell.robot.calibration.settle_time_s,
        calibration_mode=MountingMode.EYE_IN_HAND,
        on_event=lambda evt, data: print(f"  [{evt}] {data.get('reason', data.get('accepted', ''))}", flush=True),
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    result = routine.run_with_poses(
        tcp_poses, dataset_save_path=f"{save_dir}/eih_wrist_dataset_{marker}.json",
    )

    # 5) validate vs the oracle + source-aware gate.
    d_t, d_r = transform_delta(result.transform, oracle)
    # The config-schema bands are structurally a QualityBandsMm (excellent/good/marginal floats),
    # so the cast is a typing narrowing, not a conversion.
    quality = str(classify_rmse(result.rmse_mm, cast("QualityBandsMm", cell.robot.calibration.quality_bands_mm)))
    gate_mm = 1.0 if marker == "ground_truth" else 5.0
    ok = d_t < gate_mm and d_r < 1.0 and quality in ("good", "excellent")
    print("\n--- RESULT ---", flush=True)
    print(f"  accepted samples: {result.num_samples}", flush=True)
    print(f"  calibrated T_cam_to_TCP mm: {np.round(np.asarray(result.transform.to_matrix())[:3, 3], 2)}", flush=True)
    print(f"  AX=XB rmse: {result.rmse_mm:.4f} mm | max_error: {result.max_error_mm:.4f} | quality: {quality}", flush=True)
    print(f"  ERROR vs oracle: translation={d_t:.3f} mm, rotation={d_r:.3f} deg", flush=True)
    print(f"  GATE (<{gate_mm:.0f} mm, <1 deg, quality good+): {'PASS' if ok else 'CHECK'}", flush=True)
    # Accepted against planned viewpoints is the number the artifact never carries: the ArUco
    # source silently skips a view it cannot detect, and AX=XB degrades with the spread it lost.
    _log_result = _LOG.info if ok else _LOG.warning
    _log_result(
        "EIH %s/%s solved: %d/%d viewpoints accepted, rmse=%.4f mm max_error=%.4f mm quality=%s, "
        "vs oracle dt=%.3f mm dr=%.3f deg -> gate(<%.0f mm, <1 deg, good+)=%s",
        camera_id, marker, result.num_samples, len(tcp_poses), result.rmse_mm, result.max_error_mm,
        quality, d_t, d_r, gate_mm, "PASS" if ok else "CHECK: the artifacts below are saved anyway",
    )

    # 6) persist the EIH result (typed transform dict; eye-in-hand has no Extrinsics carrier).
    cal_path = Path(save_dir) / f"eih_wrist_cam_{marker}.json"
    cal_path.write_text(
        json.dumps(
            {
                "schema": "willy.willy_sim.eih_calibration/2",
                "mode": str(result.mode),
                "t_cam_to_tool": transform_to_dict(result.transform),
                "rmse_mm": result.rmse_mm,
                "max_error_mm": result.max_error_mm,
                "quality": quality,
                "num_samples": result.num_samples,
                "error_vs_oracle_mm": d_t,
                "error_vs_oracle_deg": d_r,
                "marker_source": marker,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\n  saved {cal_path} and {result.dataset_path}", flush=True)
    # Also persist the typed, camera-id-keyed CAMERA->TOOL artifact so this wrist camera drops straight
    # into the central grasping.fusion.cameras map (mounting_mode: eye_in_hand). The custom dict above is
    # kept for its richer validation metadata; this one is what the resolver-map builder loads.
    try:
        from src.calibration.serialization import save_cam_to_tool

        typed_path = save_cam_to_tool(f"{save_dir}/eih_{camera_id}.json", result.transform, rig_id=camera_id)
        _LOG.info("saved %s + typed %s (rig_id=%s) + dataset %s", cal_path, typed_path, camera_id,
                  result.dataset_path)
        print(f"  saved typed {typed_path} (rig_id={camera_id})", flush=True)
    except Exception as exc:  # noqa: BLE001 (typed save is a bonus; a frame mismatch must not fail the run)
        # The run reads as successful and the rich JSON exists, but the keyed artifact, the one the
        # resolver-map builder loads, does not, so this camera never reaches grasping.fusion.cameras.
        _LOG.warning(
            "typed cam_to_tool save skipped (%s): eih_%s.json was not written, so this camera will "
            "not appear in the resolver map; %s still holds the transform", exc, camera_id, cal_path,
        )
        print(f"  (typed cam_to_tool save skipped: {exc})", flush=True)
    print("========== END EIH CALIBRATION ==========", flush=True)

    if not headless:
        arm.session.step_n(900)
    return {"ok": ok, "error_mm": d_t, "error_deg": d_r, "quality": quality}


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy eye-in-hand wrist-camera calibration (Isaac).")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-headless", action="store_true", help="alias for --gui")
    ap.add_argument("--marker", type=str, default="ground_truth", choices=["ground_truth", "aruco"])
    ap.add_argument("--camera", type=str, default="wrist",
                    help="camera id for the keyed typed artifact eih_<camera>.json (rig_id).")
    ap.add_argument("--data-dir", type=str, default=None, help="override the sim config tree")
    add_cell_arguments(ap)   # --robot-model / --profile (artifacts namespace per robot)
    args = ap.parse_args()
    calibrate(headless=not (args.gui or args.no_headless), marker=args.marker, camera_id=args.camera,
              data_dir=args.data_dir, cell_kwargs=cell_profile_kwargs(args))


if __name__ == "__main__":
    main()
