"""Eye-in-hand wrist camera: mount, visual inspection and ground-truth validation.

Mounts the wrist camera (``scene.mount_wrist_camera``), shows that the camera rigidly follows the
wrist, captures what the wrist sees, and checks the optical convention by back-projecting the
object centroid against ground truth. Nothing here touches the pick path.

What it checks, all printed in one block:
  1. The camera tracks wrist_3_link: its world pose changes consistently after an arm move.
  2. T_cam_to_TCP is invariant across arm poses (rigid mount and correct math), so the delta is tiny.
  3. The empirical CAMERA to BASE transform from the Isaac projection: it back-projects the object
     centroid and compares it with the object's true base position. The Umeyama fit RMSE is the
     calibration metric; an oblique view sees the object surface, not its volumetric centre, so a
     centroid delta of a few cm is expected.
  4. Saves wrist RGB and depth PNGs under :data:`CACHE_DIR` so the viewpoint is inspectable.

Run (Isaac bundled python):
    <isaac-sim>\\python.bat -m src.willy_sim.inspect_wrist_cam            # headless
    <isaac-sim>\\python.bat -m src.willy_sim.inspect_wrist_cam --gui      # visible window
    ...\\python.bat -m src.willy_sim.inspect_wrist_cam --gui --ox 80 --oy 30   # tune the mount offset

All heavy imports are lazy / on-box only; importing the module stays macOS/CI-safe.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.config import load_sim_config, require_robot
from src.willy_sim.calibration.hand_eye import transform_delta as _transform_delta
from src.willy_sim.scene import (
    OBJECT_PRIM,
    camera_to_base_ground_truth,  # noqa: F401 (used in the inspect body)
    camera_to_tcp_ground_truth,
    mount_wrist_camera,
)
from src.geometry import Frame, Pose

CACHE_DIR = "logs/willy_sim"

# Two straight-down viewing poses, both reachable: TCP approach down, meaning TCP +Z onto base -Z,
# via quat [1,0,0,0], a 180 deg rotation about base X. Two slightly different arm configs are what
# confirm T_cam_to_TCP is invariant (rigid mount and correct empirical fit).
VIEWPOSE_A = Pose(
    position_mm=np.array([450.0, 0.0, 350.0]),
    quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
    frame=Frame.BASE,
    label="eih-viewpose-a",
)
VIEWPOSE_B = Pose(
    position_mm=np.array([430.0, 0.0, 380.0]),
    quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
    frame=Frame.BASE,
    label="eih-viewpose-b",
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mount + inspect the eye-in-hand wrist camera.")
    ap.add_argument("--gui", action="store_true", help="visible window (headless=False)")
    ap.add_argument("--ox", type=float, default=None, help="override wrist-frame X offset (mm)")
    ap.add_argument("--oy", type=float, default=None, help="override wrist-frame Y offset (mm)")
    ap.add_argument("--oz", type=float, default=None, help="override wrist-frame Z offset (mm)")
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_sim_config(args.data_dir)
    wcam = require_robot(cfg).sim.cameras["wrist"]
    # The wrist camera's mount, resolution and near-clip are Optional in the schema but required here.
    if (
        wcam.mount_offset_mm is None or wcam.mount_aim_mm is None or wcam.mount_up_hint is None
        or wcam.resolution is None or wcam.near_clip_m is None
    ):
        raise ValueError(
            "sim wrist camera config requires mount_offset_mm / mount_aim_mm / mount_up_hint / "
            "resolution / near_clip_m."
        )
    offset = list(wcam.mount_offset_mm)
    for idx, val in enumerate((args.ox, args.oy, args.oz)):  # CLI overrides for tuning
        if val is not None:
            offset[idx] = val

    # The shared boot prefix. This diagnostic builds the arm without a safety preflight.
    cell = bootstrap_sim_cell(args.data_dir, headless=not args.gui, safety=False)
    arm, gripper = cell.arm, cell.gripper
    gripper.open()  # open jaw so the object is visible from the wrist

    wrist_cam = mount_wrist_camera(
        arm.session, prim_path=wcam.prim_path, offset_mm=(offset[0], offset[1], offset[2]),
        aim_target_mm=wcam.mount_aim_mm, up_hint=wcam.mount_up_hint,
        resolution=wcam.resolution, near_clip_m=wcam.near_clip_m,
    )

    print("\n========== WRIST-CAM INSPECT OUTPUT (paste back) ==========", flush=True)
    print(f"mount offset (wrist frame, mm): {tuple(offset)}", flush=True)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.willy_sim.perception import GroundTruthPerceptionSource, object_mask_from_frame

    # --- 1) camera tracks the wrist (world pose at home vs viewpose A) ---
    cam_pos_home_m, _ = wrist_cam.get_world_pose()
    print("\n--- MOVING to viewpose A (450,0,350, approach down) ---", flush=True)
    result = arm.move(VIEWPOSE_A)
    print(f"  move status: {getattr(result.status, 'value', result.status)}, {result.message}", flush=True)
    arm.session.step_n(15)
    cam_pos_view_m, _ = wrist_cam.get_world_pose()
    moved = float(np.linalg.norm((np.asarray(cam_pos_view_m) - np.asarray(cam_pos_home_m)) * 1000.0))
    print(f"  wrist-cam world pos HOME mm: {np.round(np.asarray(cam_pos_home_m) * 1000.0, 1)}", flush=True)
    print(f"  wrist-cam world pos VIEW mm: {np.round(np.asarray(cam_pos_view_m) * 1000.0, 1)}", flush=True)
    print(f"  => camera moved {moved:.1f} mm with the arm (tracks if hundreds of mm)", flush=True)

    # --- 2) perception + segmentation diagnostics @ viewpose A ---
    # The instance-id segmentation annotator on a freshly-mounted camera is flaky: depth is
    # reliable immediately, but the segmentation graph needs ``app.update()`` pumping to populate,
    # the same render pump real-vision RGB needs. So pump ``app.update()`` plus a step and
    # re-acquire until the object mask is non-empty.
    obj_world_mm = np.asarray(SingleRigidPrim(OBJECT_PRIM).get_world_pose()[0], dtype=np.float64) * 1000.0
    perc = GroundTruthPerceptionSource(
        camera=wrist_cam, target_prim_path=OBJECT_PRIM, session=arm.session, warmup_steps=3
    )
    app = getattr(arm.session, "app", None)
    frame = perc.acquire()
    cur = wrist_cam.get_current_frame()
    mask = np.zeros(np.asarray(frame.depth_map).shape, dtype=bool)  # overwritten in the loop's 1st pass
    for attempt in range(6):
        for _ in range(20):
            arm.session.step(render=True)
            if app is not None:
                app.update()
        frame = perc.acquire()
        cur = wrist_cam.get_current_frame()
        mask_e = object_mask_from_frame(cur, OBJECT_PRIM)
        mask = np.zeros(np.asarray(frame.depth_map).shape, dtype=bool) if mask_e is None else np.asarray(mask_e, dtype=bool)
        if int(mask.sum()) > 0:
            break
        print(f"  (seg warmup attempt {attempt + 1}: object mask empty, pumping app.update()...)", flush=True)
    depth = np.asarray(frame.depth_map, dtype=np.float64)
    k = np.asarray(frame.intrinsics)
    print("\n--- WRIST VIEW @ A ---", flush=True)
    print(f"  object true base pos mm: {np.round(obj_world_mm, 1)}", flush=True)
    fin = depth[np.isfinite(depth) & (depth > 0)]
    if fin.size:
        print(f"  wrist depth mm: min={fin.min():.0f} max={fin.max():.0f} (finite {fin.size}/{depth.size})", flush=True)
    iid = cur.get("instance_id_segmentation") if isinstance(cur, dict) else None
    seg_data = np.asarray(iid["data"]) if isinstance(iid, dict) and iid.get("data") is not None else None
    if seg_data is not None:
        uids, ucnt = np.unique(seg_data, return_counts=True)
        print(f"  seg id->count: {{ {', '.join(f'{int(i)}:{int(c)}' for i, c in zip(uids, ucnt))} }}", flush=True)
    print(f"  object mask pixels: {int(mask.sum())}", flush=True)

    # --- 3) empirical CAMERA to BASE (Isaac projection + Umeyama) + back-projection check ---
    print("\n--- CALIBRATION ORACLE (empirical CAMERA->BASE, convention-agnostic) ---", flush=True)
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        print("  !! object not visible: tune --ox/--oy/--oz so the wrist sees the object.", flush=True)
    else:
        ys, xs = np.where(valid)
        u, v = float(xs.mean()), float(ys.mean())
        d = float(np.median(depth[valid]))
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        p_cam = np.array([(u - cx) / fx * d, (v - cy) / fy * d, d], dtype=np.float64)
        t_cam_to_base_a, rmse_a = camera_to_base_ground_truth(wrist_cam)
        p_base = t_cam_to_base_a.apply_point(p_cam)
        err_fit = float(np.linalg.norm(p_base - obj_world_mm))
        cal_ok = rmse_a < 1.0
        print(f"  CAMERA->BASE Umeyama fit rmse: {rmse_a:.3f} mm  "
              f"{'<== PASS (<1 mm: exact)' if cal_ok else '<< CHECK'}", flush=True)
        print(f"  object centroid -> base (oracle): {np.round(p_base, 1)} vs true {np.round(obj_world_mm, 1)} "
              f"(delta={err_fit:.1f} mm)", flush=True)
        print("    note: an oblique view sees the object SURFACE, not its volumetric centre "
              "(50 mm-tall object), so a few-cm delta here is expected; the fit rmse is the calibration metric.",
              flush=True)

        # save what the wrist sees @ A
        try:
            import cv2  # type: ignore[import-not-found]

            rgba = cur.get("rgb") if isinstance(cur, dict) else None
            if rgba is not None:
                cv2.imwrite(f"{CACHE_DIR}/_eih_wrist_rgb.png", np.asarray(rgba)[..., :3][..., ::-1])
            if fin.size:
                lo, hi = float(fin.min()), float(fin.max())
                vis = (np.clip((depth - lo) / max(1e-6, hi - lo), 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(f"{CACHE_DIR}/_eih_wrist_depth.png", vis)
            print(f"  saved {CACHE_DIR}/_eih_wrist_rgb.png and _eih_wrist_depth.png", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  (image save skipped: {exc})", flush=True)

        # --- 4) T_cam_to_TCP invariance across two viewposes ---
        t_cam_to_tcp_a, _ = camera_to_tcp_ground_truth(wrist_cam, arm)
        print("\n--- MOVING to viewpose B (430,0,380) for invariance ---", flush=True)
        rb = arm.move(VIEWPOSE_B)
        print(f"  move status: {getattr(rb.status, 'value', rb.status)}", flush=True)
        arm.session.step_n(15)
        perc.acquire()  # refresh the camera depth frame at B
        t_cam_to_tcp_b, _ = camera_to_tcp_ground_truth(wrist_cam, arm)
        d_t, d_r = _transform_delta(t_cam_to_tcp_a, t_cam_to_tcp_b)
        m = np.asarray(t_cam_to_tcp_a.to_matrix())
        print("\n--- T_cam_to_TCP (CAMERA->TOOL, the EIH calibration target) ---", flush=True)
        print(f"  translation mm: {np.round(m[:3, 3], 2)}", flush=True)
        print(f"  invariance A vs B: d_translation={d_t:.2f} mm, d_rotation={d_r:.3f} deg "
              f"(both ~0 => correct + rigid)", flush=True)

    print("\n========== END WRIST-CAM INSPECT OUTPUT ==========", flush=True)

    if args.gui:
        print("\n[GUI] holding ~20 s: look at the wrist camera placement in the viewport.", flush=True)
        arm.session.step_n(1200)


if __name__ == "__main__":
    main()
