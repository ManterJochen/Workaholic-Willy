"""Dual-cam clutter pick demo recorder (Isaac): a watchable cinematic MP4 of the prompt-select pick.

Films one dual-cam pick on the three-distinct-cube clutter scene: the overhead camera localizes the
prompted cube, then the wrist eye-in-hand camera moves in and grasps it, leaving the distractors
where they are. Reuses the ``run_fused_pick`` and ``run_eih_pick`` machinery (``build_service``, the
true-extrinsic coarse localize, ``topdown_view_pose``, the calibrated eye-in-hand grasp) and adds
only a fixed cinematic camera, frame capture and MP4 encode, the way ``run_eih_demo`` does. The MP4
plays at a low fps so the pick is easy to watch.

On-box only (needs Isaac). Run with Isaac's bundled python from the repo root:
    <isaac-sim>\\python.bat -m src.willy_sim.run_clutter_demo --prompt "the green cube"
    ...\\python.bat -m src.willy_sim.run_clutter_demo --prompt "the red cube" --fps 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.willy_sim.run_eih_pick import (
    _load_calibrated_eih,
    build_service,
    coarse_centroid_base,
    topdown_view_pose,
)
from src.willy_sim.run_fused_pick import OBJECT_PRIM, _clutter_specs, _select_by_prompt

DEFAULT_OUT = "logs/demo/clutter_pick_demo.mp4"
# Cinematic camera (metres): an elevated front-right 3/4 view of the cell, aimed at the workspace.
CINE_POS_M = (2.05, -1.82, 1.68)
CINE_TARGET_M = (0.45, 0.0, 0.14)
CINE_RES = (1280, 720)


def record_clutter_demo(*, out_path: str = DEFAULT_OUT, headless: bool = True, fps: int = 18,
                        prompt: str = "the green cube",
                        view_height_mm: float = 275.0, hold_steps: int = 45) -> dict:
    """Record one prompt-select clutter pick as a cinematic MP4; returns a summary dict.

    The recording uses the ground-truth overhead localize, which is visually identical to the vision
    path that ``run_fused_pick --coarse vision`` drives. The wrist grasp is the calibrated
    eye-in-hand pick.
    """
    import cv2  # type: ignore[import-not-found]

    objects = _clutter_specs()
    chosen_idx = _select_by_prompt(objects, prompt)
    chosen = objects[chosen_idx]
    chosen_prim = f"{OBJECT_PRIM}_{chosen_idx}"
    print(f"=== CLUTTER DEMO: prompt={prompt!r} -> '{chosen.name}' ({chosen_prim}) among {len(objects)} cubes ===",
          flush=True)

    service, arm, gripper, handles, cfg, _vp, _cell = build_service(
        headless=headless, objects_override=objects, wrist_target_prim=chosen_prim,
        continuous_guard=True,  # continuous guard on the EIH closed-form path; natural_aim does not apply
    )
    session = arm.session
    app = getattr(session, "app", None)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    from src.willy_sim.perception import GroundTruthPerceptionSource
    from src.willy_sim.scene import camera_to_base_ground_truth
    from src.robot.core import JointPositions

    sim = cfg.robot.sim
    cal = _load_calibrated_eih("ground_truth")
    if cal is None:
        raise SystemExit("clutter demo requires the calibrated EIH artifact (eih_wrist_cam_ground_truth.json)")
    p_cam_tool = np.asarray(cal.to_matrix(), dtype=np.float64)[:3, 3]

    # --- cinematic camera (orient via Isaac's canonical set_camera_view) --------------------------
    cine = Camera(prim_path="/World/Cameras/Cinematic", position=np.array(CINE_POS_M), resolution=CINE_RES)
    cine.initialize()
    cine.add_distance_to_image_plane_to_frame()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    set_camera_view(eye=list(CINE_POS_M), target=list(CINE_TARGET_M), camera_prim_path="/World/Cameras/Cinematic")

    # --- overhead localize (true CV-optical extrinsic; near-clip 0.05 + depth annotator) ----------
    overhead_cam = handles.camera
    park_q = np.asarray(sim.park_joint_positions, dtype=np.float64)
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    for fn in (lambda: overhead_cam.add_distance_to_image_plane_to_frame(),
               lambda: overhead_cam.set_clipping_range(0.05, 1.0e6)):
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass
    arm.move_joint(JointPositions(park_q))
    for _ in range(max(40, sim.scene_setup.render_warmup_steps)):
        session.step(render=True)
        if app is not None:
            app.update()
    t_overhead_true, _ = camera_to_base_ground_truth(overhead_cam)

    overhead = GroundTruthPerceptionSource(
        camera=overhead_cam, target_prim_path=chosen_prim, session=session,
        warmup_steps=sim.scene_setup.render_warmup_steps, ground_truth_depth=True, grasp_lift_mm=0.0,
    )

    def _grab() -> np.ndarray | None:
        fr = cine.get_current_frame()
        rgba = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgba is None else np.asarray(rgba)[..., :3].copy()

    # --- frame capture: wrap session.step so every motion step is filmed --------------------------
    frames: list[np.ndarray] = []
    state = {"on": False}
    orig_step = session.step

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001
        orig_step(dt_s, render=True)
        if state["on"]:
            if app is not None:
                app.update()
            f = _grab()
            if f is not None:
                frames.append(f)

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(steps: int) -> None:
        for _ in range(steps):
            session.step(render=True)

    # --- film one pick: park, overhead localize, via home, then wrist grasp ------------------------
    prims = [SingleRigidPrim(spec[0]) for spec in handles.object_specs]
    homes = [(np.asarray(p.get_world_pose()[0]), np.asarray(p.get_world_pose()[1])) for p in prims]
    gripper.open()
    for p, (hp, hq) in zip(prims, homes):
        p.set_world_pose(position=hp, orientation=hq)
    arm.move_joint(JointPositions(park_q))
    session.step_n(20)

    state["on"] = True
    _hold(hold_steps)                                   # establish: 3 cubes, arm parked
    coarse_c = coarse_centroid_base(overhead.acquire(), t_overhead_true)
    if coarse_c is None:
        raise SystemExit("overhead found no chosen object")
    arm.move_joint(JointPositions(home_q))              # via home (reachable approach)
    arm.move(topdown_view_pose(coarse_c, p_cam_tool, view_height_mm))  # wrist over the chosen cube
    _hold(hold_steps // 2)
    report = service.pick()                             # grasp the chosen cube (AutonomousGraspService)
    _hold(hold_steps)                                   # hold on the lifted cube
    state["on"] = False
    session.step = orig_step  # type: ignore[assignment]

    z_lift = (float(np.asarray(prims[chosen_idx].get_world_pose()[0])[2]) - float(homes[chosen_idx][0][2])) * 1000.0
    outcome = getattr(getattr(report, "outcome", None), "value", getattr(report, "outcome", None))

    # --- encode MP4 + keyframe PNGs ---------------------------------------------------------------
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise SystemExit("no frames captured: cinematic RGB never populated")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 ships no stubs
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
    for f in frames:
        writer.write(f[..., ::-1])
    writer.release()
    for i, idx in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        cv2.imwrite(str(out.with_name(f"{out.stem}_key{i}.png")), frames[idx][..., ::-1])

    dur = len(frames) / float(fps)
    print("\n==== CLUTTER DEMO RECORDED ====", flush=True)
    print(f"  prompt={prompt!r} chosen={chosen.name!r} outcome={outcome} lift={z_lift:.1f} mm", flush=True)
    print(f"  frames={len(frames)} fps={fps} duration={dur:.1f}s -> {out}  (+ 4 keyframe PNGs)", flush=True)
    return {"frames": len(frames), "fps": fps, "duration_s": dur, "lift_mm": z_lift, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a cinematic MP4 of the Willy dual-cam clutter pick (Isaac).")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=18, help="playback fps (lower = slower/easier to watch)")
    ap.add_argument("--prompt", type=str, default="the green cube")
    ap.add_argument("--gui", action="store_true", help="also open the live window")
    args = ap.parse_args()
    record_clutter_demo(out_path=args.out, headless=not args.gui, fps=args.fps, prompt=args.prompt)


if __name__ == "__main__":
    main()
