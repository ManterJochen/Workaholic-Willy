"""Eye-in-hand wrist-camera pick demo recorder for Isaac: a cinematic MP4 of one pick.

Runs the eye-in-hand pick once through ``run_eih_pick.build_service`` and its
AutonomousGraspService, films it from a fixed cinematic 3/4 camera, and encodes the result as an
MP4. Frames are captured every sim step, with short hold pauses at the perceive and lifted moments
so each phase is readable, and the MP4 is written at a low playback fps because the live GUI runs
too fast to follow. The clip and four keyframe PNGs land at ``--out``, ``logs/demo/`` by default.

Requires Isaac, so it runs on the sim box only. Run with Isaac's bundled python from the repo root:
    <isaac-sim>\\python.bat -m src.willy_sim.run_eih_demo --fps 18
    ...\\python.bat -m src.willy_sim.run_eih_demo --gui          # also open the live window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.willy_sim.run_eih_pick import build_service

DEFAULT_OUT = "logs/demo/eih_pick_demo.mp4"
# Cinematic camera (metres): an elevated front-right 3/4 view of the cell, aimed at the workspace.
# Oriented via Isaac's set_camera_view(eye, target), which is the convention-correct call.
CINE_POS_M = (2.05, -1.82, 1.68)
CINE_TARGET_M = (0.45, 0.0, 0.14)
CINE_RES = (1280, 720)


def record_demo(*, out_path: str = DEFAULT_OUT, headless: bool = True, fps: int = 18,
                capture_every: int = 1, hold_steps: int = 55) -> dict:
    """Record one eye-in-hand pick as a cinematic MP4. Returns a summary dict."""
    import cv2  # type: ignore[import-not-found]

    # build_service starts the SimulationApp; the isaacsim.* namespaces only exist after that.
    service, arm, gripper, handles, cfg, view_pose, _cell = build_service(
        headless=headless, continuous_guard=True,  # continuous guard (closed-form path; natural_aim N/A)
    )
    session = arm.session
    app = getattr(session, "app", None)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    # --- cinematic camera ---------------------------------------------------
    # Orient via Isaac's canonical set_camera_view (eye/target). Hand-rolled look-at quaternions get
    # the Isaac sensor-camera optical convention wrong and the camera films the empty dome instead.
    # add_distance_to_image_plane_to_frame activates the render product for a post-reset camera.
    cine = Camera(
        prim_path="/World/Cameras/Cinematic",
        position=np.array(CINE_POS_M, dtype=np.float64),
        resolution=CINE_RES,
    )
    cine.initialize()
    cine.add_distance_to_image_plane_to_frame()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    set_camera_view(eye=list(CINE_POS_M), target=list(CINE_TARGET_M), camera_prim_path="/World/Cameras/Cinematic")
    # warm up the render so the RGB annotator populates before the first capture
    for _ in range(60):
        session.step(render=True)
        if app is not None:
            app.update()

    def _grab() -> np.ndarray | None:
        fr = cine.get_current_frame()
        rgba = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgba is None else np.asarray(rgba)[..., :3].copy()

    probe = _grab()
    print(f"cinematic cam ready: frame {None if probe is None else probe.shape} "
          f"mean={None if probe is None else round(float(np.mean(probe)), 1)}", flush=True)

    # --- frame capture: wrap session.step so every motion step is filmed -----
    frames: list[np.ndarray] = []
    state = {"on": False, "n": 0}
    orig_step = session.step

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001
        orig_step(dt_s, render=True)
        if state["on"]:
            state["n"] += 1
            if state["n"] % capture_every == 0:
                if app is not None:
                    app.update()
                f = _grab()
                if f is not None:
                    frames.append(f)

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(steps: int) -> None:
        for _ in range(steps):
            session.step(render=True)

    # --- reset, then film: hold at view pose, pick, hold lifted --------------
    obj = SingleRigidPrim(handles.object_prim_path)
    home_pos, home_quat = obj.get_world_pose()
    gripper.open()
    arm.move(view_pose)
    obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
    try:
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))
    except Exception:  # noqa: BLE001
        pass
    arm.session.step_n(30)

    state["on"] = True
    _hold(hold_steps)                 # "perceiving" pause at the top-down view pose
    report = service.pick()           # the pick through AutonomousGraspService, filmed throughout
    _hold(hold_steps)                 # hold on the lifted object
    state["on"] = False
    session.step = orig_step          # type: ignore[assignment]

    z_lift = float(np.asarray(obj.get_world_pose()[0])[2]) - float(np.asarray(home_pos)[2])
    outcome = getattr(getattr(report, "outcome", None), "value", getattr(report, "outcome", None))

    # --- encode MP4 + a few keyframe PNGs for inspection --------------------
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise SystemExit("no frames captured: cinematic RGB never populated")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 ships no stubs
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
    for f in frames:
        writer.write(f[..., ::-1])  # RGB -> BGR
    writer.release()
    for i, idx in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        cv2.imwrite(str(out.with_name(f"{out.stem}_key{i}.png")), frames[idx][..., ::-1])

    dur = len(frames) / float(fps)
    print("\n==== DEMO RECORDED ====", flush=True)
    print(f"  outcome={outcome} lift={z_lift * 1000.0:.1f} mm | frames={len(frames)} fps={fps} duration={dur:.1f}s", flush=True)
    print(f"  saved {out}  (+ 4 keyframe PNGs alongside)", flush=True)
    return {"frames": len(frames), "fps": fps, "duration_s": dur, "lift_mm": z_lift * 1000.0, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a cinematic MP4 of the Willy EIH pick (Isaac).")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=18, help="playback fps (lower = slower/easier to watch)")
    ap.add_argument("--capture-every", type=int, default=1, help="capture 1 of every N sim steps")
    ap.add_argument("--hold-steps", type=int, default=55, help="pause length at perceive + lifted moments")
    ap.add_argument("--gui", action="store_true", help="also open the live window")
    args = ap.parse_args()
    record_demo(out_path=args.out, headless=not args.gui, fps=args.fps,
                capture_every=args.capture_every, hold_steps=args.hold_steps)


if __name__ == "__main__":
    main()
