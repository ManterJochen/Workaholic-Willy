"""Dense-vision pick demo recorder (Isaac): a split-screen cinematic MP4 of the vision-language pick.

A watchable record of the whole perception and grasp chain on real YCB objects:

* left panel: the overhead vision view, with GroundingDINO detection boxes, SAM2 masks and the object
  names (sugar box / tomato soup can / red cracker box), the prompted target highlighted. This is
  "what the robot sees" (the vision-language grounding).
* right panel: a cinematic 3/4 view of the arm grasping the prompted object and lifting it. This is
  "what the robot does".

Reuses ``run_dense_pick.build_service`` (``ycb=True, vision=True``), so the demo films the real pick
and never a mock. The perception panel is drawn here rather than taken from the calculator's debug
PNG, so it shows every detection, its name and the target.

On-box only (needs Isaac + the cached GroundingDINO/SAM2 weights). Run with Isaac's bundled python from
the repo root:

    <isaac-sim>\\python.bat -m src.willy_sim.run_dense_demo --prompt "the sugar box"
    ...\\python.bat -m src.willy_sim.run_dense_demo --gui     # also open the live window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.willy_sim.run_dense_pick import build_service

DEFAULT_OUT = "logs/demo/ycb_vision_demo.mp4"
# Cinematic camera (metres): an elevated front-right 3/4 view aimed at the YCB workspace (objects at
# x~0.45 m). Oriented via Isaac's set_camera_view(eye, target), the only convention-correct way.
CINE_POS_M = (2.05, -1.82, 1.68)
CINE_TARGET_M = (0.45, -0.12, 0.08)
CINE_RES = (1100, 760)  # the right (cinematic) panel resolution
_TARGET_COLOR = (60, 220, 60)        # BGR: the prompted target (green)
_DISTRACTOR_COLORS = [(80, 170, 255), (255, 150, 70), (200, 120, 255)]  # BGR: distractors


def _draw_perception_panel(rgb, segmentations, target_label, prompt, panel_h):  # noqa: ANN001, ANN202
    """Draw GroundingDINO boxes + SAM2 masks + names on the overhead RGB; highlight the target. Returns BGR."""
    import cv2  # type: ignore[import-not-found]

    img = np.ascontiguousarray(np.asarray(rgb)[..., ::-1]).copy()  # convert RGB to BGR for OpenCV
    di = 0
    for seg in segmentations:
        is_target = getattr(seg, "label", "") == target_label
        col = _TARGET_COLOR if is_target else _DISTRACTOR_COLORS[di % len(_DISTRACTOR_COLORS)]
        if not is_target:
            di += 1
        mask = np.asarray(getattr(seg, "mask", None)).astype(bool) if getattr(seg, "mask", None) is not None else None
        if mask is not None and mask.any() and mask.shape == img.shape[:2]:
            tint = img.copy()
            tint[mask] = col
            img = cv2.addWeighted(tint, 0.35, img, 0.65, 0)
        bbox = getattr(seg, "bbox_xyxy", None) or getattr(seg, "derived_bbox_xyxy", None)
        label = str(getattr(seg, "label", "object"))
        if bbox is not None:
            x0, y0, x1, y1 = (int(v) for v in bbox)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 3 if is_target else 2)
            tag = label + ("  <- TARGET" if is_target else "")
            ty = max(20, y0 - 8)
            cv2.putText(img, tag, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, tag, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
    cap = f'Prompt: "{prompt}"   |   GroundingDINO + SAM2'
    cv2.putText(img, cap, (14, img.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, cap, (14, img.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # scale to the panel height, preserving aspect
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * (panel_h / float(h)))))
    return cv2.resize(img, (new_w, panel_h))


def record_demo(*, out_path: str = DEFAULT_OUT, prompt: str = "the sugar box", headless: bool = True,
                fps: int = 18, hold_steps: int = 50, capture_every: int = 2,
                max_attempts: int = 6, lift_ok_mm: float = 30.0) -> dict:
    """Record one dense-vision pick as a split-screen (perception | cinematic) MP4. Returns a summary."""
    import cv2  # type: ignore[import-not-found]

    service, arm, gripper, handles, cfg, target_idx, target_label = build_service(
        prompt=prompt, headless=headless, ycb=True, vision=True,
        natural_aim=True, continuous_guard=True,  # pick the clean IK branch and guard it (no self-collision)
    )
    session = arm.session
    app = getattr(session, "app", None)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    # --- cinematic camera (canonical eye/target orientation) ----------------
    cine = Camera(prim_path="/World/Cameras/Cinematic", position=np.array(CINE_POS_M, dtype=np.float64),
                  resolution=CINE_RES)
    cine.initialize()
    cine.add_distance_to_image_plane_to_frame()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    set_camera_view(eye=list(CINE_POS_M), target=list(CINE_TARGET_M),
                    camera_prim_path="/World/Cameras/Cinematic")
    for _ in range(60):
        session.step(render=True)
        if app is not None:
            app.update()

    def _grab_cine() -> "np.ndarray | None":
        fr = cine.get_current_frame()
        rgba = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgba is None else np.asarray(rgba)[..., :3].copy()

    # --- reset + settle all objects (mirror run_gate) -----------------------
    park_q = np.asarray(cfg.robot.sim.park_joint_positions, dtype=np.float64)
    prims = [p for (p, _label) in handles.object_specs]
    objs = [SingleRigidPrim(p) for p in prims]
    homes = [o.get_world_pose() for o in objs]
    # --- capture wiring: film the cinematic cam each sim step. app.update() runs on every step, not only
    # on the captured ones, so the filmed pick is as reliable as the unwrapped runner; capture_every only
    # thins the saved frames for a shorter clip. Tying app.update() to capture_every destabilises the grasp.
    frames: list[np.ndarray] = []
    state = {"on": False, "n": 0}
    orig_step = session.step
    every = max(1, int(capture_every))

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001
        orig_step(dt_s, render=True)
        if app is not None:
            app.update()
        if state["on"]:
            state["n"] += 1
            if state["n"] % every == 0:
                f = _grab_cine()
                if f is not None:
                    frames.append(f)

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(steps: int) -> None:
        for _ in range(steps):
            session.step(render=True)

    def _reset_scene() -> None:
        gripper.open()
        arm.move_to_joints(JointPositions(park_q))
        for o, (hp, hq) in zip(objs, homes):
            o.set_world_pose(position=np.asarray(hp), orientation=np.asarray(hq))
            try:
                o.set_linear_velocity(np.zeros(3))
                o.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
        arm.session.step_n(120)
        arm.move_to_joints(JointPositions(park_q))  # park: clean, non-occluding overhead view
        for _ in range(max(20, cfg.robot.sim.scene_setup.render_warmup_steps)):
            session.step(render=True)

    # --- perception panel (once; the scene is identical across resets) -------
    _reset_scene()
    perception = service.runtime.orchestrator.perception
    pframe = perception.acquire()
    panel = None
    if getattr(pframe, "rgb", None) is not None and pframe.segmentations:
        panel = _draw_perception_panel(pframe.rgb, pframe.segmentations, target_label, prompt, CINE_RES[1])
        print(f"perception panel: {len(pframe.segmentations)} segs "
              f"{[getattr(s, 'label', '?') for s in pframe.segmentations]}", flush=True)
    else:
        print("WARNING: no vision rgb/segmentations for the perception panel", flush=True)

    # --- record-until-lift: re-pick (reset each retry) until the box actually lifts, keep the good take ----
    # The vision grasp on the thin on-edge box is not perfect, and a demo must show a real lift: a miss is
    # retried from a fresh reset and only the lifting take's frames are kept.
    home_z = float(np.asarray(homes[target_idx][0])[2])
    z_lift = 0.0
    outcome = None
    for attempt in range(max(1, max_attempts)):
        if attempt > 0:
            _reset_scene()
        frames.clear()
        state["n"] = 0
        state["on"] = True
        _hold(hold_steps)                 # hold on the scene (the perception panel reads)
        report = service.pick()           # the real dense-vision pick, filmed throughout
        _hold(hold_steps)                 # hold on the lifted object
        state["on"] = False
        z_lift = (float(np.asarray(objs[target_idx].get_world_pose()[0])[2]) - home_z) * 1000.0
        outcome = getattr(getattr(report, "outcome", None), "value", getattr(report, "outcome", None))
        if z_lift >= float(lift_ok_mm):
            print(f"take {attempt}: lift {z_lift:.1f} mm >= {lift_ok_mm:.0f} -> keep", flush=True)
            break
        print(f"take {attempt}: lift {z_lift:.1f} mm < {lift_ok_mm:.0f} -> retry", flush=True)
    session.step = orig_step          # type: ignore[assignment]

    # --- composite split-screen [perception | cinematic] + encode MP4 -------
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise SystemExit("no cinematic frames captured; RGB never populated")
    panel_h = frames[0].shape[0]
    if panel is None:
        panel = np.zeros((panel_h, panel_h, 3), dtype=np.uint8)
    elif panel.shape[0] != panel_h:
        panel = cv2.resize(panel, (int(panel.shape[1] * panel_h / panel.shape[0]), panel_h))
    composed: list[np.ndarray] = []
    for f in frames:
        right = np.ascontiguousarray(f[..., ::-1])  # convert RGB to BGR
        composed.append(np.hstack([panel, right]))
    big_h, big_w = composed[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 ships no stubs
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (big_w, big_h))
    for c in composed:
        writer.write(c)
    writer.release()
    for i, idx in enumerate(np.linspace(0, len(composed) - 1, 4).astype(int)):
        cv2.imwrite(str(out.with_name(f"{out.stem}_key{i}.png")), composed[idx])

    dur = len(composed) / float(fps)
    print("\n==== VISION DEMO RECORDED ====", flush=True)
    print(f"  prompt={prompt!r} target={target_label!r} outcome={outcome} lift={z_lift:.1f} mm", flush=True)
    print(f"  frames={len(composed)} fps={fps} duration={dur:.1f}s size={big_w}x{big_h}", flush=True)
    print(f"  saved {out}  (+ 4 keyframe PNGs alongside)", flush=True)
    return {"frames": len(composed), "fps": fps, "duration_s": dur, "lift_mm": z_lift,
            "outcome": outcome, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a split-screen cinematic MP4 of the Willy dense-vision pick.")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--prompt", type=str, default="the sugar box")
    ap.add_argument("--fps", type=int, default=18, help="playback fps (lower = slower/easier to watch)")
    ap.add_argument("--hold-steps", type=int, default=50, help="pause length at the scene + lifted moments")
    ap.add_argument("--capture-every", type=int, default=2, help="capture 1 of every N sim steps (shorter clip)")
    ap.add_argument("--max-attempts", type=int, default=6, help="re-pick until the box lifts (keep the good take)")
    ap.add_argument("--lift-ok", type=float, default=30.0, help="min lift (mm) that counts as a good take")
    ap.add_argument("--gui", action="store_true", help="also open the live window")
    args = ap.parse_args()
    record_demo(out_path=args.out, prompt=args.prompt, headless=not args.gui,
                fps=args.fps, hold_steps=args.hold_steps, capture_every=args.capture_every,
                max_attempts=args.max_attempts, lift_ok_mm=args.lift_ok)


if __name__ == "__main__":
    main()
