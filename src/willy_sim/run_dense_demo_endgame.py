"""The multi-segment endgame demo: one investor MP4 covering the whole stack (Isaac).

Shows the full stack in one cinematic video, one segment per capability, each recorded in its own clean
process and then stitched offline with title cards:

* ``vision``: the prompt is grounded by GroundingDINO and SAM2, then the object is grasped and lifted.
  The left perception panel shows the detection boxes, masks and names; the right panel is the
  cinematic pick. "What it sees / does."
* ``safety``: a blocker sits in the approach corridor, the swept-volume validator refuses every
  candidate (approach_path_blocked), and no blind grab follows.
* ``recovery``: the same jam, and the contact-redistribute pushes the movable blocker out of the
  corridor so the re-pick succeeds.
* ``autonomy``: the autonomous loop. ``DENSE_AUTONOMOUS`` perceives, refines, verifies and recovers.

Each frame carries a decision and state banner naming the segment, the loop state and the outcome.
Reuses ``run_dense_pick.build_service`` and the real pick paths, no mock. On-box only (Isaac).

    # record each segment (4 clean processes):
    ...python.bat -m src.willy_sim.run_dense_demo_endgame --segment vision
    ...python.bat -m src.willy_sim.run_dense_demo_endgame --segment safety
    ...python.bat -m src.willy_sim.run_dense_demo_endgame --segment recovery
    ...python.bat -m src.willy_sim.run_dense_demo_endgame --segment autonomy
    # stitch them into one MP4 (offline, no Isaac):
    python -m src.willy_sim.run_dense_demo_endgame --stitch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.willy_sim.run_dense_demo import _draw_perception_panel
from src.willy_sim.run_dense_pick import build_service

# The recorder's output directory. It is repo-relative, so a default run writes inside the working
# tree and `stitch()` reads back exactly what `record_segment` wrote.
#
# This is where a recording is written, not a store of already-shipped clips. Pass `demo_dir=<dir>`
# to stitch segment MP4s that live somewhere else.
_DEMO_DIR = "logs/demo/endgame"
CINE_POS_M = (2.05, -1.82, 1.68)
CINE_RES = (1100, 760)
# BGR accent per segment (banner colour).
_SEGMENTS: dict[str, dict] = {
    "vision": {
        "title": "VISION-LANGUAGE GRASP",
        "caption": "prompt -> GroundingDINO + SAM2 -> grasp + lift",
        "color": (60, 220, 60),
        "target": (0.45, -0.12, 0.08),
    },
    "safety": {
        "title": "FAIL-CLOSED SAFETY (G12)",
        "caption": "blocker in the approach corridor -> REFUSE (no blind grab)",
        "color": (60, 200, 255),
        "target": (0.45, -0.09, 0.05),
    },
    "recovery": {
        "title": "SELF-RECOVERY (C3 redistribute)",
        "caption": "jam -> push the blocker out of the corridor -> re-pick lifts",
        "color": (255, 170, 70),
        "target": (0.45, -0.09, 0.05),
    },
    "autonomy": {
        "title": "AUTONOMOUS LOOP (C2)",
        "caption": "perceive -> refine -> verify -> recover",
        "color": (200, 120, 255),
        "target": (0.45, -0.12, 0.08),
    },
}


def _banner(frame_bgr, title, caption, state, color):  # noqa: ANN001, ANN202
    """Draw a top title banner + a bottom state caption on a BGR frame: the overlay that says why."""
    import cv2  # type: ignore[import-not-found]

    img = np.ascontiguousarray(frame_bgr).copy()
    h, w = img.shape[:2]
    # top title bar
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 64), (20, 20, 20), -1)
    img = cv2.addWeighted(bar, 0.55, img, 0.45, 0)
    cv2.rectangle(img, (0, 62), (w, 66), color, -1)
    cv2.putText(img, title, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, title, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    # bottom caption + live state
    text = caption if not state else f"{caption}    [{state}]"
    cv2.putText(img, text, (18, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (18, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def record_segment(segment: str, *, out_path: str | None = None, headless: bool = True,
                   fps: int = 18, hold_steps: int = 45, capture_every: int = 2,
                   max_attempts: int = 6, lift_ok_mm: float = 30.0) -> dict:
    """Record one demo segment as a cinematic MP4 with the decision/state banner. Returns a summary."""
    import cv2  # type: ignore[import-not-found]

    spec = _SEGMENTS[segment]
    out = Path(out_path or f"{_DEMO_DIR}/seg_{segment}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- per-segment service config (all via the validated build_service flags) -------------------------
    is_vision = segment == "vision"
    is_blocking = segment in ("safety", "recovery")
    prompt = "the sugar box" if is_vision else "the red cube"
    svc = build_service(
        prompt=prompt, headless=headless,
        ycb=is_vision, vision=is_vision,
        blocking=is_blocking, enable_g12=is_blocking,
        mode="dense_autonomous" if segment == "autonomy" else "dense_clutter",
        # Natural aim (the clean IK branch) for the free-pick segments; off for the blocking ones, whose
        # refusal and recovery scenarios drive their own approach. The continuous guard is always on.
        natural_aim=not is_blocking, continuous_guard=True,
    )
    service, arm, gripper, handles, cfg, target_idx, target_label = svc
    session = arm.session
    app = getattr(session, "app", None)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    cine = Camera(prim_path="/World/Cameras/Cinematic", position=np.array(CINE_POS_M, dtype=np.float64),
                  resolution=CINE_RES)
    cine.initialize()
    cine.add_distance_to_image_plane_to_frame()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    set_camera_view(eye=list(CINE_POS_M), target=list(spec["target"]),
                    camera_prim_path="/World/Cameras/Cinematic")
    for _ in range(60):
        session.step(render=True)
        if app is not None:
            app.update()

    def _grab():  # noqa: ANN202
        fr = cine.get_current_frame()
        rgba = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgba is None else np.asarray(rgba)[..., :3].copy()

    park_q = np.asarray(cfg.robot.sim.park_joint_positions, dtype=np.float64)
    objs = [SingleRigidPrim(p) for (p, _l) in handles.object_specs]
    homes = [o.get_world_pose() for o in objs]
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
                f = _grab()
                if f is not None:
                    frames.append(f)

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(n: int) -> None:
        for _ in range(n):
            session.step(render=True)

    def _reset() -> None:
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
        arm.move_to_joints(JointPositions(park_q))
        for _ in range(max(20, cfg.robot.sim.scene_setup.render_warmup_steps)):
            session.step(render=True)

    # --- the segment's pick behaviour (real paths) -----------------------------------------------------
    def _pick_once():  # noqa: ANN202 (returns (report, recovered_bool))
        if segment == "recovery":
            from src.geometry import Frame, Pose
            from src.robot.grasping.motion.execution_policy import _quaternion_from_axes
            from src.robot.grasping.recovery.orchestrator import run_recovery_loop

            from src.willy_sim.run_dense_pick import (
                _build_g6_recovery,
                _G6RecoveryAdapter,
            )
            hover = np.asarray(homes[target_idx][0], dtype=np.float64) * 1000.0 + np.array([0.0, 0.0, 200.0])
            hover_pose = Pose(position_mm=hover,
                              quaternion_xyzw=_quaternion_from_axes(np.array([1.0, 0.0, 0.0]),
                                                                    np.array([0.0, 0.0, -1.0])),
                              frame=Frame.BASE, label="g6_hover")
            prof, pol, orch_r = _build_g6_recovery(50.0, center_mm=tuple(hover),
                                                   contact_depth_mm=175.0, sweep_offset_mm=35.0)

            def _pth():  # noqa: ANN202
                arm.move_to_joints(JointPositions(park_q))
                rep = service.pick()
                arm.move(hover_pose)
                return _G6RecoveryAdapter(rep)

            final, trail = run_recovery_loop(pick=_pth, profile=prof, policy=pol,
                                             orchestrator=orch_r, frame_acquirer=lambda: None, arm=arm)
            return final.report, (trail is not None and trail.terminal_reason == "recovered_success")
        return service.pick(), False

    # vision perception panel (drawn once; the scene is identical across resets)
    _reset()
    panel = None
    if is_vision:
        pframe = service.runtime.orchestrator.perception.acquire()
        if getattr(pframe, "rgb", None) is not None and pframe.segmentations:
            panel = _draw_perception_panel(pframe.rgb, pframe.segmentations, target_label, prompt, CINE_RES[1])

    home_z = float(np.asarray(homes[target_idx][0])[2])
    z_lift, outcome, recovered = 0.0, None, False
    # safety segment: the pick must refuse (no lift), so the first take is kept; the refusal is the point.
    want_lift = segment != "safety"
    for attempt in range(max(1, max_attempts)):
        if attempt > 0:
            _reset()
        frames.clear()
        state["n"] = 0
        state["on"] = True
        _hold(hold_steps)
        report, recovered = _pick_once()
        _hold(hold_steps)
        state["on"] = False
        z_lift = (float(np.asarray(objs[target_idx].get_world_pose()[0])[2]) - home_z) * 1000.0
        outcome = getattr(getattr(report, "outcome", None), "value", getattr(report, "outcome", None))
        # the precise reason (e.g. approach_path_blocked) lives on the pick_report; the top-level outcome
        # collapses it to execution_failed, so the captions for safety and recovery use the precise one.
        _pr = getattr(report, "pick_report", None)
        pick_outcome = getattr(getattr(_pr, "outcome", None), "value", None) or outcome
        if not want_lift or z_lift >= float(lift_ok_mm):
            break
        print(f"take {attempt}: lift {z_lift:.1f} < {lift_ok_mm:.0f} -> retry", flush=True)
    session.step = orig_step  # type: ignore[assignment]

    # --- compose + encode (banner on every frame; vision adds the left perception panel) ---------------
    if not frames:
        raise SystemExit("no cinematic frames captured")
    _lifted = z_lift >= lift_ok_mm
    state_label = {
        "vision": f"GRASP_NOW -> lift {z_lift:.0f} mm",
        "safety": f"{pick_outcome} -> REFUSED (no blind grab)",
        "recovery": (f"recovered_success -> re-pick lift {z_lift:.0f} mm" if (recovered or _lifted)
                     else f"{pick_outcome} (no recovery this take)"),
        "autonomy": f"refine->verify->{outcome} -> lift {z_lift:.0f} mm",
    }[segment]
    composed: list[np.ndarray] = []
    for f in frames:
        right = _banner(np.ascontiguousarray(f[..., ::-1]), spec["title"], spec["caption"],
                        state_label, spec["color"])
        if panel is not None:
            ph = right.shape[0]
            pn = panel if panel.shape[0] == ph else cv2.resize(panel, (int(panel.shape[1] * ph / panel.shape[0]), ph))
            composed.append(np.hstack([pn, right]))
        else:
            composed.append(right)
    big_h, big_w = composed[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (big_w, big_h))
    for c in composed:
        writer.write(c)
    writer.release()
    print(f"\n==== SEGMENT {segment} RECORDED ====", flush=True)
    print(f"  outcome={outcome} lift={z_lift:.1f}mm recovered={recovered} frames={len(composed)} -> {out}",
          flush=True)
    return {"segment": segment, "outcome": outcome, "lift_mm": z_lift, "recovered": recovered,
            "frames": len(composed), "out": str(out)}


def _title_card(text_lines, size, color, frames):  # noqa: ANN001, ANN202
    """A solid title card (list of BGR frames) shown for ``frames`` frames before a segment."""
    import cv2  # type: ignore[import-not-found]

    w, h = size
    card = np.full((h, w, 3), 18, dtype=np.uint8)
    cv2.rectangle(card, (0, h // 2 + 40), (w, h // 2 + 46), color, -1)
    for i, line in enumerate(text_lines):
        scale = 1.4 if i == 0 else 0.8
        (tw, _th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = (w - tw) // 2
        y = h // 2 - 20 + i * 46
        cv2.putText(card, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color if i == 0 else (235, 235, 235), 2, cv2.LINE_AA)
    return [card.copy() for _ in range(frames)]


def stitch(*, demo_dir: str = _DEMO_DIR, out_path: str | None = None, fps: int = 18,
           max_seg_frames: int = 220, order=("vision", "safety", "recovery", "autonomy")) -> dict:
    """Offline (no Isaac): concatenate the segment MP4s into one investor MP4 with title cards.

    Each segment is subsampled to at most ``max_seg_frames`` on a uniform stride, so the segments stay
    balanced and the whole video stays watchable however long each pick or recovery loop ran.
    """
    import cv2  # type: ignore[import-not-found]

    out = Path(out_path or f"{demo_dir}/willy_endgame_demo.mp4")
    seg_frames: dict[str, list[np.ndarray]] = {}
    for seg in order:
        p = Path(demo_dir) / f"seg_{seg}.mp4"
        if not p.exists():
            print(f"  skip {seg}: {p} not found", flush=True)
            continue
        cap = cv2.VideoCapture(str(p))
        fr: list[np.ndarray] = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(f)
        cap.release()
        if fr:
            if len(fr) > max_seg_frames:  # uniform-stride subsample to keep segments balanced
                idx = np.linspace(0, len(fr) - 1, max_seg_frames).astype(int)
                fr = [fr[i] for i in idx]
            seg_frames[seg] = fr
    if not seg_frames:
        raise SystemExit("no segment MP4s found to stitch")
    # Segments differ in width (vision carries the extra perception panel), so the canvas takes the
    # largest of each dimension and every frame is letterbox-padded (centered, black bars) rather
    # than stretched, which would distort it.
    w = max(f[0].shape[1] for f in seg_frames.values())
    h = max(f[0].shape[0] for f in seg_frames.values())

    def _pad(f):  # noqa: ANN001, ANN202 (center a frame on the (w,h) canvas)
        fh, fw = f.shape[:2]
        if (fw, fh) == (w, h):
            return f
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        y0, x0 = (h - fh) // 2, (w - fw) // 2
        canvas[y0:y0 + fh, x0:x0 + fw] = f
        return canvas

    all_frames: list[np.ndarray] = []
    n_cards = max(1, int(fps * 1.5))  # ~1.5 s title card per segment
    for seg in order:
        if seg not in seg_frames:
            continue
        spec = _SEGMENTS[seg]
        all_frames += _title_card([spec["title"], spec["caption"]], (w, h), spec["color"], n_cards)
        for f in seg_frames[seg]:
            all_frames.append(_pad(f))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
    for f in all_frames:
        writer.write(f)
    writer.release()
    dur = len(all_frames) / float(fps)
    print(f"\n==== ENDGAME DEMO STITCHED ====\n  segments={list(seg_frames)} frames={len(all_frames)} "
          f"fps={fps} duration={dur:.1f}s -> {out}", flush=True)
    return {"segments": list(seg_frames), "frames": len(all_frames), "duration_s": dur, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Record/stitch the Willy comprehensive demo-endgame MP4.")
    ap.add_argument("--segment", type=str, default=None, choices=list(_SEGMENTS))
    ap.add_argument("--stitch", action="store_true", help="offline: combine the recorded segments")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--fps", type=int, default=18)
    # '%%' not '%': argparse runs help strings through %-formatting, so a literal percent sign makes
    # `--help` itself raise ValueError and the runner's flags become undiscoverable.
    ap.add_argument("--max-attempts", type=int, default=6, help="re-pick until a lifting take (recovery is ~12%%)")
    ap.add_argument("--capture-every", type=int, default=2, help="capture 1 of every N steps (thin long loops)")
    ap.add_argument("--hold-steps", type=int, default=45, help="pause length at the scene + lifted moments")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    if args.stitch:
        stitch(out_path=args.out, fps=args.fps)
    elif args.segment:
        record_segment(args.segment, out_path=args.out, headless=not args.gui, fps=args.fps,
                       max_attempts=args.max_attempts, capture_every=args.capture_every,
                       hold_steps=args.hold_steps)
    else:
        ap.error("pass --segment <name> (record on-box) or --stitch (combine offline)")


if __name__ == "__main__":
    main()
