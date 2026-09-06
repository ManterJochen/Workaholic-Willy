"""Suction pick demo recorder: a watchable cinematic MP4 of the seal-gated suction pick.

Films the suction pick (perceive, seal-gate with the analytical model, descend, attach, lift) from a
fixed cinematic 3/4 camera and encodes a slow MP4. It follows the ``run_eih_demo`` recorder pattern:
Isaac ``set_camera_view`` for the only convention-correct orientation, and ``session.step`` wrapped so
every step is captured.

The object is carried kinematically here, not by the ``--pick`` runtime weld. Adding a physics joint
mid-play freezes Isaac's render product, so the filmed lift would stay frozen at the contact pose while
the physics lifts the object. The film instead tracks the object to the wrist every filmed step, a
render-safe kinematic carry, still gated by the analytical seal: the decision, where the seal goes and
whether it seals at all, is the real model, and only the visual hold is a surrogate. The physics-real
lift runs under ``run_suction_pick --pick``.

On-box only (needs Isaac). Run with Isaac's bundled python from the repo root:
    <isaac-sim>\\python.bat -m src.willy_sim.run_suction_demo --fps 18
    ...\\python.bat -m src.willy_sim.run_suction_demo --gui          # also open the live window
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.willy_sim.harness.bootstrap import bootstrap_sim_cell

DEFAULT_OUT = "logs/demo/suction_pick_demo.mp4"
# Cinematic camera (metres): an elevated front-right 3/4 view of the cell, aimed at the workspace. The
# same framing as run_eih_demo, which is convention-correct for the UR5e cell.
CINE_POS_M = (2.05, -1.82, 1.68)
CINE_TARGET_M = (0.45, 0.0, 0.14)
CINE_RES = (1280, 720)


def record_demo(*, out_path: str = DEFAULT_OUT, headless: bool = True, fps: int = 18,
                capture_every: int = 1, hold_steps: int = 40, lift_mm: float = 130.0,
                seal_threshold: float = 0.5) -> dict:
    """Record one seal-gated suction pick (kinematic carry) as a cinematic MP4. Returns a summary dict."""
    import cv2  # type: ignore[import-not-found]

    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import (
        SuctionConfig,
        synthesize_suction_grasps,
    )

    from src.willy_sim.perception import GroundTruthPerceptionSource
    from src.willy_sim.scene import ARM_PRIM, WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import SUCTION_CUP, mount_visible_suction_cup

    # bootstrap_sim_cell starts the SimulationApp; the isaacsim.* namespaces only exist after that.
    cell = bootstrap_sim_cell(None, headless=headless)
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    session = arm.session
    app = getattr(session, "app", None)
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)

    # Show a real suction cup (visual only) and hide the baked 2F-85 jaw, for a coherent suction cell. The
    # 2F-85 stays in the articulation for IK health; this is pure render, so IK and the carry are unaffected.
    import omni.usd  # type: ignore[import-not-found]
    _stage = omni.usd.get_context().get_stage()
    _cup_root = mount_visible_suction_cup(_stage, ARM_PRIM, SUCTION_CUP, hide_jaw=True)
    print(f"visible suction cup authored at {_cup_root}", flush=True)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    # --- cinematic camera (orient via Isaac's canonical set_camera_view) -----
    cine = Camera(prim_path="/World/Cameras/Cinematic",
                  position=np.array(CINE_POS_M, dtype=np.float64), resolution=CINE_RES)
    cine.initialize()
    cine.add_distance_to_image_plane_to_frame()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001 (clipping best-effort)
        pass
    set_camera_view(eye=list(CINE_POS_M), target=list(CINE_TARGET_M),
                    camera_prim_path="/World/Cameras/Cinematic")
    for _ in range(60):  # warm up the render so the RGB annotator populates before capture
        session.step(render=True)
        if app is not None:
            app.update()

    def _grab() -> np.ndarray | None:
        fr = cine.get_current_frame()
        rgba = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgba is None else np.asarray(rgba)[..., :3].copy()

    probe = _grab()
    print(f"cinematic cam ready: frame {None if probe is None else probe.shape}", flush=True)

    obj = SingleRigidPrim(handles.object_prim_path)
    wrist = SingleRigidPrim(WRIST_LINK_PRIM)

    # --- frame capture + the render-safe kinematic carry --------------------
    frames: list[np.ndarray] = []
    state: dict[str, Any] = {"on": False, "n": 0, "carry": False, "p_rel": None, "r_rel": None}
    orig_step = session.step

    def _carry_object() -> None:
        # Track the object to the wrist so the render shows it riding with the tool (no runtime joint).
        wp, wq = wrist.get_world_pose()
        r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
        pos = np.asarray(wp, dtype=np.float64) + r_w @ state["p_rel"]
        q = from_rotation_matrix(r_w @ state["r_rel"])  # xyzw
        obj.set_world_pose(position=pos, orientation=np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001 (velocity zeroing is best-effort)
            pass

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001, ARG001
        orig_step(dt_s, render=True)
        if state["carry"] and state["p_rel"] is not None:
            _carry_object()
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

    # Start at the usual home pose (the raw post-boot articulation pose is not the config home).
    arm.move_joint(JointPositions(home_q))
    arm.session.step_n(10)
    z_before = float(np.asarray(obj.get_world_pose()[0])[2])
    state["on"] = True
    _hold(hold_steps)  # opening beat

    # --- perceive the true object-top geometry (park + near-clip) -----------
    _park = getattr(sim, "park_joint_positions", None)
    park_q = np.asarray(_park if _park is not None else sim.home_joint_positions, dtype=np.float64)
    arm.move_joint(JointPositions(park_q))
    arm.session.step_n(15)
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    perception = GroundTruthPerceptionSource(
        camera=handles.camera, target_prim_path=handles.object_prim_path, session=arm.session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps), ground_truth_depth=False,
    )
    frame = perception.acquire()
    mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
    tries = 0
    while (mask.ndim != 2 or not mask.any()) and tries < 10:
        arm.session.step_n(5)
        frame = perception.acquire()
        mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
        tries += 1

    payload = float(getattr(getattr(sim.scene_setup, "object", None), "mass_kg", 0.2)) * 1000.0
    grasps = synthesize_suction_grasps(
        mask, frame.depth_map, frame.intrinsics,
        camera_to_base=handles.camera_to_base, payload_mass_g=payload,
        config=SuctionConfig(scorer=AnalyticalSuctionScorer(), max_results=8, min_quality=0.05),
    )
    if not grasps or grasps[0].seal_score < seal_threshold:
        state["on"] = False
        session.step = orig_step  # type: ignore[assignment]
        session.stop()
        raise SystemExit(f"suction demo: no sealable candidate (grasps={len(grasps)}); nothing to film")
    best = grasps[0]
    pos = np.asarray(best.position_mm, dtype=np.float64)
    approach = np.asarray(best.approach, dtype=np.float64)
    approach = approach / (float(np.linalg.norm(approach)) or 1.0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    closing = seed - approach * float(np.dot(seed, approach))
    closing = closing / (float(np.linalg.norm(closing)) or 1.0)
    quat = _quaternion_from_axes(closing, approach)

    # --- approach: home, standoff, contact (filmed, carry off) --------------
    arm.move_joint(JointPositions(home_q))
    arm.session.step_n(10)
    for label, gap in (("standoff", 60.0), ("contact", 22.0)):
        arm.move(Pose(position_mm=pos - approach * gap, quaternion_xyzw=quat, frame=Frame.BASE,
                      label=f"suction-{label}"))
    arm.session.step_n(6)

    # --- engage (vacuum on): capture the object pose in the wrist frame + start the carry ----
    wp, wq = wrist.get_world_pose()
    op, oq = obj.get_world_pose()
    r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
    r_o = to_rotation_matrix(np.array([oq[1], oq[2], oq[3], oq[0]], dtype=np.float64))
    state["p_rel"] = r_w.T @ (np.asarray(op, dtype=np.float64) - np.asarray(wp, dtype=np.float64))
    state["r_rel"] = r_w.T @ r_o
    state["carry"] = True
    _hold(6)  # a beat at contact with the seal engaged

    # --- lift (up along -approach): the carry hook rides the object with the tool every step ----
    arm.move(Pose(position_mm=pos - approach * 22.0 + (-approach) * float(lift_mm),
                  quaternion_xyzw=quat, frame=Frame.BASE, label="suction-lift"))
    _hold(hold_steps)  # hold on the lifted object
    z_lift = (float(np.asarray(obj.get_world_pose()[0])[2]) - z_before) * 1000.0
    state["on"] = False
    state["carry"] = False
    session.step = orig_step  # type: ignore[assignment]

    # --- encode MP4 + a few keyframe PNGs -----------------------------------
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        session.stop()
        raise SystemExit("no frames captured; cinematic RGB never populated")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 ships no stubs
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
    for f in frames:
        writer.write(f[..., ::-1])  # RGB -> BGR
    writer.release()
    for i, idx in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        cv2.imwrite(str(out.with_name(f"{out.stem}_key{i}.png")), frames[idx][..., ::-1])

    dur = len(frames) / float(fps)
    print("\n==== SUCTION DEMO RECORDED ====", flush=True)
    print(f"  seal={best.seal_score:.3f} object_rose={z_lift:.1f} mm | frames={len(frames)} fps={fps} "
          f"duration={dur:.1f}s", flush=True)
    print(f"  saved {out}  (+ 4 keyframe PNGs alongside)", flush=True)
    session.stop()
    return {"seal": best.seal_score, "frames": len(frames), "fps": fps, "duration_s": dur,
            "lift_mm": z_lift, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a cinematic MP4 of the Willy suction pick (Isaac).")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=18, help="playback fps (lower = slower/easier to watch)")
    ap.add_argument("--capture-every", type=int, default=1, help="capture 1 of every N sim steps")
    ap.add_argument("--hold-steps", type=int, default=40, help="pause length at the opening + lifted beats")
    ap.add_argument("--lift-mm", type=float, default=130.0)
    ap.add_argument("--gui", action="store_true", help="also open the live window")
    args = ap.parse_args()
    record_demo(out_path=args.out, headless=not args.gui, fps=args.fps,
                capture_every=args.capture_every, hold_steps=args.hold_steps, lift_mm=args.lift_mm)


if __name__ == "__main__":
    main()
