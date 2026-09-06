"""Combined KLT modality demo (Isaac): one video showing which gripper wins where.

Three filmed segments concatenated into one cinematic MP4, telling the complementary-modality story on
the real KLT geometry:
  1. Jaw, flat tray:    the 2F-85 picks a cube from a shallow tray it can reach       [succeeds]
  2. Jaw, deep KLT:     the 2F-85 tries the real full-depth (146 mm) small_KLT and
                        cannot reach the cube resting on its floor                     [fails]
  3. Suction, deep KLT: the suction cup descends the same deep KLT and lifts the cube  [succeeds]

Each segment is its own Isaac boot, because SimulationApp is a singleton, so this runs one segment per
invocation via --segment, and --concat then stitches the three clips plus title cards into the final
MP4. The deep segments show the real small_KLT visual mesh over the reliable fixture-box physics. Jaw
picks reuse ``build_service`` and ``service.pick()``; the suction pick reuses the render-safe kinematic
carry from ``run_suction_demo``.

On-box only. Run with Isaac's bundled python from the repo root:
    ...\\python.bat -m src.willy_sim.run_klt_combined_demo --segment jaw-flat --out logs/demo/klt_1_jaw_flat.mp4
    ...\\python.bat -m src.willy_sim.run_klt_combined_demo --segment jaw-deep --out logs/demo/klt_2_jaw_deep.mp4
    ...\\python.bat -m src.willy_sim.run_klt_combined_demo --segment suction-deep --out logs/demo/klt_3_suction.mp4
    python -m src.willy_sim.run_klt_combined_demo --concat logs/demo/klt_1_jaw_flat.mp4,logs/demo/klt_2_jaw_deep.mp4,logs/demo/klt_3_suction.mp4 --out logs/demo/klt_combined.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

CINE_EYE = [1.55, -1.45, 1.55]        # elevated 3/4 view that sees into a 146 mm-deep bin (the eye _h39d_binpick_record uses)
CINE_TARGET = [0.45, -0.02, 0.05]
CINE_RES = (1200, 820)
FPS = 24                               # playback rate of the encoded clip
CAPTURE_EVERY = 3                      # capture 1 of every N sim steps; larger means fewer frames
HOLD = 22                              # beat length at the opening, lifted and final states
KLT_Z_MM = 73.2                        # places the real small_KLT visual so its floor sits ~ the table z=0
KLT_Z_SHALLOW_MM = 35.1               # a Z-squashed KLT (floor at z=0) for the shallow-tray jaw segment
KLT_ZSCALE_SHALLOW = 0.48             # 0.48*146 ~= 70 mm deep, which the jaw reaches; same mesh, one look
GREEN = "the green cube"               # centred target (y=+5), where the fingers clear both walls


def _write_mp4(frames: list[np.ndarray], out: str, fps: int = FPS, keyframes: int = 6) -> float:
    """Encode captured RGB frames into an mp4 (BGR) plus a few keyframe PNGs; returns the duration (s)."""
    import cv2  # type: ignore[import-not-found]

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise SystemExit(f"no frames captured for {out}")
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))  # type: ignore[attr-defined]
    for f in frames:
        wr.write(np.ascontiguousarray(f[..., ::-1]))
    wr.release()
    for i, idx in enumerate(np.linspace(0, len(frames) - 1, keyframes).astype(int)):
        cv2.imwrite(str(p.with_name(f"{p.stem}_key{i}.png")), np.ascontiguousarray(frames[int(idx)][..., ::-1]))
    dur = len(frames) / float(fps)
    print(f"=== {out}: {len(frames)} frames @{fps}fps ({dur:.1f}s) (+{keyframes} PNGs) ===", flush=True)
    return dur


def _setup_cine(session: Any) -> tuple[Any, Any]:
    """Author the cinematic camera + return (camera, grab_fn) after a render warm-up."""
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    cine = Camera(prim_path="/World/Cameras/Cinematic",
                  position=np.array(CINE_EYE, dtype=np.float64), resolution=CINE_RES)
    cine.initialize()
    cine.add_distance_to_image_plane_to_frame()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001 (clipping best-effort)
        pass
    set_camera_view(eye=CINE_EYE, target=CINE_TARGET, camera_prim_path="/World/Cameras/Cinematic")
    app = getattr(session, "app", None)
    for _ in range(60):  # warm the render so the RGB annotator populates before capture
        session.step(render=True)
        if app is not None:
            app.update()

    def _grab() -> np.ndarray | None:
        fr = cine.get_current_frame()
        rgb = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgb is None else np.asarray(rgb)[..., :3].copy()

    return cine, _grab


def record_jaw(out: str, *, deep: bool, hold_steps: int = HOLD) -> int:
    """Film a 2F-85 jaw pick attempt.

    ``deep=False`` gives the shallow tray, where the pick succeeds; ``deep=True`` gives the real deep
    KLT, where it fails. Both use the same real small_KLT mesh as the sole visible bin, with the
    fixture walls hidden: the shallow segment Z-squashes it by ``KLT_ZSCALE_SHALLOW`` so the jaw can
    reach, and the deep segment is the full 146 mm KLT.
    """
    from src.willy_sim.run_dense_pick import build_service, require_robot
    from src.robot.core import JointPositions

    bin_h = 146.0 if deep else 70.0
    klt = (450.0, 0.0, KLT_Z_MM) if deep else (450.0, 0.0, KLT_Z_SHALLOW_MM, KLT_ZSCALE_SHALLOW)
    service, arm, gripper, handles, cfg, target_idx, target_label = build_service(
        headless=True, prompt=GREEN, enable_bin=True,
        bin_half_width_mm=146.5, bin_half_width_y_mm=94.7, bin_height_mm=bin_h, finger_tool=True,
        motion_planner="curobo", real_klt_bin=klt,
    )
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    robot = require_robot(cfg)
    park_q = np.asarray(robot.sim.park_joint_positions, dtype=np.float64)
    objs = [SingleRigidPrim(p) for (p, _l) in handles.object_specs]
    homes = [o.get_world_pose() for o in objs]
    session = arm.session
    app = getattr(session, "app", None)

    _cine, grab = _setup_cine(session)
    frames: list[np.ndarray] = []
    rec = {"on": False}
    n = {"v": 0}
    orig = session.step

    def cap(dt: Any = None, *, render: bool = False) -> None:  # noqa: ARG001
        orig(dt, render=True)
        if app is not None:
            app.update()
        if rec["on"]:
            n["v"] += 1
            if n["v"] % CAPTURE_EVERY == 0:
                f = grab()
                if f is not None:
                    frames.append(f)

    session.step = cap  # type: ignore[assignment]

    # Clean reset: open jaw, park (clean overhead perceive), re-home + settle every object.
    gripper.open()
    arm.move_to_joints(JointPositions(park_q))
    for o, (hp, hq) in zip(objs, homes):
        o.set_world_pose(position=np.asarray(hp, dtype=np.float64), orientation=np.asarray(hq, dtype=np.float64))
        try:
            o.set_linear_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
    session.step_n(60)

    rec["on"] = True
    for _ in range(hold_steps):  # opening beat
        session.step(render=True)
    report = service.pick()
    out_val = getattr(getattr(report, "outcome", None), "value", None)
    for _ in range(hold_steps):  # hold on the final state
        session.step(render=True)
    rec["on"] = False
    session.step = orig  # type: ignore[assignment]
    tgt_z = float(np.asarray(objs[target_idx].get_world_pose()[0])[2]) * 1000.0
    print(f"[jaw {'deep' if deep else 'flat'}] outcome={out_val} target_z={tgt_z:.1f}mm", flush=True)
    _write_mp4(frames, out)
    session.stop()
    return 0


def record_suction_deep(out: str, *, hold_steps: int = HOLD, lift_mm: float = 130.0,
                        seal_threshold: float = 0.5) -> int:
    """Film the suction cup picking the cube from the real full-depth KLT (render-safe kinematic carry)."""
    from src.config.schema.robot.sim_schema import SimObjectConfig

    from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
    from src.willy_sim.run_dense_pick import _make_bin_fixtures

    fixtures = _make_bin_fixtures(center_xy_mm=(450.0, 0.0), half_width_mm=94.7, half_width_y_mm=146.5,
                                  height_mm=146.0, thickness_mm=8.0)
    green = SimObjectConfig(name="klt_target", shape="cube", size_mm=(40.0, 40.0, 40.0),
                            position_mm=(450.0, 0.0, 20.0), mass_kg=0.15, color=(0.20, 0.65, 0.30))
    cell = bootstrap_sim_cell(None, headless=True, scene_kwargs={
        "objects_override": [green], "bin_walls": fixtures, "real_klt_bin": (450.0, 0.0, KLT_Z_MM)})
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    session = arm.session
    app = getattr(session, "app", None)
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)

    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import SuctionConfig, synthesize_suction_grasps

    from src.willy_sim.perception import GroundTruthPerceptionSource
    from src.willy_sim.scene import ARM_PRIM, WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import SUCTION_CUP, mount_visible_suction_cup

    import omni.usd  # type: ignore[import-not-found]

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    stage = omni.usd.get_context().get_stage()
    mount_visible_suction_cup(stage, ARM_PRIM, SUCTION_CUP, hide_jaw=True)

    _cine, grab = _setup_cine(session)
    obj = SingleRigidPrim(handles.object_prim_path)
    wrist = SingleRigidPrim(WRIST_LINK_PRIM)
    frames: list[np.ndarray] = []
    state: dict[str, Any] = {"on": False, "carry": False, "p_rel": None, "r_rel": None, "n": 0}
    orig = session.step

    def _carry() -> None:
        wp, wq = wrist.get_world_pose()
        r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
        pos = np.asarray(wp, dtype=np.float64) + r_w @ state["p_rel"]
        q = from_rotation_matrix(r_w @ state["r_rel"])
        obj.set_world_pose(position=pos, orientation=np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass

    def cap(dt: Any = None, *, render: bool = False) -> None:  # noqa: ARG001
        orig(dt, render=True)
        if state["carry"] and state["p_rel"] is not None:
            _carry()
        if state["on"]:
            state["n"] += 1
            if state["n"] % CAPTURE_EVERY == 0:
                if app is not None:
                    app.update()
                f = grab()
                if f is not None:
                    frames.append(f)

    session.step = cap  # type: ignore[assignment]

    def _hold(k: int) -> None:
        for _ in range(k):
            session.step(render=True)

    arm.move_joint(JointPositions(home_q))
    arm.session.step_n(10)
    z_before = float(np.asarray(obj.get_world_pose()[0])[2])
    state["on"] = True
    _hold(hold_steps)

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
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps), ground_truth_depth=False)
    frame = perception.acquire()
    mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
    tries = 0
    while (mask.ndim != 2 or not mask.any()) and tries < 10:
        arm.session.step_n(5)
        frame = perception.acquire()
        mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
        tries += 1
    grasps = synthesize_suction_grasps(
        mask, frame.depth_map, frame.intrinsics, camera_to_base=handles.camera_to_base, payload_mass_g=150.0,
        config=SuctionConfig(scorer=AnalyticalSuctionScorer(), max_results=8, min_quality=0.05))
    if not grasps or grasps[0].seal_score < seal_threshold:
        state["on"] = False
        session.step = orig  # type: ignore[assignment]
        session.stop()
        raise SystemExit(f"suction-deep: no sealable candidate (grasps={len(grasps)})")
    best = grasps[0]
    pos = np.asarray(best.position_mm, dtype=np.float64)
    approach = np.asarray(best.approach, dtype=np.float64)
    approach = approach / (float(np.linalg.norm(approach)) or 1.0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    closing = seed - approach * float(np.dot(seed, approach))
    closing = closing / (float(np.linalg.norm(closing)) or 1.0)
    quat = _quaternion_from_axes(closing, approach)

    arm.move_joint(JointPositions(home_q))
    arm.session.step_n(10)
    for label, gap in (("standoff", 60.0), ("contact", 22.0)):
        arm.move(Pose(position_mm=pos - approach * gap, quaternion_xyzw=quat, frame=Frame.BASE,
                      label=f"suction-{label}"))
    arm.session.step_n(6)

    wp, wq = wrist.get_world_pose()
    op, oq = obj.get_world_pose()
    r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
    r_o = to_rotation_matrix(np.array([oq[1], oq[2], oq[3], oq[0]], dtype=np.float64))
    state["p_rel"] = r_w.T @ (np.asarray(op, dtype=np.float64) - np.asarray(wp, dtype=np.float64))
    state["r_rel"] = r_w.T @ r_o
    state["carry"] = True
    _hold(6)
    arm.move(Pose(position_mm=pos - approach * 22.0 + (-approach) * float(lift_mm), quaternion_xyzw=quat,
                  frame=Frame.BASE, label="suction-lift"))
    _hold(hold_steps)
    z_lift = (float(np.asarray(obj.get_world_pose()[0])[2]) - z_before) * 1000.0
    state["on"] = False
    state["carry"] = False
    session.step = orig  # type: ignore[assignment]
    print(f"[suction deep] seal={best.seal_score:.3f} object_rose={z_lift:.1f}mm", flush=True)
    _write_mp4(frames, out)
    session.stop()
    return 0


def concat(clips: list[str], out: str, fps: int = FPS) -> int:
    """Stitch the segment clips into one MP4 with a title card before each (no Isaac needed)."""
    import cv2  # type: ignore[import-not-found]

    titles = ["1. Parallel-Greifer (2F-85); flache Schale: greift",
              "2. Parallel-Greifer; tiefer KLT: kommt nicht an den Boden",
              "3. Saug-Greifer; tiefer KLT: greift"]
    caps = [cv2.VideoCapture(c) for c in clips]
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))  # type: ignore[attr-defined]
    total = 0
    for ci, cap in enumerate(caps):
        card = np.zeros((h, w, 3), dtype=np.uint8)
        title = titles[ci] if ci < len(titles) else f"Segment {ci + 1}"
        cv2.putText(card, title, (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        for _ in range(int(fps * 1.0)):  # 1.0 s title card
            wr.write(card)
            total += 1
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            wr.write(fr)
            total += 1
        cap.release()
    wr.release()
    print(f"=== COMBINED: {total} frames @{fps}fps ({total / fps:.1f}s) -> {out} ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="H3.2 combined KLT modality demo (jaw flat / jaw deep / suction deep).")
    ap.add_argument("--segment", choices=["jaw-flat", "jaw-deep", "suction-deep"], default=None)
    ap.add_argument("--concat", type=str, default=None, help="comma-separated clip paths to stitch")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    if args.concat:
        return concat([c.strip() for c in args.concat.split(",")], args.out)
    if args.segment == "jaw-flat":
        return record_jaw(args.out, deep=False)
    if args.segment == "jaw-deep":
        return record_jaw(args.out, deep=True)
    if args.segment == "suction-deep":
        return record_suction_deep(args.out)
    ap.error("pass --segment {jaw-flat,jaw-deep,suction-deep} or --concat <clips>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
