"""cuRobo motion-planning demo (Isaac): the positive bin-pick, side by side.

Records the same blocking scene twice, a red cube with a blocker 5 mm off its +X face in a Y-row of
neighbours, and stitches the two takes side by side so the contrast is the message:

* ``blind``:  the default open-loop straight-line approach (``motion_planner="ik"``). The close yaws
              onto base-X, the reachable blind-IK branch, and the jaw drives straight into the +X
              column, ramming the blocker out of the bin. There is no collision-aware planning.
* ``curobo``: the planned approach (``motion_planner="curobo"``). cuRobo plans the gripper, as
              mesh-fit tool0 spheres, collision-free around the registered obstacles, threads the
              corridor-clear -Y close, lifts the cube and leaves the blocker untouched.

Two clean processes (Isaac plus the cuRobo sidecar), then an offline side-by-side compose that needs
no Isaac:

    ...python.bat -m src.willy_sim.run_curobo_demo --record blind
    ...python.bat -m src.willy_sim.run_curobo_demo --record curobo
    python -m src.willy_sim.run_curobo_demo --compose

On-box only (Isaac). A sim capability; motion safety is not certified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.robot.safety.planning.world import planner_cuboid

from src.willy_sim.run_dense_demo_endgame import CINE_POS_M, CINE_RES, _banner
from src.willy_sim.run_dense_pick import _blocking_specs, build_service

_DEMO_DIR = "logs/demo/curobo"
_CINE_TARGET = (0.47, -0.09, 0.04)  # frame the red target + the +X blocker
_PLANNERS: dict[str, dict] = {
    "blind": {"title": "BLIND straight-line  (no motion planning)",
              "caption": "open-loop IK -> the jaw drives into the +X column", "color": (60, 60, 235)},
    "curobo": {"title": "cuRobo motion planning",
               "caption": "collision-free plan -> threads the corridor-clear grasp", "color": (90, 220, 90)},
}


def record(planner: str, *, out_path: str | None = None, headless: bool = True, fps: int = 18,
           hold_steps: int = 45, capture_every: int = 2) -> dict:
    """Record one blocking pick (``planner`` = 'blind' | 'curobo') as a cinematic MP4 + return a summary."""
    import cv2  # type: ignore[import-not-found]

    spec = _PLANNERS[planner]
    out = Path(out_path or f"{_DEMO_DIR}/seg_{planner}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    is_curobo = planner == "curobo"

    # blind: default IK, align_closing_to_base_x on and no G12, so it rams. curobo: align off, and G12
    # selects the corridor-clear -Y close at an 8 mm margin, with an interpolated vertical retreat.
    service, arm, gripper, handles, cfg, target_idx, target_label = build_service(
        prompt="the red cube", headless=headless, blocking=True,
        motion_planner="curobo" if is_curobo else "ik",
        enable_g12=is_curobo, approach_margin_mm=8.0, retreat_steps=5 if is_curobo else 1,
    )
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
    set_camera_view(eye=list(CINE_POS_M), target=list(_CINE_TARGET), camera_prim_path="/World/Cameras/Cinematic")
    for _ in range(60):
        session.step(render=True)
        if app is not None:
            app.update()

    def _grab():  # noqa: ANN202
        fr = cine.get_current_frame()
        rgba = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgba is None else np.asarray(rgba)[..., :3].copy()

    park_q = np.asarray(cfg.robot.sim.park_joint_positions, dtype=np.float64)
    specs = _blocking_specs()
    labels = [lbl for (_p, lbl) in handles.object_specs]
    objs = [SingleRigidPrim(p) for (p, _l) in handles.object_specs]
    homes = [o.get_world_pose() for o in objs]
    b_idx = next((i for i, lbl in enumerate(labels) if "blocker" in lbl.lower()), None)
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
        arm.session.step_n(80)

    _reset()
    if is_curobo:  # register the neighbour bodies + a floor so cuRobo plans the gripper clear of them
        cub = []
        for i, lbl in enumerate(labels):
            if i == target_idx:
                continue
            pm, qw = objs[i].get_world_pose()
            d = specs[i].size_mm if i < len(specs) else (40.0, 40.0, 40.0)
            # Written out rather than through `planner_cuboid`, and this is the one place
            # where that is right: these poses come from the simulator with a full
            # orientation, and the helper writes a yaw about base Z because that is what a
            # fitted box has. The pose here is already metres and WXYZ, straight off the prim.
            cub.append({"name": lbl.replace(" ", "_"), "dims_m": [float(x) / 1000.0 for x in d],
                        "pose": [float(pm[0]), float(pm[1]), float(pm[2]),
                                 float(qw[0]), float(qw[1]), float(qw[2]), float(qw[3])]})
        cub.append(planner_cuboid("floor", (0.0, 0.0, -26.0), (2000.0, 2000.0, 50.0)))
        arm.set_curobo_world(cub)

    home_z = float(np.asarray(homes[target_idx][0])[2])
    b0 = np.asarray(objs[b_idx].get_world_pose()[0], np.float64) * 1000.0 if b_idx is not None else np.zeros(3)
    state["on"] = True
    for _ in range(hold_steps):
        session.step(render=True)
    try:
        report = service.pick()
        outcome = getattr(getattr(report, "outcome", None), "value", None)
    except Exception as exc:  # noqa: BLE001 (a raised pick still yields a watchable take)
        outcome = f"raised:{type(exc).__name__}"
    for _ in range(hold_steps):
        session.step(render=True)
    state["on"] = False
    session.step = orig_step  # type: ignore[assignment]

    z_lift = (float(np.asarray(objs[target_idx].get_world_pose()[0])[2]) - home_z) * 1000.0
    b1 = np.asarray(objs[b_idx].get_world_pose()[0], np.float64) * 1000.0 if b_idx is not None else np.zeros(3)
    blocker_moved = float(np.linalg.norm(b1 - b0))
    # The blind ram launches the cube through the bin, so its z_lift is meaningless; the shove is what
    # the caption reports. For cuRobo the caption reports the real lift and the untouched blocker.
    if is_curobo:
        live = f"cube +{z_lift:.0f} mm   blocker {blocker_moved:.0f} mm (untouched)"
    else:
        live = f"blocker SHOVED {blocker_moved:.0f} mm   ->  scene wrecked (no planning)"

    if not frames:
        raise SystemExit("no cinematic frames captured")
    composed = [_banner(np.ascontiguousarray(f[..., ::-1]), spec["title"], spec["caption"], live, spec["color"])
                for f in frames]
    h, w = composed[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))  # type: ignore[attr-defined]
    for c in composed:
        writer.write(c)
    writer.release()
    print(f"\n==== {planner} RECORDED ====\n  outcome={outcome} lift={z_lift:.1f}mm "
          f"blocker_moved={blocker_moved:.1f}mm frames={len(composed)} -> {out}", flush=True)
    try:
        arm.disconnect()
    except Exception:  # noqa: BLE001
        pass
    return {"planner": planner, "outcome": outcome, "lift_mm": z_lift, "blocker_moved_mm": blocker_moved,
            "frames": len(composed), "out": str(out)}


def _read_mp4(path: Path) -> list[np.ndarray]:
    import cv2  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def _subsample(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    """Uniform-stride subsample to at most ``target`` frames (keeps the whole pick, just thinner)."""
    if len(frames) <= target:
        return frames
    idx = np.linspace(0, len(frames) - 1, num=target).round().astype(int)
    return [frames[i] for i in idx]


def compose(*, demo_dir: str = _DEMO_DIR, out_path: str | None = None, fps: int = 18,
            max_panel_frames: int = 380) -> dict:
    """Compose the two takes side by side, blind against cuRobo. Offline: no Isaac needed.

    Each take is subsampled to a tight, balanced length so neither panel just freezes, then padded to
    equal length, under a header.
    """
    import cv2  # type: ignore[import-not-found]

    out = Path(out_path or f"{demo_dir}/curobo_binpick_demo.mp4")
    left = _subsample(_read_mp4(Path(demo_dir) / "seg_blind.mp4"), max_panel_frames)
    right = _subsample(_read_mp4(Path(demo_dir) / "seg_curobo.mp4"), max_panel_frames)
    if not left or not right:
        raise SystemExit(f"missing takes: blind={len(left)} curobo={len(right)} (record both first)")
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))    # freeze the last frame so both panels end together
    right += [right[-1]] * (n - len(right))
    h = min(left[0].shape[0], right[0].shape[0])

    def _fit(f):  # noqa: ANN001, ANN202
        return f if f.shape[0] == h else cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))

    pairs = [np.hstack([_fit(a), _fit(b)]) for a, b in zip(left, right)]
    bh, bw = pairs[0].shape[:2]
    header = 54
    canvas0 = np.full((bh + header, bw, 3), 18, dtype=np.uint8)
    title = "MOTION PLANNING: blind straight-line  vs  cuRobo  (same scene, same target)"
    (tw, _t), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (bw, bh + header))  # type: ignore[attr-defined]
    for p in pairs:
        c = canvas0.copy()
        cv2.putText(c, title, ((bw - tw) // 2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
        c[header:, :, :] = p
        writer.write(c)
    writer.release()
    print(f"\n==== composed {n} frames -> {out} ====", flush=True)
    return {"frames": n, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="cuRobo motion-planning bin-pick demo (blind vs cuRobo).")
    ap.add_argument("--record", choices=list(_PLANNERS), help="record one take (its own Isaac process)")
    ap.add_argument("--compose", action="store_true", help="stitch the two takes side by side (offline)")
    ap.add_argument("--no-headless", action="store_true")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--capture-every", type=int, default=2)
    args = ap.parse_args()
    if args.record:
        record(args.record, headless=not args.no_headless, fps=args.fps, capture_every=args.capture_every)
    elif args.compose:
        compose(fps=args.fps)
    else:
        ap.error("pass --record blind | --record curobo | --compose")


if __name__ == "__main__":
    main()
