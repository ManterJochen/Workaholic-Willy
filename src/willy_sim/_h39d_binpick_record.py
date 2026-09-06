"""Visual diagnosis: film one bin-pick attempt to an MP4.

Makes visible a failure that only appears inside a bin: ``exec_failed`` with ``wall_reject=None``
while the same pick without a bin lifts. Reuses ``build_service`` and the ``run_build_stack``
cine-record pattern from ``run_dense_pick``.
"""
from __future__ import annotations

import sys

import numpy as np

from src.willy_sim.run_dense_pick import build_service, require_robot
from src.robot.core import JointPositions


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "logs/demo/h39d_binpick_diag.mp4"
    bin_h = float(sys.argv[2]) if len(sys.argv) > 2 else 146.0  # deep KLT by default
    service, arm, gripper, handles, cfg, target_idx, target_label = build_service(
        headless=True, enable_bin=True, bin_half_width_mm=110.0, bin_half_width_y_mm=130.0,
        bin_height_mm=bin_h, finger_tool=True,
    )
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    robot = require_robot(cfg)
    park_q = np.asarray(robot.sim.park_joint_positions, dtype=np.float64)
    prims = [p for (p, _l) in handles.object_specs]
    objs = [SingleRigidPrim(p) for p in prims]
    homes = [o.get_world_pose() for o in objs]
    session = arm.session
    app = getattr(session, "app", None)

    # Cinematic 3/4 view, elevated enough to see into a 146 mm-deep bin. Bin centre ~ (0.45, 0); red ~ (0.45,-0.095).
    eye, target = [1.55, -1.45, 1.55], [0.45, -0.06, 0.02]
    cine = Camera(prim_path="/World/Cameras/Cinematic",
                  position=np.array(eye, dtype=np.float64), resolution=(1200, 820))
    cine.initialize()
    cine.add_distance_to_image_plane_to_frame()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    set_camera_view(eye=eye, target=target, camera_prim_path="/World/Cameras/Cinematic")

    frames: list[np.ndarray] = []
    rec = {"v": False}
    n = {"v": 0}
    orig = session.step

    def cap(dt=None, *, render=False):  # noqa: ANN001, ANN202
        orig(dt, render=True)
        if app is not None:
            app.update()
        if rec["v"]:
            n["v"] += 1
            if n["v"] % 2 == 0:  # every 2 steps gives a smooth slow clip of the single pick
                fr = cine.get_current_frame()
                rgb = fr.get("rgb") if isinstance(fr, dict) else None
                if rgb is not None:
                    frames.append(np.asarray(rgb)[..., :3].copy())

    session.step = cap  # type: ignore[assignment]
    for _ in range(60):
        session.step(render=True)

    # clean reset: open jaw, park (clean overhead perceive), re-home + settle every object
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
    print(f"[rec] target={target_label!r} pre-pick pos_mm="
          f"{np.round(np.asarray(objs[target_idx].get_world_pose()[0]) * 1000.0, 1)}", flush=True)

    rec["v"] = True  # Film the pick
    report = service.pick()
    out_val = getattr(getattr(report, "outcome", None), "value", None)
    print(f"[rec] pick outcome={out_val} post-pick target pos_mm="
          f"{np.round(np.asarray(objs[target_idx].get_world_pose()[0]) * 1000.0, 1)}", flush=True)
    for _ in range(40):  # hold (still filming) so the final state is visible
        session.step(render=True)
    rec["v"] = False

    if frames:
        from pathlib import Path

        import cv2  # type: ignore[import-not-found]
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 18.0, (w, h))  # type: ignore[attr-defined]
        for f in frames:
            wr.write(np.ascontiguousarray(f[..., ::-1]))
        wr.release()
        for i, idx in enumerate(np.linspace(0, len(frames) - 1, 8).astype(int)):  # 8 keyframes to scrub
            cv2.imwrite(str(p.with_name(f"{p.stem}_key{i}.png")),
                        np.ascontiguousarray(frames[int(idx)][..., ::-1]))
        print(f"=== MP4: {len(frames)} frames @18fps ({len(frames)/18.0:.1f}s) -> {p} (+8 PNGs) ===", flush=True)
    else:
        print("=== MP4: no frames captured ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
