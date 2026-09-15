"""4g.0: where the physical hand is, against every frame the guard and the planner place a hand in.

Whether the guard's hand points where the tool does rests on which axis is the approach. 4a.7 measured the
guard's hand 108 degrees off cuRobo's tool0 +Z, and the recorded evidence does not agree about the axis:

* the committed ur5e bundle puts `wrist_3` on frame 6 as a cylinder about Z and the 2F-85 arrays along +Y;
* every sphere map is fitted from those arrays and hangs on cuRobo's tool0, which 4a.7 measured as frame 6;
* the sim driver records that the gripper approaches along Lula's tool0 +Y, and hands cuRobo a tool0 pose
  target in that convention, and its picks pass.

So this reads the hand out of the physics scene itself, where no frame name can mislead, and asks each frame
where that hand is. The answer is two distances: how far the guard's hand (the bundle arrays
placed on frame 6) and the planner's hand (the sphere map placed on cuRobo's tool0) sit from the physical one.

Nothing is changed and nothing moves under a planner: the articulation is set to each joint vector directly.

Usage, on the box and alone on it (Isaac python)::

    python.bat scripts/isaac/probe_hand_frames.py --json logs/step4/probe_4g0_hand_frames.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

#: The joint vectors of 4a.7 after the cell's own home, so the two probes are read at the same places.
_EXTRA_JOINTS = (
    (0.5, -1.2, 1.3, -1.7, -1.57, 0.3),
    (-0.4, -1.8, 1.9, -1.2, -1.4, -0.6),
)
#: The arm's own rigid bodies. Every other rigid body under the arm prim is the hand.
_ARM_BODIES = frozenset({
    "world", "base_link", "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
})
_BUNDLE = _ROOT / "src" / "robot" / "safety" / "data" / "ur5e_collision_meshes.npz"
_MAP = _ROOT / "src" / "robot" / "safety" / "planning" / "robot" / "robotiq_2f85_gripper_spheres.yml"


# ------------------------------------------------------------------------------------------------ geometry
def _yaw(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _quat_wxyz_matrix(wxyz: Any) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _transform(rotation: np.ndarray, translation_mm: Any) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64)
    out[:3, 3] = np.asarray(translation_mm, dtype=np.float64)
    return out


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ np.asarray(points, dtype=np.float64).T).T + transform[:3, 3]


def _local(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3].T @ (np.asarray(point, dtype=np.float64) - transform[:3, 3])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / (na * nb)))))


def _rotation_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    cosine = (float(np.trace(r1.T @ r2)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _thin_axis(points: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """The direction of least spread of a point set, signed toward ``toward``: a disc's axis."""
    centred = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    axis = vectors[:, 0]
    return axis if float(np.dot(axis, toward - points.mean(axis=0))) >= 0.0 else -axis


def _round(value: Any, digits: int = 2) -> Any:
    return np.round(np.asarray(value, dtype=np.float64), digits).tolist()


# ---------------------------------------------------------------------------------------------------- USD
def _usd_world(prim: Any) -> np.ndarray:
    from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64).T


def _rigid_bodies(stage: Any, arm_prim: str) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]

    found: dict[str, Any] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(arm_prim), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            found[str(prim.GetPath())] = prim
    return found


def _body_geometry(body: Any, metres_per_unit: float) -> tuple[np.ndarray, str]:
    """The body's own mesh points in the body frame, millimetres, collision meshes first.

    The mesh-to-body transform is authored and static, so it is read from USD whatever the physics state;
    only the body's world pose is taken from the running scene.
    """
    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

    body_world = _usd_world(body)
    inverse = np.linalg.inv(body_world)
    for kind in ("collision", "visual"):
        chunks: list[np.ndarray] = []
        iterator = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for prim in iterator:
            if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                iterator.PruneChildren()  # a nested body is counted as itself, not as part of this one
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if kind == "collision" and not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if not points:
                continue
            raw = np.asarray([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
            relative = inverse @ _usd_world(prim)
            chunks.append(_apply(relative, raw) * metres_per_unit * 1000.0)
        if chunks:
            return np.vstack(chunks), kind
    return np.zeros((0, 3), dtype=np.float64), "none"


def _named(stage: Any, arm_prim: str, names: tuple[str, ...]) -> dict[str, Any]:
    from pxr import Usd  # type: ignore[import-not-found]

    out: dict[str, Any] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(arm_prim), Usd.TraverseInstanceProxies()):
        if prim.GetName() in names and prim.GetName() not in out:
            out[prim.GetName()] = prim
    return out


# ---------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_hand_frames", description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=str, default="logs/step4/probe_4g0_hand_frames.json")
    args = parser.parse_args(argv)

    from src.willy_sim.harness.bootstrap import bootstrap_sim_cell

    cell = bootstrap_sim_cell(None, headless=True, safety=False)

    import omni.usd  # type: ignore[import-not-found]
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from pxr import UsdGeom  # type: ignore[import-not-found]

    import yaml

    from src.robot.drivers.ur.tool_frame import tool_frame_matrix
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm
    from src.willy_sim.scene.constants import ARM_PRIM

    arm = cell.arm
    robot = cell.robot
    stage = omni.usd.get_context().get_stage()
    metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    yaw_deg = float(robot.safety.self_collision.kinematics_base_yaw_deg)
    model = str(robot.safety.self_collision.kinematics_model or "ur5e")
    arm_names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

    bodies = _rigid_bodies(stage, ARM_PRIM)
    hand_paths = sorted(p for p in bodies if p.rsplit("/", 1)[-1] not in _ARM_BODIES)
    wrist_paths = [p for p in bodies if p.rsplit("/", 1)[-1] == "wrist_3_link"]
    report: dict[str, Any] = {
        "model": model, "base_yaw_deg": yaw_deg, "metres_per_unit": metres_per_unit,
        "rigid_bodies": sorted(p.rsplit("/", 1)[-1] for p in bodies),
        "hand_bodies": [p.rsplit("/", 1)[-1] for p in hand_paths],
        "poses": [],
    }
    if not hand_paths or len(wrist_paths) != 1:
        report["why"] = "the scene holds no hand body, or not exactly one wrist_3_link, so there is nothing to place"
        _write(args.json, report)
        print(json.dumps(report, indent=2))
        return 2

    geometry = {p: _body_geometry(bodies[p], metres_per_unit) for p in [*hand_paths, wrist_paths[0]]}
    report["geometry_kind"] = {p.rsplit("/", 1)[-1]: geometry[p][1] for p in geometry}
    report["geometry_points"] = {p.rsplit("/", 1)[-1]: int(len(geometry[p][0])) for p in geometry}
    frames_named = _named(stage, ARM_PRIM, ("tool0", "flange"))
    inner = [p for p in hand_paths if "inner_finger" in p.rsplit("/", 1)[-1] and "knuckle" not in p]

    with np.load(_BUNDLE) as data:
        guard_hand = np.vstack([np.asarray(data[f"{k}__v"], dtype=np.float64) for k in ("gripper", "lfinger", "rfinger")])
    spheres = yaml.safe_load(_MAP.read_text(encoding="utf-8"))["collision_spheres"]["tool0"]
    planner_hand = np.asarray([s["center"] for s in spheres], dtype=np.float64) * 1000.0
    tool = robot.gripper.tool_frame
    tool_T = tool_frame_matrix(tool.offset_mm, tool.rotation_quat_xyzw)

    client = arm._get_curobo_client()  # noqa: SLF001 (the sim's own planner client, so its tool0 is the one it plans with)
    home = tuple(float(v) for v in (robot.sim.home_joint_positions or ()))
    vectors = ([home] if len(home) == 6 else []) + [tuple(q) for q in _EXTRA_JOINTS]
    subset = arm._arm_subset  # noqa: SLF001 (set the articulation directly: nothing plans, nothing is judged)
    solver = arm._kin_solver  # noqa: SLF001
    assert subset is not None and solver is not None

    for joints in vectors:
        q = np.asarray(joints, dtype=np.float64)
        subset.set_joint_positions(q)
        subset.apply_action(joint_positions=q)
        arm.session.step_n(3)
        read_back = np.asarray(subset.get_joint_positions(), dtype=np.float64)

        def world_of(path: str) -> tuple[np.ndarray, np.ndarray]:
            pos_m, quat = SingleRigidPrim(path).get_world_pose()
            physics = _transform(_quat_wxyz_matrix(quat), np.asarray(pos_m, dtype=np.float64) * 1000.0)
            usd = _usd_world(bodies[path])
            usd[:3, 3] = usd[:3, 3] * metres_per_unit * 1000.0
            return physics, usd

        placed: dict[str, np.ndarray] = {}
        usd_disagreement_mm = 0.0
        for path in geometry:
            physics, usd = world_of(path)
            usd_disagreement_mm = max(usd_disagreement_mm, float(np.linalg.norm(physics[:3, 3] - usd[:3, 3])))
            placed[path] = _apply(physics, geometry[path][0])
        hand_world = np.vstack([placed[p] for p in hand_paths])
        hand_centroid = hand_world.mean(axis=0)
        wrist_world = placed[wrist_paths[0]]

        links = ur_link_transforms_mm(model, q)
        assert links is not None
        rb = _yaw(yaw_deg)
        dh6 = _transform(rb @ links[6][:3, :3], rb @ links[6][:3, 3])
        lula_pos_m, lula_rot = solver.compute_forward_kinematics("tool0", q)
        lula = _transform(np.asarray(lula_rot, dtype=np.float64), np.asarray(lula_pos_m, dtype=np.float64) * 1000.0)
        fk = client.fk([float(q[arm_names.index(n)]) for n in client.joint_names])
        frames: dict[str, np.ndarray] = {"dh_frame_6": dh6, "lula_tool0": lula}
        if fk is not None:
            frames["curobo_tool0"] = _transform(_quat_wxyz_matrix(fk[1]), np.asarray(fk[0], dtype=np.float64) * 1000.0)
        for name, prim in frames_named.items():
            parent = str(prim.GetParent().GetPath())
            if parent in bodies:  # a frame prim under a body: its authored offset on the body's physics pose
                physics, _ = world_of(parent)
                offset = np.linalg.inv(_usd_world(bodies[parent])) @ _usd_world(prim)
                offset[:3, 3] = offset[:3, 3] * metres_per_unit * 1000.0
                frames[f"usd_{name}"] = physics @ offset

        wrist_axis = _thin_axis(wrist_world, hand_centroid)
        per_frame = {}
        for name, frame in frames.items():
            direction = _local(frame, hand_centroid)
            per_frame[name] = {
                "origin_from_dh6_mm": round(float(np.linalg.norm(frame[:3, 3] - dh6[:3, 3])), 3),
                "rotation_from_dh6_deg": round(_rotation_deg(dh6[:3, :3], frame[:3, :3]), 3),
                "hand_centroid_in_frame_mm": _round(direction),
                "hand_angle_to_axes_deg": {
                    axis: round(_angle_deg(direction, np.eye(3)[i]), 2) for i, axis in enumerate(("+X", "+Y", "+Z"))
                },
                "wrist3_axis_in_frame": _round(frame[:3, :3].T @ wrist_axis, 4),
            }

        guard_world = _apply(dh6, guard_hand)
        pose: dict[str, Any] = {
            "joints": _round(q, 4),
            "joint_readback_error_rad": round(float(np.max(np.abs(read_back - q))), 5),
            "usd_vs_physics_body_origin_mm": round(usd_disagreement_mm, 3),
            "hand_centroid_world_mm": _round(hand_centroid),
            "wrist3_axis_to_hand_deg": round(_angle_deg(wrist_axis, hand_centroid - wrist_world.mean(axis=0)), 2),
            "frames": per_frame,
            "guard_hand": {
                "centroid_error_mm": round(float(np.linalg.norm(guard_world.mean(axis=0) - hand_centroid)), 2),
                "direction_error_deg": round(_angle_deg(guard_world.mean(axis=0) - dh6[:3, 3], hand_centroid - dh6[:3, 3]), 2),
            },
        }
        if "curobo_tool0" in frames:
            cur = frames["curobo_tool0"]
            planner_world = _apply(cur, planner_hand)
            pose["planner_hand"] = {
                "centroid_error_mm": round(float(np.linalg.norm(planner_world.mean(axis=0) - hand_centroid)), 2),
                "direction_error_deg": round(_angle_deg(planner_world.mean(axis=0) - cur[:3, 3], hand_centroid - cur[:3, 3]), 2),
            }
            pose["lula_vs_curobo_tool0"] = {
                "position_mm": round(float(np.linalg.norm(lula[:3, 3] - cur[:3, 3])), 3),
                "rotation_deg": round(_rotation_deg(lula[:3, :3], cur[:3, :3]), 3),
            }
        tcp = lula @ tool_T
        pose["sim_tcp"] = {
            "approach_to_hand_deg": round(_angle_deg(tcp[:3, 2], hand_centroid - lula[:3, 3]), 2),
        }
        if len(inner) == 2:
            pads = np.vstack([placed[p].mean(axis=0) for p in inner]).mean(axis=0)
            pose["sim_tcp"]["distance_to_inner_finger_midpoint_mm"] = round(float(np.linalg.norm(tcp[:3, 3] - pads)), 2)
        report["poses"].append(pose)

        print(f"\njoints {pose['joints']} (readback error {pose['joint_readback_error_rad']} rad, "
              f"USD vs physics {pose['usd_vs_physics_body_origin_mm']} mm)")
        for name, row in per_frame.items():
            print(f"  {name:14s} origin {row['origin_from_dh6_mm']:8.3f} mm from dh6, rotated {row['rotation_from_dh6_deg']:7.3f} deg; "
                  f"hand centroid {row['hand_centroid_in_frame_mm']} angles {row['hand_angle_to_axes_deg']}")
        print(f"  guard hand   {pose['guard_hand']}")
        print(f"  planner hand {pose.get('planner_hand')}")
        print(f"  lula vs curobo tool0 {pose.get('lula_vs_curobo_tool0')}; sim tcp {pose['sim_tcp']}; "
              f"wrist_3 axis to hand {pose['wrist3_axis_to_hand_deg']} deg")

    _write(args.json, report)
    print(f"\nwrote {args.json}")
    try:
        client.close()
    except Exception:  # noqa: BLE001 (the process is exiting either way)
        pass
    return 0


def _write(path: str, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
