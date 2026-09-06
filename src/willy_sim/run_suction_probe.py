"""Attach probe for Isaac 5.1's surface gripper: does it form and hold an attach?

Spawns a flat box, places a static surface gripper a few cm above it pointing down, closes it, steps
physics, and polls the status until it reads "Closed" with the box in gripped_objects; then raises the
gripper and checks the box follows, which is the hold. Isaac's built-in `isaacsim.robot.surface_gripper`
is a binary proximity-joint surrogate. No arm, no mount, no gate. Default-off and on-box only (the
`isaacsim` imports are lazy, after bootstrap); nothing in the pick path imports it.

The API is `robot_schema.CreateSurfaceGripper(stage, path)` plus
`robot_schema.ApplyAttachmentPointAPI(joint)` (forwardAxis, clearanceOffset) plus the gripper's
ATTACHMENT_POINTS relationship to the D6 joint or joints; drive it with
`GripperView.apply_gripper_action([0.5/-0.5])` and read status with
`GripperStatus(get_surface_gripper_status()[0])`. There is no `create_surface_gripper` free function in
this build.

The attachment-point joint needs a dynamic body. A standalone kinematic cup with body1 left empty makes
PhysX refuse ("cannot create a joint between static bodies") and the bond never forms: the status stays
Open. So the surface gripper cannot be a standalone prim; the cup must hang on a dynamic articulation,
the UR5e arm, mounted on wrist_3_link before closing. This standalone probe is therefore the negative
control, and ``suction_mount`` holds the arm-mounted attach that does bond.

    <isaac-sim>\\python.bat -m src.willy_sim.run_suction_probe
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from src.config.schema.robot import SimObjectConfig
from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.scene import OBJECT_PRIM

PROBE_PRIM = "/World/SuctionProbe"
GRIPPER_PRIM = f"{PROBE_PRIM}/SurfaceGripper"


CUP_PRIM = f"{PROBE_PRIM}/cup"
JOINT_PRIM = f"{PROBE_PRIM}/attach_joint"


def _author_static_gripper(stage: Any, position_m: tuple[float, float, float], *,
                           max_grip_distance_m: float, force_limit: float, retry_interval_s: float) -> Any:
    """Author the full surface-gripper recipe and return its ``GripperView``.

    The recipe is a kinematic cup rigid body, the surface-gripper schema prim, and a D6 attachment-point
    joint (IsaacAttachmentPointAPI, forwardAxis Z pointing down at the box) whose body0 is the cup, linked
    through the gripper's ATTACHMENT_POINTS relationship. Without that attachment-point joint no bond
    forms. The cup is kinematic so that raising its xform later drags the bonded box.
    """
    from isaacsim.robot.surface_gripper import GripperView  # type: ignore[import-not-found]
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore[import-not-found]
    from usd.schema.isaac import robot_schema  # type: ignore[import-not-found]

    UsdGeom.Xform.Define(stage, PROBE_PRIM)
    # Kinematic cup rigid body just above the box (a small cuboid). Kinematic = movable + not gravity-driven.
    cup = UsdGeom.Cube.Define(stage, CUP_PRIM)
    cup.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(cup).SetScale(Gf.Vec3f(0.01, 0.01, 0.005))  # 20x20x10 mm
    UsdGeom.XformCommonAPI(cup).SetTranslate(Gf.Vec3d(*position_m))
    rb = UsdPhysics.RigidBodyAPI.Apply(cup.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(cup.GetPrim())
    UsdPhysics.MassAPI.Apply(cup.GetPrim()).CreateMassAttr(0.2)

    # The surface-gripper schema prim (status + force/grip-distance attrs + ATTACHMENT_POINTS/gripped rels).
    robot_schema.CreateSurfaceGripper(stage, GRIPPER_PRIM)

    # The D6 attachment-point joint: body0 = cup; body1 left open (the gripper bonds it to the target on close).
    # forwardAxis Z + localRot0 flipping local +Z to world -Z (down at the box, cup sits above it).
    joint = UsdPhysics.Joint.Define(stage, JOINT_PRIM)
    joint.CreateBody0Rel().SetTargets([CUP_PRIM])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))  # 180 deg about X, so local +Z points down
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateJointEnabledAttr(True)
    jp = joint.GetPrim()
    robot_schema.ApplyAttachmentPointAPI(jp)
    jp.GetAttribute(robot_schema.Attributes.FORWARD_AXIS.name).Set("Z")
    jp.GetAttribute(robot_schema.Attributes.CLEARANCE_OFFSET.name).Set(0.008)

    # Link the attachment point to the gripper.
    gp = stage.GetPrimAtPath(GRIPPER_PRIM)
    gp.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets([JOINT_PRIM])

    view = GripperView(paths=GRIPPER_PRIM)
    view.set_surface_gripper_properties(
        max_grip_distance=[max_grip_distance_m],
        coaxial_force_limit=[force_limit],
        shear_force_limit=[force_limit],
        retry_interval=[retry_interval_s],
    )
    return view


def run_probe(*, headless: bool = True, data_dir: str | None = None,
              max_grip_distance_m: float = 0.05, gap_mm: float = 15.0, lift_mm: float = 100.0) -> int:
    print("=== H7 surface-gripper ATTACH probe ===", flush=True)
    # A flat box (60x60x20 mm) so a top suction has a big sealable face; a single object lands on OBJECT_PRIM.
    box = SimObjectConfig(name="flat box", size_mm=(60.0, 60.0, 20.0), position_mm=(450.0, 0.0, 10.0), mass_kg=0.05)
    cell = bootstrap_sim_cell(data_dir, headless=headless, scene_kwargs={"objects_override": [box]})
    arm, session = cell.arm, cell.arm.session

    import omni.usd  # type: ignore[import-not-found]
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    # Park the arm out of the way so it cannot interfere with the box / the static gripper.
    from src.robot.core import JointPositions
    home_q = np.asarray(cell.sim.home_joint_positions, dtype=np.float64)
    arm.move_joint(JointPositions(home_q))
    session.step_n(30)

    obj = SingleRigidPrim(OBJECT_PRIM)
    box_pos0 = np.asarray(obj.get_world_pose()[0], dtype=np.float64)  # m
    box_top_m = float(box_pos0[2]) + 0.010  # +half the 20mm box
    print(f"box at {np.round(box_pos0 * 1000, 1)} mm, top ~{box_top_m * 1000:.1f} mm", flush=True)

    stage = omni.usd.get_context().get_stage()
    grip_z = box_top_m + gap_mm / 1000.0
    try:
        view = _author_static_gripper(
            stage, (float(box_pos0[0]), float(box_pos0[1]), grip_z),
            max_grip_distance_m=max_grip_distance_m, force_limit=1.0e6, retry_interval_s=2.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to author surface gripper: {type(exc).__name__}: {exc}", flush=True)
        return 2
    session.step_n(10)
    print(f"surface gripper authored at z={grip_z * 1000:.1f} mm, MaxGripDistance={max_grip_distance_m * 1000:.0f} mm", flush=True)

    def _status() -> tuple[str, list[str]]:
        try:
            st = list(view.get_surface_gripper_status())
            gr = list(view.get_gripped_objects())
            return (str(st[0]) if st else "?", gr if isinstance(gr, list) else list(gr))
        except Exception as exc:  # noqa: BLE001
            return (f"status-err:{type(exc).__name__}:{exc}", [])

    print(f"pre-close status={_status()}", flush=True)
    # Close: the gripper goes from Open through Closing to Closed only on a physics tick where a body sits
    # within MaxGripDistance along ForwardAxis. Step, then poll.
    view.apply_gripper_action(np.array([1.0]))
    closed = False
    for k in range(20):
        session.step_n(3)
        st, gr = _status()
        print(f"  close step {k}: status={st} gripped={gr}", flush=True)
        if st == "Closed" and gr:
            closed = True
            break
    if not closed:
        print("ATTACH DID NOT FORM (stuck Open/Closing or no gripped object): the H7.0R failure mode.", flush=True)
        return 1

    # Hold: raise the kinematic cup and check the box follows (the bonded attachment joint drags it).
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath(CUP_PRIM)).SetTranslate(  # type: ignore[attr-defined]
        Gf.Vec3d(float(box_pos0[0]), float(box_pos0[1]), grip_z + lift_mm / 1000.0)
    )
    session.step_n(40)
    box_pos1 = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
    rose_mm = (box_pos1[2] - box_pos0[2]) * 1000.0
    st, gr = _status()
    print(f"after raising gripper +{lift_mm:.0f} mm: box rose {rose_mm:.1f} mm, status={st} gripped={gr}", flush=True)
    held = rose_mm >= 0.5 * lift_mm
    print(f"\n=== ATTACH {'+ HOLD OK' if held else 'formed but HOLD weak'}: attach_formed=True held={held} "
          f"(box rose {rose_mm:.1f}/{lift_mm:.0f} mm) ===", flush=True)
    return 0 if held else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="H7 Isaac surface-gripper attach probe (measure>map).")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--max-grip-distance-mm", type=float, default=50.0)
    ap.add_argument("--gap-mm", type=float, default=15.0)
    ap.add_argument("--lift-mm", type=float, default=100.0)
    args = ap.parse_args()
    return run_probe(headless=not args.gui, data_dir=args.data,
                     max_grip_distance_m=args.max_grip_distance_mm / 1000.0,
                     gap_mm=args.gap_mm, lift_mm=args.lift_mm)


if __name__ == "__main__":
    raise SystemExit(main())
