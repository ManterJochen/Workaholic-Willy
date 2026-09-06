"""Author an Isaac surface-gripper suction anchor on the UR5e wrist (arm-mounted attach).

The attachment-point joint needs a dynamic body. A standalone kinematic cup with body1 left empty makes
PhysX refuse ("cannot create a joint between static bodies") and no bond ever forms, so the attach must
hang on a dynamic articulation; ``run_suction_probe`` is the standalone negative control. Two further
constraints shape it:

* a separate primitive cup body fixed-jointed to the wrist is fragile: a compliant maximal-coordinate
  joint, or a collision body overlapping the 2F-85 fingers, destabilises the long lever, wrist_3 spins
  and the home config is corrupted;
* the "None" UR5e gripper variant, an arm with no merged end-effector, breaks the Lula IK pre-resolve for
  every top-down reach, while the baked 2F-85 cell reaches them clean.

So this anchors the surface gripper directly on ``wrist_3_link``, a rigid articulation link, in the baked
2F-85 cell: no separate body and no collision. The attachment-point joint's anchor sits
``anchor_offset_mm`` down the gripper approach (wrist +Y), so on close the surface gripper bonds a body
within ``max_grip_distance_mm`` of the grasp centre to the dynamic wrist. Authoring only; every
``isaacsim`` and ``pxr`` import is lazy, so the module imports on macOS and in CI. No module under
``src/robot`` imports this one, and ``run_suction_pick``, which does import it at module scope, stays
import-safe because of that laziness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grippers.sim import SuctionCupProfile

__all__ = [
    "SuctionCupSpec", "SUCTION_CUP", "mount_suction_cup",
    "engage_suction_weld", "release_suction_weld",
    "mount_visible_suction_cup", "mount_real_suction_gripper", "hide_baked_jaw",
    "REAL_GRIPPER_USD", "REAL_GRIPPER_TIP_MM",
]

# The shipped Isaac 5.1 UR10 vacuum gripper mesh: a real industrial suction end-effector, a shaft and tube
# down to a vacuum cup, plus a wrist camera and light. Referenced visual-only on the UR5e wrist so the
# suction cell shows a real tool instead of an authored placeholder cylinder. The asset is modelled along
# its local +X with the mount at x~=0 and the cup tip at x~=161 mm; size ~165x106x100 mm. Each mesh carries
# a Collision API and there is one internal Suction_Joint; both are neutralised on mount, so the tool is
# display-only, the weld or carry does the attach, and physics and IK are untouched.
REAL_GRIPPER_USD = "/Isaac/Robots/UniversalRobots/ur10/grippers/short_gripper.usd"
REAL_GRIPPER_TIP_MM = 161.0  # cup-face offset along the approach (wrist +Y) after the mount rotation


@dataclass(frozen=True)
class SuctionCupSpec:
    """How to author an Isaac surface-gripper suction anchor on the UR5e wrist.

    The anchor sits ``anchor_offset_mm`` from ``wrist_3_link`` along the gripper approach (wrist +Y, the
    2F-85 convention), which is about the grasp centre, with the attachment forwardAxis pointing along
    that approach and so downwards at a top-down pose. On close the surface gripper bonds a body within
    ``max_grip_distance_mm`` of the anchor to the rigid, dynamic wrist link; on lift the bonded body rides
    with the arm.
    """

    name: str = "suction_cup"
    # Offset from wrist_3_link to the attachment anchor along the approach (wrist +Y), in mm. That is about
    # the 2F-85 grasp centre, 132 mm.
    anchor_offset_mm: float = 132.0
    # Joint local frame into the wrist_3_link frame: maps the joint's forwardAxis (local Z) onto wrist +Y,
    # the approach, so the suction points the way the jaws would. A -90 deg rotation about wrist X.
    forward_axis_rot_wxyz: tuple[float, float, float, float] = (0.70710678, -0.70710678, 0.0, 0.0)
    # Surface-gripper bond parameters.
    max_grip_distance_mm: float = 120.0  # how far a body may sit along forwardAxis and still bond (generous:
    #                                      the anchor sits ~the grasp centre, the object top is ~40-70 mm below)
    clearance_offset_mm: float = 8.0     # gantry-example default (the attachment stand-off)
    coaxial_force_limit: float = 1.0e6
    shear_force_limit: float = 1.0e6
    retry_interval_s: float = 0.1  # how often the gripper retries the bond; a small value engages promptly


SUCTION_CUP = SuctionCupSpec()


def mount_suction_cup(stage: Any, arm_prim_path: str, spec: SuctionCupSpec) -> str:
    """Author the surface-gripper attach anchor on ``wrist_3_link``; return the gripper prim path.

    Anchors directly on the dynamic ``wrist_3_link`` articulation link, with no separate cup body, so
    there is nothing to destabilise the arm. Author this after ``arm.connect()``: the attachment-point
    joint is ``excludeFromArticulation``, a maximal-coordinate joint, so it must not be present when
    ``SingleArticulation.initialize()`` runs, which would break the Lula IK pre-resolve. Returns the
    surface gripper prim path for :class:`IsaacSuctionGripper`. ``pxr`` and ``isaacsim`` imports are lazy.
    """
    from isaacsim.robot.surface_gripper import GripperView  # type: ignore[import-not-found]
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore[import-not-found]
    from usd.schema.isaac import robot_schema  # type: ignore[import-not-found]

    suction_root = f"{arm_prim_path}/suction"
    gripper_path = f"{suction_root}/SurfaceGripper"
    wrist = f"{arm_prim_path}/wrist_3_link"
    joint_path = f"{wrist}/suction_attach_joint"

    UsdGeom.Xform.Define(stage, suction_root)

    # The surface-gripper schema prim (status + force/grip-distance attrs + ATTACHMENT_POINTS/gripped rels).
    robot_schema.CreateSurfaceGripper(stage, gripper_path)

    # The D6 attachment-point joint anchored on wrist_3_link: body0 = the wrist link (dynamic), body1 open
    # (the gripper bonds it to the target on close). localPos0 = anchor along wrist +Y (~the grasp centre);
    # localRot0 maps the joint forwardAxis (local Z) onto wrist +Y so the suction points along the approach.
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([wrist])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, float(spec.anchor_offset_mm) / 1000.0, 0.0))
    w, x, y, z = spec.forward_axis_rot_wxyz
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateJointEnabledAttr(True)
    jp = joint.GetPrim()
    robot_schema.ApplyAttachmentPointAPI(jp)
    jp.GetAttribute(robot_schema.Attributes.FORWARD_AXIS.name).Set("Z")
    jp.GetAttribute(robot_schema.Attributes.CLEARANCE_OFFSET.name).Set(spec.clearance_offset_mm / 1000.0)

    gp = stage.GetPrimAtPath(gripper_path)
    gp.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets([joint_path])

    # Set the bond properties (critical: an unset MaxGripDistance never bonds). Best-effort here (a GripperView
    # may not resolve before the first play when this is authored via the bootstrap pre-play hook); the driver
    # re-asserts them post-play in IsaacSuctionGripper.connect().
    try:
        GripperView(paths=gripper_path).set_surface_gripper_properties(
            max_grip_distance=[spec.max_grip_distance_mm / 1000.0],
            coaxial_force_limit=[spec.coaxial_force_limit],
            shear_force_limit=[spec.shear_force_limit],
            retry_interval=[spec.retry_interval_s],
        )
    except Exception:  # noqa: BLE001 (pre-play GripperView may not resolve; the driver sets props post-play)
        pass
    return gripper_path


def engage_suction_weld(
    stage: Any,
    wrist_prim_path: str,
    object_prim_path: str,
    local_pos0_m: tuple[float, float, float],
    local_rot0_wxyz: tuple[float, float, float, float],
    *,
    joint_name: str = "suction_grip_weld",
) -> str:
    """Author a runtime ``FixedJoint`` welding ``object_prim_path`` to the wrist at its current relative pose.

    This is the physics-real stand-in for the vacuum bond: Isaac's binary surface gripper will not bond a
    non-root articulation link, so once the analytical seal says the contact is sealable the object is
    locked rigidly to ``wrist_3_link`` with a maximal-coordinate fixed joint, the same kind of constraint
    between an articulation and a free body that a real grasp is. ``local_pos0_m`` and ``local_rot0_wxyz``
    place the joint anchor on the wrist at the object's current pose in the wrist frame, so the object is
    not snapped: it stays exactly where the cup met it. body1's local frame is identity. The object then
    rides with the arm on a lift; :func:`release_suction_weld` drops it (vacuum off). The ``pxr`` import is
    lazy. Returns the joint path.
    """
    from pxr import Gf, UsdPhysics  # type: ignore[import-not-found]

    joint_path = f"{wrist_prim_path}/{joint_name}"
    fj = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fj.CreateBody0Rel().SetTargets([wrist_prim_path])
    fj.CreateBody1Rel().SetTargets([object_prim_path])
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(c) for c in local_pos0_m)))
    w, x, y, z = local_rot0_wxyz
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # A free rigid body is not part of the UR5e articulation, so this must be a maximal-coordinate joint,
    # like the surface gripper's own attachment joint; else PhysX rejects adding it to the topology at runtime.
    fj.CreateExcludeFromArticulationAttr(True)
    fj.CreateJointEnabledAttr(True)
    return joint_path


def release_suction_weld(stage: Any, joint_path: str) -> bool:
    """Release a weld authored by :func:`engage_suction_weld` (vacuum off). Returns True if a joint was removed."""
    prim = stage.GetPrimAtPath(joint_path)
    if prim and prim.IsValid():
        stage.RemovePrim(joint_path)
        return True
    return False


def hide_baked_jaw(stage: Any, arm_prim_path: str) -> int:
    """Make the baked Robotiq 2F-85 moving-jaw meshes invisible so a suction cell shows a cup on the stub.

    Hides only the moving finger assembly (knuckle, finger, tip, pad) and keeps the base coupler
    ``robotiq_85_base_link``, the mount stub the cup sits on, so the cup is not floating. The 2F-85 stays
    in the articulation, where it keeps the Lula IK pre-resolve healthy; the ``None`` gripper variant
    breaks IK. Only the render is hidden. Visibility is a pure display attr: it does not touch physics,
    collision or DOFs, so IK and the lift are unaffected. Returns the number of prims hidden. Lazy ``pxr``.
    """
    from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

    # The moving-jaw parts only; not "base", since the coupler stub stays visible so the cup mounts on it.
    keys = ("knuckle", "finger", "fingertip", "tip", "pad")
    hidden = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(arm_prim_path)):
        name = prim.GetName().lower()
        if "base" in name:
            continue
        if any(k in name for k in keys):
            img = UsdGeom.Imageable(prim)
            if img:
                img.MakeInvisible()
                hidden += 1
    return hidden


def mount_visible_suction_cup(
    stage: Any,
    arm_prim_path: str,
    spec: SuctionCupSpec,
    *,
    profile: "SuctionCupProfile | None" = None,
    hide_jaw: bool = True,
) -> str:
    """Author a visible suction-cup end-effector on the wrist, a stem and a cup, so the cell looks like one.

    Purely visual USD geometry, no collider and no body, parented under ``wrist_3_link`` so it rides with
    the arm: a thin metal stem, then a dark rubber cup, laid along the gripper approach (wrist +Y, the
    2F-85 convention) so the cup face sits at about the grasp centre and contact point. The cup and shaft
    radii come from ``profile`` (a :class:`~src.robot.grippers.sim.SuctionCupProfile`) when one is given,
    so the ``slim`` cup renders slim, and fall back to this function's own placeholder radii otherwise.
    With ``hide_jaw`` the baked 2F-85 jaw meshes are hidden (:func:`hide_baked_jaw`) so only the cup shows.
    This does not change physics or IK: the 2F-85 stays in the articulation for IK health and the weld or
    carry does the attach. Returns the visual root path.
    """
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    wrist = f"{arm_prim_path}/wrist_3_link"
    root = f"{wrist}/suction_visual"
    UsdGeom.Xform.Define(stage, root)

    def _cyl(name: str, radius_m: float, y0_m: float, y1_m: float, color: tuple[float, float, float]) -> None:
        cyl = UsdGeom.Cylinder.Define(stage, f"{root}/{name}")
        cyl.CreateAxisAttr("Y")  # lay the cylinder along wrist +Y (the approach), not the default Z
        cyl.CreateRadiusAttr(float(radius_m))
        cyl.CreateHeightAttr(float(y1_m - y0_m))
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        # The cylinder is centred at the origin along its axis, so translate it to span [y0, y1].
        UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(0.0, float((y0_m + y1_m) / 2.0), 0.0))

    # A shaft from the wrist flange (Y=0, so it is rigidly connected to the arm and not floating) down to
    # the grasp centre, then a wide dark rubber cup at the contact tip. The anchor, which is the grasp
    # centre, is ``anchor_offset_mm`` (~132) along +Y; the cup face sits about there so it meets the object
    # top at contact. The baked 2F-85 base coupler stays visible (hide_baked_jaw keeps it) as the mount stub.
    a = float(spec.anchor_offset_mm) / 1000.0
    shaft_r_m = 0.013 if profile is None else float(profile.shaft_radius_mm) / 1000.0
    cup_r_m = 0.024 if profile is None else float(profile.cup_radius_mm) / 1000.0
    _cyl("shaft", shaft_r_m, 0.0, a - 0.002, (0.30, 0.30, 0.33))
    _cyl("cup", cup_r_m, a - 0.006, a + 0.026, (0.10, 0.10, 0.12))

    if hide_jaw:
        hide_baked_jaw(stage, arm_prim_path)
    return root


def mount_real_suction_gripper(
    stage: Any,
    arm_prim_path: str,
    *,
    hide_jaw: bool = True,
    yaw_deg: float = 90.0,
    flange_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Reference the real Isaac UR10 vacuum gripper mesh on the wrist as a visual-only end-effector.

    Mounts :data:`REAL_GRIPPER_USD`, a real industrial suction tool with a shaft and tube down to a vacuum
    cup plus a wrist camera and light, under ``wrist_3_link``, so the suction cell shows a real gripper
    instead of the authored placeholder cylinder of :func:`mount_visible_suction_cup`. The asset is
    modelled along its local +X with the cup tip at ~161 mm; a ``yaw_deg`` rotation about the wrist Z aims
    that +X along the wrist +Y approach (the 2F-85 convention), so the cup face sits at about the grasp
    centre and contact point at a top-down pose. Every collider on the referenced subtree is disabled and
    the internal ``Suction_Joint`` is deactivated, so the tool is pure display geometry: it rides the wrist
    but never perturbs PhysX or the UR5e articulation and IK, and the render-safe carry or weld does the
    actual attach. With ``hide_jaw`` the baked 2F-85 jaw meshes are hidden (:func:`hide_baked_jaw`). ``pxr``
    and ``isaacsim`` imports are lazy. Returns the visual root path.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]
    from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

    root = (get_assets_root_path() or "").rstrip("/")
    wrist = f"{arm_prim_path}/wrist_3_link"
    vis_root = f"{wrist}/suction_gripper"
    # short_gripper.usd has no default prim, so add_reference_to_stage maps its root prims under vis_root,
    # an Xform. Aim its +X along wrist +Y and offset the flange if needed.
    add_reference_to_stage(root + REAL_GRIPPER_USD, vis_root)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(vis_root))
    xf.ClearXformOpOrder()
    if any(flange_offset_mm):
        xf.AddTranslateOp().Set(Gf.Vec3d(*(float(c) / 1000.0 for c in flange_offset_mm)))
    if yaw_deg:
        xf.AddRotateZOp().Set(float(yaw_deg))

    # Neutralise physics so the tool is display-only: disable every collider + deactivate the suction joint.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(vis_root)):
        if "Joint" in str(prim.GetTypeName()):
            prim.SetActive(False)
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)

    if hide_jaw:
        hide_baked_jaw(stage, arm_prim_path)
    return vis_root
