"""Sim work-cell gripper specs: which end-effector to mount, as data.

Two families, both selected by a config key and driven by the vendor-neutral sim drivers:

* Parallel jaws: :data:`MOUNTED_GRIPPERS`, keyed by ``SimConfig.gripper_mount``. By default the sim
  uses the UR5e asset's baked ``Gripper`` variant (the Robotiq 2F-85); when a key is named the scene
  builder selects the ``"None"`` variant and mounts that standalone vendor gripper on the wrist:
  reference it, drop its articulation root, add a fixed joint ``wrist_3_link -> base`` so it merges
  into the UR5e articulation. :class:`~src.robot.grippers.sim.IsaacGripper` then drives it via the
  matching :class:`~src.robot.grippers.sim.GripperProfile`, so a new gripper is data here, not new
  driver code.
* Suction cups: :data:`SUCTION_CUPS`, keyed by ``SimConfig.suction_cup``. Each entry is a
  :class:`~src.robot.grippers.sim.SuctionCupProfile` (cup geometry plus vacuum threshold); a finer
  cup is a new entry, not new code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.robot.grippers.sim import (
    ROBOTIQ_2F85_PROFILE,
    SCHUNK_EGU50_PROFILE,
    SCHUNK_EZU35_PROFILE,
    SLIM_SUCTION_CUP,
    STANDARD_SUCTION_CUP,
    GripperProfile,
    SuctionCupProfile,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot.sim_schema import SimConfig

__all__ = [
    "MountedGripperSpec",
    "MOUNTED_GRIPPERS",
    "ROBOTIQ_2F85_MOUNT",
    "SCHUNK_EGU50_MOUNT",
    "SCHUNK_EZU35_MOUNT",
    "SUCTION_CUPS",
    "resolve_suction_cup",
]


@dataclass(frozen=True)
class MountedGripperSpec:
    """How to mount and drive a standalone vendor gripper USD on the UR5e wrist.

    ``mount_rotation_matrix`` maps the gripper's local axes into the ``wrist_3_link`` frame so the
    gripper's approach lands on the 2F-85 convention: approach = wrist_3 +Y.
    """

    name: str
    usd_asset_path: str  # relative to the Isaac assets_root
    base_prim_name: str  # the base rigid body (under the referenced prim) the fixed joint attaches to
    profile: GripperProfile  # the IsaacGripper profile mapping width to the driven joint
    # Rotation carrying the gripper's local frame onto the wrist_3_link frame (approach onto wrist +Y).
    mount_rotation_matrix: tuple[tuple[float, float, float], ...]
    flange_offset_mm: tuple[float, float, float]  # wrist_3_link to gripper base translation (wrist frame, mm)
    tcp_offset_mm: float  # gripper base to grasp centre along the approach (mm); feeds the grasp depth

    @property
    def flange_to_tcp_offset_mm(self) -> tuple[float, float, float]:
        """The composed flange to grasp-centre translation, in the wrist frame.

        ``flange_offset_mm`` (wrist_3 to the gripper base) plus ``tcp_offset_mm`` along the
        approach, which every mounted spec puts on wrist +Y; that is what ``mount_rotation_matrix``
        is for. Composing them here is what makes ``tcp_offset_mm`` a live value: without it,
        mounting a gripper would swap only its width profile while the arm kept using the 2F-85's
        132 mm, an offset error of 12.1 mm for the EGU-50 and 18.0 mm for the EZU-35. The 2F-85
        composes to exactly 132.0, so the default cell is unchanged.
        """
        fx, fy, fz = self.flange_offset_mm
        return (float(fx), float(fy) + float(self.tcp_offset_mm), float(fz))
    # Driven-joint PD drive gains for the mounted gripper (overrides the vendor USD defaults). ``None``
    # leaves the asset's gains untouched. A firmer grip (higher stiffness, damping scaled to keep the
    # ratio) holds the object against the lift's inertial force: the EGU-50's soft vendor default
    # (stiffness 1000) grips statically but lets the cube slip out during the 100 mm lift.
    drive_stiffness: float | None = None
    drive_damping: float | None = None
    # Per-rigid-body mass (kg) for the mounted gripper bodies (housing + fingers); ``None`` keeps the
    # vendor USD masses. A heavy end-effector destabilises the reactive RMPflow move: it cannot drive
    # the last ~20 mm to the grasp pose and the IK-refine times out. A lighter, uniform mass keeps the
    # move convergent. Set at mount time, before the articulation initialises. Sim-only end-effector
    # mass; it does not model the real gripper's inertia.
    body_mass_kg: float | None = None


# Schunk EGU-50 (parallel, a different vendor from the baked Robotiq 2F-85). Local frame: approach =
# local +Z (grasp centre 115 mm out), closing = local +Y. The 2F-85's mounted grasp frame, which the
# pick geometry is tuned to: approach = wrist_3 +Y, closing = wrist_3 +X. ``mount_rotation_matrix``
# carries the EGU-50 onto that same orientation. ``flange_offset_mm`` is 29.1 mm along wrist +Y, the
# approach: the EGU-50's FGR-mesh grasp centre sits ~24 mm shallower than the 2F-85 inner-finger-body
# midpoint the pick depth is tuned to, so at zero offset the gripper grips ~24 mm too high and knocks
# the object over; +29.1 mm extends it so the grasp centre reaches the object centre. The matrix is the
# joint localRot0 input under scene.py's ``from_rotation_matrix`` to ``Gf.Quatf`` convention; the
# values are empirical.
SCHUNK_EGU50_MOUNT = MountedGripperSpec(
    name="schunk_egu50",
    usd_asset_path="/Isaac/Robots/Schunk/egu_50/schunk_egu_50.usd",
    base_prim_name="SCHUNK_1491540_EGU_50_EI_M_B_000ohne0",
    profile=SCHUNK_EGU50_PROFILE,
    mount_rotation_matrix=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    flange_offset_mm=(0.0, 29.1, 0.0),
    tcp_offset_mm=115.0,
    # Firm the soft vendor default (stiffness 1000, damping 10) by a factor of 12 so the grip holds the
    # cube through the lift under natural-aim, the safety branch. That branch is mandatory here: in the
    # flipped park-pi branch the EGU-50 self-collides (forearm against gripper, 0.000 mm) and Coal
    # rejects the pose.
    drive_stiffness=12000.0,
    drive_damping=120.0,
)

# Schunk EZU-35 (3-finger centric, the multifinger generalization). Same mount harness as the
# EGU-50. Standalone, world = local: the 3 fingers sit at 120 deg and the approach axis is local +Z
# (housing to finger centroid, 125 mm), identical to the EGU-50, so the EGU-50's mount_rotation_matrix
# carries over. The finger pad-mid is ~140 mm out along the approach, against the EGU-50's ~115 mm plus
# a 29.1 mm flange, ~144 mm from the wrist, so a small flange keeps the same grasp depth.
# drive_stiffness and drive_damping reuse the EGU-50's firm grip; the soft vendor default drops the
# object on the lift. The width table is the 3-finger inscribed diameter.
#
# Status: this mount harness covers the 3-finger centric vendor swap (mount, 1-DOF drive, depth and
# reach). The 3-finger equal grip needs one piece that is not in this static spec: the two PhysxMimic
# follower fingers (GBA2/GBA3) must be firmed and continuously target-synced to Jaw_Drive each physics
# step, otherwise a firm follower drive fights the mimic and jams the gripper. That sync is not in
# IsaacGripper. The free-cylinder pick is blocked by the descent-roll (see ``_cylinder_specs``).
# Default-off: only build_service(gripper_mount="schunk_ezu35") reaches this.
SCHUNK_EZU35_MOUNT = MountedGripperSpec(
    name="schunk_ezu35",
    usd_asset_path="/Isaac/Robots/Schunk/ezu_35/schunk_ezu_35.usd",
    base_prim_name="SCHUNK_1582096_EZU_35_MB_N_B__000ohne0",
    profile=SCHUNK_EZU35_PROFILE,
    mount_rotation_matrix=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),  # EGU-50's (approach = local +Z)
    flange_offset_mm=(0.0, -26.0, 0.0),  # Retract: the move places tool0 for the 2F-85 TCP, and the EZU-35's
    # long fingers (tips ~170mm from the base) otherwise reach ~30mm past that and ram the table before
    # the grasp; the descent blocks and the move times out ~20mm short. -26mm pulls the gripper back so
    # its effective grip depth matches the EGU-50's (~144mm), clearing the table. The housing overlaps
    # the wrist region visually; the fixed joint filters that collision, and Coal uses the arm-capsule
    # fallback here, so there is no false reject.
    tcp_offset_mm=140.0,
    drive_stiffness=12000.0,
    drive_damping=120.0,
    body_mass_kg=0.1,  # light uniform mass keeps the reactive move stable for the chunky 3-finger gripper
)

# Robotiq 2F-85 as a standalone mount. Isaac's ``ur3e.usd`` bakes no gripper variant (at byte level,
# ur5e.usd carries the token ``Robotiq_2F_85`` while ur3e.usd carries neither ``Gripper`` nor
# ``Robotiq``), so a UR3e cell must mount the very same gripper the UR5e cell gets baked in.
#
# Every value here is read off the baked mount, not tuned. The ur5e asset's authored coupling joint
# ``/World/UR5e/joints/robot_gripper_joint`` gives body0=wrist_3_link, body1=Robotiq_2F_85/base_link,
# localPos0 == localPos1 == (0,0,0) and localRot0 == (w=-0.5, x=0.5, y=0.5, z=0.5), whose matrix is
# exactly ``((0,1,0),(0,0,1),(1,0,0))``, with localRot1 = identity. The mount is therefore a pure
# re-orientation at zero offset, the ISO-9409-1-50 flange being identical across the UR e-series.
# Reproducing those inputs lands the mounted gripper on the same grasp frame the pick geometry is
# tuned to (approach = wrist_3 +Y, closing = +X), which is why, unlike the EGU-50, it needs no
# flange_offset.
#
# The standalone asset's body and joint tree is identical to the baked variant (``base_link``,
# ``Joints/finger_joint`` and the knuckle/finger bodies) and it carries no ``root_joint``, which
# ``_mount_standalone_gripper`` tolerates. ``base_prim_name`` is a nested path because the reference
# composes the asset's ``Robotiq_2F_85`` prim under the mount path. Drive gains stay at the vendor
# defaults: this is the same gripper the baked cell drives, so firming it as the EGU-50 needs would be
# an unmeasured change.
ROBOTIQ_2F85_MOUNT = MountedGripperSpec(
    name="robotiq_2f85",
    usd_asset_path="/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd",
    base_prim_name="Robotiq_2F_85/base_link",
    profile=ROBOTIQ_2F85_PROFILE,
    mount_rotation_matrix=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    flange_offset_mm=(0.0, 0.0, 0.0),
    tcp_offset_mm=132.0,  # the baked 2F-85 grasp-centre offset the pick depth is tuned to (robot.sim.yaml)
)

MOUNTED_GRIPPERS: dict[str, MountedGripperSpec] = {
    SCHUNK_EGU50_MOUNT.name: SCHUNK_EGU50_MOUNT,
    SCHUNK_EZU35_MOUNT.name: SCHUNK_EZU35_MOUNT,
    ROBOTIQ_2F85_MOUNT.name: ROBOTIQ_2F85_MOUNT,
}


# Suction cups (keyed by ``SimConfig.suction_cup``). The cup mounts directly on the dynamic
# ``wrist_3_link`` with no separate body, so a cup is its geometry plus vacuum threshold: a
# ``SuctionCupProfile``. ``None`` (the default) uses the standard cup.
SUCTION_CUPS: dict[str, SuctionCupProfile] = {
    STANDARD_SUCTION_CUP.name: STANDARD_SUCTION_CUP,
    SLIM_SUCTION_CUP.name: SLIM_SUCTION_CUP,
}


def resolve_suction_cup(
    sim_cfg: "SimConfig | None" = None, *, override: str | None = None
) -> SuctionCupProfile:
    """Resolve the suction-cup profile for the sim cell.

    ``override`` (a CLI flag) wins, then ``sim.suction_cup`` from config, then the standard cup.
    Raises ``KeyError`` on an unknown key so a typo fails loudly instead of silently mounting the
    wrong cup. This is the one place a runner turns "which cup" into the
    :class:`SuctionCupProfile` that drives the sim driver width, the visible cup geometry and the
    collision envelope: one decision, not three.
    """
    key = override or (getattr(sim_cfg, "suction_cup", None) if sim_cfg is not None else None)
    if not key:
        return STANDARD_SUCTION_CUP
    return SUCTION_CUPS[key]
