"""Tests for the SelfCollisionGuard and capsule helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.config.schema.robot import (
    FixtureBoxConfig,
    RobotSafetyConfig,
    SelfCollisionSafetyConfig,
    WorkspaceLimitsConfig,
)
from src.geometry import Frame, Pose
from src.geometry.quaternion import IDENTITY_QUAT_XYZW
from src.robot.core import JointPositions, MotionCommand, RobotCapabilities
from src.robot.safety import (
    SafetyContext,
    SafetyPreflight,
    SafetyReason,
    SelfCollisionGuard,
)
from src.robot.safety._capsule import (
    AxisAlignedBox,
    Capsule,
    capsule_box_distance_mm,
    capsule_capsule_distance_mm,
    segment_segment_distance_mm,
)
from src.robot.safety._ur_kinematics import ur_link_origins_mm


def _ur_arm():
    return type(
        "Arm",
        (),
        {
            "capabilities": RobotCapabilities(
                vendor="ur", model="ur5e", dof=6,
                has_native_fk=True,
                is_simulated=False,
                                         controller=None, owns_a_controller=True,
            ),
        },
    )()


def _kuka_arm():
    return type(
        "Arm",
        (),
        {
            "capabilities": RobotCapabilities(
                vendor="kuka", model="kr6-r900", dof=6,
                has_native_fk=True,
                is_simulated=False,
                                         controller=None, owns_a_controller=True,
            ),
        },
    )()


def _reporting(vendor: str, model: str, *, simulated: bool = True):
    """An arm that reports ``vendor`` and ``model`` and nothing else."""
    return type(
        "Arm",
        (),
        {
            "capabilities": RobotCapabilities(
                vendor=vendor, model=model, dof=6,
                has_native_fk=True,
                is_simulated=simulated,
                                         controller=None, owns_a_controller=vendor not in ("sim", "dummy"),
            ),
        },
    )()


def _sim_arm():
    # What IsaacRobotArm reports since P0d D5: vendor 'sim' and the UR it drives.
    return _reporting("sim", "ur5e")


def _pose(x_mm: float = 400.0, y_mm: float = 0.0, z_mm: float = 300.0) -> Pose:
    return Pose(
        position_mm=np.array([x_mm, y_mm, z_mm], dtype=np.float64),
        quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
        frame=Frame.BASE,
        label="tcp",
    )


class CapsuleGeometryTests(unittest.TestCase):
    def test_segment_segment_parallel_distance(self) -> None:
        d = segment_segment_distance_mm(
            np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]),
            np.array([0.0, 50.0, 0.0]), np.array([100.0, 50.0, 0.0]),
        )
        self.assertAlmostEqual(d, 50.0, places=6)

    def test_segment_segment_crossing_distance_is_zero(self) -> None:
        d = segment_segment_distance_mm(
            np.array([-100.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]),
            np.array([0.0, -100.0, 0.0]), np.array([0.0, 100.0, 0.0]),
        )
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_capsule_capsule_signed_distance(self) -> None:
        a = Capsule(np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), 20.0)
        b = Capsule(np.array([0.0, 50.0, 0.0]), np.array([100.0, 50.0, 0.0]), 20.0)
        signed = capsule_capsule_distance_mm(a, b)
        # 50 mm centre-to-centre - (20 + 20) = 10 mm.
        self.assertAlmostEqual(signed, 10.0, places=6)

    def test_capsule_capsule_penetration_is_negative(self) -> None:
        a = Capsule(np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), 30.0)
        b = Capsule(np.array([0.0, 40.0, 0.0]), np.array([100.0, 40.0, 0.0]), 30.0)
        signed = capsule_capsule_distance_mm(a, b)
        # 40 mm centres - (30 + 30) = -20 mm (penetrating).
        self.assertAlmostEqual(signed, -20.0, places=6)

    def test_capsule_box_distance(self) -> None:
        # Capsule horizontal at y=50, radius 10. Box at origin, ±20.
        c = Capsule(np.array([-50.0, 50.0, 0.0]), np.array([50.0, 50.0, 0.0]), 10.0)
        box = AxisAlignedBox(
            center_mm=np.array([0.0, 0.0, 0.0]),
            half_extents_mm=np.array([20.0, 20.0, 20.0]),
        )
        signed = capsule_box_distance_mm(c, box)
        # Nearest point on capsule line at y=50 → distance to box face = 30 → minus r=10 ⇒ 20.
        self.assertAlmostEqual(signed, 20.0, places=4)


class URKinematicsTests(unittest.TestCase):
    def test_ur5e_zero_joints_returns_seven_origins(self) -> None:
        origins = ur_link_origins_mm("ur5e", np.zeros(6))
        self.assertIsNotNone(origins)
        self.assertEqual(len(origins), 7)
        # Base origin is exactly 0,0,0.
        np.testing.assert_allclose(origins[0], [0.0, 0.0, 0.0])
        # Shoulder lifts to the base height of UR5e (d_1 = 162.5 mm).
        self.assertAlmostEqual(origins[1][2], 162.5, places=3)

    def test_unknown_model_returns_none(self) -> None:
        self.assertIsNone(
            ur_link_origins_mm("franka", np.zeros(7))
        )

    def test_dof_mismatch_returns_none(self) -> None:
        self.assertIsNone(ur_link_origins_mm("ur5e", np.zeros(5)))


class SelfCollisionGuardTests(unittest.TestCase):
    def _ctx(self, pose=None, joints=None, arm=None) -> SafetyContext:
        return SafetyContext(
            command=MotionCommand.MOVE_TO,
            target_pose=pose,
            target_joints=joints,
            arm=arm,
        )

    def test_ur_zero_pose_accepts(self) -> None:
        cfg = SelfCollisionSafetyConfig()
        guard = SelfCollisionGuard(cfg)
        d = guard.evaluate(self._ctx(
            pose=_pose(),
            joints=JointPositions([0.0, -1.5, 1.5, 0.0, 0.0, 0.0]),
            arm=_ur_arm(),
        ))
        self.assertTrue(d.accepted, msg=d.message)

    #: Flange origin for ``_FIXTURE_JOINTS`` on a ur5e, from ``ur_link_transforms_mm`` (mm).
    _FIXTURE_JOINTS = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)
    _FLANGE_MM = (-422.3, -232.9, 486.7)

    def test_fixture_intersect_rejects(self) -> None:
        """A fixture around the tool is refused by BOTH backends.

        The pose and the joints must agree. The capsule path roots the tool in ``target_pose``; the
        exact-mesh path roots every link in ``target_joints``. A context whose two halves disagree is
        not something a driver can produce (the UR driver sets ``target_joints = self.ik(pose)``), and
        a test built on one only proves which half its backend happens to read.
        """
        fx = FixtureBoxConfig(
            name="post",
            center_mm=self._FLANGE_MM,
            half_extents_mm=(50.0, 50.0, 50.0),
        )
        for backend in ("capsule", "fcl"):
            with self.subTest(backend=backend):
                cfg = SelfCollisionSafetyConfig(fixtures=[fx], backend=backend)
                guard = SelfCollisionGuard(cfg)
                d = guard.evaluate(self._ctx(
                    pose=_pose(*self._FLANGE_MM),
                    joints=JointPositions(list(self._FIXTURE_JOINTS)),
                    arm=_ur_arm(),
                ))
                self.assertFalse(d.accepted, msg=d.message)
                self.assertIs(d.reason, SafetyReason.SELF_COLLISION)

    #: Where a perceived box has to sit for each backend to be able to see it at all.
    #:
    #: The two backends check different things, which the module docstring states and this pins: the
    #: capsule proxy roots the tool in ``target_pose`` and compares it against boxes, while the
    #: exact-mesh path roots every link in ``target_joints`` and compares those. A test that put one
    #: box in one place would prove only whichever half its backend happens to read.
    _PERCEIVED_CASES = (
        ("capsule", (600.0, 200.0, 300.0), (600.0, 200.0, 300.0)),
        ("fcl", (-422.3, -232.9, 486.7), (-422.3, -232.9, 486.7)),
    )

    def test_a_perceived_obstacle_refuses_the_same_way_a_declared_one_does(self) -> None:
        """The planner and this guard have to be looking at the same cell.

        A planner routing around a tote the guard cannot see gives the worst of both: a path that
        avoids the tote, and a gate that would have passed one straight through it. So an obstacle a
        camera saw is refused exactly as a fixture written in the config is, and it says which one.
        """
        from src.robot.safety._capsule import AxisAlignedBox

        for backend, tool_mm, box_mm in self._PERCEIVED_CASES:
            with self.subTest(backend=backend):
                guard = SelfCollisionGuard(SelfCollisionSafetyConfig(backend=backend))
                ctx = self._ctx(
                    pose=_pose(*tool_mm),
                    joints=JointPositions(list(self._FIXTURE_JOINTS)),
                    arm=_ur_arm(),
                )
                self.assertTrue(guard.evaluate(ctx).accepted, "nothing is there yet")

                guard.set_perceived_fixtures(
                    [
                        AxisAlignedBox(
                            center_mm=np.asarray(box_mm, dtype=np.float64),
                            half_extents_mm=np.asarray([60.0, 60.0, 60.0], dtype=np.float64),
                            name="seen_00_tote",
                        )
                    ]
                )
                refused = guard.evaluate(ctx)
                self.assertFalse(refused.accepted, msg=refused.message)
                self.assertIs(refused.reason, SafetyReason.SELF_COLLISION)
                self.assertIn(
                    "seen_00_tote", refused.message,
                    "a refusal that will not say which box refused sends an operator to look at all "
                    "of them",
                )

    def test_clearing_the_perceived_obstacles_puts_the_cell_back(self) -> None:
        """The moment nobody can vouch for what a camera saw, the guard must stop believing it."""
        from src.robot.safety._capsule import AxisAlignedBox

        spot = (600.0, 200.0, 300.0)
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig(backend="capsule"))
        ctx = self._ctx(
            pose=_pose(*spot),
            joints=JointPositions(list(self._FIXTURE_JOINTS)),
            arm=_ur_arm(),
        )
        guard.set_perceived_fixtures(
            [
                AxisAlignedBox(
                    center_mm=np.asarray(spot, dtype=np.float64),
                    half_extents_mm=np.asarray([60.0, 60.0, 60.0], dtype=np.float64),
                    name="seen_00",
                )
            ]
        )
        self.assertFalse(guard.evaluate(ctx).accepted)

        guard.set_perceived_fixtures([])
        self.assertTrue(guard.evaluate(ctx).accepted)

    def test_a_declared_fixture_survives_every_perceived_one(self) -> None:
        """Clearing what a camera saw must never take the bench with it."""
        from src.robot.safety._capsule import AxisAlignedBox

        spot = (600.0, 200.0, 300.0)
        declared = FixtureBoxConfig(
            name="post", center_mm=spot, half_extents_mm=(60.0, 60.0, 60.0)
        )
        guard = SelfCollisionGuard(
            SelfCollisionSafetyConfig(fixtures=[declared], backend="capsule")
        )
        ctx = self._ctx(
            pose=_pose(*spot),
            joints=JointPositions(list(self._FIXTURE_JOINTS)),
            arm=_ur_arm(),
        )

        guard.set_perceived_fixtures(
            [
                AxisAlignedBox(
                    center_mm=np.asarray([2000.0, 0.0, 0.0], dtype=np.float64),
                    half_extents_mm=np.asarray([10.0, 10.0, 10.0], dtype=np.float64),
                    name="seen_00",
                )
            ]
        )
        guard.set_perceived_fixtures([])

        refused = guard.evaluate(ctx)
        self.assertFalse(refused.accepted, msg=refused.message)
        self.assertIn("post", refused.message)

    def test_the_capsule_proxy_cannot_see_the_tool_on_a_joint_only_context(self) -> None:
        """The reason the shipped default is ``fcl``.

        ``gate_joint_target`` builds a context with joints and NO pose: every commanded joint move,
        and cuRobo's final-configuration gate. The capsule path builds its tool capsule from
        ``target_pose``, so on that context the gripper is not modelled at all and a finger driven
        into the forearm is accepted. The exact-mesh path places the gripper meshes from the joints.
        """
        folded = JointPositions([1.95, 0.38, -1.33, -0.55, 2.00, 0.79])
        ctx = self._ctx(joints=folded, arm=_ur_arm())

        capsule = SelfCollisionGuard(SelfCollisionSafetyConfig(backend="capsule"))
        self.assertTrue(capsule.evaluate(ctx).accepted, "the proxy is expected to be blind here")

        mesh = SelfCollisionGuard(SelfCollisionSafetyConfig(backend="fcl"))
        decision = mesh.evaluate(ctx)
        if mesh._fcl_status != "ok":  # no engine / no bundle -> it degraded to the proxy, nothing to assert
            self.skipTest(f"exact-mesh backend unavailable: {mesh._fcl_status}")
        self.assertFalse(decision.accepted)
        self.assertIn("lfinger", decision.message or "")

    def test_fcl_backend_falls_back_to_capsule_without_joints(self) -> None:
        # backend='fcl' that cannot run the mesh check (no joints/arm, or python-fcl absent)
        # falls back cleanly to the capsule path -- NOT an UNAVAILABLE stub, no crash.
        cfg = SelfCollisionSafetyConfig(backend="fcl")
        guard = SelfCollisionGuard(cfg)
        d = guard.evaluate(self._ctx(pose=_pose()))
        self.assertTrue(d.accepted)  # trivial base+tool scene accepts via the capsule fallback

    def test_fcl_backend_module_flags_real_self_penetration(self) -> None:
        # The fcl mesh backend catches the flipped-branch self-penetration (forearm|gripper,
        # ~43 mm) the capsule proxy SKIPS, and accepts the clean natural branch. Skipped if the
        # optional python-fcl / mesh bundle is absent.
        from src.robot.safety._fcl_self_collision import make_backend
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm
        backend = make_backend("ur5e")
        if backend is None:
            self.skipTest("python-fcl / ur5e mesh bundle not available (optional dependency)")
        flipped = np.array([3.011, -2.227, -1.915, 0.994, -3.010, -0.005])
        natural = np.array([-0.682, -1.106, 1.878, -0.773, -0.681, -3.141])
        hit_flipped = backend.evaluate(ur_link_transforms_mm("ur5e", flipped), 180.0, (), 10.0)
        hit_natural = backend.evaluate(ur_link_transforms_mm("ur5e", natural), 180.0, (), 10.0)
        self.assertIsNotNone(hit_flipped)
        assert hit_flipped is not None  # for type-checkers
        self.assertIn("forearm", hit_flipped[0])
        self.assertIsNone(hit_natural)

    def test_kuka_no_arm_chain_still_checks_tool_and_fixtures(self) -> None:
        # KUKA has no description; arm-vs-arm coverage and the base are absent, but the
        # tool capsule still pairs against the fixtures.
        fx = FixtureBoxConfig(
            name="enclosure",
            center_mm=(400.0, 0.0, 300.0),
            half_extents_mm=(50.0, 50.0, 50.0),
        )
        cfg = SelfCollisionSafetyConfig(fixtures=[fx])
        guard = SelfCollisionGuard(cfg)
        d = guard.evaluate(self._ctx(
            pose=_pose(),
            joints=JointPositions([0.0] * 6),
            arm=_kuka_arm(),
        ))
        # Tool capsule sits inside the fixture -> reject.
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.SELF_COLLISION)

    def test_an_arm_reporting_a_model_nobody_describes_has_no_arm_capsules(self) -> None:
        # Only base + tool + fixture: the Isaac arm's old answer, a KUKA's model, the dummy's.
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig())
        for arm in (_reporting("sim", "isaac-sim"), _kuka_arm(), _reporting("dummy", "dummy-6dof")):
            with self.subTest(model=arm.capabilities.model):
                ctx = self._ctx(pose=_pose(), joints=JointPositions([0.0] * 6), arm=arm)
                self.assertIsNone(guard._ur_arm_capsules(ctx))
                self.assertIsNone(guard.model_for(arm))

    def test_a_sim_arm_reporting_the_ur_it_drives_gets_ur_arm_capsules(self) -> None:
        # No key since P0d D5: the Isaac arm names its UR and the guard takes the chain from that.
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig())
        ctx = self._ctx(pose=_pose(), joints=JointPositions([0.0] * 6), arm=_sim_arm())
        caps = guard._ur_arm_capsules(ctx)
        self.assertIsNotNone(caps)
        assert caps is not None  # for type-checkers
        self.assertGreaterEqual(len(caps), 5)  # UR5e -> 6 link capsules from 7 origins
        self.assertEqual("ur5e", guard.model_for(_sim_arm()))

    def test_the_vendor_does_not_enter_the_rule(self) -> None:
        """The owner's choice, pinned both ways (in aurora, 2026-09-22): the model decides, through the arm registry.

        A UR driven by another vendor's transport (ROS 2, later) is judged as the UR it reports, and a vendor 'ur'
        double reporting a model nobody describes gets nothing: the vendor string is in neither answer.
        """
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig())
        self.assertEqual("ur3e", guard.model_for(_reporting("ros2", "ur3e", simulated=False)))
        self.assertIsNone(guard.model_for(_reporting("ur", "isaac-sim", simulated=False)))

    def test_one_guard_judges_one_arm(self) -> None:
        """The exact meshes are cached per guard, and the model now comes from each arm: a second one is refused."""
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig(backend="fcl"))
        guard._exact_mesh_backend("ur5e")
        guard._exact_mesh_backend("ur5e")  # the same arm again is the cache, not a refusal
        with self.assertRaises(RuntimeError) as caught:
            guard._exact_mesh_backend("ur3e")
        self.assertIn("one safety preflight per arm", str(caught.exception))

    def test_the_base_yaw_rotates_arm_capsules(self) -> None:
        # The base-frame reconcile rotates the DH arm-link capsules about +Z. 180 deg -> [-x, -y, z]
        # (the measured Isaac-UR5e-USD-vs-official-UR-DH offset, the Isaac arm's own number since 2026-10-17).
        # A real arm's 0.0 leaves the capsules untouched.
        from src.willy_sim.guard_numbers import ISAAC

        joints = JointPositions([0.1, -1.2, 1.4, -1.7, -1.57, 0.0])
        ctx = self._ctx(pose=_pose(), joints=joints, arm=_sim_arm())
        base = SelfCollisionGuard(SelfCollisionSafetyConfig())
        rot = SelfCollisionGuard(SelfCollisionSafetyConfig(), base_yaw_deg=ISAAC.base_yaw_deg)
        self.assertEqual((0.0, 180.0), (base.base_yaw_deg, rot.base_yaw_deg))
        caps0 = base._ur_arm_capsules(ctx)
        caps180 = rot._ur_arm_capsules(ctx)
        assert caps0 is not None and caps180 is not None  # for type-checkers
        self.assertEqual(len(caps0), len(caps180))
        for c0, c1 in zip(caps0, caps180, strict=True):
            np.testing.assert_allclose(c1.p0, [-c0.p0[0], -c0.p0[1], c0.p0[2]], atol=1e-6)
            np.testing.assert_allclose(c1.p1, [-c0.p1[0], -c0.p1[1], c0.p1[2]], atol=1e-6)
            self.assertEqual(c0.radius_mm, c1.radius_mm)  # pure rotation: radius unchanged

    def test_the_fingers_lie_along_the_closing_axis_as_the_hands_registry_file_says(self) -> None:
        """The capsule proxy's tool is the hand's fingers from its registry file (since 2026-10-17, the owner's cell's
        ``tool_model: finger``): along the grasp's closing axis (R[:, 0]), centred on the TCP, as long as the open jaw
        is across its fingers' outer faces and as thick as a finger is wide. A guard built with no hand models the hand
        every arm bundle carries, the 2F-85: 85 + 2 x 31.35 = 147.7 mm and 13.5 mm, where the 150 and 16 measured off
        its USD stood."""
        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.hand import planner_hand

        for hand, span, radius in ((None, 147.7, 13.5), ("robotiq_hande", 49.99 + 2 * 10.80, 29.24 / 2)):
            with self.subTest(hand=hand or "none"):
                if hand is None:
                    guard = SelfCollisionGuard(SelfCollisionSafetyConfig())
                else:
                    named = planner_hand(RobotConfig.model_validate({"gripper": {
                        "model": hand, "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}]}}))
                    guard = SelfCollisionGuard(SelfCollisionSafetyConfig(), hand=named)
                cap = guard._tool_capsule(self._ctx(pose=_pose()))
                assert cap is not None  # for type-checkers
                self.assertAlmostEqual(radius, cap.radius_mm, places=9)
                # A rod along base X (identity quat: closing = +X), centred on the TCP at (400, 0, 300).
                np.testing.assert_allclose(cap.p0, [400.0 - span / 2.0, 0.0, 300.0], atol=1e-9)
                np.testing.assert_allclose(cap.p1, [400.0 + span / 2.0, 0.0, 300.0], atol=1e-9)

    def test_the_base_is_a_column_up_to_the_shoulder_the_description_places(self) -> None:
        """The capsule proxy's base: from the mounting face to the shoulder joint of the arm's description (its first
        joint's ``d_m``), 80 mm round; none for an arm nobody describes, whose base is not guessed."""
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig())
        for model, shoulder in (("ur5e", 162.5), ("ur10", 127.3)):
            with self.subTest(model=model):
                base = guard._base_capsule(model)
                assert base is not None  # for type-checkers
                np.testing.assert_allclose(base.p0, [0.0, 0.0, 0.0], atol=1e-12)
                np.testing.assert_allclose(base.p1, [0.0, 0.0, shoulder], atol=1e-9)
                self.assertEqual(80.0, base.radius_mm)
        self.assertIsNone(guard._base_capsule(None))

    def test_the_capsule_keys_are_refused_naming_where_their_numbers_come_from(self) -> None:
        from pydantic import ValidationError

        from src.config.schema._removed import removed_key_sentence

        for key, said in (("tool_model", "registry file"), ("tool_finger_span_mm", "registry file"),
                          ("tool_length_mm", "registry file"), ("base_height_mm", "arm's description"),
                          ("link_radii_mm", "60 mm"), ("kinematics_base_yaw_deg", "guard_numbers.py"),
                          ("whole_path_judge", "judged whole first")):
            with self.subTest(key=key):
                with self.assertRaises(ValidationError):
                    RobotSafetyConfig.model_validate({"self_collision": {key: None}})
                self.assertIn(said, removed_key_sentence(f"robot.safety.self_collision.{key}") or "")

    def test_broadphase_cull_matches_brute(self) -> None:
        # The broadphase sphere cull is conservative -> byte-identical verdict to the brute path.
        from src.robot.safety._fcl_self_collision import make_backend
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm
        backend = make_backend("ur5e")
        if backend is None:
            self.skipTest("mesh backend not available (optional dependency)")
        for q in (np.array([3.011, -2.227, -1.915, 0.994, -3.010, -0.005]),   # flipped (collision)
                  np.array([3.14, -1.0, 0.9, 0.0, 0.0, 0.0])):                 # park (clean)
            t = ur_link_transforms_mm("ur5e", q)
            self.assertEqual(backend.evaluate(t, 180.0, (), 8.0, broadphase=False),
                             backend.evaluate(t, 180.0, (), 8.0, broadphase=True))

    def test_broadphase_cull_matches_brute_over_a_sweep_with_fixtures(self) -> None:
        """The cull is what every exact-mesh verdict now runs behind, so two configurations is thin.

        The argument is geometric: a pair whose bounding spheres cannot be within the margin cannot
        violate it, because the spheres bound the meshes. This is the measurement beside the argument,
        and it includes declared fixtures, which the case above does not: a bench under the arm and a
        bin wall beside it, the two shapes a real cell declares most often.
        """
        from src.robot.safety._capsule import AxisAlignedBox
        from src.robot.safety._fcl_self_collision import make_backend
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm
        backend = make_backend("ur5e")
        if backend is None:
            self.skipTest("mesh backend not available (optional dependency)")
        fixtures = (
            AxisAlignedBox(
                name="bench", center_mm=np.array([400.0, 0.0, -60.0]),
                half_extents_mm=np.array([600.0, 600.0, 50.0]),
            ),
            AxisAlignedBox(
                name="bin_wall", center_mm=np.array([600.0, 300.0, 200.0]),
                half_extents_mm=np.array([200.0, 20.0, 200.0]),
            ),
        )
        rng = np.random.default_rng(20260912)
        refused = 0
        for index in range(60):
            t = ur_link_transforms_mm("ur5e", rng.uniform(-np.pi, np.pi, 6))
            if t is None:
                continue
            brute = backend.evaluate(t, 180.0, fixtures, 10.0, broadphase=False)
            culled = backend.evaluate(t, 180.0, fixtures, 10.0, broadphase=True)
            with self.subTest(configuration=index):
                self.assertEqual(brute, culled)
            refused += brute is not None
        # The control: a sweep that refused nothing would agree trivially, because both paths would
        # only ever have returned None.
        self.assertGreater(refused, 10, "the sweep found almost nothing to refuse")

    def test_continuous_monitor_margin_and_failsafe(self) -> None:
        # The continuous monitor STOPS on a collision within margin, PASSES a clear config, and
        # fail-safes to STOP when a check overruns its budget. SOFTWARE avoidance, not certified.
        from src.robot.safety.continuous_monitor import (
            ContinuousCollisionMonitor,
            ContinuousGuardProfile,
            MonitorStatus,
        )
        flipped = np.array([3.011, -2.227, -1.915, 0.994, -3.010, -0.005])
        park = np.array([3.14, -1.0, 0.9, 0.0, 0.0, 0.0])
        mon = ContinuousCollisionMonitor.from_model(
            "ur5e", 180.0, (), ContinuousGuardProfile(enabled=True, margin_mm=8.0))
        if mon is None:
            self.skipTest("mesh backend not available (optional dependency)")
        self.assertIs(mon.check(flipped).status, MonitorStatus.COLLISION)
        self.assertIs(mon.check(park).status, MonitorStatus.OK)
        tiny = ContinuousCollisionMonitor.from_model(
            "ur5e", 180.0, (), ContinuousGuardProfile(enabled=True, margin_mm=8.0, max_check_ms=0.0))
        assert tiny is not None  # same backend availability as mon
        v = tiny.check(park)  # clear config -> only the timeout fail-safe can stop it
        self.assertIs(v.status, MonitorStatus.TIMEOUT)
        self.assertTrue(v.stop)


class SelfCollisionWiringTests(unittest.TestCase):
    def test_every_cell_wires_the_guard(self) -> None:
        pf = SafetyPreflight.from_safety_config(RobotSafetyConfig(), WorkspaceLimitsConfig(), arm_model=None)
        self.assertIn("self_collision", pf.guard_names)

    def test_the_switch_that_left_it_out_is_refused(self) -> None:
        """No cell switches a guard off since 2026-10-17 (the owner's cleanup)."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RobotSafetyConfig.model_validate({"self_collision": {"enforce": False}})


if __name__ == "__main__":
    unittest.main()


class MeshBackendModelGateTests(unittest.TestCase):
    """The exact-mesh backend must be model-GENERAL and its degradation must never be silent.

    It used to be hardcoded to ur5e, so any other robot (the UR3e bring-up) fell through to the capsule
    proxy with NO signal -- and that proxy's default 60 mm link radius already over-rejects legitimate
    UR5e reach-down grasps, so a UR3e cell would have looked like "the robot cannot grasp" instead of
    "this cell has no exact-mesh self-collision authority".
    """

    @staticmethod
    def _known_model_without_geometry() -> tuple[str, str]:
        """A model with a DH row and a mesh directory that does not hold its bundle, as (model, mesh_dir).

        The condition is made rather than found, through the same ``mesh_dir`` a cell can set. Every
        model named here has eventually been baked, so a hunt for an unbaked arm comes back empty and
        both tests below skip themselves, and a skip is not a pass. Whatever this repository ships,
        there is always a directory without this arm in it.
        """
        import tempfile

        from src.config.arms import available_arms

        folder = tempfile.mkdtemp()
        return available_arms()[0], folder

    def test_status_distinguishes_unknown_model_from_missing_geometry(self) -> None:
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        # a model with no bundled DH chain cannot place link meshes at all
        self.assertEqual(mesh_backend_status("definitely-not-a-robot"), "unknown_model")
        # this one IS a known model (DH bundled) and has no committed mesh bundle. Proving the
        # token is "no_bundle" and not "unknown_model" is what proves the ur5e hardcode is gone.
        model, folder = self._known_model_without_geometry()
        self.assertEqual(mesh_backend_status(model, folder), "no_bundle", model)
        # The control on the control: the same arm in the directory that DOES hold it answers otherwise.
        self.assertNotEqual(mesh_backend_status(model), "no_bundle", model)

    def test_bundled_models_are_known_and_have_geometry(self) -> None:
        """A model with a committed bundle must never report unknown_model or no_bundle.
        (Whether the engine imports is host-dependent, so "ok" and "no_engine" are both acceptable.)"""
        from src.robot.safety._fcl_self_collision import mesh_backend_status
        from src.config.arms import available_arms
        from src.robot.safety.planning.environment import collision_mesh_bundle

        committed = [m for m in available_arms() if collision_mesh_bundle(m).is_file()]
        self.assertGreaterEqual(len(committed), 2, "a check over one bundle proves nothing")
        for model in committed:
            with self.subTest(model=model):
                self.assertIn(mesh_backend_status(model), {"ok", "no_engine"})

    def test_make_backend_logs_and_degrades_for_a_model_without_geometry(self) -> None:
        from src.robot.safety import _fcl_self_collision as fcl

        model, folder = self._known_model_without_geometry()
        with self.assertLogs(fcl.__name__, level="WARNING") as caught:
            backend = fcl.make_backend(model, folder)
        self.assertIsNone(backend, f"no {model} mesh bundle -> must fall back, not fabricate one")
        joined = "\n".join(caught.output)
        self.assertIn(model, joined)
        self.assertIn("no_bundle", joined)
        self.assertIn("capsule guard", joined)  # the operator is told WHAT it degraded to

    def test_unknown_model_also_degrades_loudly(self) -> None:
        from src.robot.safety import _fcl_self_collision as fcl

        with self.assertLogs(fcl.__name__, level="WARNING") as caught:
            self.assertIsNone(fcl.make_backend("definitely-not-a-robot"))
        self.assertIn("unknown_model", "\n".join(caught.output))


class UR3eCollisionBundleTests(unittest.TestCase):
    """The baked ur3e bundle is a SAFETY artifact -- assert its shape and that it is really a UR3e."""

    @staticmethod
    def _bundle(model: str):
        from src.robot.safety.planning.environment import collision_mesh_bundle

        return np.load(collision_mesh_bundle(model))

    def test_has_every_link_and_the_gripper_in_the_right_dh_frames(self) -> None:
        d = self._bundle("ur3e")
        expected = {"shoulder": 1, "upper_arm": 2, "forearm": 3,
                    "wrist_1": 4, "wrist_2": 5, "wrist_3": 6,
                    "gripper": 6, "lfinger": 6, "rfinger": 6}
        self.assertEqual({k.split("__")[0] for k in d.files}, set(expected))
        for name, frame in expected.items():
            self.assertEqual(int(d[f"{name}__frame"][0]), frame, name)
            v, f = d[f"{name}__v"], d[f"{name}__f"]
            self.assertEqual(v.shape[1], 3, name)
            self.assertEqual(f.shape[1], 3, name)
            # every face must index a real vertex (the arm links are triangle soups straight from the USD;
            # the 2F-85 meshes are indexed -- both are valid, but neither may dangle)
            self.assertGreaterEqual(int(f.min()), 0, name)
            self.assertLess(int(f.max()), v.shape[0], name)

    def test_tool0_gripper_geometry_is_shared_with_ur5e(self) -> None:
        """The 2F-85 sits on the same ISO flange on every UR e-series, so DH-frame-6 geometry is identical
        (measured: X_{tool0<-DH6} == identity for both models). Copied verbatim -> must be bit-equal."""
        a, b = self._bundle("ur3e"), self._bundle("ur5e")
        for g in ("gripper", "lfinger", "rfinger"):
            np.testing.assert_array_equal(a[f"{g}__v"], b[f"{g}__v"], err_msg=g)

    def test_arm_links_are_actually_ur3e_sized(self) -> None:
        """Guards against baking the WRONG robot's meshes into a ur3e bundle -- the failure mode that would
        silently corrupt the guard. UR3e links are materially shorter than UR5e's (DH a2/a3: 243.55/213.2
        vs 425/392.2 mm), so the long axis of the upper arm + forearm must shrink accordingly."""
        a, b = self._bundle("ur3e"), self._bundle("ur5e")
        for link, dh_ratio in (("upper_arm", 243.55 / 425.0), ("forearm", 213.2 / 392.2)):
            span_a = float(np.ptp(a[f"{link}__v"], axis=0).max())
            span_b = float(np.ptp(b[f"{link}__v"], axis=0).max())
            self.assertLess(span_a, span_b, link)
            # within 25% of the DH length ratio (the mesh also covers the joint housing beyond the link)
            self.assertAlmostEqual(span_a / span_b, dh_ratio, delta=0.25 * dh_ratio,
                                   msg=f"{link}: {span_a:.1f}/{span_b:.1f} mm vs DH ratio {dh_ratio:.3f}")


class TheCouplingPlateReachesTheGuardTests(unittest.TestCase):
    """A bundle stamped ``mounting_face`` starts at the hand's own face, not at the flange.

    ⛔ **THE GUARD READ NEITHER THE STAMP NOR A PLATE UNTIL 2026-09-10.** The bake module states the
    obligation in its own words: MOUNTING_FACE means "whatever plate sits between that face and the
    flange has to be added before the planner sees it". ``gripper_spheres.py`` reads the stamp and
    carries it to the on-box cuRobo builder, which adds the plate via ``--coupling-mm``.
    ``_fcl_self_collision.py`` had no parameter that could carry the number, so a Hand-E cell ran two
    collision models of the same hand differing by one plate thickness: the planner's with it, the
    guard's without.

    ⚠ The direction is the CONSERVATIVE one for arm-versus-hand (the guard's hand sits nearer the
    arm than the real one), so this is a disagreement rather than a hole. The defect is that a plan
    produced against one geometry was judged against another and nothing compared them.
    """

    @staticmethod
    def _backend(coupling_mm: float):
        from src.robot.safety._fcl_self_collision import make_backend

        return make_backend("ur5e", None, "robotiq_hande", coupling_mm=coupling_mm)

    def test_the_hande_bundle_is_stamped_mounting_face(self) -> None:
        """The premise, derived from the file rather than asserted. If a re-bake ever stamps it
        FLANGE, the coupling key stops applying and this says so before anything else drifts. One
        file since UM lane S08: the hand is its own bundle, composed onto every arm."""
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[1] / "src/robot/safety/data/robotiq_hande_hand_meshes.npz"
        self.assertTrue(path.is_file(), "no Hand-E bundle, so this whole class proves nothing")
        with np.load(path, allow_pickle=True) as z:
            self.assertIn("gripper__origin", z.files)
            self.assertEqual(str(np.asarray(z["gripper__origin"]).reshape(-1)[0]), "mounting_face")

    def test_the_2f85_bundle_carries_no_origin_and_is_therefore_flange(self) -> None:
        """The control. A bundle baked from a composed arm asset already sits where the hand is
        bolted, so it must NOT be shifted and the absent key is what says so."""
        import pathlib

        data = pathlib.Path(__file__).resolve().parents[1] / "src/robot/safety/data"
        z = np.load(data / "ur5e_collision_meshes.npz", allow_pickle=True)
        self.assertNotIn("gripper__origin", z.files)

    def test_a_declared_coupling_moves_the_hand_and_only_the_hand(self) -> None:
        """⭐ THE ONE THAT MATTERS. ``wrist_3`` shares frame 6 with the hand, so a shift applied by
        FRAME rather than by NAME would carry an arm link along with it."""
        a, b = self._backend(0.0), self._backend(12.0)
        if a is None or b is None:
            self.skipTest("no collision engine on this host")
        for hand in ("gripper", "lfinger", "rfinger"):
            with self.subTest(part=hand):
                delta = float(np.asarray(b._sph_c[hand]).reshape(-1)[1]
                              - np.asarray(a._sph_c[hand]).reshape(-1)[1])
                self.assertAlmostEqual(delta, 12.0, places=6)
        for arm in ("wrist_3", "wrist_2", "forearm", "upper_arm", "shoulder"):
            with self.subTest(part=arm):
                delta = float(np.asarray(b._sph_c[arm]).reshape(-1)[1]
                              - np.asarray(a._sph_c[arm]).reshape(-1)[1])
                self.assertAlmostEqual(delta, 0.0, places=6,
                                       msg=f"{arm} is an ARM link and must not move with the hand")

    def test_a_flange_bundle_ignores_the_coupling_entirely(self) -> None:
        """The other control: declaring a plate must do nothing to a bundle that already includes it,
        or a cell that sets the key once would silently move its 2F-85 as well."""
        from src.robot.safety._fcl_self_collision import make_backend

        a = make_backend("ur5e", None, None, coupling_mm=0.0)
        b = make_backend("ur5e", None, None, coupling_mm=12.0)
        if a is None or b is None:
            self.skipTest("no collision engine on this host")
        for part in ("gripper", "lfinger", "wrist_3"):
            with self.subTest(part=part):
                np.testing.assert_allclose(np.asarray(a._sph_c[part]),
                                           np.asarray(b._sph_c[part]))
