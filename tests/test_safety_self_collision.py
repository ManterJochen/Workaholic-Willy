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
                supports_joint_move=True, supports_linear_move=True,
                supports_async_move=False,
                has_native_fk=True, has_native_ik=True,
                has_force_control=False, is_simulated=False,
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
                supports_joint_move=True, supports_linear_move=True,
                supports_async_move=False,
                has_native_fk=True, has_native_ik=True,
                has_force_control=False, is_simulated=False,
            ),
        },
    )()


def _sim_arm():
    # Physically a UR5e, but reports vendor='sim' / model='isaac-sim' (like IsaacRobotArm).
    return type(
        "Arm",
        (),
        {
            "capabilities": RobotCapabilities(
                vendor="sim", model="isaac-sim", dof=6,
                supports_joint_move=True, supports_linear_move=True,
                supports_async_move=False,
                has_native_fk=True, has_native_ik=True,
                has_force_control=False, is_simulated=True,
            ),
        },
    )()


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
        # KUKA has no DH table; arm-vs-arm coverage is absent, but the
        # tool capsule still pairs against the base and fixtures.
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

    def test_sim_without_kinematics_model_has_no_arm_capsules(self) -> None:
        # A non-UR vendor gets NO arm-vs-arm capsules by default (only base+tool+fixture).
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig())
        ctx = self._ctx(pose=_pose(), joints=JointPositions([0.0] * 6), arm=_sim_arm())
        self.assertIsNone(guard._ur_arm_capsules(ctx))

    def test_sim_with_kinematics_model_gets_ur_arm_capsules(self) -> None:
        # kinematics_model='ur5e' lets the sim (a UR5e) opt into the UR DH -> real arm-vs-arm.
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig(kinematics_model="ur5e"))
        ctx = self._ctx(pose=_pose(), joints=JointPositions([0.0] * 6), arm=_sim_arm())
        caps = guard._ur_arm_capsules(ctx)
        self.assertIsNotNone(caps)
        assert caps is not None  # for type-checkers
        self.assertGreaterEqual(len(caps), 5)  # UR5e -> 6 link capsules from 7 origins

    def test_kinematics_base_yaw_rotates_arm_capsules(self) -> None:
        # The base-frame reconcile rotates the DH arm-link capsules about +Z. 180 deg -> [-x, -y, z]
        # (the measured Isaac-UR5e-USD-vs-official-UR-DH offset). Default 0.0 leaves the capsules untouched.
        joints = JointPositions([0.1, -1.2, 1.4, -1.7, -1.57, 0.0])
        ctx = self._ctx(pose=_pose(), joints=joints, arm=_sim_arm())
        base = SelfCollisionGuard(SelfCollisionSafetyConfig(kinematics_model="ur5e"))
        rot = SelfCollisionGuard(
            SelfCollisionSafetyConfig(kinematics_model="ur5e", kinematics_base_yaw_deg=180.0)
        )
        caps0 = base._ur_arm_capsules(ctx)
        caps180 = rot._ur_arm_capsules(ctx)
        assert caps0 is not None and caps180 is not None  # for type-checkers
        self.assertEqual(len(caps0), len(caps180))
        for c0, c1 in zip(caps0, caps180, strict=True):
            np.testing.assert_allclose(c1.p0, [-c0.p0[0], -c0.p0[1], c0.p0[2]], atol=1e-6)
            np.testing.assert_allclose(c1.p1, [-c0.p1[0], -c0.p1[1], c0.p1[2]], atol=1e-6)
            self.assertEqual(c0.radius_mm, c1.radius_mm)  # pure rotation: radius unchanged

    def test_tool_capsule_geometry_is_config_driven(self) -> None:
        # tool_length_mm / tool_radius_mm flow into the tool capsule.
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig(tool_length_mm=200.0, tool_radius_mm=40.0))
        cap = guard._tool_capsule(self._ctx(pose=_pose()))
        self.assertIsNotNone(cap)
        assert cap is not None  # for type-checkers
        self.assertEqual(cap.radius_mm, 40.0)
        # Identity quat -> tool +Z is the approach; the capsule runs BACK from the TCP toward the
        # flange, so tcp z=300 - 200 mm -> z=100. (This assertion read +200 -> 500 until 2026-08-09;
        # see the direction test below for why that was wrong.)
        self.assertAlmostEqual(float(cap.p1[2]), 100.0, places=3)

    def test_tool_capsule_runs_from_the_grasp_centre_TOWARD_THE_FLANGE(self) -> None:
        """The tool's material lies BEHIND the grasp centre, and the capsule must lie where it does.

        ``ctx.target_pose`` is the commanded pose, and every pose this stack commands is the TCP --
        the grasp centre, between the fingers. The gripper body, the coupling and the flange are all
        on the far side of it, along -approach; ahead of it there is only the small fingertip overhang
        and then the object.

        MEASURED 2026-08-09 by running the guard: rooted at the TCP and run FORWARD, a top-down grasp
        at (300, 0, 37) produced a capsule to (300, 0, -113) at r=70 -- 113 mm of phantom solid tool
        below the table surface, where the object and the table are and the gripper is not, while the
        real 2F-85 body (z 37..169) went unmodelled entirely. Wrong in both directions at once: it
        invents collisions where the gripper is absent and misses them where it is present.
        """
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig(tool_length_mm=150.0, tool_radius_mm=70.0))
        # A top-down grasp: 180 deg about Y, so the TCP's +Z (approach) points along base -Z.
        pose = Pose(
            position_mm=np.array([300.0, 0.0, 37.0]),
            quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
            frame=Frame.BASE,
            label="topdown-grasp",
        )
        cap = guard._tool_capsule(self._ctx(pose=pose))
        assert cap is not None  # for type-checkers
        lowest = min(float(cap.p0[2]), float(cap.p1[2]))
        highest = max(float(cap.p0[2]), float(cap.p1[2]))
        self.assertAlmostEqual(lowest, 37.0, places=3, msg="the capsule must not reach below the TCP")
        self.assertAlmostEqual(highest, 187.0, places=3, msg="it must extend up toward the flange")
        self.assertGreaterEqual(lowest, 0.0, "no part of the modelled tool may sit below the table")

    def test_finger_tool_model_lays_capsule_along_closing_axis(self) -> None:
        # tool_model='finger' -> a thin capsule ALONG the grasp closing axis (R[:, 0]),
        # length tool_finger_span_mm, radius tool_finger_radius_mm. Identity quat -> closing = +X.
        guard = SelfCollisionGuard(
            SelfCollisionSafetyConfig(
                tool_model="finger", tool_finger_radius_mm=16.0, tool_finger_span_mm=150.0
            )
        )
        cap = guard._tool_capsule(self._ctx(pose=_pose()))
        self.assertIsNotNone(cap)
        assert cap is not None  # for type-checkers
        self.assertEqual(cap.radius_mm, 16.0)
        # Horizontal rod along base X (closing), centred on the tcp (400,0,300): endpoints +/-75 mm in X.
        self.assertAlmostEqual(float(cap.p0[0]), 325.0, places=3)
        self.assertAlmostEqual(float(cap.p1[0]), 475.0, places=3)
        self.assertAlmostEqual(float(cap.p0[2]), 300.0, places=3)
        self.assertAlmostEqual(float(cap.p1[2]), 300.0, places=3)

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
    def test_enforce_true_registers_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"self_collision": {"enforce": True}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig())
        self.assertIn("self_collision", pf.guard_names)

    def test_enforce_false_omits_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"self_collision": {"enforce": False}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig())
        self.assertNotIn("self_collision", pf.guard_names)


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
    def _known_model_without_geometry() -> str:
        """A model whose DH chain is bundled and whose mesh bundle is not.

        Read off the tree rather than written down, because baking a bundle for one of these
        is a normal thing to do and must move this example rather than break the test.
        """
        from src.robot.safety._ur_kinematics import UR_DH_TABLES_M
        from src.robot.safety.planning.environment import collision_mesh_bundle

        for model in sorted(UR_DH_TABLES_M):
            if not collision_mesh_bundle(model, None).exists():
                return model
        raise AssertionError("every model with a DH chain now ships geometry; this test needs "
                             "a different example of the no_bundle branch")

    def test_status_distinguishes_unknown_model_from_missing_geometry(self) -> None:
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        # a model with no bundled DH chain cannot place link meshes at all
        self.assertEqual(mesh_backend_status("definitely-not-a-robot"), "unknown_model")
        # this one IS a known model (DH bundled) and has no committed mesh bundle. Proving the
        # token is "no_bundle" and not "unknown_model" is what proves the ur5e hardcode is gone.
        self.assertEqual(mesh_backend_status(self._known_model_without_geometry()), "no_bundle")

    def test_bundled_models_are_known_and_have_geometry(self) -> None:
        """A model with a committed bundle must never report unknown_model or no_bundle.
        (Whether the engine imports is host-dependent, so "ok" and "no_engine" are both acceptable.)"""
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        for model in ("ur5e", "ur3e", "ur10e"):
            self.assertIn(mesh_backend_status(model), {"ok", "no_engine"}, model)

    def test_make_backend_logs_and_degrades_for_a_model_without_geometry(self) -> None:
        from src.robot.safety import _fcl_self_collision as fcl

        model = self._known_model_without_geometry()
        with self.assertLogs(fcl.__name__, level="WARNING") as caught:
            backend = fcl.make_backend(model)
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
