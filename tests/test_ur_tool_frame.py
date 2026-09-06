"""The flange->TCP contract on a real UR cell: declared, composed or verified -- never assumed.

Every Pose crossing the RobotArm Protocol is the TCP (the grasp centre). What the CONTROLLER calls the
TCP is whatever sits in its tool register, and this repo had no representation of that fact outside the
sim driver. MEASURED 2026-08-09 from the repo's own numbers: at the UR factory default (identity at the
flange) a top-down grasp commanding z=37 mm drives the FLANGE to 37 mm and the fingertips to z=-95 mm --
95 mm below the table -- and nothing downstream catches it, because the WorkspaceGuard bounds the
COMMANDED number and no profile declares a table fixture.

Bucket (2): analytical + mock-driven, never run against a physical controller.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import RobotConnectionError
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.pose import URPose
from src.robot.drivers.ur.tool_frame import (
    compare_tool_frames,
    derive_active_tool_frame,
    tool_frame_matrix,
)
from src.robot.safety._ur_kinematics import ur_link_transforms_mm

#: The 2F-85 on a UR flange, as the Isaac driver hardcodes it (sim/arm.py): 132 mm along flange +Y plus
#: a -90 deg rotation about flange X, mapping the TCP +Z (approach) onto flange +Y.
_2F85_OFFSET = (0.0, 132.0, 0.0)
_2F85_QUAT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
_Q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]


def _cfg(source="willy", **tool):
    return RobotConfig.model_validate({
        "vendor": "ur",
        "safety": {"payload": {"enforce": False}},
        "gripper": {"tool_frame": {
            "source": source, "offset_mm": _2F85_OFFSET, "rotation_quat_xyzw": _2F85_QUAT, **tool,
        }},
    })


def _conn(controller_tool, model="ur5e"):
    """A mocked connection whose FK reports a controller carrying ``controller_tool`` (None = bare)."""
    links = ur_link_transforms_mm(model, np.asarray(_Q, dtype=np.float64))
    assert links is not None
    active = links[-1] if controller_tool is None else links[-1] @ controller_tool
    conn = MagicMock()
    conn.is_connected = True
    conn.get_joint_positions.return_value = list(_Q)
    conn.fk.return_value = URPose.from_T(active).to_ur_list()
    # connect()'s tool-frame check reads the controller's CURRENT TCP via fk_current (the
    # no-argument getForwardKinematics), because the q-form is corrupted by any preceding
    # motion -- see URConnection.fk_current for the measurement. Same active frame here.
    conn.fk_current.return_value = URPose.from_T(active).to_ur_list()
    conn.is_steady.return_value = True
    # what the controller reports as "the current TCP pose" is the same active frame
    conn.get_tcp_pose.return_value = URPose.from_T(active).to_ur_list()
    return conn


class DerivationTests(unittest.TestCase):
    """The measurement the whole contract rests on, and why it needs no new SDK symbol."""

    def test_the_controllers_tool_frame_is_recovered_exactly(self):
        """X = inv(base->flange from the bundled DH table) @ getForwardKinematics(q).

        Both halves already exist and are already on the safety-critical path -- the singularity guard
        finite-differences arm.fk() on every gated Cartesian move -- so this check's SDK dependency is a
        strict SUBSET of what move() already requires. No setTcp, no getTCPOffset, no hasattr probe:
        none of those appears anywhere in this repo, and the mock spec lists are grepped from the DRIVER
        rather than the SDK, so a suite alone cannot tell a real RTDE method from an invented one.
        (Since 2026-08-17 ur_rtde==1.6.5 IS installed here and every symbol the driver calls was
        verified present against it -- so that gap is now closed by a separate measurement, not by
        this suite.)
        """
        declared = tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT)
        for model in ("ur3e", "ur5e", "ur10e"):
            with self.subTest(model=model):
                links = ur_link_transforms_mm(model, np.asarray(_Q, dtype=np.float64))
                assert links is not None
                observed = derive_active_tool_frame(model, _Q, links[-1] @ declared)
                assert observed is not None
                d_t, d_r = compare_tool_frames(observed, declared)
                self.assertLess(d_t, 1e-6)
                self.assertLess(d_r, 1e-3)

    def test_the_two_modes_predict_measurably_different_observations(self):
        """A pass proves WHICH mode the cell is in, not merely that some number matched.

        'willy' composes on our side, so the controller must be bare -> the derivation is identity.
        'polyscope' means the controller holds the transform -> the derivation is the declared one.
        They differ by the whole tool: 132.0 mm and 90.0 deg for the 2F-85.
        """
        declared = tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT)
        links = ur_link_transforms_mm("ur5e", np.asarray(_Q, dtype=np.float64))
        assert links is not None
        bare = derive_active_tool_frame("ur5e", _Q, links[-1])
        assert bare is not None
        d_t, d_r = compare_tool_frames(bare, declared)
        self.assertAlmostEqual(d_t, 132.0, places=3)
        self.assertAlmostEqual(d_r, 90.0, places=3)

    def test_an_underivable_model_returns_none_rather_than_a_wrong_number(self):
        self.assertIsNone(derive_active_tool_frame("not-a-robot", _Q, np.eye(4)))


class ConnectGateTests(unittest.TestCase):
    def test_undeclared_refuses_before_the_socket_opens(self):
        """Silence is an unmeasured cell, not a default."""
        arm = URRobotArm(RobotConfig.model_validate(
            {"vendor": "ur", "safety": {"payload": {"enforce": False}}}))
        arm._conn = _conn(None)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        arm._conn.connect.assert_not_called()
        msg = str(ctx.exception)
        self.assertIn("undeclared", msg)
        self.assertIn("willy", msg)
        self.assertIn("polyscope", msg)

    def test_willy_mode_accepts_a_bare_controller(self):
        arm = URRobotArm(_cfg("willy"))
        arm._conn = _conn(None)
        arm.connect()
        arm._conn.disconnect.assert_not_called()

    def test_willy_mode_refuses_a_controller_that_already_holds_a_tool(self):
        """Composing on our side while the controller also composes double-applies the transform."""
        arm = URRobotArm(_cfg("willy"))
        arm._conn = _conn(tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT))
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        arm._conn.disconnect.assert_called_once()
        self.assertIn("132", str(ctx.exception))

    def test_polyscope_mode_accepts_a_matching_controller(self):
        arm = URRobotArm(_cfg("polyscope"))
        arm._conn = _conn(tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT))
        arm.connect()
        arm._conn.disconnect.assert_not_called()

    def test_polyscope_mode_refuses_a_bare_controller(self):
        """The exact September day-one hazard: the pendant TCP was never set, so every commanded pose
        drives the flange where the grasp centre belongs -- 132 mm off, into the table."""
        arm = URRobotArm(_cfg("polyscope"))
        arm._conn = _conn(None)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        arm._conn.disconnect.assert_called_once()
        self.assertIn("132", str(ctx.exception))

    def test_every_refusal_after_the_socket_opens_leaves_the_connection_closed(self) -> None:
        """The invariant, asserted across ALL of them rather than one at a time.

        It did not hold. The "no bundled DH table for this model" refusal raised without disconnecting,
        while the mismatch refusal beside it did -- so that one path left a live RTDE control script
        owning the arm, with the payload already pushed, while the caller was told the connection had
        failed. A one-shot CLI hides that behind process exit; the operator console cannot, and it
        cannot recover either: ``URConnection.connect()`` early-returns on an already-open connection,
        so every retry skips the socket, re-pushes the payload and fails identically, forever.

        Written as a loop on purpose. The next refusal added to ``connect()`` is covered by this test
        the day it is written, which is the only version of this assertion that stays true.
        """
        cases = {
            "controller carries a tool in willy mode":
                (_cfg("willy"), _conn(tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT))),
            "controller is bare in polyscope mode":
                (_cfg("polyscope"), _conn(None)),
            "no bundled DH table for the configured model":
                (_cfg("willy"), _conn(None, model="ur5e")),
        }
        # The third case needs a model the kinematics table does not know; the config carries the model
        # and the capabilities read it, so an unknown one makes derive_active_tool_frame return None.
        cases["no bundled DH table for the configured model"][1].fk.return_value = (
            URPose.from_T(np.eye(4)).to_ur_list()
        )

        for name, (config, conn) in cases.items():
            with self.subTest(name):
                arm = URRobotArm(config)
                if name.startswith("no bundled"):
                    arm._capabilities = replace(arm._capabilities, model="ur-does-not-exist")
                arm._conn = conn
                with self.assertRaises(RobotConnectionError):
                    arm.connect()
                conn.disconnect.assert_called_once()

    def test_the_tolerance_is_honoured(self):
        """A physical arm carries a per-unit kinematic calibration, so the derivation is not exact on
        hardware: measured offline, a correctly-configured arm spreads ~1.4 mm at a 0.5 mm/5e-4 rad
        perturbation and ~4.6 mm at a pessimistic 1.0 mm/1e-3 rad. The default tolerance is 10 mm."""
        nudged = tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT).copy()
        nudged[0, 3] += 4.0
        arm = URRobotArm(_cfg("polyscope"))
        arm._conn = _conn(nudged)
        arm.connect()
        arm2 = URRobotArm(_cfg("polyscope", verify_tolerance_mm=1.0))
        arm2._conn = _conn(nudged)
        with self.assertRaises(RobotConnectionError):
            arm2.connect()


class PoseCompositionTests(unittest.TestCase):
    """A mode that declares it composes and then does not is worse than no mode at all."""

    def test_willy_mode_reports_the_TCP_from_a_bare_controller_pose(self):
        arm = URRobotArm(_cfg("willy"))
        arm._conn = _conn(None)
        arm._motion.conn = arm._conn    # the MotionController was built with the real connection
        arm.connect()
        links = ur_link_transforms_mm("ur5e", np.asarray(_Q, dtype=np.float64))
        assert links is not None
        expected = links[-1] @ tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT)
        np.testing.assert_allclose(arm.get_tcp_pose().position_mm, expected[:3, 3], atol=1e-6)

    def test_willy_mode_round_trips_a_commanded_pose(self):
        """TCP -> controller -> TCP must be identity, or the arm drifts by the tool on every move."""
        arm = URRobotArm(_cfg("willy"))
        arm._conn = _conn(None)
        tcp = Pose(position_mm=np.array([400.0, -100.0, 250.0]),
                   quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                   frame=Frame.BASE, label="target")
        back = arm._pose_from_controller(arm._pose_to_controller(tcp), label="rt")
        np.testing.assert_allclose(back.position_mm, tcp.position_mm, atol=1e-6)

    def test_willy_mode_offsets_the_commanded_pose_by_the_whole_tool(self):
        """The point of the mode: the controller is told the FLANGE target, not the grasp centre."""
        arm = URRobotArm(_cfg("willy"))
        arm._conn = _conn(None)
        tcp = Pose(position_mm=np.array([400.0, 0.0, 300.0]),
                   quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                   frame=Frame.BASE, label="target")
        sent = arm._pose_to_controller(tcp)
        moved = float(np.linalg.norm(np.asarray([sent.x, sent.y, sent.z]) - tcp.position_mm))
        self.assertAlmostEqual(moved, 132.0, places=3)

    def test_polyscope_mode_passes_poses_through_untouched(self):
        """The controller already applies the transform; applying it twice is the same 132 mm bug."""
        arm = URRobotArm(_cfg("polyscope"))
        arm._conn = _conn(tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT))
        tcp = Pose(position_mm=np.array([400.0, 0.0, 300.0]),
                   quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                   frame=Frame.BASE, label="target")
        sent = arm._pose_to_controller(tcp)
        np.testing.assert_allclose([sent.x, sent.y, sent.z], tcp.position_mm, atol=1e-9)


class SchemaTests(unittest.TestCase):
    def test_a_declared_frame_may_not_be_left_at_identity(self):
        """Declaring a source and then leaving the transform at identity says the grasp centre IS the
        flange face -- an unmeasured cell wearing a measured cell's clothes."""
        for source in ("willy", "polyscope"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                RobotConfig.model_validate({"gripper": {"tool_frame": {"source": source}}})

    def test_an_unnormalised_quaternion_is_rejected(self):
        with self.assertRaises(ValueError):
            RobotConfig.model_validate({"gripper": {"tool_frame": {
                "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
                "rotation_quat_xyzw": (0.0, 0.0, 0.0, 0.5),
            }}})

    def test_the_default_is_inert(self):
        """Every existing config keeps loading unchanged; the driver, not the schema, refuses."""
        tf = RobotConfig().gripper.tool_frame
        self.assertEqual(tf.source, "undeclared")
        self.assertEqual(tf.offset_mm, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()


class PlannerAndGuardConsumeTheTruthTests(unittest.TestCase):
    """cuRobo and the Coal/self-collision guard must both work from the RIGHT frame -- and neither may
    change what the caller asked for.

    They reach the truth by different routes, and conflating them is the bug:

    * **Coal (exact-mesh self-collision) is JOINT-based.** ``SelfCollisionGuard._evaluate_fcl`` needs
      ``ctx.target_joints`` and places the bundled meshes -- including the 2F-85 tool0 meshes -- off the
      DH chain. That is frame-independent and correct by construction, PROVIDED the joints it is given
      solve the right Cartesian target. They do, because ``move()`` pre-resolves IK through
      ``ik()`` -> ``_pose_to_controller``.
    * **cuRobo is POSE-based, and its end-effector is ``tool0``.** ``build_ur_config.py`` attaches the
      2F-85 collision spheres TO tool0 and the planner's docstring says "Plan tool0 -> pose". It never
      talks to the controller, so its goal must be the FLANGE pose in BOTH ownership modes -- polyscope
      does not reach it. Handing it the grasp centre was wrong twice over: the flange would be driven to
      the grasp point, AND the gripper's collision spheres would sit a whole tool-length past the target,
      so the "collision-free" plan would be computed for a gripper that is not where the planner thinks.
    """

    @staticmethod
    def _curobo_arm(source: str):
        cfg = RobotConfig.model_validate({
            "vendor": "ur",
            "ur": {"motion_planner": "curobo"},
            "safety": {"payload": {"enforce": False}},
            "gripper": {"tool_frame": {
                "source": source, "offset_mm": _2F85_OFFSET, "rotation_quat_xyzw": _2F85_QUAT,
            }},
        })
        arm = URRobotArm(cfg)
        arm._conn = _conn(None if source == "willy" else tool_frame_matrix(_2F85_OFFSET, _2F85_QUAT))
        planner = MagicMock()
        planner.plan.return_value = [list(_Q)]
        arm._curobo_ur = planner
        return arm, planner

    def _tcp(self):
        return Pose(position_mm=np.array([400.0, 0.0, 300.0]),
                    quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                    frame=Frame.BASE, label="grasp")

    def test_curobo_is_planned_in_the_flange_frame_in_BOTH_modes(self):
        """The polyscope case is the one that would have been missed: the controller applies the tool
        frame for moveL, but cuRobo never asks the controller anything."""
        for source in ("willy", "polyscope"):
            with self.subTest(source=source):
                arm, planner = self._curobo_arm(source)
                tcp = self._tcp()
                arm.move(tcp)
                planned = planner.plan.call_args[0][0]
                offset = float(np.linalg.norm(planned.position_mm - tcp.position_mm))
                self.assertAlmostEqual(offset, 132.0, places=3,
                                       msg="cuRobo must receive the flange goal, not the grasp centre")

    def test_the_callers_pose_is_never_mutated_and_is_what_gets_reported(self):
        """The frame conversion is an internal detail. The caller asked for a TCP pose; the executed
        result must say so, and their Pose object must come back untouched."""
        arm, planner = self._curobo_arm("willy")
        tcp = self._tcp()
        before = np.array(tcp.position_mm, copy=True)
        arm.move(tcp)
        np.testing.assert_allclose(tcp.position_mm, before, atol=0.0)
        reported = planner.execute.call_args[0][1]
        np.testing.assert_allclose(reported.position_mm, before, atol=1e-9)

    def test_the_self_collision_gate_runs_on_curobos_own_final_joints(self):
        """Joint-based, so the Coal path is immune to the frame question -- it checks the configuration
        the arm will actually be in, not a pose someone might have mis-framed."""
        arm, planner = self._curobo_arm("willy")
        final = [0.2, -1.1, 1.0, -0.5, 1.4, 0.3]
        planner.plan.return_value = [list(_Q), final]
        seen = {}
        arm._gate_planned_config = lambda pose, joints: seen.setdefault("q", list(joints.values)) and None
        arm.move(self._tcp())
        np.testing.assert_allclose(seen["q"], final, atol=1e-9)

    def test_an_undeclared_tool_frame_leaves_curobo_untouched(self):
        """Undeclared never reaches motion (connect() refuses), so the conversion must be a no-op there
        rather than silently applying a zero transform that looks like a real one."""
        cfg = RobotConfig.model_validate({"vendor": "ur", "ur": {"motion_planner": "curobo"}})
        arm = URRobotArm(cfg)
        tcp = self._tcp()
        np.testing.assert_allclose(arm._pose_to_flange(tcp).position_mm, tcp.position_mm, atol=0.0)
