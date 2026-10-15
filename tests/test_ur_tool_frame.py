"""The flange->TCP contract on a real UR cell: declared, composed or verified -- never assumed.

Every Pose crossing the RobotArm Protocol is the TCP (the grasp centre). What the CONTROLLER calls the
TCP is whatever sits in its tool register, and this repo had no representation of that fact outside the
sim driver. MEASURED 2026-08-09 from the repo's own numbers: at the UR factory default (identity at the
flange) a top-down grasp commanding z=37 mm drives the FLANGE to 37 mm and the fingertips to z=-95 mm --
95 mm below the table -- and nothing downstream catches it, because the WorkspaceGuard bounds the
COMMANDED number, and a table exists only where a profile declares one.

Bucket (2): analytical + mock-driven, never run against a physical controller.

Since P0g slice 13 this file holds what the UR adds: connect's refusals against a UR controller over the UR transport,
the UR's answers to connect's three hooks (its address, its size class, its series twin), the derivation's arithmetic,
the schema of the key, and the box its motion controller applies. How a planned arm composes the declared frame is held
in ``tests/test_a_planned_arm_hands_the_controller_its_frame.py``, and connect itself, which is ``PlannedArm``'s since
slice 15 and derives the frame through the description's chain, in ``tests/test_a_planned_arm_connects_through_its_transport.py``.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import RobotConnectionError
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.pose import URPose
from src.geometry.matrix import compare_tool_frames, position_quaternion_to_matrix
from src.robot.drivers.planned.tool_frame import derive_active_tool_frame
from src.robot.safety._ur_kinematics import ur_link_transforms_mm

#: The 2F-85 on a UR flange, as the Isaac driver hardcodes it (sim/arm.py): 132 mm along flange +Y plus
#: a -90 deg rotation about flange X, mapping the TCP +Z (approach) onto flange +Y.
_2F85_OFFSET = (0.0, 132.0, 0.0)
_2F85_QUAT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
_Q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]


def _cfg(source="willy", **tool):
    return RobotConfig.model_validate({
        "vendor": "ur", "ur": {"ip": "127.0.0.1"},
        "safety": {"payload": {"enforce": False}},
        "gripper": {"model": "robotiq_2f85", "tool_frame": {
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
        declared = position_quaternion_to_matrix(_2F85_OFFSET, _2F85_QUAT)
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
        'controller' means the controller holds the transform -> the derivation is the declared one.
        They differ by the whole tool: 132.0 mm and 90.0 deg for the 2F-85.
        """
        declared = position_quaternion_to_matrix(_2F85_OFFSET, _2F85_QUAT)
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
            {"vendor": "ur", "ur": {"ip": "127.0.0.1"}, "safety": {"payload": {"enforce": False}}, "gripper": {"model": "robotiq_2f85"}}))
        arm._conn = _conn(None)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        arm._conn.connect.assert_not_called()
        msg = str(ctx.exception)
        self.assertIn("undeclared", msg)
        self.assertIn("willy", msg)
        self.assertIn("controller", msg)

    def test_willy_mode_accepts_a_bare_controller(self):
        arm = URRobotArm(_cfg("willy"))
        arm._conn = _conn(None)
        arm.connect()
        arm._conn.disconnect.assert_not_called()

    def test_willy_mode_refuses_a_controller_that_already_holds_a_tool(self):
        """Composing on our side while the controller also composes double-applies the transform."""
        arm = URRobotArm(_cfg("willy"))
        arm._conn = _conn(position_quaternion_to_matrix(_2F85_OFFSET, _2F85_QUAT))
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        arm._conn.disconnect.assert_called_once()
        self.assertIn("132", str(ctx.exception))

    def test_controller_mode_accepts_a_matching_controller(self):
        arm = URRobotArm(_cfg("controller"))
        arm._conn = _conn(position_quaternion_to_matrix(_2F85_OFFSET, _2F85_QUAT))
        arm.connect()
        arm._conn.disconnect.assert_not_called()

    def test_controller_mode_refuses_a_bare_controller(self):
        """The exact September day-one hazard: the pendant TCP was never set, so every commanded pose
        drives the flange where the grasp centre belongs -- 132 mm off, into the table."""
        arm = URRobotArm(_cfg("controller"))
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
                (_cfg("willy"), _conn(position_quaternion_to_matrix(_2F85_OFFSET, _2F85_QUAT))),
            "controller is bare in controller mode":
                (_cfg("controller"), _conn(None)),
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
        nudged = position_quaternion_to_matrix(_2F85_OFFSET, _2F85_QUAT).copy()
        nudged[0, 3] += 4.0
        arm = URRobotArm(_cfg("controller"))
        arm._conn = _conn(nudged)
        arm.connect()
        arm2 = URRobotArm(_cfg("controller", verify_tolerance_mm=1.0))
        arm2._conn = _conn(nudged)
        with self.assertRaises(RobotConnectionError):
            arm2.connect()


def _ur_over(config: RobotConfig, conn: Any) -> URRobotArm:
    """The UR driver on ``config`` over ``conn``, a mocked connection behind the real UR transport."""
    arm = URRobotArm(config)
    arm._conn = conn
    return arm


class TheUrAnswersConnectsHooksTests(unittest.TestCase):
    """Connect is ``PlannedArm``'s since P0g slice 15 and asks three hooks a vendor answers; these are the UR's
    answers, reached through the UR's own connect, so an answer the base stops asking fails here and not on a cell.
    """

    def test_a_cell_with_no_address_is_refused_before_anything_opens(self) -> None:
        config = _cfg("willy")
        config = config.model_copy(update={"ur": config.ur.model_copy(update={"ip": None})})
        arm = _ur_over(config, _conn(None))
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        self.assertIn("robot.ur.ip declares no address", str(ctx.exception))
        arm._conn.connect.assert_not_called()

    def test_a_controller_of_another_size_is_refused_and_rolled_back(self) -> None:
        conn = _conn(None)
        conn.controller_model.return_value = "UR10"
        arm = _ur_over(_cfg("willy"), conn)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        self.assertIn("a UR10 is not a UR5", str(ctx.exception))
        conn.disconnect.assert_called_once()

    def test_a_ur3e_cell_on_a_ur3_controller_is_refused_by_its_series_twin(self) -> None:
        """The pair the size class cannot separate: both report UR3, and the twin explains the controller better."""
        config = _cfg("willy")
        config = config.model_copy(update={"ur": config.ur.model_copy(update={"model": "ur3e"})})
        conn = _conn(None, model="ur3")
        conn.controller_model.return_value = "UR3"
        arm = _ur_over(config, conn)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        self.assertIn("better explained by a 'ur3'", str(ctx.exception))
        conn.disconnect.assert_called_once()

    def test_the_control_the_same_cell_on_its_own_controller_connects(self) -> None:
        """⭐ THE CONTROL: the three refusals above are the hooks', not a cell this fixture cannot connect."""
        conn = _conn(None)
        conn.controller_model.return_value = "UR5"
        arm = _ur_over(_cfg("willy"), conn)
        arm.connect()
        conn.connect.assert_called_once()
        conn.disconnect.assert_not_called()

    def test_a_weighed_payload_reaches_the_controller_through_the_ur_transport(self) -> None:
        config = _cfg("willy")
        payload = config.safety.payload.model_copy(update={"enforce": True, "mass_kg": 2.5, "cog_mm": (0.0, 0.0, 60.0)})
        config = config.model_copy(update={"safety": config.safety.model_copy(update={"payload": payload})})
        arm = _ur_over(config, _conn(None))
        arm.connect()
        arm._conn.set_payload.assert_called_once_with(2.5, (0.0, 0.0, 60.0))
        arm._conn.disconnect.assert_not_called()


class SchemaTests(unittest.TestCase):
    def test_a_declared_frame_may_not_be_left_at_identity(self):
        """Declaring a source and then leaving the transform at identity says the grasp centre IS the
        flange face -- an unmeasured cell wearing a measured cell's clothes."""
        for source in ("willy", "controller"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                RobotConfig.model_validate({"gripper": {"model": "robotiq_2f85", "tool_frame": {"source": source}}})

    def test_polyscope_is_refused_at_load_with_the_name_it_has_now(self):
        """P0e E10: the mode names who holds the tool frame, and for every vendor that is its controller."""
        with self.assertRaisesRegex(ValueError, r"'polyscope' is 'controller' since P0e.*source: controller"):
            RobotConfig.model_validate({"gripper": {"model": "robotiq_2f85", "tool_frame": {
                "source": "polyscope", "offset_mm": (0.0, 132.0, 0.0),
                "rotation_quat_xyzw": (-0.70710678, 0.0, 0.0, 0.70710678)}}})

    def test_an_unnormalised_quaternion_is_rejected(self):
        with self.assertRaises(ValueError):
            RobotConfig.model_validate({"gripper": {"model": "robotiq_2f85", "tool_frame": {
                "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
                "rotation_quat_xyzw": (0.0, 0.0, 0.0, 0.5),
            }}})

    def test_a_non_finite_number_is_refused_at_load_with_its_key(self):
        """A NaN passes the unit-norm test (every comparison with it is false); before P0e it reached the
        driver as a NaN matrix, and the geometry builder the driver uses since then refuses it there instead,
        far from the key. The key is where it is refused, under every source."""
        bad = float("nan")
        for key, value in (("offset_mm", (0.0, bad, 0.0)), ("rotation_quat_xyzw", (bad, 0.0, 0.0, 1.0)),
                           ("offset_mm", (0.0, float("inf"), 0.0))):
            for source in ("undeclared", "willy", "controller"):
                with self.subTest(key=key, value=value, source=source):
                    tf = {"source": source, "offset_mm": (0.0, 132.0, 0.0),
                          "rotation_quat_xyzw": (-0.70710678, 0.0, 0.0, 0.70710678), key: value}
                    with self.assertRaisesRegex(ValueError, rf"gripper\.tool_frame\.{key} must hold finite"):
                        RobotConfig.model_validate({"gripper": {"model": "robotiq_2f85", "tool_frame": tf}})

    def test_the_default_is_inert(self):
        """Every existing config keeps loading unchanged; the driver, not the schema, refuses."""
        tf = RobotConfig().gripper.tool_frame
        self.assertEqual(tf.source, "undeclared")
        self.assertEqual(tf.offset_mm, (0.0, 0.0, 0.0))


class _AcceptEverything:
    """A pipeline that passes every pose, so these tests are about the FRAME the controller is told."""

    name = "workspace"

    def evaluate(self, ctx):
        from src.robot.safety.decision import SafetyDecision

        return SafetyDecision.accept(self.name, message="ok")


class _HeightBox:
    """A workspace guard that bounds height only, placed between the grasp centre and the flange."""

    def __init__(self, limit_mm: float, *, keep_below: bool) -> None:
        self.limit_mm = limit_mm
        self.keep_below = keep_below
        self.judged: list[float] = []

    def is_inside_workspace(self, pose) -> bool:
        # The box is handed a base-frame Pose since P0e, as the real guard is; the height is its z in mm.
        z = float(pose.position_mm[2])
        self.judged.append(z)
        return z <= self.limit_mm if self.keep_below else z >= self.limit_mm

    def accept(self, pose) -> None:
        return None


class TheControllerBoxJudgesTheGraspCentreTests(unittest.TestCase):
    """`workspace_limits` bound the commanded TCP, and the controller path boxes whatever it is handed.

    `MotionController.move_to` runs its own full-box check on the URPose it receives. Once willy mode
    hands it the flange target, a grasp centre inside the box whose flange is outside is refused as
    WORKSPACE_REJECTED after the preflight accepted it. `move_linear` always handed over the flange;
    since `move` does as well, both refused that way until this fix, and so did the bool and async
    twins removed on 2026-09-19.
    """

    @staticmethod
    def _arm():
        from src.robot.safety import SafetyPreflight

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"ip": "127.0.0.1"},
            "motion": {"planner": "ik"},
            "safety": {"payload": {"enforce": False}},
            "gripper": {"model": "robotiq_2f85", "tool_frame": {
                "source": "willy", "offset_mm": _2F85_OFFSET, "rotation_quat_xyzw": _2F85_QUAT,
            }},
        })
        arm = URRobotArm(cfg)
        arm._conn = _conn(None)
        arm._conn.ik.return_value = list(_Q)
        arm._conn.moveJ.return_value = True
        arm._conn.moveL.return_value = True
        arm._motion.conn = arm._conn
        arm._motion._is_singularity_risky = lambda joints: False
        arm._preflight = SafetyPreflight([_AcceptEverything()])
        return arm

    def test_every_cartesian_surface_boxes_the_grasp_centre_not_the_flange(self):
        tcp = Pose(position_mm=np.array([400.0, 0.0, 300.0]),
                   quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                   frame=Frame.BASE, label="grasp")
        tcp_z = float(tcp.position_mm[2])
        for surface in ("move", "move_linear"):
            with self.subTest(surface=surface):
                arm = self._arm()
                flange_z = float(arm._pose_to_controller(tcp).z)
                self.assertGreater(abs(flange_z - tcp_z), 50.0, "the fixture needs the two apart")
                box = _HeightBox((tcp_z + flange_z) / 2.0, keep_below=tcp_z < flange_z)
                arm._motion.guard = box
                if surface == "move":
                    result = arm.move(tcp)
                    self.assertEqual(result.status.value, "executed", result.message)
                else:
                    arm.move_linear(tcp)
                self.assertEqual(len(box.judged), 1, box.judged)
                self.assertAlmostEqual(box.judged[0], tcp_z, places=6,
                                       msg=f"{surface} boxed z={box.judged[0]:.1f}, not the grasp centre")


if __name__ == "__main__":
    unittest.main()
