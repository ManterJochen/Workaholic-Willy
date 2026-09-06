"""S/P5: `URRobotArm.move()` -- the function that maps every safety verdict to a typed MotionStatus on
real hardware -- driven end to end, hardware-free.

Measured gap: `grep "URRobotArm(" tests/` found exactly one construction (an I/O test that never calls
move()). Yet move() has real September callers -- hand-eye calibration (`execution/calibration.py`) is
the first thing that runs, and `service.py` drives it every pick. This pins its contract: frame check,
the cuRobo branch, IK failure, connection faults, all six guard vetoes, and the `last_reject_status`
pass-through. The `_conn` / preflight / planner are faked exactly as the sibling UR tests do.

The real day-one risk this covers is OVER-rejection (a real arm refusing every Cartesian move), not
unguarded motion: the preflight fails CLOSED on missing data, so a bug here reads as a stuck cell, and
a stuck cell with no test is how you lose a bench day.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotConnectionError,
    RobotKinematicsError,
)
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety import SafetyDecision, SafetyReason
from src.robot.safety.planning import CuroboUnavailableError

_JOINTS = JointPositions([0.0, -1.5, 1.5, 0.0, 1.5, 0.0])


def _fake_bare_flange_conn(model: str = "ur5e"):
    """A mocked URConnection whose kinematics look like a controller running a BARE FLANGE.

    connect() now derives the controller's active tool frame (inv(DH flange) @ getForwardKinematics)
    and refuses a mismatch, so a connection mock has to answer those two calls plausibly. A bare
    flange is what `tool_frame.source: "willy"` expects, since that mode composes on Willy's side.
    """
    from src.robot.drivers.ur.pose import URPose
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm

    q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
    conn = MagicMock()
    conn.is_connected = True
    conn.get_joint_positions.return_value = q
    links = ur_link_transforms_mm(model, np.asarray(q, dtype=np.float64))
    assert links is not None
    conn.fk.return_value = URPose.from_T(links[-1]).to_ur_list()
    # connect()'s tool-frame check reads the controller's CURRENT TCP via fk_current (the
    # no-argument getForwardKinematics), because the q-form is corrupted by any preceding
    # motion -- see URConnection.fk_current for the measurement. Same active frame here.
    conn.fk_current.return_value = URPose.from_T(links[-1]).to_ur_list()
    conn.is_steady.return_value = True
    return conn


def _pose(frame: Frame = Frame.BASE) -> Pose:
    return Pose(
        position_mm=np.array([400.0, 0.0, 300.0]),
        quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
        frame=frame,
    )


def _arm(motion_planner: str = "ik", *, connected: bool = True) -> URRobotArm:
    cfg = RobotConfig.model_validate({"vendor": "ur", "ur": {"motion_planner": motion_planner}})
    arm = URRobotArm(cfg)
    arm._conn = MagicMock()
    arm._conn.is_connected = connected
    return arm


class _Preflight:
    """context_for_pose is a passthrough; evaluate() returns a pre-set decision. as_motion_result is a
    class staticmethod, so the REAL reason->status mapping still runs on our crafted decision."""

    def __init__(self, decision: SafetyDecision) -> None:
        self._decision = decision

    def context_for_pose(self, *a: object, **k: object) -> object:
        return object()

    def evaluate(self, ctx: object) -> SafetyDecision:
        return self._decision


def _accept() -> SafetyDecision:
    return SafetyDecision(accepted=True, reason=SafetyReason.OK, guard="test", message="")


class FrameAndIkPathTests(unittest.TestCase):
    def test_non_base_frame_is_invalid_target(self) -> None:
        # The frame check is first and needs no connection.
        result = _arm().move(_pose(Frame.TOOL))
        self.assertIs(result.status, MotionStatus.INVALID_TARGET)

    def test_ik_failure_is_ik_failed(self) -> None:
        arm = _arm("ik")
        arm.ik = lambda pose: (_ for _ in ()).throw(RobotKinematicsError("no solution"))  # type: ignore[assignment, misc]
        self.assertIs(arm.move(_pose()).status, MotionStatus.IK_FAILED)

    def test_ik_connection_error_is_connection_error(self) -> None:
        arm = _arm("ik")
        arm.ik = lambda pose: (_ for _ in ()).throw(RobotConnectionError("dropped"))  # type: ignore[assignment, misc]
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONNECTION_ERROR)

    def test_move_to_connection_error_is_connection_error(self) -> None:
        arm = _arm("ik")
        arm.ik = lambda pose: _JOINTS  # type: ignore[assignment, misc]
        arm.get_joint_positions = lambda: _JOINTS  # type: ignore[assignment, misc]
        arm._preflight = _Preflight(_accept())  # type: ignore[assignment]
        arm._drive_pose = lambda *a, **k: (_ for _ in ()).throw(RobotConnectionError("lost mid-move"))  # type: ignore[assignment, misc]
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONNECTION_ERROR)


class GuardVetoTests(unittest.TestCase):
    """Every guard's rejection must surface as its typed MotionStatus through move()."""

    _CASES = [
        (SafetyReason.WORKSPACE, MotionStatus.WORKSPACE_REJECTED),
        (SafetyReason.JOINT_LIMIT, MotionStatus.JOINT_LIMIT_REJECTED),
        (SafetyReason.IK_QUALITY, MotionStatus.IK_QUALITY_REJECTED),
        (SafetyReason.SELF_COLLISION, MotionStatus.SELF_COLLISION_REJECTED),
        (SafetyReason.PAYLOAD, MotionStatus.PAYLOAD_REJECTED),
        (SafetyReason.CONTINUITY, MotionStatus.CONTINUITY_REJECTED),
    ]

    def test_each_guard_veto_maps_to_its_status(self) -> None:
        for reason, status in self._CASES:
            with self.subTest(reason=reason):
                arm = _arm("ik")
                arm.ik = lambda pose: _JOINTS  # type: ignore[assignment, misc]
                arm.get_joint_positions = lambda: _JOINTS  # type: ignore[assignment, misc]
                arm._preflight = _Preflight(  # type: ignore[assignment]
                    SafetyDecision(accepted=False, reason=reason, guard=reason.value, message="veto")
                )
                # move_to must NOT be reached on a veto; make it explode if it is.
                arm._drive_pose = lambda *a, **k: (_ for _ in ()).throw(AssertionError("moved after veto"))  # type: ignore[assignment, misc]
                self.assertIs(arm.move(_pose()).status, status)


class SuccessAndPassthroughTests(unittest.TestCase):
    def _armed(self) -> URRobotArm:
        arm = _arm("ik")
        arm.ik = lambda pose: _JOINTS  # type: ignore[assignment, misc]
        arm.get_joint_positions = lambda: _JOINTS  # type: ignore[assignment, misc]
        arm._preflight = _Preflight(_accept())  # type: ignore[assignment]
        return arm

    def test_accept_then_move_success(self) -> None:
        arm = self._armed()
        arm._drive_pose = lambda *a, **k: True  # type: ignore[assignment, misc]
        arm._motion.last_reject_status = None
        self.assertTrue(arm.move(_pose()).ok)

    def test_controller_reject_carries_last_reject_status(self) -> None:
        # move_to returns False; the driver must attach the controller's CLASSIFIED cause (a real
        # near-singularity -> IK_QUALITY_REJECTED), not from_bool's generic CONTROLLER_REJECTED.
        arm = self._armed()
        arm._drive_pose = lambda *a, **k: False  # type: ignore[assignment, misc]
        arm._motion.last_reject_status = MotionStatus.IK_QUALITY_REJECTED
        self.assertIs(arm.move(_pose()).status, MotionStatus.IK_QUALITY_REJECTED)

    def test_controller_reject_without_classification_is_generic(self) -> None:
        arm = self._armed()
        arm._drive_pose = lambda *a, **k: False  # type: ignore[assignment, misc]
        arm._motion.last_reject_status = None
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONTROLLER_REJECTED)


class _FakePlanner:
    def __init__(self, *, plan_result: object, execute_result: MotionResult | None = None) -> None:
        self._plan_result, self._execute_result = plan_result, execute_result

    def plan(self, pose: Pose) -> object:
        if isinstance(self._plan_result, Exception):
            raise self._plan_result
        return self._plan_result

    def execute(self, traj: object, pose: Pose, *, vel: object = None, acc: object = None) -> MotionResult:
        assert self._execute_result is not None
        return self._execute_result


class CuroboBranchTests(unittest.TestCase):
    def _arm(self, planner: _FakePlanner | None, *, connected: bool = True) -> URRobotArm:
        arm = _arm("curobo", connected=connected)
        arm._curobo_ur = planner  # type: ignore[assignment]  # bypass the lazy builder
        return arm

    def test_disconnected_is_connection_error(self) -> None:
        self.assertIs(self._arm(None, connected=False).move(_pose()).status, MotionStatus.CONNECTION_ERROR)

    def test_curobo_unavailable_is_controller_rejected(self) -> None:
        arm = self._arm(_FakePlanner(plan_result=CuroboUnavailableError("no GPU env")))
        self.assertIs(arm.move(_pose()).status, MotionStatus.CONTROLLER_REJECTED)

    def test_no_plan_is_timeout(self) -> None:
        # Empty trajectory -> no collision-free plan -> fail safe, no blind motion.
        arm = self._arm(_FakePlanner(plan_result=[]))
        self.assertIs(arm.move(_pose()).status, MotionStatus.TIMEOUT)

    def test_planned_config_gated_before_execute(self) -> None:
        # A plan exists, but the guard rejects cuRobo's ACTUAL final config -> the matching status, and
        # execute() is never reached.
        planner = _FakePlanner(plan_result=[[0.0, -1.5, 1.5, 0.0, 1.5, 0.0]])
        arm = self._arm(planner)
        veto = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, target_pose=_pose())
        arm._gate_planned_config = lambda pose, joints: veto  # type: ignore[assignment, misc]
        self.assertIs(arm.move(_pose()).status, MotionStatus.SELF_COLLISION_REJECTED)

    def test_planned_config_accepted_executes(self) -> None:
        ok = MotionResult.from_bool(True, MotionCommand.MOVE_TO, target_pose=_pose())
        planner = _FakePlanner(plan_result=[[0.0, -1.5, 1.5, 0.0, 1.5, 0.0]], execute_result=ok)
        arm = self._arm(planner)
        arm._gate_planned_config = lambda pose, joints: None  # type: ignore[assignment, misc]
        self.assertTrue(arm.move(_pose()).ok)


if __name__ == "__main__":
    unittest.main()


class ConnectPayloadRollbackTests(unittest.TestCase):
    """`connect()` pushes the configured payload, and MUST roll the connection back if that push fails.

    The driver's own comment states the invariant: "an unverified payload on a connected controller is
    more dangerous than no connection at all" -- the controller's protective-stop model would then
    under-read the real mass. Measured 2026-08-09: the handler had NO test at all, and enumerated
    `(RuntimeError, OSError, ValueError)`, so several realistic failures skipped the rollback entirely
    and left the arm connected. A handler whose whole purpose is to fail closed must not enumerate
    failure types.
    """

    @staticmethod
    def _arm_with_payload(mass_kg: float = 2.5):
        cfg = RobotConfig.model_validate({
            "vendor": "ur",
            "safety": {"payload": {"enforce": True, "mass_kg": mass_kg, "cog_mm": (0.0, 0.0, 60.0)}},
            # a real cell must declare its tool frame too; these tests are about the PAYLOAD gates
            "gripper": {"tool_frame": {"source": "willy", "offset_mm": (0.0, 132.0, 0.0), "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}},
        })
        arm = URRobotArm(cfg)
        arm._conn = _fake_bare_flange_conn()
        return arm

    def test_successful_push_keeps_the_connection(self) -> None:
        arm = self._arm_with_payload()
        arm.connect()
        arm._conn.set_payload.assert_called_once_with(2.5, (0.0, 0.0, 60.0))
        arm._conn.disconnect.assert_not_called()

    def test_every_failure_type_rolls_the_connection_back(self) -> None:
        """AttributeError and TypeError are the realistic ones and are exactly what used to escape.

        AttributeError is what a missing/renamed `setPayload` raises -- and nobody in this repo has
        confirmed that symbol exists: ur_rtde is pinned (==1.6.3) but is not a test dependency, is not
        installed on the dev box, and the mock spec lists are grepped from the driver rather than from
        the SDK, so the suite cannot tell a real method from an invented one. TypeError is a signature
        change. If either skips the rollback, the arm stays connected with an unverified payload.
        """
        for exc_type in (AttributeError, TypeError, RuntimeError, OSError, ValueError):
            with self.subTest(exception=exc_type.__name__):
                arm = self._arm_with_payload()
                arm._conn.set_payload.side_effect = exc_type("controller said no")
                with self.assertRaises(RobotConnectionError) as ctx:
                    arm.connect()
                arm._conn.disconnect.assert_called_once()
                self.assertIn("setPayload", str(ctx.exception))
                self.assertIsInstance(ctx.exception.__cause__, exc_type)

    def test_a_keyboard_interrupt_mid_push_also_disconnects_and_propagates_unchanged(self) -> None:
        """Ctrl-C during the push leaves the same dangerous state. Disconnect, then let it through --
        it must NOT be reclassified as a RobotConnectionError, or an operator's abort looks like a
        hardware fault."""
        arm = self._arm_with_payload()
        arm._conn.set_payload.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            arm.connect()
        arm._conn.disconnect.assert_called_once()

    def test_enforce_false_never_touches_the_controller_payload(self) -> None:
        """A genuinely bare flange is expressed by enforce: false, and then the driver must leave
        whatever the pendant has alone."""
        cfg = RobotConfig.model_validate({
            "vendor": "ur", "safety": {"payload": {"enforce": False, "mass_kg": 0.0}},
            "gripper": {"tool_frame": {"source": "willy", "offset_mm": (0.0, 132.0, 0.0), "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}},
        })
        arm = URRobotArm(cfg)
        arm._conn = _fake_bare_flange_conn()
        arm.connect()
        arm._conn.set_payload.assert_not_called()
        arm._conn.disconnect.assert_not_called()


class ConnectPayloadCogGateTests(unittest.TestCase):
    """`connect()` must refuse a declared mass with an unmeasured CoG.

    The mass gate's own message already instructs the operator to "set safety.payload.mass_kg AND
    cog_mm" -- and nothing checked the second half. Measured 2026-08-09: mass_kg 2.2 with the default
    cog_mm [0,0,0] connected happily and pushed `setPayload(2.2, (0.0, 0.0, 0.0))`, telling the
    controller a 2.2 kg tool is a point mass at the flange face.

    That is the shape this repo already removed once as worse-than-nothing (the `quality_threshold_mm`
    key that documented a gate which did not exist), sitting inside the very handler held up as the
    model of fail-closed config. The physical stake: at the UR3e's shipped max_mass_kg 3.0 and the
    132 mm tool this project ships, it is 3.88 N*m of wrist torque the controller does not model --
    in the protective-stop calculation AND in the gravity compensation behind get_tcp_wrench(), which
    this driver documents as the hand-over signal.
    """

    @staticmethod
    def _arm(**payload):
        cfg = RobotConfig.model_validate(
            {"vendor": "ur", "safety": {"payload": payload}, "gripper": {"tool_frame": {"source": "willy", "offset_mm": (0.0, 132.0, 0.0), "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}}}
        )
        arm = URRobotArm(cfg)
        arm._conn = _fake_bare_flange_conn()
        return arm

    def test_declared_mass_with_default_cog_is_refused_before_the_socket_opens(self) -> None:
        arm = self._arm(enforce=True, mass_kg=2.2)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        arm._conn.connect.assert_not_called()      # fail closed BEFORE touching the network
        arm._conn.set_payload.assert_not_called()
        msg = str(ctx.exception)
        self.assertIn("cog_mm", msg)
        self.assertIn("enforce: false", msg)       # the message must name its own escape hatch

    def test_a_measured_cog_connects(self) -> None:
        arm = self._arm(enforce=True, mass_kg=2.2, cog_mm=(0.0, 0.0, 60.0))
        arm.connect()
        arm._conn.set_payload.assert_called_once_with(2.2, (0.0, 0.0, 60.0))

    def test_any_nonzero_axis_counts_as_measured(self) -> None:
        """The gate detects the literal default, not 'small' -- it must not second-guess a real
        measurement that happens to be axis-centred on two of three axes."""
        for cog in ((3.0, 0.0, 0.0), (0.0, -4.5, 0.0), (0.0, 0.0, 1.0)):
            with self.subTest(cog=cog):
                arm = self._arm(enforce=True, mass_kg=1.0, cog_mm=cog)
                arm.connect()
                arm._conn.set_payload.assert_called_once_with(1.0, cog)

    def test_the_zero_mass_gate_still_fires_first_and_unchanged(self) -> None:
        """Both gates refuse a fully-default payload block; the mass one owns that message."""
        arm = self._arm(enforce=True, mass_kg=0.0)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        self.assertIn("mass_kg: 0.0", str(ctx.exception))

    def test_enforce_false_is_unaffected(self) -> None:
        arm = self._arm(enforce=False, mass_kg=2.2)
        arm.connect()
        arm._conn.set_payload.assert_not_called()


class MoveHomeIsGatedTests(unittest.TestCase):
    """`move_home()` is the natural FIRST motion on a new cell, and it was the only commanded motion on
    the real path that no safety guard ever saw.

    It went straight to `MotionController.move_home`, which checks the workspace box, logs a warning,
    and -- in its own words -- proceeds anyway. MEASURED 2026-08-09 on the September UR3e: the shipped
    UR5e-authored `HOME_JOINTS_DEFAULT` puts the grasp centre at z = 561.9 mm and r = 466.9 mm (93.4%
    of a 500 mm reach) while that cell's `workspace_limits` stop at z = 320. So the first thing an
    operator does would have driven outside the declared box on a log line.

    Two halves, and neither subsumes the other: `gate_joint_target` runs the DESTINATION guards (joint
    limits, self-collision, payload) and deliberately SKIPS the Cartesian workspace box, because that
    guard does not apply to a point-to-point joint command -- but here the box matters, because a home
    pose is a place the arm will sit.
    """

    @staticmethod
    def _arm(home=None, **cfg_extra):
        cfg = RobotConfig.model_validate({
            "vendor": "ur", "safety": {"payload": {"enforce": False}},
            "gripper": {"tool_frame": {
                "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
                "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
            }},
            **({"home_joint_positions": home} if home is not None else {}),
            **cfg_extra,
        })
        arm = URRobotArm(cfg)
        arm._conn = MagicMock()
        arm._conn.is_connected = True
        # FK stands in for the controller: report the pose the caller wants tested.
        arm._conn.fk.return_value = [0.4, 0.0, 0.3, 0.0, 3.14, 0.0]   # (400, 0, 300) mm, inside
        arm._conn.moveJ.return_value = True
        arm._motion.conn = arm._conn   # the MotionController was built with the real connection
        return arm

    def test_a_home_pose_inside_the_box_still_moves(self) -> None:
        arm = self._arm()
        self.assertTrue(arm.move_home())
        self.assertTrue(arm._conn.moveJ.called)

    def test_a_home_pose_outside_the_workspace_is_REFUSED_not_warned(self) -> None:
        """The exact September case: the shipped default lands above a UR3e's z_max."""
        arm = self._arm()
        arm._conn.fk.return_value = [0.0, -0.22, 0.562, 0.0, 3.14, 0.0]   # z = 562 mm, outside
        self.assertFalse(arm.move_home())
        arm._conn.moveJ.assert_not_called()

    def test_a_guard_rejection_stops_it(self) -> None:
        """The destination guards must reach move_home like every other joint move."""
        from backend.src.robot.core import MotionCommand, MotionResult, MotionStatus

        arm = self._arm()
        arm._preflight = MagicMock()
        arm._preflight.gate_joint_target.return_value = MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS, message="test veto",
        )
        self.assertFalse(arm.move_home())
        arm._conn.moveJ.assert_not_called()

    def test_an_unreadable_home_pose_refuses_rather_than_proceeding(self) -> None:
        """If FK cannot be read, the box cannot be checked -- and an unchecked home move is exactly what
        this guard exists to stop."""
        arm = self._arm()
        arm._conn.fk.side_effect = RuntimeError("controller dropped")
        self.assertFalse(arm.move_home())
        arm._conn.moveJ.assert_not_called()

    def test_the_cell_can_declare_its_own_home(self) -> None:
        """The real path had NO field for this; the sim has had one all along."""
        measured_ur3e_home = (3.14, -1.0, 1.0, -1.7, -1.57, 0.0)
        arm = self._arm(home=measured_ur3e_home)
        self.assertEqual(tuple(arm._home_joints), measured_ur3e_home)

    def test_an_explicit_argument_still_wins_over_config(self) -> None:
        cfg = RobotConfig.model_validate({"vendor": "ur", "home_joint_positions": (1.0,) * 6})
        arm = URRobotArm(cfg, home_joints=[0.5] * 6)
        self.assertEqual(list(arm._home_joints), [0.5] * 6)
