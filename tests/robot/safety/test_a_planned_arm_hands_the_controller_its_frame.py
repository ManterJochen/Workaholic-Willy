"""A planned arm hands every consumer the frame it works in: the controller, the planner and the guards (P0g slice 13).

Every Pose crossing the RobotArm Protocol is the TCP (the grasp centre). What the CONTROLLER calls the TCP is whatever
sits in its tool register, and ``gripper.tool_frame.source`` declares who owns that: in ``willy`` mode the controller
runs a bare flange and the arm composes the declared transform, in ``controller`` mode the controller holds it and a
pose passes through untouched. cuRobo plans ``tool0`` and never asks the controller, so it is handed the flange goal in
both modes. Measured 2026-08-09 from the repo's own numbers: at the factory default (identity at the flange) a top-down
grasp commanding z=37 mm drives the FLANGE to 37 mm and the fingertips to z=-95 mm, 95 mm below the table.

Moved off ``tests/robot/drivers/ur/test_ur_tool_frame.py`` in P0g slice 13: the composition is ``PlannedArm``'s and reads the DECLARED
frame, so it is held on the kit's planned arm over the fake transport, and what the controller was handed is read off
the fake's tape. What stays in the UR's file is what its connect derives and refuses, and the box its motion controller
applies. Bucket (2): analytical + fake-driven, never run against a physical controller.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.matrix import position_quaternion_to_matrix
from src.robot.safety import SafetyPreflight
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from tests._arm_kit import kit_tree, planned_arm
from tests._fake_transport import chain_of, fake_transport
from tests._plan_end import OPEN_WORKSPACE, pose_where_it_ends

#: Said where each arm is built: these doubles plan or check with cuRobo and carry no camera world, so every
#: motion they command declines it, as a cuRobo cell must since the world became mandatory.
_DECLINED = "unit double: this test exercises the planner and the guard on a cuRobo arm, and no camera world is wired to it"

#: The 2F-85 on a UR flange, as the Isaac driver hardcodes it (sim/arm.py): 132 mm along flange +Y plus
#: a -90 deg rotation about flange X, mapping the TCP +Z (approach) onto flange +Y.
_2F85_OFFSET = (0.0, 132.0, 0.0)
_2F85_QUAT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
_TOOL = position_quaternion_to_matrix(_2F85_OFFSET, _2F85_QUAT)
_Q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]


def _tree(source: str, planner: str = "ik", **blocks: Any) -> Any:
    return kit_tree(planner, safety={}, gripper={"tool_frame": {
        "source": source, "offset_mm": _2F85_OFFSET, "rotation_quat_xyzw": _2F85_QUAT}}, **blocks)


def _controller(source: str) -> Any:
    """A fake controller standing at ``_Q`` whose TCP register is bare (willy) or holds the tool (controller)."""
    return fake_transport(joints=_Q, kinematics=chain_of("ur5e"), tcp_register=None if source == "willy" else _TOOL)


def _tcp(x: float = 400.0, y: float = 0.0, z: float = 300.0) -> Pose:
    return Pose(position_mm=np.array([x, y, z]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=Frame.BASE, label="grasp")


class _AcceptEverything:
    """A pipeline that passes every pose, so these tests are about the FRAME the controller is told."""

    name = "workspace"

    def evaluate(self, ctx: Any) -> Any:
        from src.robot.safety.decision import SafetyDecision

        return SafetyDecision.accept(self.name, message="ok")


def _ik_arm(source: str) -> Any:
    """An ik arm whose controller solves every pose to ``_Q`` and whose pipeline accepts everything."""
    fake = _controller(source)
    fake.always("controller_ik", list(_Q))
    return planned_arm(_tree(source), transport=fake, preflight=SafetyPreflight([_AcceptEverything()]))


def _handed(arm: Any, target: Pose, *, verb: str = "move") -> Any:
    """Command ``target`` through ``verb``; return the one move_pose call the controller was handed."""
    if verb == "move":
        result = arm.move(target)
        assert result.status.value == "executed", result.message
    else:
        arm.move_linear(target)
    (call,) = arm._transport.calls("move_pose")
    return call


class PoseCompositionTests(unittest.TestCase):
    """A mode that declares it composes and then does not is worse than no mode at all."""

    def test_willy_mode_reports_the_TCP_from_a_bare_controller_pose(self) -> None:
        arm = planned_arm(_tree("willy"), transport=_controller("willy"))
        links = ur_link_transforms_mm("ur5e", np.asarray(_Q, dtype=np.float64))
        assert links is not None
        expected = links[-1] @ _TOOL
        np.testing.assert_allclose(arm.get_tcp_pose().position_mm, expected[:3, 3], atol=1e-6)

    def test_willy_mode_round_trips_a_commanded_pose(self) -> None:
        """TCP -> controller -> TCP must be identity, or the arm drifts by the tool on every move: what the arm
        hands a bare controller, read back as the controller's TCP register, is the pose commanded."""
        arm = _ik_arm("willy")
        tcp = _tcp(400.0, -100.0, 250.0)
        arm._transport.always("read_tcp", _handed(arm, tcp).args[0])
        np.testing.assert_allclose(arm.get_tcp_pose().position_mm, tcp.position_mm, atol=1e-6)

    def test_willy_mode_offsets_the_commanded_pose_by_the_whole_tool(self) -> None:
        """The point of the mode: the controller is told the FLANGE target, not the grasp centre."""
        tcp = _tcp()
        sent = _handed(_ik_arm("willy"), tcp).args[0]
        self.assertAlmostEqual(float(np.linalg.norm(sent.position_mm - tcp.position_mm)), 132.0, places=3)

    def test_controller_mode_passes_poses_through_untouched(self) -> None:
        """The controller already applies the transform; applying it twice is the same 132 mm bug."""
        tcp = _tcp()
        sent = _handed(_ik_arm("controller"), tcp).args[0]
        np.testing.assert_allclose(sent.position_mm, tcp.position_mm, atol=1e-9)


class PlannerAndGuardConsumeTheTruthTests(unittest.TestCase):
    """cuRobo and the Coal/self-collision guard must both work from the RIGHT frame -- and neither may
    change what the caller asked for.

    They reach the truth by different routes, and conflating them is the bug:

    * Coal (exact-mesh self-collision) is JOINT-based. ``SelfCollisionGuard._evaluate_fcl`` needs
      ``ctx.target_joints`` and places the bundled meshes -- including the 2F-85 tool0 meshes -- off the
      DH chain. That is frame-independent and correct by construction, PROVIDED the joints it is given
      solve the right Cartesian target. They do, because ``move()`` pre-resolves IK through ``ik()``, which
      hands the controller the frame it runs.
    * cuRobo is POSE-based, and its end-effector is ``tool0``. ``descriptor.py`` attaches the
      2F-85 collision spheres TO tool0 and the planner's docstring says "Plan tool0 -> pose". It never
      talks to the controller, so its goal must be the FLANGE pose in BOTH ownership modes -- controller
      does not reach it. Handing it the grasp centre was wrong twice over: the flange would be driven to
      the grasp point, AND the gripper's collision spheres would sit a whole tool-length past the target,
      so the "collision-free" plan would be computed for a gripper that is not where the planner thinks.
    """

    @staticmethod
    def _curobo_arm(source: str) -> tuple[Any, MagicMock]:
        # A goal is where the stand-in's plan ends (Step 8f), and that is outside the shipped box; the box is not
        # what this class is about.
        arm: Any = planned_arm(_tree(source, "curobo", motion={"workspace": OPEN_WORKSPACE}), transport=_controller(source))
        planner = MagicMock()
        planner.plan_joint.side_effect = lambda goal, **_: [list(_Q), list(goal)]
        arm._curobo = planner
        return arm, planner

    def test_curobo_is_planned_in_the_flange_frame_in_BOTH_modes(self) -> None:
        """The controller case is the one that would have been missed: the controller applies the tool
        frame for moveL, but cuRobo never asks the controller anything."""
        for source in ("willy", "controller"):
            with self.subTest(source=source):
                arm, planner = self._curobo_arm(source)
                # The swing from the fake controller's joints clears the forearm by 9.5 mm at one sample, under the
                # guard's 10; where the arm goes is what this reads, not what it passes on the way.
                arm._preflight.gate_planned_path = lambda *_, **__: None
                tcp = _tcp()
                with arm.without_camera_world(_DECLINED):
                    arm.move(tcp)
                # cuRobo is handed a configuration, the one the arm solved the goal into: where it puts the flange is
                # the goal the planner was given, and it has to be the flange goal.
                final = planner.execute.call_args[0][0][-1]
                frames = ur_link_transforms_mm(str(arm.config.ur.model), np.asarray(final, dtype=np.float64))
                assert frames is not None
                offset = float(np.linalg.norm(np.asarray(frames[-1])[:3, 3] - tcp.position_mm))
                self.assertAlmostEqual(offset, 132.0, places=3,
                                       msg="cuRobo must receive the flange goal, not the grasp centre")

    def test_the_callers_pose_is_never_mutated_and_is_what_gets_reported(self) -> None:
        """The frame conversion is an internal detail. The caller asked for a TCP pose; the executed
        result must say so, and their Pose object must come back untouched."""
        arm, planner = self._curobo_arm("willy")
        self.enterContext(arm.without_camera_world(_DECLINED))
        # The goal is where the plan ends: a planned arm refuses a plan off its goal before execution (Step 8f).
        tcp = pose_where_it_ends(arm, _Q)
        before = np.array(tcp.position_mm, copy=True)
        arm.move(tcp)
        np.testing.assert_allclose(tcp.position_mm, before, atol=0.0)
        reported = planner.execute.call_args[0][1]
        np.testing.assert_allclose(reported.position_mm, before, atol=1e-9)

    def test_the_self_collision_gate_runs_on_curobos_own_final_joints(self) -> None:
        """Joint-based, so the Coal path is immune to the frame question -- it checks the configuration
        the arm will actually be in, not a pose someone might have mis-framed."""
        arm, planner = self._curobo_arm("willy")
        self.enterContext(arm.without_camera_world(_DECLINED))
        final = [0.2, -1.1, 1.0, -0.5, 1.4, 0.3]
        planner.plan.return_value = [list(_Q), final]
        seen: dict[str, Any] = {}
        arm._gate_planned_config = lambda pose, joints: seen.setdefault("q", list(joints.values)) and None
        arm.move(pose_where_it_ends(arm, final))
        np.testing.assert_allclose(seen["q"], final, atol=1e-9)

    def test_an_undeclared_tool_frame_leaves_curobo_untouched(self) -> None:
        """Undeclared never reaches motion (connect() refuses), so the conversion must be a no-op there
        rather than silently applying a zero transform that looks like a real one."""
        arm: Any = planned_arm(kit_tree("curobo"))
        tcp = _tcp()
        np.testing.assert_allclose(arm._pose_to_flange(tcp).position_mm, tcp.position_mm, atol=0.0)


class EveryControllerCommandIsTheFlangeInWillyModeTests(unittest.TestCase):
    """`willy` means the arm composes the tool frame, on EVERY path that reaches the controller.

    `ik()` and `move_linear` composed it. `_drive_pose`, which `move()` uses on the ik planner, did not,
    and neither did the bool `move_to` and the async `amove_to` (both removed 2026-09-19): they handed
    the grasp centre to a bare controller as its flange target. So the guard judged the joints for the right pose while
    the controller drove the flange to the grasp centre, one tool length (132 mm here) further along the approach. The
    `ursim` profile is exactly that combination: `motion_planner: ik` with `tool_frame.source: willy`.
    """

    def test_move_hands_the_controller_the_flange(self) -> None:
        arm = _ik_arm("willy")
        tcp = _tcp()
        sent = _handed(arm, tcp).args[0]
        offset = float(np.linalg.norm(sent.position_mm - tcp.position_mm))
        self.assertAlmostEqual(
            offset, 132.0, places=3,
            msg=f"move handed the controller a pose {offset:.1f} mm from the grasp "
                f"centre; a bare controller needs the flange target, 132.0 mm away")
        flange = arm._controller_pose(tcp)
        np.testing.assert_allclose(sent.position_mm, flange.position_mm, atol=1e-9)
        np.testing.assert_allclose(sent.quaternion_xyzw, flange.quaternion_xyzw, atol=1e-9)

    def test_controller_mode_hands_the_controller_the_tcp_untouched(self) -> None:
        """The control: here the controller applies the tool frame, so composing it again is the bug."""
        tcp = _tcp()
        sent = _handed(_ik_arm("controller"), tcp).args[0]
        np.testing.assert_allclose(sent.position_mm, tcp.position_mm, atol=1e-9)

    def test_the_box_a_controller_applies_is_handed_the_grasp_centre(self) -> None:
        """`robot.motion.workspace` bounds the commanded TCP: every Cartesian surface hands the controller's own box the
        grasp centre beside the flange target. How the UR's motion controller boxes what it is handed stays in
        tests/robot/drivers/ur/test_ur_tool_frame.py."""
        tcp = _tcp()
        for verb in ("move", "move_linear"):
            with self.subTest(verb=verb):
                call = _handed(_ik_arm("willy"), tcp, verb=verb)
                np.testing.assert_allclose(call.kwargs["workspace_pose"].position_mm, tcp.position_mm, atol=1e-9)
                self.assertGreater(float(np.linalg.norm(call.args[0].position_mm - tcp.position_mm)), 100.0,
                                   "the fixture needs the flange and the grasp centre apart")


if __name__ == "__main__":
    unittest.main()
