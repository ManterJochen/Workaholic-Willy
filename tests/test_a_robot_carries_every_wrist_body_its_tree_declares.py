"""A robot built from a tree carries every wrist camera body the tree declares, camera open or not.

The audit of 2026-09-23 found that ``Robot.from_tree(tree)`` with no camera handed in carried no D415 body: the
planner, the exact guard and the self filter planned the UR10 and the Hand-E without the housing or the bracket,
while ``Cell`` and ``PlannerStart`` read the same camera section and carried it. Examples 03, 04, 05 and 16, and any
recovery script after the wrist calibration, took that door. A camera that is not open still hangs on the arm, so the
body is read from the tree's camera section, not from the cameras handed in.

A body that cannot be placed yet (the camera is not calibrated) is refused, and only a stated reason lets the robot
move without it, as the calibration sweep does. ``Robot.from_config`` reads no camera section, and says so. Names
that are new with this change are imported inside the tests, so the file loads against the tree before it and each
test fails there on its own assertion or its own missing name.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.calibration.serialization import FlangeToTcp, save_cam_to_tool
from src.config import ConfigTree
from src.config.tree import LoadedTree
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution.robot import Robot
from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired
from tests._wrist_body import camera_section, camera_to_tool, declared_frame, wrist_cell, wrist_rig


def _tree(*rigs: SimpleNamespace) -> LoadedTree:
    """A loaded tree whose robot section is the wrist cell and whose camera section holds ``rigs``.

    The root is the repository's ``config/``, so the camera registry the body is checked against is the shipped one.
    """
    return LoadedTree(tree=ConfigTree.from_directory(profile=None),
                      config=SimpleNamespace(robot=wrist_cell(), camera=camera_section(*rigs)))


class _FixedOwner:
    """An open fixed camera: the one a robot is handed while its wrist camera stays shut."""

    def __init__(self) -> None:
        from tests.test_camera_world_wiring import _Owner, _rig

        self.rig_id = "overhead"
        self._owner = _Owner("overhead")
        self.rig = _rig("overhead")

    def handle(self) -> object:
        return self._owner.handle()

    def calibration(self) -> object:
        return self._owner.calibration()


class _Artifacts(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def _rig(self, *, calibrated: bool = True, **changes: object) -> SimpleNamespace:
        rig = wrist_rig(calibrated=calibrated, **changes)  # type: ignore[arg-type]
        if rig.extrinsics is not None:
            rig.extrinsics.artifact_path = str(save_cam_to_tool(
                self.root / f"eih_{rig.rig_id}.json", camera_to_tool(), rig_id=rig.rig_id,
                flange_to_tcp=FlangeToTcp.from_matrix("willy", declared_frame())))
        return rig


class TheTreeIsReadCameraOpenOrNotTests(_Artifacts):

    def test_a_robot_handed_no_camera_carries_the_declared_body_in_the_guard_and_the_planner(self) -> None:
        """⛔ The finding: before, the arm held no body and the robot said 'not handed any camera'."""
        robot = Robot.from_tree(_tree(self._rig()), gripper=None)
        assert isinstance(robot.arm, URRobotArm)
        held = robot.arm._preflight.wrist_bodies(robot.arm)  # noqa: SLF001 (what the guard and the self filter read)
        self.assertEqual([body.link_name for body in held], ["wrist_camera_wrist"])
        assert robot.wrist_bodies is not None
        self.assertEqual(robot.wrist_bodies.bodies, held)
        self.assertIn("wrist camera wrist_camera_wrist: realsense_d435", robot.wrist_body_line())
        self.assertIsNone(robot.camera_world, "no camera was handed in, so no world was built")

    def test_a_switched_off_wrist_camera_is_still_carried(self) -> None:
        robot = Robot.from_tree(_tree(self._rig(enabled=False)), gripper=None)
        self.assertEqual([body.link_name for body in robot.arm._preflight.wrist_bodies(robot.arm)],  # noqa: SLF001
                         ["wrist_camera_wrist"])

    def test_a_robot_handed_only_its_fixed_camera_still_carries_the_wrist_body(self) -> None:
        from tests.test_camera_world_wiring import _rig as fixed_rig

        tree = _tree(fixed_rig("overhead"), self._rig())
        robot = Robot.from_tree(tree, gripper=None, cameras=[_FixedOwner()])
        self.assertEqual([body.link_name for body in robot.arm._preflight.wrist_bodies(robot.arm)],  # noqa: SLF001
                         ["wrist_camera_wrist"])
        assert robot.camera_world is not None and robot.camera_world.world is not None
        self.assertEqual(robot.camera_world.cameras, ("overhead",))

    def test_a_handed_wrist_camera_the_tree_does_not_hold_is_carried_beside_the_trees(self) -> None:
        from tests._wrist_body import WristOwner

        robot = Robot.from_tree(_tree(self._rig()), gripper=None, cameras=[WristOwner("side")])
        self.assertEqual(sorted(body.link_name for body in robot.arm._preflight.wrist_bodies(robot.arm)),  # noqa: SLF001
                         ["wrist_camera_side", "wrist_camera_wrist"])

    def test_the_body_is_the_one_every_other_door_resolves(self) -> None:
        """The build, PlannerStart and the desk resolve through ``WristBodies.from_config``; so does the tree door."""
        rig = self._rig()
        expected = WristBodies.from_config(wrist_cell(), camera_section(rig))
        robot = Robot.from_tree(_tree(rig), gripper=None)
        assert robot.wrist_bodies is not None
        self.assertEqual([body.to_dict() for body in robot.wrist_bodies.bodies],
                         [body.to_dict() for body in expected.bodies])


class ABodyNothingPlacesTests(_Artifacts):

    def test_an_uncalibrated_body_is_refused_and_names_the_reason_to_give(self) -> None:
        from src.robot.execution.wrist_bodies import WristBodyUnplaced

        with self.assertRaises(WristBodyUnplaced) as caught:
            Robot.from_tree(_tree(self._rig(calibrated=False)), gripper=None)
        said = str(caught.exception)
        self.assertIn("nothing places the body on the flange", said)
        self.assertIn('Robot.from_tree(tree, unmodelled_wrist_body="<reason>")', said)
        self.assertIsInstance(caught.exception, WristBodyRequired, "caught by what already catches the refusal")

    def test_a_stated_reason_builds_the_robot_and_it_says_which_camera_it_moves_without(self) -> None:
        robot = Robot.from_tree(_tree(self._rig(calibrated=False)), gripper=None,
                                unmodelled_wrist_body="bench check before the first calibration")
        self.assertEqual(robot.arm._preflight.wrist_bodies(robot.arm), ())  # noqa: SLF001
        line = robot.wrist_body_line()
        self.assertIn("NOT carried, its body cannot be placed yet: 'wrist'", line)
        self.assertIn("bench check before the first calibration", line)
        assert robot.wrist_bodies is not None
        data = robot.wrist_bodies.to_dict()
        self.assertEqual([entry["rig_id"] for entry in data["unmodelled"]], ["wrist"])
        self.assertEqual(data["unmodelled_reason"], "bench check before the first calibration")

    def test_a_blank_reason_is_no_reason(self) -> None:
        from src.robot.execution.wrist_bodies import WristBodyUnplaced

        with self.assertRaises(WristBodyUnplaced):
            Robot.from_tree(_tree(self._rig(calibrated=False)), gripper=None, unmodelled_wrist_body="   ")

    def test_a_reason_is_read_only_for_a_body_that_cannot_be_placed(self) -> None:
        robot = Robot.from_tree(_tree(self._rig()), gripper=None, unmodelled_wrist_body="not needed")
        assert robot.wrist_bodies is not None
        self.assertEqual(len(robot.wrist_bodies.bodies), 1)
        self.assertEqual((robot.wrist_bodies.unmodelled, robot.wrist_bodies.unmodelled_reason), ((), ""))

    def test_a_reason_excuses_neither_a_missing_body_nor_an_unknown_camera(self) -> None:
        from src.robot.execution.wrist_bodies import WristBodyUnplaced

        with self.assertRaises(WristBodyRequired) as caught:
            Robot.from_tree(_tree(self._rig(body=False)), gripper=None, unmodelled_wrist_body="first calibration")
        self.assertNotIsInstance(caught.exception, WristBodyUnplaced)
        self.assertIn("margin_mm and bracket", str(caught.exception))
        with self.assertRaises(WristBodyRequired) as caught:
            Robot.from_tree(_tree(self._rig(calibrated=False, model="acme_cam")), gripper=None,
                            unmodelled_wrist_body="first calibration")
        self.assertNotIsInstance(caught.exception, WristBodyUnplaced)

    def test_one_placed_body_is_carried_beside_one_that_is_not(self) -> None:
        placed, unplaced = self._rig(), self._rig("second", calibrated=False)
        robot = Robot.from_tree(_tree(placed, unplaced), gripper=None, unmodelled_wrist_body="second is new")
        self.assertEqual([body.link_name for body in robot.arm._preflight.wrist_bodies(robot.arm)],  # noqa: SLF001
                         ["wrist_camera_wrist"])
        self.assertIn("NOT carried, its body cannot be placed yet: 'second'", robot.wrist_body_line())

    def _rig(self, rig_id: str = "wrist", **changes: object) -> SimpleNamespace:  # type: ignore[override]
        rig = super()._rig(**changes)
        if rig_id != "wrist":
            rig.rig_id = rig_id
        return rig


class TheOtherDoorsTests(_Artifacts):

    def test_a_robot_section_alone_reads_no_camera_section(self) -> None:
        """The control, and the calibration sweep's door: it builds the arm here and hands the body itself."""
        robot = Robot.from_config(wrist_cell(), gripper=None)
        self.assertIsNone(robot.wrist_bodies)
        self.assertEqual(robot.arm._preflight.wrist_bodies(robot.arm), ())  # noqa: SLF001

    def test_a_desk_tree_reads_no_geometry(self) -> None:
        robot = Robot.from_tree(ConfigTree.from_directory(profile="console_dummy").load())
        self.assertEqual(robot.wrist_body_line(), "wrist cameras  not read: this cell's planner and guard read no "
                                                  "geometry")


if __name__ == "__main__":
    unittest.main()
