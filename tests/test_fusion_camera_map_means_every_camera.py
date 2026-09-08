"""`robot.grasping.fusion.cameras` names EVERY camera this cell fuses, the primary included.

⛔⛔ IT USED TO MEAN "EVERY CAMERA EXCEPT THE PRIMARY", AND THE TWO SHIPPED PROFILES DISAGREED ABOUT
IT. `robot.sim.yaml` listed its own primary and a real cell's profile did not, so the same key taught
opposite shapes depending on which file an operator read first. Two things came out of that:

* The boot banner counts this map. A correctly configured two-camera cell listed one camera there and
  was told "this cell has 1 calibrated camera and will grasp SINGLE-VIEW. Add a camera", on every
  build, while it was fusing two.
* An operator who followed the schema's own worked example and listed the primary got the opposite
  failure: the primary was then expected among the cameras a pick WAITS for, it delivers through the
  cell's main perception source and never through the extra-camera rig, so it was permanently
  missing. A warning on every pick under `degrade`, a raised refusal on every pick under `refuse`.

So the map is the whole inventory now, and the two things that must NOT follow from that are pinned
here: the primary stays out of the list a pick waits for, and it stays out of the extra-camera rig,
because it is already open.

⚠ The primary's calibration can now be written twice, once in the map and once in the
`extrinsics_artifact_path` key beside it. That key stays, because it is the one
`from_robot_config` names when it refuses a cell with no CAMERA to BASE transform. Two artifacts for
one camera means the cell is calibrated differently depending on which loader ran, so the root
refuses it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError

from src.config.schema.app import AppConfig
from src.robot.execution.autonomous_grasp.builders import apply_orchestrator_overlays
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator


def _tree(*, primary: str, cameras: dict[str, str], scalar: str | None) -> dict:
    """The SHIPPED tree with the two halves of the question replaced.

    Built from the real config rather than by hand, because a hand-built AppConfig needs two dozen
    required fields that have nothing to do with cameras, and because starting from what ships means
    these tests fail if the shipped tree ever stops satisfying the rule they describe.
    """
    from src.config import load_config

    tree = load_config().model_dump(mode="json")
    tree["camera"]["cameras"]["primary_rig_id"] = primary
    fusion = tree["robot"]["grasping"]["fusion"]
    fusion["enabled"] = True
    fusion["extrinsics_artifact_path"] = scalar
    fusion["cameras"] = {
        cam_id: {"enabled": True, "mounting_mode": "eye_to_hand",
                 "extrinsics_artifact_path": path}
        for cam_id, path in cameras.items()
    }
    return tree


class TheCalibrationIsStatedOnceTests(unittest.TestCase):
    def test_the_primary_may_be_listed_when_both_names_agree(self) -> None:
        AppConfig.model_validate(_tree(
            primary="webcam_main",
            cameras={"webcam_main": "cal/left.json", "cam_right": "cal/right.json"},
            scalar="cal/left.json",
        ))

    def test_two_artifacts_for_the_primary_are_REFUSED_at_load(self) -> None:
        """The failure this catches is not a typo, it is a cell that is calibrated differently on two
        code paths and gives no sign of it."""
        with self.assertRaises(ValidationError) as caught:
            AppConfig.model_validate(_tree(
                primary="webcam_main",
                cameras={"webcam_main": "cal/OTHER.json", "cam_right": "cal/right.json"},
                scalar="cal/left.json",
            ))

        message = str(caught.exception)
        self.assertIn("webcam_main", message)
        self.assertIn("cal/OTHER.json", message)
        self.assertIn("cal/left.json", message)

    def test_a_map_without_the_primary_is_still_fine(self) -> None:
        """Nothing forces a cell to list its primary; a single-camera cell has no map at all."""
        AppConfig.model_validate(_tree(
            primary="webcam_main", cameras={"cam_right": "cal/right.json"}, scalar="cal/left.json",
        ))

    def test_a_cell_with_no_scalar_is_not_second_guessed(self) -> None:
        """An eye-in-hand cell leaves the scalar unset and passes a resolver in code."""
        AppConfig.model_validate(_tree(
            primary="webcam_main",
            cameras={"webcam_main": "cal/left.json", "cam_right": "cal/right.json"},
            scalar=None,
        ))


class ThePrimaryIsNotWAITED_ForTests(unittest.TestCase):
    """It delivers through the cell's main perception source, so it can never arrive via the rig."""

    @staticmethod
    def _configured_ids(cameras: list[str], *, primary: str | None) -> tuple[str, ...]:
        """A REAL grasping config through the REAL overlay function. The attribute under test is set
        there and nowhere else, so a double would only prove that a double can hold a tuple."""
        from src.config.schema.robot.grasping_schema import RobotGraspingConfig
        from src.robot.execution.autonomous_grasp.config import GraspMode

        grasping = RobotGraspingConfig.model_validate({
            "fusion": {
                "enabled": True,
                "geometry": {"enabled": True},
                "cameras": {c: {"extrinsics_artifact_path": f"{c}.json"} for c in cameras},
            },
        })
        # A real orchestrator, because the overlay writes a dozen of its attributes on the way to
        # the one under test and a stand-in would have to grow each of them by trial and error.
        orchestrator = BinPickingOrchestrator(
            arm=SimpleNamespace(),  # type: ignore[arg-type]
            calculator=SimpleNamespace(),  # type: ignore[arg-type]
            perception=SimpleNamespace(),  # type: ignore[arg-type]
        )
        runtime = SimpleNamespace(orchestrator=orchestrator)
        # The resolver builder is stubbed because it LOADS each camera's calibration artifact and
        # refuses fail-closed when one is missing, which is right and is a different test. What is
        # under test here is which ids a pick is told to expect, and that is derived from the config
        # map rather than from what loaded: deriving it from what loaded is the defect
        # `test_camera_policy_can_actually_fire.py` exists for.
        with mock.patch(
            "src.robot.execution.autonomous_grasp.builders.build_config_frame_resolvers",
            return_value={},
        ):
            apply_orchestrator_overlays(
                runtime, grasping, resolved_mode=GraspMode.AUTO, primary_camera_id=primary)  # type: ignore[arg-type]
        return orchestrator.configured_camera_ids

    def test_the_primary_is_excluded_from_what_a_pick_waits_for(self) -> None:
        ids = self._configured_ids(["cam_left", "cam_right"], primary="cam_left")

        self.assertEqual(("cam_right",), ids)

    def test_without_a_named_primary_every_camera_is_expected(self) -> None:
        """What every caller got before, and what the simulator runners still get: they build their
        own rig and hand it in, so nothing there is delivered by a main source."""
        ids = self._configured_ids(["cam_left", "cam_right"], primary=None)

        self.assertEqual(("cam_left", "cam_right"), ids)

    def test_a_one_camera_cell_waits_for_nothing(self) -> None:
        ids = self._configured_ids(["cam_left"], primary="cam_left")

        self.assertEqual((), ids)


if __name__ == "__main__":
    unittest.main()
