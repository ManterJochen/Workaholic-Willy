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

Listing the primary cannot state its calibration twice. Every camera's calibration is declared once,
on its rig, `camera.cameras.rigs[<id>].extrinsics`, and a map entry carries `enabled` only, so an
entry that writes an artifact path is refused at load. What the root refuses across the two sections
is an id in the map that names no rig, because that camera's calibration would have nowhere to be.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError

from src.config.schema.app import AppConfig
from src.robot.execution.autonomous_grasp.builders import apply_orchestrator_overlays
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator


def _tree(*, primary: str, cameras: list[str], calibrated: bool = True) -> dict:
    """The SHIPPED tree with the two halves of the question replaced.

    Built from the real config rather than by hand, because a hand-built AppConfig needs two dozen
    required fields that have nothing to do with cameras, and because starting from what ships means
    these tests fail if the shipped tree ever stops satisfying the rule they describe.

    The primary and every mapped camera become RGB-D rigs. With ``calibrated`` each rig declares its
    calibration where it lives, ``camera.cameras.rigs[<id>].extrinsics``; the map entries carry
    ``enabled`` only.
    """
    from src.config import load_config

    tree = load_config().model_dump(mode="json")
    cameras_block = tree["camera"]["cameras"]
    template = next(rig for rig in cameras_block["rigs"] if rig["source"] == "rgbd")
    for cam_id in dict.fromkeys([primary, *cameras]):
        extrinsics = {"mounting_mode": "eye_to_hand", "artifact_path": f"cal/{cam_id}.json"} if calibrated else None
        cameras_block["rigs"].append({**template, "rig_id": cam_id, "enabled": True,
                                      "serial_number": f"serial_{cam_id}", "extrinsics": extrinsics})
    cameras_block["primary_rig_id"] = primary
    fusion = tree["robot"]["grasping"]["fusion"]
    fusion["enabled"] = True
    fusion["cameras"] = {cam_id: {"enabled": True} for cam_id in cameras}
    return tree


class TheCalibrationIsStatedOnceTests(unittest.TestCase):
    def test_the_primary_may_be_listed_beside_another_camera(self) -> None:
        AppConfig.model_validate(_tree(primary="cam_left", cameras=["cam_left", "cam_right"]))

    def test_a_second_artifact_for_a_listed_camera_is_REFUSED_at_load(self) -> None:
        """The failure this catches is not a typo, it is a cell that is calibrated differently on two
        code paths and gives no sign of it. The map entry has no key that could hold a second
        artifact."""
        tree = _tree(primary="cam_left", cameras=["cam_left", "cam_right"])
        tree["robot"]["grasping"]["fusion"]["cameras"]["cam_left"]["extrinsics_artifact_path"] = "cal/OTHER.json"

        with self.assertRaises(ValidationError) as caught:
            AppConfig.model_validate(tree)

        message = str(caught.exception)
        self.assertIn("cam_left", message)
        self.assertIn("extrinsics_artifact_path", message)

    def test_a_map_without_the_primary_is_still_fine(self) -> None:
        """Nothing forces a cell to list its primary; a single-camera cell has no map at all."""
        AppConfig.model_validate(_tree(primary="cam_left", cameras=["cam_right"]))

    def test_a_listed_rig_that_declares_no_calibration_is_not_second_guessed_at_load(self) -> None:
        """A profile that is not ready to run is not a malformed file. Whether each rig declares its
        calibration is decided when the cell is built and in the preflight, not at load."""
        AppConfig.model_validate(_tree(primary="cam_left", cameras=["cam_left", "cam_right"], calibrated=False))


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
                "cameras": {c: {"enabled": True} for c in cameras},
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
        # The resolver builder is stubbed because it loads each camera's calibration from its rig and
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
