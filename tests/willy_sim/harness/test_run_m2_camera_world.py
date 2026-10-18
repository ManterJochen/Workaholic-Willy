"""The M2 reference gate plans with a live camera world by default and says what each run stood on.

``camera_world_for`` decides what the cell boots with: the overhead camera's world and the ``sim_camera_world`` layer,
or a decline for a control run with the layer left as the caller gave it. ``camera_world_line`` is printed beside the
rate: the weakest stamp of each run counted, the runs where a camera could not vouch, and the runs that did not pick.
"""

from __future__ import annotations

import unittest

from src.robot.core.camera_world import CameraWorldDecline
from src.willy_sim.harness.camera_world import SimCameraWorld
from src.willy_sim.run_m2_pick import CAMERA_WORLD_LAYER, camera_world_for, camera_world_line


class WhatTheGateBootsWithTests(unittest.TestCase):
    def test_by_default_the_overhead_world_and_its_layer(self) -> None:
        given = {"extra_profiles": ["tilted"], "hand": "robotiq_hande"}
        world, kwargs = camera_world_for(None, given)
        self.assertEqual(SimCameraWorld("overhead"), world)
        self.assertEqual(["tilted", CAMERA_WORLD_LAYER], kwargs["extra_profiles"])
        self.assertEqual("robotiq_hande", kwargs["hand"])
        self.assertEqual(["tilted"], given["extra_profiles"], "the caller's kwargs were changed in place")

    def test_the_layer_is_added_once(self) -> None:
        _, kwargs = camera_world_for(None, {"extra_profiles": [CAMERA_WORLD_LAYER]})
        self.assertEqual([CAMERA_WORLD_LAYER], kwargs["extra_profiles"])
        _, bare = camera_world_for(None, None)
        self.assertEqual([CAMERA_WORLD_LAYER], bare["extra_profiles"])

    def test_a_reason_declines_and_leaves_the_profiles_alone(self) -> None:
        world, kwargs = camera_world_for("control: M2 without the camera world", {"extra_profiles": ["tilted"]})
        self.assertEqual(CameraWorldDecline("control: M2 without the camera world"), world)
        self.assertEqual(["tilted"], kwargs["extra_profiles"])


class TheLineBesideTheRateTests(unittest.TestCase):
    def test_it_counts_each_use_and_names_what_did_not_pick(self) -> None:
        rows = [
            {"run": 0, "succeeded": True, "camera_world": "planned", "camera_world_raised": False},
            {"run": 1, "succeeded": True, "camera_world": "planned", "camera_world_raised": False},
            {"run": 2, "succeeded": False, "camera_world": "planned", "camera_world_raised": False,
             "motion_status": "planning_failed"},
            {"run": 3, "succeeded": False, "camera_world": None, "camera_world_raised": True},
            {"run": 4, "succeeded": False, "camera_world": "planned", "camera_world_raised": False,
             "motion_status": "execution_failed", "failure": "execution_failed: the approach was refused"},
        ]
        self.assertEqual(
            "camera world: none 1, planned 4; CameraWorldUnavailable raised in 1 run(s) (3); "
            "not picked: run 2 planning_failed, run 4 execution_failed: the approach was refused",
            camera_world_line(rows),
        )

    def test_a_refused_reset_park_is_counted_beside_the_rate(self) -> None:
        rows = [
            {"run": 0, "succeeded": True, "camera_world": "planned", "camera_world_raised": False},
            {"run": 1, "succeeded": True, "camera_world": "planned", "camera_world_raised": False,
             "park": "set in the sim"},
        ]
        self.assertEqual(
            "camera world: planned 2; CameraWorldUnavailable raised in 0 run(s); "
            "reset park refused before 1 run(s): 1 (set in the sim)",
            camera_world_line(rows),
        )

    def test_the_planned_stamps_that_carried_a_keep_out_are_counted(self) -> None:
        """A PLANNED stamp says what its world left out, and the gate line says on how many it did."""
        rows = [
            {"run": 0, "succeeded": True, "camera_world": "planned", "camera_world_raised": False,
             "planned_stamps": 5, "kept_out_stamps": 5},
            {"run": 1, "succeeded": False, "camera_world": "planned", "camera_world_raised": False,
             "planned_stamps": 2, "kept_out_stamps": 1, "failure": "execution_failed",
             "motion_message": "the planner found no path"},
        ]
        self.assertEqual(
            "camera world: planned 2; CameraWorldUnavailable raised in 0 run(s); "
            "kept out on 6 of 7 planned motion stamp(s); "
            "not picked: run 1 execution_failed (the planner found no path)",
            camera_world_line(rows),
        )

    def test_a_clean_declined_run(self) -> None:
        rows = [{"run": i, "succeeded": True, "camera_world": "declined", "camera_world_raised": False}
                for i in range(2)]
        self.assertEqual("camera world: declined 2; CameraWorldUnavailable raised in 0 run(s)", camera_world_line(rows))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
