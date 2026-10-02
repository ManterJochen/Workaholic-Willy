"""The shipped desk profile runs a task as it is shipped (review of 2026-10-02).

``python -m api --profile console_dummy`` is the console at a desk with no hardware. It declared no taught pose and no
default place, so Start said "Es ist keine Standard-Ablage eingetragen", the place select offered nothing to choose
(a camera place is refused on the desk), and the dummy arm cannot be guided by hand to teach one: no task could run
until a person wrote a profile layer by hand. The profile now ships the two labelled poses the e2e's scratch layer
adds, "Ablage links" (the default place) and "Parkposition", so the desk demo runs as shipped. What is held here:

* the profile declares both poses with their labels, and ``drop_left`` as the default place;
* on the shipped tree, a console built in rehearsal and connected runs a task whose command names no place: it places at
  "Ablage links" and returns home; and the poses read back with their labels.

Honesty bucket (2): the shipped config tree as it loads; the real console, routes and ``run_task`` on the rehearsal.
"""

from __future__ import annotations

import unittest
from typing import Any


class TheDeskProfileDeclaresAPlaceTests(unittest.TestCase):
    def test_it_ships_a_labelled_default_place_and_a_park(self) -> None:
        from src.config.tree import load_tree

        tree = load_tree("console_dummy")
        self.assertTrue(tree.ok, getattr(tree, "error", tree))
        robot = tree.robot
        self.assertEqual("drop_left", robot.default_place_pose)
        poses = robot.named_poses
        self.assertEqual(("Ablage links", "Parkposition"), (poses["drop_left"].label, poses["park"].label))
        # One value per joint of the arm, as the home names them (a UR's six; the console's Home reads the same).
        self.assertEqual((6, 6), (len(poses["drop_left"].joints_deg), len(poses["park"].joints_deg)))
        self.assertIsNone(load_tree(None).robot.default_place_pose, "the base tree grew a place the cell never taught")


class TheDeskConsoleRunsATaskTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from api.app import create_app
        from api.cell import Console, set_console
        from src.config.loader import active_profile, reload_config, set_active_profile

        self._profile = active_profile()
        self.cell = Console(profile="console_dummy", stop_file=None)
        self.cell.registry.countdown_step_s = 0.05
        self._previous = set_console(self.cell)

        def restore() -> None:
            try:
                self.cell.session.release()
            finally:
                set_console(self._previous)
                set_active_profile(self._profile)
                reload_config()

        self.addCleanup(restore)
        self.client = TestClient(create_app())

    def finished(self, run_id: str) -> dict[str, Any]:
        from tests._console_task_fakes import await_run

        await_run(self.cell, run_id)
        return dict(self.client.get(f"/v1/runs/{run_id}").json())

    def test_a_command_that_names_no_place_places_at_the_default_and_returns(self) -> None:
        import tempfile
        from pathlib import Path

        self.cell.record_log_path = Path(tempfile.mkdtemp()) / "grasp_records.jsonl"
        self.assertEqual(200, self.client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)
        poses = self.client.get("/v1/poses")
        self.assertEqual(200, poses.status_code, poses.text)
        self.assertIn("Ablage links", poses.text)
        self.assertIn("Parkposition", poses.text)

        started = self.client.post("/v1/task", json={"object": "", "place": {"kind": "pose", "pose": None}})

        self.assertEqual(202, started.status_code, started.text)
        self.assertEqual(("drop_left", "Ablage links"),
                         (started.json()["plan"]["place"]["pose"], started.json()["plan"]["place"]["pose_label"]))
        body = self.finished(started.json()["id"])
        self.assertEqual(("finished", 1), (body["stop_code"], body["parts_placed"]), body)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
