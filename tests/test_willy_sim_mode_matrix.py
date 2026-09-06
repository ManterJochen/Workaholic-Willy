"""P1 — the mode-matrix harness aggregator + cell-command builder (pure parts; no Isaac).

The subprocess driver (run_matrix) is on-box only; here we pin the pure logic the harness depends on:
correct per-cell argv, honest aggregation (pass-rate, median lift, known-limited note, failures kept).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.willy_sim.run_mode_matrix import (
    aggregate_only,
    cell_command,
    format_table,
    matrix_from_cells,
)


class CellCommandTests(unittest.TestCase):
    def test_m1_easy(self) -> None:
        cmd = cell_command("py", "m1", "easy", runs=3, result_json="r.json")
        self.assertEqual(
            cmd,
            ["py", "-m", "src.willy_sim.run_m1_pick", "--mode", "easy",
             "--runs", "3", "--result-json", "r.json"],
        )

    def test_m2_carries_prompt_and_optional_artifacts(self) -> None:
        cmd = cell_command("py", "m2", "closed_loop", runs=2, result_json="r.json",
                           record_log="rec.jsonl", debug_frames="frames")
        self.assertIn("--prompt", cmd)
        self.assertIn("a red cube", cmd)
        self.assertIn("--record-log", cmd)
        self.assertIn("rec.jsonl", cmd)
        self.assertIn("--debug-frames", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "closed_loop")


class AggregatorTests(unittest.TestCase):
    def test_pass_rate_and_median_lift(self) -> None:
        cell = {
            "scene": "m2", "mode": "easy", "passed": 2, "runs": 3,
            "results": [{"lift_mm": 95.0}, {"lift_mm": 0.2}, {"lift_mm": 99.0}],
        }
        agg = matrix_from_cells([cell])
        s = agg["matrix"]["m2"]["easy"]
        self.assertAlmostEqual(s["pass_rate"], 2 / 3)
        self.assertEqual(s["lift_mm_median"], 95.0)
        self.assertEqual(s["status"], "ok")
        self.assertIsNone(s["note"])

    def test_known_limited_cell_annotated(self) -> None:
        # Both m1+closed_loop and eih+closed_loop are measured NOT-VIABLE (P1.C) and carry a note.
        for scene in ("eih", "m1"):
            agg = matrix_from_cells([{"scene": scene, "mode": "closed_loop", "passed": 0, "runs": 3, "results": []}])
            self.assertIn("NOT-VIABLE", agg["matrix"][scene]["closed_loop"]["note"])

    def test_failed_cell_is_kept_not_dropped(self) -> None:
        agg = matrix_from_cells([{"scene": "m1", "mode": "auto", "status": "timeout"}])
        s = agg["matrix"]["m1"]["auto"]
        self.assertEqual(s["status"], "timeout")
        self.assertIsNone(s["pass_rate"])  # no honest number for a failed cell

    def test_format_table_runs(self) -> None:
        agg = matrix_from_cells([{"scene": "m1", "mode": "easy", "passed": 3, "runs": 3, "results": []}])
        table = format_table(agg)
        self.assertIn("m1", table)
        self.assertIn("easy", table)


class AggregateOnlyTests(unittest.TestCase):
    def test_reads_result_jsons_from_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "m1_easy.result.json").write_text(
                json.dumps({"scene": "m1", "mode": "easy", "passed": 3, "runs": 3, "results": []}),
                encoding="utf-8",
            )
            out = Path(td) / "matrix.json"
            agg = aggregate_only(td, str(out))
            self.assertTrue(out.exists())
            self.assertEqual(agg["matrix"]["m1"]["easy"]["passed"], 3)


if __name__ == "__main__":
    unittest.main()


class RobotAxisTests(unittest.TestCase):
    """A robot dimension on the only harness that measures several cells together.

    It matters because NOTHING else guards a second robot: the U12 soak gate is robot-agnostic and
    synthetic, and stays green through a completely broken UR3e cell. Without this axis the whole UR3e
    bring-up could be broken by an unrelated change and every automated gate would still pass.
    """

    def test_the_default_robot_leaves_the_argv_untouched(self) -> None:
        """A single-robot matrix must be byte-identical to every one taken before this axis existed --
        otherwise the comparison to earlier measurements is lost."""
        from backend.src.willy_sim.run_mode_matrix import DEFAULT_ROBOT, cell_command

        self.assertEqual(
            cell_command("py", "m1", "easy", runs=3, result_json="r.json", robot_model=DEFAULT_ROBOT),
            cell_command("py", "m1", "easy", runs=3, result_json="r.json"),
        )

    def test_another_robot_is_selected_on_the_command_line(self) -> None:
        from backend.src.willy_sim.run_mode_matrix import cell_command

        cmd = cell_command("py", "m1", "easy", runs=3, result_json="r.json", robot_model="ur3e")
        self.assertEqual(cmd[cmd.index("--robot-model") + 1], "ur3e")

    def test_radial_closing_is_opt_in(self) -> None:
        """The shorter arm needs it (a UR3e cannot reach a TANGENTIAL top-down close at any azimuth);
        leaving it off must change nothing."""
        from backend.src.willy_sim.run_mode_matrix import cell_command

        self.assertNotIn(
            "--radial-closing", cell_command("py", "m1", "easy", runs=1, result_json="r.json"))
        self.assertIn(
            "--radial-closing",
            cell_command("py", "m1", "easy", runs=1, result_json="r.json", radial_closing=True))

    def test_cells_are_labelled_by_robot_only_when_it_is_not_the_default(self) -> None:
        from backend.src.willy_sim.run_mode_matrix import scene_label

        self.assertEqual(scene_label("m1"), "m1")
        self.assertEqual(scene_label("m1", "ur5e"), "m1")
        self.assertEqual(scene_label("m1", "ur3e"), "ur3e/m1")

    def test_two_robots_appear_as_separate_rows(self) -> None:
        """The point of the axis: both cells measured side by side, neither hidden by the other."""
        from backend.src.willy_sim.run_mode_matrix import matrix_from_cells

        agg = matrix_from_cells([
            {"scene": "m1", "robot": "ur5e", "mode": "easy", "passed": 10, "runs": 10, "results": []},
            {"scene": "ur3e/m1", "robot": "ur3e", "mode": "easy", "passed": 9, "runs": 10, "results": []},
        ])
        self.assertEqual(agg["matrix"]["m1"]["easy"]["pass_rate"], 1.0)
        self.assertEqual(agg["matrix"]["ur3e/m1"]["easy"]["pass_rate"], 0.9)

    def test_known_limits_follow_the_scene_not_the_robot(self) -> None:
        """The KNOWN_LIMITED notes describe the REFINER, so they must still annotate a cell running on
        another arm -- otherwise a known-unviable combination would read as an unexplained failure."""
        from backend.src.willy_sim.run_mode_matrix import matrix_from_cells

        agg = matrix_from_cells([
            {"scene": "ur3e/m2", "robot": "ur3e", "mode": "closed_loop",
             "passed": 0, "runs": 3, "results": []},
        ])
        self.assertIn("NOT-VIABLE", agg["matrix"]["ur3e/m2"]["closed_loop"]["note"] or "")
