"""The dataset on disk: resume that is exact rather than approximate, and rejections that are visible.

A path-traced run takes days. Two things then stop being conveniences and become requirements: a crash
must cost one scene rather than eighteen hours, and a scene that was dropped must be *findable* — a
silent gap in the numbering looks identical to a scene that was never requested.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datagen.render.writer import RENDER_ERROR_BUDGET, DatasetWriter, SceneRecord, decode_depth_png


class WriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.writer = DatasetWriter(self.root, "v1_test")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_layout_is_created(self) -> None:
        self.assertTrue((self.root / "v1_test" / "scenes").is_dir())

    def test_an_empty_dataset_has_nothing_completed(self) -> None:
        self.assertEqual(self.writer.completed(), {})

    def test_a_finished_scene_is_skipped_on_resume(self) -> None:
        self.writer.append_index(SceneRecord("sparse_000000", 0, "sparse", "ok", ("wrist",), objects=3))
        reopened = DatasetWriter(self.root, "v1_test")
        self.assertIn("sparse_000000", reopened.completed())
        self.assertEqual(reopened.completed()["sparse_000000"].objects, 3)

    def test_a_rejected_scene_counts_as_completed(self) -> None:
        """Re-running it would reject it again -- same seed, same physics. Retrying is pure cost."""
        self.writer.append_index(
            SceneRecord("pile_000004", 4, "pile", "unstable", note="still moving after 2 settles")
        )
        completed = DatasetWriter(self.root, "v1_test").completed()
        self.assertIn("pile_000004", completed)
        self.assertEqual(completed["pile_000004"].status, "unstable")
        self.assertIn("still moving", completed["pile_000004"].note)

    def _crash(self, scene_id: str, index: int = 7) -> None:
        self.writer.append_index(SceneRecord(
            scene_id, index, "bin", "render_error",
            note="the depth buffer never settled -- 611 px disagreed over a budget of 307"))

    def test_a_crashed_scene_is_RE_ATTEMPTED_while_it_still_has_budget(self) -> None:
        """A render_error is not a verdict about the scene, it is a bug in the renderer, and the
        whole point of fixing that bug is to render the scene it lost."""
        self._crash("bin_000014")
        self.assertNotIn("bin_000014", DatasetWriter(self.root, "v1_test").completed())

    def test_a_scene_that_KEEPS_crashing_is_given_up_on(self) -> None:
        """⚠ THE DEFECT THIS EXISTS FOR. `build` aborts after `_CONSECUTIVE_ERROR_LIMIT` scenes fail
        in a row, so three reproducibly-failing scenes sitting next to each other in the pending list
        abort EVERY run at the same place -- an unattended corpus can never terminate. MEASURED on the
        WS2 corpus: `bin_000014`, `pile_000061` and `pile_000084` are adjacent, each fails the
        depth-settle check on exactly 611 px against a budget of 307 every single time, and the
        shard's index grew three more `render_error` rows per restart while its scene count stayed
        at 191."""
        for _ in range(RENDER_ERROR_BUDGET):
            self._crash("bin_000014")
        completed = DatasetWriter(self.root, "v1_test").completed()
        self.assertIn("bin_000014", completed)
        # Returned as the FAILURE it is, so a summary counts it as an error and nothing reads it as ok.
        self.assertEqual(completed["bin_000014"].status, "render_error")
        self.assertIn("611 px", completed["bin_000014"].note)

    def test_a_LATER_SUCCESS_clears_the_tally(self) -> None:
        """Isaac's colour buffer is genuinely intermittent -- five consecutive runs once lost the same
        scene and a sixth rendered it perfectly. A scene that finally rendered is not exhausted, and
        leaving it counted would hide a real record behind a stale failure count."""
        for _ in range(RENDER_ERROR_BUDGET):
            self._crash("bin_000014")
        self.writer.append_index(SceneRecord("bin_000014", 7, "bin", "ok", ("wrist",), objects=5))
        completed = DatasetWriter(self.root, "v1_test").completed()
        self.assertEqual(completed["bin_000014"].status, "ok")
        self.assertEqual(completed["bin_000014"].objects, 5)

    def test_a_BIGGER_budget_re_attempts_it_which_is_the_after_a_fix_workflow(self) -> None:
        """Someone who has just fixed a renderer bug wants exactly the scenes it lost."""
        for _ in range(RENDER_ERROR_BUDGET):
            self._crash("bin_000014")
        reopened = DatasetWriter(self.root, "v1_test")
        self.assertIn("bin_000014", reopened.completed())
        self.assertNotIn("bin_000014",
                         reopened.completed(render_error_budget=RENDER_ERROR_BUDGET + 1))

    def test_the_summary_counts_a_given_up_scene_as_an_ERROR(self) -> None:
        """It must not quietly become part of the yield."""
        self.writer.append_index(SceneRecord("sparse_000000", 0, "sparse", "ok", ("wrist",)))
        for _ in range(RENDER_ERROR_BUDGET):
            self._crash("bin_000014")
        summary = DatasetWriter(self.root, "v1_test").summary()
        self.assertEqual(summary["by_status"].get("ok"), 1)
        self.assertEqual(summary["by_status"].get("render_error"), 1)

    def test_previously_failed_names_the_KNOWN_BAD_backlog(self) -> None:
        """`build`'s consecutive-failure guard means "three in a row means Isaac is broken". That
        reading is WRONG for a scene that already failed on an earlier run, and where those scenes sit
        makes it matter: a resume's pending list is every incomplete index in order, so they come
        FIRST, ahead of everything never attempted. MEASURED 2026-08-24 -- shard0 resumed with four of
        them at the head, failed three, aborted, and never reached scene 236. Three restarts got
        exactly as far."""
        self._crash("bin_000014")
        self._crash("pile_000061")
        self.writer.append_index(SceneRecord("sparse_000002", 2, "sparse", "ok", ("wrist",)))
        self.writer.append_index(SceneRecord("pile_000004", 4, "pile", "unstable"))
        failed = DatasetWriter(self.root, "v1_test").previously_failed()
        self.assertEqual(failed, {"bin_000014", "pile_000061"})

    def test_a_scene_that_LATER_RENDERED_is_not_known_bad(self) -> None:
        """Isaac's colour buffer is intermittent; a scene that came good is not a known risk."""
        self._crash("bin_000014")
        self.writer.append_index(SceneRecord("bin_000014", 7, "bin", "ok", ("wrist",)))
        self.assertEqual(DatasetWriter(self.root, "v1_test").previously_failed(), set())

    def test_an_empty_index_has_no_backlog(self) -> None:
        self.assertEqual(self.writer.previously_failed(), set())

    def test_the_index_is_flushed_per_scene(self) -> None:
        """An interrupted run must lose at most the scene it was rendering."""
        self.writer.append_index(SceneRecord("a", 0, "sparse", "ok"))
        lines = (self.root / "v1_test" / "index.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["scene_id"], "a")

    def test_the_summary_counts_by_status_and_family(self) -> None:
        for record in (
            SceneRecord("s0", 0, "sparse", "ok"),
            SceneRecord("s1", 1, "sparse", "ok"),
            SceneRecord("p0", 2, "pile", "unstable"),
            SceneRecord("p1", 3, "pile", "ok"),
        ):
            self.writer.append_index(record)
        summary = self.writer.summary()
        self.assertEqual(summary["scenes_in_index"], 4)
        self.assertEqual(summary["by_status"], {"ok": 3, "unstable": 1})
        self.assertEqual(summary["by_family"]["pile"], {"ok": 1, "unstable": 1})

    def test_the_reject_rate_per_family_is_recoverable(self) -> None:
        """The number that says whether piles are 5% unstable or 40% -- worth knowing, not guessing."""
        for i in range(10):
            status = "unstable" if i % 5 == 0 else "ok"
            self.writer.append_index(SceneRecord(f"p{i}", i, "pile", status))
        pile = self.writer.summary()["by_family"]["pile"]
        self.assertEqual(pile["unstable"] / (pile["ok"] + pile["unstable"]), 0.2)

    def test_scene_json_is_written_and_readable(self) -> None:
        path = self.writer.write_json("s0", "scene.json", {"objects": 3, "family": "sparse"})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["objects"], 3)

    def test_dataclasses_and_numpy_scalars_serialise(self) -> None:
        record = SceneRecord("s0", 0, "sparse", "ok", ("wrist",), {"wrist": "rendered"})
        path = self.writer.write_json("s0", "record.json", {"record": record, "n": np.int64(7)})
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["record"]["scene_id"], "s0")
        self.assertEqual(payload["record"]["view_outcomes"]["wrist"], "rendered")
        self.assertEqual(payload["n"], 7)

    def test_depth_survives_the_png_round_trip_in_millimetres(self) -> None:
        import cv2  # type: ignore[import-not-found]

        from datagen.render.writer import encode_depth_png

        depth = np.random.default_rng(0).uniform(200.0, 1500.0, (32, 32))
        path = self.writer.write_png("s0", "depth.png", encode_depth_png(depth))
        loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        self.assertEqual(loaded.dtype, np.uint16)
        np.testing.assert_allclose(decode_depth_png(loaded), np.round(depth), atol=0.5)

    def test_rgb_keeps_its_channel_order(self) -> None:
        """The classic silent corruption: a red dataset that is quietly blue."""
        import cv2  # type: ignore[import-not-found]

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[..., 0] = 255  # pure red in RGB
        path = self.writer.write_png("s0", "rgb.png", rgb)
        loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)  # cv2 reads BGR
        self.assertEqual(tuple(loaded[0, 0]), (0, 0, 255), "red must survive as red")

    def test_provenance_and_attribution_land_at_the_dataset_root(self) -> None:
        self.writer.write_provenance({"code_commit": "abc123"})
        self.writer.write_attribution("ATTRIBUTION — v1_test\n")
        self.assertIn("abc123", (self.root / "v1_test" / "provenance.json").read_text(encoding="utf-8"))
        self.assertIn("v1_test", (self.root / "v1_test" / "ATTRIBUTION").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
