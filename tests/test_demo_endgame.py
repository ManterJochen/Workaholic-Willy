"""C4 / demo-endgame — off-box tests for the pure-cv2 demo helpers (banner, title card, stitch).

The recording itself is on-box (Isaac); these pin the composition logic that runs without a simulator.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class DemoEndgameTests(unittest.TestCase):
    def test_banner_preserves_shape_and_draws(self) -> None:
        from src.willy_sim.run_dense_demo_endgame import _banner

        frame = np.zeros((300, 600, 3), dtype=np.uint8)
        out = _banner(frame, "VISION-LANGUAGE GRASP", "prompt -> grasp", "GRASP_NOW", (60, 220, 60))
        self.assertEqual(out.shape, frame.shape)
        self.assertTrue(out.any())  # the banner + text were drawn onto the black frame

    def test_banner_handles_empty_state(self) -> None:
        from src.willy_sim.run_dense_demo_endgame import _banner

        out = _banner(np.zeros((240, 320, 3), dtype=np.uint8), "T", "c", "", (200, 120, 255))
        self.assertEqual(out.shape, (240, 320, 3))

    def test_title_card_count_and_size(self) -> None:
        from src.willy_sim.run_dense_demo_endgame import _title_card

        cards = _title_card(["FAIL-CLOSED SAFETY", "blocker -> refuse"], (320, 240), (60, 200, 255), 5)
        self.assertEqual(len(cards), 5)
        self.assertEqual(cards[0].shape, (240, 320, 3))
        self.assertTrue(cards[0].any())

    def test_stitch_combines_segment_mp4s_with_title_cards(self) -> None:
        import cv2  # type: ignore[import-not-found]

        from src.willy_sim.run_dense_demo_endgame import stitch

        with tempfile.TemporaryDirectory() as td:
            for seg in ("vision", "safety"):
                w = cv2.VideoWriter(
                    str(Path(td) / f"seg_{seg}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                    18.0, (320, 240),
                )
                for _ in range(6):
                    w.write(np.full((240, 320, 3), 30, dtype=np.uint8))
                w.release()
            res = stitch(demo_dir=td, fps=18, order=("vision", "safety", "recovery", "autonomy"))
            self.assertEqual(set(res["segments"]), {"vision", "safety"})  # only the 2 present are used
            self.assertTrue(Path(res["out"]).exists())
            # 2 segments x 6 frames + 2 title cards (~27 frames each at 18fps*1.5) -> well over the raw 12
            self.assertGreater(res["frames"], 12)

    def test_stitch_raises_when_no_segments(self) -> None:
        from src.willy_sim.run_dense_demo_endgame import stitch

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                stitch(demo_dir=td)


if __name__ == "__main__":
    unittest.main()
