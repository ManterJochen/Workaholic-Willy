"""Two defects in one bench tool: a number that could only be 0 % or 100 %, and the wrong camera.

⛔⛔ **THE NUMBER.** `acquire()` overwrites the depth inside every mask with one scalar
(`depth_mm = np.where(mask, grasp_depth_mm, depth_mm)`, guarded by `if vals.size:`), and this CLI
then counted zeros on that same array. A mask with even one real depth pixel reported 0 holes; a mask
that was entirely holes skipped the overwrite and reported 100 %. Nothing in between -- on the number
the module docstring calls "the real-hardware risk" and "a bench-tuning signal", and which an
operator reads at the first real camera to decide whether their depth is good enough.

⛔ **THE CAMERA.** This file selected its rig with `rgbd_backend == "realsense"` while
`build_real_components` (cells.py:118) selected with `source == "rgbd"`. Fifteen lines below the
selection, this module's own comment says the exerciser exists "to prove the SAME chain the cell runs
-- if the two acquire their frames differently, a green bench run stops being evidence about the
cell." A different SELECTION is a stronger disagreement than a different acquisition.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.robot.perception.__main__ import _find_rgbd_rig, _RawDepthTap


class _Frame:
    def __init__(self, color: np.ndarray, depth: np.ndarray) -> None:
        self.color = color
        self.depth = depth


class _Streamer:
    """Hands back one fixed frame and counts grabs, like the fakes in
    tests/test_realsense_perception_source.py."""

    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth
        self.grabs = 0
        self.released = 0

    def grab(self) -> _Frame:
        self.grabs += 1
        return _Frame(np.zeros((*self.depth.shape, 3), np.uint8), self.depth.copy())

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)

    def release(self) -> None:
        self.released += 1


def _source(streamer, *, warmup=0, mask=None):
    """The REAL adapter over fakes -- the point is what `acquire()` does to the depth."""
    from src.models.segmentation.types import SegmentationResult
    from src.robot.perception import RealSenseVisionPerceptionSource

    h, w = mask.shape
    det = SimpleNamespace(box=(0.0, 0.0, float(w), float(h)), label="cube", score=0.9)

    class _Backend:
        def perceive(self, image_bgr, prompt):  # noqa: ANN001, ANN202
            from src.models.perception_backend import PerceivedObject

            seg = SegmentationResult(
                label="cube", score=0.9, bbox_xyxy=(0, 0, w, h),
                mask=mask.astype(np.uint8), mask_area_px=int(mask.sum()),
                centroid_xy=(w / 2.0, h / 2.0), derived_bbox_xyxy=(0, 0, w, h),
                inference_time_s=0.0, timestamp_utc="2026-09-05T00:00:00Z",
            )
            return (PerceivedObject(detection=det, segmentation=seg),)

    return RealSenseVisionPerceptionSource(
        streamer=streamer, backend=_Backend(), prompt="cube", warmup_grabs=warmup,
    )


class TheTapSeesWhatTheAdapterConsumedTests(unittest.TestCase):
    def _scene(self):
        """An 8x8 depth with 15 holes, 8 of them inside a 16-pixel mask."""
        depth = np.full((8, 8), 400.0)
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True                      # 16 px
        depth[2:4, 2:6] = 0.0                      # 8 holes inside the mask
        depth[0, 0:7] = 0.0                        # 7 holes outside it
        return depth, mask

    def test_the_frame_the_cli_used_to_read_can_only_say_zero_or_one_hundred(self) -> None:
        """⛔ THE DEFECT, AS AN ASSERTION ABOUT THE OLD ARITHMETIC. Truth is 8 of 16; reading
        `frame.depth_map` gives 0 of 16, because the mask was overwritten with a constant."""
        depth, mask = self._scene()
        streamer = _Streamer(depth)
        frame = _source(streamer, mask=mask).acquire()

        rendered = np.asarray(frame.depth_map)
        seg_mask = np.asarray(frame.segmentations[0].mask).astype(bool)
        self.assertEqual(int((rendered[seg_mask] == 0).sum()), 0, "the old number, on real holes")
        self.assertEqual(int((depth[seg_mask] == 0).sum()), 8, "the truth")

    def test_the_tap_reports_the_truth_for_both_branches(self) -> None:
        depth, mask = self._scene()
        streamer = _Streamer(depth)
        tap = _RawDepthTap(streamer)
        frame = _source(tap, mask=mask).acquire()
        seg_mask = np.asarray(frame.segmentations[0].mask).astype(bool)
        self.assertEqual(int((tap.last_depth_mm[seg_mask] == 0).sum()), 8)
        self.assertEqual(int((tap.last_depth_mm == 0).sum()), 15, "the whole-image line too")

        # The all-holes branch: the adapter skips the overwrite, so the OLD reading was 100 % here.
        # The tap must agree with it rather than differ -- both ends of the two-valued flag.
        dead = np.full((8, 8), 400.0)
        dead[2:6, 2:6] = 0.0
        tap2 = _RawDepthTap(_Streamer(dead))
        _source(tap2, mask=mask).acquire()
        self.assertEqual(int((tap2.last_depth_mm[mask] == 0).sum()), 16)

    def test_the_tap_costs_no_extra_device_frame_and_holds_the_LAST_one(self) -> None:
        """⭐ WHY A TAP AND NOT A SECOND GRAB. A pre-grab reads a DIFFERENT frame from the one the
        masks were cut from, and reads it before the warmup grabs have settled auto-exposure."""
        depth, mask = self._scene()
        streamer = _Streamer(depth)
        tap = _RawDepthTap(streamer)
        _source(tap, warmup=5, mask=mask).acquire()
        self.assertEqual(streamer.grabs, 6, "5 warmup + 1 real, exactly as without the tap")

    def test_the_tap_delegates_everything_else(self) -> None:
        """The adapter and the calibration code duck-type the whole handle surface."""
        streamer = _Streamer(np.zeros((4, 4)))
        tap = _RawDepthTap(streamer)
        self.assertTrue((np.asarray(tap.get_intrinsics()) == np.eye(3)).all())
        tap.release()
        self.assertEqual(streamer.released, 1)

    def test_a_frame_with_no_depth_leaves_the_snapshot_alone(self) -> None:
        """An RGBDFrame may legally carry an empty depth array, and a StereoFrame has no `.depth` at
        all -- the tap must not raise on either."""
        class _NoDepth:
            def grab(self):  # noqa: ANN202
                return SimpleNamespace(color=np.zeros((2, 2, 3), np.uint8), left=None, right=None)

        tap = _RawDepthTap(_NoDepth())
        tap.grab()
        self.assertIsNone(tap.last_depth_mm)


class TheTwoPredicatesAgreeTests(unittest.TestCase):
    def _cfg(self, *rigs):
        return SimpleNamespace(cameras=SimpleNamespace(rigs=list(rigs)))

    @staticmethod
    def _cell_predicate(camera_cfg):
        """`build_real_components`' own filter, copied here so the two can be compared."""
        return [r for r in camera_cfg.cameras.rigs if getattr(r, "source", None) == "rgbd"]

    def test_a_rig_the_cell_would_pick_is_no_longer_invisible_to_the_exerciser(self) -> None:
        """⛔ THE DIVERGENCE. `rgbd_backend` defaults to "opencv" (cam_schema.py:172), so a rig with
        `source: rgbd` and no explicit backend was chosen as PRIMARY by the cell and reported as
        "found 0 realsense rigs" by the tool whose job is to prove the cell's chain."""
        rig = SimpleNamespace(rig_id="overhead", source="rgbd", rgbd_backend="opencv", enabled=True)
        cfg = self._cfg(rig)
        self.assertEqual([r.rig_id for r in self._cell_predicate(cfg)], ["overhead"])
        self.assertIs(_find_rgbd_rig(cfg, None), rig)

    def test_the_two_predicates_now_select_the_same_set_on_the_shipped_trees(self) -> None:
        """⭐ THE CONTROL THAT MAKES THE TEST ABOVE MORE THAN A SYNTHETIC CLAIM. Run over the real
        config, in both profiles that carry RGB-D rigs."""
        import os

        for profile in (None, "tiltcam"):
            with self.subTest(profile=profile or "(base)"):
                env = {"WILLY_PROFILE": profile} if profile else {}
                with mock.patch.dict(os.environ, env, clear=False):
                    if not profile:
                        os.environ.pop("WILLY_PROFILE", None)
                    from src.config.loader import load_config

                    cfg = load_config()
                cell = {r.rig_id for r in self._cell_predicate(cfg.camera)}
                tool = {r.rig_id for r in cfg.camera.cameras.rigs
                        if getattr(r, "source", None) == "rgbd"}
                self.assertEqual(cell, tool)
                self.assertTrue(cell, "a profile with no RGB-D rig would make this vacuous")

    def test_the_refusals_still_name_the_candidates(self) -> None:
        a = SimpleNamespace(rig_id="l", source="rgbd", rgbd_backend="realsense", enabled=True)
        b = SimpleNamespace(rig_id="r", source="rgbd", rgbd_backend="realsense", enabled=True)
        with self.assertRaises(SystemExit) as ambiguous:
            _find_rgbd_rig(self._cfg(a, b), None)
        self.assertIn("pass --rig", str(ambiguous.exception))
        self.assertIn("'l'", str(ambiguous.exception))

        with self.assertRaises(SystemExit) as missing:
            _find_rgbd_rig(self._cfg(a), "nope")
        self.assertIn("nope", str(missing.exception))

    def test_a_webcam_rig_is_still_not_an_rgbd_rig(self) -> None:
        """The widening must not become 'anything with a rig_id'."""
        cam = SimpleNamespace(rig_id="webcam_main", source="webcam_pair", enabled=True)
        with self.assertRaises(SystemExit):
            _find_rgbd_rig(self._cfg(cam), None)


class TheRenderedBlockTests(unittest.TestCase):
    def test_an_all_hole_mask_is_called_out_by_name(self) -> None:
        """⚠ THE CASE THE EXIT CODE IS STILL BLIND TO. A D435 on a specular scene returns lit RGB, so
        grounding succeeds, and dead depth -- and the adapter leaves the zeros in place, so the grasp
        depth is referenced to nothing. This CLI keeps exit 0 for it; the line exists so the operator
        is not the last to know."""
        from src.robot.perception import __main__ as cli

        depth = np.full((8, 8), 400.0)
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True
        depth[2:6, 2:6] = 0.0
        tap = _RawDepthTap(_Streamer(depth))
        frame = _source(tap, mask=mask).acquire()

        out = io.StringIO()
        with redirect_stdout(out):
            cli._print_frame_health(frame, tap, backend_note="")
        text = out.getvalue()
        self.assertIn("16/16 (100.0%)", text)
        self.assertIn("mask(s) have NO real depth", text)

    def test_an_empty_mask_is_a_segmenter_miss_not_a_camera_fault(self) -> None:
        """⛔ `mask_px == 0` divides by zero and, derived as `holes == mask_px`, reports a camera
        fault for what is a segmenter miss -- the opposite instruction to the operator. The original
        had an explicit `(empty)` branch and it is kept."""
        from src.robot.perception import __main__ as cli

        depth = np.full((4, 4), 400.0)
        tap = _RawDepthTap(_Streamer(depth))
        frame = _source(tap, mask=np.zeros((4, 4), bool)).acquire()

        out = io.StringIO()
        with redirect_stdout(out):
            cli._print_frame_health(frame, tap, backend_note="")
        self.assertIn("mask_px=0 (empty)", out.getvalue())
        self.assertNotIn("NO real depth", out.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
