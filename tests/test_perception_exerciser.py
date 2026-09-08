"""Two defects in one bench tool: a number that could only be 0 % or 100 %, and the wrong camera.

⛔⛔ **THE NUMBER.** `acquire()` overwrites the depth inside every mask with one scalar
(`depth_mm = np.where(mask, grasp_depth_mm, depth_mm)`, guarded by `if vals.size:`), and this CLI
then counted zeros on that same array. A mask with even one real depth pixel reported 0 holes; a mask
that was entirely holes skipped the overwrite and reported 100 %. Nothing in between -- on the number
the module docstring calls "the real-hardware risk" and "a bench-tuning signal", and which an
operator reads at the first real camera to decide whether their depth is good enough.

⛔ **THE CAMERA, TWICE.** This file selected its rig with `rgbd_backend == "realsense"` while
`build_real_components` selected with `source == "rgbd"`. Fifteen lines below the selection, this
module's own comment says the exerciser exists "to prove the SAME chain the cell runs -- if the two
acquire their frames differently, a green bench run stops being evidence about the cell." A different
SELECTION is a stronger disagreement than a different acquisition.

⛔⛔ **AND THEN IT HAPPENED AGAIN, THE OTHER WAY ROUND.** The cell moved to
`camera.cameras.primary_rig_id` (cells.py:132) and this file went on scanning the rig list for the
single RGB-D one. MEASURED on the shipped base profile: the cell names `webcam_main`, a webcam pair,
and refuses it; the exerciser opened `realsense_d435`, which is `enabled: false`, and said nothing.
The first divergence made the tool useless, because a strict subset can only find less. The second
made it MISLEADING, because it reported a green bench run about a camera the cell never opens.
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

    def test_the_frame_now_reports_the_holes_it_has(self) -> None:
        """The defect this class was written for is gone at its source, and the assertion is
        inverted rather than deleted, because the inversion is the proof.

        The source overwrote the depth under every mask with one scalar, so a mask holding even one
        real depth pixel reported 0 holes and a mask holding none reported 100 %. Nothing in
        between, on the figure this CLI exists to show. Truth on this scene is 8 of 16, and
        `frame.depth_map` says 8 of 16.
        """
        depth, mask = self._scene()
        streamer = _Streamer(depth)
        frame = _source(streamer, mask=mask).acquire()

        rendered = np.asarray(frame.depth_map)
        seg_mask = np.asarray(frame.segmentations[0].mask).astype(bool)
        self.assertEqual(int((rendered[seg_mask] == 0).sum()), 8, "what the sensor measured")
        self.assertEqual(int((depth[seg_mask] == 0).sum()), 8, "the truth")

    def test_the_tap_is_now_a_second_copy_of_what_the_frame_carries(self) -> None:
        """Said out loud rather than left to be discovered. `_RawDepthTap` was built because
        `frame.depth_map` could not be trusted for a hole count. It can now, and the frame also
        publishes `surface_depth_map`, so the tap holds a third copy of one array. It still
        guarantees that the depth and the masks come from the same grab, which is why it stays.
        """
        depth, mask = self._scene()
        tap = _RawDepthTap(_Streamer(depth))
        frame = _source(tap, mask=mask).acquire()

        np.testing.assert_array_equal(tap.last_depth_mm, frame.depth_map)
        np.testing.assert_array_equal(tap.last_depth_mm, frame.surface_depth_map)

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


class TheTwoToolsNameTheSameCameraTests(unittest.TestCase):
    """⛔ TWICE NOW THEY HAVE NOT, and the second time was worse than the first.

    First the exerciser filtered on ``rgbd_backend == "realsense"`` while the cell filtered on
    ``source == "rgbd"``, so the exerciser was blind to rigs the cell would pick. That closed, and
    then the cell moved to ``camera.cameras.primary_rig_id`` while this file still scanned the rig
    list. Measured on the shipped base profile at that point: the cell named ``webcam_main`` and
    refused it, the exerciser opened ``realsense_d435``. A strict subset became two different
    cameras, which is the worse failure of the two: the first made the tool useless, the second made
    it misleading.
    """

    def _cfg(self, primary, *rigs):  # noqa: ANN001, ANN202
        return SimpleNamespace(
            cameras=SimpleNamespace(primary_rig_id=primary, rigs=list(rigs)))

    def _rig(self, rig_id, source="rgbd", enabled=True, backend="realsense"):  # noqa: ANN001, ANN202
        return SimpleNamespace(
            rig_id=rig_id, source=source, enabled=enabled, rgbd_backend=backend)

    def test_the_default_is_the_key_the_cell_reads(self) -> None:
        second = self._rig("other")
        cfg = self._cfg("overhead", self._rig("overhead"), second)
        self.assertEqual(_find_rgbd_rig(cfg, None).rig_id, "overhead")
        # And it is the key, not "the first" or "the only": with two candidates the old code demanded
        # --rig, and the key answers the question instead.
        self.assertEqual(
            _find_rgbd_rig(self._cfg("other", self._rig("overhead"), second), None).rig_id, "other")

    def test_rig_overrides_the_key(self) -> None:
        cfg = self._cfg("overhead", self._rig("overhead"), self._rig("wrist"))
        self.assertEqual(_find_rgbd_rig(cfg, "wrist").rig_id, "wrist")

    def test_a_non_rgbd_rig_is_refused_in_the_cells_own_terms(self) -> None:
        """⭐ THE OPERATOR CARRIES THE SENTENCE BETWEEN TWO TOOLS. ``build_real_components`` refuses a
        primary that is not RGB-D and points at this exerciser; if the exerciser then opened it
        happily, the advice would contradict the refusal that produced it."""
        cfg = self._cfg(
            "webcam_main",
            self._rig("webcam_main", source="webcam_pair"),
            self._rig("realsense_d435"),
        )
        with self.assertRaises(SystemExit) as caught:
            _find_rgbd_rig(cfg, None)
        message = str(caught.exception)
        self.assertIn("webcam_pair", message)
        self.assertIn("synthesised from depth", message)
        # And it names what would work, rather than the flag that would name it.
        self.assertIn("realsense_d435", message)

    def test_a_config_with_no_rgbd_rig_at_all_says_so(self) -> None:
        """A different problem from the wrong rig, and the empty list is what distinguishes them."""
        cfg = self._cfg("webcam_main", self._rig("webcam_main", source="webcam_pair"))
        with self.assertRaises(SystemExit) as caught:
            _find_rgbd_rig(cfg, None)
        self.assertIn("no RGB-D rig at all", str(caught.exception))

    def test_a_disabled_rig_is_opened_with_a_note_rather_than_refused(self) -> None:
        """⚠ NOT THE SAME AS THE CASE ABOVE, AND THE DIFFERENCE IS NOT SEVERITY. ``enabled: false`` is
        a config state the operator may be about to change, and this is the tool they check with
        first. A missing depth channel is a fact about the hardware that no config edit repairs."""
        cfg = self._cfg("oblique_L", self._rig("oblique_L", enabled=False))
        out = io.StringIO()
        with redirect_stdout(out):
            rig = _find_rgbd_rig(cfg, None)
        self.assertEqual(rig.rig_id, "oblique_L")
        self.assertIn("enabled: false", out.getvalue())
        self.assertIn("A cell will refuse it", out.getvalue())

    def test_an_unnamed_rig_and_an_unset_key_both_say_what_to_do(self) -> None:
        with self.assertRaises(SystemExit) as missing:
            _find_rgbd_rig(self._cfg("overhead", self._rig("overhead")), "nope")
        self.assertIn("nope", str(missing.exception))
        with self.assertRaises(SystemExit) as unset:
            _find_rgbd_rig(self._cfg(None, self._rig("overhead")), None)
        self.assertIn("primary_rig_id", str(unset.exception))

    def test_over_the_shipped_profiles_the_two_tools_agree(self) -> None:
        """⭐ THE CONTROL, AND THE ONE THAT WOULD HAVE CAUGHT BOTH DIVERGENCES. Everything above is
        synthetic; this reads the real config. For every profile the exerciser must open the rig the
        cell names, or refuse it for the reason the cell refuses it. It must never open a different
        one."""
        import os

        checked = 0
        for profile in (None, "tiltcam", "ur3e,tiltcam", "sim", "console_dummy"):
            with self.subTest(profile=profile or "(base)"):
                env = dict(os.environ)
                env.pop("WILLY_PROFILE", None)
                if profile:
                    env["WILLY_PROFILE"] = profile
                with mock.patch.dict(os.environ, env, clear=True):
                    from src.config.loader import load_config

                    cfg = load_config()
                named = cfg.camera.cameras.primary_rig_id
                out = io.StringIO()
                try:
                    with redirect_stdout(out):
                        chosen = _find_rgbd_rig(cfg.camera, None).rig_id
                except SystemExit:
                    chosen = named  # a refusal is about the rig the cell named, which is agreement
                self.assertEqual(chosen, named)
                checked += 1
        self.assertGreaterEqual(checked, 3, "a test over no profile passes loudest")


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
        self.assertIn("mask(s) have no real depth at all", text)

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
        self.assertNotIn("no real depth at all", out.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
