"""A rig recording for research keeps its infrared images: ``realsense.record_for_research``, one switch, off by default.

The owner, 2026-10-09 ("alles in einem Schalter"). The depth review of the owner's cell found the D415 smearing every
far edge into a ramp of depths that SFE's footprint follows, and could only reconstruct the masks from the debug
pictures: the views files held no mask, no box, no infrared image and no depth before the filters. Monday's recording
keeps all of it, behind one key on the rig: both infrared images (Y8, at the depth mode), the raw depth, the colour
frame's exposure metadata and every segmentation's mask, box, label, score, SAM2's predicted IoUs and target per look,
and the camera once.

What is pinned, against a stand-in SDK (``tests/_realsense_fakes.py``) and, where it can run with no camera, the real
librealsense:

  * off, the camera is asked for colour and depth alone, the frames carry nothing more, and the views file is today's,
    byte for byte;
  * on, both infrared streams are asked at the depth mode, every frame carries them and the depth as the sensor sent it,
    copied before any filter runs, and the camera's facts, read once;
  * a camera that cannot stream infrared refuses to open with a sentence;
  * the views file is format 2, readable by every reader of format 1, and each look's target is the part the pick went
    for, as the pick loop associated it across the looks;
  * SAM2 keeps its three predicted IoUs on each segmentation.
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.camera.setup.image_taking.frames import CameraFacts, ResearchCapture, RGBDFrame
from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer, _extrinsics_mm
from src.robot.execution.pick_run import _targets_of, keep_pick_views
from src.robot.execution.record_views import (
    COLOR_METADATA,
    RECORD_VIEWS_FORMAT,
    RECORD_VIEWS_FORMAT_WITHOUT_RESEARCH,
    record_views,
)
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.perception import RealSenseVisionPerceptionSource
from tests._realsense_fakes import Frameset, Pipeline, Profile, rig, sdk

try:  # pyrealsense2 is pinned in requirements.txt, and the wheel is absent on some hosts
    import pyrealsense2 as rs

    _SKIP = ""
except Exception as exc:  # pragma: no cover - exercised only where the wheel is missing
    rs = None  # type: ignore[assignment]
    _SKIP = f"pyrealsense2 not installed ({exc.__class__.__name__})"

_LOGGER = "src.camera.setup.image_taking.rgbd"
_K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]])


def _opened(pipeline: Pipeline, **rig_keys: Any) -> RealSenseRGBDStreamer:
    streamer = RealSenseRGBDStreamer(rig(**rig_keys), rs_module=sdk(pipeline))
    streamer.open()
    return streamer


# ---------------------------------------------------------------------------------------------------------------------
# The streams and the frames
# ---------------------------------------------------------------------------------------------------------------------


class TheStreamsAndTheFramesTests(unittest.TestCase):

    def setUp(self) -> None:
        Frameset.infrared_reads = 0

    def test_off_the_camera_is_asked_for_colour_and_depth_alone_and_the_frame_carries_nothing_more(self) -> None:
        pipeline = Pipeline()
        streamer = _opened(pipeline)
        self.addCleanup(streamer.release)
        assert pipeline.config is not None
        self.assertEqual([("color", 64, 48, "bgr8", 30), ("depth", 32, 24, "z16", 30)], pipeline.config.streams)
        frame = streamer.grab()
        self.assertIsNone(frame.research)
        self.assertEqual(0, Frameset.infrared_reads, "a grab read an infrared image with the recording off")
        self.assertEqual(1007, int(frame.depth[0, 0]), "the spatial filter and the alignment no longer ran")

    def test_on_both_infrared_images_are_asked_at_the_depth_mode_in_y8(self) -> None:
        pipeline = Pipeline()
        streamer = _opened(pipeline, research=True)
        self.addCleanup(streamer.release)
        assert pipeline.config is not None
        self.assertEqual([("color", 64, 48, "bgr8", 30), ("depth", 32, 24, "z16", 30),
                          ("infrared", 1, 32, 24, "y8", 30), ("infrared", 2, 32, 24, "y8", 30)],
                         pipeline.config.streams)

    def test_on_every_frame_carries_the_infrared_images_and_the_depth_before_the_filters_copied(self) -> None:
        pipeline = Pipeline()
        streamer = _opened(pipeline, research=True)
        self.addCleanup(streamer.release)
        frame = streamer.grab()
        research = frame.research
        assert isinstance(research, ResearchCapture)
        assert research.ir_left is not None and research.ir_right is not None and research.raw_depth is not None
        self.assertEqual(((24, 32), np.uint8), (research.ir_left.shape, research.ir_left.dtype))
        self.assertTrue(np.all(research.ir_left == 40) and np.all(research.ir_right == 60))
        # The depth as the sensor sent it: 1000 raw units at the depth mode, where the frame's depth went through the
        # spatial filter (+7 in the double) and the alignment to the colour grid.
        self.assertEqual(((24, 32), np.uint16), (research.raw_depth.shape, research.raw_depth.dtype))
        self.assertTrue(np.all(research.raw_depth == 1000))
        self.assertEqual((48, 64), frame.depth.shape)
        self.assertTrue(np.all(frame.depth == 1007))
        assert pipeline.last is not None
        self.assertFalse(np.shares_memory(research.raw_depth, pipeline.last.depth.data))
        self.assertFalse(np.shares_memory(research.ir_left, pipeline.last.infrared[1].data))
        self.assertEqual({"actual_exposure": 156.0, "gain_level": 64.0, "white_balance": 4600.0, "auto_exposure": 1.0},
                         dict(research.color_metadata))

    def test_on_the_camera_facts_are_read_once_with_its_lenses_extrinsics_and_settings(self) -> None:
        pipeline = Pipeline()
        streamer = _opened(pipeline, research=True)
        self.addCleanup(streamer.release)
        first, second = streamer.grab().research, streamer.grab().research
        assert first is not None and second is not None
        self.assertIs(first.camera, second.camera, "the facts are read again for every frame")
        facts = first.camera
        self.assertEqual({"color", "depth", "ir_left", "ir_right"}, set(facts.intrinsics))
        np.testing.assert_allclose([[32.0, 0.0, 16.0], [0.0, 32.0, 12.0], [0.0, 0.0, 1.0]], facts.intrinsics["depth"])
        np.testing.assert_allclose([0.01, -0.02, 0.0, 0.0, 0.003], facts.distortion["ir_left"])
        # The double's cameras sit along x: colour at -15 mm, the left imager (the depth's) at 0, the right at 55.
        np.testing.assert_allclose([15.0, 0.0, 0.0], facts.extrinsics_mm["depth_to_color"][:3, 3])
        np.testing.assert_allclose([15.0, 0.0, 0.0], facts.extrinsics_mm["ir_left_to_color"][:3, 3])
        np.testing.assert_allclose([-55.0, 0.0, 0.0], facts.extrinsics_mm["ir_left_to_ir_right"][:3, 3])
        self.assertAlmostEqual(0.001, facts.depth_units_m)
        said = json.loads(json.dumps(facts.said))
        self.assertEqual({"name": "Intel RealSense D415", "serial": "SN-415", "firmware": "5.16.0.1", "usb": "3.2"},
                         said["device"])
        self.assertEqual({"width": 32, "height": 24, "distortion_model": "brown_conrady", "format": "y8"},
                         said["streams"]["ir_right"])
        self.assertEqual(156.0, said["color_sensor"]["held"]["exposure"])
        self.assertEqual(4600.0, said["color_sensor"]["options"]["white_balance"])
        self.assertEqual({"auto_exposure": None, "exposure": None, "gain": None, "auto_white_balance": None,
                          "white_balance": None}, said["color_sensor"]["asked"])
        self.assertEqual(150.0, said["depth_sensor"]["laser_power"])
        self.assertEqual(4, len(said["filters"]), "spatial and temporal between the two disparity transforms")
        self.assertEqual({"filter_smooth_alpha": 0.5}, said["filters"][1]["options"])

    def test_the_open_says_it_streams_both_infrared_images(self) -> None:
        streamer = RealSenseRGBDStreamer(rig(research=True), rs_module=sdk(Pipeline()))
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            streamer.open()
        self.addCleanup(streamer.release)
        opened = next(r.getMessage() for r in logs.records if "opened" in r.getMessage())
        self.assertIn("colour 64x48 + depth 32x24 + infrared 1 and 2 32x24 Y8 (record_for_research) @ 30 fps", opened)

    def test_a_camera_that_cannot_stream_infrared_refuses_with_a_sentence(self) -> None:
        pipeline = Pipeline(infrared=False)
        streamer = RealSenseRGBDStreamer(rig(research=True), rs_module=sdk(pipeline))
        with self.assertRaises(RuntimeError) as caught:
            streamer.open()
        text = str(caught.exception)
        for part in ("could not start colour 64x48, depth 32x24 and both infrared images 32x24 Y8 at 30 fps",
                     "Couldn't resolve requests", "realsense.record_for_research asks for both infrared images",
                     "switch record_for_research off"):
            with self.subTest(part=part):
                self.assertIn(part, text)
        self.assertFalse(streamer.is_opened())

    def test_a_camera_whose_profile_holds_no_infrared_after_all_refuses_and_stops(self) -> None:
        class _Forgets(Pipeline):
            def start(self, config: Any) -> Profile:
                profile = super().start(config)
                profile.streams = {key: stream for key, stream in profile.streams.items() if key[0] != "infrared"}
                return profile

        pipeline = _Forgets()
        streamer = RealSenseRGBDStreamer(rig(research=True), rs_module=sdk(pipeline))
        with self.assertRaises(RuntimeError) as caught:
            streamer.open()
        self.assertIn("streams no left infrared image (infrared 1) at this mode", str(caught.exception))
        self.assertTrue(pipeline.stopped)
        self.assertFalse(streamer.is_opened())

    def test_a_frameset_without_an_infrared_image_keeps_the_frame_and_is_said_once(self) -> None:
        streamer = _opened(Pipeline(drops=(2,)), research=True)
        self.addCleanup(streamer.release)
        with self.assertLogs(_LOGGER, level="WARNING") as said:
            frames = [streamer.grab() for _ in range(3)]
        for frame in frames:
            assert frame.research is not None
            self.assertIsNone(frame.research.ir_right)
            self.assertIsNotNone(frame.research.ir_left)
        self.assertEqual(1, len(said.records), said.output)
        self.assertIn("without ir_right", said.output[0])


# ---------------------------------------------------------------------------------------------------------------------
# The frame the pick perceives
# ---------------------------------------------------------------------------------------------------------------------


class _Streamer:
    def __init__(self, frame: Any) -> None:
        self.frame = frame

    def grab(self) -> Any:
        return self.frame

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


class _Nothing:
    def detect_all(self, bgr: np.ndarray, prompt: str) -> list:
        return []

    def segment_detection(self, bgr: np.ndarray, det: Any) -> Any:  # pragma: no cover - nothing is detected
        raise AssertionError("nothing was detected")


def _capture(**arrays: Any) -> ResearchCapture:
    facts = CameraFacts(said={"device": {"serial": "SN-415"}}, depth_units_m=0.001,
                        intrinsics={"color": _K, "depth": _K, "ir_left": _K}, distortion={"color": np.zeros(5)},
                        extrinsics_mm={"depth_to_color": np.eye(4), "ir_left_to_color": np.eye(4)})
    return ResearchCapture(camera=facts, **arrays)


class ThePerceptionFrameTests(unittest.TestCase):

    def _acquired(self, frame: Any) -> PerceptionFrame:
        source = RealSenseVisionPerceptionSource(streamer=_Streamer(frame), detector=_Nothing(), segmenter=_Nothing(),
                                                 prompt="x", warmup_grabs=0)
        return source.acquire()

    def test_the_frame_the_pick_perceives_carries_what_the_camera_recorded(self) -> None:
        capture = _capture(raw_depth=np.full((48, 64), 900, np.uint16))
        frame = self._acquired(RGBDFrame(color=np.zeros((48, 64, 3), np.uint8),
                                         depth=np.full((48, 64), 900, np.uint16), research=capture))
        self.assertIs(capture, frame.research)

    def test_a_frame_of_any_other_camera_carries_none(self) -> None:
        plain = RGBDFrame(color=np.zeros((48, 64, 3), np.uint8), depth=np.full((48, 64), 900, np.uint16))
        double = mock.MagicMock(color=np.zeros((48, 64, 3), np.uint8), depth=np.full((48, 64), 900, np.uint16),
                                captured_at_s=None)
        for frame in (plain, double):
            with self.subTest(frame=type(frame).__name__):
                self.assertIsNone(self._acquired(frame).research)


# ---------------------------------------------------------------------------------------------------------------------
# The views file
# ---------------------------------------------------------------------------------------------------------------------


def _todays_views_file(looks: list[Any], cloud: Any, path: Path) -> None:
    """The views writer of format 1 as it stood before format 2 (2026-10-09), word for word, for the byte comparison."""

    def matrix(value: Any) -> np.ndarray:
        if value is None:
            return np.full((4, 4), np.nan)
        to_matrix = getattr(value, "to_matrix", None)
        held = np.asarray(to_matrix() if callable(to_matrix) else value, dtype=np.float64)
        return held if held.shape == (4, 4) else np.full((4, 4), np.nan)

    arrays: dict[str, np.ndarray] = {"format": np.asarray(1, dtype=np.int64)}
    labels: list[str] = []
    names: list[str] = []
    for index, look in enumerate(looks):
        frame = getattr(look, "frame", look)
        labels.append(str(getattr(look, "label", "") or ""))
        names.append(str(getattr(look, "name", "") or ""))
        rgb = getattr(frame, "rgb", None)
        if rgb is not None:
            arrays[f"rgb_{index}"] = np.asarray(rgb, dtype=np.uint8)
        arrays[f"depth_{index}"] = np.asarray(frame.depth_map, dtype=np.float32)
        surface = getattr(frame, "surface_depth_map", None)
        if surface is not None:
            arrays[f"surface_depth_{index}"] = np.asarray(surface, dtype=np.float32)
        arrays[f"intrinsics_{index}"] = np.asarray(frame.intrinsics, dtype=np.float64).reshape(3, 3)
        arrays[f"tool_to_base_{index}"] = matrix(getattr(frame, "tool_pose", None))
        arrays[f"camera_to_base_{index}"] = matrix(getattr(look, "camera_to_base", None))
    arrays["labels"] = np.asarray(labels, dtype=np.str_)
    arrays["views"] = np.asarray(names, dtype=np.str_)
    held = np.zeros((0, 3)) if cloud is None else np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    arrays["target_cloud_base_mm"] = held.astype(np.float32)
    np.savez_compressed(path, **arrays)


def _segmentation(mask: np.ndarray, box: tuple[int, int, int, int], label: str, score: float,
                  ious: tuple[float, ...] | None) -> Any:
    return SimpleNamespace(mask=mask.astype(np.uint8), bbox_xyxy=box, label=label, score=score,
                           metadata={} if ious is None else {"device": "cpu", "predicted_iou": ious})


def _looks(*, research: bool) -> tuple[list[Any], dict[str, Any]]:
    """Two looks of one pick: the first with three segmentations, the second with one, colour on the first only."""
    height, width = 48, 64
    rng = np.random.default_rng(3)
    masks = [np.zeros((height, width), bool) for _ in range(4)]
    masks[0][5:20, 4:30] = True
    masks[1][22:40, 30:61] = True
    masks[2][1:3, 1:3] = True
    masks[3][10:30, 10:20] = True
    segs_0 = (_segmentation(masks[0], (4, 5, 30, 20), "grey cube", 0.91, (0.81, 0.93, 0.97)),
              _segmentation(masks[1], (30, 22, 61, 40), "grey cube", 0.88, (0.70, 0.92, 0.96)),
              _segmentation(masks[2], (1, 1, 3, 3), "yellow bin", 0.42, None))
    segs_1 = (_segmentation(masks[3], (10, 10, 20, 30), "grey cube", 0.77, (0.9, 0.91, 0.99)),)
    capture = _capture(ir_left=rng.integers(0, 255, (24, 32), dtype=np.uint8),
                       ir_right=rng.integers(0, 255, (24, 32), dtype=np.uint8),
                       raw_depth=rng.integers(0, 3000, (24, 32), dtype=np.uint16),
                       color_metadata={"actual_exposure": 156.0, "gain_level": 64.0, "auto_exposure": 1.0})
    tool = np.eye(4)
    tool[:3, 3] = (100.0, -700.0, 400.0)
    frames = [
        PerceptionFrame(depth_map=np.full((height, width), 650.0), intrinsics=_K, segmentations=segs_0,
                        rgb=rng.integers(0, 255, (height, width, 3), dtype=np.uint8),
                        surface_depth_map=np.full((height, width), 651.0), tool_pose=None,
                        research=capture if research else None),
        PerceptionFrame(depth_map=np.full((height, width), 700.0), intrinsics=_K, segmentations=segs_1),
    ]
    looks = [SimpleNamespace(label="home", name="wrist@home", frame=frames[0], camera_to_base=tool),
             SimpleNamespace(label="look 1", name="wrist@look 1", frame=frames[1], camera_to_base=None)]
    return looks, {"capture": capture, "masks": masks}


class TheViewsFileTests(unittest.TestCase):

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.cloud = np.arange(30, dtype=np.float64).reshape(10, 3)

    def test_with_no_camera_recording_for_research_the_file_is_todays_byte_for_byte(self) -> None:
        looks, _ = _looks(research=False)
        written = record_views(looks, target_cloud_base_mm=self.cloud, directory=self.folder, name="pick",
                               targets=(1, 0))
        assert written is not None
        today = self.folder / "today.npz"
        _todays_views_file(looks, self.cloud, today)
        self.assertEqual(today.read_bytes(), written.read_bytes())
        with np.load(written, allow_pickle=False) as views:
            self.assertEqual(RECORD_VIEWS_FORMAT_WITHOUT_RESEARCH, int(views["format"]))
            self.assertEqual(1, int(views["format"]))

    def test_with_a_look_recorded_for_research_the_file_is_format_2_and_holds_every_array_of_it(self) -> None:
        looks, made = _looks(research=True)
        written = record_views(looks, target_cloud_base_mm=self.cloud, directory=self.folder, name="pick",
                               targets=(1, None))
        assert written is not None
        capture = made["capture"]
        with np.load(written, allow_pickle=False) as views:
            self.assertEqual(2, RECORD_VIEWS_FORMAT)
            self.assertEqual(RECORD_VIEWS_FORMAT, int(views["format"]))
            np.testing.assert_array_equal(capture.ir_left, views["ir_left_0"])
            np.testing.assert_array_equal(capture.ir_right, views["ir_right_0"])
            np.testing.assert_array_equal(capture.raw_depth, views["raw_depth_0"])
            self.assertEqual((np.uint8, np.uint8, np.uint16),
                             (views["ir_left_0"].dtype, views["ir_right_0"].dtype, views["raw_depth_0"].dtype))
            self.assertNotIn("ir_left_1", views.files, "a look its camera recorded nothing more for got an image")
            self.assertEqual(("actual_exposure", "gain_level", "white_balance", "auto_exposure"), COLOR_METADATA)
            np.testing.assert_array_equal([156.0, 64.0, np.nan, 1.0], views["color_metadata_0"])
            unpacked = np.unpackbits(views["seg_masks_0"], axis=1, count=48 * 64).reshape(-1, 48, 64).astype(bool)
            np.testing.assert_array_equal(np.stack(made["masks"][:3]), unpacked)
            unpacked = np.unpackbits(views["seg_masks_1"], axis=1, count=48 * 64).reshape(-1, 48, 64).astype(bool)
            np.testing.assert_array_equal(made["masks"][3][None], unpacked)
            np.testing.assert_array_equal([[4, 5, 30, 20], [30, 22, 61, 40], [1, 1, 3, 3]], views["seg_boxes_0"])
            self.assertEqual(np.int32, views["seg_boxes_0"].dtype)
            self.assertEqual(["grey cube", "grey cube", "yellow bin"], views["seg_labels_0"].tolist())
            np.testing.assert_allclose([0.91, 0.88, 0.42], views["seg_scores_0"], rtol=1e-6)
            np.testing.assert_allclose([[0.81, 0.93, 0.97], [0.70, 0.92, 0.96], [np.nan] * 3],
                                       views["seg_predicted_iou_0"], rtol=1e-6)
            self.assertEqual((1, -1), (int(views["seg_target_0"]), int(views["seg_target_1"])))
            self.assertEqual({"device": {"serial": "SN-415"}}, json.loads(str(views["camera"])))
            self.assertAlmostEqual(0.001, float(views["camera_depth_units_m"]))
            np.testing.assert_array_equal(_K, views["camera_depth_intrinsics"])
            np.testing.assert_array_equal(np.zeros(5), views["camera_color_distortion"])
            np.testing.assert_array_equal(np.eye(4), views["camera_depth_to_color"])
            self.assertNotIn("camera_depth_distortion", views.files, "a lens the camera did not report was invented")
            # The infrared lenses and extrinsics go with each look; ones the camera did not report read NaN.
            np.testing.assert_array_equal(_K, views["ir_intrinsics_0"][0])
            self.assertEqual(((2, 3, 3), (2, 5), (2, 4, 4)), (views["ir_intrinsics_0"].shape,
                                                              views["ir_distortion_0"].shape,
                                                              views["ir_extrinsics_0"].shape))
            self.assertTrue(np.all(np.isnan(views["ir_intrinsics_0"][1])))
            np.testing.assert_array_equal(np.eye(4), views["ir_extrinsics_0"][0])
            self.assertTrue(np.all(np.isnan(views["ir_extrinsics_0"][1])))
            self.assertNotIn("ir_intrinsics_1", views.files)

    def test_a_reader_of_format_1_reads_format_2_as_it_read_format_1(self) -> None:
        plain = record_views(_looks(research=False)[0], target_cloud_base_mm=self.cloud, directory=self.folder,
                             name="plain")
        researched = record_views(_looks(research=True)[0], target_cloud_base_mm=self.cloud, directory=self.folder,
                                  name="research", targets=(0, 0))
        assert plain is not None and researched is not None
        with np.load(plain, allow_pickle=False) as old, np.load(researched, allow_pickle=False) as new:
            for key in old.files:
                if key == "format":
                    continue
                with self.subTest(key=key):
                    np.testing.assert_array_equal(old[key], new[key])
            added = set(new.files) - set(old.files)
            prefixes = ("rgb_", "depth_", "surface_depth_", "intrinsics_", "tool_to_base_", "camera_to_base_",
                        "labels", "views", "target_cloud_base_mm", "format")
            self.assertEqual([], sorted(key for key in added if key.startswith(prefixes)))
            # How the readers of format 1 count the looks (the D415 probe's --from-npz, the image reviews).
            self.assertEqual(sum(1 for key in old.files if key.startswith("rgb_")),
                             sum(1 for key in new.files if key.startswith("rgb_")))
            self.assertEqual(len(old["labels"]), len(new["labels"]))
        with zipfile.ZipFile(researched) as archive:
            self.assertTrue(all(info.filename.endswith(".npy") for info in archive.infolist()))

    def test_a_mask_of_another_size_than_its_frame_is_refused(self) -> None:
        looks, _ = _looks(research=True)
        looks[1].frame = PerceptionFrame(depth_map=np.full((48, 64), 700.0), intrinsics=_K,
                                         segmentations=(_segmentation(np.ones((10, 10), bool), (0, 0, 9, 9),
                                                                      "grey cube", 0.5, None),))
        with self.assertRaises(ValueError) as caught:
            record_views(looks, directory=self.folder)
        self.assertIn("(10, 10)", str(caught.exception))


# ---------------------------------------------------------------------------------------------------------------------
# Which segmentation of each look is the target
# ---------------------------------------------------------------------------------------------------------------------


class TheTargetOfEachLookTests(unittest.TestCase):

    def test_the_judged_look_names_its_own_and_every_other_look_the_one_it_associated(self) -> None:
        first = SimpleNamespace(name="wrist@home", blob_objects=(3, 5))
        judged_look = SimpleNamespace(name="wrist@look 1", blob_objects=(0, 2))
        unseen = SimpleNamespace(name="wrist@look 2", blob_objects=(1,))
        judged = SimpleNamespace(look=judged_look, target_index=2, members={"wrist@home": 0, "wrist@look 1": 1})
        self.assertEqual((3, 2, None), _targets_of((first, judged_look, unseen), judged))

    def test_a_pick_that_judged_nothing_has_no_target_in_any_look(self) -> None:
        views = (SimpleNamespace(name="wrist@home", blob_objects=(0,)),)
        self.assertEqual((None,), _targets_of(views, None))
        self.assertEqual((None,), _targets_of(views, mock.MagicMock()))

    def test_a_campaign_keeps_each_looks_target_in_a_file_recorded_for_research(self) -> None:
        looks, _ = _looks(research=True)
        looks[0].blob_objects, looks[1].blob_objects = (0, 1), (0,)
        judged = SimpleNamespace(look=looks[0], target_index=1, members={"wrist@home": 1, "wrist@look 1": 0},
                                 target_cloud_base_mm=None, result=None)
        service = SimpleNamespace(looked_around=SimpleNamespace(views=tuple(looks), judged=judged), ranked_looked=None)
        report = SimpleNamespace(looks=("home", "look 1"), telemetry={"attempt_id": "pick-0001"})
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch("src.robot.execution.record_views.RECORD_VIEWS_DIR", folder), \
                mock.patch("src.robot.execution.pick_run._keep_debug_images", return_value=[]):
            written = keep_pick_views(service, report, name="run000")
            self.assertTrue(Path(written).name.startswith("pick-0001"), written)
            with np.load(written, allow_pickle=False) as views:
                self.assertEqual(2, int(views["format"]))
                self.assertEqual((1, 0), (int(views["seg_target_0"]), int(views["seg_target_1"])))


# ---------------------------------------------------------------------------------------------------------------------
# SAM2's own rating of its masks
# ---------------------------------------------------------------------------------------------------------------------


class Sam2KeepsItsPredictedIousTests(unittest.TestCase):

    def test_each_segmentation_carries_the_three_ious_sam2_predicted_for_its_masks(self) -> None:
        import torch

        from src.models.segmentation.realtime.segmenter import Sam2Segmenter

        height, width = 48, 64
        masks = torch.full((1, 3, height, width), -1.0)
        masks[0, 0, 10:30, 10:40] = 1.0     # the first mask, the one kept
        masks[0, 2, 0:40, 0:60] = 1.0
        processor = mock.MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 8, 8),
                                                 "original_sizes": torch.tensor([[height, width]])})
        processor.image_processor.post_process_masks.return_value = [masks]
        model = mock.MagicMock(return_value=SimpleNamespace(pred_masks=torch.zeros(1, 1, 3, 8, 8),
                                                            iou_scores=torch.tensor([[[0.81, 0.93, 0.97]]])))
        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        segmenter.device = torch.device("cpu")
        segmenter.optim = None
        segmenter._model_dtype = None
        segmenter.save_debug = False
        segmenter.keep_largest_component = True
        segmenter.morph_kernel_size = 5
        segmenter.logger = logging.getLogger("tests.sam2")
        segmenter.processor, segmenter.model = processor, model

        result = segmenter.segment(np.zeros((height, width, 3), np.uint8), (10, 10, 40, 30), "grey cube", 0.9)
        np.testing.assert_allclose([0.81, 0.93, 0.97], result.metadata["predicted_iou"], rtol=1e-6)
        # The mask kept is still the first of the three (its corners rounded by the open with a 5 px ellipse), not
        # the best rated: the IoUs are kept for whoever records them, and nothing chooses by them.
        self.assertTrue(result.mask[15:25, 15:35].all())
        self.assertFalse(result.mask[32:40, 42:60].any(), "the kept mask is no longer the first of the three")

    def test_an_output_without_ious_keeps_none(self) -> None:
        from src.models.segmentation.realtime.segmenter import Sam2Segmenter

        self.assertEqual((), Sam2Segmenter._predicted_ious(SimpleNamespace(pred_masks=None)))


# ---------------------------------------------------------------------------------------------------------------------
# The real librealsense, no camera attached
# ---------------------------------------------------------------------------------------------------------------------


@unittest.skipIf(rs is None, _SKIP)
class TheRealSdkTests(unittest.TestCase):
    """What the double answers is what librealsense names: the stand-in's names, held against the real module."""

    def test_the_names_the_colour_block_and_the_research_recording_reach_exist(self) -> None:
        for enum_name, members in (
            ("option", ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance")),
            ("stream", ("infrared",)),
            ("format", ("y8",)),
            ("frame_metadata_value", COLOR_METADATA),
            ("distortion", ("brown_conrady",)),
        ):
            enum = getattr(rs, enum_name)
            for member in members:
                with self.subTest(enum=enum_name, member=member):
                    self.assertTrue(hasattr(enum, member), f"rs.{enum_name}.{member} is gone")
        for owner, method in (("device", "first_color_sensor"), ("sensor", "is_depth_sensor"),
                              ("options", "get_supported_options"), ("options", "get_option_range"),
                              ("composite_frame", "get_infrared_frame"), ("stream_profile", "get_extrinsics_to"),
                              ("frame", "supports_frame_metadata"), ("frame", "get_frame_metadata")):
            with self.subTest(method=f"{owner}.{method}"):
                self.assertTrue(hasattr(getattr(rs, owner), method), f"rs.{owner}.{method} is gone")
        self.assertEqual("exposure", rs.option.exposure.name)

    def test_the_extrinsics_matrix_maps_a_point_as_librealsense_does(self) -> None:
        extrinsics = rs.extrinsics()
        turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # 90 degrees about z, row by row
        extrinsics.rotation = [float(v) for v in turn.T.reshape(-1)]            # librealsense keeps it column-major
        extrinsics.translation = [0.015, -0.002, 0.003]
        point_m = [0.1, 0.2, 0.3]
        expected_mm = np.asarray(rs.rs2_transform_point_to_point(extrinsics, point_m)) * 1000.0
        mapped = _extrinsics_mm(extrinsics) @ np.array([100.0, 200.0, 300.0, 1.0])
        np.testing.assert_allclose(expected_mm, mapped[:3], atol=1e-3)

    def test_a_software_device_hands_back_the_extrinsics_registered_between_two_streams(self) -> None:
        device = rs.software_device()
        depth_sensor, colour_sensor = device.add_sensor("Depth"), device.add_sensor("Color")

        def stream(kind: Any, uid: int, fmt: Any, bpp: int) -> Any:
            video = rs.video_stream()
            video.type, video.index, video.uid = kind, 0, uid
            video.width, video.height, video.fps, video.bpp, video.fmt = 64, 48, 30, bpp, fmt
            lens = rs.intrinsics()
            lens.width, lens.height, lens.ppx, lens.ppy, lens.fx, lens.fy = 64, 48, 32.0, 24.0, 60.0, 60.0
            lens.model, lens.coeffs = rs.distortion.brown_conrady, [0.0] * 5
            video.intrinsics = lens
            return video

        depth = depth_sensor.add_video_stream(stream(rs.stream.depth, 0, rs.format.z16, 2))
        colour = colour_sensor.add_video_stream(stream(rs.stream.color, 1, rs.format.bgr8, 3))
        registered = rs.extrinsics()
        registered.rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        registered.translation = [0.015, 0.0, 0.0]
        depth.register_extrinsics_to(colour, registered)
        np.testing.assert_allclose([15.0, 0.0, 0.0], _extrinsics_mm(depth.get_extrinsics_to(colour))[:3, 3], atol=1e-4)
        np.testing.assert_allclose([-15.0, 0.0, 0.0], _extrinsics_mm(colour.get_extrinsics_to(depth))[:3, 3], atol=1e-4)
        intr = depth.as_video_stream_profile().get_intrinsics()
        self.assertEqual("brown_conrady", intr.model.name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
