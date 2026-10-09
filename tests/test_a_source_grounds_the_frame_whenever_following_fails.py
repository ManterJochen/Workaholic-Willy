"""A camera source grounds the frame whenever following fails (``RealSenseVisionPerceptionSource.acquire(follow=)``).

Handed a follow, the source finds the parts a task kept again on the frame it grabbed, with no detector: the follow
reads the frame and boxes the parts, the backend's segmenter cuts the boxes (``segment_boxes``), every mask goes through
the label mapping and the colour check as every mask does, and the follow accepts all of them or none. Any doubt, and
the same frame is grounded by the detector as before (the owner, 2026-10-09). One grab either way, and ``last_route``
says which. Real: the source, its ``TwoStageBackend`` and the kept scene; stand-ins: the camera (a ray-cast frame of
two grey cubes, ``tests/test_a_kept_scene_follows_only_what_stands_where_it_stood``), a grounder that boxes every cube
it is shown, and a segmenter that cuts the part most of a box shows.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.models.detection.types import Detection
from src.robot.perception import RealSenseVisionPerceptionSource
from tests._wrist_views import K
from tests.test_a_kept_scene_follows_only_what_stands_where_it_stood import (
    LOOK,
    PROMPT,
    Part,
    Shot,
    cut,
    kept_of,
    shoot,
    two_grey_cubes,
)


class _Camera:
    """The wrist camera, holding one shot; it counts its grabs."""

    def __init__(self, shot: Shot) -> None:
        self.shot = shot
        self.grabs = 0

    def grab(self) -> RGBDFrame:
        self.grabs += 1
        return RGBDFrame(color=self.shot.bgr.copy(), depth=np.round(self.shot.depth).astype(np.uint16))

    def get_intrinsics(self) -> np.ndarray:
        return K.copy()


@dataclass(frozen=True)
class _Seg:
    mask: np.ndarray
    label: str


class _Grounder:
    """Boxes every part of the shot, labelled with the prompt's words, and counts what it was asked."""

    def __init__(self, camera: _Camera) -> None:
        self.camera = camera
        self.asked: list[str] = []

    def detect_all(self, bgr: Any, prompt: str) -> list[Detection]:
        self.asked.append(prompt)
        found = []
        for index in range(len(self.camera.shot.parts)):
            rows, cols = np.nonzero(self.camera.shot.hit == index)
            box = [float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)]
            found.append(Detection(box=box, x_center=(box[0] + box[2]) / 2.0, y_center=(box[1] + box[3]) / 2.0,
                                   label=prompt, score=1.0))
        return found


class _Segmenter:
    """Cuts the part most of a box shows, whole; ``raise_on_label`` makes a box of that label raise."""

    def __init__(self, camera: _Camera, *, raise_on_label: str = "") -> None:
        self.camera = camera
        self.raise_on_label = raise_on_label
        self.cut: list[str] = []

    def segment_detection(self, bgr: Any, det: Any) -> _Seg:
        self.cut.append(det.label)
        if self.raise_on_label and det.label == self.raise_on_label:
            raise RuntimeError("SAM2 out of memory")
        return _Seg(mask=cut(self.camera.shot, det.box).astype(np.uint8), label=det.label)


def _source(shot: Shot, **keywords: Any) -> "tuple[RealSenseVisionPerceptionSource, _Camera, _Grounder, _Segmenter]":
    camera = _Camera(shot)
    grounder = _Grounder(camera)
    segmenter = _Segmenter(camera, **keywords)
    source = RealSenseVisionPerceptionSource(streamer=camera, detector=grounder, segmenter=segmenter, prompt=PROMPT,
                                             object_labels=("grey cube",), warmup_grabs=2)
    return source, camera, grounder, segmenter


def _follow(kept: Any, **keywords: Any) -> Any:
    """The follow a pick loop hands its first look: the frame placed at the camera the shot was taken from."""

    def follow(depth_mm: np.ndarray, bgr: np.ndarray, intrinsics: np.ndarray, tool_pose: Any) -> Any:
        return kept.following(depth_mm, bgr, intrinsics, keywords.get("camera", kept.camera_to_base),
                              look=keywords.get("look", LOOK), prompt=keywords.get("prompt", PROMPT))

    return follow


class AFollowedFrameAsksNoDetectorTests(unittest.TestCase):
    def test_the_parts_are_followed_from_one_grab_with_no_detector(self) -> None:
        scene = two_grey_cubes()
        source, camera, grounder, segmenter = _source(shoot(scene))

        frame = source.acquire(follow=_follow(kept_of(shoot(scene))))

        self.assertEqual([], grounder.asked)
        self.assertEqual(3, camera.grabs, "two warm-ups and one grab, as for a grounded frame")
        self.assertEqual(["grey cube", "grey cube"], [seg.label for seg in frame.segmentations])
        self.assertEqual(["grey cube", "grey cube"], segmenter.cut)
        self.assertEqual(("followed", "2 part(s) followed, nothing new in depth"), source.last_route)
        self.assertEqual(0, source.backend.failures)

    def test_a_frame_handed_no_follow_is_grounded_and_says_no_route(self) -> None:
        scene = two_grey_cubes()
        source, camera, grounder, _ = _source(shoot(scene))

        frame = source.acquire()

        self.assertEqual([PROMPT], grounder.asked)
        self.assertEqual(3, camera.grabs)
        self.assertEqual(2, len(frame.segmentations))
        self.assertIsNone(source.last_route)


class AFollowThatFailsGroundsTheSameFrameTests(unittest.TestCase):
    def _grounded(self, source: Any, camera: _Camera, grounder: _Grounder, frame: Any) -> str:
        self.assertEqual([PROMPT], grounder.asked, "the frame is grounded by the detector")
        self.assertEqual(3, camera.grabs, "on the frame already grabbed")
        self.assertEqual(2, len(frame.segmentations))
        route, why = source.last_route
        self.assertEqual("grounded", route)
        self.assertTrue(why.startswith("not followed: "))
        return why

    def test_a_scene_that_changed_is_grounded(self) -> None:
        scene = two_grey_cubes()
        now = shoot(scene)
        source, camera, grounder, _ = _source(now)
        # The second cube stood 100 mm away when it was kept: now it stands where depth read the bench.
        kept = kept_of(shoot([scene[0], Part("grey", (60.0, -660.0))]))

        why = self._grounded(source, camera, grounder, source.acquire(follow=_follow(kept)))

        self.assertIn("changed in depth", why)

    def test_a_box_the_segmenter_cannot_cut_is_grounded_and_counts_no_failure(self) -> None:
        scene = two_grey_cubes()
        # The boxes of the parts kept carry their kept label; the detector's carry the prompt's words, and cut.
        source, camera, grounder, _ = _source(shoot(scene), raise_on_label="grey cube")

        frame = source.acquire(follow=_follow(kept_of(shoot(scene))))

        self.assertEqual([PROMPT], grounder.asked)
        self.assertIn("did not cut every box", source.last_route[1])
        self.assertEqual(2, len(frame.segmentations))
        self.assertEqual(0, source.backend.failures, "a box not cut is no failure of the model")

    def test_a_part_the_colour_check_refuses_is_grounded(self) -> None:
        scene = two_grey_cubes()
        green = shoot([replace(scene[0], colour="green"), scene[1]])
        source, camera, grounder, _ = _source(green)
        # A memory that kept the green part under the grey label, its colour green: only the colour check can tell.
        kept = kept_of(green)

        why = self._grounded(source, camera, grounder, source.acquire(follow=_follow(kept)))

        self.assertIn("colour check", why)

    def test_a_mask_not_accepted_is_grounded(self) -> None:
        scene = two_grey_cubes()
        source, camera, grounder, _ = _source(shoot([scene[0].moved(dx=15.0), scene[1]]))

        why = self._grounded(source, camera, grounder, source.acquire(follow=_follow(kept_of(shoot(scene)))))

        self.assertIn("moved 15", why)

    def test_a_follow_that_raises_is_grounded(self) -> None:
        scene = two_grey_cubes()
        source, camera, grounder, _ = _source(shoot(scene))

        def broken(*_: Any) -> Any:
            raise ValueError("a frame the memory cannot read")

        why = self._grounded(source, camera, grounder, source.acquire(follow=broken))

        self.assertIn("ValueError", why)

    def test_a_backend_that_cuts_no_boxes_is_grounded(self) -> None:
        scene = two_grey_cubes()
        camera = _Camera(shoot(scene))
        grounder = _Grounder(camera)

        class _NoBoxes:
            def perceive(self, bgr: Any, prompt: str) -> Any:
                from src.models.perception_backend import TwoStageBackend

                return TwoStageBackend(detector=grounder, segmenter=_Segmenter(camera)).perceive(bgr, prompt)

        source = RealSenseVisionPerceptionSource(streamer=camera, backend=_NoBoxes(), prompt=PROMPT,
                                                 object_labels=("grey cube",), warmup_grabs=2)

        why = self._grounded(source, camera, grounder, source.acquire(follow=_follow(kept_of(shoot(scene)))))

        self.assertIn("cuts no boxes", why)

    def test_another_phrase_is_grounded(self) -> None:
        scene = two_grey_cubes()
        source, camera, grounder, _ = _source(shoot(scene))

        why = self._grounded(source, camera, grounder,
                             source.acquire(follow=_follow(kept_of(shoot(scene)), prompt="each separate bolt")))

        self.assertIn("phrase", why)

    def test_the_next_frame_handed_no_follow_says_no_route_again(self) -> None:
        scene = two_grey_cubes()
        source, _, _, _ = _source(shoot(scene))
        source.acquire(follow=_follow(kept_of(shoot(scene))))

        source.acquire()

        self.assertIsNone(source.last_route)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
