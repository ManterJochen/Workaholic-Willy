"""A part a phrase singled out of its kind is resolved once, then followed where it stood (the owner, 2026-10-12).

A task's ``which`` ("the gray cube closest to the blue object") is the phrase its picks ground, and the relation in it is
read off one image. On the cell (2026-10-12, 10:14 and 10:20) every pick's later looks refused to follow the part by its
projected box (41 % of its mask seen past a neighbour, 2.7 % of its surface past its footprint) and grounded the
relation again from their own side: one cube each time, or none (3 of 11). The relation is now resolved once, by the
look that grounded it, and a frame whose follow of the part is refused once the follow placed it in BASE grounds the
part's kind instead (``PickPrompt.kind``, "each separate grey cube"): the part is the segmentation most of whose surface
stands on its kept footprint, and every other part of the kind is a neighbour, never a target.

* the source grounds the kind on a refused follow and keeps the part where it stood; a part found nowhere it stood
  leaves the frame without a target, and is not resolved again;
* a phrase that singles out nothing, and a follow that cannot place its frame, ground the phrase as before;
* the task names the kind beside its which, the service hands it to the camera and gives it back;
* in a pick, a later look whose projected box is refused grounds the kind and ranks the part singled out alone.

Real: the camera source, its ``TwoStageBackend``, the kept scene, and in the last the pick service and its pick loop;
stand-ins as in ``tests/test_a_source_grounds_the_frame_whenever_following_fails`` and
``tests/test_a_later_look_follows_the_parts_its_first_look_saw``.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.models.detection.types import Detection
from tests.robot.perception.test_a_kept_scene_follows_only_what_stands_where_it_stood import (
    PROMPT,
    Part,
    kept_of,
    shoot,
    two_grey_cubes,
    view_of,
)
from tests.robot.grasping.motion.test_a_later_look_follows_the_parts_its_first_look_saw import _LeakingSegmenter, _pick
from tests.camera.setup.image_taking.test_a_source_grounds_the_frame_whenever_following_fails import _follow, _source
from tests.robot.execution.test_a_task_picks_the_grey_parts_and_ends_without_a_long_search import _Grounder

#: The phrase the operator's sentence singled the part out by; :data:`PROMPT` is its kind.
RELATION = "the grey cube nearest the robot"
#: The label a part of the kind goes by where it is not the one singled out.
NOT_IT = "grey cube, not the one singled out"


def _singled_out(shot: Any, index: int = 0) -> Any:
    """What the look that grounded :data:`RELATION` kept: the one cube the relation named."""
    return kept_of(shot, prompt=RELATION, view=view_of(shot, masks=[shot.hit == index], labels=["grey cube"]))


def _relating_source(shot: Any, kind: str = PROMPT) -> Any:
    source, camera, grounder, segmenter = _source(shot)
    source.set_prompt(RELATION, object_labels=("grey cube",), kind=kind)
    return source, camera, grounder, segmenter


class ARefusedFollowGroundsTheKindTests(unittest.TestCase):
    def test_the_part_singled_out_is_kept_where_it_stood_and_its_kind_beside_it_are_neighbours(self) -> None:
        scene = two_grey_cubes()
        kept = _singled_out(shoot(scene))
        # The cube moved 15 mm since it was kept: more than a follow lets it, as far as its footprint reaches.
        source, _, grounder, _ = _relating_source(shoot([scene[0].moved(dx=15.0), scene[1]]))

        frame = source.acquire(follow=_follow(kept, prompt=RELATION))

        self.assertEqual([PROMPT], grounder.asked, "the kind is grounded, not the relation")
        self.assertEqual(["grey cube", NOT_IT], [seg.label for seg in frame.segmentations])
        route, why = source.last_route
        self.assertEqual("grounded", route)
        self.assertIn("moved 15", why)
        self.assertIn(f"the kind {PROMPT!r} grounded: 1 of 1 part(s) singled out found where they stood, 1 other", why)

    def test_a_part_found_nowhere_it_stood_leaves_the_frame_no_target(self) -> None:
        scene = two_grey_cubes()
        kept = _singled_out(shoot(scene))
        # The cube singled out is gone; the other stands where it stood, and a third came.
        source, _, grounder, _ = _relating_source(shoot([scene[1], Part("grey", (-60.0, -700.0))]))

        frame = source.acquire(follow=_follow(kept, prompt=RELATION))

        self.assertEqual([PROMPT], grounder.asked)
        self.assertEqual([NOT_IT, NOT_IT], [seg.label for seg in frame.segmentations])
        self.assertIn("0 of 1 part(s) singled out found where they stood", source.last_route[1])
        self.assertIn("stands nowhere it stood", source.last_route[1])

    def test_a_followed_frame_asks_no_detector_as_before(self) -> None:
        scene = two_grey_cubes()
        source, _, grounder, _ = _relating_source(shoot(scene))

        frame = source.acquire(follow=_follow(_singled_out(shoot(scene)), prompt=RELATION))

        self.assertEqual([], grounder.asked)
        self.assertEqual(["grey cube"], [seg.label for seg in frame.segmentations])
        self.assertEqual("followed", source.last_route[0])


class ThePhraseIsGroundedAsBeforeTests(unittest.TestCase):
    def test_a_phrase_that_singles_out_nothing_grounds_itself_and_every_part_stays_a_target(self) -> None:
        scene = two_grey_cubes()
        source, _, grounder, _ = _source(shoot([scene[0].moved(dx=15.0), scene[1]]))

        frame = source.acquire(follow=_follow(kept_of(shoot(scene))))

        self.assertEqual([PROMPT], grounder.asked)
        self.assertEqual(["grey cube", "grey cube"], [seg.label for seg in frame.segmentations])
        self.assertNotIn("the kind", source.last_route[1])

    def test_a_follow_that_cannot_place_its_frame_grounds_the_relation(self) -> None:
        scene = two_grey_cubes()
        source, _, grounder, _ = _relating_source(shoot(scene))

        def broken(*_: Any) -> Any:
            raise ValueError("a frame the memory cannot read")

        frame = source.acquire(follow=broken)

        self.assertEqual([RELATION], grounder.asked, "where the part stood cannot be read on this frame")
        self.assertEqual(["grey cube", "grey cube"], [seg.label for seg in frame.segmentations])

    def test_a_frame_handed_no_follow_grounds_the_relation(self) -> None:
        source, _, grounder, _ = _relating_source(shoot(two_grey_cubes()))

        source.acquire()

        self.assertEqual([RELATION], grounder.asked, "the look that resolves the relation")


class TheTaskNamesTheKindBesideItsWhichTests(unittest.TestCase):
    def test_a_which_names_its_kind_and_a_kind_alone_names_none(self) -> None:
        from tests.robot.execution.test_a_task_grounds_the_one_part_its_sentence_singles_out import _plan, run

        singled = run(["part"], plan=_plan("gray cube", which="the gray cube closest to the blue object",
                                           source="on the black mat")).service.prompts[0]
        every = run(["part"], plan=_plan("gray cube", source="on the black mat")).service.prompts[0]
        unnamed = run(["part"], plan=_plan("", which="the upper object")).service.prompts[0]

        self.assertEqual(("the gray cube closest to the blue object", "each separate gray cube on the black mat"),
                         (singled.phrase, singled.kind))
        self.assertEqual("", every.kind)
        self.assertEqual("", unnamed.kind, "with no kind named every part is a target: the which is grounded alone")

    def test_the_service_hands_the_kind_to_the_camera_and_gives_it_back(self) -> None:
        from src.robot.execution.autonomous_grasp import PickPrompt
        from tests.robot.execution.autonomous_grasp.test_a_sorts_pick_prompt_passes_only_its_kinds import _service

        service, source = _service()
        previous = service.set_prompt(PickPrompt(phrase=RELATION, target_label="grey cube",
                                                 object_labels=("grey cube",), kind=PROMPT))
        self.assertEqual((RELATION, PROMPT), (source.prompt, source.kind))

        back = service.set_prompt(previous)

        self.assertEqual(PROMPT, back.kind)
        self.assertEqual("", source.kind)


class _Relating(_Grounder):
    """Boxes the one part :data:`RELATION` names ("grey 2"), and every part for any other phrase."""

    def detect_all(self, bgr: Any, prompt: str) -> list[Detection]:
        found = super().detect_all(bgr, prompt)
        if prompt != RELATION:
            return found
        index = [part.name for part in self.mat.shown].index("grey 2")
        rows, cols = np.nonzero(self.mat.hit == index)
        box = [float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)]
        return [Detection(box=box, x_center=(box[0] + box[2]) / 2.0, y_center=(box[1] + box[3]) / 2.0, label=prompt,
                          score=1.0)]


def _asked(log: list[Any]) -> list[str]:
    return [entry[1] for entry in log if isinstance(entry, tuple) and entry[:1] == ("ground",)]


class ALaterLookFindsThePartSingledOutAmongItsKindTests(unittest.TestCase):
    def test_a_refused_projection_grounds_the_kind_and_ranks_the_part_singled_out_alone(self) -> None:
        _, log, perceived, calculator = _pick(follow_looks=True, segmenter=_LeakingSegmenter, grounder=_Relating,
                                              prompt=RELATION, kind=PROMPT)

        self.assertEqual([RELATION, PROMPT], _asked(log))
        self.assertIn("1 of 1 part(s) singled out found where they stood, 2 other", perceived[1].route_reason)
        later = [name for frame, name in calculator.by_frame if frame == 2]
        self.assertEqual(["grey 2"], later, "the later look ranks the part singled out, its kind beside it neighbours")
        self.assertTrue({"grey 1", "grey 3"} <= set(calculator.neighbours))

    def test_without_a_kind_the_later_look_grounds_the_relation_again(self) -> None:
        _, log, _, _ = _pick(follow_looks=True, segmenter=_LeakingSegmenter, grounder=_Relating, prompt=RELATION)

        self.assertEqual([RELATION, RELATION], _asked(log))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
