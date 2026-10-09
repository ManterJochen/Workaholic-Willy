"""A kept scene follows only what stands where it stood (``src/robot/perception/kept_scene.py``, the owner, 2026-10-09).

A task asks the detector once and then follows its parts: the next pick's first look finds them again with SAM2 on
their boxes, no detector asked, where nothing changed in depth and every part passes every check, all or nothing, and is
grounded as before where anything does not. These pin each check on synthetic frames: a straight-down wrist camera over
a bench at z 0, 40 mm cubes on it, ray-cast (``tests/_wrist_views.render``), and a segmenter that cuts the part most of a
box shows, as SAM2 cut the cell's cubes from their projected boxes (IoU 0.73 to 0.80 of the depth blob, 0 % outside the
footprint, ``speedmap4/reuse/sam2_follow.py``).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion
from tests._wrist_views import HEIGHT, K, WIDTH, Box, render

#: A camera 600 mm over the bench, looking straight down, the image's x along BASE x.
CAMERA = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, -600.0], [0.0, 0.0, -1.0, 600.0], [0.0, 0.0, 0.0, 1.0]])
#: The owner's black mat and the parts' colours, BGR.
MAT_BGR = (34, 40, 42)
COLOURS = {"grey": (156, 160, 162), "green": (60, 160, 60), "red": (40, 40, 200)}
LOOK = "wrist@home"
PROMPT = "each separate grey cube"


@dataclass(eq=False)
class Part:
    """A box on the bench: its colour, its middle (BASE xy), its sides and its height, and its label."""

    colour: str
    xy: tuple[float, float]
    size: float = 40.0
    height: float = 40.0
    label: str = "grey cube"

    def moved(self, dx: float = 0.0, dy: float = 0.0) -> "Part":
        return replace(self, xy=(self.xy[0] + dx, self.xy[1] + dy))

    @property
    def box(self) -> Box:
        x, y = self.xy
        half = self.size / 2.0
        return Box((x - half, y - half, 0.0), (x + half, y + half, self.height), self.label)


@dataclass(eq=False)
class Shot:
    """One frame of the scene: measured depth (mm), BGR colour, which part each pixel shows (-1 the bench)."""

    parts: list[Part]
    depth: np.ndarray
    bgr: np.ndarray
    hit: np.ndarray
    camera: np.ndarray = field(default_factory=lambda: CAMERA.copy())


def shoot(parts: list[Part], *, camera: np.ndarray = CAMERA, holes: tuple[tuple[int, int, int, int], ...] = ()) -> Shot:
    """The scene rendered from ``camera``; ``holes`` are pixel windows ``(x0, y0, x1, y1)`` where no depth was read."""
    depth, hit = render(camera, tuple(part.box for part in parts))
    depth = np.round(depth).astype(np.float64)
    bgr = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    bgr[...] = MAT_BGR
    for index, part in enumerate(parts):
        bgr[hit == index] = COLOURS[part.colour]
    for x0, y0, x1, y1 in holes:
        depth[y0:y1, x0:x1] = 0.0
    return Shot(parts=list(parts), depth=depth, bgr=bgr, hit=hit, camera=np.asarray(camera, dtype=np.float64))


def view_of(shot: Shot, *, masks: "list[np.ndarray] | None" = None, labels: "list[str] | None" = None,
            name: str = LOOK) -> Any:
    """What a pick loop's look keeps of ``shot`` (``_LookView``): its segmentations, each part's surface in BASE less
    the pixels behind a depth step and the grazing ones, as ``BinPickingOrchestrator._look_view`` keeps them."""
    from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps
    from src.robot.grasping.multiview.scene_geometry import grazing_pixels, to_base_mm

    masks = masks if masks is not None else [shot.hit == index for index in range(len(shot.parts))]
    labels = labels if labels is not None else [part.label for part in shot.parts]
    segmentations = tuple(SimpleNamespace(mask=mask.astype(np.uint8), label=label) for mask, label in zip(masks, labels))
    clouds = []
    for mask in masks:
        surface = np.asarray(mask).astype(bool)
        surface = surface & ~pixels_behind_depth_steps(surface, shot.depth)
        surface &= ~grazing_pixels(surface, shot.depth, K)
        clouds.append(to_base_mm(surface, shot.depth, K, shot.camera))
    frame = SimpleNamespace(segmentations=segmentations, rgb=shot.bgr[..., ::-1], intrinsics=K.copy())
    return SimpleNamespace(frame=frame, depth=shot.depth, camera_to_base=shot.camera, clouds=tuple(clouds), name=name)


def kept_of(shot: Shot, **keywords: Any) -> Any:
    """The memory a task keeps of ``shot``'s first look: its grey cubes, on the bench (support 0)."""
    from src.robot.perception.kept_scene import KeptScene

    options = {"label": "grey cube", "prompt": PROMPT, "pick": 1, "support_mm": 0.0, **keywords}
    view = options.pop("view", None) or view_of(shot)
    kept = KeptScene.of_view(view, **options)
    assert kept is not None
    return kept


def cut(shot: Shot, box: tuple[float, float, float, float]) -> np.ndarray:
    """SAM2 on ``box``, as the cell's SAM2 cut a cube: the part most of the box shows, whole."""
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    window = shot.hit[y0:y1, x0:x1]
    shown = window[window >= 0]
    if not shown.size:
        return np.zeros(shot.hit.shape, dtype=bool)
    return shot.hit == int(np.bincount(shown).argmax())


def follow(kept: Any, shot: Shot, *, look: str = LOOK, prompt: str = PROMPT, regions: tuple[Any, ...] = (),
           masks: "list[np.ndarray] | None" = None) -> "tuple[Any, Any]":
    """What the next pick's first look makes of ``shot``: the boxes (or why not), and the masks accepted (or why not)."""
    following = kept.following(shot.depth, shot.bgr, K, shot.camera, look=look, prompt=prompt, regions=regions)
    if following.why:
        return following, None
    cut_masks = masks if masks is not None else [cut(shot, box) for box, _ in following.boxes]
    return following, following.accept(shot.bgr, cut_masks)


def followed(kept: Any, shot: Shot, **keywords: Any) -> bool:
    following, accepted = follow(kept, shot, **keywords)
    return not following.why and accepted is not None and not accepted.why


def why(kept: Any, shot: Shot, **keywords: Any) -> str:
    following, accepted = follow(kept, shot, **keywords)
    return following.why or (accepted.why if accepted is not None else "")


def two_grey_cubes() -> list[Part]:
    return [Part("grey", (-60.0, -600.0)), Part("grey", (60.0, -560.0))]


class APartThatStandsWhereItStoodIsFollowedTests(unittest.TestCase):
    def test_an_unchanged_part_is_followed(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))

        following, accepted = follow(kept, shoot(scene))

        self.assertEqual("", following.why)
        self.assertEqual(["grey cube", "grey cube"], [label for _, label in following.boxes])
        self.assertEqual("", accepted.why)
        self.assertEqual(2, len(accepted.masks))
        self.assertIn("nothing new in depth", following.said)

    def test_each_box_holds_its_own_part_only(self) -> None:
        scene = two_grey_cubes()
        shot = shoot(scene)
        following, _ = follow(kept_of(shot), shot)

        for index, (box, _) in enumerate(following.boxes):
            self.assertTrue(np.array_equal(cut(shot, box), shot.hit == index), f"box {index} cut another part")

    def test_a_part_nudged_5_mm_is_followed_where_its_new_mask_stands(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))
        nudged = shoot([scene[0].moved(dx=5.0), scene[1]])

        following, accepted = follow(kept, nudged)

        self.assertEqual("", following.why)
        self.assertEqual("", accepted.why)
        self.assertTrue(np.array_equal(accepted.masks[0], nudged.hit == 0), "the mask is this frame's, not the kept")

    def test_a_part_moved_8_mm_is_boxed_wide_enough_for_its_mask_to_say_so(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))

        self.assertTrue(followed(kept, shoot([scene[0].moved(dy=8.0), scene[1]])))

    def test_a_part_moved_15_mm_is_grounded_again(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))

        said = why(kept, shoot([scene[0].moved(dx=15.0), scene[1]]))

        self.assertIn("part 0", said)
        self.assertIn("moved 15", said)

    def test_a_part_that_creeps_8_mm_a_pick_is_grounded_at_its_third_follow(self) -> None:
        from src.robot.perception.kept_scene import KeptScene

        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))
        verdicts = []
        for step in (1, 2, 3):
            now = shoot([scene[0].moved(dx=8.0 * step), scene[1]])
            following, accepted = follow(kept, now)
            ok = not following.why and not accepted.why
            verdicts.append(ok)
            if not ok:
                self.assertIn("crept 24", following.why or accepted.why)
                break
            # The next memory: this followed look, its parts keeping where their grounding found them.
            masks = list(accepted.masks)
            kept = KeptScene.of_view(view_of(now, masks=masks, labels=["grey cube"] * 2), label="grey cube",
                                     prompt=PROMPT, pick=step + 1, support_mm=0.0, before=kept)
            self.assertIsNotNone(kept)
            self.assertEqual(1, kept.grounded_at_pick, "a followed look keeps the pick its parts were grounded at")
        self.assertEqual([True, True, False], verdicts)

    def test_a_green_part_in_a_grey_part_s_place_is_grounded_again(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))

        said = why(kept, shoot([replace(scene[0], colour="green"), scene[1]]))

        self.assertIn("colour", said)

    def test_a_part_that_is_gone_is_grounded_again(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))

        said = why(kept, shoot([scene[1]]))

        self.assertIn("gone", said)


class AMaskThatDoesNotFitItsPartGroundsTheFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = two_grey_cubes()
        self.kept = kept_of(shoot(self.scene))
        self.shot = shoot(self.scene)

    def _masks(self) -> list[np.ndarray]:
        return [self.shot.hit == 0, self.shot.hit == 1]

    def test_a_mask_leaking_20_mm_past_its_part_is_grounded(self) -> None:
        import cv2

        masks = self._masks()
        # 20 mm at the bench, 600 mm from the camera: about 15 px of mat beside the cube.
        masks[0] = cv2.dilate(masks[0].astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)

        said = why(self.kept, self.shot, masks=masks)

        self.assertIn("part 0", said)

    def test_a_mask_of_half_its_part_is_grounded(self) -> None:
        masks = self._masks()
        rows, cols = np.nonzero(masks[0])
        half = masks[0].copy()
        half[:, : int(np.median(cols))] = False

        said = why(self.kept, self.shot, masks=[half, masks[1]])

        self.assertIn("px", said)
        self.assertIn("part 0", said)

    def test_a_mask_of_the_neighbour_is_grounded(self) -> None:
        masks = self._masks()

        self.assertIn("part 0", why(self.kept, self.shot, masks=[masks[1], masks[1]]))

    def test_masks_fewer_than_the_parts_are_grounded(self) -> None:
        following = self.kept.following(self.shot.depth, self.shot.bgr, K, self.shot.camera, look=LOOK, prompt=PROMPT)

        self.assertIn("mask(s) came back", following.accept(self.shot.bgr, [self.shot.hit == 0]).why)

    def test_an_accepted_mask_is_clipped_to_its_part_s_footprint(self) -> None:
        masks = self._masks()
        stray = masks[0].copy()
        # A few pixels of bench far from the part: under the leak bound, and never handed on.
        stray[10:13, 10:13] = True
        _, accepted = follow(self.kept, self.shot, masks=[stray, masks[1]])

        self.assertEqual("", accepted.why)
        self.assertFalse(accepted.masks[0][10:13, 10:13].any())
        self.assertTrue(np.array_equal(accepted.masks[0], masks[0]))


class SomethingNewInDepthGroundsTheFrameTests(unittest.TestCase):
    def test_a_part_set_down_on_the_bench_is_grounded(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))

        said = why(kept, shoot([*scene, Part("grey", (0.0, -680.0))]))

        self.assertIn("changed in depth", said)

    def test_a_part_set_down_where_the_kept_frame_read_holes_is_seen_the_second_way(self) -> None:
        """The shiny profiles of the cell read 36 % holes: a part set down there is new only against those holes."""
        scene = two_grey_cubes()
        added = Part("grey", (0.0, -680.0))
        now = shoot([*scene, added])
        # The kept frame read no depth where the part stands now, 5 px round it.
        rows, cols = np.nonzero(now.hit == 2)
        holes = ((int(cols.min()) - 5, int(rows.min()) - 5, int(cols.max()) + 6, int(rows.max()) + 6),)
        kept = kept_of(shoot(scene, holes=holes))

        self.assertIn("changed in depth", why(kept, now))
        # The control: without the holes the part is new the first way too.
        self.assertIn("changed in depth", why(kept_of(shoot(scene)), now))

    def test_a_person_s_arm_over_the_bench_is_grounded(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))
        arm = Part("red", (0.0, -720.0), size=120.0, height=300.0, label="arm")

        self.assertIn("changed in depth", why(kept, shoot([*scene, arm])))

    def test_a_drop_in_a_region_the_task_keeps_out_is_an_expected_change(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))
        dropped = shoot([*scene, Part("grey", (0.0, -700.0))])
        drop = ExclusionRegion.circle((0.0, -700.0), 60.0, reason="the drop at pose 'drop_left'")

        self.assertTrue(followed(kept, dropped, regions=(drop,)))
        self.assertIn("changed in depth", why(kept, dropped))

    def test_the_taken_part_s_spot_reading_the_bench_is_followed(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))
        taken = kept.without(kept.parts[0].cloud_base_mm)

        self.assertEqual(1, len(taken.parts))
        self.assertTrue(followed(taken, shoot([scene[1]])))

    def test_30_mm_of_anything_over_the_taken_part_s_spot_is_grounded(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))
        taken = kept.without(kept.parts[0].cloud_base_mm)
        under = Part("red", scene[0].xy, size=40.0, height=30.0, label="the part under it")

        said = why(taken, shoot([under, scene[1]]))

        self.assertIn("taken part's spot", said)


class AnotherPhraseOrLookIsGroundedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = two_grey_cubes()
        self.kept = kept_of(shoot(self.scene))

    def test_another_phrase_is_grounded(self) -> None:
        self.assertIn("phrase", why(self.kept, shoot(self.scene), prompt="each separate red cube"))

    def test_another_first_look_is_grounded(self) -> None:
        self.assertIn("first look", why(self.kept, shoot(self.scene), look="wrist@look 1"))

    def test_another_lens_is_grounded(self) -> None:
        shot = shoot(self.scene)
        following = self.kept.following(shot.depth, shot.bgr, K * 1.01, shot.camera, look=LOOK, prompt=PROMPT)

        self.assertIn("lens", following.why)

    def test_a_memory_with_no_part_left_never_follows(self) -> None:
        empty = replace(self.kept, parts=())

        self.assertIn("always asked of the detector", why(empty, shoot([])))

    def test_a_memory_that_knows_no_support_never_follows(self) -> None:
        self.assertIn("support", why(replace(self.kept, support_mm=None), shoot(self.scene)))


class WhatALookKeepsTests(unittest.TestCase):
    def test_only_the_parts_of_the_target_label_are_kept(self) -> None:
        scene = [*two_grey_cubes(), Part("red", (0.0, -700.0), label="red object, not grey cube")]

        kept = kept_of(shoot(scene))

        self.assertEqual(2, len(kept.parts))
        self.assertEqual({"grey cube"}, {part.label for part in kept.parts})

    def test_every_label_is_kept_where_the_task_names_none(self) -> None:
        scene = [Part("grey", (-60.0, -600.0), label="cube"), Part("red", (60.0, -560.0), label="block")]

        self.assertEqual(["cube", "block"], [part.label for part in kept_of(shoot(scene), label=None).parts])

    def test_a_part_in_a_region_the_task_keeps_out_is_not_kept(self) -> None:
        scene = [*two_grey_cubes(), Part("grey", (0.0, -720.0))]
        bin_region = ExclusionRegion.rectangle((0.0, -720.0), (100.0, 100.0), reason="the blue bin it places into")

        kept = kept_of(shoot(scene), regions=(bin_region,))

        self.assertEqual(2, len(kept.parts))

    def test_a_surface_the_parts_lie_on_is_no_part_to_keep(self) -> None:
        """A mat the detector boxed under the part's words: wide and flat, the pick loop never takes it either."""
        scene = [Part("grey", (0.0, -780.0), size=300.0, height=10.0), *two_grey_cubes()]

        kept = kept_of(shoot(scene))

        self.assertEqual(2, len(kept.parts))

    def test_a_segmentation_larger_than_a_part_keeps_nothing(self) -> None:
        """A mask that ran from a cube 300 mm across the bench: no part, and no surface either."""
        from src.robot.perception.kept_scene import KeptScene

        shot = shoot(two_grey_cubes())
        merged = shot.hit == 0
        merged[170:190, 505:520] = True  # the bench about 250 mm to +x of the cube
        view = view_of(shot, masks=[merged, shot.hit == 1])

        self.assertIsNone(KeptScene.of_view(view, label="grey cube", prompt=PROMPT, support_mm=0.0))

    def test_a_look_with_no_part_keeps_nothing(self) -> None:
        from src.robot.perception.kept_scene import KeptScene

        self.assertIsNone(KeptScene.of_view(view_of(shoot([])), label="grey cube", prompt=PROMPT, support_mm=0.0))

    def test_the_gripped_part_is_the_nearest_kept_one_within_15_mm(self) -> None:
        kept = kept_of(shoot(two_grey_cubes()))
        cloud = kept.parts[1].cloud_base_mm + np.array([10.0, 0.0, 0.0])

        taken = kept.without(cloud)

        self.assertEqual(1, len(taken.parts))
        self.assertAlmostEqual(-60.0, taken.parts[0].centre[0], delta=3.0)
        self.assertEqual(1, len(taken.taken))

    def test_a_gripped_part_no_kept_part_stands_near_keeps_nothing(self) -> None:
        kept = kept_of(shoot(two_grey_cubes()))

        self.assertIsNone(kept.without(kept.parts[1].cloud_base_mm + np.array([20.0, 0.0, 0.0])))
        self.assertIsNone(kept.without(None))


class ALaterLookFindsTheFirstLook_sPartsByTheirProjectedBoxesTests(unittest.TestCase):
    """Map2's F: the parts of a pick's first look, projected into its later looks, the camera moved and the parts not."""

    def setUp(self) -> None:
        from src.robot.perception.kept_scene import KeptScene

        self.scene = two_grey_cubes()
        self.first = KeptScene.of_view(view_of(shoot(self.scene)), label="grey cube")
        # Another look: the camera 120 mm to the side and tilted toward the parts.
        tilt = np.radians(15.0)
        rotation = np.array([[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]])
        self.later = CAMERA.copy()
        self.later[:3, :3] = CAMERA[:3, :3] @ rotation
        self.later[:3, 3] = (-150.0, -600.0, 580.0)

    def test_every_part_of_the_first_look_is_boxed_and_its_mask_accepted(self) -> None:
        shot = shoot(self.scene, camera=self.later)

        following = self.first.projected(shot.depth, shot.bgr, K, shot.camera)
        self.assertEqual("", following.why)
        masks = [cut(shot, box) for box, _ in following.boxes]

        self.assertEqual("", following.accept(shot.bgr, masks).why)
        self.assertIn("projected boxes", following.said)

    def test_a_projected_mask_reaching_past_its_part_is_grounded(self) -> None:
        import cv2

        shot = shoot(self.scene, camera=self.later)
        following = self.first.projected(shot.depth, shot.bgr, K, shot.camera)
        masks = [cut(shot, box) for box, _ in following.boxes]
        masks[1] = cv2.dilate(masks[1].astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)

        self.assertIn("past the part's footprint", following.accept(shot.bgr, masks).why)

    def test_a_projected_mask_of_half_its_part_is_grounded(self) -> None:
        shot = shoot(self.scene, camera=self.later)
        following = self.first.projected(shot.depth, shot.bgr, K, shot.camera)
        masks = [cut(shot, box) for box, _ in following.boxes]
        rows, cols = np.nonzero(masks[0])
        masks[0][:, : int(np.median(cols))] = False

        self.assertIn("px", following.accept(shot.bgr, masks).why)

    def test_a_part_the_later_look_does_not_frame_is_grounded(self) -> None:
        away = CAMERA.copy()
        away[:3, 3] = (900.0, -600.0, 600.0)
        shot = shoot(self.scene, camera=away)

        self.assertIn("falls in this look's frame", self.first.projected(shot.depth, shot.bgr, K, shot.camera).why)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
