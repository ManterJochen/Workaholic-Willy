"""A build made with the rest of its closing line is the build ``_build`` makes alone: the same verdict, the same
causes, the same numbers, and where a number lies on its threshold it is ``_build``'s own.

``support_footprint._build_many`` makes a line's builds at once (``batched``, ``robot.grasping.batched_builds``). Its
ladder (``_ladder``) is ``_run_unit``'s tilt, ``_rolled`` and ``_build``'s binormal on every rung at once, to the bit,
for every tilt and roll of the three searches and a thousand random axes. Over ten thousand anchors jittered off the
lines of five scenes, with every search forced, each build's verdict, causes and numbers are ``_build``'s. And the three
numbers it reads over many builds at once and holds 1e-6 off their thresholds hand a build to ``_build`` where they lie
on one: a hand the guard's distance and its slack from a camera's box, a corridor point on a pixel's half, a finger on
the edge of the solid the guard holds under it. That build is made alone, once, and answers as it does alone; half a
millimetre or a quarter of a pixel off the threshold, the line decides it.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.robot.grasping.generation import support_footprint as sf
from src.robot.grasping.generation.scene_obstacles import SeenEnvelope, SeenInTheFrame
from src.robot.grasping.generation.support_footprint import (
    COARSE,
    FINE,
    ROLLED,
    HandFloor,
    SfeInputs,
    plan_support_footprint,
)
from tests.test_a_lines_builds_made_at_once_answer_to_the_bit import said
from tests.test_sfe_says_why_it_refused import box_cloud, hande_jaw
from tests.test_sfe_units_in_any_order_give_the_search_its_answer import scenes


def _bits(value: Any) -> bytes:
    return np.asarray(value, dtype=np.float64).tobytes()


class TheLadderIsTheLoopsOwnTests(unittest.TestCase):
    def test_every_rung_of_every_search_is_the_loops_axis_approach_and_binormal_to_the_bit(self) -> None:
        prism = sf.reconstruct_support_prism(box_cloud((-20.0, -15.0, 0.0), (20.0, 15.0, 40.0)), 0.0)
        assert prism is not None
        rng = np.random.default_rng(20261009)
        raw_axes = [np.array([prism.u[0], prism.u[1], 0.0]), np.array([prism.v[0], prism.v[1], 0.0])]
        raw_axes += [np.array([x, y, 0.0]) for x, y in rng.standard_normal((1000, 2))]
        ladders = ((sf._TILTS_DEG, (0.0,)), (sf._FINE_TILTS_DEG, (0.0,)), (sf._TILTS_DEG, sf._FINE_ROLL_DEG))
        rungs = 0
        for raw in raw_axes:
            a2 = raw[:2] / max(float(np.linalg.norm(raw[:2])), sf._EPS)
            axis = np.array([a2[0], a2[1], 0.0])
            for tilts, rolls in ladders:
                ladder = sf._ladder(prism, axis, tilts, rolls)
                row = 0
                for index, tilt in enumerate(tilts):
                    for sign in ((1.0,) if tilt == 0.0 else (1.0, -1.0)):
                        tilted = sf._rodrigues(np.array([0.0, 0.0, -1.0]), axis, sign * np.radians(tilt))
                        tilted /= float(np.linalg.norm(tilted))
                        for roll in rolls:
                            turned, approach = sf._rolled(axis, tilted, roll)
                            binormal = sf._cross3(approach, turned)
                            binormal /= max(float(np.linalg.norm(binormal)), sf._EPS)
                            self.assertEqual(_bits(turned), _bits(ladder.axis[row]))
                            self.assertEqual(_bits(approach), _bits(ladder.approach[row]))
                            self.assertEqual(_bits(binormal), _bits(ladder.binormal[row]))
                            self.assertEqual(_bits(-turned), _bits(ladder.back[row]))
                            self.assertEqual(_bits(prism.normals @ turned[:2]), _bits(ladder.facing[row]))
                            off = (float(tilt) if roll == 0.0 else
                                   math.degrees(math.acos(max(-1.0, min(1.0, -float(approach[2]))))))
                            self.assertEqual(_bits(off), _bits(ladder.off_vertical[row]))
                            self.assertIn(row, ladder.rungs[index].tolist())
                            row += 1
                            rungs += 1
                self.assertEqual(row, ladder.size)
        self.assertGreater(rungs, 90_000)


class EveryBuildIsBuildsOwnTests(unittest.TestCase):
    def test_ten_thousand_jittered_builds_get_builds_verdict_causes_and_numbers(self) -> None:
        rng = np.random.default_rng(7)
        builds, kept = 0, 0
        differ: list[tuple[Any, ...]] = []
        for name, (cloud, keywords) in scenes().items():
            inputs = SfeInputs(cloud, counting=True, batched=True, **keywords)
            plan = plan_support_footprint(inputs)
            assert plan is not None
            obstacles, side, _every = plan.grids()
            for search in (COARSE, FINE, ROLLED):
                for axis_index in range(len(plan.searches[search][0])):
                    for frac_index in range(len(sf._FRACS)):
                        line = plan.line(search, axis_index, frac_index)
                        if line is None:
                            continue
                        ladder = plan.ladder(search, axis_index, line.axis)
                        # Two heights on the line, each jittered off it: every rung of the ladder at each.
                        z = rng.uniform(plan.prism.z0 + 1.0, plan.prism.z1, size=2)
                        at = np.column_stack([np.full(2, line.mid[0]), np.full(2, line.mid[1]), z])
                        at = at + rng.normal(scale=4.0, size=(2, 3))
                        anchors = np.repeat(at, ladder.size, axis=0)
                        rows = np.tile(np.arange(ladder.size), 2)
                        many, causes = sf._build_many(plan, anchors, ladder, rows, counting=True)
                        for build, row in enumerate(rows.tolist()):
                            counted: dict[str, int] = {}
                            alone = sf._build(plan.prism, anchors[build].copy(), ladder.axis[row],
                                              ladder.approach[row], plan.jaw, obstacles, inputs.support_height_mm,
                                              palm_aware=inputs.palm_aware, score_weights=inputs.score_weights,
                                              refusals=counted, side=side, tilt_deg=ladder.off_vertical[row],
                                              seen=inputs.seen_envelope, floor=inputs.hand_floor)
                            if (said(alone) != said(many[build])
                                    or tuple(k for k, n in counted.items() for _ in range(n)) != causes[build]):
                                differ.append((name, search, axis_index, row, alone is None, causes[build]))
                            builds += 1
                            kept += alone is not None
        self.assertEqual([], differ[:5], f"{len(differ)} of {builds} builds differ")
        self.assertGreater(builds, 10_000)
        self.assertGreater(kept, 100, "the scenes keep grasps too")


# ---------------------------------------------------------------------------------------------------------------------
# On a threshold, a build is made alone
# ---------------------------------------------------------------------------------------------------------------------


def _an_open_cube(**keywords: Any) -> tuple[Any, Any, np.ndarray, int]:
    """A 40 mm cube on the bench, its plan with ``keywords``, its first vertical build's ladder, anchor and rung."""
    cloud = box_cloud((-20.0, -20.0, 0.0), (20.0, 20.0, 40.0))
    plan = plan_support_footprint(SfeInputs(cloud, jaw=hande_jaw(), counting=True, batched=True, **keywords))
    assert plan is not None
    line = plan.line(COARSE, 0, 0)
    assert line is not None
    ladder = plan.ladder(COARSE, 0, line.axis)
    return plan, ladder, np.array([line.mid[0], line.mid[1], line.heights[0]]), 0


def _made_alone(test: unittest.TestCase, plan: Any, ladder: Any, anchor: np.ndarray, row: int, *,
                times: int = 1) -> None:
    """The build is handed to ``_build`` ``times`` (once on a threshold, never off it), and answers what ``_build``
    answers alone."""
    obstacles, side, _every = plan.grids()
    inputs = plan.inputs
    with mock.patch.object(sf, "_build", wraps=sf._build) as alone:
        many, causes = sf._build_many(plan, anchor[None, :].copy(), ladder, np.array([row]), counting=True)
    test.assertEqual(times, alone.call_count, "made alone" if times else "made with its line")
    counted: dict[str, int] = {}
    expected = sf._build(plan.prism, anchor.copy(), ladder.axis[row], ladder.approach[row], plan.jaw, obstacles,
                         inputs.support_height_mm, palm_aware=inputs.palm_aware, score_weights=inputs.score_weights,
                         refusals=counted, side=side, tilt_deg=ladder.off_vertical[row], seen=inputs.seen_envelope,
                         floor=inputs.hand_floor)
    test.assertEqual(said(expected), said(many[0]))
    test.assertEqual(tuple(k for k, n in counted.items() for _ in range(n)), causes[0])


class OnAThresholdABuildIsMadeAloneTests(unittest.TestCase):
    def test_a_hand_the_guards_distance_and_its_slack_from_a_box_is_made_alone(self) -> None:
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.safety.planning.perceived import SeenBox
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

        cfg = hande_cell()
        hand = build_gripper_geometry(cfg.grasping.gripper_geometry)
        far = SeenBox(centre_mm=(0.0, 0.0, -500.0), rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                      half_extents_mm=(5.0, 5.0, 5.0))
        probe = SeenEnvelope.of([far], gripper_model=hand, open_width_mm=float(cfg.gripper.max_width_mm),
                                distance_mm=3.0)
        assert probe is not None
        plan, ladder, anchor, row = _an_open_cube()
        turn = np.column_stack([ladder.axis[row], ladder.binormal[row], ladder.approach[row]])
        centres = anchor + probe._swept_centres @ turn.T
        reach = np.abs(turn) @ probe._swept_halves.T
        # A box beyond the hand's farthest reach along +x, exactly the guard's 3 mm and the 1 mm slack off it, and
        # half a millimetre farther, where the line decides.
        k = int(np.argmax(centres[:, 0] + reach[0]))
        edge = float(centres[k, 0] + reach[0, k])
        for gap, times in ((4.0, 1), (4.5, 0)):
            box = SeenBox(centre_mm=(edge + gap + 10.0, float(centres[k, 1]), float(centres[k, 2])),
                          rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                          half_extents_mm=(10.0, 10.0, 10.0))
            envelope = SeenEnvelope.of([box], gripper_model=hand, open_width_mm=float(cfg.gripper.max_width_mm),
                                       distance_mm=3.0)
            assert envelope is not None
            said_many = int(envelope.refuses_many(anchor[None, :], turn[None])[0])
            self.assertEqual(sf._UNSURE if times else sf._NO, said_many)
            plan, ladder, anchor, row = _an_open_cube(seen_envelope=envelope)
            _made_alone(self, plan, ladder, anchor, row, times=times)

    def test_a_corridor_point_on_a_pixels_half_is_made_alone(self) -> None:
        plan, ladder, anchor, _ = _an_open_cube(side_approaches=True)
        row = int(ladder.rungs[1][0])   # 15 degrees: the seen test is asked
        jaw = plan.jaw
        batch = plan.batch()
        x, a, b = ladder.axis[row:row + 1], ladder.approach[row:row + 1], ladder.binormal[row:row + 1]
        swept = sf._fingers_many(anchor[None, :] + (jaw.aperture_mm / 2.0) * x, a, b, jaw, batch.open_lengths,
                                 batch.widths)[0]
        point = swept[swept[:, 2] >= 0.0][0]
        to_camera = np.eye(4)
        to_camera[2, 3] = 500.0   # the camera's z is BASE z, 500 mm under the bench: every point in front of it
        z = point[2] + 500.0
        fx = 500.0
        cx = 160.5 - fx * point[0] / z
        while fx * point[0] / z + cx != 160.5:
            cx = float(np.nextafter(cx, 160.5 if fx * point[0] / z + cx < 160.5 else -np.inf))
        # On the pixel's half, and a quarter of a pixel off it, where the line decides.
        for shift, times in ((0.0, 1), (0.25, 0)):
            seen = SeenInTheFrame(depth_mm=np.full((240, 320), 5000.0), to_camera=to_camera, fx=fx, fy=fx,
                                  cx=cx + shift, cy=120.0)
            asked = (swept[:, 2] >= 0.0)[None, :]
            self.assertEqual(sf._UNSURE if times else sf._NO, int(seen.unseen_many(swept[None, :, :], asked)[0]))
            plan, ladder, anchor, _ = _an_open_cube(side_approaches=True, corridor_seen=seen)
            _made_alone(self, plan, ladder, anchor, row, times=times)

    def test_a_finger_on_the_edge_of_the_solid_under_it_is_made_alone(self) -> None:
        from src.robot.safety.planning.support_surfaces import SupportSolid

        plan, ladder, anchor, row = _an_open_cube()
        jaw = plan.jaw
        axis, approach, binormal = ladder.axis[row], ladder.approach[row], ladder.binormal[row]
        many, _ = sf._build_many(plan, anchor[None, :].copy(), ladder, np.array([row]), counting=True)
        self.assertIsNotNone(many[0], "the build stands without the solid")
        # The closed finger on the +axis side, where ``_build`` stands it: its lowest row, the tip.
        contacts = sf._pad_contacts(plan.prism, anchor, axis, approach, binormal, jaw)
        assert contacts is not None
        finger = sf._finger_points(sf._on_the_axis(contacts[3], anchor, axis) + jaw.finger_thickness_mm / 2.0 * axis,
                                   approach, binormal, jaw)
        tip = finger[np.argmin(finger[:, 2])]
        along = int(np.argmax(np.abs(axis)))
        x = axis[None, :]
        half = np.array([2.0, 2.0, 5.0])
        centre = np.array(tip, dtype=np.float64)
        centre[2] = tip[2] - 3.0 - half[2] + 1.0   # its top 1 mm under the guard's 3 mm below the tip
        # Its edge along the closing axis on the tip; and half a millimetre short of it, where the line decides.
        for short, times in ((0.0, 1), (0.5, 0)):
            centre[along] = tip[along] - np.sign(x[0, along]) * (half[along] + short)
            solid = SupportSolid(name="s", kind="support", surface=0, centre_mm=centre.copy(), half_extents_mm=half,
                                 rotation=np.eye(3), local_plane=np.array([0.0, 0.0, 0.0]), excess_mm=0.0,
                                 band_mm=0.0, allowance_mm=0.0)
            floor = HandFloor(solids=(solid,), distance_mm=3.0)
            said_many = int(floor.under_many(finger[None, :, :], np.ones((1, finger.shape[0]), bool))[0])
            self.assertEqual(sf._UNSURE if times else sf._NO, said_many)
            plan, ladder, anchor, row = _an_open_cube(hand_floor=floor)
            _made_alone(self, plan, ladder, anchor, row, times=times)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
