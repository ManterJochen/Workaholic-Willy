"""The grasp reference — the one artefact in D5 that must be right before any number it produces means
anything.

Two layers are protected here. The **solid geometry**, because every contact, normal and clearance is
derived from it: each closed-form answer is checked against a brute-force numeric one, so a sign error
in a quaternion or a slab test cannot pass by being self-consistent. And the **verdict**, on cases whose
answer is known by hand — a cube gripped across a face is a grasp, the same cube gripped along its
diagonal is not, and a 100 mm span does not fit an 85 mm jaw whatever else is true about it.
"""

from __future__ import annotations

import collections
import unittest

import numpy as np

from datagen.assets.procedural import primitive_for_kind
from datagen.grasps.shapes import Solid, quat_to_matrix, solid_from_scene, wall_solid
from datagen.grasps.verdict import (
    STANDARD_CUP,
    JawGrasp,
    JawModel,
    Rejection,
    SuctionGrasp,
    check_jaw_grasp,
    check_suction_grasp,
    suction_payload_ok,
)

DOWN = np.array([0.0, 0.0, -1.0])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def cube(size: float = 40.0, centre=(0.0, 0.0, 20.0), quat=IDENTITY) -> Solid:
    return solid_from_scene("box", (size, size, size), centre, quat, asset_id="cube")


#: A 40 x 40 x 90 mm block standing on the table, gripped 70 mm up. A small object gripped at its
#: MID-HEIGHT is not a usable fixture any more, and that is a real fact rather than a nuisance: the
#: measured 2F-85 pad is 41.5 mm long, so a 40 mm cube gripped at z = 20 puts the fingertip below the
#: table. Every test that used one was encoding a grasp the hardware cannot perform.
BLOCK_GRASP_Z = 70.0


def block(width: float = 40.0, height: float = 90.0, quat=IDENTITY, centre=None) -> Solid:
    middle = centre if centre is not None else (0.0, 0.0, height / 2.0)
    return solid_from_scene("box", (width, width, height), middle, quat, asset_id="block")


def brute_force_span(solid: Solid, origin, direction, *, reach: float = 400.0):
    """Where a densely marched ray enters and leaves — the independent check on ``line_span``."""
    ts = np.linspace(-reach, reach, 400_001)
    points = np.asarray(origin, dtype=np.float64)[None, :] + ts[:, None] * np.asarray(direction)
    inside = solid.contains_many(points)
    if not inside.any():
        return None
    hit = np.flatnonzero(inside)
    return float(ts[hit[0]]), float(ts[hit[-1]])


class SolidGeometryTests(unittest.TestCase):
    """Closed form vs brute force. Same question, two entirely different routes to the answer."""

    def _compare(self, solid: Solid, origin, direction) -> None:
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        exact = solid.line_span(origin, direction)
        marched = brute_force_span(solid, origin, direction)
        if marched is None:
            self.assertIsNone(exact, f"{solid.kind}: closed form found a hit the march did not")
            return
        assert exact is not None, f"{solid.kind}: closed form missed a hit the march found"
        self.assertAlmostEqual(exact[0], marched[0], delta=0.05, msg=f"{solid.kind} entry")
        self.assertAlmostEqual(exact[1], marched[1], delta=0.05, msg=f"{solid.kind} exit")

    def test_line_span_matches_a_marched_ray_for_every_primitive(self) -> None:
        rng = np.random.default_rng(20260813)
        quat = np.array([0.24, -0.11, 0.37, 0.89])
        quat /= np.linalg.norm(quat)
        solids = [
            solid_from_scene("box", (40.0, 70.0, 25.0), (10.0, -5.0, 30.0), quat),
            solid_from_scene("can", (36.0, 36.0, 90.0), (-20.0, 15.0, 45.0), quat),
            solid_from_scene("sphere", (50.0, 50.0, 50.0), (5.0, 5.0, 25.0), IDENTITY),
        ]
        for solid in solids:
            for _ in range(12):
                origin = rng.uniform(-120.0, 120.0, size=3)
                direction = rng.normal(size=3)
                self._compare(solid, origin, direction)

    def test_a_ray_that_misses_is_reported_as_a_miss(self) -> None:
        solid = cube(40.0)
        self.assertIsNone(solid.line_span(np.array([0.0, 0.0, 500.0]), np.array([1.0, 0.0, 0.0])))

    def test_support_width_matches_an_explicit_surface_model(self) -> None:
        """``support_width_mm`` decides who is too wide for the jaw, so it gets an independent check:
        the width of an explicitly enumerated surface (box corners, cylinder rims) projected on ``d``.
        """
        rng = np.random.default_rng(7)
        quat = np.array([0.5, 0.5, 0.1, 0.7])
        quat /= np.linalg.norm(quat)
        cases = [
            solid_from_scene("plate", (90.0, 60.0, 12.0), (0, 0, 40), quat),
            solid_from_scene("tube", (30.0, 30.0, 85.0), (0, 0, 40), quat),
            solid_from_scene("sphere", (44.0, 44.0, 44.0), (0, 0, 40), IDENTITY),
        ]
        for solid in cases:
            h = solid.half_extent_mm
            if solid.kind == "box":
                body = np.asarray(np.meshgrid(*[[-1.0, 1.0]] * 3)).reshape(3, -1).T * h
            elif solid.kind == "cylinder":
                theta = np.linspace(0.0, 2 * np.pi, 2000, endpoint=False)
                ring = np.stack([h[0] * np.cos(theta), h[0] * np.sin(theta), np.zeros_like(theta)], 1)
                body = np.vstack([ring + [0, 0, h[2]], ring - [0, 0, h[2]]])
            else:
                body = None
            for _ in range(10):
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                if body is None:
                    expected = 2.0 * float(h[0])
                else:
                    projected = solid.to_world(body) @ d
                    expected = float(projected.max() - projected.min())
                self.assertAlmostEqual(solid.support_width_mm(d), expected, delta=0.05,
                                       msg=f"{solid.kind} along {np.round(d, 2)}")

    def test_normals_point_outward(self) -> None:
        rng = np.random.default_rng(3)
        for solid in (cube(40.0), solid_from_scene("can", (30.0, 30.0, 80.0), (0, 0, 40), IDENTITY),
                      solid_from_scene("sphere", (50.0, 50.0, 50.0), (0, 0, 25), IDENTITY)):
            for _ in range(20):
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                origin = solid.centre_mm + direction * solid.bounding_radius_mm() * 2.0
                span = solid.line_span(origin, -direction)
                assert span is not None
                surface = origin - direction * span[0]
                normal = solid.normal_at(surface)
                # Stepping OUT along the normal must leave the solid; stepping in must stay inside.
                self.assertFalse(solid.contains(surface + normal * 1.0), f"{solid.kind}: normal inward")
                self.assertTrue(solid.contains(surface - normal * 1.0), f"{solid.kind}: normal detached")

    def test_the_kind_table_is_the_renderers(self) -> None:
        """If these drifted, the labels would describe a shape that was never rendered."""
        self.assertEqual(primitive_for_kind("flange"), "box")
        self.assertEqual(primitive_for_kind("bowl"), "box")
        self.assertEqual(primitive_for_kind("bottle"), "cylinder")
        self.assertEqual(primitive_for_kind("sphere"), "sphere")

    def test_a_quaternion_round_trips(self) -> None:
        quat = np.array([0.18, 0.44, -0.27, 0.83])
        quat /= np.linalg.norm(quat)
        R = quat_to_matrix(quat)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)


class JawVerdictTests(unittest.TestCase):
    def test_a_face_grasp_on_a_standing_block_is_a_grasp(self) -> None:
        solid = block()
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, BLOCK_GRASP_Z]), DOWN, np.array([1.0, 0.0, 0.0]), 40.0),
            solid)
        self.assertTrue(verdict.ok, verdict.detail or verdict.reason)
        self.assertAlmostEqual(verdict.object_span_mm, 40.0, places=6)
        self.assertAlmostEqual(verdict.contact_angle_deg, 0.0, places=6)

    def test_the_same_block_across_its_diagonal_is_not(self) -> None:
        """45 deg contacts are outside any sane friction cone -- the jaw would squeeze it out."""
        solid = block()
        axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, BLOCK_GRASP_Z]), DOWN, axis, 56.6), solid)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, Rejection.NOT_ANTIPODAL)
        self.assertAlmostEqual(verdict.contact_angle_deg, 45.0, delta=0.5)

    def test_an_object_wider_than_the_jaw_is_refused(self) -> None:
        solid = solid_from_scene("box", (100.0, 40.0, 90.0), (0.0, 0.0, 45.0), IDENTITY)
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, BLOCK_GRASP_Z]), DOWN, np.array([1.0, 0.0, 0.0]), 100.0),
            solid)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, Rejection.TOO_WIDE)
        self.assertAlmostEqual(verdict.object_span_mm, 100.0, places=6)

    def test_the_narrow_axis_of_that_same_object_IS_graspable(self) -> None:
        """The pair that matters: a verdict must depend on the closing axis, not on the object."""
        solid = solid_from_scene("box", (100.0, 40.0, 90.0), (0.0, 0.0, 45.0), IDENTITY)
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, BLOCK_GRASP_Z]), DOWN, np.array([0.0, 1.0, 0.0]), 40.0),
            solid)
        self.assertTrue(verdict.ok, verdict.detail or verdict.reason)

    def test_a_neighbour_in_the_way_blocks_the_fingers(self) -> None:
        solid = block()
        neighbour = solid_from_scene("box", (40.0, 40.0, 90.0), (56.0, 0.0, 45.0), IDENTITY,
                                     asset_id="neighbour")
        grasp = JawGrasp(np.array([0.0, 0.0, BLOCK_GRASP_Z]), DOWN, np.array([1.0, 0.0, 0.0]), 40.0)
        self.assertTrue(check_jaw_grasp(grasp, solid).ok, "clear scene should still pass")
        verdict = check_jaw_grasp(grasp, solid, [neighbour])
        self.assertFalse(verdict.ok)
        self.assertIn(verdict.reason, (Rejection.FINGER_COLLISION, Rejection.APPROACH_BLOCKED))
        self.assertIn("neighbour", verdict.detail)

    def test_a_grasp_below_the_table_is_refused(self) -> None:
        """Regression: the FINGERTIP leads the contact patch, so it is the part that hits the table.

        Sampling the finger from the contact backwards let this exact pose through — the 10 mm of
        finger ahead of the contact, which is what was under the table, was never looked at.
        """
        solid = solid_from_scene("box", (40.0, 40.0, 12.0), (0.0, 0.0, 6.0), IDENTITY)
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, 2.0]), DOWN, np.array([1.0, 0.0, 0.0]), 40.0), solid)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, Rejection.BELOW_TABLE)

    def test_a_non_perpendicular_axis_is_refused_rather_than_silently_projected(self) -> None:
        solid = cube(40.0)
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, 20.0]), DOWN, np.array([0.0, 0.3, -0.95]), 40.0), solid)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, Rejection.AXIS_NOT_PERPENDICULAR)

    def test_only_a_band_of_sphere_sizes_is_top_down_graspable(self) -> None:
        """A consequence of the MEASURED pad, and a real constraint rather than a modelling quirk.

        A ball resting on the table is gripped at its equator and the pad reaches 20.75 mm past the
        contact, so anything much under 54 mm across puts the fingertip below the table; above 85 mm it
        no longer fits the jaw at all. The graspable band is narrow and it is not an artefact.
        """
        small = solid_from_scene("sphere", (40.0, 40.0, 40.0), (0.0, 0.0, 20.0), IDENTITY)
        verdict = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, 20.0]), DOWN, np.array([1.0, 0.0, 0.0]), 40.0), small)
        self.assertFalse(verdict.ok, "a 40 mm ball cannot be gripped without going under the table")
        self.assertEqual(verdict.reason, Rejection.BELOW_TABLE)

        fits = solid_from_scene("sphere", (70.0, 70.0, 70.0), (0.0, 0.0, 35.0), IDENTITY)
        through = check_jaw_grasp(
            JawGrasp(np.array([0.0, 0.0, 35.0]), DOWN, np.array([1.0, 0.0, 0.0]), 70.0), fits)
        self.assertTrue(through.ok, through.detail or through.reason)
        off = check_jaw_grasp(
            JawGrasp(np.array([0.0, 40.0, 35.0]), DOWN, np.array([1.0, 0.0, 0.0]), 40.0), fits)
        self.assertFalse(off.ok, "past the shoulder the contacts leave the friction cone")

    def test_a_tilted_approach_on_a_tilted_object_still_works(self) -> None:
        """The owner's case: objects do not lie flat, so the verdict must not assume they do."""
        angle = np.radians(30.0)
        quat = np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])
        solid = solid_from_scene("box", (40.0, 40.0, 40.0), (0.0, 0.0, 90.0), quat)
        R = quat_to_matrix(quat)
        axis = R @ np.array([1.0, 0.0, 0.0])       # the tilted face normal pair
        approach = R @ np.array([0.0, 0.0, -1.0])
        verdict = check_jaw_grasp(JawGrasp(solid.centre_mm, approach, axis, 40.0), solid)
        self.assertTrue(verdict.ok, verdict.detail or verdict.reason)
        self.assertAlmostEqual(verdict.approach_tilt_deg, 30.0, delta=0.5)

    def test_the_friction_coefficient_is_what_moves_the_antipodal_line(self) -> None:
        solid = block()
        # 15 deg, deliberately away from 20 deg: there the pad's corner lands on the box EDGE, where
        # the surface normal is discontinuous and the verdict jumps. That is a stated limitation of
        # taking the pad's extreme point as the contact, not a friction effect, and a test sitting on
        # top of it would be testing the discontinuity.
        axis = np.array([np.cos(np.radians(15.0)), np.sin(np.radians(15.0)), 0.0])
        grasp = JawGrasp(np.array([0.0, 0.0, BLOCK_GRASP_Z]), DOWN, axis, 47.8)
        self.assertTrue(check_jaw_grasp(grasp, solid, model=JawModel(friction_coefficient=0.5)).ok)
        self.assertFalse(check_jaw_grasp(grasp, solid, model=JawModel(friction_coefficient=0.2)).ok)


class SuctionVerdictTests(unittest.TestCase):
    def test_a_flat_top_face_seals(self) -> None:
        solid = cube(60.0, centre=(0.0, 0.0, 30.0))
        verdict = check_suction_grasp(
            SuctionGrasp(np.array([0.0, 0.0, 60.0]), DOWN), solid)
        self.assertTrue(verdict.ok, verdict.detail or verdict.reason)

    def test_a_cup_hanging_over_the_edge_does_not(self) -> None:
        solid = cube(60.0, centre=(0.0, 0.0, 30.0))
        verdict = check_suction_grasp(
            SuctionGrasp(np.array([26.0, 0.0, 60.0]), DOWN), solid)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, Rejection.SEAL_NOT_FLAT)

    def test_a_barrel_too_curved_for_the_rim_does_not(self) -> None:
        """A 30 mm cup on a 40 mm cylinder: the rim sits ~3 mm off the plane. Physics, not opinion."""
        solid = solid_from_scene("can", (40.0, 40.0, 100.0), (0.0, 0.0, 50.0), IDENTITY)
        verdict = check_suction_grasp(
            SuctionGrasp(np.array([0.0, -20.0, 50.0]), np.array([0.0, 1.0, 0.0])), solid)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, Rejection.SEAL_NOT_FLAT)

    def test_the_slim_cup_seals_where_the_standard_one_cannot(self) -> None:
        from datagen.grasps.verdict import SLIM_CUP

        solid = solid_from_scene("can", (60.0, 60.0, 100.0), (0.0, 0.0, 50.0), IDENTITY)
        contact = SuctionGrasp(np.array([0.0, -30.0, 50.0]), np.array([0.0, 1.0, 0.0]))
        self.assertFalse(check_suction_grasp(contact, solid, cup=STANDARD_CUP).ok)
        self.assertTrue(check_suction_grasp(contact, solid, cup=SLIM_CUP).ok)

    def test_pressing_sideways_on_a_top_face_is_refused(self) -> None:
        solid = cube(60.0, centre=(0.0, 0.0, 30.0))
        verdict = check_suction_grasp(
            SuctionGrasp(np.array([0.0, 0.0, 60.0]), np.array([1.0, 0.0, 0.0])), solid)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, Rejection.APPROACH_NOT_NORMAL)

    def test_a_bin_wall_blocks_the_cup_path(self) -> None:
        solid = cube(40.0, centre=(0.0, 0.0, 20.0))
        lid = wall_solid((0.0, 0.0, 70.0), (100.0, 100.0, 4.0), name="lid")
        clear = check_suction_grasp(SuctionGrasp(np.array([0.0, 0.0, 40.0]), DOWN), solid)
        self.assertTrue(clear.ok, clear.detail or clear.reason)
        blocked = check_suction_grasp(SuctionGrasp(np.array([0.0, 0.0, 40.0]), DOWN), solid, [lid])
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.reason, Rejection.CUP_COLLISION)

    def test_payload_is_a_separate_question_from_sealing(self) -> None:
        self.assertTrue(suction_payload_ok(0.5, STANDARD_CUP))
        self.assertFalse(suction_payload_ok(5.0, STANDARD_CUP))


def _payload(objects, *, family: str = "sparse", dropped=(), walls=()) -> dict:
    """A scene.json shaped exactly like the renderer writes one, with settled poses."""
    return {
        "dropped_objects": list(dropped),
        "settled_poses_mm_xyzw": {
            str(i): [list(position), list(quat)] for i, (_, position, quat) in enumerate(objects)
        },
        "spec": {
            "family": family, "bin_walls": list(walls),
            "objects": [{"asset_id": asset_id, "mass_kg": 0.2, "color_rgb": [0.5, 0.5, 0.5],
                         "position_mm": list(position)} for asset_id, position, _ in objects],
        },
    }


class LabelTests(unittest.TestCase):
    """The label set is the recall denominator, so what it contains has to be defensible."""

    EXTENTS = {"proc_00000_box": (40.0, 40.0, 60.0), "proc_00001_plate": (90.0, 70.0, 10.0),
               "proc_00002_can": (36.0, 36.0, 90.0)}

    def test_every_label_passes_the_verdict_it_came_from(self) -> None:
        """The invariant that makes the set a reference: a label IS a grasp, checked, not proposed."""
        from datagen.grasps.labels import label_scene, scene_geometry

        payload = _payload([("proc_00000_box", (0.0, 0.0, 30.0), IDENTITY),
                            ("proc_00002_can", (200.0, 0.0, 45.0), IDENTITY)])
        labels, _ = label_scene(payload, self.EXTENTS, "s")
        geometry = scene_geometry(payload, self.EXTENTS, "s")
        self.assertTrue(labels)
        for label in labels:
            if label.kind != "jaw":
                continue
            verdict = check_jaw_grasp(
                JawGrasp(np.asarray(label.position_mm), np.asarray(label.approach),
                         np.asarray(label.closing_axis), label.width_mm),
                geometry.objects[label.instance_id], geometry.obstacles_for(label.instance_id))
            self.assertTrue(verdict.ok, f"{label.kind} label rejected as {verdict.reason}")

    def test_a_flat_plate_on_the_table_gets_NO_jaw_label(self) -> None:
        """Its only sub-aperture axis is the vertical one, and gripping that means going under the
        table. This is the finding that separates 814 jaw-graspable objects from 806 that are not."""
        from datagen.grasps.labels import label_scene

        payload = _payload([("proc_00001_plate", (0.0, 0.0, 5.0), IDENTITY)])
        labels, rejected = label_scene(payload, self.EXTENTS, "s")
        self.assertEqual([label for label in labels if label.kind == "jaw"], [])
        self.assertIn(Rejection.BELOW_TABLE, rejected)
        self.assertTrue([label for label in labels if label.kind == "suction"],
                        "...but it is a perfectly good suction target, which is the point")

    def test_labels_use_the_settled_pose_and_skip_dropped_objects(self) -> None:
        from datagen.grasps.labels import scene_geometry

        payload = _payload([("proc_00000_box", (0.0, 0.0, 30.0), IDENTITY),
                            ("proc_00002_can", (200.0, 0.0, 45.0), IDENTITY)], dropped=(1,))
        payload["settled_poses_mm_xyzw"]["0"] = [[11.0, 22.0, 33.0], [0.0, 0.0, 0.0, 1.0]]
        geometry = scene_geometry(payload, self.EXTENTS, "s")
        self.assertEqual(sorted(geometry.objects), [0])
        np.testing.assert_allclose(geometry.objects[0].centre_mm, [11.0, 22.0, 33.0])

    def test_a_bin_wall_removes_labels_that_a_clear_table_allows(self) -> None:
        from datagen.grasps.labels import label_scene

        objects = [("proc_00000_box", (0.0, 0.0, 30.0), IDENTITY)]
        clear, _ = label_scene(_payload(objects), self.EXTENTS, "s")
        walled, _ = label_scene(
            _payload(objects, family="bin",
                     walls=[{"center_mm": [34.0, 0.0, 60.0], "half_extents_mm": [4.0, 120.0, 60.0],
                             "name": "bin_x_pos"}]),
            self.EXTENTS, "s")
        jaw_clear = sum(1 for label in clear if label.kind == "jaw")
        jaw_walled = sum(1 for label in walled if label.kind == "jaw")
        self.assertLess(jaw_walled, jaw_clear, "a wall 4 mm from the object must cost some grasps")

    def test_the_label_row_round_trips(self) -> None:
        from datagen.grasps.labels import label_scene

        labels, _ = label_scene(_payload([("proc_00000_box", (0.0, 0.0, 30.0), IDENTITY)]),
                                self.EXTENTS, "s")
        row = labels[0].as_row()
        self.assertEqual(set(row) >= {"scene_id", "instance_id", "kind", "position_mm", "approach",
                                      "closing_axis", "width_mm"}, True)
        self.assertEqual(len(row["position_mm"]), 3)


class EvaluationMathTests(unittest.TestCase):
    """``summarise`` turns rows into the numbers a decision is made on; it gets its own arithmetic."""

    def _write(self, tmp, rows) -> None:
        (tmp / "grasp_eval.jsonl").write_text(
            "\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_a_role_with_labels_and_NO_candidate_still_appears(self) -> None:
        """⛔ ABSENCE HAS TO BE READABLE, and for one day it was not.

        `by_part_role` reported only roles a candidate MATCHED, so "this corpus has no handles" and
        "this stack never grasps a handle" produced the identical table: no handle row. MEASURED
        2026-08-31, that ambiguity cost two diagnostics -- the first read the missing row as "the
        matcher rejects handles" and the second as "handle objects get no candidate", and the table
        could support neither.

        Worse, the block's own comment already promised the pair ("`labels` counts how many of that
        role were there to find") while the code counted only candidates. A described design that was
        never implemented reads exactly like one that was.
        """
        import tempfile
        from pathlib import Path

        from datagen.eval.ladder import summarise

        base = {"config": "default", "family": "packed", "view": "v", "visibility": 1.0}
        rows = [
            # A mug: 4 body labels and 3 handle labels. Two candidates, both matching the BODY.
            # ⚠ THE DISTANCE COLUMNS ARE PART OF THE FIXTURE NOW. `summarise` gates an attribution
            # at `MATCH_ACCEPT_COST` working points, re-derived from these columns, so a row with a
            # role and no distance is "never measured" rather than a match. Without them this test
            # would have gone quietly green with an EMPTY table, which is the very thing it pins.
            {**base, "scene_id": "a", "instance_id": 0, "outcome": "candidate", "valid": True,
             "reason": "none", "jaw_labels": 7, "jaw_labels_by_role": {"body": 4, "handle": 3},
             "match_part_role": "body", "match_position_mm": 2.0, "match_approach_deg": 3.0,
             "match_axis_deg": 4.0},
            {**base, "scene_id": "a", "instance_id": 0, "outcome": "candidate", "valid": False,
             "reason": "too_wide", "jaw_labels": 7, "jaw_labels_by_role": {"body": 4, "handle": 3},
             "match_part_role": "body", "match_position_mm": 5.0, "match_approach_deg": 6.0,
             "match_axis_deg": 7.0},
            # A hammer the stack refused outright. Its grip labels are exactly the ones that went
            # ungrasped, so dropping this row would hide the finding.
            {**base, "scene_id": "a", "instance_id": 1, "outcome": "no_candidate", "jaw_labels": 5,
             "jaw_labels_by_role": {"grip": 5}},
            # A handle nobody could see is not a handle the stack declined to grasp.
            {**base, "scene_id": "a", "instance_id": 2, "outcome": "not_visible", "jaw_labels": 2,
             "jaw_labels_by_role": {"handle": 2}},
        ]
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            self._write(tmp, rows)
            report = summarise(tmp)
        roles = report["by_config"]["default"]["by_part_role"]

        self.assertEqual(set(roles), {"body", "handle", "grip"},
                         "a role with labels must appear even with zero candidates")
        self.assertEqual(roles["body"]["candidates"], 2)
        self.assertEqual(roles["body"]["valid"], 1)
        self.assertEqual(roles["body"]["labels"], 4)
        # THE ONE THE TABLE EXISTS FOR: handles were there to find and nothing matched one.
        self.assertEqual(roles["handle"]["candidates"], 0)
        self.assertEqual(roles["handle"]["labels"], 3)
        self.assertEqual(roles["handle"]["precision"], 0.0)
        self.assertEqual(roles["grip"]["labels"], 5)
        self.assertEqual(roles["grip"]["objects"], 1)
        # Counted apart, not folded in.
        self.assertEqual(roles["handle"]["objects_not_visible"], 1)
        self.assertEqual(roles["handle"]["objects"], 1)

    def test_SUCTION_gets_the_same_part_table_as_the_jaw(self) -> None:
        """⛔ IT DID NOT, AND ITS EMPTY TABLE READ AS "no parts here".

        `by_part_role` was built from `jaw_labels_by_role` alone, so the suction rung reported an
        empty table BY CONSTRUCTION -- indistinguishable from a corpus with no composite in it.

        MEASURED 2026-08-31 on the `v4_s0` shard: 108 fully visible objects whose only jaw-graspable
        feature is a handle received ZERO jaw candidates. Whether a cup reaches those objects is
        exactly the follow-up question, and 34.1 % of objects are too short to jaw-grasp top-down at
        all, so the cup is not a footnote.
        """
        import tempfile
        from pathlib import Path

        from datagen.eval.ladder import summarise

        base = {"config": "suction", "family": "packed", "view": "v", "visibility": 1.0,
                "kind": "suction"}
        rows = [
            {**base, "scene_id": "a", "instance_id": 0, "outcome": "candidate", "valid": True,
             "reason": "none", "suction_labels": 6,
             "suction_labels_by_role": {"body": 4, "handle": 2}, "match_part_role": "body",
             # A cup has no closing axis, so 0.0 is the CONVENTION rather than a measurement.
             "match_position_mm": 1.0, "match_approach_deg": 2.0, "match_axis_deg": 0.0},
            {**base, "scene_id": "a", "instance_id": 1, "outcome": "no_candidate",
             "suction_labels": 3, "suction_labels_by_role": {"handle": 3}},
        ]
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            self._write(tmp, rows)
            roles = summarise(tmp)["by_config"]["suction"]["by_part_role"]
        self.assertEqual(set(roles), {"body", "handle"})
        self.assertEqual(roles["body"]["candidates"], 1)
        self.assertEqual(roles["handle"]["labels"], 5)
        self.assertEqual(roles["handle"]["candidates"], 0)

    def test_precision_and_coverage_are_what_they_say(self) -> None:
        import tempfile
        from pathlib import Path

        from datagen.eval.ladder import summarise

        base = {"config": "default", "family": "sparse", "view": "v", "outcome": "candidate",
                "jaw_labels": 5, "visibility": 1.0}
        rows = [
            {**base, "scene_id": "a", "instance_id": 0, "valid": True, "reason": "none",
             "match_position_mm": 1.0, "match_approach_deg": 2.0, "match_axis_deg": 3.0},
            {**base, "scene_id": "a", "instance_id": 0, "valid": False, "reason": "too_wide",
             "match_position_mm": 90.0, "match_approach_deg": 80.0, "match_axis_deg": 80.0},
            {**base, "scene_id": "a", "instance_id": 1, "valid": False, "reason": "too_wide",
             "match_position_mm": 90.0, "match_approach_deg": 80.0, "match_axis_deg": 80.0},
            {**base, "scene_id": "a", "instance_id": 2, "outcome": "no_candidate"},
        ]
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            self._write(tmp, rows)
            report = summarise(tmp)
        entry = report["by_config"]["default"]
        self.assertEqual(entry["candidates"], 3)
        self.assertAlmostEqual(entry["precision"], 1 / 3, places=4)
        # Three objects carry labels; exactly one produced a valid candidate.
        self.assertEqual(entry["objects_with_labels"], 3)
        self.assertAlmostEqual(entry["coverage"], 1 / 3, places=4)
        # An object the calculator said nothing about is a MISS, not an absence.
        self.assertEqual(entry["objects_no_candidate"], 1)

    def test_the_recall_curve_rises_with_the_tolerance(self) -> None:
        import tempfile
        from pathlib import Path

        from datagen.eval.ladder import summarise

        base = {"config": "default", "family": "sparse", "view": "v", "outcome": "candidate",
                "jaw_labels": 3, "visibility": 1.0, "valid": True, "reason": "none"}
        rows = [
            {**base, "scene_id": "a", "instance_id": 0, "match_position_mm": 3.0,
             "match_approach_deg": 3.0, "match_axis_deg": 3.0},
            {**base, "scene_id": "a", "instance_id": 1, "match_position_mm": 18.0,
             "match_approach_deg": 25.0, "match_axis_deg": 25.0},
        ]
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            self._write(tmp, rows)
            curve = summarise(tmp)["by_config"]["default"]["recall_curve"]
        self.assertAlmostEqual(curve["5mm/5deg"], 0.5)
        self.assertAlmostEqual(curve["20mm/30deg"], 1.0)
        self.assertLessEqual(curve["2mm/5deg"], curve["30mm/45deg"], "a curve must be monotone")

    def test_a_match_needs_ALL_THREE_coordinates(self) -> None:
        """Otherwise a candidate pointing the wrong way 'matches' by standing in the right place."""
        from datagen.eval.ladder import _best_match
        from datagen.grasps.labels import GraspLabel

        label = GraspLabel("s", 0, "jaw", (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0),
                           40.0, 0.0, 0.0)
        dp, da, dc, matched = _best_match(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]),
                                          np.array([0.0, 1.0, 0.0]), [label])
        self.assertAlmostEqual(dp, 0.0)
        self.assertAlmostEqual(da, 0.0)
        self.assertAlmostEqual(dc, 90.0, places=3)
        # ⭑ AND THE LABEL ITSELF, which the matcher used to discard. Without it nothing downstream can
        # say WHICH grasp a proposal matched, so the `part_role` the corpus carries on every composite
        # label was unreachable and the affordance head had no number at all.
        self.assertIs(matched, label)

    def test_the_matcher_returns_nothing_matched_when_there_are_no_labels(self) -> None:
        from datagen.eval.ladder import _best_match

        dp, da, dc, matched = _best_match(np.zeros(3), np.array([0.0, 0.0, -1.0]),
                                          np.array([1.0, 0.0, 0.0]), [])
        self.assertIsNone(matched)
        self.assertEqual((dp, da, dc), (float("inf"), float("inf"), float("inf")))


class CameraRoundTripTests(unittest.TestCase):
    def test_unproject_inverts_project(self) -> None:
        """The fused-cloud rung stands on this. A back-projection that disagreed with the forward one
        by a sign would build a perfectly self-consistent cloud in the wrong place."""
        from datagen.render.camera import intrinsics_matrix, look_at_camera_to_base, unproject_to_base

        intrinsics = intrinsics_matrix((640, 480), 55.0)
        camera_to_base = look_at_camera_to_base(
            np.array([300.0, 120.0, 400.0]), np.array([450.0, 0.0, 20.0]))
        depth = np.zeros((480, 640), dtype=np.float64)
        mask = np.zeros_like(depth, dtype=bool)
        wanted = np.array([[450.0, 0.0, 20.0], [500.0, 40.0, 60.0], [400.0, -30.0, 10.0]])
        pixels = project_to_pixels_for_test(wanted, camera_to_base, intrinsics)
        base_to_camera = np.linalg.inv(camera_to_base)
        for point, (u, v) in zip(wanted, pixels, strict=True):
            row, column = int(round(v)), int(round(u))
            depth[row, column] = float((base_to_camera[:3, :3] @ point + base_to_camera[:3, 3])[2])
            mask[row, column] = True
        recovered = unproject_to_base(depth, mask, camera_to_base, intrinsics)
        for point in wanted:
            self.assertLess(float(np.linalg.norm(recovered - point, axis=1).min()), 1.5,
                            f"{point} did not survive the round trip")


def project_to_pixels_for_test(points, camera_to_base, intrinsics):  # noqa: ANN001, ANN201
    from datagen.render.camera import project_to_pixels

    return project_to_pixels(points, camera_to_base, intrinsics)


class PhysicsSampleTests(unittest.TestCase):
    """The Isaac half cannot run here. What CAN be checked off-box is checked off-box."""

    def test_the_grasp_frame_is_orthonormal_and_ordered(self) -> None:
        """Columns are (closing, binormal, approach). A silent transpose here would rotate every
        trial by 90 deg and every failure would look like a bad grasp."""
        from datagen.grasps.physics import _grasp_frame

        approach = np.array([0.3, -0.2, -0.9])
        approach /= np.linalg.norm(approach)
        closing = np.array([0.9, 0.1, 0.2])
        frame = _grasp_frame(approach, closing)
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(frame)), 1.0, places=9)
        np.testing.assert_allclose(frame[:, 2], approach, atol=1e-9)
        self.assertAlmostEqual(float(frame[:, 0] @ approach), 0.0, places=9)

    def test_the_sample_is_stratified_and_reproducible(self) -> None:
        import json as _json
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        labels = [{"scene_id": "s", "instance_id": i, "kind": "jaw", "position_mm": [0, 0, 0],
                   "approach": [0, 0, -1], "closing_axis": [1, 0, 0], "width_mm": 40.0,
                   "approach_tilt_deg": 0.0, "contact_angle_deg": 0.0} for i in range(50)]
        evaluated = [{"scene_id": "s", "instance_id": i % 7, "kind": "jaw", "outcome": "candidate",
                      "valid": i % 2 == 0, "reason": "none" if i % 2 == 0 else "not_antipodal",
                      "family": "bin", "config": "default", "position_mm": [1.0, 2.0, 3.0],
                      "approach": [0, 0, -1], "closing_axis": [1, 0, 0],
                      "commanded_width_mm": 30.0} for i in range(60)]
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "grasps.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in labels) + "\n", encoding="utf-8")
            (tmp / "grasp_eval.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in evaluated) + "\n", encoding="utf-8")
            first = sample_trials(tmp, per_class=10)
            second = sample_trials(tmp, per_class=10)
        sources = {trial.source for trial in first}
        # The rejected side is the half that makes this a validation rather than a self-check.
        self.assertIn("label", sources)
        self.assertIn("valid", sources)
        self.assertIn("rejected:not_antipodal", sources)
        self.assertEqual([t.as_row() for t in first], [t.as_row() for t in second],
                         "the sample is part of the measurement, so it has to be reproducible")

    def _two_rung_root(self, directory):
        """A corpus whose eval file mixes two rungs, one of them four times the size of the other."""
        import json as _json

        labels = [{"scene_id": "s", "instance_id": i, "kind": "jaw", "position_mm": [0, 0, 0],
                   "approach": [0, 0, -1], "closing_axis": [1, 0, 0], "width_mm": 40.0,
                   "approach_tilt_deg": 0.0, "contact_angle_deg": 0.0} for i in range(20)]
        evaluated = []
        for rung, count in (("sfe_fused", 10), ("deep", 40)):
            for i in range(count):
                evaluated.append({
                    "scene_id": "s", "instance_id": i % 5, "kind": "jaw", "outcome": "candidate",
                    "valid": i % 2 == 0, "reason": "none" if i % 2 == 0 else "not_antipodal",
                    "family": "bin", "config": rung,
                    # Distinct poses, or the identity dedupe collapses the two rungs into one.
                    "position_mm": [float(i), 2.0, 3.0 if rung == "deep" else 4.0],
                    "approach": [0, 0, -1], "closing_axis": [1, 0, 0],
                    "commanded_width_mm": 30.0})
        (directory / "grasps.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in labels) + "\n", encoding="utf-8")
        (directory / "grasp_eval.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in evaluated) + "\n", encoding="utf-8")
        return directory

    def _corpus_with_floors(self, directory):
        """An eval file that mixes an analytic rung with the ladder's FLOOR rungs."""
        import json as _json

        labels = [{"scene_id": "s", "instance_id": i, "kind": "jaw", "position_mm": [0, 0, 0],
                   "approach": [0, 0, -1], "closing_axis": [1, 0, 0], "width_mm": 40.0,
                   "approach_tilt_deg": 0.0, "contact_angle_deg": 0.0} for i in range(10)]
        evaluated = []
        # The floors outnumber the analytic rung four to one, which is the real ratio: they propose
        # 12 poses per object-view where `sfe_fused` proposes a handful.
        # ⚠ A DISTINCT, DETERMINISTIC z PER RUNG. The first version separated them by `hash(rung) % 7`
        # -- and `hash()` on a str is randomised per process, so the rungs collided at some seeds and
        # the identity dedupe silently swallowed one. A fixture that depends on PYTHONHASHSEED is the
        # "passes by luck" defect this file already caught once elsewhere.
        for offset, (rung, count) in enumerate((("sfe_fused", 10), ("floor_random", 40),
                                                ("floor_topdown", 40))):
            for i in range(count):
                evaluated.append({
                    "scene_id": "s", "instance_id": i % 5, "kind": "jaw", "outcome": "candidate",
                    "valid": i % 2 == 0, "reason": "none" if i % 2 == 0 else "not_antipodal",
                    "family": "bin", "config": rung,
                    "position_mm": [float(i), 2.0, 3.0 + 50.0 * offset],
                    "approach": [0, 0, -1], "closing_axis": [1, 0, 0],
                    "commanded_width_mm": 30.0})
        (directory / "grasps.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in labels) + "\n", encoding="utf-8")
        (directory / "grasp_eval.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in evaluated) + "\n", encoding="utf-8")
        return directory

    def test_the_LADDER_FLOORS_never_reach_the_default_physics_draw(self) -> None:
        """They exist to give the ladder a SCALE, not to be shaken. They propose 12 poses per
        object-view against the analytic rungs' handful, so an unrestricted draw would spend most of
        an Isaac night on deliberate nonsense -- and the `held` labels the ranker trains on would come
        from random poses. Proven byte-identical against HEAD on v1_proof (192 trials) as well: no
        corpus written before the floors existed carries such a row, so nothing already measured moves.
        """
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._corpus_with_floors(Path(name))
            drawn = sample_trials(root, per_class=6, seed=3)
        configs = {t.config for t in drawn}
        self.assertNotIn("floor_random", configs)
        self.assertNotIn("floor_topdown", configs)
        self.assertIn("sfe_fused", configs)

    def test_a_floor_can_still_be_asked_for_BY_NAME(self) -> None:
        """Excluding them by default must not make them unreachable -- shaking a floor is exactly how
        one would establish what the physics anchor's own floor is."""
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._corpus_with_floors(Path(name))
            drawn = sample_trials(root, per_class=6, seed=3, configs=("floor_topdown",))
        self.assertEqual({t.config for t in drawn}, {"floor_topdown"})

    def test_without_configs_the_draw_is_exactly_what_it_always_was(self) -> None:
        """PROVEN against HEAD on the real v1_proof corpus too: 192 trials, byte-identical. This is
        the cheap version that keeps it that way, because the default is a LOCKED measurement."""
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._two_rung_root(Path(name))
            drawn = sample_trials(root, per_class=4, seed=1)
        # The analytic label pool is drawn, which is the half a restricted draw gives up.
        self.assertIn("label", {t.source for t in drawn})
        self.assertIn("", {t.config for t in drawn})
        # And the rung is NOT part of the stratum key: both rungs sit in one (source, family) pool,
        # so which of them a stratum happens to yield is the luck of the draw. Asserting that both
        # appear would be asserting the opposite of the property -- measured, at per_class=4 there is
        # a ~38 % chance per stratum of drawing no `sfe_fused` at all, and seed 1 does exactly that.
        self.assertTrue({t.config for t in drawn} & {"sfe_fused", "deep"})

    def test_configs_RESTRICTS_to_those_rungs_and_gives_each_its_own_strata(self) -> None:
        """`grasp_eval.jsonl` mixes every rung. An unrestricted draw is dominated by whichever rung
        produced the most rows -- here `deep` outnumbers `sfe_fused` four to one, so a comparison
        without this would be a weighting rather than a comparison."""
        import tempfile
        from collections import Counter
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._two_rung_root(Path(name))
            unrestricted = sample_trials(root, per_class=4, seed=1)
            drawn = sample_trials(root, per_class=4, seed=1, configs=("sfe_fused", "deep"))
        self.assertEqual({t.config for t in drawn}, {"sfe_fused", "deep"})
        counts = Counter(t.config for t in drawn)
        self.assertEqual(counts["sfe_fused"], counts["deep"],
                         "each rung gets its own strata, so equal availability draws equally")
        # The point of the flag: without it the four-to-one row imbalance survives into the draw.
        loose = Counter(t.config for t in unrestricted)
        self.assertGreater(loose["deep"], loose["sfe_fused"])

    def test_restricting_by_rung_DROPS_the_analytic_label_pool(self) -> None:
        """Those rows carry no rung, so they would arrive as `config=""` and add a third arm to a
        two-arm comparison. Measured before the code did this: 136 trials of which 32 were labels."""
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._two_rung_root(Path(name))
            drawn = sample_trials(root, per_class=4, seed=1, configs=("deep",))
        self.assertNotIn("label", {t.source for t in drawn})
        self.assertEqual({t.config for t in drawn}, {"deep"})

    def test_an_unknown_rung_draws_nothing_rather_than_falling_back(self) -> None:
        """A typo in a rung name must not quietly return the whole corpus under that name."""
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._two_rung_root(Path(name))
            self.assertEqual(sample_trials(root, per_class=4, configs=("typo",)), [])

    def _paired_root(self, directory, missing_arm_objects: int = 2):
        """Two arms proposing on the same objects, plus a few where only one arm proposed."""
        import json as _json

        rows = []
        # 10 objects both arms reached. `sfe_fused` ranks its best candidate LAST in file order, so a
        # test that takes the first row instead of rank 0 fails.
        for obj in range(10):
            for arm, base_z in (("sfe_fused", 3.0), ("deep", 60.0)):
                for rank in (2, 1, 0):
                    rows.append({
                        "scene_id": "bin_000001", "instance_id": obj, "kind": "jaw",
                        "outcome": "candidate", "rank": rank,
                        # Our own verdict says NO for every rank-0 pick -- the anchor must draw them
                        # anyway, or it grades the mimicry twice.
                        "valid": rank != 0, "reason": "none" if rank != 0 else "not_antipodal",
                        "family": "bin", "config": arm,
                        "position_mm": [float(obj), float(rank), base_z],
                        "approach": [0, 0, -1], "closing_axis": [1, 0, 0],
                        "commanded_width_mm": 30.0})
        # objects only `deep` proposed on
        for obj in range(100, 100 + missing_arm_objects):
            rows.append({
                "scene_id": "bin_000001", "instance_id": obj, "kind": "jaw",
                "outcome": "candidate", "rank": 0, "valid": True, "reason": "none",
                "family": "bin", "config": "deep",
                "position_mm": [float(obj), 0.0, 90.0],
                "approach": [0, 0, -1], "closing_axis": [1, 0, 0], "commanded_width_mm": 30.0})
        (directory / "grasps.jsonl").write_text("", encoding="utf-8")
        (directory / "grasp_eval.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return directory

    def test_a_PAIRED_draw_puts_every_object_in_EVERY_arm(self) -> None:
        """The comparison `sample_trials` cannot make. MEASURED on v1_proof at per_class=8:
        stratified drew 153 vs 104 trials over 113 vs 74 objects sharing only ELEVEN of them, so the
        two arms were compared on almost disjoint samples and object difficulty sat inside the
        result. Paired: 32 vs 32 over the same 32 objects."""
        import tempfile
        from collections import Counter
        from pathlib import Path

        from datagen.grasps.physics import paired_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._paired_root(Path(name))
            drawn = paired_trials(root, configs=("sfe_fused", "deep"), per_class=10, seed=1)
        counts = Counter(t.config for t in drawn)
        self.assertEqual(counts["sfe_fused"], counts["deep"])
        per_arm = {arm: {(t.scene_id, t.instance_id) for t in drawn if t.config == arm}
                   for arm in ("sfe_fused", "deep")}
        self.assertEqual(per_arm["sfe_fused"], per_arm["deep"])
        self.assertTrue(per_arm["deep"])

    def test_pick_rank_takes_the_GENERATORS_OWN_first_choice(self) -> None:
        """Promotion asks "when this thing picks, does its pick hold" -- so the trial has to be the
        pick, not a candidate somewhere down its list. The fixture writes rank 0 LAST, so taking the
        first row in file order fails here."""
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import paired_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._paired_root(Path(name))
            drawn = paired_trials(root, configs=("sfe_fused", "deep"), per_class=10, seed=1)
        # rank 0 rows are the ones with y == 0 in the fixture's position.
        self.assertTrue(all(t.position_mm[1] == 0.0 for t in drawn),
                        "a trial came from a candidate the generator did not choose")

    def test_the_anchor_draws_a_pick_OUR_VERDICT_REJECTED(self) -> None:
        """THE WHOLE POINT OF THE ANCHOR. A generator trained on labels derived from our analytic
        verdict is partly graded on mimicry; pre-filtering its candidates by that same verdict would
        grade the mimicry twice. Every rank-0 pick in the fixture is one we call `not_antipodal`."""
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import paired_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._paired_root(Path(name))
            drawn = paired_trials(root, configs=("sfe_fused", "deep"), per_class=10, seed=1)
        self.assertTrue(drawn)
        self.assertTrue(all(t.source.startswith("rejected") for t in drawn),
                        "the anchor filtered by the referee it exists to check")

    def test_an_object_only_ONE_arm_reached_is_excluded_by_default(self) -> None:
        """A missing arm is a REFUSAL, not a tie, and pairing it against nothing is not a comparison.
        The refusals still matter -- they are logged, because "held 60 % of what it proposed" says
        nothing without "and it proposed on 40 % fewer objects"."""
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import paired_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._paired_root(Path(name), missing_arm_objects=3)
            strict = paired_trials(root, configs=("sfe_fused", "deep"), per_class=50, seed=1)
            loose = paired_trials(root, configs=("sfe_fused", "deep"), per_class=50, seed=1,
                                  require_all=False)
        strict_objects = {(t.scene_id, t.instance_id) for t in strict}
        loose_objects = {(t.scene_id, t.instance_id) for t in loose}
        self.assertEqual(len(strict_objects), 10)
        self.assertEqual(len(loose_objects), 13)
        self.assertNotIn(("bin_000001", 100), strict_objects)
        self.assertIn(("bin_000001", 100), loose_objects)

    def test_the_paired_draw_is_reproducible(self) -> None:
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import paired_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._paired_root(Path(name))
            first = paired_trials(root, configs=("sfe_fused", "deep"), per_class=4, seed=11)
            second = paired_trials(root, configs=("sfe_fused", "deep"), per_class=4, seed=11)
        self.assertEqual([t.as_row() for t in first], [t.as_row() for t in second])

    def test_it_refuses_a_draw_that_cannot_be_paired(self) -> None:
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import paired_trials

        with tempfile.TemporaryDirectory() as name:
            root = self._paired_root(Path(name))
            with self.assertRaises(ValueError):
                paired_trials(root, configs=("sfe_fused",), per_class=4)
            with self.assertRaises(ValueError):
                paired_trials(root, configs=("sfe_fused", "deep"), pick="whatever")

    def test_every_trial_names_the_file_and_line_it_came_from(self) -> None:
        """The join key back to the candidate's features, and it has to be an INDEX.

        Measured on the v1_proof corpus 2026-08-24: joining a physics result to its candidate on
        (scene, instance, position, approach, closing, width) instead would be ambiguous for **53.6 %**
        of 148,216 candidates, because a `label` and the `valid` row describing the same grasp carry
        the same pose. A float-equality join does not fail loudly; it drops what it cannot match and
        reports a smaller dataset.
        """
        import json as _json
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        labels = [{"scene_id": f"bin_{i:03d}", "instance_id": i, "kind": "jaw",
                   "position_mm": [float(i), 0.0, 0.0], "approach": [0, 0, -1],
                   "closing_axis": [1, 0, 0], "width_mm": 40.0} for i in range(12)]
        # Deliberately the SAME poses as the labels: this is the case the pose join cannot separate.
        evaluated = [{"scene_id": f"bin_{i:03d}", "instance_id": i, "kind": "jaw",
                      "outcome": "candidate", "valid": True, "reason": "none", "family": "bin",
                      "position_mm": [float(i), 0.0, 0.0], "approach": [0, 0, -1],
                      "closing_axis": [1, 0, 0], "commanded_width_mm": 40.0} for i in range(12)]
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "grasps.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in labels), encoding="utf-8")
            (tmp / "grasp_eval.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in evaluated), encoding="utf-8")
            trials = sample_trials(tmp, per_class=100)
            lines = {
                "grasps": (tmp / "grasps.jsonl").read_text(encoding="utf-8").splitlines(),
                "grasp_eval": (tmp / "grasp_eval.jsonl").read_text(encoding="utf-8").splitlines(),
            }

        self.assertEqual(len(trials), 24, "both files, every row")
        keys = [(t.origin, t.row_index) for t in trials]
        self.assertEqual(len(set(keys)), len(keys), "the join key has to be unique")
        poses = {(t.scene_id, t.instance_id, t.position_mm, t.approach) for t in trials}
        self.assertEqual(len(poses), 12, "the poses collide by construction; the key must not")
        for trial in trials:
            row = _json.loads(lines[trial.origin][trial.row_index])
            self.assertEqual(row["scene_id"], trial.scene_id)
            self.assertEqual(row["instance_id"], trial.instance_id)
            self.assertEqual(tuple(row["position_mm"]), trial.position_mm)

    def test_the_key_survives_a_round_trip_through_the_result_row(self) -> None:
        """`as_row` is what the shake writes. A key that does not reach the file is not a key."""
        from datagen.grasps.physics import PhysicsTrial

        row = PhysicsTrial("bin_001", 3, "valid", (1.0, 2.0, 3.0), (0.0, 0.0, -1.0),
                           (1.0, 0.0, 0.0), 40.0, origin="grasp_eval", row_index=17).as_row()
        self.assertEqual(row["origin"], "grasp_eval")
        self.assertEqual(row["row_index"], 17)

    def test_the_draw_is_stratified_by_family_not_only_by_source(self) -> None:
        """The docstring claimed this from the start; the code keyed its pools by source alone.

        `family` was carried on every trial and never read, so the draw was a statement about whichever
        family has the most rows -- in the real corpus `bin`, which is also the hardest family by every
        measure this project has taken.
        """
        import json as _json
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        # One family swamps the other 20:1. Stratified, the draw takes 3 of each anyway.
        evaluated = (
            [{"scene_id": f"bin_{i:04d}", "instance_id": 0, "kind": "jaw", "outcome": "candidate",
              "valid": True, "reason": "none", "family": "bin", "position_mm": [float(i), 0.0, 0.0],
              "approach": [0, 0, -1], "closing_axis": [1, 0, 0], "commanded_width_mm": 40.0}
             for i in range(200)]
            + [{"scene_id": f"pile_{i:04d}", "instance_id": 0, "kind": "jaw",
                "outcome": "candidate", "valid": True, "reason": "none", "family": "pile",
                "position_mm": [0.0, float(i), 0.0], "approach": [0, 0, -1],
                "closing_axis": [1, 0, 0], "commanded_width_mm": 40.0} for i in range(10)])
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "grasps.jsonl").write_text("", encoding="utf-8")
            (tmp / "grasp_eval.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in evaluated), encoding="utf-8")
            trials = sample_trials(tmp, per_class=3)

        drawn = collections.Counter(t.family for t in trials)
        self.assertEqual(dict(drawn), {"bin": 3, "pile": 3},
                         "per_class is per stratum, so a 20:1 imbalance draws evenly")

    def test_a_repeated_candidate_is_drawn_once_and_it_is_the_first_copy(self) -> None:
        """`grasp_eval.jsonl` is APPENDED across runs, so it holds each candidate several times.

        Measured on v1_proof: 130,382 jaw candidate rows are only 53,212 distinct grasps (40.8 %). A
        draw over LINES therefore costs twice -- it weights a grasp by how often someone re-evaluated
        it, and only the FIRST copy has a feature row in the corpus, so 41.6 % of a run's labels could
        not be joined to anything. After deduping, 14,700 of 14,700 drawn trials were joinable.
        """
        import json as _json
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        candidate = {"scene_id": "bin_0001", "instance_id": 0, "kind": "jaw", "view": "cam0",
                     "outcome": "candidate", "valid": True, "reason": "none", "family": "bin",
                     "position_mm": [10.0, 20.0, 30.0], "approach": [0, 0, -1],
                     "closing_axis": [1, 0, 0], "commanded_width_mm": 40.0}
        other = {**candidate, "position_mm": [90.0, 20.0, 30.0]}
        # The same grasp four times, then a different one, then the first grasp again.
        rows = [candidate, candidate, candidate, candidate, other, candidate]
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "grasps.jsonl").write_text("", encoding="utf-8")
            (tmp / "grasp_eval.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
            trials = sample_trials(tmp, per_class=100)

        self.assertEqual(len(trials), 2, "six rows, two grasps")
        self.assertEqual(sorted(t.row_index for t in trials), [0, 4],
                         "the FIRST copy of each, which is the row the corpus keeps")

    def test_the_identity_is_claimed_before_the_reason_filter(self) -> None:
        """Two eval configs can disagree about WHY the same grasp is invalid. Measured: 2 in 14,700.

        The corpus dedupes without knowing about `reasons`, so its survivor is the first candidate row
        full stop. Filtering first made this sampler skip that row and draw a later copy of the same
        grasp -- which then had no feature row to join to.
        """
        import json as _json
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        base = {"scene_id": "bin_0001", "instance_id": 0, "kind": "jaw", "view": "cam0",
                "outcome": "candidate", "valid": False, "family": "bin",
                "position_mm": [10.0, 20.0, 30.0], "approach": [0, 0, -1],
                "closing_axis": [1, 0, 0], "commanded_width_mm": 40.0}
        rows = [{**base, "reason": "approach_blocked"},      # not in `reasons` -- but IS the corpus's
                {**base, "reason": "not_antipodal"}]         # the copy the old code would have drawn
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "grasps.jsonl").write_text("", encoding="utf-8")
            (tmp / "grasp_eval.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
            trials = sample_trials(tmp, per_class=100)

        self.assertEqual(trials, [], "the identity was claimed by row 0, which no reason admits")

    def test_a_label_and_a_candidate_with_the_same_pose_are_both_drawn(self) -> None:
        """They live in different files with different line spaces, so they are different trials.

        Today they cannot collide anyway -- `grasp_identity` carries the view and analytic labels have
        none -- but the dedupe is written per file so that stays true if labels ever gain one.
        """
        import json as _json
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        pose = {"position_mm": [10.0, 20.0, 30.0], "approach": [0, 0, -1], "closing_axis": [1, 0, 0]}
        label = {"scene_id": "bin_0001", "instance_id": 0, "kind": "jaw", "view": "cam0",
                 "width_mm": 40.0, **pose}
        candidate = {"scene_id": "bin_0001", "instance_id": 0, "kind": "jaw", "view": "cam0",
                     "outcome": "candidate", "valid": True, "reason": "none", "family": "bin",
                     "commanded_width_mm": 40.0, **pose}
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "grasps.jsonl").write_text(_json.dumps(label), encoding="utf-8")
            (tmp / "grasp_eval.jsonl").write_text(_json.dumps(candidate), encoding="utf-8")
            trials = sample_trials(tmp, per_class=100)

        self.assertEqual({t.source for t in trials}, {"label", "valid"})
        self.assertEqual({t.origin for t in trials}, {"grasps", "grasp_eval"})

    def test_rows_without_the_candidate_vectors_are_skipped_not_faked(self) -> None:
        """Older eval files carry no geometry to replay. Replaying a zero vector would be worse."""
        import json as _json
        import tempfile
        from pathlib import Path

        from datagen.grasps.physics import sample_trials

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "grasps.jsonl").write_text("", encoding="utf-8")
            (tmp / "grasp_eval.jsonl").write_text(_json.dumps(
                {"scene_id": "s", "instance_id": 0, "kind": "jaw", "outcome": "candidate",
                 "valid": True, "reason": "none", "family": "bin"}) + "\n", encoding="utf-8")
            self.assertEqual(sample_trials(tmp, per_class=5), [])


class PredictedMaskTests(unittest.TestCase):
    """The second mask source. Everything here is arithmetic; the models never load."""

    def test_iou_is_intersection_over_union_and_survives_empties(self) -> None:
        from datagen.grasps.masks import _iou

        a = np.zeros((10, 10), dtype=bool)
        a[2:6, 2:6] = True                       # 16 px
        b = np.zeros((10, 10), dtype=bool)
        b[4:8, 4:8] = True                       # 16 px, overlapping in a 2x2 corner
        self.assertAlmostEqual(_iou(a, b), 4 / 28, places=6)
        self.assertEqual(_iou(a, a), 1.0)
        self.assertEqual(_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)), 0.0,
                         "two empty masks are not a perfect match; they are no measurement")

    def test_a_detection_below_the_floor_is_a_false_positive_not_a_forced_match(self) -> None:
        """A box that grazes an object must not be credited with having found it."""
        from datagen.grasps.masks import MIN_MATCH_IOU, _match

        truth = np.zeros((20, 20), dtype=bool)
        truth[0:10, 0:10] = True
        graze = np.zeros((20, 20), dtype=bool)
        graze[9:11, 9:11] = True
        self.assertLess(_iou_of(graze, truth), MIN_MATCH_IOU)
        self.assertEqual(_match([graze], {3: truth}), {})

    def test_each_object_is_claimed_once_and_by_its_best_detection(self) -> None:
        """Two boxes on one object is the common clutter failure; the better one wins, once."""
        from datagen.grasps.masks import _match

        truth = np.zeros((20, 20), dtype=bool)
        truth[0:10, 0:10] = True
        good = truth.copy()
        good[0, 0] = False
        rough = np.zeros((20, 20), dtype=bool)
        rough[0:7, 0:7] = True
        assignment = _match([rough, good], {5: truth})
        self.assertEqual(assignment, {1: 5}, "the closer mask takes the object")

    def test_kinds_come_off_the_asset_ids(self) -> None:
        from datagen.grasps.masks import _kinds_in_scene

        payload = {"spec": {"objects": [{"asset_id": "proc_00000_flange"},
                                        {"asset_id": "proc_00013_can"},
                                        {"asset_id": "proc_00007_can"}]}}
        self.assertEqual(_kinds_in_scene(payload), ["can", "flange"])

    def test_summary_recall_counts_missed_objects_not_just_found_ones(self) -> None:
        """The denominator is every visible object, so a detector that finds one of five scores 20%."""
        import json as _json
        import tempfile
        from pathlib import Path

        from datagen.grasps.masks import summarise_masks

        rows = [{"scene_id": "s", "view": "v", "instance_id": i, "found": i == 0,
                 "iou": 0.9 if i == 0 else 0.0, "true_px": 100, "pred_px": 90 if i == 0 else 0,
                 "visibility": 1.0, "family": "bin"} for i in range(5)]
        rows.append({"scene_id": "s", "view": "v", "instance_id": -1, "detections": 9,
                     "segmented": 9, "matched": 1, "objects": 5, "prompt": "box.",
                     "detect_ms": 1.0, "segment_ms": 1.0})
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "masks_pred.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            report = summarise_masks(tmp)
        self.assertEqual(report["objects"], 5)
        self.assertEqual(report["recall"], 0.2)
        self.assertEqual(report["iou"]["median"], 0.9,
                         "IoU is reported over the FOUND objects; a miss has no mask to score")
        self.assertEqual(report["unmatched_detections_per_view"], 8.0,
                         "eight boxes matched nothing, and each one is an object a pick loop would try")


def _iou_of(a: np.ndarray, b: np.ndarray) -> float:
    from datagen.grasps.masks import _iou

    return _iou(a, b)


if __name__ == "__main__":
    unittest.main()


class TheMatchGateTests(unittest.TestCase):
    """A part role may only be attributed when the candidate is CLOSE to that label.

    ⛔ THERE WAS NO THRESHOLD. `_best_match` returns the argmin however far away it is, so every
    candidate on a labelled object got the nearest label's role. MEASURED 2026-08-31 on
    `logs/dl/partbase`: of 454 jaw candidates 106 carry a match, cost min 1.82, median 5.14, max
    14.71 working points, and NOT ONE is inside 1.0. The 27 rows behind `body precision 0.6667` sit
    2.27 to 7.35 out, so that number described which label was nearest, not which part was grasped.

    ⚠ THE GATE IS IN `summarise`, NOT IN THE ROW WRITER, and three skeptics had to point that out. A
    writer that stripped the role would leave `by_part_role_unattributed` structurally 0 on every
    fresh file, because the report skips an empty role before it ever reaches the cost test. The
    counter could then only count rows an older writer produced. That is the inert-switch shape, and
    the test below is differential rather than a read-back precisely because of it.
    """

    def _report(self, rows: list[dict], **kwargs) -> dict:
        import json
        import tempfile
        from pathlib import Path

        from datagen.eval.ladder import summarise

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            payload = chr(10).join(json.dumps(row) for row in rows) + chr(10)
            (tmp / "grasp_eval.jsonl").write_text(payload, encoding="utf-8")
            return summarise(tmp, **kwargs)["by_config"]["default"]

    def _row(self, **over) -> dict:
        base = {"config": "default", "family": "packed", "view": "v", "visibility": 1.0,
                "kind": "jaw", "scene_id": "a", "instance_id": 0, "outcome": "candidate",
                "valid": True, "reason": "none", "jaw_labels": 3,
                "jaw_labels_by_role": {"handle": 3}, "match_part_role": "handle"}
        return {**base, **over}

    def test_a_CLOSE_match_is_attributed(self) -> None:
        entry = self._report([self._row(match_position_mm=4.0, match_approach_deg=8.0,
                                        match_axis_deg=6.0)])
        self.assertEqual(entry["by_part_role"]["handle"]["candidates"], 1)
        self.assertEqual(entry["by_part_role_unattributed"], {"too_far": 0, "never_measured": 0})

    def test_a_FAR_match_is_counted_APART_rather_than_attributed(self) -> None:
        """⭑ THE ONE. 30 mm is 3.0 working points, which is inside the range every real candidate
        in the artefacts occupies."""
        entry = self._report([self._row(match_position_mm=30.0, match_approach_deg=8.0,
                                        match_axis_deg=6.0)])
        self.assertEqual(entry["by_part_role"]["handle"]["candidates"], 0)
        self.assertEqual(entry["by_part_role_unattributed"]["too_far"], 1)
        # The role still appears, with its denominator, or absence is unreadable again.
        self.assertEqual(entry["by_part_role"]["handle"]["labels"], 3)

    def test_a_row_with_NO_distance_is_never_measured_and_NOT_too_far(self) -> None:
        """Three absences, and collapsing them is the defect. A file older than the distance columns
        has not been measured; saying "too far" about it would be an invented result."""
        entry = self._report([self._row()])
        self.assertEqual(entry["by_part_role_unattributed"],
                         {"too_far": 0, "never_measured": 1})

    def test_the_WORST_of_the_three_decides_it(self) -> None:
        """Close in position and wrong in approach is not a match, which is what the max is for."""
        entry = self._report([self._row(match_position_mm=1.0, match_approach_deg=90.0,
                                        match_axis_deg=1.0)])
        self.assertEqual(entry["by_part_role_unattributed"]["too_far"], 1)

    def test_the_tolerance_is_a_KNOB_and_the_report_stamps_which_one_it_used(self) -> None:
        """The report is a pure function of the rows, so a looser reading costs a re-summarise."""
        rows = [self._row(match_position_mm=25.0, match_approach_deg=8.0, match_axis_deg=6.0)]
        tight = self._report(rows)
        loose = self._report(rows, match_cost=3.0)
        self.assertEqual(tight["by_part_role"]["handle"]["candidates"], 0)
        self.assertEqual(loose["by_part_role"]["handle"]["candidates"], 1)
        self.assertEqual(tight["match_accept_cost"], 1.0)
        self.assertEqual(loose["match_accept_cost"], 3.0)


class TheRowSchemaTests(unittest.TestCase):
    """A row older than a field is not a row that means zero.

    ⛔ MEASURED 2026-08-31: `summarise` read `row.get("jaw_labels_by_role") or {}`, so all 1,991 rows
    of `logs/dl/partbase` -- written before the field existed -- produced a committed report saying
    "body labels 0" beside "precision 0.6667". The absent field and a field meaning zero were the same
    thing, which is the defect the field was added to fix, inverted.

    ⚠ AND A MIXTURE MUST REFUSE, not average. Rows judged against the primary solid alone and rows
    judged against every part are two different experiments. This repo already records what pooling
    cost once: a directory reporting 60.33 % where the run that produced it measured 59.28 %.
    """

    def _report(self, rows: list[dict], **kwargs) -> dict:
        import json
        import tempfile
        from pathlib import Path

        from datagen.eval.ladder import summarise

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            payload = chr(10).join(json.dumps(row) for row in rows) + chr(10)
            (tmp / "grasp_eval.jsonl").write_text(payload, encoding="utf-8")
            return summarise(tmp, **kwargs)

    def _row(self, **over) -> dict:
        base = {"config": "default", "family": "packed", "view": "v", "visibility": 1.0,
                "kind": "jaw", "scene_id": "a", "instance_id": 0, "outcome": "candidate",
                "valid": True, "reason": "none", "jaw_labels": 0}
        return {**base, **over}

    def test_a_CURRENT_row_reports(self) -> None:
        from datagen.eval.ladder import EVAL_ROW_SCHEMA

        entry = self._report([self._row(row_schema=EVAL_ROW_SCHEMA)])["by_config"]["default"]
        self.assertEqual(entry["by_part_role_status"], "reported")
        self.assertEqual(entry["row_schema"], EVAL_ROW_SCHEMA)

    def test_a_row_with_NO_stamp_is_treated_as_the_OLDEST_generation(self) -> None:
        """The conservative default: an unstamped row predates the stamp, so it predates everything."""
        entry = self._report([self._row()])["by_config"]["default"]
        self.assertEqual(entry["row_schema"], 1)
        self.assertIn("unreadable", entry["by_part_role_status"])

    def test_a_row_that_predates_the_PART_VERDICT_says_so(self) -> None:
        entry = self._report([self._row(row_schema=2)])["by_config"]["default"]
        self.assertIn("part-aware verdict", entry["by_part_role_status"])

    def test_a_MIXTURE_of_generations_REFUSES(self) -> None:
        """⭑ THE ONE. Two verdict rules in one config key is two experiments reported as one."""
        with self.assertRaises(ValueError) as caught:
            self._report([self._row(row_schema=1), self._row(row_schema=3, instance_id=1)])
        message = str(caught.exception)
        self.assertIn("two experiments", message)
        self.assertIn("reverdict", message)

    def test_the_writer_stamps_what_the_reader_expects(self) -> None:
        """A stamp the reader does not know about is worse than none: it would read as the oldest."""
        import inspect

        from datagen.eval import ladder as evaluate

        source = inspect.getsource(evaluate._evaluate_view)  # noqa: SLF001 - the stamp IS the subject
        self.assertEqual(source.count('"row_schema": EVAL_ROW_SCHEMA'), 2,
                         "both the jaw and the suction base dict must carry the stamp")


class TheReportNamesItsUnitTests(unittest.TestCase):
    """Every `objects_*` key counts an object-VIEW, and for months the name did not say so.

    ⛔ MEASURED on `logs/dl/partbase`: `objects_with_labels` reads **49** while an independent join
    over the same files counts **19** physical objects, because a scene is rendered from three cameras
    and one object contributes an entry per view. Both numbers were right; they were compared as
    though they measured the same thing, and that cost an evening.

    ⚠ THE FIX IS NOT A CONVERSION. A view is the correct unit for coverage: a pick happens from one
    camera, and an object the stack can take from one angle and not another really is two different
    situations. Silently reinterpreting the per-view keys would move every number this project has
    ever reported. The physical count is reported BESIDE them, and the report says which is which.
    """

    def _report(self, rows: list[dict]) -> dict:
        import json
        import tempfile
        from pathlib import Path

        from datagen.eval.ladder import summarise

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            payload = chr(10).join(json.dumps(row) for row in rows) + chr(10)
            (tmp / "grasp_eval.jsonl").write_text(payload, encoding="utf-8")
            return summarise(tmp)["by_config"]["default"]

    def test_one_object_in_three_views_counts_three_times_and_once(self) -> None:
        base = {"config": "default", "family": "packed", "visibility": 1.0, "kind": "jaw",
                "scene_id": "a", "instance_id": 0, "outcome": "candidate", "valid": True,
                "reason": "none", "jaw_labels": 4, "row_schema": 3}
        entry = self._report([{**base, "view": v} for v in ("overhead", "left", "right")])
        self.assertEqual(entry["objects_with_labels"], 3, "the per-view unit must be unchanged")
        self.assertEqual(entry["distinct_objects_with_labels"], 1)
        self.assertEqual(entry["views_per_object"], 3.0)

    def test_the_report_says_which_key_counts_what(self) -> None:
        """A number whose unit is only in a docstring is a number that gets compared to the wrong
        thing, which is exactly what happened."""
        base = {"config": "default", "family": "packed", "visibility": 1.0, "kind": "jaw",
                "scene_id": "a", "instance_id": 0, "outcome": "candidate", "valid": True,
                "reason": "none", "jaw_labels": 1, "view": "v", "row_schema": 3}
        entry = self._report([base])
        self.assertIn("OBJECT-VIEWS", entry["unit_note"])
        self.assertIn("distinct_objects_", entry["unit_note"])
