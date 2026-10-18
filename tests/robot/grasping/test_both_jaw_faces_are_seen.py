"""Were both faces a parallel jaw closes on measured? What a wrist look answers before it stops.

One depth view sees one side of a part, and a jaw closes on two. A grasp the generator placed on a
cloud that holds only the near face has inferred the far one from the silhouette. ``jaw_faces_seen``
asks, per side, whether enough of the patch the pad would touch was measured, so the pick loop can tell
a grasp it saw both contacts of from one it guessed the second contact of.

The scenes are analytic: a box on a table at z = 0, sampled face by face on a 1 mm grid with half a
millimetre of depth noise, and a camera sees exactly the faces that turn toward it, which is exact for
a convex box. The hands are the registry's own numbers (config/grippers), so a re-measured hand moves
these tests with it rather than leaving them asserting a hand nobody has. Where a verdict turns on how
densely and how noisily a face is sampled, the scene says so: 0.7 mm is about the footprint of a D415
pixel on a face seen at 45 degrees from 0.45 m, and a millimetre of noise about its depth error there.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.config.grippers import load_gripper
from src.robot.grasping import scene
from src.robot.grasping.multiview.unseen_side import (
    FACE_BAND_MM,
    FACE_CELL_MIN_POINTS,
    FACE_CELL_MM,
    FACE_MIN_CELLS,
    FACE_MIN_COVERAGE,
    SUPPORT_MARGIN_MM,
    JawFacesSeen,
    jaw_faces_seen,
)

#: The part stands here, and the camera looks at it from this far, about where a wrist D415 works.
_CENTRE_XY = (500.0, 0.0)
_RANGE_MM = 450.0


def _jaw(model: str) -> dict[str, float]:
    jaw = load_gripper(model).jaw
    return {
        "pad_ahead_mm": float(jaw.pad_ahead_mm),
        "pad_behind_mm": float(jaw.pad_behind_mm),
        "finger_width_mm": float(jaw.finger_width_mm),
    }


def _yaw(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _box(size: tuple[float, float, float], *, yaw_deg: float = 0.0, seed: int = 7,
         noise_mm: float = 0.5, step_mm: float = 1.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every face of a box standing on the table, as (points, outward normal), BASE mm."""
    rng = np.random.default_rng(seed)
    half = np.asarray(size, dtype=np.float64) / 2.0
    centre = np.array([_CENTRE_XY[0], _CENTRE_XY[1], half[2]])
    rotation = _yaw(yaw_deg)
    faces = []
    for axis in range(3):
        u_axis, v_axis = (i for i in range(3) if i != axis)
        u = np.arange(-half[u_axis], half[u_axis] + 1e-9, step_mm)
        v = np.arange(-half[v_axis], half[v_axis] + 1e-9, step_mm)
        grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
        for sign in (-1.0, 1.0):
            local = np.zeros((grid_u.size, 3))
            local[:, u_axis] = grid_u.ravel()
            local[:, v_axis] = grid_v.ravel()
            local[:, axis] = sign * half[axis]
            normal = rotation @ np.eye(3)[axis] * sign
            points = local @ rotation.T + centre
            points += np.outer(rng.normal(0.0, noise_mm, len(points)), normal)
            faces.append((points, normal))
    return faces


def _tilted(deg: float) -> np.ndarray:
    """Straight down, tilted about the closing axis (BASE X) by ``deg``, as the support-footprint
    estimator tilts an approach."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([0.0, s, -c])


def _camera(bearing_deg: float, elevation_deg: float, height_mm: float = 20.0) -> np.ndarray:
    b, e = math.radians(bearing_deg), math.radians(elevation_deg)
    return np.array([
        _CENTRE_XY[0] + _RANGE_MM * math.cos(e) * math.cos(b),
        _CENTRE_XY[1] + _RANGE_MM * math.cos(e) * math.sin(b),
        height_mm + _RANGE_MM * math.sin(e),
    ])


def _seen_from(faces: list[tuple[np.ndarray, np.ndarray]], camera: np.ndarray) -> np.ndarray:
    """The faces that turn toward the camera. A convex box hides nothing else from it."""
    kept = [points for points, normal in faces
            if float(normal @ (camera - points.mean(axis=0))) > 1e-6]
    return np.vstack(kept)


def _judge(points: np.ndarray, *, width: float, grasp_z: float, yaw_deg: float = 0.0,
           model: str = "robotiq_hande", support: float | None = 0.0,
           approach: np.ndarray | tuple[float, float, float] = (0.0, 0.0, -1.0)) -> JawFacesSeen:
    return jaw_faces_seen(
        points,
        centre_mm=(_CENTRE_XY[0], _CENTRE_XY[1], grasp_z),
        approach=approach,
        closing_axis=_yaw(yaw_deg) @ np.array([1.0, 0.0, 0.0]),
        grip_width_mm=width,
        support_height_mm=support,
        **_jaw(model),
    )


class BothJawFacesAreSeenTests(unittest.TestCase):
    def test_a_top_view_of_a_cube_sees_neither_jaw_face(self) -> None:
        """A camera straight overhead measures the top and nothing the pads touch."""
        cube = _box((40.0, 40.0, 40.0))
        faces = _judge(_seen_from(cube, _camera(0.0, 90.0)), width=40.0, grasp_z=20.0)
        self.assertEqual(faces.as_pair(), (False, False))
        self.assertFalse(faces.both)
        self.assertEqual(len(faces.missing), 2)
        for face in (faces.negative, faces.positive):
            self.assertEqual(face.cells_seen, 0)
            self.assertEqual(face.cells_touchable, 48, "the whole Hand-E patch can touch a 40 mm cube")

    def test_a_45_degree_view_sees_the_near_face_only(self) -> None:
        """What a tilted wrist look sees: the face toward it, and a far face it can only have
        inferred."""
        for yaw in (0.0, 37.0):
            with self.subTest(yaw=yaw):
                cube = _box((40.0, 40.0, 40.0), yaw_deg=yaw)
                axis = _yaw(yaw) @ np.array([1.0, 0.0, 0.0])
                seen = _seen_from(cube, _camera(180.0 + yaw, 45.0))
                faces = _judge(seen, width=40.0, grasp_z=20.0, yaw_deg=yaw)
                self.assertTrue(faces.negative.seen)
                self.assertEqual(faces.negative.cells_seen, 48)
                self.assertFalse(faces.positive.seen)
                self.assertEqual(faces.positive.points, 0)
                self.assertEqual(faces.missing, (faces.positive,))
                self.assertEqual(faces.as_pair(), (True, False))
                far = faces.missing[0]
                self.assertEqual(far.side, 1)
                np.testing.assert_allclose(far.normal, axis, atol=1e-12)
                centre = np.array([_CENTRE_XY[0], _CENTRE_XY[1], 20.0]) + 20.0 * axis
                np.testing.assert_allclose(far.centre_mm, centre, atol=1e-9)
                np.testing.assert_allclose(faces.negative.normal, -axis, atol=1e-12)

    def test_two_opposite_views_fused_see_both_faces(self) -> None:
        cube = _box((40.0, 40.0, 40.0))
        fused = np.vstack([_seen_from(cube, _camera(180.0, 45.0)),
                           _seen_from(cube, _camera(0.0, 45.0))])
        faces = _judge(fused, width=40.0, grasp_z=20.0)
        self.assertTrue(faces.both)
        self.assertEqual(faces.missing, ())
        self.assertEqual(faces.as_pair(), (True, True))

    def test_the_patch_is_the_hands_not_a_constant(self) -> None:
        """The 20.91 x 29.24 mm Hand-E pad grids into 8 x 6 cells of at most 4 mm; the 2F-85's
        38 x 27 mm pad into 7 x 10, of which the two rows reaching lower than 5 mm over the table
        cannot touch a block gripped 20 mm up, so 56."""
        block = _box((40.0, 40.0, 60.0))
        seen = _seen_from(block, _camera(180.0, 45.0))
        hande = _judge(seen, width=40.0, grasp_z=20.0, model="robotiq_hande")
        f85 = _judge(seen, width=40.0, grasp_z=20.0, model="robotiq_2f85")
        self.assertEqual(hande.negative.cells_touchable, 48)
        self.assertEqual(f85.negative.cells_touchable, 56)
        self.assertNotEqual(hande.negative.cells_touchable, f85.negative.cells_touchable)

    def test_table_points_behind_the_part_do_not_count_as_a_face(self) -> None:
        """A mask that bleeds onto the table behind the far face puts table points where the far pad
        would close. Only the support height tells them from the part: without it they are band
        points, with it they are the table."""
        part = _box((40.0, 40.0, 20.0))
        seen = _seen_from(part, _camera(180.0, 45.0, height_mm=10.0))
        rng = np.random.default_rng(3)
        xs, ys = np.meshgrid(np.arange(520.5, 530.0, 1.0), np.arange(-20.0, 20.5, 1.0), indexing="ij")
        table = np.column_stack([xs.ravel(), ys.ravel(), rng.uniform(0.0, 4.0, xs.size)])
        cloud = np.vstack([seen, table])
        unaware = _judge(cloud, width=40.0, grasp_z=10.0, support=None)
        self.assertGreater(unaware.positive.points, 0, "the table points do fall in the far band")
        faces = _judge(cloud, width=40.0, grasp_z=10.0, support=0.0)
        self.assertEqual(faces.positive.points, 0)
        self.assertFalse(faces.positive.seen)
        self.assertTrue(faces.negative.seen)
        # Of the six rows, the lowest two reach under the support margin and the top two lie within a
        # band of the part's top: two rows of eight are left to judge either face on.
        self.assertEqual(faces.negative.cells_touchable, 16)

    def test_the_rim_of_a_flat_parts_top_is_not_its_far_face(self) -> None:
        """The inward half of the band reaches under the top. On a 12 mm part gripped at its middle
        the pad stands higher than the part, and the rim of the top a 45 degree look always sees falls
        in the far band: with the ceiling 2 mm over the top it filled 8 of 24 cells, the 30 % a face
        needs, and one look read both faces as seen. Within a band of the top a point of the face and a
        point of the rim are one measurement, so those cells count for neither face, and a part this
        flat has no height left that is neither table nor rim: the look says it cannot confirm a
        contact rather than confirming the wrong one."""
        part = _box((40.0, 40.0, 12.0))
        faces = _judge(_seen_from(part, _camera(180.0, 45.0, height_mm=6.0)), width=40.0, grasp_z=6.0)
        self.assertFalse(faces.positive.seen)
        self.assertEqual(faces.positive.points, 0)
        self.assertEqual(faces.positive.cells_touchable, 0)
        self.assertEqual(faces.as_pair(), (False, False))

    def test_a_narrow_part_is_judged_on_the_patch_it_can_touch(self) -> None:
        """An 8 mm plate stood on edge fills 2 of the Hand-E's 8 columns. Asked against the whole
        48-cell patch it would need 15 seen cells and could never have more than 12. A mask that bleeds
        onto the table all round it must not widen that patch either, so the part's width is read off
        its points over the support, not off the table's."""
        plate = _box((40.0, 8.0, 40.0))
        seen = _seen_from(plate, _camera(180.0, 45.0))
        faces = _judge(seen, width=40.0, grasp_z=20.0)
        self.assertEqual(faces.negative.cells_touchable, 12)
        self.assertEqual(faces.negative.cells_seen, 12)
        self.assertTrue(faces.negative.seen)
        self.assertFalse(faces.positive.seen)
        rng = np.random.default_rng(5)
        xs, ys = np.meshgrid(np.arange(470.0, 530.5, 1.0), np.arange(-20.0, 20.5, 1.0), indexing="ij")
        ring = (np.abs(ys) > 4.0).ravel()
        table = np.column_stack([xs.ravel(), ys.ravel(), rng.uniform(0.0, 3.0, xs.size)])[ring]
        bled = _judge(np.vstack([seen, table]), width=40.0, grasp_z=20.0)
        self.assertEqual(bled.negative.cells_touchable, 12)
        self.assertTrue(bled.negative.seen)

    def test_a_wider_body_beyond_the_band_does_not_widen_the_patch(self) -> None:
        """The part's width across the binormal is read near the grasp. A flange 10 mm past the far
        face, 60 mm across, is part of the same cloud, but the pads close on the 8 mm stem, and a
        patch widened to the flange would ask 15 seen cells of a stem that can show 12."""
        stem = _seen_from(_box((40.0, 8.0, 40.0)), _camera(180.0, 45.0))
        xs, ys = np.meshgrid(np.arange(530.0, 550.5, 1.0), np.arange(-30.0, 30.5, 1.0), indexing="ij")
        flange = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 40.0)])
        faces = _judge(np.vstack([stem, flange]), width=40.0, grasp_z=20.0)
        self.assertEqual(faces.negative.cells_touchable, 12)
        self.assertTrue(faces.negative.seen)

    def test_a_side_face_of_a_narrow_part_is_not_its_far_face(self) -> None:
        """A part narrower than the pad shows a look from off the closing axis one of its sides, and
        that side runs the length of the part: its strip within the band of the far face lies where the
        far pad would close, filling the edge column of the patch on every row. Read as the far face,
        one look, or two from the same side, would confirm a contact no camera faced. The side runs
        through the part between the two faces, where a jaw face never does, and that is what tells
        it from the face; a margin off the part's percentile width does not, once the depth noise
        spreads the side past it. Narrow parts, looks 5 to 40 degrees off the axis, one or two from the
        same side, grasps at mid-height and 4 mm under the top, approaches tilted 15 and 30 degrees:
        none of them sees the far face. The 12 by 10 mm post is gripped across its 12 mm, where only
        2 mm lie between the bands against 11 in each: the side is told by its density there, not by
        its count."""
        for length, width in ((40.0, 8.0), (40.0, 12.0), (40.0, 20.0), (12.0, 10.0)):
            for noise in (0.5, 1.0):
                part = _box((length, width, 40.0), noise_mm=noise, step_mm=0.7, seed=int(width))
                scenes: list[tuple[str, np.ndarray, float, np.ndarray | tuple[float, float, float]]] = []
                for delta in (5.0, 20.0, 40.0):
                    one = _seen_from(part, _camera(180.0 + delta, 45.0))
                    two = np.vstack([one, _seen_from(part, _camera(180.0 - delta, 45.0))])
                    scenes += [(f"one look {delta:g} deg off, grasp at mid-height", one, 20.0, (0, 0, -1)),
                               (f"one look {delta:g} deg off, grasp 4 mm under the top", one, 36.0, (0, 0, -1)),
                               (f"two looks 180 +- {delta:g} deg", two, 20.0, (0, 0, -1))]
                for tilt in (15.0, 30.0):
                    scenes.append((f"approach tilted {tilt:g} deg", _seen_from(part, _camera(200.0, 45.0)),
                                   20.0, _tilted(tilt)))
                for label, cloud, grasp_z, approach in scenes:
                    with self.subTest(length=length, width=width, noise=noise, scene=label):
                        faces = _judge(cloud, width=length, grasp_z=grasp_z, approach=approach)
                        self.assertFalse(faces.positive.seen)
                        self.assertFalse(faces.both)
                        self.assertIn(faces.positive, faces.missing)

    def test_a_narrow_part_seen_from_both_sides_has_both_faces(self) -> None:
        """What the side rule must not cost: a narrow part whose far face a look did face still has it,
        also when both looks stand off the axis and each sees a side running through the part."""
        for length, width in ((40.0, 8.0), (40.0, 12.0), (40.0, 20.0), (12.0, 10.0)):
            for noise in (0.5, 1.0):
                part = _box((length, width, 40.0), noise_mm=noise, step_mm=0.7, seed=int(width))
                for bearings in ((180.0, 0.0), (195.0, 340.0), (160.0, 20.0)):
                    with self.subTest(length=length, width=width, noise=noise, bearings=bearings):
                        cloud = np.vstack([_seen_from(part, _camera(b, 45.0)) for b in bearings])
                        faces = _judge(cloud, width=length, grasp_z=20.0)
                        self.assertTrue(faces.both)

    def test_a_thin_grip_counts_no_point_for_both_faces(self) -> None:
        """Gripped across an 8 mm plate, a band reaching the full 6 mm inward would pass the centre and
        take the near face's points for the far one. The bands stop a millimetre short of the centre.
        One look at the plate, its near face read 2.5 mm deep, well inside the band: the far face gets
        none of it."""
        plate = _seen_from(_box((8.0, 40.0, 40.0)), _camera(180.0, 45.0))
        deep = plate.copy()
        near = plate[:, 0] < _CENTRE_XY[0] - 2.0
        deep[near, 0] += 2.5
        for label, cloud in (("as measured", plate), ("read 2.5 mm deep", deep)):
            with self.subTest(label):
                faces = _judge(cloud, width=8.0, grasp_z=20.0)
                self.assertTrue(faces.negative.seen)
                self.assertEqual(faces.positive.points, 0)
                self.assertEqual(faces.as_pair(), (True, False))

    def test_a_row_reaching_over_the_ceiling_is_not_touchable(self) -> None:
        """The ceiling asks for the whole cell under it, not its centre. On a 40 mm cube measured
        without noise the ceiling stands at 34 mm; a Hand-E row is 3.485 mm tall. The top row centred
        at 33 mm reaches 34.74 and cannot be touched, so 40 of 48 cells; the same row ending at 33.9
        mm can, so 48."""
        jaw = _jaw("robotiq_hande")
        length = jaw["pad_ahead_mm"] + jaw["pad_behind_mm"]
        row = length / math.ceil(length / FACE_CELL_MM)
        self.assertAlmostEqual(row, 3.485, places=9)
        # The approach points down, so the row nearest the palm is the highest: its centre stands this
        # far over the grasp centre.
        top_row = jaw["pad_behind_mm"] - row / 2.0
        seen = _seen_from(_box((40.0, 40.0, 40.0), noise_mm=0.0), _camera(180.0, 45.0))
        straddling = _judge(seen, width=40.0, grasp_z=33.0 - top_row)
        self.assertEqual(straddling.negative.cells_touchable, 40)
        under = _judge(seen, width=40.0, grasp_z=33.9 - row / 2.0 - top_row)
        self.assertEqual(under.negative.cells_touchable, 48)

    def test_a_grasp_just_under_the_top_does_not_read_the_rim_as_the_far_face(self) -> None:
        """The support-footprint estimator's first anchor is 4 mm under the top. There the pad stands
        mostly over the part, and the rim of the top a 45 degree look always sees would fill the far
        band's upper rows: with the ceiling 2 mm over the top, one look read both faces on every part
        height. Wholly a band under the top, 16 cells are left, the near face fills them and the far
        face has none."""
        faces = _judge(_seen_from(_box((40.0, 40.0, 40.0)), _camera(180.0, 45.0)), width=40.0,
                       grasp_z=36.0)
        self.assertEqual(faces.negative.cells_touchable, 16)
        self.assertTrue(faces.negative.seen)
        self.assertEqual(faces.positive.points, 0)
        self.assertEqual(faces.as_pair(), (True, False))

    def test_a_neighbour_beyond_the_band_is_not_the_face(self) -> None:
        """A taller neighbour 10 mm past the far face faces the camera where the far pad would be.
        It lies beyond the band, so it is not read as the face the camera could not see; the same
        neighbour 4 mm off would be, which is why the band is no wider than the error it absorbs."""
        cube = _box((40.0, 40.0, 40.0))
        seen = _seen_from(cube, _camera(180.0, 45.0))
        ys, zs = np.meshgrid(np.arange(-20.0, 20.5, 1.0), np.arange(0.0, 60.5, 1.0), indexing="ij")

        def neighbour(x: float) -> np.ndarray:
            return np.column_stack([np.full(ys.size, x), ys.ravel(), zs.ravel()])

        far = _judge(np.vstack([seen, neighbour(530.0)]), width=40.0, grasp_z=20.0)
        self.assertEqual(far.positive.points, 0)
        self.assertFalse(far.positive.seen)
        self.assertEqual(far.negative.cells_seen, 48, "the neighbour changes nothing near")
        inside = neighbour(520.0 + FACE_BAND_MM - 2.0)
        near = _judge(np.vstack([seen, inside]), width=40.0, grasp_z=20.0)
        self.assertGreater(near.positive.points, 0)

    def test_the_support_margin_is_the_scenes(self) -> None:
        """The table is the table by the same margin the scene plans with, not a second opinion."""
        self.assertEqual(SUPPORT_MARGIN_MM, scene.SUPPORT_READ_ERROR_MM)

    def test_a_face_needs_its_share_of_the_patch_and_never_fewer_than_the_floor(self) -> None:
        """The cells a face needs are max(FACE_MIN_CELLS, ceil(coverage x touchable)): 15 of the
        Hand-E's 48, 3 of 10, and 3 of 4, where the share alone would ask for 2."""
        cube = _box((40.0, 40.0, 40.0))
        face = _judge(_seen_from(cube, _camera(180.0, 45.0)), width=40.0, grasp_z=20.0).negative
        self.assertEqual(face.cells_needed, max(FACE_MIN_CELLS, math.ceil(FACE_MIN_COVERAGE * 48)))
        self.assertEqual(face.cells_needed, 15)
        ten = type(face)(side=-1, centre_mm=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                         points=30, cells_seen=3, cells_touchable=10)
        self.assertEqual(ten.cells_needed, 3)
        self.assertTrue(ten.seen)
        four = type(face)(side=-1, centre_mm=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                          points=20, cells_seen=2, cells_touchable=4)
        self.assertEqual(four.cells_needed, FACE_MIN_CELLS)
        self.assertFalse(four.seen, "half of a tiny patch is still only two cells")
        none = type(face)(side=-1, centre_mm=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                          points=0, cells_seen=0, cells_touchable=0)
        self.assertFalse(none.seen, "a face with nothing it could touch was not seen")

    def test_a_flying_pixel_is_not_a_seen_cell(self) -> None:
        """Fewer than FACE_CELL_MIN_POINTS in a cell is noise, however many cells it is spread over."""
        cube = _box((40.0, 40.0, 40.0))
        seen = _seen_from(cube, _camera(180.0, 45.0))
        ys, zs = np.meshgrid(np.arange(-12.0, 13.0, 4.0), np.arange(12.0, 29.0, 3.5), indexing="ij")
        sparse = np.column_stack([np.full(ys.size, 521.0), ys.ravel(), zs.ravel()])
        sparse = np.repeat(sparse, FACE_CELL_MIN_POINTS - 1, axis=0)
        faces = _judge(np.vstack([seen, sparse]), width=40.0, grasp_z=20.0)
        self.assertGreater(faces.positive.points, 0)
        self.assertEqual(faces.positive.cells_seen, 0)

    def test_a_cell_with_exactly_the_minimum_is_seen(self) -> None:
        """The same 35 cells with FACE_CELL_MIN_POINTS each are seen, and 35 of 48 is a face."""
        cube = _box((40.0, 40.0, 40.0))
        seen = _seen_from(cube, _camera(180.0, 45.0))
        ys, zs = np.meshgrid(np.arange(-12.0, 13.0, 4.0), np.arange(12.0, 29.0, 3.5), indexing="ij")
        sparse = np.column_stack([np.full(ys.size, 521.0), ys.ravel(), zs.ravel()])
        sparse = np.repeat(sparse, FACE_CELL_MIN_POINTS, axis=0)
        faces = _judge(np.vstack([seen, sparse]), width=40.0, grasp_z=20.0)
        self.assertEqual(faces.positive.points, 35 * FACE_CELL_MIN_POINTS)
        self.assertEqual(faces.positive.cells_seen, 35)
        self.assertTrue(faces.positive.seen)

    def test_an_empty_cloud_sees_nothing_and_a_closing_axis_along_the_approach_is_refused(self) -> None:
        faces = _judge(np.empty((0, 3)), width=40.0, grasp_z=20.0)
        self.assertEqual(faces.as_pair(), (False, False))
        self.assertEqual(faces.negative.cells_touchable, 0)
        with self.assertRaises(ValueError):
            jaw_faces_seen(np.zeros((4, 3)), centre_mm=(0.0, 0.0, 0.0), approach=(0.0, 0.0, -1.0),
                           closing_axis=(0.0, 0.0, 1.0), grip_width_mm=40.0, **_jaw("robotiq_hande"))


if __name__ == "__main__":
    unittest.main()
