"""A wrist camera's body is its registry housing and the rig's bracket, grown and placed by its own calibration.

As the owner decided: the boxes are written in the colour camera's optical frame, grown by the rig's margin, and
placed on tool0 by the recorded flange to TCP times the calibrated CAMERA to TOOL, so the body and the pick frame are
placed from the same numbers. The evidence is keyed without the camera, and what stands in for a measurement is a
proof that the sphere fill holds every point of every box.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.config.cameras import load_camera
from src.config.schema.cameras import OpticalBox
from src.calibration.serialization import FlangeToTcp
from src.robot.safety.planning._curobo_body_links import WRIST_BODY_IGNORE, BodyLinkError
from src.robot.safety.planning._declared_body import Box, box_spheres, cover_refusal, inflated
from src.robot.safety.planning.body_link import WristBody
from tests._wrist_body import camera_to_tool, declared_frame, wrist_body

_BRACKET = OpticalBox(size_mm=(60.0, 10.0, 40.0), centre_mm=(30.0, 18.0, -30.0))


def _body(*, margin_mm: float = 5.0, bracket: object = None, record: np.ndarray | None = None) -> WristBody:
    return WristBody.from_parts(
        rig_id="wrist", spec=load_camera("realsense_d435"), bracket=bracket, margin_mm=margin_mm,
        camera_to_tool=camera_to_tool(),
        flange_to_tcp=FlangeToTcp.from_matrix("willy", declared_frame() if record is None else record),
        artifact_path="eih_wrist.json")


class ThePlacementTests(unittest.TestCase):
    def test_the_housing_centre_is_the_record_times_the_calibration_times_the_registry_offset(self) -> None:
        body = _body()
        housing = load_camera("realsense_d435").housing
        by_hand = declared_frame() @ camera_to_tool().to_matrix() @ np.array([*housing.centre_mm, 1.0])
        corners = body.guard_parts()["wrist_camera_wrist__v"][:8]
        np.testing.assert_allclose(corners.mean(axis=0), by_hand[:3], atol=1e-9)

    def test_an_identity_calibration_puts_the_housing_at_the_record_times_its_offset(self) -> None:
        from src.geometry import Frame, Transform

        identity = Transform(translation_mm=np.zeros(3), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                             from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
        body = WristBody.from_parts(rig_id="wrist", spec=load_camera("realsense_d435"), bracket=None, margin_mm=1.0,
                                    camera_to_tool=identity,
                                    flange_to_tcp=FlangeToTcp.from_matrix("willy", declared_frame()))
        centre = body.guard_parts()["wrist_camera_wrist__v"][:8].mean(axis=0)
        expected = declared_frame() @ np.array([32.5, 0.0, -8.275, 1.0])
        np.testing.assert_allclose(centre, expected[:3], atol=1e-9)

    def test_the_margin_grows_every_box(self) -> None:
        body = _body(margin_mm=5.0, bracket=_BRACKET)
        self.assertEqual([box.half_extents_mm for box in body.boxes],
                         [(45.075 + 5.0, 12.575 + 5.0, 12.575 + 5.0), (30.0 + 5.0, 5.0 + 5.0, 20.0 + 5.0)])

    def test_housing_and_bracket_are_one_part_and_one_link(self) -> None:
        body = _body(bracket=_BRACKET)
        parts = body.guard_parts()
        self.assertEqual(sorted(parts), ["wrist_camera_wrist__f", "wrist_camera_wrist__frame", "wrist_camera_wrist__v"])
        self.assertEqual(parts["wrist_camera_wrist__v"].shape, (16, 3))
        self.assertEqual(int(parts["wrist_camera_wrist__frame"][0]), 6)
        link = body.link()
        self.assertEqual((link["link"], link["parent"]), ("wrist_camera_wrist", "tool0"))
        self.assertEqual(link["ignore"], list(WRIST_BODY_IGNORE))
        self.assertEqual(len(link["spheres"]), sum(len(spheres) for spheres in body.spheres_by_box()))

    def test_the_link_transform_is_the_placement_in_metres(self) -> None:
        body = _body()
        placement = body.placement()
        np.testing.assert_allclose(body.link()["fixed_transform"][:3], placement[:3, 3] / 1000.0, atol=1e-15)

    def test_the_self_filter_spheres_sit_where_the_guard_boxes_sit(self) -> None:
        body = _body()
        vertices = body.guard_parts()["wrist_camera_wrist__v"]
        centres = np.array([centre for centre, _ in body.envelope_spheres_mm()])
        self.assertTrue(np.all(centres.min(axis=0) >= vertices.min(axis=0) - 1e-6))
        self.assertTrue(np.all(centres.max(axis=0) <= vertices.max(axis=0) + 1e-6))

    def test_a_margin_that_is_not_above_zero_and_a_placement_that_is_not_rigid_are_refused(self) -> None:
        with self.assertRaises(BodyLinkError):
            _body(margin_mm=0.0)
        from src.geometry import Frame, Transform

        base = Transform(translation_mm=np.zeros(3), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                         from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        with self.assertRaises(BodyLinkError) as caught:
            WristBody.from_parts(rig_id="wrist", spec=load_camera("realsense_d435"), bracket=None, margin_mm=1.0,
                                 camera_to_tool=base, flange_to_tcp=FlangeToTcp.from_matrix("willy", declared_frame()))
        self.assertIn("placed by CAMERA to TOOL", str(caught.exception))

    def test_the_resolved_body_of_a_wrist_owner_is_this_body(self) -> None:
        self.assertEqual(wrist_body().placement_mm, _body().placement_mm)


class TheCoverProofTests(unittest.TestCase):
    """Every point of every grown box lies inside the fill, proven on the grid rather than sampled."""

    def _boxes(self) -> list[Box]:
        boxes = []
        for model in ("realsense_d405", "realsense_d415", "realsense_d435", "realsense_d435i"):
            housing = load_camera(model).housing
            boxes.append(inflated(Box(model, tuple(housing.centre_mm),  # type: ignore[arg-type]
                                      tuple(v / 2.0 for v in housing.size_mm)), by_mm=5.0))  # type: ignore[arg-type]
        boxes.append(Box("bracket", (30.0, 18.0, -30.0), (30.0, 5.0, 20.0)))
        return boxes

    def test_every_fill_the_writer_makes_is_proven_complete(self) -> None:
        for box in self._boxes():
            for reach in (4.0, 8.0, 20.0):
                with self.subTest(box=box.name, reach=reach):
                    self.assertIsNone(cover_refusal(box, box_spheres(box, reach_mm=reach)))

    def test_a_shrunk_radius_is_a_hole(self) -> None:
        box = self._boxes()[2]
        spheres = box_spheres(box, reach_mm=8.0)
        spheres[3] = {**spheres[3], "radius": spheres[3]["radius"] - 1e-6}
        refusal = cover_refusal(box, spheres)
        assert refusal is not None
        self.assertIn("outside its sphere, so the spheres leave a hole in the box, which is a false clear", refusal)

    def test_a_missing_sphere_is_refused(self) -> None:
        box = self._boxes()[2]
        refusal = cover_refusal(box, box_spheres(box, reach_mm=8.0)[1:])
        assert refusal is not None
        self.assertIn("a cover proof needs exactly one sphere per grid point", refusal)

    def test_shifted_centres_are_refused(self) -> None:
        box = self._boxes()[2]
        shifted = [{"center": [c[0] + 1e-5, c[1], c[2]], "radius": s["radius"]}
                   for s in box_spheres(box, reach_mm=8.0) for c in [s["center"]]]
        self.assertIsNotNone(cover_refusal(box, shifted))

    def test_a_box_grown_past_its_fill_is_refused(self) -> None:
        box = self._boxes()[2]
        spheres = box_spheres(box, reach_mm=8.0)
        self.assertIsNotNone(cover_refusal(inflated(box, by_mm=0.01), spheres))

    def test_the_proof_is_not_vacuous_a_sample_agrees(self) -> None:
        """The control: a 1 mm sample of the box sits inside the union for a fill the proof admits, and a point in the
        hole a shrunk sphere leaves is outside it."""
        box = Box("d435", (32.5, 0.0, -8.275), (50.075, 17.575, 17.575))
        spheres = box_spheres(box, reach_mm=8.0)
        self.assertIsNone(cover_refusal(box, spheres))
        centres = np.array([s["center"] for s in spheres]) * 1000.0
        radii = np.array([s["radius"] for s in spheres]) * 1000.0
        low = np.asarray(box.centre_mm) - np.asarray(box.half_extents_mm)
        high = np.asarray(box.centre_mm) + np.asarray(box.half_extents_mm)
        grid = np.stack(np.meshgrid(*[np.arange(low[i], high[i] + 1e-9, 1.0) for i in range(3)], indexing="ij"),
                        axis=-1).reshape(-1, 3)
        grid = np.vstack([grid, [low, high]])
        inside = np.zeros(len(grid), dtype=bool)
        for centre, radius in zip(centres, radii):
            inside |= np.linalg.norm(grid - centre, axis=1) <= radius + 1e-6
        self.assertTrue(bool(inside.all()))
        corner_owner = int(np.argmin(np.linalg.norm(centres - low, axis=1)))
        self.assertLessEqual(np.linalg.norm(low - centres[corner_owner]), radii[corner_owner] + 1e-6)
        self.assertGreater(np.linalg.norm(low - centres[corner_owner]), radii[corner_owner] - 1e-3)


if __name__ == "__main__":
    unittest.main()
