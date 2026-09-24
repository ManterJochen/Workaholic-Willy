"""A wrist camera tilted 45 degrees grasps the part, not the table behind it (owner's cell, 2026-09-23).

The owner's D415 rides on the wrist tilted about 45 degrees forward. One or two pixels of depth-to-colour
misregistration at a part's far top edge put depth from behind the part inside its mask. Straight down that is the
table under the edge, inside the footprint; at 45 degrees it is the table 40 to 46 mm behind the part. With the
table read a few millimetres high (hand-eye or TCP height error) those points survived the support-footprint
stage's 2 mm floor, its convex hull reached back to them, and the best grasp line lay about 37 mm from the centre of
a 40 mm cube: the jaws closed behind the part, which this sensorless jaw cannot report.

The scene is ray-cast rather than rendered: a 40 mm cube on a table at z = 0, a pinhole with the D415 colour
intrinsics at 1280x720 (fx 925), 500 mm from the cube, depth shifted down by one or two rows (the far edge is at
the top of the image), the base frame read 3 mm high, 1 mm Gaussian depth noise and whole-millimetre depth as the
D415 streams it. The same seeds run through the Located path example 13 uses and through the cell
path's calculator.
"""

from __future__ import annotations

import functools
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

_W, _H, _F = 1280, 720, 925.0
_K = np.array([[_F, 0.0, 640.0], [0.0, _F, 360.0], [0.0, 0.0, 1.0]])
_CENTRE = np.array([500.0, -300.0])
_SIZE = 40.0
_SEEDS = range(5)


@functools.lru_cache(maxsize=None)
def _hande() -> Any:
    from src.config.loader import load_robot_section

    return load_robot_section(profile="hande")


def _camera_to_base(tilt_deg: float, distance_mm: float = 500.0) -> np.ndarray:
    """A camera ``distance_mm`` from the cube's centre, looking at it ``tilt_deg`` off straight down, from -y."""
    tilt = np.radians(tilt_deg)
    optical = np.array([0.0, np.sin(tilt), -np.cos(tilt)])
    right = np.array([1.0, 0.0, 0.0])
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, np.cross(optical, right), optical])
    matrix[:3, 3] = np.array([_CENTRE[0], _CENTRE[1], _SIZE / 2.0]) - distance_mm * optical
    return matrix


def _ray_cast(camera_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The cube's mask and the depth of every pixel, cube or table, in millimetres along the optical axis."""
    rotation, origin = camera_to_base[:3, :3], camera_to_base[:3, 3]
    u, v = np.meshgrid(np.arange(_W), np.arange(_H))
    rays = np.stack([(u - _K[0, 2]) / _F, (v - _K[1, 2]) / _F, np.ones(u.shape)], axis=-1) @ rotation.T
    low = np.array([_CENTRE[0] - _SIZE / 2, _CENTRE[1] - _SIZE / 2, 0.0])
    high = np.array([_CENTRE[0] + _SIZE / 2, _CENTRE[1] + _SIZE / 2, _SIZE])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low, t_high = (low - origin) / rays, (high - origin) / rays
    enter = np.nanmax(np.minimum(t_low, t_high), axis=-1)
    leave = np.nanmin(np.maximum(t_low, t_high), axis=-1)
    cube = (leave >= enter) & (enter > 0.0)
    t = np.where(cube, enter, -origin[2] / rays[..., 2])
    return cube, t * (rays @ rotation[:, 2])


def _frame(tilt_deg: float, *, shift_px: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mask, depth_mm as uint16, camera_to_base as the cell believes it)`` for one noisy frame."""
    camera_to_base = _camera_to_base(tilt_deg)
    mask, depth = _ray_cast(camera_to_base)
    if shift_px:
        shifted = depth.copy()
        shifted[shift_px:, :] = depth[:-shift_px, :]
        depth = shifted
    noisy = depth + np.random.default_rng(seed).normal(0.0, 1.0, depth.shape)
    believed = camera_to_base.copy()
    believed[2, 3] += 3.0
    return mask, np.clip(np.round(noisy), 0, 65535).astype(np.uint16), believed


def _off_the_centre(position_mm: np.ndarray, closing_axis: np.ndarray) -> tuple[float, float]:
    """How far the grasp line lies from the cube's centre, across it and along it, in millimetres."""
    axis = np.asarray(closing_axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    offset = np.r_[np.asarray(position_mm, dtype=np.float64)[:2] - _CENTRE, 0.0]
    along = float(offset @ axis)
    return float(np.linalg.norm(offset - along * axis)), abs(along)


# --------------------------------------------------------------------------- the Located path, as example 13 runs it


@dataclass(frozen=True)
class _Segmentation:
    label: str
    score: float
    bbox_xyxy: tuple
    mask: np.ndarray


class _Handle:
    def __init__(self, depth: np.ndarray) -> None:
        self.rig_id = "wrist"
        self._depth = depth

    def grab(self):  # noqa: ANN201
        from src.camera.setup.image_taking.frames import RGBDFrame

        return RGBDFrame(color=np.zeros((_H, _W, 3), dtype=np.uint8), depth=self._depth, captured_at_s=10.0)

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


class _WristCamera:
    """The owner's wrist D415 as the locator sees an open camera: CAMERA to TOOL is the tilt, the TCP carries the rest."""

    def __init__(self, depth: np.ndarray, camera_to_base: np.ndarray) -> None:
        from src.calibration.rig_calibration import RigCalibration
        from src.calibration.serialization import FlangeToTcp
        from src.geometry import Frame, Transform

        rotation_only = np.eye(4)
        rotation_only[:3, :3] = camera_to_base[:3, :3]
        self.rig_id = "wrist"
        self.source = "rgbd"
        self._handle = _Handle(depth)
        self._calibration = RigCalibration(
            rig_id="wrist", mounting_mode="eye_in_hand", artifact_path="eih.json",
            transform=Transform.from_matrix(rotation_only, from_frame=Frame.CAMERA, to_frame=Frame.TOOL),
            shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5,
            flange_to_tcp=FlangeToTcp.from_matrix("willy", np.eye(4)))

    def handle(self) -> _Handle:
        return self._handle

    def calibration(self):  # noqa: ANN201
        return self._calibration


def _located_best(tilt_deg: float, *, shift_px: int, seed: int) -> Any:
    from src.geometry import Frame, Pose
    from src.robot.perception.locator import Locator

    mask, depth, believed = _frame(tilt_deg, shift_px=shift_px, seed=seed)
    tcp = Pose(position_mm=believed[:3, 3].copy(), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE)
    seen = SimpleNamespace(detection=SimpleNamespace(score=0.9, box=None),
                           segmentation=_Segmentation(label="red cube", score=0.9, bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
                                                      mask=mask))
    backend = SimpleNamespace(perceive=lambda _bgr, _prompt: (seen,))
    tool_frame = SimpleNamespace(source="willy", offset_mm=(0.0, 0.0, 0.0), rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                                 verify_tolerance_mm=1.0)
    located = Locator.from_parts(camera=_WristCamera(depth, believed), backend=backend, tool_pose=lambda: tcp,
                                 tool_frame=tool_frame).locate("a red cube")
    return located.scene(0, _hande()).grasps()


class TheLocatedPathGraspsOnThePartTests(unittest.TestCase):
    def _assert_on_the_part(self, tilt_deg: float, shift_px: int) -> None:
        for seed in _SEEDS:
            grasps = _located_best(tilt_deg, shift_px=shift_px, seed=seed)
            best = grasps.best
            with self.subTest(seed=seed):
                assert best is not None, grasps.render()
                across, along = _off_the_centre(best.position_mm, best.closing_axis)
                # A Hand-E pad is 29 mm wide; within 10 mm of the centre line it stays on a 40 mm face. The straight-down
                # scene with nothing leaking sits 0.3 to 9.2 mm off, the thirds SFE also tries.
                self.assertLessEqual(across, 10.0, grasps.render())
                self.assertLessEqual(along, 2.0, grasps.render())
                self.assertLessEqual(best.grip_width_mm, 44.0, grasps.render())
                self.assertGreater(-float(best.approach[2]), 0.999, "a vertical approach")

    def test_the_owners_45_degree_view_with_2_px_of_misregistration_grasps_on_the_part(self) -> None:
        self._assert_on_the_part(45.0, 2)

    def test_one_pixel_of_misregistration_too(self) -> None:
        self._assert_on_the_part(45.0, 1)

    def test_a_camera_looking_straight_down_still_grasps_on_the_part(self) -> None:
        self._assert_on_the_part(0.0, 2)


# --------------------------------------------------------------------------- the cell path, through the calculator


@functools.lru_cache(maxsize=None)
def _calculator() -> Any:
    from src.robot.grasping.calculator_factory import build_calculator

    cfg = _hande()
    return build_calculator(
        cfg, camera_matrix=_K, max_grip_width_mm=cfg.gripper.max_width_mm,
        min_grip_width_mm=cfg.gripper.min_width_mm,
        support_footprint_geometry=cfg.grasping.geometry.stage == "support_footprint",
        support_footprint_inflate_mm=cfg.grasping.geometry.inflate_mm)


def _cell_compute(tilt_deg: float, *, shift_px: int, seed: int) -> tuple[list, dict]:
    from src.geometry import Frame
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.grasping.collision import SupportPlane

    cfg = _hande()
    mask, depth, believed = _frame(tilt_deg, shift_px=shift_px, seed=seed)
    calculator = _calculator()
    candidates = calculator.compute(
        SimpleNamespace(mask=mask), depth.astype(np.float64), camera_to_base=believed,
        support_plane=SupportPlane(np.array([0.0, 0.0, 1.0]), float(cfg.grasping.support.height_mm), Frame.BASE),
        gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry),
        min_table_clearance_mm=float(cfg.grasping.support.min_clearance_mm))
    return candidates, dict(calculator.last_telemetry)


class TheCellPathGraspsOnThePartTests(unittest.TestCase):
    def test_the_calculator_reaches_the_footprint_stage_and_grasps_on_the_part(self) -> None:
        """The silhouette of the tilted cube spans its top and its near side, 54.5 mm, wider than the 49.99 mm stroke.

        It used to end every attempt there with no candidate, before the support-footprint stage, whose candidates
        replace the silhouette's anyway, was reached.
        """
        for seed in _SEEDS:
            candidates, telemetry = _cell_compute(45.0, shift_px=2, seed=seed)
            with self.subTest(seed=seed):
                self.assertEqual(0, telemetry["candidates_silhouette"])
                self.assertEqual("support_footprint", telemetry.get("geometry_stage"), telemetry)
                self.assertTrue(candidates, telemetry)
                best = candidates[0]
                across, along = _off_the_centre(best.position, best.axis)
                self.assertLessEqual(across, 10.0, telemetry)
                self.assertLessEqual(along, 2.0, telemetry)
                self.assertLessEqual(best.grip_width_mm, 44.0, telemetry)
                self.assertGreater(-float(best.approach[2]), 0.999, "a vertical approach")

    def test_the_calculator_leaves_the_background_behind_the_edge_out_of_the_footprint(self) -> None:
        """A band at most three pixels deep along the far edge when it is misregistered, and nothing when it is not."""
        mask, _depth, _believed = _frame(45.0, shift_px=0, seed=0)
        columns = int(np.count_nonzero(mask.any(axis=0)))
        _candidates, clean = _cell_compute(45.0, shift_px=0, seed=0)
        self.assertEqual(0, clean["support_footprint_depth_step_pixels"], clean)
        _candidates, leaked = _cell_compute(45.0, shift_px=2, seed=0)
        self.assertGreaterEqual(leaked["support_footprint_depth_step_pixels"], columns, leaked)
        self.assertLessEqual(leaked["support_footprint_depth_step_pixels"], 3 * columns, leaked)

    def test_with_no_support_plane_an_empty_silhouette_still_ends_the_attempt(self) -> None:
        """No plane, no footprint stage: the silhouette is all there is, and its empty answer stands."""
        mask, depth, believed = _frame(45.0, shift_px=2, seed=0)
        calculator = _calculator()
        self.assertEqual([], calculator.compute(SimpleNamespace(mask=mask), depth.astype(np.float64),
                                                camera_to_base=believed))
        self.assertEqual(0, calculator.last_telemetry["candidates_silhouette"])
        self.assertNotIn("geometry_stage", calculator.last_telemetry)


# --------------------------------------------------------------------------- the footprint's own hull


def _clean_top_and_side(seed: int) -> np.ndarray:
    """The cube's measured surface at 45 degrees, nothing leaking: what the camera sees of the part itself."""
    from src.robot.grasping.multiview.scene_geometry import to_base_mm

    camera_to_base = _camera_to_base(45.0)
    mask, depth = _ray_cast(camera_to_base)
    noisy = depth + np.random.default_rng(seed).normal(0.0, 1.0, depth.shape)
    return to_base_mm(mask, noisy, _K, camera_to_base)


def _leak(count: int, seed: int) -> np.ndarray:
    """``count`` table points 3 +- 1 mm high, 20 to 45 mm behind the part's far edge: the audit's verifier band."""
    rng = np.random.default_rng(100 + seed)
    return np.column_stack([rng.uniform(_CENTRE[0] - 20.0, _CENTRE[0] + 20.0, count),
                            rng.uniform(_CENTRE[1] + 40.0, _CENTRE[1] + 65.0, count),
                            rng.normal(3.0, 1.0, count)])


class TheFootprintLeavesLowFragmentsOutTests(unittest.TestCase):
    """The hull is built from the body; a low band lying apart from it is not the part."""

    def test_a_band_of_table_behind_the_part_does_not_stretch_the_hull(self) -> None:
        from src.robot.grasping.generation.support_footprint import reconstruct_support_prism

        clean = _clean_top_and_side(2)
        for count in (10, 30, 63, 100):
            with self.subTest(leaked_points=count):
                prism = reconstruct_support_prism(np.vstack([clean, _leak(count, count)]), 0.0)
                assert prism is not None
                self.assertLessEqual(float(prism.hull[:, 1].max()) - _CENTRE[1], _SIZE / 2.0 + 3.0)

    def test_the_grasp_stays_on_the_part(self) -> None:
        from src.robot.grasping.generation.support_footprint import (
            SupportFootprintJaw,
            generate_support_footprint_grasps,
        )

        jaw = SupportFootprintJaw.from_robot_config(_hande())
        clean = _clean_top_and_side(2)
        for count in (10, 30, 63, 100):
            with self.subTest(leaked_points=count):
                candidates = generate_support_footprint_grasps(
                    np.vstack([clean, _leak(count, count)]), support_height_mm=0.0, jaw=jaw)
                self.assertTrue(candidates)
                across, along = _off_the_centre(candidates[0].position_mm, candidates[0].closing_axis)
                self.assertLessEqual(across, 10.0)
                self.assertLessEqual(along, 2.0)

    def test_a_left_out_fragment_is_an_obstacle_exactly_as_a_declared_one_is(self) -> None:
        """Left out of the hull, not forgotten: the plan is the one made with the fragment as a declared obstacle."""
        from src.robot.grasping.generation.support_footprint import (
            SupportFootprintJaw,
            generate_support_footprint_grasps,
        )

        jaw = SupportFootprintJaw.from_robot_config(_hande())
        clean, leak = _clean_top_and_side(2), _leak(63, 63)
        mixed = generate_support_footprint_grasps(np.vstack([clean, leak]), support_height_mm=0.0, jaw=jaw)
        declared = generate_support_footprint_grasps(clean, support_height_mm=0.0, jaw=jaw,
                                                     rigid_obstacle_points_base_mm=leak)
        self.assertEqual(len(declared), len(mixed))
        for left, right in zip(mixed, declared):
            np.testing.assert_allclose(left.position_mm, right.position_mm, atol=1e-9)
            self.assertAlmostEqual(left.score, right.score, places=12)

    def test_a_top_face_split_by_a_depth_hole_is_kept_whole(self) -> None:
        """Both halves reach the top, so both are the part: a finger in the hole would land on the part."""
        from src.robot.grasping.generation.support_footprint import reconstruct_support_prism

        clean = _clean_top_and_side(2)
        # A 20 mm stripe of shiny top face with no depth, across the whole cube.
        holed = clean[~((np.abs(clean[:, 1] - _CENTRE[1]) < 10.0) & (clean[:, 2] > _SIZE - 3.0))]
        prism = reconstruct_support_prism(holed, 0.0)
        assert prism is not None
        self.assertEqual(0, prism.trimmed.shape[0])
        self.assertGreaterEqual(float(prism.hull[:, 1].max()) - _CENTRE[1], _SIZE / 2.0 - 2.0)
        self.assertLessEqual(float(prism.hull[:, 1].min()) - _CENTRE[1], -_SIZE / 2.0 + 2.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
