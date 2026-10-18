"""`deep propose` judged grasps it had emitted in a randomly rotated frame.

⛔⛔ **THIS INVALIDATED THE ARC'S HEADLINE NUMBER.** `SampleSpec.rotate_z` defaults to True because
it is a TRAINING device: `build_sample` draws `rng.uniform(0, 2*pi)` and rotates the cloud, the
normals and the grasp targets about the cloud centroid (dataset.py:633-642). At training time that
costs nothing, because the labels rotate with the scene. `propose` served `spec = loaded.sample`
verbatim, so the net saw a rotated scene and its decoded pose came back in that rotated frame, while
`_nearest_instance` and the physics referee both judged it against the UNROTATED cloud.

MEASURED on this repository's own 40-scene proposal file (logs/dl/chain/proposals.jsonl), by
searching one angle per scene: the median distance from a proposal to the nearest object point falls
from **46.84 mm to 2.58 mm** once a single z-rotation is undone, in **40 of 40 scenes**, and the
recovered angles are spread over the whole circle. The grasps were never aimed at the table. They
were aimed correctly, in a frame nobody turned back.

⚠ SO "0 held of 307 against a 22.53 % label control" IS A MEASUREMENT OF THIS DEFECT, not of the
model. Nothing about the model's quality follows from it in either direction until the run is
repeated. The label control is unaffected: it never went through `propose`.

⚠ THE CELL NEVER HAD THIS. `calculator.py` overrides `rotate_z` with the reason written beside it
("augmentation is a training device; rotating a live scene would move the grasp it returns"), so the
runbook's judgment stage and its deployment stage were measuring two different systems.
"""

from __future__ import annotations

import ast
import dataclasses
import unittest
from pathlib import Path

import numpy as np

from src.robot.grasping.deep.corpus.sample import SampleSpec

_PROPOSE = Path("src/robot/grasping/deep/eval/propose.py")
_CALCULATOR = Path("src/robot/grasping/deep/calculator.py")


def _scene(rng: np.random.Generator) -> dict:
    """A small scene with one object above a support plane, shaped like the corpus."""
    table = np.stack([
        rng.uniform(-200.0, 200.0, 400),
        rng.uniform(-200.0, 200.0, 400),
        np.zeros(400),
    ], axis=1)
    # Deliberately NOT rotationally symmetric, so a rotation is detectable at all.
    box = np.stack([
        rng.uniform(-60.0, 60.0, 200),
        rng.uniform(-8.0, 8.0, 200),
        rng.uniform(20.0, 40.0, 200),
    ], axis=1)
    points = np.vstack([box, table])
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
    return {
        "points_mm": points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "normal_valid": np.ones(len(points), dtype=bool),
        "view_count": np.ones(len(points), dtype=np.uint8),
        "instance_id": np.concatenate([
            np.zeros(len(box), dtype=np.int16), np.full(len(table), -1, dtype=np.int16)]),
        "object_instance": np.asarray([0], dtype=np.int16),
        "object_asset_id": np.asarray(["box"], dtype="<U80"),
        "grasp_position_mm": np.zeros((0, 3), dtype=np.float32),
        "grasp_approach": np.zeros((0, 3), dtype=np.float32),
        "grasp_axis": np.zeros((0, 3), dtype=np.float32),
        "grasp_width_mm": np.zeros(0, dtype=np.float32),
        "grasp_instance": np.zeros(0, dtype=np.int16),
        "contact_points_mm": np.zeros((0, 3), dtype=np.float32),
        "contact_grasp_index": np.zeros(0, dtype=np.int32),
    }


def _principal_angle(points: np.ndarray) -> float:
    """Orientation of the cloud's dominant xy axis, in degrees, modulo 180."""
    flat = np.asarray(points, dtype=np.float64)[:, :2]
    flat = flat - flat.mean(axis=0)
    _values, vectors = np.linalg.eigh(flat.T @ flat)
    return float(np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])) % 180.0)


def _axial_spread_deg(angles: list[float]) -> float:
    """Circular spread of AXIS directions, in degrees.

    ⚠ `max - min` IS WRONG HERE and it cost this test a false failure: an axis lives on a circle of
    period 180, so 178 deg and 15 deg are 17 deg apart, not 163. The doubled-angle transform maps
    period-180 axes onto a full circle, where the mean resultant length is a real dispersion measure:
    R near 1 is tight, R near 0 is scattered.
    """
    doubled = np.deg2rad(np.asarray(angles, dtype=np.float64) * 2.0)
    resultant = float(np.hypot(np.cos(doubled).mean(), np.sin(doubled).mean()))
    # Circular standard deviation, halved back into axis degrees.
    return float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12))))) / 2.0)


class TheAugmentationIsOffAtServeTimeTests(unittest.TestCase):

    def test_propose_overrides_rotate_z(self) -> None:
        """⭐ THE REPAIR. Read off the AST rather than the text, because the comment that explains the
        defect necessarily contains the word `rotate_z` and a substring search would pass on the
        defect and fail on the fix. That trap has already cost this repository two tests in one day.
        """
        tree = ast.parse(_PROPOSE.read_text(encoding="utf-8"))
        replaces = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "replace"
            and any(kw.arg == "rotate_z" for kw in node.keywords)
        ]
        self.assertTrue(replaces, "propose does not override rotate_z at all")
        for call in replaces:
            keyword = next(kw for kw in call.keywords if kw.arg == "rotate_z")
            self.assertIsInstance(keyword.value, ast.Constant)
            self.assertFalse(keyword.value.value, "propose serves a ROTATED scene")

    def test_the_cell_overrides_it_too_and_always_did(self) -> None:
        """The two must agree, and they did not: that disagreement is how the runbook's judgment
        stage and its deployment stage came to measure different systems.

        The cell builds its spec from a dict literal, so the value is read off that dict rather than
        off a `dataclasses.replace` call.
        """
        tree = ast.parse(_CALCULATOR.read_text(encoding="utf-8"))
        found = [
            value.value
            for node in ast.walk(tree) if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "rotate_z"
            and isinstance(value, ast.Constant)
        ]

        self.assertTrue(found, "the cell no longer sets rotate_z when it builds its sample spec")
        self.assertFalse(any(found), "the cell serves a ROTATED scene")


class TheAugmentationReallyMovesTheSceneTests(unittest.TestCase):
    """⚠ THE CONTROL. If the augmentation did not move anything, the repair above would be
    decoration and the 46.84 mm would need another explanation."""

    def _angles(self, *, rotate: bool) -> list[float]:
        from src.robot.grasping.deep.corpus.sample import build_sample

        scene = _scene(np.random.default_rng(7))
        spec = dataclasses.replace(SampleSpec(grasp_set=True, points=512), rotate_z=rotate)
        return [
            _principal_angle(build_sample(scene, np.random.default_rng(seed), spec,
                                          target_instance=0)["points_m"])
            for seed in range(6)
        ]

    def test_rotate_z_scatters_the_scene_orientation(self) -> None:
        angles = self._angles(rotate=True)
        spread = _axial_spread_deg(angles)
        self.assertGreater(spread, 25.0,
                           f"the augmentation did not rotate anything: {angles} -> {spread:.1f} deg")

    def test_without_it_the_orientation_HOLDS(self) -> None:
        """The other half of the control, and the half that caught a defect in this test rather than
        in the code: the residual spread is point sampling, not rotation."""
        angles = self._angles(rotate=False)
        spread = _axial_spread_deg(angles)
        self.assertLess(spread, 20.0,
                        f"the scene moved with the augmentation off: {angles} -> {spread:.1f} deg")

    def test_the_two_are_SEPARATED_by_more_than_their_own_noise(self) -> None:
        """⭐ The comparison the two thresholds above stand on. Fixed numbers can drift into being
        vacuous; this one asks the question directly."""
        on = _axial_spread_deg(self._angles(rotate=True))
        off = _axial_spread_deg(self._angles(rotate=False))
        self.assertGreater(on, off * 2.0,
                           f"rotated {on:.1f} deg against unrotated {off:.1f} deg is not a difference")

    def test_the_default_is_ON_which_is_why_this_was_silent(self) -> None:
        """`rotate_z` defaults True because training wants it. Every artifact carries that default,
        so serving the artifact's spec verbatim serves a rotated scene."""
        self.assertTrue(SampleSpec().rotate_z)


class WhatARotatedPoseLooksLikeTests(unittest.TestCase):

    def test_a_rotated_pose_measured_against_an_unrotated_cloud_MISSES(self) -> None:
        """The defect's shape in one assertion: a pose that is correct in its own frame reads as a
        pose aimed at nothing, and that is what the referee and `_nearest_instance` both saw."""
        rng = np.random.default_rng(3)
        scene = _scene(rng)
        objects = scene["points_mm"][scene["instance_id"] >= 0].astype(np.float64)
        on_object = objects[0]

        centre = scene["points_mm"].astype(np.float64).mean(axis=0)
        centre[2] = 0.0
        angle = np.deg2rad(90.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                             [np.sin(angle), np.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
        rotated = (on_object - centre) @ rotation.T + centre

        as_written = np.linalg.norm(objects - rotated, axis=1).min()
        corrected = np.linalg.norm(objects - on_object, axis=1).min()

        self.assertEqual(corrected, 0.0, "the control pose is not on the object")
        self.assertGreater(as_written, 20.0,
                           "a rotated pose was not far from the cloud, so the 46.84 mm measured on "
                           "the real proposals needs another explanation")


if __name__ == "__main__":
    unittest.main()
