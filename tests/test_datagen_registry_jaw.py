"""The labeller takes a registry hand, checks a measured housing, and each jaw's corridor is its own.

Lane (i) D4. `label_dataset(jaw=)` knew only the three procedural jaws, so no hand this repository owns besides the
2F-85 could be labelled. A registry file (config/data/grippers) now builds the `JawModel`, with the labeller's policy at
its defaults. The housing is checked against the table exactly for a hand whose palm was measured, which leaves every
2F-85 label as it was, because the 2F-85's palm numbers are an estimate. The approach corridor a grasp table records is
the jaw's own reach plus the approach clearance, where it was the 2F-85's fixed 113.37 mm for every jaw.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.config.grippers import load_gripper
from datagen.grasps.shapes import Solid
from datagen.grasps.verdict import PROCEDURAL_JAWS, JawGrasp, JawModel, check_jaw_grasp


def _hande_numbers() -> JawModel:
    """The Hand-E file's numbers typed out, with the labeller's policy at its defaults and no palm check."""
    return JawModel(aperture_mm=49.99, min_width_mm=5.0, finger_ahead_mm=10.45, finger_behind_mm=36.53,
                    finger_thickness_mm=10.80, finger_width_mm=29.24, palm_depth_mm=104.12, palm_width_mm=75.0,
                    friction_coefficient=0.5, pad_ahead_mm=10.45, pad_behind_mm=10.46)


def _from_file(model: str) -> JawModel:
    return JawModel.from_spec(load_gripper(model, aliases=False).jaw)  # type: ignore[attr-defined]


class ARegistryHandBuildsTheJawTests(unittest.TestCase):
    def test_the_2f85_file_is_the_labellers_own_jaw(self) -> None:
        """Byte-identical by construction: the file builds exactly the jaw every 2F-85 label was made with."""
        self.assertEqual(_from_file("robotiq_2f85"), JawModel())

    def test_the_hande_file_builds_its_measured_jaw_with_the_palm_checked(self) -> None:
        self.assertEqual(_from_file("robotiq_hande"), dataclasses.replace(_hande_numbers(), check_palm=True))

    def test_the_lookup_takes_a_registry_hand_or_a_procedural_jaw(self) -> None:
        from datagen.grasps.labels import jaw_model_for  # type: ignore[attr-defined]

        self.assertEqual(jaw_model_for("robotiq_hande"), _from_file("robotiq_hande"))
        self.assertIs(jaw_model_for("narrow_55"), PROCEDURAL_JAWS["narrow_55"])
        for name in ("no_such_hand", "2f85"):
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                jaw_model_for(name)
            self.assertIn("unknown jaw", str(caught.exception))


class _LabelRun(unittest.TestCase):
    def _label(self, jaw: str | None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        """`label_dataset` over one scene, with the manifest and the assets stubbed; returns the call it made."""
        from datagen.grasps import labels

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "v5_s0"
        scene = root / "scenes" / "bin_000001"
        scene.mkdir(parents=True)
        (scene / "scene.json").write_text(json.dumps({"spec": {"family": "bin"}}), encoding="utf-8")
        seen: dict[str, Any] = {}

        def label(payload: Any, scene_id: str, **kwargs: Any) -> tuple[list[Any], dict[str, int]]:
            seen.update(kwargs)
            return [], {}

        assets = SimpleNamespace(label=label, geometry=lambda payload, scene_id: SimpleNamespace(objects={}))
        with mock.patch("datagen.prompts.build.manifest_for", return_value=({}, [])), \
             mock.patch.object(labels, "scene_assets", return_value=assets):
            report = labels.label_dataset(root, jaw=jaw)
        return root, report, seen


class LabelDatasetTakesARegistryHandTests(_LabelRun):
    def test_a_registry_hand_is_labelled_into_its_own_file_with_jaw_rows_only(self) -> None:
        root, report, seen = self._label("robotiq_hande")
        self.assertTrue((root / "grasps_jaw_robotiq_hande.jsonl").is_file())
        self.assertEqual(report["jaw_model"], "robotiq_hande")
        self.assertEqual(seen["model"], _from_file("robotiq_hande"))
        self.assertIs(seen["suction"], False, "suction does not depend on the hand; grasps.jsonl already carries it")

    def test_the_corpus_run_still_labels_suction(self) -> None:
        _root, _report, seen = self._label(None)
        self.assertIsNone(seen["model"])
        self.assertIs(seen["suction"], True)


class APerJawSceneWritesNoSuctionTests(unittest.TestCase):
    def _scene(self, **kwargs: Any) -> tuple[mock.MagicMock, mock.MagicMock]:
        from datagen.grasps import labels

        geometry = SimpleNamespace(objects={0: object()})
        with mock.patch.object(labels, "scene_geometry", return_value=geometry), \
             mock.patch.object(labels, "label_jaw_grasps", return_value=([], {})) as jaw, \
             mock.patch.object(labels, "label_suction_grasps", return_value=([], {})) as suction:
            labels.label_scene({}, {}, "bin_000001", **kwargs)
        return jaw, suction

    def test_a_scene_labelled_without_suction_skips_it(self) -> None:
        jaw, suction = self._scene(suction=False)
        jaw.assert_called_once()
        suction.assert_not_called()

    def test_a_scene_labels_suction_by_default(self) -> None:
        """The control."""
        _jaw, suction = self._scene()
        suction.assert_called_once()


#: A grasp tilted 80 degrees from straight down, closing horizontally on a 20 mm thick box standing on the table.
#: Worked out for the Hand-E: the lowest finger sample stays 3.8 mm above the table, and the housing, 75 mm across the
#: nearly vertical binormal, reaches 10.5 mm under it.
_TILT = np.radians(80.0)


def _box() -> Solid:
    return Solid("box", np.array([10.0, 10.0, 20.0]), np.eye(3), np.array([0.0, 0.0, 20.0]))


def _tilted_grasp() -> JawGrasp:
    approach = np.array([np.sin(_TILT), 0.0, -np.cos(_TILT)])
    return JawGrasp(np.array([0.0, 0.0, 20.0]), approach, np.array([0.0, 1.0, 0.0]), 20.0)


class AMeasuredHousingIsCheckedTests(unittest.TestCase):
    def test_the_grasp_closes_when_the_palm_is_not_checked(self) -> None:
        """The control: every check but the housing passes, so the refusal below is the palm's alone."""
        verdict = check_jaw_grasp(_tilted_grasp(), _box(), model=_hande_numbers())
        self.assertTrue(verdict.ok, f"{verdict.reason}: {verdict.detail}")

    def test_a_measured_housing_under_the_table_refuses(self) -> None:
        palm_checked = dataclasses.replace(_hande_numbers(), check_palm=True)
        verdict = check_jaw_grasp(_tilted_grasp(), _box(), model=palm_checked)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "palm_under_table")


class TheCorridorFollowsTheJawTests(unittest.TestCase):
    _FROM_BELOW = {"scene_id": "bin_000001", "instance_id": 0, "kind": "jaw", "_row_index": 0,
                   "position_mm": [0.0, 0.0, 115.0], "approach": [0.0, 0.0, 1.0], "closing_axis": [1.0, 0.0, 0.0],
                   "width_mm": 30.0}

    def test_the_2f85_corridor_is_the_constant(self) -> None:
        """The control: 33.37 + 80.0, so a table made without a jaw is what it was."""
        from datagen.corpus import clouds

        self.assertAlmostEqual(JawModel().finger_behind_mm + JawModel().approach_clearance_mm,
                               clouds._APPROACH_START_MM)  # noqa: SLF001
        self.assertEqual(clouds.scene_grasp_table([dict(self._FROM_BELOW)])["grasp_approach_admissible"].tolist(),
                         [True])

    def test_a_table_for_the_hande_starts_its_corridor_at_its_own_reach(self) -> None:
        """36.53 + 80.0 = 116.53 mm: a grasp 115 mm up, approached from below, starts under the table for the Hand-E."""
        from datagen.corpus import clouds

        table = clouds.scene_grasp_table([dict(self._FROM_BELOW)], model=_hande_numbers())
        self.assertEqual(table["grasp_approach_admissible"].tolist(), [False])


if __name__ == "__main__":
    unittest.main()
