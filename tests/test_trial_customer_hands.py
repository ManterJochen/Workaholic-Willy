"""Two customer hands whose answers are known before the chain runs (customer chain lane C5b).

``scripts/trial/customer_hands.py`` is a trial instrument. These tests hold its known answers to arithmetic and to the
committed Hand-E, each with a control showing the comparison can come out the other way.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "src" / "robot" / "safety" / "data"


def _tool():
    name = "_trial_customer_hands"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "trial" / "customer_hands.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AcmeDimsTests(unittest.TestCase):
    def test_acme_dims_is_a_valid_measured_hand_whose_boxes_are_known(self) -> None:
        import yaml

        from src.config.schema.grippers import GripperSpec, ParallelJawSpec
        from src.robot.safety.planning.robot.hand_from_dimensions import refusal_for

        tool = _tool()
        jaw = ParallelJawSpec.model_validate(tool.ACME_DIMS_JAW)
        self.assertIsNone(refusal_for(jaw))
        GripperSpec.model_validate(yaml.safe_load(tool.registry_text())["gripper"])
        self.assertEqual(tool.acme_dims_boxes(), tool.ACME_DIMS_BOXES)

    def test_the_expected_boxes_carry_the_inflation(self) -> None:
        """⭐ THE CONTROL: at no inflation the same numbers do not render, so they do not match by coincidence."""
        tool = _tool()
        self.assertNotEqual(tool.acme_dims_boxes(inflation_mm=0.0), tool.ACME_DIMS_BOXES)


class AcmeMeshTests(unittest.TestCase):
    def _inverse(self, out: Path, rotation: np.ndarray) -> dict:
        import trimesh

        tool = _tool()
        approach = "XYZ".index(tool.VENDOR_WORDS["approach"][1])
        parts = {}
        for part in tool.PARTS:
            mesh = trimesh.load(str(out / f"{part}.stl"), force="mesh", process=False)
            points = np.asarray(mesh.vertices, dtype=np.float64) * tool.VENDOR_SCALE_TO_MM
            points[:, approach] -= tool.VENDOR_MOUNT_FACE_MM
            parts[part] = points @ rotation.T
        return parts

    def _distance(self, parts: dict, source: dict) -> float:
        from scipy.spatial import cKDTree

        return max(float(cKDTree(np.asarray(source[f"{part}__v"], dtype=np.float64)).query(points)[0].max())
                   for part, points in parts.items())

    def test_the_vendor_stl_undone_by_its_recorded_transform_is_the_hande_bundle(self) -> None:
        tool = _tool()
        out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        tool.export_stl("robotiq_hande", out)
        with np.load(_DATA / "robotiq_hande_hand_meshes.npz") as data:
            source = {key: np.array(data[key]) for key in data.files}
        rotation = np.asarray(tool.VENDOR_ROTATION, dtype=np.float64)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
        self.assertLessEqual(self._distance(self._inverse(out, rotation), source), 1e-3)
        # ⭐ THE CONTROL: the inverse with the rotation instead of its transpose is not the hand.
        self.assertGreater(self._distance(self._inverse(out, rotation.T), source), 50.0)

    def test_the_mesh_writer_reads_the_recorded_axes_as_the_same_rotation(self) -> None:
        from src.robot.safety.planning.robot.hand_from_mesh import VendorAxes

        tool = _tool()
        axes = VendorAxes.from_words(**tool.VENDOR_WORDS)
        self.assertEqual(tuple(tuple(row) for row in axes.rotation), tool.VENDOR_ROTATION)

    def test_compare_reads_zero_for_one_hand_and_more_than_a_millimetre_for_two(self) -> None:
        tool = _tool()
        same = tool.compare("robotiq_hande", "robotiq_hande", tolerance_mm=0.0)
        self.assertEqual((same.worst_mm, same.exit_code), (0.0, 0))
        other = tool.compare("robotiq_hande", "schunk_egu50", tolerance_mm=1.0)
        self.assertGreater(other.worst_mm, 1.0)
        self.assertEqual(other.exit_code, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
