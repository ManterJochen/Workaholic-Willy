"""The tables the stack types by hand agree with Universal Robots' pinned description, read without ROS.

``scripts/curobo/_ur_description.py`` reads the vendored config (``test_ur_description_is_pinned.py`` holds its bytes):
the kinematics as joint origins, the joint limits with UR's ``!degrees`` tag, and which arm's folder each link's mesh
comes from. This file proves the reader against what the stack already trusts, so that the URDF writer built on it
starts from numbers that are shown to be the same numbers:

* ``dh_rows`` reproduces ``UR_DH_TABLES_M`` for every model that has a row, to float noise;
* ``mesh_owner`` reproduces the borrow map ``fetch_ur_meshes.py`` pins meshes by;
* the elbow is limited to half a turn in every model, the planning limit UM6 adopts for the planner.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import pathlib
import tempfile
import unittest

from src.robot.safety._ur_kinematics import UR_DH_TABLES_M

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODULE = _ROOT / "scripts" / "curobo" / "_ur_description.py"
_FETCH = _ROOT / "scripts" / "curobo" / "fetch_ur_meshes.py"
_TOLERANCE = 1e-12


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ur = _load(_MODULE, "_ur_description_for_tests")

    def test_degrees_are_radians(self) -> None:
        self.assertEqual(self.ur.parse_yaml("a: !degrees 180\nb: !degrees  -90.0\n"), {"a": math.pi, "b": -math.pi / 2})

    def test_a_plain_safe_load_cannot_read_the_files(self) -> None:
        """The control for the tag: without the reader's constructor the same text does not load at all."""
        import yaml

        with self.assertRaises(yaml.YAMLError):
            yaml.safe_load("a: !degrees 180\n")

    def test_an_unknown_tag_refuses_and_names_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "odd.yaml"
            path.write_text("a: !radians 3\n", encoding="utf-8")
            with self.assertRaises(self.ur.DescriptionError) as caught:
                self.ur.read_yaml(path)
        self.assertIn("odd.yaml", str(caught.exception))

    def test_the_models_are_the_pinned_models(self) -> None:
        models = self.ur.models()
        self.assertEqual(len(models), 14, models)
        self.assertLessEqual(set(UR_DH_TABLES_M), set(models))
        self.assertEqual(list(models), sorted(models))


class TheTablesAgreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ur = _load(_MODULE, "_ur_description_for_table_tests")
        cls.fetch = _load(_FETCH, "_fetch_ur_meshes_for_table_tests")

    def test_every_dh_row_the_stack_types_is_the_description_s(self) -> None:
        for model, authority in UR_DH_TABLES_M.items():
            rows = self.ur.dh_rows(model)
            self.assertEqual(len(rows), 6)
            for i, (row, ref) in enumerate(zip(rows, authority)):
                for value, expected, field in zip(row, (ref.a_m, ref.d_m, ref.alpha_rad), ("a", "d", "alpha")):
                    with self.subTest(model=model, row=i, field=field):
                        self.assertAlmostEqual(value, expected, delta=_TOLERANCE)

    def test_a_moved_origin_moves_the_row(self) -> None:
        """The control: a kinematics block with wrist_1 raised by a micrometre must not produce the authority's d4."""
        kinematics = dict(self.ur.load("ur5e").kinematics)
        kinematics["wrist_1"] = dataclasses.replace(kinematics["wrist_1"], z=kinematics["wrist_1"].z + 1e-6)
        rows = self.ur.dh_rows_from(kinematics, where="a perturbed ur5e")
        self.assertNotAlmostEqual(rows[3][1], UR_DH_TABLES_M["ur5e"][3].d_m, delta=_TOLERANCE)

    def test_kinematics_the_dh_form_cannot_hold_refuse(self) -> None:
        kinematics = dict(self.ur.load("ur5e").kinematics)
        kinematics["shoulder"] = dataclasses.replace(kinematics["shoulder"], x=1e-3)
        with self.assertRaises(self.ur.DescriptionError) as caught:
            self.ur.dh_rows_from(kinematics, where="a shifted ur5e shoulder")
        self.assertIn("shoulder", str(caught.exception))

    def test_the_mesh_owners_are_the_borrow_map(self) -> None:
        for model in UR_DH_TABLES_M:
            for link in self.ur.LINKS:
                path = self.ur.load(model).meshes[link].collision
                stem = pathlib.PurePosixPath(path).stem
                with self.subTest(model=model, link=link):
                    self.assertEqual(self.ur.mesh_owner(model, link), self.fetch.mesh_owner(model, stem))

    def test_a_borrowed_link_is_seen_as_borrowed(self) -> None:
        """The control for the loop above: ur16e's wrists sit in ur10e's folder, its forearm in its own."""
        self.assertEqual(self.ur.mesh_owner("ur16e", "wrist_1"), "ur10e")
        self.assertEqual(self.ur.mesh_owner("ur16e", "forearm"), "ur16e")

    def test_the_twins_are_twins(self) -> None:
        self.assertEqual(self.ur.dh_rows("ur7e"), self.ur.dh_rows("ur5e"))
        self.assertEqual(self.ur.dh_rows("ur12e"), self.ur.dh_rows("ur10e"))
        self.assertNotEqual(self.ur.dh_rows("ur7e"), self.ur.dh_rows("ur10e"))

    def test_every_elbow_is_limited_to_half_a_turn(self) -> None:
        for model in self.ur.models():
            limit = self.ur.load(model).joint_limits["elbow_joint"]
            with self.subTest(model=model):
                self.assertAlmostEqual(limit.lower_rad, -math.pi, delta=1e-12)
                self.assertAlmostEqual(limit.upper_rad, math.pi, delta=1e-12)
            other = self.ur.load(model).joint_limits["shoulder_pan_joint"]
            with self.subTest(model=model, joint="shoulder_pan_joint"):
                self.assertAlmostEqual(other.upper_rad, 2.0 * math.pi, delta=1e-12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
