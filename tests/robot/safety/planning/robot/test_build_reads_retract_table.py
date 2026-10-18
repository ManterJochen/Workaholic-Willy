"""What the builder declares about the planner rather than inheriting it (UM5, UM6, B4).

Two things a descriptor used to carry by accident: the elbow limit of whichever Isaac description was copied, and
``position_limit_clip`` from whichever template the content happened to hold. Both decide what the planner may propose,
so the builder writes them itself. Source checks, because the builder runs under the cuRobo interpreter where this
repository cannot be imported; the arithmetic they name is unit tested in tests/scripts/arms/test_planner_joint_envelope.py.
"""

from __future__ import annotations

import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
_BUILD = _ROOT / "scripts" / "arms" / "descriptor.py"


class TheBuilderDeclaresThePlannerEnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _BUILD.read_text(encoding="utf-8")

    def test_the_source_was_read(self) -> None:
        """The control for every scan below: this file is the builder and nothing else."""
        self.assertIn("YML_OUT.write_text(", self.source)
        self.assertIn("_ur_template.yml", self.source)

    def test_the_elbow_is_clamped_before_the_description_is_written(self) -> None:
        """A clamp after the write leaves the file on disk with the wide elbow, and the descriptor reads the file."""
        self.assertIn("clamp_to_planner_limits(urdf", self.source)
        self.assertLess(self.source.index("clamp_to_planner_limits(urdf"), self.source.index("URDF_OUT.write_text("))

    def test_the_clip_is_written_and_not_inherited_from_a_template(self) -> None:
        self.assertIn('cs["position_limit_clip"] = _arm_file.position_limit_clip_rad(MODEL)', self.source)

    def test_the_provenance_records_what_the_planner_may_do(self) -> None:
        for key in ('"planner_joint_limits"', '"position_limit_clip"'):
            with self.subTest(key=key):
                self.assertIn(key, self.source)


class TheBuilderReadsTheCommittedRetractTests(unittest.TestCase):
    """UM5: the descriptor carries the pose the rule chose on the exact meshes, and no silent fallback to Isaac's."""

    def setUp(self) -> None:
        self.source = _BUILD.read_text(encoding="utf-8")

    def test_isaacs_pose_is_no_longer_the_retract(self) -> None:
        self.assertNotIn('lula["default_q"]', self.source)
        self.assertIn("read_arm_retract(", self.source)

    def test_the_spheres_come_from_the_committed_fit(self) -> None:
        """The control: the retract left Isaac's Lula file in UM5 and the spheres in B6; P0f F5 took the file out."""
        self.assertIn("link_spheres(_fitted_doc", self.source)
        self.assertNotIn('lula["collision_spheres"]', self.source)

    def test_the_provenance_records_the_table_it_read(self) -> None:
        for key in ('"retract"', "arm_rows_sha256", "judged_hands"):
            with self.subTest(key=key):
                self.assertIn(key, self.source)


class TheReaderRefusesWhatWasNotJudgedTests(unittest.TestCase):
    def setUp(self) -> None:
        import importlib.util
        import sys
        import tempfile

        path = _ROOT / "scripts" / "arms" / "_retract_rule.py"
        spec = importlib.util.spec_from_file_location("_retract_rule_for_builder", path)
        assert spec is not None and spec.loader is not None
        self.rule = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.rule
        spec.loader.exec_module(self.rule)
        self.table = Path(self.enterContext(tempfile.TemporaryDirectory())) / "retract.yaml"
        self.table.write_text(
            "rule:\n  guard_margin_mm: 10.0\n  reserve_mm: 3.0\n"
            "  anchor_source: Isaac Lula default_q\n"
            "  pairs_with_no_retract: []\n"
            "retracts:\n"
            "  - arm: ur5\n    hand: robotiq_2f85\n    plate_mm: 0.0\n    planner_margin_mm: 10.0\n"
            "    retract: [0.0, -1.0, 0.9, 0.0, 0.0, -1.5]\n    steps: [0, 0, 0, 0, -6]\n"
            "    anchor: [0.0, -1.0, 0.9, 0.0, 0.0, 0.0]\n"
            "  - arm: ur5\n    hand: robotiq_hande\n    plate_mm: 0.0\n    planner_margin_mm: 10.0\n"
            "    retract: [0.0, -1.0, 0.9, 0.0, 0.0, -1.5]\n    steps: [0, 0, 0, 0, -6]\n"
            "    anchor: [0.0, -1.0, 0.9, 0.0, 0.0, 0.0]\n"
            "  - arm: ur5\n    hand: robotiq_hande\n    plate_mm: 20.0\n    planner_margin_mm: 10.0\n"
            "    retract: [0.0, -1.0, 0.9, 0.0, 0.0, -1.5]\n    steps: [0, 0, 0, 0, -6]\n"
            "    anchor: [0.0, -1.0, 0.9, 0.0, 0.0, 0.0]\n",
            encoding="utf-8",
        )

    def test_a_judged_arm_reads_back_with_what_it_was_judged_against(self) -> None:
        row = self.rule.read_arm_retract(self.table, "ur5")
        self.assertEqual(row.retract, [0.0, -1.0, 0.9, 0.0, 0.0, -1.5])
        self.assertEqual(row.steps, [0, 0, 0, 0, -6])
        self.assertEqual(row.hands, {"robotiq_2f85": [0.0], "robotiq_hande": [0.0, 20.0]})
        self.assertEqual(len(row.rows_sha256), 64)

    def test_an_arm_the_table_never_judged_refuses_and_names_the_generator(self) -> None:
        """⭐ THE CONTROL: no silent fallback. An arm with no row has no retract, and the build stops."""
        with self.assertRaises(self.rule.RetractMissing) as ctx:
            self.rule.read_arm_retract(self.table, "ur10")
        self.assertIn("scripts/arms/retract.py", str(ctx.exception))
        self.assertIn("ur10", str(ctx.exception))



class TheDescriptorProvenanceHangsOnItsOwnRowsTests(unittest.TestCase):
    """A descriptor carries its arm's retract provenance into ``composed_sha256``, so what moves it re-gates evidence.

    MEASURED 2026-09-23 on the committed table, before this: the hash moved between an LF and a CRLF checkout of the
    same table, and with a row of another arm, and the 47 committed evidence files matched the descriptors built on
    2026-09-16 while 0 matched a descriptor built from the same scripts at HEAD (aurora; Willy's 57 the same way,
    measured 2026-10-13).
    """

    def setUp(self) -> None:
        import tempfile

        import yaml

        from src.robot.safety.planning.robot import retract_table

        self.reader = retract_table
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.lf = retract_table.TABLE_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.table = yaml.safe_load(self.lf.decode("utf-8"))

    def _hash(self, name: str, data: bytes, arm: str = "ur5e") -> str:
        path = self.folder / name
        path.write_bytes(data)
        return self.reader.read_arm_retract(path, arm).rows_sha256

    def _edited(self, arm: str) -> bytes:
        import copy

        import yaml

        table = copy.deepcopy(self.table)
        row = next(row for row in table["retracts"] if row["arm"] == arm)
        row["retract"][5] = float(row["retract"][5]) + 0.001
        return yaml.safe_dump(table, sort_keys=False, default_flow_style=False).encode("utf-8")

    def test_the_checkout_s_line_endings_are_outside_it(self) -> None:
        self.assertEqual(self._hash("lf.yaml", self.lf), self._hash("crlf.yaml", self.lf.replace(b"\n", b"\r\n")))

    def test_another_arm_s_rows_are_outside_it(self) -> None:
        self.assertEqual(self._hash("lf.yaml", self.lf), self._hash("ur3.yaml", self._edited("ur3")))

    def test_its_own_rows_are_inside_it(self) -> None:
        """The control: a hash that no row moves would let a re-judged retract keep its old evidence."""
        self.assertNotEqual(self._hash("lf.yaml", self.lf), self._hash("ur5e.yaml", self._edited("ur5e")))


if __name__ == "__main__":
    unittest.main()
