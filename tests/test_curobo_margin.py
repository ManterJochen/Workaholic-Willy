"""Telling the cuRobo planner what clearance the safety guard will demand.

cuRobo plans against a SPHERE model and returns the first path it thinks is free; the guard then
re-checks that path's final configuration against the EXACT MESHES and rejects anything inside
``min_distance_mm``. Nothing connected the two, so the planner kept offering configurations the guard
was always going to refuse -- measured on-box as ``forearm|wrist_2`` clearances of 9.44-9.47 mm against
a 10.000 mm margin, losing three of ten picks to what read like bad grasping.

cuRobo scores a pair as ``sphere_distance - (buffer_a + buffer_b)``, so half the margin per link yields
exactly the margin per pair. These tests pin that arithmetic and, just as importantly, the cases where
the config must be left alone.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.robot.safety.planning._curobo_margin import (
    apply_self_collision_margin,
    derive_margin_config_file,
)


def _config(**buffers: float) -> dict:
    return {
        "robot_cfg": {
            "kinematics": {
                "self_collision_buffer": dict(buffers or {"forearm_link": 0.0, "shoulder_link": 0.07}),
                "collision_spheres": {"forearm_link": [{"center": [0, 0, 0], "radius": 0.05}]},
            },
            "dynamics": {"max_acc": 1.0},
        },
    }


def _buffers(config: dict) -> dict:
    return config["robot_cfg"]["kinematics"]["self_collision_buffer"]


class MarginArithmeticTests(unittest.TestCase):
    def test_half_the_margin_lands_on_each_link(self) -> None:
        """cuRobo subtracts BOTH links' buffers from a pair's distance, so half each == the full margin
        for the pair -- which is what the guard measures."""
        out, adjusted = apply_self_collision_margin(_config(a=0.0, b=0.0), margin_mm=10.0)
        self.assertEqual(adjusted, 2)
        self.assertAlmostEqual(_buffers(out)["a"], 0.005)
        self.assertAlmostEqual(_buffers(out)["b"], 0.005)
        self.assertAlmostEqual(_buffers(out)["a"] + _buffers(out)["b"], 0.010)  # == 10 mm per PAIR

    def test_existing_buffers_are_added_to_not_replaced(self) -> None:
        """Vendor buffers encode real geometry (the UR shoulder ships 0.07 m). Overwriting them would
        trade one engine disagreement for a worse one."""
        out, _ = apply_self_collision_margin(_config(shoulder_link=0.07, forearm_link=0.0), 10.0)
        self.assertAlmostEqual(_buffers(out)["shoulder_link"], 0.075)
        self.assertAlmostEqual(_buffers(out)["forearm_link"], 0.005)

    def test_the_source_config_is_never_mutated(self) -> None:
        src = _config(a=0.0)
        apply_self_collision_margin(src, 10.0)
        self.assertEqual(_buffers(src)["a"], 0.0)

    def test_nothing_else_in_the_config_is_touched(self) -> None:
        src = _config(a=0.0)
        out, _ = apply_self_collision_margin(src, 10.0)
        self.assertEqual(
            out["robot_cfg"]["kinematics"]["collision_spheres"],
            src["robot_cfg"]["kinematics"]["collision_spheres"],
        )
        self.assertEqual(out["robot_cfg"]["dynamics"], src["robot_cfg"]["dynamics"])


class LeaveItAloneTests(unittest.TestCase):
    def test_zero_margin_is_a_no_op(self) -> None:
        """The default. A cell without a self-collision margin must get the planner it always got."""
        src = _config(a=0.0, b=0.07)
        out, adjusted = apply_self_collision_margin(src, 0.0)
        self.assertEqual(adjusted, 0)
        self.assertEqual(out, src)

    def test_a_negative_margin_is_a_no_op_not_a_shrink(self) -> None:
        """Shrinking cuRobo's buffers would make it plan CLOSER than the vendor intended -- the one
        direction this must never move."""
        src = _config(a=0.05)
        out, adjusted = apply_self_collision_margin(src, -10.0)
        self.assertEqual(adjusted, 0)
        self.assertEqual(_buffers(out)["a"], 0.05)

    def test_a_config_without_the_buffer_block_is_returned_unchanged(self) -> None:
        """Inventing the key on a third-party robot description would be a worse failure than visibly
        doing nothing -- the caller logs the zero and the operator can see the margin was not applied."""
        src = {"robot_cfg": {"kinematics": {"collision_spheres": {}}}}
        out, adjusted = apply_self_collision_margin(src, 10.0)
        self.assertEqual(adjusted, 0)
        self.assertEqual(out, src)

    def test_a_structurally_foreign_config_does_not_raise(self) -> None:
        for src in ({}, {"robot_cfg": {}}, {"robot_cfg": {"kinematics": None}}):
            out, adjusted = apply_self_collision_margin(src, 10.0)
            self.assertEqual(adjusted, 0)
            self.assertEqual(out, src)

    def test_non_numeric_buffer_entries_are_skipped_not_coerced(self) -> None:
        src = _config()
        src["robot_cfg"]["kinematics"]["self_collision_buffer"]["odd"] = "0.01"
        out, adjusted = apply_self_collision_margin(src, 10.0)
        self.assertEqual(_buffers(out)["odd"], "0.01")
        self.assertEqual(adjusted, 2)  # the two numeric links, not the string


class FileRoundTripTests(unittest.TestCase):
    def test_the_derived_file_is_loadable_and_carries_the_margin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "robot.yml"
            dst = Path(tmp) / "robot_margin.yml"
            src.write_text(yaml.safe_dump(_config(a=0.0, b=0.07)), encoding="utf-8")
            adjusted = derive_margin_config_file(str(src), str(dst), 10.0)
            self.assertEqual(adjusted, 2)
            loaded = yaml.safe_load(dst.read_text(encoding="utf-8"))
            self.assertAlmostEqual(_buffers(loaded)["a"], 0.005)
            self.assertAlmostEqual(_buffers(loaded)["b"], 0.075)


class SidecarImportabilityTests(unittest.TestCase):
    def test_the_module_is_importable_without_the_willy_package(self) -> None:
        """The sidecar runs on py3.10 in the cuRobo env and CANNOT import Willy -- it picks this module
        up as a plain sibling of the server script. So it must not reach for the package."""
        source = Path(
            "src/robot/safety/planning/_curobo_margin.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from backend", source)
        self.assertNotIn("from .", source)
        self.assertNotIn("import backend", source)


if __name__ == "__main__":
    unittest.main()
