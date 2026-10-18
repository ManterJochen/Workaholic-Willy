"""What the Isaac runners read off the arm and import from the drivers, counted per file, so the counts only fall.

``tests/test_robot_boundaries.py`` holds everything above the drivers to the arm contract (rule I: no driver import;
rule III: no member read off an arm that no Protocol declares), and it leaves ``src/willy_sim`` out on purpose:
the runners drive ``IsaacRobotArm`` directly, and the sim arm is not a ``PlannedArm`` until phase PY-b. Out of scope
also meant unmeasured, so a runner could start reading a new private member tomorrow and nothing would say so. These
are the three measures the contract test uses, taken over ``willy_sim`` with its own helpers
(``_arm_member``, ``_declared_members``, ``_imported_modules``), pinned per file as measured on 2026-10-14 (P0g slice
16): 16 private reads in 6 files, 151 reads of an undeclared member in 31 (140 of them ``session``, the Isaac session
``tests/test_every_planned_arm_stays_inside_the_fence.py`` counts as the one exposed link), and 17 imports of a driver or
gripper package in 11.

The lists may only shrink. A file that reads or imports more than it did fails, a file not listed that starts fails,
and a file that reads or imports fewer must lower its number in the same commit, so the list stays the measure. PY-b,
which moves the sim arm onto ``PlannedArm``, is where most of them go.
"""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path

from tests.test_robot_boundaries import _arm_member, _declared_members, _imported_modules

_SIM = Path(__file__).resolve().parents[1] / "src" / "willy_sim"

#: File -> count, measured 2026-10-14. Each list may only shrink.
_PRIVATE_READS: dict[str, int] = {
    "run_attribute_pick.py": 1,
    "run_dense_pick.py": 10,
    "run_eih_pick.py": 1,
    "run_fused_pick.py": 1,
    "run_m2_pick.py": 2,
    "run_suction_pick.py": 1,
}

_UNDECLARED_READS: dict[str, int] = {
    "_h39d_binpick_record.py": 1,
    "config.py": 1,
    "harness/bootstrap.py": 5,
    "inspect_wrist_cam.py": 7,
    "run_attribute_pick.py": 5,
    "run_bin_clearing_demo.py": 2,
    "run_clutter_demo.py": 1,
    "run_curobo_demo.py": 3,
    "run_dense_demo.py": 2,
    "run_dense_pick.py": 13,
    "run_eih_calibrate.py": 5,
    "run_eih_demo.py": 2,
    "run_eih_pick.py": 9,
    "run_eth_calibrate.py": 6,
    "run_fused_pick.py": 10,
    "run_industrial_bin_pick.py": 1,
    "run_industrial_suction_demo.py": 1,
    "run_industrial_suction_pick.py": 1,
    "run_klt_combined_demo.py": 8,
    "run_m1_pick.py": 2,
    "run_m2_pick.py": 5,
    "run_multiview_pick.py": 1,
    "run_occlusion_probe.py": 1,
    "run_pile_baseline.py": 4,
    "run_shake_label.py": 3,
    "run_sorting_demo.py": 3,
    "run_suction_demo.py": 7,
    "run_suction_pick.py": 35,
    "run_suction_probe.py": 1,
}

_DRIVER_IMPORTS: dict[str, int] = {
    "config.py": 1,
    "grippers.py": 2,
    "harness/bootstrap.py": 4,
    "harness/gate.py": 2,
    "harness/reach.py": 1,
    "perception/ground_truth.py": 1,
    "perception/vision.py": 1,
    "run_suction_pick.py": 1,
    "scene/build.py": 2,
    "scene/cameras.py": 1,
    "suction_mount.py": 1,
}


def _measured() -> tuple[Counter[str], Counter[str], Counter[str]]:
    """(private reads, undeclared reads, driver imports) per file of ``willy_sim``, relative to it."""
    declared = _declared_members()
    private: Counter[str] = Counter()
    undeclared: Counter[str] = Counter()
    imports: Counter[str] = Counter()
    for path in sorted(_SIM.rglob("*.py")):
        rel = path.relative_to(_SIM).as_posix()
        if "__pycache__" in rel:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for module in _imported_modules(node):
                if module.startswith(("src.robot.drivers", "src.robot.grippers")):
                    imports[rel] += 1
            member = _arm_member(node)
            if not member or member.startswith("__"):
                continue
            if member.startswith("_"):
                private[rel] += 1
            elif member not in declared:
                undeclared[rel] += 1
    return private, undeclared, imports


class TheSimRunnersAreCountedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.measured = dict(zip(("private reads", "undeclared reads", "driver imports"), _measured(), strict=True))
        self.listed = {"private reads": _PRIVATE_READS, "undeclared reads": _UNDECLARED_READS,
                       "driver imports": _DRIVER_IMPORTS}

    def test_no_file_reads_or_imports_more_than_it_did(self) -> None:
        for kind, measured in self.measured.items():
            with self.subTest(kind=kind):
                listed = self.listed[kind]
                grown = {name: (count, listed.get(name, 0)) for name, count in measured.items()
                         if count > listed.get(name, 0)}
                self.assertEqual({}, grown, "(now, listed): read or import through the contract instead")

    def test_the_lists_are_the_measure(self) -> None:
        """A number above what the file does would admit a new read later without anyone deciding it."""
        for kind, measured in self.measured.items():
            with self.subTest(kind=kind):
                stale = {name: (measured.get(name, 0), listed) for name, listed in self.listed[kind].items()
                         if measured.get(name, 0) != listed}
                self.assertEqual({}, stale, "(now, listed): lower the number, or remove the file")

    def test_the_scan_sees_the_runners(self) -> None:
        """The positive control: the scan finds the session reads it was written for, and a clean module has none."""
        _private, undeclared, _imports = self.measured.values()
        self.assertGreater(sum(undeclared.values()), 100, "the scan stopped seeing arm.session")
        tree = ast.parse("arm.session.step()\nx = arm._preflight\ny = arm.move\n")
        self.assertEqual(["_preflight", "move", "session"], sorted(m for n in ast.walk(tree) if (m := _arm_member(n))))


if __name__ == "__main__":
    unittest.main()
