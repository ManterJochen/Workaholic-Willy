"""The trial copy is provably the tree, and the original provably did not move (customer chain lane C5a).

``scripts/trial/tree_copy.py`` is a trial instrument, not product code. These tests hold its four questions to cases
with known answers, each with the control that shows the question can come out the other way.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]


def _tool():
    name = "_trial_tree_copy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "trial" / "tree_copy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_repo(case: unittest.TestCase) -> Path:
    root = Path(case.enterContext(tempfile.TemporaryDirectory())) / "orig"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_bytes(b"ignored.txt\n")
    (root / "tracked.txt").write_bytes(b"one\ntwo\n")
    (root / "untracked.txt").write_bytes(b"three\n")
    (root / "ignored.txt").write_bytes(b"never compared\n")
    subprocess.run(["git", "-C", str(root), "add", "tracked.txt", ".gitignore"], check=True)
    return root


@unittest.skipIf(shutil.which("git") is None, "git is not on PATH, and every question here is asked of git")
class TheCopyIsTheTreeTests(unittest.TestCase):
    def test_a_complete_copy_passes_and_a_missing_or_rewritten_file_is_named(self) -> None:
        tool = _tool()
        original = _git_repo(self)
        copy = original.parent / "copy"
        shutil.copytree(original, copy, ignore=shutil.ignore_patterns(".git", "ignored.txt"))
        self.assertEqual(tool.complete(original, copy).exit_code, 0, "the ignored file is not git's to demand")

        (copy / "tracked.txt").write_bytes(b"one\r\ntwo\r\n")
        report = tool.complete(original, copy)
        self.assertEqual((report.exit_code, report.differing), (1, ("tracked.txt",)))

        (copy / "tracked.txt").unlink()
        report = tool.complete(original, copy)
        self.assertEqual((report.exit_code, report.missing), (1, ("tracked.txt",)))

    def test_a_copy_inside_the_original_is_refused(self) -> None:
        """MEASURED 2026-09-17: a drive relative target put the copy inside the original tree."""
        tool = _tool()
        original = _git_repo(self)
        with self.assertRaises(tool.CannotAsk):
            tool.copy_tree(original, original / "trial_copy")
        self.assertFalse((original / "trial_copy").exists())
        # THE CONTROL: beside the original, the same copy is made.
        self.assertGreater(tool.copy_tree(original, original.parent / "trial_copy"), 0)

    def test_where_names_a_path_that_resolves_outside_the_expected_root(self) -> None:
        tool = _tool()
        report = tool.where(expect_root=_REPO)
        self.assertEqual(report.exit_code, 0, report.render())
        from src.robot.safety.planning import environment

        elsewhere = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(environment, "COLLISION_MESH_DIR", elsewhere):
            report = tool.where(expect_root=_REPO)
        self.assertEqual(report.exit_code, 1)
        self.assertIn(str(elsewhere.resolve()), report.render())

    def test_touched_tests_hold_the_thirteen_files_the_audit_named(self) -> None:
        tool = _tool()
        thirteen = {f"tests/{name}.py" for name in (
            "test_gripper_hands_are_measured", "test_bundle_provenance", "test_gripper_spheres",
            "test_one_hand_one_name", "test_self_filter_envelope", "test_the_declared_frame_agrees_with_the_hand",
            "test_retract_table", "test_committed_evidence", "test_a_hand_says_where_it_came_from",
            "test_the_planner_starts_only_with_evidence", "test_gripper_registry", "test_sim_mount_is_derived",
            "test_arm_profiles")}
        with mock.patch.object(tool, "_git", return_value=b""):
            derived = set(tool.touched_tests(_REPO))
            self.assertLessEqual(thirteen, derived)
            # ⭐ THE CONTROL, by removal: with no patterns the status alone (patched empty) derives none of them.
            self.assertFalse(thirteen & set(tool.touched_tests(_REPO, patterns=())))

    def test_compare_names_a_watched_file_or_descriptor_that_changed(self) -> None:
        tool = _tool()
        tree = _git_repo(self)
        watched = tree / "tests"
        watched.mkdir()
        (watched / "test_x.py").write_bytes(b"x = 1\n")
        content = tree.parent / "content"
        (content / "configs" / "robot").mkdir(parents=True)
        (content / "configs" / "robot" / "willy_ur5e.yml").write_bytes(b"a: 1\n")
        before = tool.snapshot(tree, curobo_content=content)
        self.assertEqual(tool.compare(tree, before).exit_code, 0, "compare is not simply always red")

        (content / "configs" / "robot" / "willy_x.yml").write_bytes(b"b: 2\n")
        report = tool.compare(tree, before)
        self.assertEqual(report.exit_code, 1)
        self.assertIn("willy_x.yml", report.render())

        (content / "configs" / "robot" / "willy_x.yml").unlink()
        (watched / "test_x.py").write_bytes(b"x = 2\n")
        report = tool.compare(tree, before)
        self.assertEqual(report.changed, ("tests/test_x.py",))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
