"""An older tree is brought to the keys this Willy reads, every comment kept (``python -m src.config migrate``).

A key that left the schema on purpose is refused at load with its sentence (``src/config/schema/_removed.py``), and the
owner's trees write a dozen of them (the owner, 2026-10-16: "ein Baum mit altem Schlüssel wird beim Laden abgelehnt,
dazu ein Umzugs-Skript"). ``src/config/migrate.py`` rewrites a tree's own files: a key that left goes with the comment
written directly above it, a key that moved goes under its new parent (into the other section's file where its section
changed), a word that changed is rewritten, and two blocks of one key become one. Nothing round-trips YAML, so every
other byte stays as it was.

Pinned here on copies of the shipped tree with old keys written in: each kind of change, the keys of how a cell moves
that went under ``robot.motion`` on 2026-10-17 (a home in radians written in degrees), the line ending a file was
written with, a dry run that writes nothing, a write whose result does not load put back byte for byte, the CLI, and
the owner's own tree where this machine holds it.
"""

from __future__ import annotations

import contextlib
import io
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from src.config.__main__ import main as config_cli
from src.config.loader import load_config, reload_config
from src.config.migrate import migrate_tree

_SHIPPED = Path(__file__).resolve().parents[2] / "config"
#: The owner's cell tree of 2026-10-14 (the Robotiq one), where this machine holds a copy.
_OWNERS = Path("D:/helper_cell/new_version_logs/robominds_config/config")


def _tree(case: unittest.TestCase) -> Path:
    """A copy of the shipped tree in a folder of its own, removed after the test."""
    folder = Path(tempfile.mkdtemp())
    case.addCleanup(shutil.rmtree, folder, ignore_errors=True)
    root = folder / "config"
    shutil.copytree(_SHIPPED, root)
    case.addCleanup(reload_config)
    return root


def _write_in(path: Path, after: str, lines: str) -> None:
    """``lines`` written into ``path`` right after the line ``after``, which the file holds once."""
    text = path.read_text(encoding="utf-8")
    assert text.count(after) == 1, (path, after)
    path.write_text(text.replace(after, after + lines), encoding="utf-8", newline="\n")


class EachKindOfChangeTests(unittest.TestCase):
    def test_a_key_that_left_goes_with_the_comment_written_above_it(self) -> None:
        root = _tree(self)
        robot = root / "robot" / "robot.yaml"
        _write_in(robot, "  grasping:\n", "    # How often a pick may try again.\n    max_attempts: 5\n")
        before = robot.read_bytes()

        planned = migrate_tree(root)

        self.assertEqual(["delete robot.grasping.max_attempts"], [f"{c.action} {c.key}" for c in planned.changes])
        self.assertIn("MAX_ATTEMPTS", planned.changes[0].said)
        self.assertEqual(before, robot.read_bytes(), "a dry run wrote a file")
        written = migrate_tree(root, write=True)
        self.assertTrue(written.written, written.refused)
        text = robot.read_text(encoding="utf-8")
        self.assertNotIn("max_attempts", text)
        self.assertNotIn("How often a pick may try again", text)
        self.assertEqual((_SHIPPED / "robot" / "robot.yaml").read_text(encoding="utf-8"), text)

    def test_a_key_that_moved_goes_into_its_new_sections_file_and_the_later_value_counts(self) -> None:
        root = _tree(self)
        _write_in(root / "robot" / "robot.yaml", "  grasping:\n",
                  "    recovery:\n      enabled: true\n      critical_parts: true\n")

        written = migrate_tree(root, write=True)

        self.assertTrue(written.written, written.refused)
        actions = [f"{c.action} {c.key}" for c in written.changes]
        self.assertIn("move robot.grasping.recovery.critical_parts", actions)
        self.assertIn("delete robot.grasping.recovery", actions)
        self.assertNotIn("recovery:", (root / "robot" / "robot.yaml").read_text(encoding="utf-8"))
        reload_config()
        self.assertIs(True, load_config(root).runtime.task.critical_parts, "the moved value did not win")

    def test_a_word_that_changed_is_rewritten_in_place(self) -> None:
        root = _tree(self)
        robot = root / "robot" / "robot.yaml"
        text = robot.read_text(encoding="utf-8")
        robot.write_text(text.replace('      source: "undeclared"', '      source: "polyscope"', 1), encoding="utf-8",
                         newline="\n")

        # What is read is the word: the shipped tree's identity frame is no controller's, which the load refuses.
        planned = migrate_tree(root)

        self.assertEqual(["reword robot.gripper.tool_frame.source"], [f"{c.action} {c.key}" for c in planned.changes])
        self.assertIn('      source: "controller"', planned.texts[robot.resolve()])

    def test_two_blocks_of_one_key_become_one(self) -> None:
        root = _tree(self)
        robot = root / "robot" / "robot.yaml"
        text = robot.read_text(encoding="utf-8")
        robot.write_text(text.replace("  grasping:\n", "  motion:\n    planner: ik\n\n  grasping:\n", 1), encoding="utf-8",
                         newline="\n")

        written = migrate_tree(root, write=True)

        self.assertTrue(written.written, written.refused)
        # The second block's keys join the first, and the planner the two then write is merged in the next pass.
        self.assertEqual(["merge robot.motion", "merge robot.motion.planner"],
                         [f"{c.action} {c.key}" for c in written.changes])
        self.assertEqual(1, robot.read_text(encoding="utf-8").count("\n  motion:\n"))
        reload_config()
        self.assertEqual("ik", load_config(root).robot.motion.planner, "the later block's value did not win")


class AFileWhoseSectionLeftTests(unittest.TestCase):
    def test_the_stereo_matcher_file_goes_and_a_refused_write_brings_it_back(self) -> None:
        root = _tree(self)
        matcher = root / "camera" / "stereomatcher.yaml"
        matcher.write_text("# The SGBM settings.\nstereomatcher:\n  numDisparities: 64\n", encoding="utf-8")

        written = migrate_tree(root, write=True)

        self.assertTrue(written.written, written.refused)
        self.assertEqual(["delete camera.stereomatcher"], [f"{c.action} {c.key}" for c in written.changes])
        self.assertFalse(matcher.exists())

        matcher.write_text("stereomatcher:\n  numDisparities: 64\n", encoding="utf-8")
        _write_in(root / "robot" / "robot.yaml", "  grasping:\n", "    no_such_key: 1\n")
        refused = migrate_tree(root, write=True)
        self.assertFalse(refused.written)
        self.assertEqual("stereomatcher:\n  numDisparities: 64\n", matcher.read_text(encoding="utf-8"))


class AModelsFileLeftWithNoKeyTests(unittest.TestCase):
    def test_the_hand_file_goes_when_both_its_blocks_left(self) -> None:
        root = _tree(self)
        hand = root / "models" / "hand.yaml"
        hand.write_text("# MediaPipe.\nhanddetect:\n  max_hands: 2\ngesturedetect:\n  max_hands: 2\n", encoding="utf-8")

        written = migrate_tree(root, write=True)

        self.assertTrue(written.written, written.refused)
        self.assertFalse(hand.exists())
        self.assertIn("delete models", [f"{c.action} {c.key}" for c in written.changes])


class TheFilesAreKeptTests(unittest.TestCase):
    def test_a_file_written_with_crlf_keeps_crlf(self) -> None:
        root = _tree(self)
        robot = root / "robot" / "robot.yaml"
        _write_in(robot, "  grasping:\n", "    max_attempts: 5\n")
        robot.write_bytes(robot.read_bytes().replace(b"\n", b"\r\n"))

        self.assertTrue(migrate_tree(root, write=True).written)

        raw = robot.read_bytes()
        self.assertNotIn(b"max_attempts", raw)
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"), "a line lost its CR")

    def test_a_write_whose_tree_still_does_not_load_puts_every_file_back(self) -> None:
        root = _tree(self)
        robot = root / "robot" / "robot.yaml"
        _write_in(robot, "  grasping:\n", "    max_attempts: 5\n    no_such_key: 1\n")
        before = robot.read_bytes()

        migration = migrate_tree(root, write=True)

        self.assertFalse(migration.written)
        self.assertIn("does not load", migration.refused)
        self.assertEqual(before, robot.read_bytes(), "a refused write left a file changed")

    def test_a_tree_that_writes_only_living_keys_is_left_alone(self) -> None:
        root = _tree(self)
        migration = migrate_tree(root, write=True)
        self.assertEqual([], migration.changes)
        self.assertIn("nothing to change", migration.render())


class TheCommandTests(unittest.TestCase):
    def _cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = config_cli(["migrate", *argv])
        return code, out.getvalue()

    def test_without_write_it_says_what_it_would_change_and_changes_nothing(self) -> None:
        root = _tree(self)
        robot = root / "robot" / "robot.yaml"
        _write_in(robot, "  grasping:\n", "    max_attempts: 5\n")
        before = robot.read_bytes()

        code, said = self._cli(str(root))

        self.assertEqual(0, code)
        self.assertIn("robot/robot.yaml:", said)
        self.assertIn("delete robot.grasping.max_attempts", said)
        self.assertIn("nothing written (add --write)", said)
        self.assertEqual(before, robot.read_bytes())

    def test_with_write_it_writes_and_says_the_tree_loads(self) -> None:
        root = _tree(self)
        _write_in(root / "robot" / "robot.yaml", "  grasping:\n", "    max_attempts: 5\n")

        code, said = self._cli(str(root), "--write")

        self.assertEqual(0, code)
        self.assertIn("written, and the tree loads", said)

    def test_a_write_that_does_not_load_exits_1(self) -> None:
        root = _tree(self)
        _write_in(root / "robot" / "robot.yaml", "  grasping:\n", "    max_attempts: 5\n    no_such_key: 1\n")
        self.assertEqual(1, self._cli(str(root), "--write")[0])


#: A home of round degrees, as the pendant shows it, and as an older layer wrote it, in radians.
_HOME_DEG = (-90.0, -100.0, -110.0, -60.0, 90.0, 0.0)
#: A cell's own layer as an older Willy wrote it: every key of how a cell moves beside robot.motion, the home in
#: radians on one line.
_OLD_MOTION_LAYER = f"""\
# An older cell's own layer.
robot:
  motion_limits:
    max_velocity: 0.8          # rad/s
    max_acceleration: 0.4
  workspace_limits:
    x_min: -400.0
    x_max: 400.0
    y_min: -800.0
    y_max: -200.0
    z_min: 20.0
    z_max: 700.0
  # Where the arm goes home, facing the bench.
  home_joint_positions: [{", ".join(repr(math.radians(v)) for v in _HOME_DEG)}]
  named_poses:
    drop_left:
      joints_deg: [-60.0, -95.0, -120.0, -55.0, 90.0, 0.0]
      label: "Ablage links"
      screen: clear
  default_place_pose: drop_left
  place:
    rim_air_mm: 10.0
    carry: over_the_rim
"""


class TheKeysOfHowACellMovesGoUnderMotionTests(unittest.TestCase):
    def test_an_older_layer_is_moved_under_motion_its_home_in_degrees_and_loads(self) -> None:
        root = _tree(self)
        layer = root / "robot" / "robot.oldcell.yaml"
        layer.write_text(_OLD_MOTION_LAYER, encoding="utf-8")

        written = migrate_tree(root, write=True)

        self.assertTrue(written.written, written.refused)
        actions = {f"{c.action} {c.key}" for c in written.changes}
        for said in ("move robot.motion_limits.max_velocity", "move robot.motion_limits.max_acceleration",
                     "delete robot.motion_limits", "move robot.workspace_limits", "convert robot.home_joint_positions",
                     "move robot.named_poses", "move robot.default_place_pose", "move robot.place",
                     "delete robot.motion.place.rim_air_mm"):
            self.assertIn(said, actions)
        text = layer.read_text(encoding="utf-8")
        for old in ("motion_limits", "workspace_limits", "home_joint_positions", "rim_air_mm"):
            self.assertNotIn(old, text)
        self.assertIn("# Where the arm goes home, facing the bench.", text)
        reload_config()
        motion = load_config(root, profile="oldcell").robot.motion
        self.assertEqual((0.8, 0.4), (motion.max_velocity, motion.max_acceleration))
        self.assertEqual((-400.0, 400.0, -800.0, -200.0, 20.0, 700.0),
                         (motion.workspace.x_min, motion.workspace.x_max, motion.workspace.y_min,
                          motion.workspace.y_max, motion.workspace.z_min, motion.workspace.z_max))
        assert motion.home_deg is not None
        for stated, degrees in zip(motion.home_deg, _HOME_DEG, strict=True):
            self.assertAlmostEqual(degrees, stated, delta=1e-4)
        self.assertEqual((-60.0, -95.0, -120.0, -55.0, 90.0, 0.0), motion.named_poses["drop_left"].joints_deg)
        self.assertEqual(("Ablage links", "clear"),
                         (motion.named_poses["drop_left"].label, motion.named_poses["drop_left"].screen))
        self.assertEqual("drop_left", motion.default_place_pose)
        # The place's rim air is the owner's cell's since 2026-10-17: the key left, the carry stays the layer's.
        self.assertEqual("over_the_rim", motion.place.carry)


class TheOwnersTreeTests(unittest.TestCase):
    def test_the_owners_tree_migrates_and_loads(self) -> None:
        if not _OWNERS.is_dir():
            raise unittest.SkipTest("the owner's tree is not on this machine")
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.addCleanup(reload_config)
        root = folder / "config"
        shutil.copytree(_OWNERS, root)

        migration = migrate_tree(root, write=True)

        self.assertTrue(migration.written, migration.refused)
        self.assertGreater(len(migration.changes), 0)


if __name__ == "__main__":
    unittest.main()
