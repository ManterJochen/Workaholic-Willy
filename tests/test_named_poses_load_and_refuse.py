"""Poses taught in the console load as named joint poses in degrees, and anything a task could not drive to is refused
at load (owner decision 15; build plan 1.8, library L11).

A taught pose is where a task puts its part (the default place) or where it returns to; it lives in
``robot.named_poses`` under its NAME, the YAML key and the word the command reader hands on, with a free LABEL the
chat and the cards say ("Ablage links"). What is held here, on the schema and on a tree of two layers:

* a name is an ASCII identifier of at most 32 characters, no Python keyword, never ``home`` (Home stays
  ``robot.home_joint_positions``), no word YAML reads as true, false or null (``yes``, ``off``, ``null``) and none of
  the loader's own words (``__null__``); the console and the schema read one rule (``pose_name_refusal``);
* the joints keep the looks' rules: degrees, finite, at most a full turn either way, one per joint of the home where it
  is set, and every pose as long as the others;
* a label is free text, Unicode and spaces, at most 40 characters, with no newline and no " #" (a comment mark the
  line editor would cut a later rewrite at); a pose answers to its name and to its label, and no two poses answer to
  one word (without case, spaces folded): an unlabelled pose is said by its name;
* only a clear pose or one in the planner's band is a taught pose: an unscreened or refused verdict is refused at load;
  a pose written by hand carries none, and a task screens it before it moves;
* ``robot.default_place_pose`` names a pose of the tree, or nothing;
* poses of two layers merge KEY BY KEY, as every mapping of the tree does: a pose named in both takes the later layer's
  keys over the earlier one's, so a pose written by hand into a later layer with its joints alone keeps the screen a
  lower layer gave other joints. A screen read off the tree is therefore no proof about the joints beside it: a task
  screens every pose before it moves (build plan 1.3.3), and the pose door writes all five keys of a pose at once.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.config.loader import load_config, reload_config
from src.config.schema.robot import RobotConfig
from src.config.schema.robot.robot_schema import NamedPoseConfig, pose_label_refusal, pose_name_refusal

_SHIPPED = Path(__file__).resolve().parents[1] / "config"
DROP = [-60.0, -95.0, -120.0, -55.0, 90.0, 0.0]
PARK = [-90.0, -80.0, -110.0, -80.0, 90.0, 0.0]


def _pose(joints: "list[float] | None" = None, **fields: Any) -> dict[str, Any]:
    return {"joints_deg": list(DROP if joints is None else joints), "label": "Ablage links",
            "taught_at": "2026-10-01T09:12:00+02:00", "screen": "clear", "note": "", **fields}


def _robot(**fields: Any) -> RobotConfig:
    return RobotConfig.model_validate({"vendor": "ur", **fields})


class APoseLoadsTests(unittest.TestCase):
    def test_a_taught_pose_loads_in_degrees_with_its_label_and_its_screen(self) -> None:
        robot = _robot(named_poses={"drop_left": _pose(), "park": _pose(PARK, label="Parken", screen="band",
                                                                       note="in the planner's cushion band: ...")},
                       default_place_pose="drop_left")
        drop = robot.named_poses["drop_left"]
        self.assertIsInstance(drop, NamedPoseConfig)
        self.assertEqual(tuple(DROP), drop.joints_deg)
        self.assertEqual("Ablage links", drop.label)
        self.assertEqual("clear", drop.screen)
        self.assertEqual("2026-10-01T09:12:00+02:00", drop.taught_at)
        self.assertEqual("band", robot.named_poses["park"].screen)
        self.assertEqual("drop_left", robot.default_place_pose)

    def test_no_pose_and_no_default_is_the_shipped_default(self) -> None:
        robot = _robot()
        self.assertEqual({}, dict(robot.named_poses))
        self.assertIsNone(robot.default_place_pose)

    def test_a_pose_written_by_hand_needs_only_its_joints(self) -> None:
        robot = _robot(named_poses={"park": {"joints_deg": PARK}})
        park = robot.named_poses["park"]
        self.assertEqual(("", None, None, ""), (park.label, park.taught_at, park.screen, park.note))

    def test_a_label_may_be_any_words_within_its_bounds(self) -> None:
        for label in ("Ablage links", "Ablage für Teile", "#2 links", "Kiste 3 (blau)", "x" * 40):
            with self.subTest(label=label):
                self.assertEqual(label, _robot(named_poses={"drop": _pose(label=label)}).named_poses["drop"].label)


class ANameIsAPoseNameTests(unittest.TestCase):
    def test_names_that_name_no_pose_are_refused_saying_the_rule(self) -> None:
        for name, said in (("drop left", "ASCII"), ("1drop", "ASCII"), ("drop-left", "ASCII"), ("drüben", "ASCII"),
                           ("class", "keyword"), ("home", "home"), ("Home", "home"), ("x" * 33, "32")):
            with self.subTest(name=name):
                with self.assertRaises(ValidationError) as caught:
                    _robot(named_poses={name: _pose()})
                self.assertIn(said, str(caught.exception))
                self.assertTrue(pose_name_refusal(name))

    def test_the_names_a_cell_uses_are_names(self) -> None:
        for name in ("drop_left", "park", "Ablage_1", "x" * 32, "_spare", "yes_bin", "y", "n", "on_the_left", "_null"):
            with self.subTest(name=name):
                self.assertEqual("", pose_name_refusal(name))
                _robot(named_poses={name: _pose()})

    def test_words_yaml_reads_as_true_false_or_null_are_no_names(self) -> None:
        """The name is the YAML key the pose door writes bare (``yes:``), and PyYAML reads ``yes``, ``on``, ``off``,
        ``true``, ``false`` and ``null`` as a boolean or as nothing: such a pose could never be written, and the arm
        would have been freed for it (review of 2026-10-01)."""
        for name in ("yes", "no", "on", "off", "true", "false", "null", "Yes", "ON", "NULL", "fALSE", "nO"):
            with self.subTest(name=name):
                refused = pose_name_refusal(name)
                self.assertIn("YAML", refused)
                with self.assertRaises(ValidationError):
                    _robot(named_poses={name: _pose()})
        for name in ("True", "False", "None"):              # Python's own words, refused as keywords first
            with self.subTest(name=name):
                self.assertIn("keyword", pose_name_refusal(name))

    def test_the_loaders_own_words_are_no_names(self) -> None:
        """``__null__`` is the loader's overlay reset: a default place under that name reads back as none."""
        for name in ("__null__", "__init__", "__x__"):
            with self.subTest(name=name):
                self.assertIn("loader", pose_name_refusal(name))
                with self.assertRaises(ValidationError):
                    _robot(named_poses={name: _pose()})

    def test_the_console_and_the_schema_read_one_rule(self) -> None:
        from src.robot.execution.teach import name_refusal

        for name in ("drop_left", "drop left", "home", "HOME", "def", "x" * 33, "", "drop.left", "yes", "__null__"):
            with self.subTest(name=name):
                self.assertEqual(pose_name_refusal(name), name_refusal(name))


class TheJointsKeepTheLooksRulesTests(unittest.TestCase):
    def test_joints_a_task_could_not_drive_to_are_refused(self) -> None:
        for joints, said in (([], "names no joint"), ([math.nan, 0, 0, 0, 0, 0], "finite"),
                             ([361.0, 0, 0, 0, 0, 0], "full turn"), ([-400.0, 0, 0, 0, 0, 0], "full turn")):
            with self.subTest(joints=joints):
                with self.assertRaises(ValidationError) as caught:
                    _robot(named_poses={"drop": _pose(joints)})
                self.assertIn(said, str(caught.exception))
                self.assertIn("robot.named_poses", str(caught.exception))

    def test_a_pose_names_one_value_per_joint_of_the_home(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _robot(home_joint_positions_deg=[0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
                   named_poses={"drop": _pose(DROP[:5])})
        self.assertIn("home", str(caught.exception))
        _robot(home_joint_positions_deg=[0.0, -90.0, 90.0, -90.0, -90.0, 0.0], named_poses={"drop": _pose()})

    def test_every_pose_names_as_many_joints_as_the_others(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _robot(named_poses={"drop": _pose(), "park": _pose(PARK[:5], label="Parken")})
        self.assertIn("joints", str(caught.exception))


class ALabelIsTextALineCanHoldTests(unittest.TestCase):
    def test_labels_the_line_editor_could_not_rewrite_are_refused(self) -> None:
        for label, said in (("x" * 41, "40"), ("Ablage\nlinks", "newline"), ("Ablage #2", "#"),
                            ("Ablage\t#2", "#")):
            with self.subTest(label=label):
                with self.assertRaises(ValidationError) as caught:
                    _robot(named_poses={"drop": _pose(label=label)})
                self.assertIn(said, str(caught.exception))
                self.assertTrue(pose_label_refusal(label))

    def test_two_poses_never_share_a_label(self) -> None:
        """The command reader hands on a pose by its label: two poses called "Ablage links" would make the operator's
        words name either, and a part could go to the other one."""
        with self.assertRaises(ValidationError) as caught:
            _robot(named_poses={"drop": _pose(), "drop_2": _pose(PARK, label=" ablage  LINKS ")})
        self.assertIn("Ablage links", str(caught.exception))
        _robot(named_poses={"drop": _pose(label=""), "drop_2": _pose(PARK, label="")})   # no label: the name is said

    def test_no_two_poses_answer_to_one_word(self) -> None:
        """A pose answers to its label and to its name, and an unlabelled pose is said by its name (the command reader
        is handed label -> name pairs). An unlabelled 'park' beside a pose labelled 'Park' would make one spoken word
        name either, and a part could go to the other one (review of 2026-10-01)."""
        for poses, word in (({"park": _pose(PARK, label=""), "drop": _pose(label="Park")}, "park"),
                            ({"park": _pose(PARK, label="Parken"), "drop": _pose(label=" PARK ")}, "park"),
                            ({"Park": _pose(PARK, label=""), "park": _pose(label="")}, "park"),
                            ({"drop": _pose(label="Ablage"), "ablage": _pose(PARK, label="")}, "ablage")):
            with self.subTest(poses=sorted(poses)):
                with self.assertRaises(ValidationError) as caught:
                    _robot(named_poses=poses)
                self.assertIn(f"answer to {word!r}", str(caught.exception))
        # ⭐ THE CONTROL: a pose's own name as its label is its own word, no other pose's.
        _robot(named_poses={"park": _pose(PARK, label="park"), "drop": _pose(label="Ablage")})

    def test_an_empty_label_is_refused_where_the_console_writes_one(self) -> None:
        self.assertTrue(pose_label_refusal(""))
        self.assertEqual("", pose_label_refusal("Ablage links"))


class OnlyAScreenedPoseIsATaughtPoseTests(unittest.TestCase):
    def test_a_verdict_no_move_may_go_to_is_refused_at_load(self) -> None:
        for screen in ("unscreened", "guard_refused", "planner_refused", "error", "CLEAR!"):
            with self.subTest(screen=screen):
                with self.assertRaises(ValidationError):
                    _robot(named_poses={"drop": _pose(screen=screen)})

    def test_a_pose_has_no_keys_beyond_its_five(self) -> None:
        with self.assertRaises(ValidationError):
            _robot(named_poses={"drop": _pose(rig="wrist")})


class TheDefaultPlaceNamesAPoseTests(unittest.TestCase):
    def test_a_default_that_names_no_pose_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _robot(named_poses={"drop": _pose()}, default_place_pose="drop_left")
        for part in ("default_place_pose", "drop_left", "drop"):
            self.assertIn(part, str(caught.exception))
        with self.assertRaises(ValidationError):
            _robot(default_place_pose="drop")

    def test_none_is_no_default(self) -> None:
        self.assertIsNone(_robot(named_poses={"drop": _pose()}, default_place_pose=None).default_place_pose)


class PosesMergeAcrossLayersTests(unittest.TestCase):
    """A pose of the base and one of the cell's own layer are both in the tree; one named in both merges key by key,
    the later layer's keys winning."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def _append(self, path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def test_the_cell_layer_adds_its_poses_to_the_ones_below(self) -> None:
        self._append(self.tmp / "robot" / "robot.ur10.yaml",
                     "  named_poses:\n    park:\n      joints_deg: [-90.0, -80.0, -110.0, -80.0, 90.0, 0.0]\n"
                     "      label: \"Parken\"\n")
        (self.tmp / "robot" / "robot.cell.yaml").write_text(
            "robot:\n  named_poses:\n    drop_left:\n      joints_deg: [-60.0, -95.0, -120.0, -55.0, 90.0, 0.0]\n"
            "      label: \"Ablage links\"\n      screen: clear\n    park:\n      joints_deg: "
            "[-91.0, -80.0, -110.0, -80.0, 90.0, 0.0]\n      label: \"Parken\"\n      screen: band\n"
            "  default_place_pose: drop_left\n", encoding="utf-8")
        reload_config()
        robot = load_config(self.tmp, profile="ur10,cell").robot
        self.assertEqual({"drop_left", "park"}, set(robot.named_poses))
        self.assertEqual(-91.0, robot.named_poses["park"].joints_deg[0], "the later layer's pose wins")
        self.assertEqual("band", robot.named_poses["park"].screen)
        self.assertEqual("drop_left", robot.default_place_pose)
        reload_config()
        below = load_config(self.tmp, profile="ur10").robot
        self.assertEqual({"park"}, set(below.named_poses))
        self.assertIsNone(below.default_place_pose)

    def test_a_pose_named_in_two_layers_merges_key_by_key(self) -> None:
        """⚠ What a screen read off a merged tree is worth: a pose the cell's layer names with its joints alone keeps the
        label, the screen and the note a lower layer gave OTHER joints. So no caller trusts ``screen`` as proof about the
        joints beside it: a task screens every pose before it moves (build plan 1.3.3), and the pose door writes all
        five keys of a pose at once, so a pose it wrote never mixes layers."""
        self._append(self.tmp / "robot" / "robot.ur10.yaml",
                     "  named_poses:\n    park:\n      joints_deg: [-90.0, -80.0, -110.0, -80.0, 90.0, 0.0]\n"
                     "      label: \"Parken\"\n      screen: clear\n      note: \"screened on the old bracket\"\n")
        (self.tmp / "robot" / "robot.cell.yaml").write_text(
            "robot:\n  named_poses:\n    park:\n      joints_deg: [10.0, -80.0, -110.0, -80.0, 90.0, 0.0]\n",
            encoding="utf-8")
        reload_config()
        park = load_config(self.tmp, profile="ur10,cell").robot.named_poses["park"]
        self.assertEqual(10.0, park.joints_deg[0], "the later layer's joints")
        self.assertEqual(("Parken", "clear", "screened on the old bracket"), (park.label, park.screen, park.note),
                         "the lower layer's record, kept beside joints it never screened")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
