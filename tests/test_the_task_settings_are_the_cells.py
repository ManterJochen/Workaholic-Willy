"""A task's settings are the cell's, not each browser's, and the console writes them through a door of their own.

The owner, 2026-10-16 (round 7): "the 5 task prefs (Enter starts, Einmal/Bis leer, Blicke, Ablegen ohne Angabe, Kein
Teil genannt) move from the browser into runtime.yaml (cell-wide)"; 2026-10-17: the Settings page writes them into the
cell's own layer through the same checked door as the taught poses ("Konsole schreibt sie"). So:

* ``runtime.task`` holds the five beside ``critical_parts``, at the defaults each browser kept until then: Enter starts,
  once, the cell's default place, the card asking for a part, the looks when needed. Each takes its closed set alone,
  and a place only ``default``, ``pose:<name>`` or ``camera:<phrase>``, the phrase at most what a card's field takes,
  on one line and with no ``" #"``. The shipped ``app/runtime.yaml`` and the all-keys reference write the defaults;
* ``set_task_settings`` writes a change into the runtime overlay of the chain's LAST layer (``app/runtime.cell.yaml``
  under the owner's chain) and nowhere else, as one transaction, and the tree loads it back; a setting chosen again is
  rewritten on its own line, every comment kept. A value the loader refuses leaves every byte as it was, and a name that
  is no setting is refused before anything is written;
* the rule of a taught pose holds (Q10): a chain with no layer over the shipped tree is refused, and so is a last layer
  git does not keep out. A cell's own tree outside the repository takes them in its own ``runtime.yaml``;
* the generic writer (``set_keys``, behind ``PATCH /v1/config``) refuses every task setting with a sentence of its own.

The console's answer over HTTP, and its literals against these, live with the console's tests
(``tests/test_api_the_cells_task_settings_are_read_and_written.py``): a library test imports no web framework.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.edit import (
    TASK_KEYS,
    WRITABLE,
    WriteRefused,
    set_keys,
    set_task_settings,
    task_layer_refusal,
    task_target_file,
    writable,
)
from src.config.loader import load_config, reload_config
from src.config.schema.runtime import RuntimeConfig, TaskConfig, task_place, task_place_text
from src.config.tree import ConfigTree
from tests.test_a_pose_is_written_only_through_its_own_door import _git, _the_shipped_tree, _tree_hash

_ROOT = Path(__file__).resolve().parents[1]
_SHIPPED = _ROOT / "config"
CHAIN = "ur10,hande,cell"
LAYERS = ("ur10", "hande", "cell")
#: Each browser's defaults until 2026-10-17, and the cell's switch beside them.
DEFAULTS = {"start": "enter", "scope": "once", "place": "default", "anything": False, "looks": "when_needed",
            "critical_parts": False}
#: Every setting chosen otherwise than shipped.
CHOSEN = {"start": "card", "scope": "until_empty", "place": "camera:yellow bin", "anything": True, "looks": "every",
          "critical_parts": True}


class TheSettingsAreTheCellsTests(unittest.TestCase):
    def test_each_browsers_defaults_are_the_cells(self) -> None:
        self.assertEqual(DEFAULTS, TaskConfig().model_dump())
        self.assertEqual(TaskConfig(), RuntimeConfig().task)

    def test_the_shipped_file_and_the_reference_write_the_defaults(self) -> None:
        for path in (_SHIPPED / "app" / "runtime.yaml", _SHIPPED / "all_keys" / "app" / "runtime.yaml"):
            with self.subTest(path=path.relative_to(_ROOT).as_posix()):
                written = yaml.safe_load(path.read_text(encoding="utf-8"))["runtime"]["task"]
                self.assertEqual(DEFAULTS, written)
        reload_config()
        self.assertEqual(TaskConfig(), load_config(_SHIPPED, profile=None).runtime.task)

    def test_each_setting_takes_its_closed_set_and_nothing_else(self) -> None:
        for key, value in CHOSEN.items():
            with self.subTest(key=key):
                self.assertEqual(value, getattr(TaskConfig.model_validate({key: value}), key))
        for key, value in (("start", "now"), ("scope", "twice"), ("looks", "all of them"), ("looks", ""),
                           ("anything", "maybe"), ("critical_parts", [True]), ("start", None)):
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                TaskConfig.model_validate({key: value})
        with self.assertRaises(ValidationError):
            TaskConfig.model_validate({"taskStart": "card"})


class APlaceIsOneOfThreeTests(unittest.TestCase):
    def test_the_default_place_a_taught_pose_or_what_the_camera_finds(self) -> None:
        for written, read in (("default", ("default", "")), ("pose:drop_left", ("pose", "drop_left")),
                              (" camera:  yellow bin ", ("camera", "yellow bin")), ("camera:bin: left", ("camera", "bin: left"))):
            with self.subTest(place=written):
                place = TaskConfig.model_validate({"place": written}).place
                self.assertEqual(read, task_place(place))
                self.assertEqual(place, task_place_text(*read))

    def test_a_place_of_no_form_is_refused_saying_the_forms(self) -> None:
        for written in ("", "somewhere", "Pose:park", "default:", "drop_left", "pose", "camera"):
            with self.subTest(place=written), self.assertRaises(ValidationError) as caught:
                TaskConfig.model_validate({"place": written})
            for form in ("'default'", "'pose:<name>'", "'camera:<phrase>'"):
                self.assertIn(form, str(caught.exception))

    def test_a_pose_under_no_pose_name_is_refused(self) -> None:
        for written in ("pose:", "pose:home", "pose:drop left", "pose:yes", "pose:__null__"):
            with self.subTest(place=written), self.assertRaises(ValidationError) as caught:
                TaskConfig.model_validate({"place": written})
            self.assertIn("no pose name", str(caught.exception))

    def test_a_camera_phrase_is_what_a_cards_field_takes_on_one_line(self) -> None:
        self.assertEqual("camera:" + "y" * 200, TaskConfig.model_validate({"place": "camera:" + "y" * 200}).place)
        for written in ("camera:", "camera:   ", "camera:" + "y" * 201, "camera:bin #2", "camera:yellow\nbin"):
            with self.subTest(place=written), self.assertRaises(ValidationError) as caught:
                TaskConfig.model_validate({"place": written})
            self.assertIn("a card would not take", str(caught.exception))


class _Case(unittest.TestCase):
    """A scratch copy of the shipped tree with the owner's own cell layer, ``robot.cell.yaml``, holding a comment."""

    CELL_TEXT = "# The cell's own layer: what this cell measured, taught and chose. Not in git.\n"

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        (self.tmp / "robot" / "robot.cell.yaml").write_text(self.CELL_TEXT, encoding="utf-8")
        self.layer = self.tmp / "app" / "runtime.cell.yaml"
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def write(self, values: dict[str, Any], *, layers: tuple[str, ...] = LAYERS, profile: str | None = CHAIN) -> Any:
        return set_task_settings(values, root=self.tmp, layers=layers, profile=profile)

    def task(self, profile: str = CHAIN) -> TaskConfig:
        reload_config()
        return load_config(self.tmp, profile=profile).runtime.task


class AChangeLandsInTheCellsOwnLayerTests(_Case):
    def test_it_lands_in_the_runtime_overlay_of_the_last_layer_and_loads_back(self) -> None:
        before = _tree_hash(self.tmp)

        result = self.write(CHOSEN)

        self.assertTrue(result.applied, result.message)
        self.assertEqual((self.layer,), result.files)
        after = _tree_hash(self.tmp)
        self.assertEqual([str(Path("app") / "runtime.cell.yaml")], sorted(n for n in after if after[n] != before.get(n)))
        self.assertEqual(TaskConfig(**CHOSEN), self.task())
        self.assertEqual(TaskConfig(), self.task(profile="ur10,hande"), "a lower layer took the cell's choice")
        self.assertEqual({f"runtime.task.{key}": value for key, value in CHOSEN.items()}, result.values)

    def test_the_file_is_named_before_anything_is_written(self) -> None:
        self.assertEqual(self.layer, task_target_file(self.tmp, LAYERS))
        self.assertEqual(self.layer, ConfigTree(root=self.tmp, profile=CHAIN, layers=LAYERS).task_file())
        self.assertFalse(self.layer.exists(), "naming the file wrote it")
        self.assertIsNone(task_target_file(_SHIPPED, ()))
        with _the_shipped_tree():
            self.assertIsNone(task_target_file(self.tmp, ()))

    def test_a_setting_chosen_again_is_rewritten_on_its_own_line_and_every_comment_stays(self) -> None:
        self.layer.write_text("# chosen at the cell\nruntime:\n  task:\n    start: card   # the owner's own\n"
                              "    looks: every\n", encoding="utf-8")

        result = self.write({"start": "enter", "scope": "until_empty"})

        self.assertTrue(result.applied, result.message)
        lines = self.layer.read_text(encoding="utf-8").splitlines()
        self.assertEqual(["# chosen at the cell", "runtime:", "  task:", '    start: "enter"  # the owner\'s own',
                          "    looks: every", '    scope: "until_empty"'], lines)
        self.assertEqual(("enter", "until_empty", "every"), (self.task().start, self.task().scope, self.task().looks))

    def test_the_tree_writes_through_the_same_door(self) -> None:
        tree = ConfigTree(root=self.tmp, profile=CHAIN, layers=LAYERS)
        self.assertEqual("", tree.task_layer_refusal())
        result = tree.write_task_settings({"looks": "first", "place": "pose:drop_left"})
        self.assertTrue(result.applied, result.message)
        self.assertEqual(("first", "pose:drop_left"), (self.task().looks, self.task().place))

    def test_a_cells_own_tree_outside_the_repository_takes_them_with_no_layer(self) -> None:
        """The owner, 2026-10-05: a customer's tree (``--data``) is wholly the cell's own, as for a taught pose."""
        base = self.tmp / "app" / "runtime.yaml"
        result = self.write({"scope": "until_empty"}, layers=(), profile=None)
        self.assertTrue(result.applied, result.message)
        self.assertEqual((base,), result.files)
        self.assertEqual("until_empty", self.task(profile="").scope)


class ABadChangeIsRefusedAndNothingIsWrittenTests(_Case):
    def test_a_value_the_loader_refuses_leaves_every_byte_as_it_was(self) -> None:
        self.assertTrue(self.write({"start": "card"}).applied)
        before = _tree_hash(self.tmp)
        for values, key in (({"start": "now"}, "start"), ({"place": "camera:bin #2"}, "place"),
                            ({"looks": "all of them"}, "looks"), ({"scope": "until_empty", "anything": "maybe"},
                                                                  "anything")):
            with self.subTest(values=values):
                result = self.write(values)
                self.assertFalse(result.applied)
                self.assertIs(WriteRefused.INVALID_VALUE, result.refused)
                for part in (f"runtime.task.{key}", "runtime.cell.yaml"):
                    self.assertIn(part, result.message)
                self.assertEqual(before, _tree_hash(self.tmp))
        self.assertEqual(("card", "once"), (self.task().start, self.task().scope))

    def test_a_name_that_is_no_setting_is_refused_before_anything_is_written(self) -> None:
        before = _tree_hash(self.tmp)
        result = self.write({"start": "card", "taskStart": "card"})
        self.assertIs(WriteRefused.UNKNOWN_KEY, result.refused)
        self.assertEqual("runtime.task.taskStart", result.refused_key)
        self.assertEqual(before, _tree_hash(self.tmp))
        self.assertFalse(self.layer.exists())


class AChainWithNoLayerOfTheCellsOwnTests(_Case):
    def test_no_layer_over_the_shipped_tree_is_refused_with_nothing_written(self) -> None:
        """⛔ Over the shipped tree a chain with no layer would write the shared ``runtime.yaml`` every cell reads."""
        before = _tree_hash(self.tmp)
        with _the_shipped_tree():
            result = self.write({"start": "card"}, layers=(), profile=None)
            said = task_layer_refusal(self.tmp, ())
        self.assertIs(WriteRefused.NO_LAYER, result.refused)
        for part in ("runtime.yaml", "robot.cell.yaml", "Nothing was written."):
            self.assertIn(part, result.message)
        self.assertIn(said, result.message)
        self.assertEqual(before, _tree_hash(self.tmp))


@unittest.skipUnless(shutil.which("git"), "git is not on PATH")
class GitKeepsTheCellsLayerOutTests(_Case):
    """Q10 = A, for the task settings as for a taught pose: the scratch tree lies in a git work tree of its own."""

    def setUp(self) -> None:
        super().setUp()
        _git(self.tmp.parent, "init", "-q")

    def test_a_layer_git_does_not_ignore_is_refused_naming_the_line_to_add(self) -> None:
        before = _tree_hash(self.tmp)
        result = self.write({"start": "card"})
        self.assertIs(WriteRefused.NO_LAYER, result.refused)
        for part in ("data/app/runtime.cell.yaml", ".gitignore", "not git-ignored", "task settings"):
            self.assertIn(part, result.message)
        self.assertEqual(before, _tree_hash(self.tmp))

    def test_a_layer_git_ignores_takes_them(self) -> None:
        (self.tmp.parent / ".gitignore").write_text("data/**/*.cell.yaml\n", encoding="utf-8")
        self.assertEqual("", task_layer_refusal(self.tmp, LAYERS))
        self.assertTrue(self.write({"start": "card"}).applied)
        self.assertEqual("card", self.task().start)

    def test_a_tree_with_no_app_folder_is_asked_from_the_folder_above_and_takes_its_first_file(self) -> None:
        """A cell's own tree may hold no ``app/`` folder yet: git is asked from the tree, and the write makes both."""
        shutil.rmtree(self.tmp / "app")
        (self.tmp.parent / ".gitignore").write_text("data/**/*.cell.yaml\n", encoding="utf-8")
        self.assertEqual("", task_layer_refusal(self.tmp, LAYERS))
        result = self.write({"anything": True})
        self.assertTrue(result.applied, result.message)
        self.assertTrue(self.task().anything)


class TheGenericWriterRefusesTheTaskSettingsTests(_Case):
    def test_a_task_setting_is_not_writable_and_says_where_it_is_chosen(self) -> None:
        before = _tree_hash(self.tmp)
        for key in (*TASK_KEYS, "runtime.task"):
            with self.subTest(key=key):
                result = set_keys({key: "card"}, root=self.tmp, layers=LAYERS, profile=CHAIN)
                self.assertIs(WriteRefused.NOT_WRITABLE, result.refused)
                self.assertEqual(key, result.refused_key)
                for part in ("task setting", "Settings page", "PATCH /v1/task/settings"):
                    self.assertIn(part, result.message)
                via_tree = ConfigTree(root=self.tmp, profile=CHAIN, layers=LAYERS).write({key: "card"}, connected=False)
                self.assertIs(WriteRefused.NOT_WRITABLE, via_tree.refused)
        self.assertEqual(before, _tree_hash(self.tmp))

    def test_the_writable_set_offers_none_and_a_key_that_is_none_stays_unknown(self) -> None:
        self.assertEqual(tuple(f"runtime.task.{key}" for key in DEFAULTS), TASK_KEYS)
        self.assertFalse([entry.path for entry in WRITABLE if entry.path.startswith("runtime.task")])
        self.assertIsNone(writable("runtime.task.start"))
        result = set_keys({"runtime.task.taskStart": "card"}, root=self.tmp, layers=LAYERS, profile=CHAIN)
        self.assertIs(WriteRefused.UNKNOWN_KEY, result.refused)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
