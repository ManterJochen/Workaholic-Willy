"""A change to a loaded tree is a load, and a noun takes the tree once (surface step, owner 2026-09-18, Q2 and Q3).

Five examples changed config with `model_copy`, which runs no validator and skips the named hand fill, so naming the
Hand-E kept the 2F-85's widths and finger geometry and raised nothing. `LoadedTree.with_values` sets dotted keys on
top of the files and runs the load itself, so a value given in memory meets every check a value written in a layer
meets, and is refused in the same words.

Every test here is red on the code before this step because `LoadedTree` has no `with_values`, no `app_config`, no
`robot`, no `root`, no `profile` and no `values`: each one reaches one of them. The one exception says so itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from src.config.edit import read_key
from src.config.hand_numbers import HAND_KEYS, HandNumbers
from src.config.loader import ConfigError, load_config, load_robot_config, reload_config
from src.config.tree import ConfigTree, default_data_dir

_IN_MEMORY = "LoadedTree.with_values"


def _numbers(model: str) -> dict[str, Any]:
    """The thirteen robot keys the registry hand ``model`` supplies, dotted below ``robot``."""
    return {key: value for key, _field, value in HandNumbers.from_registry(model=model).values}


def _dotted(block: Any, prefix: str = "") -> dict[str, Any]:
    """A YAML mapping as dotted keys to leaves; a list is a leaf, as a layer sets it whole."""
    if not isinstance(block, dict):
        return {prefix: block}
    out: dict[str, Any] = {}
    for key, value in block.items():
        out.update(_dotted(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def _digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


class _Clean(unittest.TestCase):
    """`WILLY_PROFILE` and the adaptation overlay cleared, so each tree is the one its test names."""

    def setUp(self) -> None:
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("WILLY_PROFILE", None)
        os.environ.pop("WILLY_ADAPTATION_OVERLAY", None)

    def copy_of_the_tree(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="willy_with_values_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        self.addCleanup(reload_config)
        root = tmp / "data"
        shutil.copytree(default_data_dir(), root)
        return root

    def ursim(self) -> Any:
        loaded = ConfigTree.from_directory(profile="ursim").load()
        self.assertTrue(loaded.ok, loaded.error)
        self.assertEqual(loaded.config.robot.gripper.model, "robotiq_2f85", "the control: ursim names the 2F-85")
        return loaded


class AHandNamedInMemoryTests(_Clean):
    def test_the_hand_e_takes_its_thirteen_numbers_where_model_copy_kept_the_2f85s(self) -> None:
        loaded = self.ursim()
        hande, f2f85 = _numbers("robotiq_hande"), _numbers("robotiq_2f85")
        self.assertEqual(len(hande), len(HAND_KEYS))
        self.assertEqual(len(HAND_KEYS), 13)
        self.assertNotEqual(hande["gripper.max_width_mm"], f2f85["gripper.max_width_mm"],
                            "the two hands must differ, or this proves nothing")

        # The gap, as it stands before this step: model_copy renames the hand and keeps every number.
        robot = loaded.config.robot
        copied = robot.model_copy(update={"gripper": robot.gripper.model_copy(update={"model": "robotiq_hande"})})
        self.assertEqual(copied.gripper.model, "robotiq_hande")
        self.assertEqual({key: read_key(copied, key) for key in f2f85}, f2f85)

        changed = loaded.with_values({"robot.gripper.model": "robotiq_hande"})
        self.assertTrue(changed.ok, changed.error)
        self.assertEqual(changed.config.robot.gripper.model, "robotiq_hande")
        for key, value in hande.items():
            with self.subTest(key=key):
                self.assertEqual(read_key(changed.config.robot, key), value)

    def test_a_value_given_in_memory_loads_as_the_same_value_written_in_a_layer(self) -> None:
        """The `hande` layer, given key by key in memory over the base tree, is the `hande` tree."""
        layer = yaml.safe_load((default_data_dir() / "robot" / "robot.hande.yaml").read_text(encoding="utf-8"))
        self.assertEqual(sorted(p.name for p in default_data_dir().rglob("*.hande.yaml")), ["robot.hande.yaml"],
                         "the control: every file of the hande layer is given in memory below")
        from_the_layer = ConfigTree.from_directory(profile="hande").load()
        in_memory = ConfigTree.from_directory(profile=None).load().with_values(_dotted(layer))
        self.assertTrue(from_the_layer.ok, from_the_layer.error)
        self.assertTrue(in_memory.ok, in_memory.error)
        self.assertEqual(in_memory.config, from_the_layer.config)
        self.assertEqual(in_memory.hands, from_the_layer.hands)


class RefusedAsALoadRefusesTests(_Clean):
    def _trial(self, root: Path, chain: str, body: str) -> Any:
        (root / "robot" / "robot.trial.yaml").write_text(body, encoding="utf-8")
        reload_config()
        return ConfigTree.from_directory(root=root, profile=chain).load()

    def test_a_schema_refusal_is_the_loads_sentence_with_memory_named_as_the_place(self) -> None:
        """`mass_kg` is written by the ursim layer. Given -1.0 in memory, the refusal is the one a layer writing
        -1.0 gets, line for line, except the line that says where: it names memory, not the ursim line."""
        root = self.copy_of_the_tree()
        before = self._trial(root, "ursim,trial", "robot: {}\n")
        self.assertTrue(before.ok, before.error)
        self.assertIn("robot.ursim.yaml", before.explain("robot.safety.payload.mass_kg").set_in)

        in_memory = before.with_values({"robot.safety.payload.mass_kg": -1.0})
        in_a_layer = self._trial(root, "ursim,trial", "robot:\n  safety:\n    payload:\n      mass_kg: -1.0\n")
        self.assertFalse(in_memory.ok)
        self.assertEqual(in_memory.exit_code, 1)
        self.assertFalse(in_a_layer.ok)

        said, written = in_memory.error.splitlines(), in_a_layer.error.splitlines()
        self.assertEqual(len(said), len(written))
        differing = [(a, b) for a, b in zip(said, written) if a != b]
        self.assertEqual(len(differing), 1, differing)
        self.assertEqual(differing[0][0].strip(), f"at {_IN_MEMORY}")
        self.assertIn("robot.trial.yaml", differing[0][1])
        self.assertNotIn("robot.ursim.yaml", in_memory.error)

    def test_a_named_hand_contradiction_is_the_loads_sentence_with_memory_named(self) -> None:
        root = self.copy_of_the_tree()
        before = self._trial(root, "ursim,trial", "robot: {}\n")
        in_memory = before.with_values({"robot.gripper.max_width_mm": 30.0})
        in_a_layer = self._trial(root, "ursim,trial", "robot:\n  gripper:\n    max_width_mm: 30.0\n")
        self.assertFalse(in_memory.ok)
        self.assertFalse(in_a_layer.ok)
        place = re.search(r"\(written in (.+?)\)", in_a_layer.error)
        self.assertIsNotNone(place, in_a_layer.error)
        assert place is not None
        self.assertIn("robot.trial.yaml", place.group(1))
        self.assertEqual(in_memory.error, in_a_layer.error.replace(place.group(1), _IN_MEMORY))

    def test_an_item_the_list_does_not_hold_is_a_verdict_and_one_it_holds_is_set(self) -> None:
        loaded = self.ursim()
        rigs = loaded.config.camera.cameras.rigs
        self.assertFalse(rigs[1].enabled, "the control: the second rig ships disabled")

        refused = loaded.with_values({f"camera.cameras.rigs[{len(rigs)}].enabled": True})
        self.assertFalse(refused.ok)
        self.assertIn(f"names item {len(rigs)} of camera.cameras.rigs, which holds {len(rigs)} item(s)",
                      refused.error)

        enabled = loaded.with_values({"camera.cameras.rigs[1].enabled": True})
        self.assertTrue(enabled.ok, enabled.error)
        self.assertTrue(enabled.config.camera.cameras.rigs[1].enabled)

    def test_a_key_that_is_no_dotted_path_or_a_model_object_is_a_programmer_error(self) -> None:
        loaded = self.ursim()
        for key in ("", "robot..gripper", "robot.gripper[", "camera.cameras.rigs[0][1]"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                loaded.with_values({key: 1})
        not_a_string: Any = ("robot", "gripper")
        with self.assertRaises(TypeError):
            loaded.with_values({not_a_string: 1})
        not_a_mapping: Any = [("robot.gripper.model", "robotiq_hande")]
        with self.assertRaises(TypeError):
            loaded.with_values(not_a_mapping)
        with self.assertRaises(TypeError):
            loaded.with_values({"robot.gripper": loaded.config.robot.gripper})


class NothingElseChangesTests(_Clean):
    def test_the_tree_it_came_from_is_unchanged_and_no_file_is_written(self) -> None:
        root = self.copy_of_the_tree()
        before = _digest(root)
        loaded = ConfigTree.from_directory(root=root, profile="ursim").load()
        self.assertTrue(loaded.ok, loaded.error)
        config = loaded.config

        changed = loaded.with_values({"robot.gripper.model": "robotiq_hande"})
        self.assertTrue(changed.ok, changed.error)
        self.assertIs(loaded.config, config)
        self.assertEqual(loaded.config.robot.gripper.model, "robotiq_2f85")
        self.assertEqual(loaded.config.robot.gripper.max_width_mm, _numbers("robotiq_2f85")["gripper.max_width_mm"])
        self.assertEqual(dict(loaded.values), {})
        self.assertIs(load_config(root, profile="ursim"), config, "the cached load is the one it was")
        self.assertIsNot(changed.config, config)
        self.assertEqual(_digest(root), before)

    def test_values_stack_and_the_last_value_given_for_a_place_wins(self) -> None:
        first = self.ursim().with_values({"robot.gripper.model": "robotiq_hande", "robot.safety.payload.mass_kg": 1.25})
        second = first.with_values({"robot.gripper": {"model": "schunk_egu50"}})
        self.assertTrue(second.ok, second.error)
        self.assertEqual(dict(second.values),
                         {"robot.safety.payload.mass_kg": 1.25, "robot.gripper.model": "schunk_egu50"})
        self.assertEqual(second.config.robot.safety.payload.mass_kg, 1.25)
        self.assertEqual(second.config.robot.gripper.max_width_mm,
                         _numbers("schunk_egu50")["gripper.max_width_mm"])
        self.assertEqual(dict(first.values),
                         {"robot.gripper.model": "robotiq_hande", "robot.safety.payload.mass_kg": 1.25})


class TheQuestionsNameMemoryTests(_Clean):
    def test_explain_says_a_value_given_in_memory_is_set_there(self) -> None:
        loaded = self.ursim()
        self.assertIn("robot.ursim.yaml", loaded.explain("robot.gripper.model").set_in, "the control")

        changed = loaded.with_values({"robot.gripper.model": "robotiq_hande"})
        said = changed.explain("robot.gripper.model")
        self.assertEqual(said.value, "robotiq_hande")
        self.assertEqual(said.set_in, _IN_MEMORY)
        self.assertTrue(any("robot.ursim.yaml" in layer.location and not layer.winner for layer in said.layers))
        self.assertNotIn("why (comment above that line)", said.render())

        derived = changed.explain("robot.gripper.max_width_mm")
        self.assertEqual(derived.value, _numbers("robotiq_hande")["gripper.max_width_mm"])
        self.assertIn("grippers/robotiq_hande.yaml", derived.set_in)

    def test_decisions_say_a_value_given_in_memory_is_set_there(self) -> None:
        changed = self.ursim().with_values({"robot.gripper.model": "robotiq_hande"})
        said = changed.decisions(section="robot.gripper")
        row = said[said.index("robot.gripper.model"):]
        self.assertIn(_IN_MEMORY, row.splitlines()[1])
        derived = said[said.index("robot.gripper.max_width_mm"):]
        self.assertIn("grippers/robotiq_hande.yaml", derived.splitlines()[1])

    def test_both_halves_name_the_values_given_in_memory(self) -> None:
        loaded = self.ursim()
        self.assertNotIn("in memory", loaded.render(), "the control: a load renders as it did")
        changed = loaded.with_values({"robot.gripper.model": "robotiq_hande"})
        text = changed.render()
        self.assertIn("in memory: robot.gripper.model", text)
        self.assertTrue(text.isascii())
        self.assertFalse(text.endswith("\n"))
        payload = changed.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["values"], {"robot.gripper.model": "robotiq_hande"})
        self.assertEqual(loaded.to_dict()["values"], {})

        refused = loaded.with_values({"robot.safety.payload.mass_kg": -1.0})
        self.assertTrue(refused.render().startswith(
            "config error, with robot.safety.payload.mass_kg given in memory:\n"))


class TheDoorNamesTests(_Clean):
    """What a `from_tree` door reads, under names that stay (owner Q2)."""

    def test_a_door_reads_the_sections_the_root_and_the_chain_from_the_tree(self) -> None:
        loaded = self.ursim()
        self.assertIs(loaded.app_config, loaded.config)
        self.assertIs(loaded.robot, loaded.config.robot)
        self.assertEqual(loaded.root, default_data_dir())
        self.assertEqual(loaded.profile, "ursim")
        self.assertEqual(loaded.layers, ("ursim",))
        self.assertEqual(dict(loaded.values), {})

    def test_a_tree_that_did_not_load_refuses_the_door_with_its_own_refusal(self) -> None:
        refused = ConfigTree.from_directory(profile="nosuch").load()
        self.assertFalse(refused.ok)
        for name in ("app_config", "robot"):
            with self.subTest(name=name):
                with self.assertRaises(ConfigError) as caught:
                    getattr(refused, name)
                self.assertEqual(str(caught.exception), refused.error)

    def test_a_tree_without_a_robot_block_refuses_with_the_loaders_sentence(self) -> None:
        root = self.copy_of_the_tree()
        (root / "robot" / "robot.yaml").unlink()
        loaded = ConfigTree.from_directory(root=root, profile=None).load()
        self.assertTrue(loaded.ok, loaded.error)
        with self.assertRaises(ConfigError) as door:
            loaded.robot
        with self.assertRaises(ConfigError) as loader:
            load_robot_config(root, profile=None)
        self.assertEqual(str(door.exception), str(loader.exception))


class ADirectoryWithoutProfilesTests(_Clean):
    """The owner: "es muss auch moeglich sein, dass man eine Directory ohne Profile reingeben kann, wie bei
    from_config wie jetzt". A copy of the tree with every overlay and both registries removed is such a directory."""

    def _a_directory_without_profiles(self) -> Path:
        root = self.copy_of_the_tree()
        for path in list(root.rglob("*.yaml")):
            if "." in path.stem:
                path.unlink()
        shutil.rmtree(root / "grippers")
        shutil.rmtree(root / "cameras")
        return root

    def test_it_loads_through_the_tree_as_load_config_reads_it(self) -> None:
        """⚠ GREEN BEFORE THIS STEP, ON PURPOSE. It pins what the owner asked to keep: a directory with no
        profile layers loads through the tree, with the profile unset and no WILLY_PROFILE, and with profile=None,
        and its config is the object `load_config(path)` returns."""
        root = self._a_directory_without_profiles()
        for loaded in (ConfigTree.from_directory(root=root).load(),
                       ConfigTree.from_directory(root=root, profile=None).load()):
            with self.subTest(profile=loaded.tree.profile):
                self.assertTrue(loaded.ok, loaded.error)
                self.assertEqual(loaded.tree.layers, ())
                self.assertEqual(loaded.chain, "(no profile)")
                self.assertIs(loaded.config, load_config(root))

    def test_a_value_given_in_memory_loads_there_too(self) -> None:
        root = self._a_directory_without_profiles()
        loaded = ConfigTree.from_directory(root=root, profile=None).load()
        changed = loaded.with_values({"robot.safety.payload.mass_kg": 1.25})
        self.assertTrue(changed.ok, changed.error)
        self.assertEqual(changed.config.robot.safety.payload.mass_kg, 1.25)
        self.assertEqual(changed.root, root.resolve())
        self.assertIsNone(changed.profile)

    def test_a_hand_named_in_memory_meets_the_registry_check_a_file_meets(self) -> None:
        """A directory with no `grippers/` that names a hand is refused by the tree (customer chain lane C1f),
        whether its file names the hand or `with_values` does, in the same words."""
        root = self._a_directory_without_profiles()
        in_memory = ConfigTree.from_directory(root=root, profile=None).load().with_values(
            {"robot.gripper.model": "robotiq_2f85"})
        robot_yaml = root / "robot" / "robot.yaml"
        text = robot_yaml.read_text(encoding="utf-8")
        self.assertEqual(text.count("    # model: robotiq_2f85\n"), 1, "the control: the base tree names no hand")
        robot_yaml.write_text(text.replace("    # model: robotiq_2f85\n", "    model: robotiq_2f85\n"),
                              encoding="utf-8")
        reload_config()
        in_the_file = ConfigTree.from_directory(root=root, profile=None).load()
        self.assertFalse(in_memory.ok)
        self.assertFalse(in_the_file.ok)
        self.assertEqual(in_memory.error, in_the_file.error)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
