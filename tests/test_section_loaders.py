"""Each part of the product loads its own config section, and a fault in one leaves the others loadable.

Every `from_config` door validated the WHOLE tree: `load_robot_config` is `load_config` plus a getattr,
and `AppConfig` requires the camera files and three model blocks. So a broken camera file refused an arm
that needs no camera, a missing speech block refused a camera, and a malformed detector file refused
speech. The owner decided on 2026-09-11 that robot, camera, perception and speech each load only their
own section, through the same profile chain.

`load_config` is unchanged and still validates everything. These loaders are the doors for the parts
that are used on their own.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from src.config import ConfigError, load_config, reload_config
from src.config.loader import (
    load_camera_section,
    load_perception_section,
    load_robot_section,
    load_speech_section,
)

_DATA = Path(__file__).resolve().parents[1] / "config"
_PERCEPTION_FIELDS = ("segmenter", "rtdetr", "pipeline", "weights_root")


class _ScratchTree(unittest.TestCase):
    """A private copy of the shipped tree, so a test can break one file without touching the repo."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="willy_sections_")
        self.root = Path(self._tmp) / "data"
        shutil.copytree(_DATA, self.root)
        reload_config()

    def tearDown(self) -> None:
        reload_config()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _move_rtdetr(self) -> dict:
        """Take the `rtdetr` block out of `object.yaml`, the optional perception key, and hand it back."""
        objects = self.root / "models" / "object.yaml"
        data = yaml.safe_load(objects.read_text(encoding="utf-8"))
        self.assertIn("rtdetr", data, "the shipped tree stopped writing the key this test moves")
        block = data.pop("rtdetr")
        objects.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return block

    def _assert_loads(self, loader) -> None:
        try:
            loader(self.root, profile=None)
        except ConfigError as exc:
            self.fail(f"{loader.__name__} refused although the fault is in another section: {exc}")


class OnAHealthyTreeEachSectionIsTheWholeTreesSectionTests(_ScratchTree):
    def test_each_section_equals_the_same_section_of_load_config(self) -> None:
        for profile in (None, "sim", "ur3e,hande"):
            with self.subTest(profile=profile):
                whole = load_config(self.root, profile=profile)
                self.assertEqual(load_robot_section(self.root, profile=profile), whole.robot)
                self.assertEqual(load_camera_section(self.root, profile=profile), whole.camera)
                self.assertEqual(load_speech_section(self.root, profile=profile), whole.models.stt)
                perception = load_perception_section(self.root, profile=profile)
                for name in _PERCEPTION_FIELDS:
                    self.assertEqual(getattr(perception, name), getattr(whole.models, name), name)


class ThePerceptionSectionIsWhatThePerceptionSpecReadsTests(unittest.TestCase):
    """The perception section is its own model, so it could drift from the whole `models` block.

    Its fields are derived from `PerceptionSpec`, the one consumer, rather than listed a third time, and
    each one must carry the whole block's annotation, requiredness and default.
    """

    def test_its_fields_are_the_specs_fields_with_the_whole_blocks_annotations(self) -> None:
        import dataclasses

        from src.config.schema.app import ModelsConfig, PerceptionModelsConfig
        from src.models.perception_spec import PerceptionSpec

        spec = tuple(item.name for item in dataclasses.fields(PerceptionSpec))
        self.assertEqual(tuple(PerceptionModelsConfig.model_fields), spec)
        self.assertEqual(spec, _PERCEPTION_FIELDS, "the four the scratch-tree tests compare moved")
        for name, info in PerceptionModelsConfig.model_fields.items():
            whole = ModelsConfig.model_fields[name]
            with self.subTest(field=name):
                self.assertEqual(info.annotation, whole.annotation)
                self.assertEqual(info.is_required(), whole.is_required())
                self.assertEqual(info.default, whole.default)

    def test_the_type_a_public_loader_returns_is_on_the_public_schema_surface(self) -> None:
        import src.config.schema as public
        from src.config.schema.app import PerceptionModelsConfig

        self.assertIs(getattr(public, "PerceptionModelsConfig", None), PerceptionModelsConfig)
        self.assertIn("PerceptionModelsConfig", public.__all__)


class AFaultStaysInItsSectionTests(_ScratchTree):
    def test_a_malformed_camera_file_refuses_only_the_camera(self) -> None:
        (self.root / "camera" / "cam.yaml").write_text("cameras: [never closed\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.root, profile=None)
        with self.assertRaises(ConfigError):
            load_camera_section(self.root, profile=None)
        for loader in (load_robot_section, load_speech_section, load_perception_section):
            with self.subTest(loader=loader.__name__):
                self._assert_loads(loader)

    def test_a_missing_speech_block_refuses_only_speech(self) -> None:
        (self.root / "models" / "stt.yaml").unlink()
        with self.assertRaises(ConfigError):
            load_config(self.root, profile=None)
        with self.assertRaises(ConfigError) as ctx:
            load_speech_section(self.root, profile=None)
        self.assertIn("stt", str(ctx.exception))
        for loader in (load_robot_section, load_camera_section, load_perception_section):
            with self.subTest(loader=loader.__name__):
                self._assert_loads(loader)

    def test_a_robot_file_the_schema_rejects_refuses_only_the_robot(self) -> None:
        path = self.root / "robot" / "robot.yaml"
        text, count = re.subn(r"(?m)^robot:[ \t]*$", "robot:\n  not_a_key: 1",
                              path.read_text(encoding="utf-8"), count=1)
        self.assertEqual(count, 1, "the scratch edit did not find the top-level robot key")
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.root, profile=None)
        with self.assertRaises(ConfigError) as ctx:
            load_robot_section(self.root, profile=None)
        self.assertIn("robot.not_a_key", str(ctx.exception),
                      "the section names the key the way the whole-tree load names it")
        self.assertIn("robot.yaml", str(ctx.exception), "and points at the file the key was written in")
        for loader in (load_camera_section, load_speech_section, load_perception_section):
            with self.subTest(loader=loader.__name__):
                self._assert_loads(loader)

    def test_a_malformed_detector_file_refuses_perception_and_leaves_speech(self) -> None:
        (self.root / "models" / "object.yaml").write_text(
            "pipeline: [never closed\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_perception_section(self.root, profile=None)
        for loader in (load_robot_section, load_camera_section, load_speech_section):
            with self.subTest(loader=loader.__name__):
                self._assert_loads(loader)

    def test_an_unreadable_file_that_cannot_hold_a_perception_key_leaves_perception_loadable(self) -> None:
        """All four perception keys are written in `object.yaml`, `segmenting.yaml` or `weights.yaml`, so a broken
        `stt.yaml` could not have held one of them."""
        for name in ("stt.yaml",):
            path = self.root / "models" / name
            original = path.read_text(encoding="utf-8")
            path.write_text("broken: [never closed\n", encoding="utf-8")
            try:
                with self.subTest(file=name):
                    self._assert_loads(load_perception_section)
            finally:
                path.write_text(original, encoding="utf-8")

    def test_a_key_written_nowhere_beside_an_unreadable_file_refuses_rather_than_defaults(self) -> None:
        """The other half of the rule. With `rtdetr` written in no file, a broken `stt.yaml` might
        have held it, so perception refuses and names the file instead of taking the schema default."""
        self._move_rtdetr()
        # Control: with every file readable, the default for a key nobody wrote is an honest answer.
        self._assert_loads(load_perception_section)
        (self.root / "models" / "stt.yaml").write_text("stt: [never closed\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_perception_section(self.root, profile=None)
        self.assertIn("stt.yaml", str(ctx.exception))
        self.assertIn("rtdetr", str(ctx.exception))


class AnUnknownModelsKeyIsNotDroppedTests(_ScratchTree):
    """`load_config` refuses a top-level models key the schema does not know. A section that copied only
    the keys it reads would drop a misspelled one without a word and take the schema default in its
    place, which is the fallback this loader refuses everywhere else. An unknown key is a section's
    fault when it sits in a file that section reads, or when one of the section's own keys is missing
    and the unknown one may be it spelled wrong."""

    def test_a_misspelled_key_in_a_file_perception_reads_refuses_perception_and_leaves_speech(self) -> None:
        path = self.root / "models" / "object.yaml"
        text, count = re.subn(r"(?m)^rtdetr:", "rtdter:", path.read_text(encoding="utf-8"), count=1)
        self.assertEqual(count, 1, "the shipped object.yaml stopped writing rtdetr")
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.root, profile=None)
        with self.assertRaises(ConfigError) as ctx:
            load_perception_section(self.root, profile=None)
        self.assertIn("rtdter", str(ctx.exception))
        self.assertIn("object.yaml", str(ctx.exception))
        self._assert_loads(load_speech_section)

    def test_an_unknown_key_beside_all_four_still_refuses_perception(self) -> None:
        path = self.root / "models" / "object.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "\nnot_a_model_key: 1\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_perception_section(self.root, profile=None)
        self.assertIn("not_a_model_key", str(ctx.exception))
        self._assert_loads(load_speech_section)

    def test_a_misspelled_key_in_a_file_of_its_own_refuses_the_section_missing_that_key(self) -> None:
        rtdetr = self._move_rtdetr()
        (self.root / "models" / "extra.yaml").write_text(
            yaml.safe_dump({"rtdter": rtdetr}, sort_keys=False), encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_perception_section(self.root, profile=None)
        self.assertIn("rtdter", str(ctx.exception))
        self._assert_loads(load_speech_section)

    def test_an_unknown_key_in_the_speech_file_refuses_speech_and_leaves_perception(self) -> None:
        path = self.root / "models" / "stt.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "\nstt_extra: 1\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_speech_section(self.root, profile=None)
        self.assertIn("stt_extra", str(ctx.exception))
        self._assert_loads(load_perception_section)


class AnUnknownKeyIsNamedWhereItIsWrittenTests(_ScratchTree):
    """An unknown key written in an overlay or in a profile-only file is reported at that file. Merged data
    carries one path, and naming it sends an operator to a file that holds no such key."""

    def test_a_misspelled_key_in_an_overlay_names_the_overlay(self) -> None:
        # The shipped tree carries no models overlay since 2026-10-17; a cell's own is the case.
        overlay = self.root / "models" / "object.sim.yaml"
        overlay.write_text("rtdetr:\n  threshold: 0.3\nrtdter: {}\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(self.root, profile="sim")
        with self.assertRaises(ConfigError) as ctx:
            load_perception_section(self.root, profile="sim")
        message = str(ctx.exception)
        self.assertIn("rtdter", message)
        self.assertIn("object.sim.yaml", message)
        self.assertNotIn(str(self.root / "models" / "object.yaml") + ":", message)

    def test_a_misspelled_key_in_a_profile_only_file_names_the_layer_that_wrote_it(self) -> None:
        rtdetr = self._move_rtdetr()
        (self.root / "models" / "extra.sim.yaml").write_text(
            yaml.safe_dump({"rtdter": rtdetr}, sort_keys=False), encoding="utf-8")
        (self.root / "models" / "extra.ur3e.yaml").write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_perception_section(self.root, profile="sim,ur3e")
        message = str(ctx.exception)
        self.assertIn("rtdter", message)
        self.assertIn("extra.sim.yaml", message)
        self.assertNotIn("extra.ur3e.yaml", message)


class TheRuleHoldsUnderAProfileChainTests(_ScratchTree):
    """The fault rule has to hold for overlays and for profile-only files, not only for base files."""

    def test_a_broken_overlay_of_a_perception_file_refuses_perception_under_that_profile(self) -> None:
        overlay = self.root / "models" / "object.sim.yaml"  # a cell's own overlay; the shipped tree carries none
        overlay.write_text("pipeline: [never closed\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_perception_section(self.root, profile="sim")
        self.assertIn("object.sim.yaml", str(ctx.exception))
        self._assert_loads(load_perception_section)
        load_speech_section(self.root, profile="sim")

    def test_a_profile_only_file_contributes_its_keys_and_its_faults(self) -> None:
        rtdetr = self._move_rtdetr()
        extra = self.root / "models" / "extra.sim.yaml"
        extra.write_text(yaml.safe_dump({"rtdetr": rtdetr}, sort_keys=False), encoding="utf-8")
        whole = load_config(self.root, profile="sim")
        self.assertIsNotNone(whole.models.rtdetr, "the profile-only file must be what supplies the key")
        self.assertEqual(load_perception_section(self.root, profile="sim").rtdetr, whole.models.rtdetr)
        extra.write_text("rtdetr: [never closed\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_perception_section(self.root, profile="sim")
        self.assertIn("extra.sim.yaml", str(ctx.exception))
        self.assertIn("rtdetr", str(ctx.exception))
        load_speech_section(self.root, profile="sim")


class ASectionNamesWhatItRefusesTests(_ScratchTree):
    def test_a_duplicate_of_a_key_the_section_reads_names_both_files(self) -> None:
        stt = yaml.safe_load((self.root / "models" / "stt.yaml").read_text(encoding="utf-8"))
        (self.root / "models" / "zz_second.yaml").write_text(
            yaml.safe_dump(stt, sort_keys=False), encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_speech_section(self.root, profile=None)
        self.assertIn("stt.yaml", str(ctx.exception))
        self.assertIn("zz_second.yaml", str(ctx.exception))
        self._assert_loads(load_perception_section)

    def test_each_section_names_a_schema_fault_by_its_place_in_the_tree(self) -> None:
        cases = (
            (load_camera_section, ("camera", "cam.yaml"), "cameras", "camera.cameras.not_a_key"),
            (load_speech_section, ("models", "stt.yaml"), "stt", "models.stt.not_a_key"),
            (load_perception_section, ("models", "object.yaml"), "rtdetr", "models.rtdetr.not_a_key"),
        )
        for loader, (folder, name), block, dotted in cases:
            with self.subTest(loader=loader.__name__):
                path = self.root / folder / name
                original = path.read_text(encoding="utf-8")
                data = yaml.safe_load(original)
                data[block] = {**data[block], "not_a_key": 1}
                path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
                try:
                    with self.assertRaises(ConfigError) as ctx:
                        loader(self.root, profile=None)
                    self.assertIn(dotted, str(ctx.exception))
                    self.assertIn(name, str(ctx.exception), "the refusal points at the file")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_an_empty_robot_block_is_called_empty_rather_than_missing(self) -> None:
        (self.root / "robot" / "robot.yaml").write_text("robot:\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_robot_section(self.root, profile=None)
        self.assertNotIn("does not exist", str(ctx.exception))
        self.assertIn("empty", str(ctx.exception))


class TheCameraSectionCarriesARigCalibrationTests(_ScratchTree):
    """`camera.cameras.rigs[<id>].extrinsics` is a camera section key, so the camera section loader alone reads it."""

    def test_the_camera_section_loader_carries_the_key(self) -> None:
        cam = self.root / "camera" / "cam.yaml"
        text = cam.read_text(encoding="utf-8").replace("\r\n", "\n")
        # The rig's own list item, so the block lands among that rig's keys at their indentation,
        # whatever comments sit between its other keys.
        marker = "    - rig_id: realsense_d435\n"
        self.assertEqual(text.count(marker), 1)
        block = ("      extrinsics:\n        mounting_mode: eye_to_hand\n"
                 "        artifact_path: calibration/eth_realsense_d435.json\n")
        cam.write_text(text.replace(marker, marker + block), encoding="utf-8")
        camera = load_camera_section(self.root, profile=None)
        rig = next(r for r in camera.cameras.rigs if r.rig_id == "realsense_d435")
        assert rig.extrinsics is not None
        # Written relative, read against the tree's folder (src/config/paths.py).
        self.assertEqual((rig.extrinsics.mounting_mode, rig.extrinsics.artifact_path),
                         ("eye_to_hand", (self.root.resolve() / "calibration/eth_realsense_d435.json").as_posix()))


if __name__ == "__main__":
    unittest.main()
