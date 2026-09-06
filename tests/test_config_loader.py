from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.config import ConfigError, load_config, reload_config
from src.config.loader import available_profiles, set_active_profile


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "config"


def _copy_data_tree(tmp_dir: str) -> Path:
    target = Path(tmp_dir) / "data"
    shutil.copytree(DATA_DIR, target)
    return target


class ConfigLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_profile = os.environ.get("WILLY_PROFILE")
        self._previous_robot_ip = os.environ.get("WILLY_TEST_ROBOT_IP")
        set_active_profile(None)
        os.environ.pop("WILLY_TEST_ROBOT_IP", None)
        reload_config()

    def tearDown(self) -> None:
        set_active_profile(self._previous_profile)
        if self._previous_robot_ip is None:
            os.environ.pop("WILLY_TEST_ROBOT_IP", None)
        else:
            os.environ["WILLY_TEST_ROBOT_IP"] = self._previous_robot_ip
        reload_config()

    def test_default_config_loads_and_is_immutable(self) -> None:
        cfg = load_config()

        self.assertEqual(cfg.camera.cameras.active_mode, "auto")
        self.assertEqual(cfg.camera.hand_eye.eye_to_hand.mode, "eye_to_hand")
        self.assertEqual(cfg.camera.hand_eye.eye_in_hand.mode, "eye_in_hand")
        self.assertEqual(cfg.robot.vendor, "ur")

        with self.assertRaises(ValidationError):
            cfg.camera.cameras.active_mode = "rig"

    def test_profile_is_part_of_loader_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            (root / "app" / "runtime.ops.yaml").write_text(
                "runtime:\n"
                "  image_encoding:\n"
                "    frame_quality: 64\n",
                encoding="utf-8",
            )

            base_cfg = load_config(root)
            set_active_profile("ops")
            profiled_cfg = load_config(root)

        self.assertEqual(base_cfg.runtime.image_encoding.frame_quality, 60)
        self.assertEqual(profiled_cfg.runtime.image_encoding.frame_quality, 64)

    def test_profile_overlay_deep_merge_null_and_list_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            (root / "app" / "runtime.qa.yaml").write_text(
                "runtime:\n"
                "  image_encoding:\n"
                "    frame_quality: 66\n",
                encoding="utf-8",
            )
            (root / "camera" / "stereomatcher.qa.yaml").write_text(
                "stereomatcher:\n"
                "  uniquenessRatio: 12\n"
                "  speckleRange: null\n",
                encoding="utf-8",
            )
            (root / "camera" / "cam.qa.yaml").write_text(
                "cameras:\n"
                "  active_mode: rig\n"
                "  active_rig_id: qa_rig\n"
                "  rigs:\n"
                "    - rig_id: qa_rig\n"
                "      enabled: true\n"
                "      source: webcam_pair\n"
                "      frame_size: [640, 480]\n"
                "      fps: 30\n"
                "      backend: 700\n"
                "      max_cam_scan: 2\n"
                "      cam_left_id: 0\n"
                "      cam_right_id: 1\n"
                "      min_pairs: 4\n"
                "      max_pairs: 10\n"
                "      calibration_paths:\n"
                "        base_dir: calibration/qa_rig\n",
                encoding="utf-8",
            )

            set_active_profile("qa")
            cfg = load_config(root)

        self.assertEqual(cfg.runtime.image_encoding.frame_quality, 66)
        # A plain `null` overlay leaf KEEPS the base value, and the sibling the overlay never
        # named survives untouched -- the two halves of deep-merge, shown on one block.
        self.assertEqual(cfg.camera.stereomatcher.uniqueness_ratio, 12)   # overlay wins
        self.assertEqual(cfg.camera.stereomatcher.speckle_range, 1)       # null KEEPS the base
        self.assertEqual(cfg.camera.stereomatcher.temporal_alpha, 0.0)    # untouched sibling
        self.assertEqual([rig.rig_id for rig in cfg.camera.cameras.rigs], ["qa_rig"])

    def test_profile_overlay_reset_sentinel_unsets_a_base_value(self) -> None:
        # A plain `null` overlay leaf KEEPS the base (test above); the reset sentinel "__null__" is the one
        # way a profile overlay can force a base field back to None. This is exactly how the `sim` profile
        # drops the production models.*.optim.torch_dtype: auto (fp16 weights) back to unset — the sim
        # detector's validated fp32-weights recall path (fp16 was measured to drop the small overhead cube).
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            (root / "models" / "object.qa.yaml").write_text(
                "objectdetector:\n"
                "  optim:\n"
                '    torch_dtype: "__null__"\n'
                '    attn_implementation: "__null__"\n',
                encoding="utf-8",
            )
            set_active_profile("qa")
            cfg = load_config(root)

        optim = cfg.models.objectdetector.optim
        self.assertIsNone(optim.torch_dtype)          # base "auto" reset to None by the sentinel
        self.assertIsNone(optim.attn_implementation)  # base "eager" reset to None by the sentinel
        self.assertTrue(optim.channels_last)          # a field the overlay didn't touch keeps its base value

    def test_env_substitution_supports_defaults_and_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            robot_path = root / "robot" / "robot.yaml"
            robot_text = robot_path.read_text(encoding="utf-8")
            robot_path.write_text(
                robot_text.replace(
                    'ip: "192.168.1.100"',
                    'ip: "${WILLY_TEST_ROBOT_IP:-10.0.0.42}"',
                ),
                encoding="utf-8",
            )

            cfg = load_config(root)
            self.assertEqual(cfg.robot.ur.ip, "10.0.0.42")

            reload_config()
            os.environ["WILLY_TEST_ROBOT_IP"] = "10.0.0.99"
            cfg = load_config(root)
            self.assertEqual(cfg.robot.ur.ip, "10.0.0.99")

            reload_config()
            os.environ.pop("WILLY_TEST_ROBOT_IP", None)
            robot_path.write_text(
                robot_text.replace(
                    'ip: "192.168.1.100"',
                    'ip: "${WILLY_TEST_ROBOT_IP}"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "WILLY_TEST_ROBOT_IP"):
                load_config(root)

    def test_strict_schema_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            runtime_path = root / "app" / "runtime.yaml"
            runtime_path.write_text(
                runtime_path.read_text(encoding="utf-8").replace(
                    "  image_encoding:\n",
                    "  unexpected_key: true\n  image_encoding:\n",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "Extra inputs are not permitted"):
                load_config(root)

    def test_file_scoped_errors_for_missing_and_duplicate_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            # cam.yaml is a loader-declared REQUIRED input; the assertion is that a missing one is
            # named in the error, not that this particular file is required.
            (root / "camera" / "cam.yaml").unlink()

            with self.assertRaisesRegex(ConfigError, "cam.yaml"):
                load_config(root)

        reload_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            (root / "models" / "dupe.yaml").write_text(
                "stt:\n  model_id: duplicate\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "duplicate model key"):
                load_config(root)

    def test_typoed_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            set_active_profile("missing")

            with self.assertRaisesRegex(ConfigError, "WILLY_PROFILE"):
                load_config(root)


class ProfileChainTests(unittest.TestCase):
    """``WILLY_PROFILE`` accepts a chain of layers applied left-to-right.

    The layers of a real configuration are independent dimensions (the sim cell / which robot it drives
    / which camera rig it has). Composing them keeps each measured value in exactly one file instead of
    once per combination — and lets one dimension be varied alone, which is what makes an observed
    difference attributable to it.
    """

    def setUp(self) -> None:
        self._previous_profile = os.environ.get("WILLY_PROFILE")
        set_active_profile(None)
        reload_config()

    def tearDown(self) -> None:
        set_active_profile(self._previous_profile)
        reload_config()

    @staticmethod
    def _write_layers(root: Path) -> None:
        # Two scalars in ONE block is the whole point: the top layer must win on the key it names
        # and leave its sibling untouched. `stereomatcher` is the vehicle because it is the
        # smallest real block that still has two independent scalars -- `runtime.image_encoding`
        # was the original one and is down to a single field since the dead-key sweep.
        (root / "camera" / "stereomatcher.base.yaml").write_text(
            "stereomatcher:\n"
            "  uniquenessRatio: 12\n"
            "  speckleRange: 2\n",
            encoding="utf-8",
        )
        (root / "camera" / "stereomatcher.top.yaml").write_text(
            "stereomatcher:\n"
            "  speckleRange: 4\n",
            encoding="utf-8",
        )

    def test_later_layers_win_and_untouched_keys_survive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            self._write_layers(root)
            set_active_profile("base,top")
            cfg = load_config(root)

        self.assertEqual(cfg.camera.stereomatcher.speckle_range, 4)      # top layer wins
        self.assertEqual(cfg.camera.stereomatcher.uniqueness_ratio, 12)  # base layer survives

    def test_layer_order_is_significant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            self._write_layers(root)
            set_active_profile("top,base")
            reversed_cfg = load_config(root)

        self.assertEqual(reversed_cfg.camera.stereomatcher.speckle_range, 2)

    def test_a_single_layer_behaves_exactly_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            self._write_layers(root)
            set_active_profile("base")
            cfg = load_config(root)

        self.assertEqual(cfg.camera.stereomatcher.uniqueness_ratio, 12)
        self.assertEqual(cfg.camera.stereomatcher.speckle_range, 2)

    def test_whitespace_and_trailing_separators_are_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            self._write_layers(root)
            set_active_profile(" base , top , ")
            cfg = load_config(root)

        self.assertEqual(cfg.camera.stereomatcher.speckle_range, 4)

    def test_every_layer_must_exist(self) -> None:
        """Fail closed PER LAYER. A typo'd layer would otherwise merge as a silent no-op, and the cell
        would come up carrying another robot's geometry — exactly the failure the check prevents."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            self._write_layers(root)
            set_active_profile("base,typo")

            with self.assertRaisesRegex(ConfigError, "typo"):
                load_config(root)

    def test_chains_are_cached_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            self._write_layers(root)
            set_active_profile("base")
            one = load_config(root)
            set_active_profile("base,top")
            two = load_config(root)

        self.assertEqual(one.camera.stereomatcher.speckle_range, 2)
        self.assertEqual(two.camera.stereomatcher.speckle_range, 4)

    def test_profile_only_model_files_merge_across_layers(self) -> None:
        """A profile-only model file refined by a later layer must MERGE, not collide: the second layer
        is stating a delta on the same model, not introducing a duplicate."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            self._write_layers(root)
            # Turn stt.yaml into a PROFILE-ONLY file (no non-profile base counterpart), then have a
            # second layer refine it -- the branch that discovers such files.
            (root / "models" / "stt.yaml").rename(root / "models" / "stt.base.yaml")
            (root / "models" / "stt.top.yaml").write_text(
                "stt:\n  language: english\n", encoding="utf-8",
            )
            set_active_profile("base,top")
            cfg = load_config(root)

        self.assertEqual(cfg.models.stt.language, "english")             # later layer wins
        self.assertEqual(cfg.models.stt.model_id, "openai/whisper-small")  # earlier layer survives


class AvailableProfilesTests(unittest.TestCase):
    """What ``available_profiles()`` discovers, and what it therefore promises.

    Written 2026-08-09 to close a gap: the function's only coverage came indirectly, through
    ``profile_status()`` in ``test_config_editor_cli.py``, and that file went when
    ``backend/config/editor.py`` was deleted as unreachable. The function itself did not go -- it has
    live callers in ``examples/scripts/validate_config.py`` and ``inspect_profile.py``, which print the
    menu a newcomer picks from. So the coverage is rebuilt here, directly.
    """

    def test_a_layer_is_discovered_from_a_filename_anywhere_under_the_tree(self) -> None:
        """``rglob``, not one directory: layers live in ``robot/``, ``camera/`` and ``models/`` alike."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data"
            (root / "robot").mkdir(parents=True)
            (root / "models").mkdir(parents=True)
            (root / "robot" / "robot.yaml").write_text("robot: {}\n", encoding="utf-8")
            (root / "robot" / "robot.ur3e.yaml").write_text("robot: {}\n", encoding="utf-8")
            (root / "models" / "object.ur3e.yaml").write_text("object: {}\n", encoding="utf-8")
            (root / "models" / "object.tiltcam.yaml").write_text("object: {}\n", encoding="utf-8")

            found = available_profiles(root)

        # Sorted, de-duplicated across directories, and a file with no `.layer.` segment contributes
        # nothing -- `robot.yaml` is the base, not a profile called "yaml".
        self.assertEqual(found, ["tiltcam", "ur3e"])

    def test_the_shipped_layers_are_all_discoverable(self) -> None:
        """Names an operator types. A renamed overlay would drop one out of every printed menu."""
        found = set(available_profiles(DATA_DIR))
        self.assertLessEqual({"sim", "ur3e", "tiltcam"}, found)

    def test_any_new_overlay_silently_joins_the_menu(self) -> None:
        """The known sharp edge, pinned rather than merely noted.

        Discovery is by filename, so adding ``robot.experiment.yaml`` makes "experiment" appear in the
        list an operator is offered -- with no registration step and nothing asserting it should be
        there. That is a real property of the design (it is what makes layers cheap to add); this test
        exists so the next person meets it in a test rather than in a demo.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _copy_data_tree(tmp_dir)
            before = available_profiles(root)
            (root / "robot" / "robot.experiment.yaml").write_text("robot: {}\n", encoding="utf-8")
            after = available_profiles(root)

        self.assertNotIn("experiment", before)
        self.assertIn("experiment", after)

    def test_it_answers_for_an_empty_tree_instead_of_raising(self) -> None:
        """The callers print ``available_profiles() or '(none)'``, so an empty list must be reachable."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty = Path(tmp_dir) / "data"
            empty.mkdir()
            self.assertEqual(available_profiles(empty), [])


if __name__ == "__main__":
    unittest.main()