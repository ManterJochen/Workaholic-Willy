"""A relative path in a config tree is read against the tree's folder, wherever that folder is.

The owner, 2026-09-24: once the config lives in a folder outside the repository, the paths in it no
longer resolve. Measured on the code before this rule, with a copy of `config/` at
`%TEMP%/willy_cell_*/cell`, its calibration beside it at `cell/calibration/eth_realsense_d435.json`, and
the program started in another folder:

    camera.cameras.rigs['realsense_d435'].extrinsics names calibration/eth_realsense_d435.json
    (eye_to_hand), which does not load: there is no file at that path relative to the working
    directory, C:\\Users\\...\\willy_elsewhere_1cmbg5f3.

Every consumer did `Path(value)`, so a relative path was read against wherever the program started, and
the tree's own folder played no part. The rule now (`src/config/paths.py`): a relative path is read
against the tree's folder, `${WILLY_PROJECT_ROOT}/...` names the repository, an absolute path is read as
written, and a schema default, written nowhere, names the repository's file.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import re
import shutil
import tempfile
import typing
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from src.calibration.extrinsics import Extrinsics
from src.calibration.rig_calibration import RigArtifactMissing, RigCalibration
from src.calibration.serialization import save_extrinsics
from src.config import ConfigError, load_config, load_tree, reload_config
from src.config.loader import load_camera_section, load_robot_section, load_speech_section
from src.config.paths import (
    PROJECT_ROOT_ANCHOR,
    is_config_path_field,
    path_note,
    repository_root,
    resolve_config_path,
)
from src.config.schema._base import ConfigPath, StrictModel
from src.config.schema.app import AppConfig
from src.config.schema.camera import RGBDDeviceRigConfig
from src.geometry import Frame, Transform

_REPO = Path(__file__).resolve().parents[1]
_RIG = "realsense_d435"
_RIG_MARKER = f"    - rig_id: {_RIG}\n"


def _posix(path: Path) -> str:
    return Path(os.path.normpath(path)).as_posix()


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.count(old) == 1, f"{old!r} is not written once in {path}"
    _write(path, text.replace(old, new))


def _artifact(path: Path) -> Path:
    return save_extrinsics(path, Extrinsics(
        transform=Transform(translation_mm=np.array([400.0, -25.0, 812.5]),
                            quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                            from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        rmse_mm=0.8, max_error_mm=1.5, num_samples=22,
        captured_at=dt.datetime(2026, 9, 15, tzinfo=dt.timezone.utc), rig_id=_RIG))


class _TreeOutsideTheRepository(unittest.TestCase):
    """A copy of the shipped tree in a folder outside the repository, run from a third folder.

    The third folder is the point: the working directory must play no part in where a path points.
    """

    def setUp(self) -> None:
        self.outside = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="willy_cell_"))).resolve()
        self.root = self.outside / "cell"
        shutil.copytree(_REPO / "config", self.root)
        elsewhere = self.enterContext(tempfile.TemporaryDirectory(prefix="willy_elsewhere_"))
        self.enterContext(contextlib.chdir(elsewhere))
        reload_config()
        self.addCleanup(reload_config)

    def _declare_the_calibration(self, artifact_path: str) -> None:
        block = ("      extrinsics:\n        mounting_mode: eye_to_hand\n"
                 f"        artifact_path: {artifact_path}\n")
        _edit(self.root / "camera" / "cam.yaml", _RIG_MARKER, _RIG_MARKER + block)

    def _rig(self) -> RGBDDeviceRigConfig:
        rigs = load_config(self.root, profile=None).camera.cameras.rigs
        return next(rig for rig in rigs if rig.rig_id == _RIG)


class ARelativePathIsReadAgainstTheConfigFolderTests(_TreeOutsideTheRepository):

    def test_the_owners_cell_loads_its_calibration_from_beside_its_config(self) -> None:
        """The report, end to end: the artifact beside the tree, the block naming it relatively."""
        _artifact(self.root / "calibration" / "eth_realsense_d435.json")
        self._declare_the_calibration("calibration/eth_realsense_d435.json")
        rig = self._rig()
        assert rig.extrinsics is not None
        self.assertEqual(rig.extrinsics.artifact_path, _posix(self.root / "calibration" / "eth_realsense_d435.json"))
        calibration = RigCalibration.from_config(_RIG, rig.extrinsics)
        np.testing.assert_allclose(calibration.camera_to_base().translation_mm, [400.0, -25.0, 812.5])

    def test_the_folder_is_the_trees_root_and_not_the_folder_of_the_file(self) -> None:
        """cam.yaml sits in camera/, and its relative path is still read from the tree's root."""
        self._declare_the_calibration("calibration/eth_realsense_d435.json")
        rig = self._rig()
        assert rig.extrinsics is not None
        self.assertNotIn("/camera/calibration/", rig.extrinsics.artifact_path)

    def test_a_dot_dot_leaves_the_folder_and_is_folded_away(self) -> None:
        self._declare_the_calibration("../shared/eth_realsense_d435.json")
        rig = self._rig()
        assert rig.extrinsics is not None
        self.assertEqual(rig.extrinsics.artifact_path, _posix(self.outside / "shared" / "eth_realsense_d435.json"))

    def test_every_door_reads_it_the_same_way(self) -> None:
        """The section loaders validate under the same folder as the whole-tree load."""
        self._declare_the_calibration("calibration/eth_realsense_d435.json")
        whole = self._rig().extrinsics
        section = next(r for r in load_camera_section(self.root, profile=None).cameras.rigs if r.rig_id == _RIG)
        self.assertEqual(section.extrinsics, whole)
        tree = load_tree(None, root=self.root)
        self.assertTrue(tree.ok, tree.error)
        rig = next(r for r in tree.app_config.camera.cameras.rigs if r.rig_id == _RIG)
        self.assertEqual(rig.extrinsics, whole)

    def test_a_value_given_in_memory_is_read_against_the_same_folder(self) -> None:
        tree = load_tree(None, root=self.root).with_values({"models.stt.vad_model_path": "weights/vad.jit"})
        self.assertTrue(tree.ok, tree.error)
        self.assertEqual(tree.app_config.models.stt.vad_model_path, _posix(self.root / "weights" / "vad.jit"))

    def test_the_stereo_paths_derived_from_a_relative_base_follow_it(self) -> None:
        _edit(self.root / "camera" / "cam.yaml", "base_dir: ../calibration/webcam_main", "base_dir: stereo/main")
        rig = next(r for r in load_config(self.root, profile=None).camera.cameras.rigs if r.rig_id == "webcam_main")
        paths = rig.calibration_paths
        base = _posix(self.root / "stereo" / "main")
        self.assertEqual((paths.base_dir, paths.stereo_map_file, paths.left_images_glob),
                         (base, f"{base}/stereoMap.xml", f"{base}/left/*.png"))


class AnAbsolutePathAndTheAnchorTests(_TreeOutsideTheRepository):

    def test_an_absolute_path_is_read_as_written(self) -> None:
        written = (self.outside / "elsewhere" / "eth.json").as_posix()
        self._declare_the_calibration(written)
        rig = self._rig()
        assert rig.extrinsics is not None
        self.assertEqual(rig.extrinsics.artifact_path, written)

    def test_a_home_path_is_expanded(self) -> None:
        self._declare_the_calibration("~/cell/eth.json")
        rig = self._rig()
        assert rig.extrinsics is not None
        self.assertEqual(rig.extrinsics.artifact_path, _posix(Path("~/cell/eth.json").expanduser()))

    def test_the_shipped_model_paths_still_name_the_repositorys_weights_from_a_copied_tree(self) -> None:
        """They are written with the anchor, so a copy of config/ outside the repository keeps them."""
        from src.utility.paths import weights_root

        models = load_config(self.root, profile=None).models
        for name, path in (("objectdetector", models.objectdetector.model_path),
                           ("segmenter", models.segmenter.model_path),
                           ("stt", models.stt.model_path), ("vad", models.stt.vad_model_path)):
            with self.subTest(model=name):
                self.assertTrue(path.startswith(weights_root().as_posix() + "/"), path)

    def test_the_anchor_follows_the_environment_variable_and_survives_a_windows_value(self) -> None:
        """Left in place by the substitution: a backslashed value inside the shipped double-quoted
        YAML string would otherwise be read as escapes and stop the file from parsing."""
        other = self.outside / "other checkout"
        with mock.patch.dict(os.environ, {"WILLY_PROJECT_ROOT": str(other)}):  # backslashed on Windows
            stt = load_speech_section(self.root, profile=None)
        self.assertEqual(stt.vad_model_path,
                         _posix(other.resolve() / "assets/models/hf/speech/silero-vad-6.2.1/silero_vad.jit"))

    def test_the_anchor_takes_no_default_and_no_place_but_the_start(self) -> None:
        stt = self.root / "models" / "stt.yaml"
        original = stt.read_text(encoding="utf-8").replace("\r\n", "\n")
        line = next(ln for ln in original.splitlines() if ln.strip().startswith("vad_model_path:"))
        for written in ('"${WILLY_PROJECT_ROOT:-/opt/willy}/x.jit"', '"weights/${WILLY_PROJECT_ROOT}/x.jit"',
                        '"${WILLY_PROJECT_ROOT}x.jit"'):
            with self.subTest(written=written):
                _write(stt, original.replace(line, f"  vad_model_path: {written}"))
                with self.assertRaises(ConfigError) as caught:
                    load_speech_section(self.root, profile=None)
                self.assertIn("models.stt.vad_model_path", str(caught.exception))
                self.assertIn(PROJECT_ROOT_ANCHOR, str(caught.exception))


class TheAnchorIsRefusedOutsideAPathFieldTests(_TreeOutsideTheRepository):
    """The substitution leaves the anchor in the text for the path rule, and only a path field has that rule.

    Measured before the fix, 2026-09-24: ``robot.sim.assets_root: "${WILLY_PROJECT_ROOT}/isaac_assets"``
    loaded and held the literal text, the asset root Isaac would be handed. Before the anchor existed the same line
    was substituted, or refused when the variable was unset; it had become the one ``${...}`` that could
    pass through a load unread.
    """

    def _write_the_sim_asset_root(self, written: str) -> None:
        _edit(self.root / "robot" / "robot.sim.yaml", "    assets_root: null\n", f"    assets_root: {written}\n")

    def test_a_key_that_names_no_file_refuses_the_anchor_and_names_the_key(self) -> None:
        self._write_the_sim_asset_root('"${WILLY_PROJECT_ROOT}/isaac_assets"')
        doors = {"whole tree": lambda: load_config(self.root, profile="sim"),
                 "robot section": lambda: load_robot_section(self.root, profile="sim")}
        for door, load in doors.items():
            with self.subTest(door=door):
                with self.assertRaises(ConfigError) as caught:
                    load()
                message = str(caught.exception)
                self.assertIn("robot.sim.assets_root", message)
                self.assertIn(PROJECT_ROOT_ANCHOR, message)
                self.assertIn("robot.sim.yaml", message)

    def test_the_anchor_with_a_default_is_refused_there_too(self) -> None:
        """Left as written by the substitution as well, so it would reach the key as text just the same."""
        self._write_the_sim_asset_root('"${WILLY_PROJECT_ROOT:-/opt/willy}/isaac_assets"')
        with self.assertRaises(ConfigError) as caught:
            load_robot_section(self.root, profile="sim")
        self.assertIn("robot.sim.assets_root", str(caught.exception))

    def test_a_path_field_in_the_same_tree_still_reads_the_repository(self) -> None:
        """The control: the refusal is about the key, not about the anchor."""
        loaded = load_config(self.root, profile="sim")
        self.assertIsNone(loaded.robot.sim.assets_root)
        self.assertEqual(loaded.models.stt.vad_model_path,
                         _posix(repository_root() / "assets/models/hf/speech/silero-vad-6.2.1/silero_vad.jit"))

    def test_a_model_built_in_code_refuses_it_in_a_string_and_in_a_list_and_reads_it_in_a_path(self) -> None:
        """The rule sits on the schema's base, so no door around the loader lets the text through either."""
        probe = _AnchorProbe(weights=f"{PROJECT_ROOT_ANCHOR}/assets/x.bin", note="plain text")
        self.assertEqual(probe.weights, _posix(repository_root() / "assets/x.bin"))
        for given in ({"note": f"{PROJECT_ROOT_ANCHOR}/isaac"}, {"tags": ["a", f"see {PROJECT_ROOT_ANCHOR}"]},
                      {"labels": {"k": f"{PROJECT_ROOT_ANCHOR}/x"}}):
            with self.subTest(given=given):
                with self.assertRaises(ValidationError) as caught:
                    _AnchorProbe.model_validate(given)
                self.assertIn(PROJECT_ROOT_ANCHOR, str(caught.exception))
                self.assertEqual(caught.exception.errors()[0]["loc"][0], next(iter(given)))


class _AnchorProbe(StrictModel):
    note: str = ""
    tags: tuple[str, ...] = ()
    labels: dict[str, str] = {}
    weights: ConfigPath = ""


class TheAnchorIsWrittenInAnyYamlStyleTests(_TreeOutsideTheRepository):
    """Unquoted inside ``{...}`` or ``[...]``, the anchor's own braces are YAML flow indicators.

    Measured before the fix, 2026-09-24: ``extrinsics: {mounting_mode: eye_to_hand, artifact_path:
    ${WILLY_PROJECT_ROOT}/calibration/eth.json}`` stopped the file with "while parsing a flow mapping",
    where any other ``${VAR}`` in the same place loads, because the substitution replaces it first. The
    variable is set to a Windows spelling throughout: the anchor is left out of the substitution so that
    a backslashed value never lands inside a double-quoted string, where ``\\U`` and ``\\n`` are escapes.
    """

    def setUp(self) -> None:
        super().setUp()
        self.other = self.outside / "new checkout" / "tools"  # `\n` and `\t` once spelled with backslashes
        self.enterContext(mock.patch.dict(os.environ, {"WILLY_PROJECT_ROOT": str(self.other)}))

    def _in_the_other_checkout(self, rest: str) -> str:
        return _posix(self.other.resolve() / rest)

    def test_unquoted_in_a_flow_mapping_it_loads_and_resolves(self) -> None:
        _write(self.root / "models" / "stt.yaml",
               "stt: {model_id: openai/whisper-large-v3-turbo, model_path: ${WILLY_PROJECT_ROOT}/weights/whisper,\n"
               "      vad_model_path: ${WILLY_PROJECT_ROOT}/weights/silero_vad.jit, samplerate: 16000,\n"
               "      blocksize: 8000, channels: 1, dtype: int16, language: auto, task: transcribe, local: true}\n")
        stt = load_speech_section(self.root, profile=None)
        self.assertEqual((stt.model_path, stt.vad_model_path),
                         (self._in_the_other_checkout("weights/whisper"),
                          self._in_the_other_checkout("weights/silero_vad.jit")))

    def test_unquoted_in_a_flow_mapping_inside_a_block_it_loads_and_resolves(self) -> None:
        _edit(self.root / "camera" / "cam.yaml", _RIG_MARKER, _RIG_MARKER + (
            "      extrinsics: {mounting_mode: eye_to_hand, "
            "artifact_path: ${WILLY_PROJECT_ROOT}/calibration/eth.json}\n"))
        rig = self._rig()
        assert rig.extrinsics is not None
        self.assertEqual(rig.extrinsics.artifact_path, self._in_the_other_checkout("calibration/eth.json"))

    def test_unquoted_in_a_flow_sequence_it_loads_as_the_anchor_and_resolves(self) -> None:
        """No path key of the schema is a list, so this reads the file the way every door reads it."""
        from src.config.loader import _load_yaml

        written = self.outside / "flow.yaml"
        _write(written, "paths: [${WILLY_PROJECT_ROOT}/a.jit, {b: ${WILLY_PROJECT_ROOT}/b.jit}, "
                        "[${WILLY_PROJECT_ROOT}/c.jit]]\n")
        data = _load_yaml(written)
        self.assertEqual(data, {"paths": [f"{PROJECT_ROOT_ANCHOR}/a.jit", {"b": f"{PROJECT_ROOT_ANCHOR}/b.jit"},
                                          [f"{PROJECT_ROOT_ANCHOR}/c.jit"]]})
        self.assertEqual(resolve_config_path(data["paths"][0], None), self._in_the_other_checkout("a.jit"))

    def test_the_block_and_quoted_forms_still_load_and_resolve(self) -> None:
        stt = self.root / "models" / "stt.yaml"
        original = stt.read_text(encoding="utf-8").replace("\r\n", "\n")
        line = next(ln for ln in original.splitlines() if ln.strip().startswith("vad_model_path:"))
        for written in ("${WILLY_PROJECT_ROOT}/x.jit", '"${WILLY_PROJECT_ROOT}/x.jit"',
                        "'${WILLY_PROJECT_ROOT}/x.jit'", "${WILLY_PROJECT_ROOT}/x.jit  # a comment"):
            with self.subTest(written=written):
                _write(stt, original.replace(line, f"  vad_model_path: {written}"))
                self.assertEqual(load_speech_section(self.root, profile=None).vad_model_path,
                                 self._in_the_other_checkout("x.jit"))

    def test_a_yaml_error_quotes_the_anchor_as_written(self) -> None:
        """Whatever the loader parses in its place stays out of the refusal the operator reads, in the
        quoted line too, which the parser cuts short: ``stt: {vad_model_path: ${WILLY_PROJECT ...``."""
        _write(self.root / "models" / "stt.yaml", "stt: {vad_model_path: ${WILLY_PROJECT_ROOT}/x.jit\n")
        with self.assertRaises(ConfigError) as caught:
            load_speech_section(self.root, profile=None)
        self.assertIn("{vad_model_path: ${WILLY_PROJECT", str(caught.exception))
        self.assertNotIn("__WILLY", str(caught.exception))

    def test_the_text_the_loader_parses_in_the_anchors_place_cannot_be_written(self) -> None:
        """It is turned back into the anchor after parsing, so a file that wrote it would gain an anchor."""
        from src.config.loader import _ANCHOR_STAND_IN

        _edit(self.root / "models" / "stt.yaml", 'dtype: "int16"', f'dtype: "{_ANCHOR_STAND_IN}"')
        with self.assertRaises(ConfigError) as caught:
            load_speech_section(self.root, profile=None)
        self.assertIn(_ANCHOR_STAND_IN, str(caught.exception))
        self.assertIn("stt.yaml", str(caught.exception))


class ADefaultNamesTheRepositorysFileTests(_TreeOutsideTheRepository):

    def test_an_unwritten_default_names_the_committed_artifact_in_the_repository(self) -> None:
        """Nobody wrote it, so the tree's folder plays no part: the committed v1 success model is found."""
        artifact = load_config(self.root, profile=None).robot.grasping.success_model.artifact_dir
        self.assertEqual(artifact, _posix(repository_root() / "assets/models/success_probability/v1"))
        self.assertTrue((Path(artifact) / "model.json").is_file(), artifact)

    def test_a_rig_that_names_no_calibration_folder_gets_the_repositorys(self) -> None:
        cam = self.root / "camera" / "cam.yaml"
        text = cam.read_text(encoding="utf-8").replace("\r\n", "\n")
        start = text.index(_RIG_MARKER)
        block = text.index("      calibration_paths:\n", start)
        end = text.index("base_dir: ../calibration/realsense_d435\n", block) + len("base_dir: ../calibration/realsense_d435\n")
        _write(cam, text[:block] + text[end:])
        self.assertEqual(self._rig().calibration_paths.base_dir, _posix(repository_root() / "calibration" / _RIG))

    def test_decisions_do_not_call_an_anchored_default_a_decision(self) -> None:
        loaded = load_tree(None, root=self.root)
        self.assertNotIn("success_model.artifact_dir", loaded.decisions(section="robot.grasping"))


#: Keys named like a path that name no file on this machine, so the rule does not apply to them.
_NOT_A_FILE = {
    "prim_path": "a USD stage path, /World/...",
    "robot_prim_path": "a USD stage path",
    "gripper_prim_path": "a USD stage path",
    "usd_asset_path": "relative to the Isaac asset root, and Isaac resolves it",
    "assets_root": "the Isaac asset root, which may be a URL (omniverse://, https://) that Isaac resolves",
}


class EveryPathKeyIsReadByTheRuleTests(unittest.TestCase):
    """The rule holds only for the fields typed ConfigPath, so the schema is held to it as a whole."""

    def test_every_key_named_like_a_path_is_typed_config_path(self) -> None:
        """A key added later as a bare ``str`` is read against the working directory again, and silently:
        nothing fails until a tree outside the repository is loaded, which is the report this file answers."""
        untyped = [
            f"{model.__name__}.{name}" for model, name, field in _fields(AppConfig)
            if re.search(r"(path|_dir|_file|_glob|_root|_folder)$", name) and name not in _NOT_A_FILE
            and not is_config_path_field(field)
        ]
        self.assertEqual(untyped, [], "type these ConfigPath (src.config.schema._base), or name in _NOT_A_FILE "
                                      "why the key names no file")

    def test_every_path_default_is_empty_or_anchored_and_validated(self) -> None:
        """The line between written and default is pydantic's: a default is validated only when the field
        says so. A relative default would then be read against whichever tree loaded it, and an anchored
        one left unvalidated would reach a consumer as the literal ``${WILLY_PROJECT_ROOT}/...``."""
        checked = 0
        for model, name, field in _fields(AppConfig):
            if not is_config_path_field(field):
                continue
            checked += 1
            if field.is_required():
                continue
            default = field.get_default(call_default_factory=True)
            with self.subTest(field=f"{model.__name__}.{name}", default=default):
                if default in (None, ""):
                    continue
                self.assertTrue(str(default).startswith(PROJECT_ROOT_ANCHOR + "/"))
                self.assertTrue(field.validate_default, "an anchored default must set validate_default=True")
        self.assertGreaterEqual(checked, 20, "a guard over no path field passes loudest")

    def test_no_default_outside_a_path_field_writes_the_anchor(self) -> None:
        """StrictModel refuses the anchor in a field that is not a path, but only in a value it validates,
        and pydantic does not validate a default: one written there would reach its reader as text."""
        anchored = [
            f"{model.__name__}.{name}" for model, name, field in _fields(AppConfig)
            if not is_config_path_field(field) and not field.is_required()
            and PROJECT_ROOT_ANCHOR in repr(field.get_default(call_default_factory=True))
        ]
        self.assertEqual(anchored, [])


class AMissingFileSaysHowThePathWasReadTests(_TreeOutsideTheRepository):

    def test_the_refusal_names_the_value_as_written_the_resolved_path_and_the_rule(self) -> None:
        self._declare_the_calibration("calibration/eth_realsense_d435.json")
        rig = self._rig()
        with self.assertRaises(RigArtifactMissing) as caught:
            RigCalibration.from_config(_RIG, rig.extrinsics)
        message = str(caught.exception)
        self.assertIn(_posix(self.root / "calibration" / "eth_realsense_d435.json"), message)
        self.assertIn("The config gives it as calibration/eth_realsense_d435.json", message)
        self.assertIn(f"a relative path in a config is read against the config folder, {self.root.as_posix()}",
                      message)
        self.assertNotIn("working directory", message)

    def test_an_anchored_path_names_the_repository(self) -> None:
        note = path_note(resolve_config_path(f"{PROJECT_ROOT_ANCHOR}/assets/nothing.bin", None))
        self.assertIn(f"The config gives it as {PROJECT_ROOT_ANCHOR}/assets/nothing.bin", note)
        self.assertIn(f"is the repository, {repository_root().as_posix()}", note)

    def test_a_path_no_tree_resolved_adds_nothing(self) -> None:
        self.assertEqual(path_note((self.outside / "never-loaded.json").as_posix()), "")


class AModelBuiltInCodeHasNoFolderTests(unittest.TestCase):

    def test_a_relative_path_without_a_tree_is_left_as_written(self) -> None:
        rig = RGBDDeviceRigConfig.model_validate({
            "rig_id": "wrist", "enabled": True, "source": "rgbd", "fps": 30,
            "extrinsics": {"mounting_mode": "eye_to_hand", "artifact_path": "calibration/eth.json"}})
        assert rig.extrinsics is not None
        self.assertEqual(rig.extrinsics.artifact_path, "calibration/eth.json")


class TheShippedTreeReadsWhatItReadBeforeTests(unittest.TestCase):
    """Inside the repository, run from its root, every shipped path names the file it named before."""

    def test_the_shipped_camera_paths_still_name_the_repositorys_calibration_folder(self) -> None:
        reload_config()
        self.addCleanup(reload_config)
        for profile in (None, "eth2", "tiltcam"):
            rigs = load_config(_REPO / "config", profile=profile).camera.cameras.rigs
            for rig in rigs:
                with self.subTest(profile=profile, rig=rig.rig_id):
                    self.assertEqual(rig.calibration_paths.base_dir,
                                     _posix(_REPO / "calibration" / Path(rig.calibration_paths.base_dir).name))
                    if rig.extrinsics is not None:
                        self.assertTrue(rig.extrinsics.artifact_path.startswith(_posix(_REPO / "calibration") + "/"))


def _fields(root: type[BaseModel]) -> list[tuple[type[BaseModel], str, FieldInfo]]:
    """Every field of every model under ``root``, through unions, lists and maps."""
    found: list[tuple[type[BaseModel], str, FieldInfo]] = []
    seen: set[type] = set()
    stack: list[object] = [root]
    while stack:
        current = stack.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            if current in seen:
                continue
            seen.add(current)
            for name, field in current.model_fields.items():
                found.append((current, name, field))
                stack.append(field.annotation)
            continue
        stack.extend(typing.get_args(current))
    return found


if __name__ == "__main__":
    unittest.main()
