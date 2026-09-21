"""Can the config say where a value came from — and does a mistake point at the line that caused it?

The loader merges a chain of profile overlays and returns plain values. That is lossless about VALUES
and total about ORIGIN, which is fine for one file and actively confusing for four: with
``WILLY_PROFILE=sim,ur3e,tiltcam`` nothing could answer "which layer set this?", and the validation
error named the data DIRECTORY — so a typo in one of four merged files left the reader guessing which.
(``loader.py``'s own docstring claimed the file was named. It was not.)

These tests pin the two halves of the fix: an index that remembers file/line/layer, and an error message
that uses it. Nothing here relaxes validation — the same keys are rejected, with more said about why.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.config import ConfigError, load_config, reload_config
from src.config._provenance import Origin, index_origins, nearest_keys
from src.config.loader import set_active_profile

DATA = Path(__file__).resolve().parents[1] / "config"


class _Tree(unittest.TestCase):
    """Each test gets its own copy of the shipped tree so a deliberate typo cannot escape."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "data"
        shutil.copytree(DATA, self.root)
        set_active_profile(None)
        reload_config()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(reload_config)
        self.addCleanup(set_active_profile, None)

    def _break(self, relative: str, old: str, new: str) -> None:
        path = self.root / relative
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text, f"fixture drift: {old!r} not in {relative}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def _load_error(self, profile: str | None = None) -> str:
        set_active_profile(profile)
        reload_config()
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.root)
        return str(ctx.exception)


class OriginIndexTests(_Tree):
    def test_a_base_value_is_traced_to_its_file_and_line(self) -> None:
        origins = index_origins(self.root)
        origin = origins["robot.workspace_limits.z_max"]
        self.assertEqual(origin.file.name, "robot.yaml")
        self.assertEqual(origin.layer, "")
        self.assertGreater(origin.line, 0)
        line = origin.file.read_text(encoding="utf-8").splitlines()[origin.line - 1]
        self.assertIn("z_max", line, "the recorded line must actually contain the key")

    def test_a_layer_overrides_the_base_and_the_index_says_which(self) -> None:
        """The question the merge could not answer. `robot.ur3e.yaml` narrows the workspace box; without
        provenance a reader sees only the merged number and cannot tell it came from the robot layer."""
        origins = index_origins(self.root, ("sim", "ur3e"))
        origin = origins["robot.workspace_limits.x_max"]
        self.assertEqual(origin.file.name, "robot.ur3e.yaml")
        self.assertEqual(origin.layer, "ur3e")

    def test_the_last_layer_in_the_chain_wins(self) -> None:
        """Same precedence as the loader's left-to-right merge -- tracked instead of forgotten."""
        origins = index_origins(self.root, ("sim", "ur3e", "tiltcam"))
        self.assertEqual(origins["robot.sim.cameras.overhead.position_mm"].layer, "tiltcam")

    def test_an_inactive_layer_is_not_indexed(self) -> None:
        """An overlay for a profile that is not in the chain did not contribute the value, so claiming
        it did would be a lie in the most confusing possible place."""
        origins = index_origins(self.root, ("sim",))
        self.assertNotEqual(origins["robot.workspace_limits.x_max"].layer, "ur3e")

    def test_models_files_are_indexed_under_models(self) -> None:
        """models/*.yaml contribute their TOP-LEVEL keys directly, unlike every other section."""
        origins = index_origins(self.root, ("sim",))
        self.assertIn("models.objectdetector.optim.torch_dtype", origins)
        self.assertEqual(origins["models.objectdetector.optim.torch_dtype"].layer, "sim")

    def test_an_unreadable_file_is_skipped_not_raised(self) -> None:
        """Indexing runs while an error is ALREADY being reported; an explainer that throws on the way
        to explaining is worse than one that says less."""
        (self.root / "robot" / "robot.yaml").write_text("{{ not: valid: yaml", encoding="utf-8")
        self.assertIsInstance(index_origins(self.root), dict)  # no raise


class ErrorMessageTests(_Tree):
    def test_it_names_the_overlay_file_not_the_directory(self) -> None:
        """THE measured complaint: the same typo in robot.yaml and robot.sim.yaml used to produce
        byte-identical output, naming only the data directory."""
        self._break("robot/robot.sim.yaml", "mock_mode: false", "mock_mode_typo: false")
        msg = self._load_error("sim")
        self.assertIn("robot.sim.yaml", msg)
        self.assertIn("robot.sim.mock_mode_typo", msg)

    def test_it_names_the_line(self) -> None:
        self._break("robot/robot.sim.yaml", "mock_mode: false", "mock_mode_typo: false")
        msg = self._load_error("sim")
        path = self.root / "robot" / "robot.sim.yaml"
        line = next(i for i, t in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                    if "mock_mode_typo" in t)
        self.assertIn(f"robot.sim.yaml:{line}", msg)

    def test_it_names_the_layer_the_file_belongs_to(self) -> None:
        self._break("robot/robot.ur3e.yaml", 'robot_model: "ur3e"', 'robot_mdl: "ur3e"')
        msg = self._load_error("sim,ur3e")
        self.assertIn("[layer: ur3e]", msg)

    def test_it_reports_the_whole_chain(self) -> None:
        self._break("robot/robot.ur3e.yaml", 'robot_model: "ur3e"', 'robot_mdl: "ur3e"')
        self.assertIn("sim -> ur3e", self._load_error("sim,ur3e"))

    def test_it_suggests_the_key_the_user_meant(self) -> None:
        """`extra='forbid'` turns every typo into a hard stop, and the commonest typo is a near-miss of
        a real key: the reader knows what they meant and only needs the spelling."""
        self._break(
            "robot/robot.ur3e.yaml",
            "    park_joint_positions: [3.14, -1.7, 1.2, -1.2, -1.57, 0.0]",
            "    park_joint_position: [3.14, -1.7, 1.2, -1.2, -1.57, 0.0]",
        )
        self.assertIn("did you mean: park_joint_positions?", self._load_error("sim,ur3e"))

    def test_it_still_rejects_everything_it_rejected_before(self) -> None:
        """Strictly more text, never more tolerance: the point is a better message, not a softer gate."""
        self._break("robot/robot.yaml", "  workspace_limits:", "  workspace_limitz:")
        msg = self._load_error()
        self.assertIn("rejected on purpose", msg)

    def test_a_cross_field_failure_says_it_is_not_in_any_yaml(self) -> None:
        """Some errors have no line to point at -- a default that failed a cross-field rule. Saying so
        is more useful than pointing somewhere arbitrary."""
        self._break("robot/robot.sim.yaml", 'kinematics_model: "ur5e"', 'kinematics_model: "ur10e"')
        msg = self._load_error("sim")
        self.assertTrue(
            "not written in any YAML" in msg or "robot.sim.yaml" in msg,
            f"expected either a location or an explicit 'no location': {msg}",
        )

    def test_a_valid_tree_still_loads(self) -> None:
        set_active_profile("sim,ur3e")
        reload_config()
        self.assertIsNotNone(load_config(self.root).robot)


#: One RealSense, restated whole by a camera layer the way an operator's cell file restates it. The
#: `{extra}` slot is the rig's calibration, which is what the tests below vary.
_ONE_RIG_LAYER = (
    "cameras:\n"
    "  primary_rig_id: rs\n"
    "  rigs:\n"
    "    - rig_id: rs\n"
    "      enabled: true\n"
    "      source: {source}\n"
    "{fps}"
    "      rgbd_backend: realsense\n"
    "      calibration_paths:\n"
    "        base_dir: calibration/rs\n"
    "{extra}"
)
_FPS = "      fps: 30\n"
#: The block kept and the path left empty: the natural way to write "not calibrated yet".
_EMPTY_PATH = (
    "      extrinsics:\n"
    "        mounting_mode: eye_to_hand\n"
    '        artifact_path: ""\n'
)
#: The path key written with no value at all, and no mounting mode.
_PATH_KEY_ONLY = (
    "      extrinsics:\n"
    "        artifact_path:\n"
)


def _rig_layer(*, extra: str = "", fps: bool = True, source: str = "rgbd") -> str:
    return _ONE_RIG_LAYER.format(extra=extra, fps=_FPS if fps else "", source=source)


def _line_of(path: Path, text: str) -> int:
    return next(i for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if line.strip() == text)


class AnErrorInsideACameraRigTests(_Tree):
    """⚠ MEASURED 2026-09-21 ON A REAL CELL: every error inside a camera rig said it was written nowhere.

    A rig is an item of a list and a tagged union. The side-car index spells its keys `rigs[0].fps`,
    and pydantic's location is `rigs.0.rgbd.fps`, a list position as a plain segment and a level for
    the union's tag that no file has. The lookup missed on both, so an operator whose own layer held the
    offending key on line 13 was told "(not written in any YAML: a default or a cross-field rule)", and
    was sent looking for a default.
    """

    def _write(self, **layer: object) -> Path:
        path = self.root / "camera" / "cam.onerig.yaml"
        path.write_text(_rig_layer(**layer), encoding="utf-8")  # type: ignore[arg-type]
        return path

    def _layer(self, **layer: object) -> str:
        self._write(**layer)
        return self._load_error("onerig")

    def test_a_key_in_a_rig_is_traced_to_the_line_that_wrote_it(self) -> None:
        msg = self._layer(extra=_EMPTY_PATH)
        line = _line_of(self.root / "camera" / "cam.onerig.yaml", 'artifact_path: ""')
        self.assertIn(f"cam.onerig.yaml:{line}", msg)
        self.assertNotIn("not written in any YAML", msg)

    def test_it_is_named_as_the_file_spells_it_with_no_union_tag(self) -> None:
        msg = self._layer(extra=_EMPTY_PATH)
        self.assertIn("camera.cameras.rigs[0].extrinsics.artifact_path", msg)
        self.assertNotIn(".rgbd.", msg)

    def test_a_required_key_a_rig_lacks_points_at_the_rig(self) -> None:
        msg = self._layer(fps=False)
        self.assertIn("camera.cameras.rigs[0].fps", msg)
        self.assertIn("(required, and the block at", msg)
        self.assertIn("cam.onerig.yaml", msg)

    def test_a_rule_over_a_whole_rig_names_the_rig_it_is_written_in(self) -> None:
        """A rig-level rule, or a tag no rig type takes, has no key of its own to point at, and a rig is
        a list position the index does not record. It is pointed at where the rig is written."""
        msg = self._layer(source="realsense")
        self.assertIn("(a rule over the block written at", msg)
        self.assertIn("cam.onerig.yaml", msg)
        self.assertNotIn("not written in any YAML", msg)

    def test_an_extrinsics_block_with_no_artifact_says_to_leave_it_out(self) -> None:
        """The natural way to write "not calibrated yet" is refused; the refusal says the way that works."""
        for extra in (_EMPTY_PATH, _PATH_KEY_ONLY):
            with self.subTest(extra=extra):
                msg = self._layer(extra=extra)
                self.assertIn("declares no extrinsics block at all: leave it out", msg)
                self.assertIn("real_cell.calibrate", msg)
                self.assertIn("set artifact_path to the file the sweep writes", msg)

    def test_the_refusal_names_the_mode_the_block_declares(self) -> None:
        """A wrist block swept with the CLI's default mode would write a CAMERA to BASE for a camera that moves."""
        msg = self._layer(extra=(
            "      extrinsics:\n"
            "        mounting_mode: eye_in_hand\n"
            '        artifact_path: ""\n'
            "        shutter_motion_tolerance_mm: 2.0\n"
            "        shutter_motion_tolerance_deg: 1.0\n"))
        self.assertIn("--mode eye_in_hand", msg)
        # ⛔ Measured by review: pasting the sweep's block over a wrist block, or deleting it, threw away
        # tolerances measured on the cell, and the loader then refused the tree until they were typed again.
        self.assertIn("comment it out, keeping the tolerances it holds", msg)
        self.assertNotIn("leave it out", msg)
        self.assertNotIn("paste", msg)

    def test_a_misspelled_artifact_key_is_named_not_called_uncalibrated(self) -> None:
        """⛔ Measured by review: a block-level refusal caught `artifact_pth:` too, and told the operator
        to delete a real calibration. The misspelling has to be the thing reported."""
        msg = self._layer(extra=(
            "      extrinsics:\n"
            "        mounting_mode: eye_to_hand\n"
            "        artifact_pth: calibration/real/eth_rs.json\n"))
        # Both halves side by side: the key that is there and should not be, on its own line, and the
        # key that should be there and is not, pointed at the block that lacks it.
        self.assertIn("camera.cameras.rigs[0].extrinsics.artifact_pth", msg)
        self.assertIn("Extra inputs are not permitted", msg)
        self.assertIn("camera.cameras.rigs[0].extrinsics.artifact_path", msg)
        self.assertIn("(required, and the block at", msg)
        self.assertNotIn("declares no extrinsics block at all", msg)

    def test_a_real_block_named_like_a_sibling_value_keeps_its_name(self) -> None:
        """⛔ Measured by review: `rgbd_backend: realsense` beside a `realsense` block no file writes made a
        guess from the YAML text drop the block, and the key printed was one that does not exist."""
        from src.config.tree import load_tree

        self._write()
        tree = load_tree("onerig", root=self.root).with_values(
            {"camera.cameras.rigs[0].realsense.laser_power_mw": -5})
        self.assertFalse(tree.ok)
        self.assertIn("camera.cameras.rigs[0].realsense.laser_power_mw", str(tree.error))
        self.assertIn("at LoadedTree.with_values", str(tree.error))

    def test_a_rig_list_given_in_memory_prints_no_tag_either(self) -> None:
        from src.config.tree import load_tree

        rig = {"rig_id": "rs", "enabled": True, "source": "rgbd", "fps": 30,
               "calibration_paths": {"base_dir": "calibration/rs"},
               "extrinsics": {"mounting_mode": "eye_to_hand", "artifact_path": ""}}
        tree = load_tree(None, root=self.root).with_values(
            {"camera.cameras.rigs": [rig], "camera.cameras.primary_rig_id": "rs"})
        self.assertIn("camera.cameras.rigs[0].extrinsics.artifact_path", str(tree.error))
        self.assertNotIn(".rgbd.", str(tree.error))
        self.assertIn("at LoadedTree.with_values", str(tree.error))

    def test_no_block_at_all_is_a_rig_that_is_not_calibrated(self) -> None:
        self._write()
        set_active_profile("onerig")
        reload_config()
        self.assertIsNone(load_config(self.root).camera.cameras.rigs[0].extrinsics)


class TheIndexMergesAsTheLoaderMergesTests(_Tree):
    """The index names the line whose value the tree HOLDS, so it has to follow `_deep_merge`'s rules."""

    def _layer(self, relative: str, text: str) -> None:
        (self.root / relative).write_text(text, encoding="utf-8")

    def test_a_layer_that_restates_the_rigs_leaves_none_of_the_base_rigs_indexed(self) -> None:
        self._layer("camera/cam.onerig.yaml", _rig_layer())
        rigs = {key: origin for key, origin in index_origins(self.root, ("onerig",)).items()
                if key.startswith("camera.cameras.rigs[")}
        self.assertTrue(rigs, "the layer's own rig should be indexed")
        self.assertEqual({"cam.onerig.yaml"}, {origin.file.name for origin in rigs.values()})
        # The base's first rig is a webcam pair; none of its keys may survive the replacement.
        self.assertNotIn("camera.cameras.rigs[0].cam_left_id", rigs)
        self.assertFalse(any(key.startswith("camera.cameras.rigs[1]") for key in rigs))

    def test_the_chain_index_forgets_it_too(self) -> None:
        from src.config._provenance import index_chains

        self._layer("camera/cam.onerig.yaml", _rig_layer())
        chains = index_chains(self.root, ("onerig",))
        self.assertNotIn("camera.cameras.rigs[0].cam_left_id", chains)
        self.assertEqual(["cam.onerig.yaml"],
                         [origin.file.name for origin in chains["camera.cameras.rigs[0].fps"]])

    def test_a_list_nobody_restates_keeps_its_record(self) -> None:
        """The control: without an overlay list, the base rigs are still where the base wrote them."""
        rigs = [key for key in index_origins(self.root) if key.startswith("camera.cameras.rigs[")]
        self.assertIn("camera.cameras.rigs[0].cam_left_id", rigs)

    def test_a_reset_forgets_everything_it_replaced(self) -> None:
        """`__null__` replaces the base value like any other scalar, so nothing inside it survives."""
        self._layer("camera/cam.resetrigs.yaml", 'cameras:\n  rigs: "__null__"\n')
        origins = index_origins(self.root, ("resetrigs",))
        self.assertEqual("cam.resetrigs.yaml", origins["camera.cameras.rigs"].file.name)
        self.assertFalse(any(key.startswith("camera.cameras.rigs[") for key in origins))

    def test_a_null_leaf_over_a_written_value_keeps_the_written_line(self) -> None:
        """⛔ Measured by review: the tree kept the base value and the index named the empty line."""
        self._layer("robot/robot.nullz.yaml", "robot:\n  workspace_limits:\n    z_max:\n")
        origin = index_origins(self.root, ("nullz",))["robot.workspace_limits.z_max"]
        self.assertEqual(("robot.yaml", ""), (origin.file.name, origin.layer))

    def test_a_null_the_earlier_layers_lack_is_the_layers_own_value(self) -> None:
        """`robot.sim.yaml` documents it: `assets_root: null` reaches the tree because the base lacks the key."""
        origin = index_origins(self.root, ("sim",))["robot.sim.assets_root"]
        self.assertEqual(("robot.sim.yaml", "sim"), (origin.file.name, origin.layer))

    def test_a_null_inside_a_list_the_layer_writes_is_its_value(self) -> None:
        self._layer("camera/cam.onerig.yaml", _rig_layer(extra="      serial_number: null\n"))
        origin = index_origins(self.root, ("onerig",))["camera.cameras.rigs[0].serial_number"]
        self.assertEqual("cam.onerig.yaml", origin.file.name)

    def test_two_layers_are_folded_before_they_meet_the_base(self) -> None:
        """⛔ Measured by review: the loader merges base with (l1 with l2), and applying the layers one after
        another lost every base key of a block l1 replaced and l2 wrote as a mapping again."""
        key = "camera.cameras.stereo_calibration.charuco_squares_x"
        self.assertEqual("cam.yaml", index_origins(self.root)[key].file.name, "fixture drift")
        self._layer("camera/cam.l1.yaml", 'cameras:\n  stereo_calibration: "__null__"\n')
        self._layer("camera/cam.l2.yaml", "cameras:\n  stereo_calibration:\n    frame_size: [640, 480]\n")
        set_active_profile("l1,l2")
        reload_config()
        held = load_config(self.root).camera.cameras.stereo_calibration
        assert held is not None
        self.assertEqual(tuple(held.frame_size), (640, 480))  # the tree holds the base block, refined by l2
        origins = index_origins(self.root, ("l1", "l2"))
        self.assertEqual(("cam.yaml", ""), (origins[key].file.name, origins[key].layer))
        self.assertEqual("cam.l2.yaml", origins["camera.cameras.stereo_calibration.frame_size"].file.name)

    def test_a_null_over_a_block_an_earlier_layer_replaced_keeps_the_base_line(self) -> None:
        self._layer("camera/cam.l1.yaml", "cameras:\n  stereo_calibration: [1, 2]\n")
        self._layer("camera/cam.l2.yaml", "cameras:\n  stereo_calibration:\n    charuco_squares_x:\n")
        origins = index_origins(self.root, ("l1", "l2"))
        self.assertEqual("cam.yaml", origins["camera.cameras.stereo_calibration.charuco_squares_x"].file.name)
        self.assertIn("camera.cameras.stereo_calibration.frame_size", origins)

    def test_a_layer_that_replaces_a_whole_section_is_the_line_named(self) -> None:
        """⛔ Measured by review: the section's base keys stayed indexed, and the error named the base file."""
        self._layer("camera/stereomatcher.l1.yaml", 'stereomatcher: "__null__"\n')
        origins = index_origins(self.root, ("l1",))
        self.assertEqual("stereomatcher.l1.yaml", origins["camera.stereomatcher"].file.name)
        self.assertFalse(any(key.startswith("camera.stereomatcher.") for key in origins))
        self.assertIn("stereomatcher.l1.yaml:1", self._load_error("l1"))

    def test_a_key_written_twice_is_the_last_write(self) -> None:
        """As `yaml.safe_load` reads it: the second value is the one the tree holds."""
        self._layer("robot/robot.twice.yaml", "robot:\n  workspace_limits:\n    z_max: 1.0\n    z_max: 2.0\n")
        self.assertEqual(4, index_origins(self.root, ("twice",))["robot.workspace_limits.z_max"].line)


class TheIndexMatchesTheLoaderOnRandomTreesTests(unittest.TestCase):
    """⛔ Measured by review with trees like these: 245 keys lost over 3000 two-layer trees.

    Each tree is merged by the loader's own `_load_yaml_with_profile` and indexed, and the two must hold
    the same keys, each named at a line that wrote the value the tree holds.
    """

    def test_random_trees_of_a_base_and_two_layers(self) -> None:
        import itertools
        import random

        import yaml

        from src.config._provenance import _flatten, _merged_files, read_value
        from src.config.loader import _load_yaml_with_profile

        rng = random.Random(20260921)
        written = itertools.count(1)  # every scalar unique, so a line can be matched to the value it wrote

        def mapping(depth: int, overlay: bool) -> dict:
            return {name: value(depth, overlay) for name in rng.sample("abcd", rng.randint(1, 3))}

        def value(depth: int, overlay: bool) -> object:
            kinds = ["int", "null"] + (["list", "map"] if depth < 3 else []) + (["reset"] if overlay else [])
            kind = rng.choice(kinds)
            if kind == "int":
                return next(written)
            if kind == "reset":
                return "__null__"
            if kind == "list":
                return [next(written) if rng.random() < 0.5 else mapping(depth + 1, overlay)
                        for _ in range(rng.randint(0, 2))]
            return mapping(depth + 1, overlay) if kind == "map" else None

        def held_keys(held: object, prefix: str, out: dict) -> None:
            if isinstance(held, dict):
                for name, item in held.items():
                    dotted = f"{prefix}.{name}" if prefix else name
                    out[dotted] = item
                    held_keys(item, dotted, out)
            elif isinstance(held, list):
                for i, item in enumerate(held):
                    held_keys(item, f"{prefix}[{i}]", out)

        with tempfile.TemporaryDirectory() as tmp:
            for trial in range(300):
                base = Path(tmp) / f"t{trial}.yaml"
                overlays = [(base.with_name(f"t{trial}.{layer}.yaml"), layer) for layer in ("l1", "l2")]
                base.write_text(yaml.safe_dump({"s": mapping(0, False)}, sort_keys=False), encoding="utf-8")
                for path, _ in overlays:
                    if rng.random() < 0.85:
                        path.write_text(yaml.safe_dump({"s": mapping(0, True)}, sort_keys=False), encoding="utf-8")
                held: dict = {}
                held_keys(_load_yaml_with_profile(base, "l1,l2", required=True), "", held)
                chains: dict = {}
                _flatten(_merged_files(base, overlays), "", chains)
                self.assertEqual(sorted(held), sorted(chains), f"trial {trial}")
                for key, item in held.items():
                    if isinstance(item, (dict, list)):
                        continue
                    line = read_value(chains[key][-1]).strip("'\"")
                    if item is None:
                        self.assertIn(line, ("", "null", "__null__"), f"trial {trial}: {key}")
                    else:
                        self.assertEqual(str(item), line, f"trial {trial}: {key}")


class TheLocationIsTranslatedNotGuessedTests(unittest.TestCase):
    """A union level is dropped where the SCHEMA puts one, and nowhere else."""

    def test_a_list_position_becomes_brackets(self) -> None:
        from src.config.loader import _bracketed

        self.assertEqual("camera.cameras.rigs[0].fps", _bracketed(["camera", "cameras", "rigs", 0, "fps"]))

    def test_an_int_map_key_keeps_the_dot_the_file_writes(self) -> None:
        """A YAML `0:` under a mapping is a key, not a position, and the index spells it with a dot."""
        from src.config.loader import _bracketed

        origins = {"robot.grasping.fusion.cameras.0.enabled": Origin(Path("/d/robot.yaml"), 6, "x", "enabled")}
        self.assertEqual("robot.grasping.fusion.cameras.0.enabled",
                         _bracketed(["robot", "grasping", "fusion", "cameras", 0, "enabled"], origins))

    def test_the_rig_union_is_read_off_the_schema(self) -> None:
        from src.config.loader import _tag_sites

        self.assertIn((("camera", "cameras", "rigs", "*"), frozenset({"webcam_pair", "single_device", "rgbd"})),
                      _tag_sites())

    def test_the_tag_is_dropped_at_its_union(self) -> None:
        from src.config.loader import _yaml_segments

        self.assertEqual(["camera", "cameras", "rigs", 0, "extrinsics", "artifact_path"],
                         _yaml_segments(("camera", "cameras", "rigs", 0, "rgbd", "extrinsics", "artifact_path"), {}))

    def test_a_field_that_shares_a_name_with_a_value_keeps_it(self) -> None:
        from src.config.loader import _yaml_segments

        for loc in (("camera", "cameras", "rigs", 0, "realsense", "laser_power_mw"),
                    ("models", "pipeline", "zero_shot", "backend"),
                    ("camera", "cameras", "rigs", 0, "quality", "warmup_frames")):
            with self.subTest(loc=loc):
                self.assertEqual(list(loc), _yaml_segments(loc, {}))

    def test_one_level_is_dropped_per_union(self) -> None:
        """⛔ Measured by review: a member field named like a tag was dropped with the tag."""
        from src.config.loader import _yaml_segments

        self.assertEqual(["camera", "cameras", "rigs", 0, "rgbd", "fps"],
                         _yaml_segments(("camera", "cameras", "rigs", 0, "rgbd", "rgbd", "fps"), {}))

    def test_a_tag_word_anywhere_else_is_an_ordinary_key(self) -> None:
        from src.config.loader import _yaml_segments

        self.assertEqual(["robot", "rgbd", "fps"], _yaml_segments(("robot", "rgbd", "fps"), {}))


class DidYouMeanTests(unittest.TestCase):
    def test_it_finds_a_near_miss(self) -> None:
        self.assertEqual(nearest_keys("tcp_offset_m", ["tcp_offset_mm", "headless"]), ["tcp_offset_mm"])

    def test_it_stays_quiet_when_nothing_is_close(self) -> None:
        """A wrong suggestion is worse than none: it sends the reader to edit the wrong key."""
        self.assertEqual(nearest_keys("wibble", ["tcp_offset_mm", "headless"]), [])


class LocationRenderingTests(unittest.TestCase):
    def test_paths_are_rendered_relative_to_the_data_root(self) -> None:
        origin = Origin(Path("/data/robot/robot.sim.yaml"), 30, "sim", "tcp_offset_mm")
        self.assertEqual(origin.location(Path("/data")), "robot/robot.sim.yaml:30  [layer: sim]".replace("/", __import__("os").sep))

    def test_a_base_file_carries_no_layer_tag(self) -> None:
        self.assertNotIn("layer", Origin(Path("/d/robot.yaml"), 4, "", "vendor").location())


if __name__ == "__main__":
    unittest.main()
