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

DATA = Path(__file__).resolve().parents[1] / "src" / "config" / "data"


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
        self._break("robot/robot.ur3e.yaml", 'gripper_mount: "robotiq_2f85"', 'gripper_mnt: "robotiq_2f85"')
        msg = self._load_error("sim,ur3e")
        self.assertIn("[layer: ur3e]", msg)

    def test_it_reports_the_whole_chain(self) -> None:
        self._break("robot/robot.ur3e.yaml", 'gripper_mount: "robotiq_2f85"', 'gripper_mnt: "robotiq_2f85"')
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
