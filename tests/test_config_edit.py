"""The guided config write: does it land in the right file, and does it leave the file intact?

Every test here runs against a COPY of the shipped tree, so a failure cannot damage the real config.

The comment assertions are the ones that matter most and they are not cosmetic. This project keeps the
evidence for a value in the comment above it (*"6/6 up to 6 mm and 0/6 at 10 mm"*), so a writer that
drops comments deletes the reason anyone can trust the number underneath. The obvious implementation --
``yaml.safe_load`` the file, patch the dict, ``safe_dump`` it back -- does exactly that: measured on
``robot.ur3e.yaml`` it turns 191 lines into 81 and 143 comment lines into none. Hence
``test_editor_module_is_the_comment_destroying_one_and_is_not_what_the_console_uses``, which pins that
difference so the cheaper implementation cannot quietly come back.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.config.edit import (
    MISSING,
    WRITABLE,
    WriteRefused,
    set_key,
    set_keys,
    target_file,
    writable,
)
from src.config.loader import active_profile, load_config, reload_config, set_active_profile

_SHIPPED = Path(__file__).resolve().parents[1] / "config"


def _comment_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("#"))


class ConfigEditTests(unittest.TestCase):
    """One scratch tree per test; the loader's global profile is restored afterwards."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        self._previous = active_profile()
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        set_active_profile(self._previous)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def _load(self, profile: str | None) -> object:
        set_active_profile(profile)
        reload_config()
        return load_config(self.tmp)

    # -- where a write lands -----------------------------------------------------------------------

    def test_a_measurement_lands_in_the_layer_being_run_not_the_shared_base(self) -> None:
        """A payload weighed on the UR3e is a fact about the UR3e, not about every cell in the tree."""
        self.assertEqual(
            target_file("robot.safety.payload.mass_kg", self.tmp, ("ur3e",)).name, "robot.ur3e.yaml"
        )
        self.assertEqual(
            target_file("robot.safety.payload.mass_kg", self.tmp, ()).name, "robot.yaml"
        )

        result = set_key("robot.safety.payload.mass_kg", 1.15, root=self.tmp, layers=("ur3e",),
                         profile="ur3e")
        self.assertTrue(result.applied, result.message)
        # The base file is untouched, so the simulator and the UR5e still see the value they had.
        self.assertNotIn("1.15", (self.tmp / "robot" / "robot.yaml").read_text(encoding="utf-8"))
        self.assertEqual(self._load("ur3e").robot.safety.payload.mass_kg, 1.15)
        self.assertEqual(self._load(None).robot.safety.payload.mass_kg, 0.0)

    # -- what the file looks like afterwards --------------------------------------------------------

    def test_the_written_file_keeps_every_comment_and_changes_only_the_intended_lines(self) -> None:
        target = self.tmp / "robot" / "robot.yaml"
        before = target.read_text(encoding="utf-8").splitlines()
        comments_before = _comment_lines(target)

        result = set_keys(
            {
                "robot.safety.payload.mass_kg": 3.42,
                "robot.safety.payload.cog_mm": [0.0, 0.0, 68.5],
            },
            root=self.tmp,
        )
        self.assertTrue(result.applied, result.message)

        after = target.read_text(encoding="utf-8").splitlines()
        self.assertEqual(_comment_lines(target), comments_before)
        self.assertEqual(len(after), len(before))
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        self.assertEqual(len(changed), 2, f"expected 2 changed lines, got {changed}")
        # The column padding this tree uses to align a block is part of the file's readability.
        self.assertIn("cog_mm:      [0.0, 0.0, 68.5]", after[changed[1]])

    def test_a_key_that_no_yaml_sets_yet_is_inserted_under_its_existing_block(self) -> None:
        """The tool-frame transform is written nowhere today, so every guided write is an insert."""
        target = self.tmp / "robot" / "robot.yaml"
        comments_before = _comment_lines(target)

        result = set_keys(
            {
                "robot.gripper.tool_frame.source": "willy",
                "robot.gripper.tool_frame.offset_mm": [0.0, 132.0, 0.0],
                "robot.gripper.tool_frame.rotation_quat_xyzw": [-0.70710678, 0.0, 0.0, 0.70710678],
            },
            root=self.tmp,
        )
        self.assertTrue(result.applied, result.message)
        self.assertEqual(_comment_lines(target), comments_before)

        frame = self._load(None).robot.gripper.tool_frame
        self.assertEqual(frame.source, "willy")
        self.assertEqual(tuple(frame.offset_mm), (0.0, 132.0, 0.0))

    def test_a_serial_is_written_into_the_right_sequence_item(self) -> None:
        """With two identical D435s the serial is the only stable identity, so the INDEX must be exact."""
        result = set_keys(
            {
                "camera.cameras.rigs[0].serial_number": "241222071234",
                "camera.cameras.rigs[1].serial_number": "241222079999",
            },
            root=self.tmp, layers=("tiltcam",), profile="tiltcam",
        )
        self.assertTrue(result.applied, result.message)
        rigs = self._load("tiltcam").camera.cameras.rigs
        self.assertEqual(rigs[0].serial_number, "241222071234")
        self.assertEqual(rigs[1].serial_number, "241222079999")

    # -- what it refuses ---------------------------------------------------------------------------

    def test_a_safety_limit_is_refused_even_though_the_schema_would_accept_it(self) -> None:
        """Not "cannot", but "will not": the value is legal and the refusal is the design."""
        result = set_key("robot.workspace_limits.z_min", -500.0, root=self.tmp)
        self.assertFalse(result.applied)
        self.assertIs(result.refused, WriteRefused.NOT_WRITABLE)
        self.assertIn("comment", result.message)

    def test_an_unknown_key_is_named_as_unknown_rather_than_written(self) -> None:
        result = set_key("robot.safety.payload.mass_kilograms", 1.0, root=self.tmp)
        self.assertFalse(result.applied)
        self.assertIs(result.refused, WriteRefused.UNKNOWN_KEY)

    def test_every_writable_key_resolves_on_a_real_loaded_tree(self) -> None:
        """A typo in the allowlist would produce a form field that can never be saved.

        Checked by navigating the loaded config rather than the schema index, because the index treats
        a list as a leaf: ``camera.cameras.rigs`` is indexed, and no field of a rig ever is. That gap
        is real -- ``explain`` reports ``rigs[0].serial_number`` as an unknown key -- but it is a gap in
        introspecting lists, not evidence that the path is wrong, and the loaded object settles it.
        """
        from src.config.edit import read_key

        cfg = self._load("tiltcam")
        for entry in WRITABLE:
            path = entry.path.replace("[*]", "[0]")
            with self.subTest(path):
                self.assertIsNot(
                    read_key(cfg, path), MISSING,
                    f"{entry.path} is offered as writable but does not exist on a loaded config",
                )
                self.assertTrue(entry.measure.strip(), f"{entry.path} offers no measuring instruction")

    def test_indexed_paths_match_their_template(self) -> None:
        self.assertIsNotNone(writable("camera.cameras.rigs[0].serial_number"))
        self.assertIsNotNone(writable("camera.cameras.rigs[7].serial_number"))
        self.assertIsNone(writable("camera.cameras.rigs[0].fps"))

    # -- the transaction ---------------------------------------------------------------------------

    def test_a_value_the_validators_reject_leaves_the_tree_byte_identical(self) -> None:
        """The whole safety argument for a machine writer: a bad edit cannot survive the write."""
        target = self.tmp / "robot" / "robot.yaml"
        before = target.read_bytes()

        result = set_key("robot.safety.payload.mass_kg", 999.0, root=self.tmp)
        self.assertFalse(result.applied, "a mass above max_mass_kg must not be accepted")
        self.assertIs(result.refused, WriteRefused.INVALID_VALUE)
        self.assertEqual(target.read_bytes(), before)
        # And the loader still works, i.e. the rollback also restored the process's cached tree.
        self.assertEqual(self._load(None).robot.safety.payload.mass_kg, 0.0)

    def test_half_a_tool_frame_is_refused_which_is_why_the_group_is_the_primitive(self) -> None:
        """Declaring an owner while the transform is still identity is rejected by a cross-field rule.

        There is no order in which the three keys validate one at a time -- the first write is always
        half a tool frame. This is the measured reason :func:`set_keys` writes a group and validates
        once, rather than looping over :func:`set_key`.
        """
        target = self.tmp / "robot" / "robot.yaml"
        before = target.read_bytes()

        result = set_key("robot.gripper.tool_frame.source", "willy", root=self.tmp)
        self.assertFalse(result.applied)
        self.assertIs(result.refused, WriteRefused.INVALID_VALUE)
        self.assertIn("identity", result.message)
        self.assertEqual(target.read_bytes(), before)

    def test_one_bad_member_rolls_back_every_file_the_group_touched(self) -> None:
        """Two sections, one failure: neither file may keep its half of the write."""
        robot = self.tmp / "robot" / "robot.yaml"
        camera = self.tmp / "camera" / "cam.yaml"
        before = (robot.read_bytes(), camera.read_bytes())

        result = set_keys(
            {
                "camera.cameras.rigs[0].serial_number": "241222071234",
                "robot.safety.payload.mass_kg": 999.0,  # rejected by max_mass_kg
            },
            root=self.tmp,
        )
        self.assertFalse(result.applied)
        self.assertEqual((robot.read_bytes(), camera.read_bytes()), before)

    def test_the_value_reported_back_is_read_out_of_the_reloaded_tree(self) -> None:
        """Echoing the request would hide coercion; the operator must see what the cell will run with."""
        result = set_key("robot.safety.payload.mass_kg", 2, root=self.tmp)
        self.assertTrue(result.applied, result.message)
        self.assertIsInstance(result.values["robot.safety.payload.mass_kg"], float)

    # -- the implementation that must not come back ---------------------------------------------------

    def test_the_obvious_yaml_round_trip_would_delete_the_evidence_and_this_writer_does_not(self) -> None:
        """Pinned as a measurement, not an opinion.

        The cheap way to write a config value is ``yaml.safe_load`` the file, patch the dict,
        ``safe_dump`` it back. It is shorter than everything in ``src/config/edit.py`` and it is
        what a future contributor will reach for. This measures the two side by side on the same file
        and the same key, so the tradeoff is a number rather than a preference.

        ``backend/config/editor.py`` was exactly that implementation. It was deleted on 2026-08-09
        after a sweep found no caller outside its own test; the lesson it taught is kept here.
        """
        import yaml

        target = self.tmp / "robot" / "robot.ur3e.yaml"
        comments_before = _comment_lines(target)
        self.assertGreater(comments_before, 100, "this fixture must be a heavily commented file")

        # The cheap implementation, in full.
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
        document["robot"]["safety"]["payload"]["mass_kg"] = 1.15
        target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.assertEqual(
            _comment_lines(target), 0,
            "a safe_load/safe_dump round-trip is expected to delete every comment in the file",
        )

        # The guided writer, same file, same key.
        shutil.rmtree(self.tmp)
        shutil.copytree(_SHIPPED, self.tmp)
        result = set_key(
            "robot.safety.payload.mass_kg", 1.15, root=self.tmp, layers=("ur3e",), profile="ur3e"
        )
        self.assertTrue(result.applied, result.message)
        self.assertEqual(_comment_lines(target), comments_before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
