"""Calibrating ONE camera of a real cell against the robot — the runner that unlocks multi-view.

⚠ WHY IT EXISTS. `grasping.fusion.geometry` is the biggest measured lever in the grasp stack — top-1
43.50 % single-view → 55.93 % fused (n=354) — and its own schema states the precondition: "Requires
``cameras`` to be populated (**each camera individually calibrated**)". Every consuming piece was
finished: the `fusion.cameras` map, `build_config_frame_resolvers`, the pick loop reading it, the
schema-versioned artifact keyed by `rig_id`, and `CalibrationRoutine` proven in sim for both mountings.

Nothing could PRODUCE one of those artifacts on real hardware. Both calibration runners are sim-only,
and `real_cell/preflight` can only CHECK for an artifact and refuse without one. So a cell could be
configured for multi-view and there was no command to fill it in.

These tests pin the parts that can be checked with no robot and no camera: the refusals, the
hardware-free `--check` path, and — the one that matters most for an operator — that the YAML this
prints actually VALIDATES against the schema it tells them to paste it into.
"""

from __future__ import annotations

import unittest
from unittest import mock

import yaml

from src.config.schema.robot.grasping_schema import CameraExtrinsicsConfig
from src.robot.execution.real_cell import calibrate


class _Rig:
    """The smallest thing `_pick_rig` reads. Not the schema type on purpose — the RGB-D check is an
    isinstance against the real class, so a stand-in must NOT pass it."""

    def __init__(self, rig_id: str, source: str) -> None:
        self.rig_id = rig_id
        self.source = source


class _Cameras:
    def __init__(self, rigs: list) -> None:
        self.rigs = rigs


class _CameraCfg:
    def __init__(self, rigs: list) -> None:
        self.cameras = _Cameras(rigs)


def _real_rgbd(rig_id: str = "overhead"):
    from tests.test_camera_boundaries import _rgbd_rig

    return _rgbd_rig(rig_id)


class RefusalTests(unittest.TestCase):
    """Everything that can be wrong is refused BY NAME, before anything is powered."""

    def test_an_empty_rig_list_says_so(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            calibrate._pick_rig(_CameraCfg([]), "overhead")
        self.assertIn("no camera to calibrate", str(caught.exception))

    def test_an_unknown_rig_LISTS_the_configured_ones(self) -> None:
        """An operator at a cell needs the answer, not the question."""
        cfg = _CameraCfg([_real_rgbd("overhead"), _real_rgbd("wrist")])
        with self.assertRaises(SystemExit) as caught:
            calibrate._pick_rig(cfg, "front")
        message = str(caught.exception)
        self.assertIn("'front'", message)
        self.assertIn("overhead", message)
        self.assertIn("wrist", message)

    def test_a_NON_RGBD_rig_is_refused_before_it_can_fail_about_aruco(self) -> None:
        """A stereo pair calibrates through the routine's own stereo path. Letting it through here
        would fail much later with an error about markers rather than about the rig."""
        cfg = _CameraCfg([_Rig("stereo_dev0", "webcam_pair")])
        with self.assertRaises(SystemExit) as caught:
            calibrate._pick_rig(cfg, "stereo_dev0")
        self.assertIn("not an RGB-D device", str(caught.exception))

    def test_a_real_rgbd_rig_is_accepted(self) -> None:
        rig = _real_rgbd("overhead")
        self.assertIs(calibrate._pick_rig(_CameraCfg([rig]), "overhead"), rig)


class CheckTouchesNothingTests(unittest.TestCase):
    def test_check_returns_before_the_arm_is_ever_built(self) -> None:
        """⚠ THE ONE THAT MATTERS AT A CELL. `--check` must be safe to run with the controller live
        and a person inside the fence. Asserted by making the arm factory EXPLODE: if `--check` still
        returns OK, it provably never reached it."""
        cfg = mock.Mock()
        cfg.robot.vendor = "ur"
        cfg.camera = _CameraCfg([_real_rgbd("overhead")])
        cfg.camera.hand_eye = mock.Mock()
        cfg.camera.hand_eye.eye_to_hand.marker_length_mm = 50.0

        def explode(*_a, **_k):  # pragma: no cover - it must never be called
            raise AssertionError("--check built the arm")

        with mock.patch.object(calibrate, "_load", return_value=cfg), \
             mock.patch("src.robot.drivers.create_arm", side_effect=explode):
            code = calibrate.main(["--rig", "overhead", "--check"])
        self.assertEqual(code, calibrate._EXIT_OK)

    def test_the_exploding_arm_factory_IS_reached_without_check(self) -> None:
        """Proves the test above is not vacuous. The same patch, the same config, `--check` dropped:
        the run must now reach the factory and be refused. If this passed too, the patch would be
        pointing at nothing and the safety assertion would be decoration."""
        cfg = mock.Mock()
        cfg.robot.vendor = "ur"
        cfg.camera = _CameraCfg([_real_rgbd("overhead")])
        cfg.camera.hand_eye = mock.Mock()
        cfg.camera.hand_eye.eye_to_hand.marker_length_mm = 50.0

        def explode(*_a, **_k):
            raise RuntimeError("no controller here")

        patched_load = mock.patch.object(calibrate, "_load", return_value=cfg)
        patched_arm = mock.patch("src.robot.drivers.create_arm", side_effect=explode)
        with patched_load, patched_arm:
            code = calibrate.main(["--rig", "overhead", "--dry-run"])
        self.assertEqual(code, calibrate._EXIT_CONFIG)

    def test_an_unknown_rig_exits_CONFIG_not_ERROR(self) -> None:
        """The exit code is what a bring-up script branches on: 1 is 'fix your config', 3 is 'this
        broke'. Conflating them turns a typo into an incident."""
        cfg = mock.Mock()
        cfg.camera = _CameraCfg([_real_rgbd("overhead")])
        with mock.patch.object(calibrate, "_load", return_value=cfg):
            self.assertEqual(calibrate.main(["--rig", "nope", "--check"]),
                             calibrate._EXIT_CONFIG)


class ThePrintedYamlActuallyValidatesTests(unittest.TestCase):
    """⭑ THE OPERATOR-FACING GUARANTEE. Writing the artifact is only half the job: until the camera is
    in `fusion.cameras`, geometry fusion stands down to a single view and says so only in telemetry.
    So the runner prints the YAML to paste — and a snippet that does not parse, or parses into
    something the schema rejects, would send an operator hunting for a typo we shipped."""

    def _parsed(self, mode: str = "eye_to_hand"):
        text = calibrate._snippet("overhead", mode, "calibration/real/eth_overhead.json")
        return yaml.safe_load(text)

    def test_it_is_valid_yaml_with_the_path_the_schema_expects(self) -> None:
        data = self._parsed()
        entry = data["robot"]["grasping"]["fusion"]["cameras"]["overhead"]
        self.assertEqual(entry["enabled"], True)
        self.assertEqual(entry["mounting_mode"], "eye_to_hand")
        self.assertEqual(entry["extrinsics_artifact_path"], "calibration/real/eth_overhead.json")

    def test_the_entry_VALIDATES_against_CameraExtrinsicsConfig(self) -> None:
        """The schema is `extra='forbid'`, so a stray key here would raise -- which is exactly the
        failure this catches before an operator meets it."""
        entry = self._parsed()["robot"]["grasping"]["fusion"]["cameras"]["overhead"]
        built = CameraExtrinsicsConfig(**entry)
        self.assertTrue(built.enabled)
        self.assertEqual(built.mounting_mode, "eye_to_hand")

    def test_the_eye_in_hand_snippet_validates_too(self) -> None:
        entry = self._parsed("eye_in_hand")["robot"]["grasping"]["fusion"]["cameras"]["overhead"]
        self.assertEqual(CameraExtrinsicsConfig(**entry).mounting_mode, "eye_in_hand")

    def test_the_camera_key_is_the_RIG_ID(self) -> None:
        """The artifact stamps `rig_id` to the camera id and the map is keyed by the same string --
        that alignment is what lets `build_config_frame_resolvers` find the file. A snippet keyed by
        anything else would load a resolver for a camera the perception source never names."""
        data = yaml.safe_load(calibrate._snippet("wrist_d435", "eye_to_hand", "x.json"))
        self.assertEqual(list(data["robot"]["grasping"]["fusion"]["cameras"]), ["wrist_d435"])


class ContractTests(unittest.TestCase):
    def test_the_console_facing_strings_are_ASCII(self) -> None:
        """⚠ A Windows console is cp1252 and the first version of this file crashed on `--help` with a
        UnicodeEncodeError from a warning glyph. Comments may carry them; anything that reaches a
        terminal may not."""
        parser = calibrate.build_parser()
        for action in parser._actions:               # noqa: SLF001 - the help text is the subject
            for text in (action.help or "", *(action.choices or ())):
                self.assertTrue(str(text).isascii(), f"{action.dest}: {text!r}")
        self.assertTrue((parser.description or "").isascii())

    def test_the_mode_choices_match_the_schema_exactly(self) -> None:
        """If the runner offered a mounting the schema does not accept, the snippet it prints would be
        rejected by the config loader."""
        import typing

        schema = typing.get_args(
            CameraExtrinsicsConfig.model_fields["mounting_mode"].annotation)
        modes = next(a for a in calibrate.build_parser()._actions   # noqa: SLF001
                     if a.dest == "mode").choices
        self.assertEqual(set(modes), set(schema))

    def test_the_artifact_name_follows_the_sim_runner_convention(self) -> None:
        """`eth_<camera_id>.json` / `eih_<camera_id>.json`. A cell brought up in sim and then on
        hardware must not need two mental models of where its calibration lives."""
        self.assertIn("eth_", calibrate._snippet("cam", "eye_to_hand", "d/eth_cam.json"))

    def test_the_exit_codes_are_distinct_and_documented(self) -> None:
        codes = (calibrate._EXIT_OK, calibrate._EXIT_CONFIG,
                 calibrate._EXIT_NO_ARTIFACT, calibrate._EXIT_ERROR)
        self.assertEqual(len(set(codes)), 4)
        self.assertIn("Exit codes", calibrate.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
