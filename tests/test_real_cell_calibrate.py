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

from src.config.schema.camera import HandEyeConfig
from src.config.schema.robot import RobotConfig
from src.robot.execution import hand_eye
from src.robot.execution.real_cell import calibrate


class _Rig:
    """The smallest thing `hand_eye._select_rig` reads. Not the schema type on purpose — the RGB-D check is an
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


def _hand_eye() -> HandEyeConfig:
    """A real hand-eye block. Its marker is validated at `--check`, so a Mock's attributes no longer pass for one:
    a dictionary named `<Mock ...>` is the unknown dictionary the check now refuses."""
    return HandEyeConfig.model_validate({"eye_to_hand": {"marker_length_mm": 50.0}})


def _real_rgbd(rig_id: str = "overhead"):
    from tests.test_camera_boundaries import _rgbd_rig

    return _rgbd_rig(rig_id)


class RefusalTests(unittest.TestCase):
    """Everything that can be wrong is refused BY NAME, before anything is powered."""

    def test_an_unknown_rig_LISTS_the_configured_ones(self) -> None:
        """An operator at a cell needs the answer, not the question."""
        cfg = _CameraCfg([_real_rgbd("overhead"), _real_rgbd("wrist")])
        rig, refusal = hand_eye._select_rig(cfg, "front")
        self.assertIsNone(rig)
        message = str(refusal)
        self.assertIn("'front'", message)
        self.assertIn("overhead", message)
        self.assertIn("wrist", message)

    def test_a_NON_RGBD_rig_is_refused_before_it_can_fail_about_aruco(self) -> None:
        """A stereo pair calibrates through the routine's own stereo path. Letting it through here
        would fail much later with an error about markers rather than about the rig."""
        cfg = _CameraCfg([_Rig("stereo_dev0", "webcam_pair")])
        rig, refusal = hand_eye._select_rig(cfg, "stereo_dev0")
        self.assertIsNone(rig)
        self.assertIn("not an RGB-D device", str(refusal))

    def test_a_real_rgbd_rig_is_accepted(self) -> None:
        rig = _real_rgbd("overhead")
        selected, refusal = hand_eye._select_rig(_CameraCfg([rig]), "overhead")
        self.assertIs(selected, rig)
        self.assertIsNone(refusal)

    def test_a_DISABLED_rig_is_refused_by_name(self) -> None:
        """MEASURED 2026-09-10: nothing on this path read `enabled`. `--check` answered "the config
        and the rig are usable" for a rig the cell does not run, and the sweep that follows would
        have moved the arm to 22 poses in front of a camera nobody switched on."""
        rig = _real_rgbd("overhead").model_copy(update={"enabled": False})
        selected, refusal = hand_eye._select_rig(_CameraCfg([rig]), "overhead")
        self.assertIsNone(selected)
        self.assertIn("enabled", str(refusal))

    def test_an_empty_rig_list_cannot_reach_this_runner(self) -> None:
        """⛔ REPLACES A TEST OF A BRANCH NOTHING COULD REACH. `_pick_rig` carried its own
        "camera.cameras.rigs is empty" refusal, and the only caller passes `cfg.camera` out of
        `load_config`, which validates `CameraSystemConfig` first. The guarantee lives THERE, so
        that is where it is pinned: a hand-built object was the only witness the branch ever had."""
        from pydantic import ValidationError

        from src.config.schema.camera import CameraSystemConfig

        with self.assertRaises(ValidationError) as caught:
            CameraSystemConfig(primary_rig_id="overhead", rigs=[])
        self.assertIn("at least one camera rig", str(caught.exception))


class CheckTouchesNothingTests(unittest.TestCase):
    def test_check_returns_before_the_arm_is_ever_built(self) -> None:
        """⚠ THE ONE THAT MATTERS AT A CELL. `--check` must be safe to run with the controller live
        and a person inside the fence. Asserted by making the arm factory EXPLODE: if `--check` still
        returns OK, it provably never reached it."""
        cfg = mock.Mock()
        cfg.robot.vendor = "ur"
        cfg.camera = _CameraCfg([_real_rgbd("overhead")])
        cfg.camera.hand_eye = _hand_eye()

        def explode(*_a, **_k):  # pragma: no cover - it must never be called
            raise AssertionError("--check built the arm")

        with mock.patch.object(calibrate, "_load", return_value=cfg), \
             mock.patch("src.robot.drivers.create_arm", side_effect=explode):
            code = calibrate.main(["--rig", "overhead", "--freedrive", "--check"])
        self.assertEqual(code, calibrate._EXIT_OK)

    def test_the_exploding_arm_factory_IS_reached_without_check(self) -> None:
        """Proves the test above is not vacuous. The same patch, `--check` dropped, a real UR robot
        section: the run must now reach the factory and be refused there. If the factory were not
        reached, the patch would be pointing at nothing and the safety assertion would be decoration.

        The readiness gate runs before the factory, so it is patched, and the call is asserted rather
        than inferred from the exit code: on a host without ur_rtde the gate refuses with the same
        exit code before the factory runs, and this test would pass for the wrong reason."""
        cfg = mock.Mock()
        cfg.robot = RobotConfig.model_validate({"vendor": "ur"})
        cfg.camera = _CameraCfg([_real_rgbd("overhead")])
        cfg.camera.hand_eye = _hand_eye()

        explode = mock.Mock(side_effect=RuntimeError("no controller here"))
        patched_load = mock.patch.object(calibrate, "_load", return_value=cfg)
        patched_gate = mock.patch("src.robot.drivers.doctor.require_arm_vendor_ready")
        patched_arm = mock.patch("src.robot.drivers.create_arm", explode)
        with patched_load, patched_gate, patched_arm:
            code = calibrate.main(["--rig", "overhead", "--freedrive", "--dry-run"])
        self.assertEqual(code, calibrate._EXIT_CONFIG)
        explode.assert_called_once()

    def test_check_REFUSES_a_disabled_rig_instead_of_calling_it_usable(self) -> None:
        """The sentence `--check` prints is what an operator acts on. Calling a switched-off rig
        "usable" sends them to the fence for a sweep that cannot see anything."""
        cfg = mock.Mock()
        cfg.robot.vendor = "ur"
        cfg.camera = _CameraCfg([_real_rgbd("overhead").model_copy(update={"enabled": False})])
        cfg.camera.hand_eye = _hand_eye()
        with mock.patch.object(calibrate, "_load", return_value=cfg):
            code = calibrate.main(["--rig", "overhead", "--freedrive", "--check"])
        self.assertEqual(code, calibrate._EXIT_CONFIG)

    def test_an_unknown_rig_exits_CONFIG_not_ERROR(self) -> None:
        """The exit code is what a bring-up script branches on: 1 is 'fix your config', 3 is 'this
        broke'. Conflating them turns a typo into an incident."""
        cfg = mock.Mock()
        cfg.camera = _CameraCfg([_real_rgbd("overhead")])
        with mock.patch.object(calibrate, "_load", return_value=cfg):
            self.assertEqual(calibrate.main(["--rig", "nope", "--freedrive", "--check"]),
                             calibrate._EXIT_CONFIG)


class ThePrintedYamlActuallyValidatesTests(unittest.TestCase):
    """The operator-facing guarantee. Writing the artifact is only half the job: until the camera's rig
    declares it in `camera.cameras.rigs[<id>].extrinsics`, the cell has no CAMERA to BASE transform for
    that camera. So the runner prints the rig block to paste, and a block that does not parse, or parses
    into something the schema rejects, would send an operator hunting for a typo in the shipped snippet."""

    def _block(self, mode: str = "eye_to_hand") -> dict:
        text = hand_eye._rig_block("overhead", mode, "calibration/real/eth_overhead.json")
        rigs = yaml.safe_load(text)["camera"]["cameras"]["rigs"]
        self.assertEqual(len(rigs), 1)
        return rigs[0]

    def test_the_snippet_is_the_rig_block(self) -> None:
        block = self._block()
        self.assertEqual(block["rig_id"], "overhead")
        self.assertEqual(block["extrinsics"], {"mounting_mode": "eye_to_hand",
                                               "artifact_path": "calibration/real/eth_overhead.json"})

    def test_the_fixed_block_VALIDATES_against_RigExtrinsicsConfig(self) -> None:
        """The schema is `extra='forbid'`, so a stray key here would raise, which is exactly the failure
        this catches before an operator meets it."""
        from src.config.schema.camera.shared_schema import RigExtrinsicsConfig

        self.assertEqual(RigExtrinsicsConfig(**self._block()["extrinsics"]).mounting_mode, "eye_to_hand")

    def test_the_wrist_block_names_the_two_tolerances_it_cannot_measure(self) -> None:
        """A wrist camera's shutter motion tolerances are a fact of the cell, so the block names them and
        leaves them for the operator, and the schema refuses the block until they are written."""
        from pydantic import ValidationError

        from src.config.schema.camera.shared_schema import RigExtrinsicsConfig

        text = hand_eye._rig_block("overhead", "eye_in_hand", "calibration/real/eih_overhead.json")
        self.assertIn("shutter_motion_tolerance_mm", text)
        self.assertIn("shutter_motion_tolerance_deg", text)
        extrinsics = self._block("eye_in_hand")["extrinsics"]
        self.assertEqual(extrinsics["mounting_mode"], "eye_in_hand")
        with self.assertRaises(ValidationError):
            RigExtrinsicsConfig(**extrinsics)
        RigExtrinsicsConfig(**extrinsics, shutter_motion_tolerance_mm=2.0, shutter_motion_tolerance_deg=0.5)

    def test_the_block_is_keyed_by_the_RIG_ID(self) -> None:
        """The artifact stamps `rig_id` to the camera id, and the block names the same rig, which is what
        lets the loader find the file for the camera the perception source opens."""
        block = yaml.safe_load(hand_eye._rig_block("wrist_d435", "eye_to_hand", "x.json"))["camera"]["cameras"]["rigs"][0]
        self.assertEqual(block["rig_id"], "wrist_d435")


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

        from src.config.schema.camera.shared_schema import RigExtrinsicsConfig

        schema = typing.get_args(
            RigExtrinsicsConfig.model_fields["mounting_mode"].annotation)
        modes = next(a for a in calibrate.build_parser()._actions   # noqa: SLF001
                     if a.dest == "mode").choices
        self.assertEqual(set(modes), set(schema))

    def test_the_artifact_name_follows_the_sim_runner_convention(self) -> None:
        """`eth_<camera_id>.json` / `eih_<camera_id>.json`. A cell brought up in sim and then on
        hardware must not need two mental models of where its calibration lives."""
        self.assertIn("eth_", hand_eye._rig_block("cam", "eye_to_hand", "d/eth_cam.json"))

    def test_the_exit_codes_are_distinct_and_documented(self) -> None:
        codes = (calibrate._EXIT_OK, calibrate._EXIT_CONFIG,
                 calibrate._EXIT_NO_ARTIFACT, calibrate._EXIT_ERROR)
        self.assertEqual(len(set(codes)), 4)
        self.assertIn("Exit codes", calibrate.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
