"""Building a whole cell in ONE call — and proving it is the same cell the two calls built.

⚠ WHAT MOVED AND WHY. `from_robot_config` is the composition root, but `calculator` and `perception`
are REQUIRED keyword arguments with no default: config alone cannot describe a camera adapter that
needs a per-run text prompt and two models on a GPU. So building a real cell took TWO calls, and the
second one — `build_real_components` — lived in `real_cell/components.py`, a package named after a CLI.

One construction shared by the runner and the console is better than two, but an example that wants to
show "here is a real cell" still had to import from a runner package to assemble one, and the
walkthrough in `examples/` reached into five modules. `build_real_cell` / `build_rehearsal_cell` put
the whole construction in the package that owns it, and say in the NAME which cell they build.

⛔ WHAT WAS DELIBERATELY NOT DONE: making `calculator` / `perception` optional on `from_robot_config`.
That buys the same single call and pays with a root that opens a CAMERA when a caller forgets an
argument. `test_the_root_still_REQUIRES_its_components` pins that decision so it cannot erode.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import unittest
from unittest import mock

from src.config.schema.robot import RobotConfig
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspService,
    CellBuildRefused,
    build_real_cell,
    build_real_components,
    build_rehearsal_cell,
    build_rehearsal_components,
)


def _cfg() -> RobotConfig:
    """A dummy-arm cell: no vendor SDK, no camera, no models — everything a rehearsal needs.

    ⚠ `grasping` IS DECLARED EXPLICITLY, and it has to be. `from_robot_config` refuses a config whose
    `grasping` block was never written down, even though the schema has a default -- "the schema
    default is intentionally not trusted on production hosts". A bare `RobotConfig()` therefore cannot
    build a cell at all, which is the guard doing its job and not a fixture detail to work around
    silently."""
    return RobotConfig.model_validate({"grasping": {}})


class OneCallEqualsTheOldTwoTests(unittest.TestCase):
    """The whole change is a RE-ROUTING. If the helper drifts from the explicit path, the console and
    the CLI start building something the composition root would not have."""

    def test_the_rehearsal_helper_wires_the_SAME_pieces_as_the_explicit_path(self) -> None:
        """⚠ THE EXPLICIT SIDE SWAPS THE VENDOR TOO, and that is the one deliberate difference this
        move introduced. `build_rehearsal_cell` now forces `vendor="dummy"` itself, because leaving
        that to the caller is what let the console build a REAL arm driver for a "rehearsal". So the
        equivalence being pinned here is against the explicit path ON THE SWAPPED CONFIG -- everything
        except the arm is identical, and the arm is the fix."""
        cfg = _cfg().model_copy(update={"vendor": "dummy"})
        calculator, perception, resolver, _cams = build_rehearsal_components(cfg)
        explicit = AutonomousGraspService.from_robot_config(
            cfg, calculator=calculator, perception=perception, frame_resolver=resolver)
        helper = build_rehearsal_cell(_cfg())

        for service in (explicit, helper):
            self.assertIsInstance(service, AutonomousGraspService)
        left, right = explicit.runtime.orchestrator, helper.runtime.orchestrator
        self.assertEqual(type(left.perception), type(right.perception))
        self.assertEqual(type(left.calculator), type(right.calculator))
        self.assertEqual(type(left.arm), type(right.arm))
        self.assertEqual(type(getattr(left, "gripper", None)),
                         type(getattr(right, "gripper", None)))

    def test_the_effective_config_survives_the_helper(self) -> None:
        """⚠ THE ONE THAT WOULD BITE SILENTLY. `from_components` leaves `effective_config=None`, which
        silences the U8/U9/U10 overlays -- a helper that took that shortcut would build a cell that
        looked identical and ran a different stack."""
        cfg = _cfg()
        helper = build_rehearsal_cell(cfg)
        self.assertIsNotNone(getattr(helper, "effective_config", None),
                             "the helper lost the config-driven overlays")

    def test_overrides_reach_the_composition_root(self) -> None:
        """A caller that needs a mode or a live device handle must not have to abandon the helper."""
        with mock.patch.object(AutonomousGraspService, "from_robot_config") as root:
            build_rehearsal_cell(_cfg(), mode="easy")
        self.assertEqual(root.call_args.kwargs["mode"], "easy")
        for required in ("calculator", "perception", "frame_resolver"):
            self.assertIn(required, root.call_args.kwargs)


class TheRootKeepsItsContractTests(unittest.TestCase):
    def test_the_root_still_REQUIRES_its_components(self) -> None:
        """⛔ THE DECISION THIS PINS. Making these optional would let a forgotten argument open a real
        RealSense from a unit test. The convenience lives in a helper whose NAME says what it costs."""
        parameters = inspect.signature(AutonomousGraspService.from_robot_config).parameters
        for name in ("calculator", "perception"):
            self.assertIs(parameters[name].default, inspect.Parameter.empty,
                          f"{name} grew a default -- the root can now build hardware implicitly")

    def test_the_door_for_own_components_is_still_open(self) -> None:
        """The sim runners, `datagen`'s probes and every test go through here. The helpers are a
        convenience for the config-driven case, never a replacement."""
        cfg = _cfg()
        calculator, perception, resolver, _cams = build_rehearsal_components(cfg)
        service = AutonomousGraspService.from_robot_config(
            cfg, calculator=calculator, perception=perception, frame_resolver=resolver)
        self.assertIsInstance(service, AutonomousGraspService)


class TheHelperInheritsTheFailClosedGuardTests(unittest.TestCase):
    def test_a_config_that_never_declared_grasping_is_REFUSED_through_the_helper_too(self) -> None:
        """The convenience must not become a way around the guard. `from_robot_config` refuses a
        config whose `grasping` block was never written down -- a helper that supplied a default mode
        to be friendly would have let a misconfigured YAML boot a degraded cell quietly."""
        with self.assertRaises(ValueError) as caught:
            build_rehearsal_cell(RobotConfig())
        self.assertIn("grasping", str(caught.exception))

    def test_an_explicit_mode_satisfies_it_through_the_helper(self) -> None:
        """The documented escape hatch still works: a caller that names the mode has opted in."""
        service = build_rehearsal_cell(RobotConfig(), mode="easy")
        self.assertIsInstance(service, AutonomousGraspService)


class ARehearsalIsAlwaysARehearsalTests(unittest.TestCase):
    """⛔ A REAL DEFECT THIS MOVE SURFACED. The CLI set `vendor="dummy"` one line before building;
    the operator CONSOLE, calling the same shared component builder, did not. So "rehearse" meant a
    dummy arm in the terminal and the CONFIGURED VENDOR'S REAL DRIVER in the browser -- which on
    connect reaches for a physical controller. That is exactly the "a browser and a terminal disagree
    about what the cell is" failure the shared builder was written to prevent, and it survived because
    the swap sat OUTSIDE the shared function."""

    def test_a_UR_config_still_yields_a_DUMMY_arm(self) -> None:
        service = build_rehearsal_cell(
            RobotConfig.model_validate({"vendor": "ur", "grasping": {}}))
        self.assertEqual(type(service.runtime.orchestrator.arm).__name__, "DummyRobotArm")

    def test_it_holds_for_every_configured_vendor(self) -> None:
        for vendor in ("ur", "kuka", "dummy"):
            with self.subTest(vendor=vendor):
                service = build_rehearsal_cell(
                    RobotConfig.model_validate({"vendor": vendor, "grasping": {}}))
                self.assertEqual(type(service.runtime.orchestrator.arm).__name__, "DummyRobotArm")

    def test_the_swap_does_not_leak_back_into_the_callers_config(self) -> None:
        """A helper that mutated the caller's config would silently turn a REAL run that happened to
        rehearse first into a dummy run."""
        cfg = RobotConfig.model_validate({"vendor": "ur", "grasping": {}})
        build_rehearsal_cell(cfg)
        self.assertEqual(str(getattr(cfg.vendor, "value", cfg.vendor)), "ur")

    def test_it_is_idempotent_for_the_CLI_that_still_swaps_first(self) -> None:
        """The runner keeps its own swap because its preflight and its banner need the swapped
        config. Swapping twice must be the same as swapping once."""
        already = RobotConfig.model_validate({"vendor": "dummy", "grasping": {}})
        service = build_rehearsal_cell(already)
        self.assertEqual(type(service.runtime.orchestrator.arm).__name__, "DummyRobotArm")

    def test_the_REAL_builder_does_NOT_swap_anything(self) -> None:
        """Otherwise a real cell would quietly become a rehearsal, which is the same defect pointing
        the other way and far worse."""
        cfg = RobotConfig.model_validate({"vendor": "ur", "grasping": {}})
        app_cfg = mock.Mock()
        app_cfg.camera.cameras.rigs = []
        with mock.patch("src.config.load_config", return_value=app_cfg),              self.assertRaises(CellBuildRefused):
            build_real_cell(cfg, prompt="x")
        self.assertEqual(str(getattr(cfg.vendor, "value", cfg.vendor)), "ur")


class NoSecondSourceTests(unittest.TestCase):
    def test_real_cell_no_longer_offers_a_cell_builder(self) -> None:
        """The point of the move: there is ONE place that knows how to build a cell."""
        import src.robot.execution.real_cell as real_cell

        for gone in ("build_real_components", "build_rehearsal_components",
                     "build_real_cell", "RehearsalPerceptionSource"):
            self.assertFalse(hasattr(real_cell, gone), f"real_cell still exposes {gone}")

    def test_the_old_modules_are_gone_rather_than_re_exported(self) -> None:
        """A compatibility shim would leave two importable addresses for one construction, which is
        the confusion this removes rather than relocates."""
        for gone in ("src.robot.execution.real_cell.components",
                     "src.robot.execution.real_cell.rehearsal"):
            with self.assertRaises(ModuleNotFoundError, msg=gone):
                __import__(gone)

    def test_both_builders_are_exported_from_the_service_package(self) -> None:
        import src.robot.execution.autonomous_grasp as package

        for name in ("build_real_cell", "build_rehearsal_cell",
                     "build_real_components", "build_rehearsal_components"):
            self.assertIn(name, package.__all__, name)


class ItStaysTorchFreeTests(unittest.TestCase):
    def test_importing_the_service_package_does_NOT_pull_torch(self) -> None:
        """⚠ A REAL REASON THE SPLIT EXISTED. `build_real_components` imports the models factory. If
        that reached module scope, every test, every sim runner and every rehearsal would pay a 2 GB
        import to build a cell that needs none of it. Checked in a SUBPROCESS, because this process
        has almost certainly imported torch already for another reason."""
        code = (
            "import sys; "
            "import src.robot.execution.autonomous_grasp as p; "
            "p.build_rehearsal_cell; p.build_real_cell; "
            "print('torch' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                check=False, cwd=str(__import__("pathlib").Path(__file__).parents[1]))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip(), "False", "importing the cell builders pulled torch")


class TheRefusalIsCATCHABLETests(unittest.TestCase):
    """⛔⛔ IT USED TO BE `SystemExit`, AND BOTH CALLERS WERE WRITTEN TO CATCH `Exception`.

    `api/routers/cell.py` answers 422 `build_refused` from an `except Exception`, and
    `real_cell/__main__.py` returns its config exit code from another. `SystemExit` is a
    `BaseException`, so neither could fire: the console answered 500 with no message and the bench
    printed a bare traceback, for a refusal an operator was meant to read as "your config says X".

    ⚠ IT WENT UNNOTICED BECAUSE NOTHING COULD REACH ONE. The cell took the first rig with
    `source: rgbd` by list position and there was always such a rig, even switched off. Naming the
    primary camera made these refusals reachable on the base profile and on tiltcam, which would have
    turned a latent 500 into a certain one on every profile there is.
    """

    def test_it_is_an_Exception_so_the_designed_handlers_see_it(self) -> None:
        self.assertTrue(issubclass(CellBuildRefused, Exception))

    def test_the_pattern_both_callers_use_actually_catches_it(self) -> None:
        """Written as the callers write it, rather than as an assertion about a class tree: what
        broke was not the type, it was that `except Exception` walked straight past it."""
        caught = None
        try:
            raise CellBuildRefused("primary_rig_id names a webcam pair")
        except Exception as exc:  # noqa: BLE001 - this IS the caller's line, verbatim
            caught = exc

        self.assertIsInstance(caught, CellBuildRefused)

    def test_it_is_NOT_a_bare_BaseException(self) -> None:
        """The guard against someone reaching for `SystemExit` again for the clean exit code. The CLI
        gets its exit code from its own handler; the refusal must not carry one."""
        self.assertFalse(issubclass(CellBuildRefused, SystemExit))


class RealCellRefusesCleanlyTests(unittest.TestCase):
    """Which camera a cell opens is named by `camera.cameras.primary_rig_id`, and every way of
    getting that wrong is refused before a model is loaded.

    ⛔ IT USED TO BE THE FIRST RIG WITH `source: rgbd`, BY LIST POSITION, and that predicate did not
    consult `enabled` while two others over the same list did. The shipped base profile ships its
    only RGB-D rig switched off, so the cell opened it anyway and the failure surfaced at the device
    instead of at the config that caused it.
    """

    @staticmethod
    def _rig(rig_id: str, *, source: str = "rgbd", enabled: bool = True) -> mock.Mock:
        rig = mock.Mock()
        rig.rig_id = rig_id
        rig.source = source
        rig.enabled = enabled
        return rig

    def _refusal(self, rigs: list, primary: str) -> str:
        app_cfg = mock.Mock()
        app_cfg.camera.cameras.rigs = rigs
        app_cfg.camera.cameras.primary_rig_id = primary
        with mock.patch("src.config.load_config", return_value=app_cfg), \
             self.assertRaises(CellBuildRefused) as caught:
            build_real_components(_cfg(), "an object")
        return str(caught.exception)

    def test_a_primary_that_names_nothing_is_refused_by_name(self) -> None:
        """The refusal an operator meets first, and it must name the fix rather than the symptom."""
        message = self._refusal([], "realsense_d435")

        self.assertIn("primary_rig_id", message)
        self.assertIn("realsense_d435", message)

    def test_a_DISABLED_primary_is_refused_rather_than_opened(self) -> None:
        """The case the old list-position rule got wrong on the shipped base profile."""
        message = self._refusal([self._rig("realsense_d435", enabled=False)], "realsense_d435")

        self.assertIn("enabled: false", message)
        self.assertIn("primary_rig_id", message)

    def test_a_primary_that_is_not_RGBD_NAMES_the_rigs_that_are(self) -> None:
        """⚠ THE LIST, NOT JUST THE DEMAND. "Name the RGB-D rig" asks the operator for the one fact
        they have just shown they do not have, and this function is holding the list while it says
        it. The base profile is exactly this case: its only enabled rig is a webcam pair."""
        message = self._refusal(
            [self._rig("webcam_main", source="webcam_pair"),
             self._rig("realsense_d435", enabled=False)],
            "webcam_main")

        self.assertIn("RGB-D", message)
        self.assertIn("realsense_d435", message)
        self.assertIn("src.robot.perception", message)

    def test_a_profile_with_no_depth_camera_at_all_says_THAT_instead(self) -> None:
        """Pointing at a standalone camera check would be advice nobody can follow: there is no
        camera to check. The two cases get different sentences because they have different fixes."""
        message = self._refusal([self._rig("webcam_main", source="webcam_pair")], "webcam_main")

        self.assertIn("no RGB-D rig", message)
        self.assertNotIn("src.robot.perception", message)

    def test_build_real_cell_propagates_that_refusal(self) -> None:
        app_cfg = mock.Mock()
        app_cfg.camera.cameras.rigs = []
        with mock.patch("src.config.load_config", return_value=app_cfg), \
             self.assertRaises(CellBuildRefused):
            build_real_cell(_cfg(), prompt="an object")


if __name__ == "__main__":
    unittest.main()
