"""The real-cell runner: the first live caller of ``from_robot_config``, and its desk rehearsal.

Until this runner existed, nothing in the repo performed a config-driven pick on a physical arm --
so the vendor driver, the gripper branch, the safety preflight, the frame resolver and record logging
had never been combined at all. They would have been combined for the first time with a powered robot
in the room.

The rehearsal path is what makes that testable here: a dummy arm and a synthetic scene drive the SAME
code path end to end, in-process, in milliseconds. These tests pin (a) the checklist's verdicts, (b)
that the pick path actually completes, and (c) the ordering contract that only bites on real hardware.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config.schema.robot import RobotConfig
from src.robot.execution.real_cell import CheckStatus, run_config_preflight
from src.robot.execution.real_cell.__main__ import main
from src.robot.execution.autonomous_grasp.rehearsal import (
    RehearsalPerceptionSource,
    rehearsal_intrinsics,
)

_GOOD_TOOL = {
    "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
    "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
}


def _named(report, name):
    return next(c for c in report.checks if c.name == name)


class PreflightTests(unittest.TestCase):
    def test_the_shipped_config_as_a_real_cell_blocks_on_the_three_known_traps(self) -> None:
        """This is the whole point of the runner: the DEFAULT state of a freshly configured real cell
        is not runnable, and each reason is one an operator would otherwise meet as a separate crash."""
        report = run_config_preflight(RobotConfig(vendor="ur"))
        blocking = {c.name for c in report.blocking}
        self.assertEqual(blocking, {"tool frame", "payload", "camera -> base"})
        self.assertFalse(report.ok)

    def test_every_blocking_check_states_a_concrete_fix(self) -> None:
        """A checklist that says "wrong" without saying "do this" just moves the guessing."""
        for c in run_config_preflight(RobotConfig(vendor="ur")).blocking:
            with self.subTest(check=c.name):
                self.assertTrue(c.fix.strip(), f"{c.name} blocks without naming a fix")

    def test_a_fully_configured_real_cell_passes(self) -> None:
        cfg = RobotConfig.model_validate({
            "vendor": "ur",
            # ur.model and self_collision.kinematics_model must agree -- a cross-field validator
            # refuses the mismatch, because a UR3e driven against UR5e link lengths computes every
            # self-collision verdict against the wrong arm.
            "ur": {"model": "ur3e"},
            "gripper": {"tool_frame": _GOOD_TOOL},
            "safety": {"payload": {"enforce": True, "mass_kg": 1.1, "cog_mm": (0.0, 0.0, 55.0)},
                       "self_collision": {"kinematics_model": "ur3e"}},
            "grasping": {"fusion": {"enabled": True, "extrinsics_artifact_path": "logs/eth.json"}},
        })
        report = run_config_preflight(cfg)
        self.assertTrue(report.ok, report.render())

    def test_sim_and_dummy_are_never_blocked(self) -> None:
        """A rehearsal must not be gated on facts that only a physical cell has."""
        for vendor in ("sim", "dummy"):
            with self.subTest(vendor=vendor):
                self.assertTrue(run_config_preflight(RobotConfig(vendor=vendor)).ok)

    def test_the_bench_items_are_reported_but_never_block(self) -> None:
        """Remote Control, brakes, the URCap: no API reports these, so the checklist names them as
        human steps rather than pretending to verify them."""
        report = run_config_preflight(RobotConfig(vendor="ur"))
        bench = [c for c in report.checks if c.status is CheckStatus.BENCH]
        self.assertTrue(bench)
        self.assertTrue(all(c not in report.blocking for c in bench))
        self.assertIn("URCap", " ".join(c.detail for c in bench))

    def test_a_declared_but_zero_cog_is_caught(self) -> None:
        cfg = RobotConfig.model_validate({
            "vendor": "ur", "safety": {"payload": {"enforce": True, "mass_kg": 2.2}},
        })
        self.assertIs(_named(run_config_preflight(cfg), "payload").status, CheckStatus.BLOCK)

    def test_the_report_renders_every_check(self) -> None:
        report = run_config_preflight(RobotConfig(vendor="ur"))
        text = report.render()
        for c in report.checks:
            self.assertIn(c.name, text)
        self.assertIn("blocking", text)


class RehearsalSceneTests(unittest.TestCase):
    def test_the_object_fits_the_gripper_it_will_be_grasped_with(self) -> None:
        """A rehearsal whose object cannot be grasped only ever exercises the failure path. The first
        version of this scene emitted a 156 mm box for an 85 mm gripper and every run returned
        no_valid_grasp -- proving the wiring but never the success path."""
        src = RehearsalPerceptionSource()
        fx = float(rehearsal_intrinsics()[0, 0])
        width_mm = 2 * src.box_px * src.plane_mm / fx
        self.assertLess(width_mm, RobotConfig().gripper.max_width_mm)

    def test_the_frame_is_well_formed_and_not_cached(self) -> None:
        src = RehearsalPerceptionSource()
        a, b = src.acquire(), src.acquire()
        self.assertEqual(a.depth_map.shape, a.segmentations[0].mask.shape)
        self.assertTrue(a.segmentations[0].mask.any())
        self.assertGreater(src.frames_served, 1)
        self.assertNotEqual(a.timestamp, b.timestamp)

    def test_the_object_stands_toward_the_camera(self) -> None:
        """Depth INSIDE the mask must be nearer than the background plane, or the grasp point lands
        behind the table."""
        frame = RehearsalPerceptionSource().acquire()
        mask = frame.segmentations[0].mask
        self.assertLess(float(frame.depth_map[mask].mean()), float(frame.depth_map[~mask].mean()))


class RunnerEndToEndTests(unittest.TestCase):
    """The CLI, in-process. Same path a real cell takes, minus the hardware."""

    def test_rehearsal_completes_the_whole_pick_path(self) -> None:
        self.assertEqual(main(["--rehearse", "--runs", "3"]), 0)

    def test_check_alone_refuses_the_shipped_real_config(self) -> None:
        """--check touches nothing and still tells the operator the cell is not runnable."""
        self.assertEqual(main(["--check"]), 1)

    def test_dry_run_builds_without_moving(self) -> None:
        """The rehearsal build is proven separately from the pick, so a build regression is not
        hidden behind a grasp failure."""
        self.assertEqual(main(["--rehearse", "--dry-run"]), 0)

    def test_the_arm_is_connected_strictly_before_the_gripper(self) -> None:
        """A LIFECYCLE CONTRACT, not a style choice: VacuumGripper.connect() drives digital I/O
        immediately, and from_robot_config deliberately never connects the arm (the caller owns the
        lifecycle). Reversed, the first real suction cell raises inside connect() with the vacuum line
        already commanded."""
        order: list[str] = []
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.grippers.null import NullGripper

        with patch.object(DummyRobotArm, "connect", autospec=True,
                          side_effect=lambda self: order.append("arm")), \
             patch.object(NullGripper, "connect", autospec=True,
                          side_effect=lambda self: order.append("gripper")):
            main(["--rehearse", "--runs", "1"])
        self.assertEqual(order[:2], ["arm", "gripper"])

    def test_a_gripper_that_refuses_does_not_leave_the_arm_connected(self) -> None:
        """Connect is a TRANSACTION: the cell comes up whole, or it does not come up.

        It did not. This branch returns before the pick block's ``finally``, so a gripper refusal left
        the arm connected -- holding the UR controller's one control script, with the payload already
        pushed -- until the process exited. A one-shot CLI survives that; the operator console shares
        this code path and cannot, so the rollback lives here where both get it.
        """
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.grippers.null import NullGripper

        order: list[str] = []
        with patch.object(DummyRobotArm, "connect", autospec=True,
                          side_effect=lambda self: order.append("arm.connect")), \
             patch.object(DummyRobotArm, "disconnect", autospec=True,
                          side_effect=lambda self: order.append("arm.disconnect")), \
             patch.object(NullGripper, "connect", autospec=True,
                          side_effect=RuntimeError("URCap socket 63352 refused")):
            code = main(["--rehearse", "--runs", "1"])

        self.assertEqual(code, 1, "a refused gripper is a config-class failure, not a pick failure")
        self.assertEqual(order, ["arm.connect", "arm.disconnect"])

    def test_a_rehearsal_takes_no_cell_lock(self) -> None:
        """Two rehearsals must be able to run at once: a dummy cell owns no controller.

        Asserted at the source -- ``cell_lock_key`` returns None for a vendor with no real controller
        address -- rather than by racing two processes, so the test cannot pass for the wrong reason
        (e.g. because the lock file happened to be free).
        """
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.cell_lock import cell_lock_key

        for vendor in ("dummy", "sim"):
            with self.subTest(vendor=vendor):
                self.assertIsNone(cell_lock_key(RobotConfig.model_validate({"vendor": vendor})))

    def test_a_real_ur_cell_locks_on_its_controller_address(self) -> None:
        """The key is the RESOURCE, not the process or the checkout -- two clients, one controller."""
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.cell_lock import cell_lock_key

        config = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "192.168.1.100"}})
        self.assertEqual(cell_lock_key(config), "ur@192.168.1.100")
        # Vendor-scoped, so a KUKA and a UR that happen to share an address are still two cells.
        kuka = RobotConfig.model_validate(
            {"vendor": "kuka", "kuka": {"controller_ip": "192.168.1.100"}}
        )
        self.assertEqual(cell_lock_key(kuka), "kuka@192.168.1.100")
        self.assertNotEqual(cell_lock_key(config), cell_lock_key(kuka))


if __name__ == "__main__":
    unittest.main()


class GraspVerificationTests(unittest.TestCase):
    """The only verifier a jaw cell can use, and the two defects that made it wrong in both directions.

    MEASURED 2026-08-09, capability first: the Robotiq driver exposes NO ``is_object_detected``, so
    ``ObjectDetectingGripperVerifier`` returns INCONCLUSIVE forever on the September jaw cell.
    ``VacuumGripper`` does expose it, and has no jaws for a width check. The two verifiers are therefore
    not competitors to be raced -- the end-effector decides which one is even feasible.

    That leaves one real question: does ``width_delta`` discriminate hold from no-hold? At the shipped
    defaults it did not, in both directions at once.
    """

    @staticmethod
    def _verify(post_mm: float, commanded_mm: float = 45.0):
        from unittest.mock import MagicMock as _MM

        from src.robot.grasping.closed_loop.verification import (
            GraspVerificationContext,
            WidthDeltaGripperVerifier,
        )
        from src.robot.grippers.robotiq import GripperController

        cfg = RobotConfig()
        gripper = GripperController(cfg.gripper, ip="127.0.0.1", driver_factory=lambda: _MM())
        return WidthDeltaGripperVerifier().verify(GraspVerificationContext(
            grasp=None, policy=cfg.grasping.verification, gripper=gripper,
            post_close_width_mm=post_mm, commanded_close_width_mm=commanded_mm,
        ))

    def test_a_thin_held_part_is_not_reported_as_an_empty_grasp(self) -> None:
        """The verifier asks "did the jaws collapse on nothing?" -- a question about the MECHANISM.
        Reading ``min_width_mm`` (5.0, a POLICY floor) put the threshold at 7 mm, so a genuinely held
        6 mm part FAILED. It now reads ``closed_width_mm`` (0.0, the 2F-85 shut on nothing)."""
        for post in (46.0, 20.0, 6.0, 3.0):
            with self.subTest(post_close_width_mm=post):
                self.assertEqual(str(self._verify(post).outcome), "passed")

    def test_collapsed_jaws_still_fail(self) -> None:
        """The fix must not blunt the case the verifier was already getting right."""
        for post in (0.0, 1.0):
            with self.subTest(post_close_width_mm=post):
                r = self._verify(post)
                self.assertEqual(str(r.outcome), "failed")
                self.assertEqual(r.reason, "jaws_collapsed_to_minimum")

    def test_a_gripper_that_never_closed_now_fails(self) -> None:
        """``width_delta_max_mm`` was None, so the verifier could not see the case it exists for: a
        close that never executed leaves the jaws at their pre-open width (~80 mm) and that was
        reported PASSED -- the cell carries air to the drop-off and logs a success."""
        for post in (80.0, 58.0):
            with self.subTest(post_close_width_mm=post):
                r = self._verify(post)
                self.assertEqual(str(r.outcome), "failed")
                self.assertEqual(r.reason, "jaws_did_not_close_enough")

    def test_the_two_width_concepts_stay_distinct(self) -> None:
        """A regression guard on the confusion itself: policy floor and mechanism must not collapse
        back into one number."""
        from unittest.mock import MagicMock as _MM

        from src.robot.grippers.robotiq import GripperController

        cfg = RobotConfig()
        g = GripperController(cfg.gripper, ip="127.0.0.1", driver_factory=lambda: _MM())
        self.assertEqual(g.closed_width_mm, 0.0)   # the mechanism
        self.assertEqual(g.min_width_mm, 5.0)      # the policy floor
        self.assertNotEqual(g.closed_width_mm, g.min_width_mm)

    def test_object_detection_is_not_available_on_the_jaw_cell(self) -> None:
        """Pins WHY width_delta is the jaw cell's only option, so a future reader does not "fix" the
        default by switching to a verifier that can only ever say INCONCLUSIVE here.

        The physical 2F-85 does report an object-detection status; this driver does not surface it, and
        whether the pinned SDK exposes it was unverifiable on this box until 2026-08-17 -- it is now
        installed, and the symbol is present."""
        from src.robot.grippers.robotiq import GripperController
        from src.robot.grippers.vacuum import VacuumGripper

        self.assertFalse(hasattr(GripperController, "is_object_detected"))
        self.assertTrue(hasattr(VacuumGripper, "is_object_detected"))


class WalkthroughExampleTests(unittest.TestCase):
    """The teaching examples must keep RUNNING, or they decay into prose that used to be true.

    They are the artifacts that read the real config, build the real objects and print real numbers,
    so a schema rename or a moved accessor breaks them silently the moment nobody runs them. These
    tests are cheap insurance against exactly that.

    ⛔ AND THE INSURANCE WAS WORTH BUYING. `examples/` carried no lint, no type check and no test but
    these two, and an inventory found its `01_hello_pick` claiming the run "clears all six safety
    guards" while NO GUARD RAN, and explaining a frame-contract refusal as "the Dummy arm has no
    kinematics". `examples/real_hardware/first_pick_walkthrough.py`, which these two tests used to
    exercise, read `robot.fixtures` — an attribute `RobotConfig` does not have.

    They now point at `scripts/examples/`, which IS in ruff and mypy in CI.
    """

    @staticmethod
    def _example(name: str):
        """Import an example by file name. They are scripts, not a package."""
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_rehearsal_pick_runs_end_to_end(self) -> None:
        self.assertEqual(self._example("03_pick").main([]), 0)

    def test_robot_setup_reads_the_real_config(self) -> None:
        """It reports the config preflight's verdict. A non-zero exit here means a CHECK came out
        wrong, which is a finding about the config -- not about the example."""
        self.assertIn(self._example("01_robot_setup").main([]), (0, 1))

    def test_it_runs_under_a_profile_chain(self) -> None:
        """Profile layering is one of the things these teach, so it has to survive being layered."""
        self.assertIn(self._example("01_robot_setup").main(["--profile", "ur3e"]), (0, 1))

    def test_every_example_answers_help(self) -> None:
        """A `--help` that raises is an example nobody can start. Cheapest possible smoke test, and
        it covers the four that need no config at all."""
        import contextlib
        import io

        for name in ("01_robot_setup", "02_calibration", "03_pick", "04_datagen",
                     "05_train", "06_full_pipeline", "07_sim"):
            with self.subTest(example=name), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    self._example(name).main(["--help"])
                self.assertEqual(caught.exception.code, 0)

    def test_the_two_frames_describe_the_same_physical_point(self) -> None:
        """Station 4's whole lesson. If this stops holding, the example is teaching a falsehood."""
        import numpy as np

        from src.geometry import Frame, Transform
        from src.robot.execution.autonomous_grasp.rehearsal import RehearsalPerceptionSource

        frame = RehearsalPerceptionSource().acquire()
        mask = frame.segmentations[0].mask
        ys, xs = np.nonzero(mask)
        k = frame.intrinsics
        z = float(frame.depth_map[mask].min())
        cam = np.array([(xs.mean() - k[0, 2]) / k[0, 0] * z, (ys.mean() - k[1, 2]) / k[1, 1] * z, z])
        t = Transform(translation_mm=np.array([400.0, 0.0, 800.0]),
                      quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                      from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        base = t.apply_point(cam)
        # 800 mm camera height minus a 750 mm range to the box top = a 50 mm object top in BASE.
        self.assertAlmostEqual(float(base[2]), 50.0, places=1)
        self.assertAlmostEqual(float(base[0]), 400.0, delta=2.0)


class CuroboProbeHonestyTests(unittest.TestCase):
    """The motion-stack banner must not claim more than it checked.

    MEASURED 2026-08-09: `probe_curobo` reports `available` from `Path(python).exists()` alone, and
    names a `robot_config` it never verifies. The sidecar resolves that name against
    `get_content_root()/configs/robot/` INSIDE the cuRobo installation -- a separate environment this
    process cannot introspect without spawning it (globbing the env root for that directory finds
    nothing), and the probe is deliberately spawn-free.

    So a UR3e cell with no `ur3e.yml` -- the September case -- passes the gate and fails inside the
    sidecar. The honest fix is to say so, the same way the arm-model cross-check was left as a human
    checkpoint rather than a check that cannot be written truthfully. `probe_collision_engine`, one
    function down, DOES verify its per-model bundle, which is exactly why the difference has to be
    visible rather than assumed.
    """

    def test_the_banner_says_the_descriptor_is_unverified(self) -> None:
        from src.robot.safety.planning.environment import probe_curobo

        summary = probe_curobo("ur3e.yml").summary
        self.assertIn("ur3e.yml", summary)
        self.assertIn("not verified", summary)

    def test_available_still_reflects_only_the_interpreter(self) -> None:
        """Pinning the semantics, so a future reader does not widen `available` and quietly make the
        banner mean something the code never established."""
        from unittest.mock import patch

        from src.robot.safety.planning import environment as env

        with patch.object(env.Path, "exists", return_value=True):
            self.assertTrue(env.probe_curobo("nonexistent_robot.yml").available)

    def test_the_runbook_carries_the_bench_step(self) -> None:
        """A gap named only in a docstring is a gap nobody at a bench will meet."""
        from pathlib import Path as _P

        text = _P("docs/runbooks/real_cell_first_pick.md").read_text(encoding="utf-8")
        self.assertIn("get_content_root", text)
        self.assertIn("build_ur_config.py", text)


class ExampleEnvironmentTests(unittest.TestCase):
    """`--profile` must not survive the example that used it.

    ⛔ REGRESSION TEST FOR A REAL, MEASURED LEAK. The examples used to do
    `os.environ["WILLY_PROFILE"] = args.profile` and never put it back. Harmless as a script -- the
    process exits -- but these examples are also CALLED: by `06_full_pipeline`, and by the tests
    above. One `--profile ur3e` run left the variable set, and
    `test_safety_guard_conformance.py::test_workspace_guard_accepts_in_box_and_rejects_out_of_box`
    then failed in the suite while passing in isolation, because it was reading a different cell's
    workspace box. The symptom appeared three files away from the cause.

    A function that mutates the process environment and does not restore it is wrong in BOTH
    contexts, so the fix lives in the examples' shared `Example` manager, not in a test fixture.
    """

    @staticmethod
    def _example(name: str):
        return WalkthroughExampleTests._example(name)

    def test_the_profile_does_not_outlive_the_example(self) -> None:
        import contextlib
        import io
        import os

        before = os.environ.get("WILLY_PROFILE")
        with contextlib.redirect_stdout(io.StringIO()):
            self._example("01_robot_setup").main(["--profile", "ur3e"])
        self.assertEqual(os.environ.get("WILLY_PROFILE"), before,
                         "the example leaked WILLY_PROFILE into the process")

    def test_it_is_restored_even_when_a_step_raises(self) -> None:
        """The restoration is in `__exit__`, so it has to survive the exception path too."""
        import contextlib
        import importlib.util
        import io
        import os
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "examples" / "_common.py"
        spec = importlib.util.spec_from_file_location("_common", path)
        assert spec is not None and spec.loader is not None
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)

        before = os.environ.get("WILLY_PROFILE")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            with common.Example("t", "raises", profile="ur3e") as run:
                self.assertEqual(os.environ["WILLY_PROFILE"], "ur3e")
                with run.step("boom"):
                    raise RuntimeError("boom")
        self.assertEqual(os.environ.get("WILLY_PROFILE"), before)
