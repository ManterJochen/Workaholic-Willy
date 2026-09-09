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

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.schema.robot import RobotConfig
from src.robot.execution.real_cell import CheckStatus, run_config_preflight
from src.robot.execution.real_cell.__main__ import main
from src.robot.execution.autonomous_grasp.rehearsal import (
    RehearsalPerceptionSource,
    rehearsal_intrinsics,
)

_ROOT = Path(__file__).resolve().parents[1]

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
        """The base profile is no longer the one that can do it, and that is the repair.

        This asserted ``main(["--rehearse", "--runs", "3"]) == 0`` on the base tree until
        2026-09-09 and passed because the build handed back a ``NullGripper``: ``gripper.vendor:
        robotiq`` cannot be reached from the dummy arm a rehearsal swaps in, so all three picks
        reported SUCCEEDED with nothing on the flange. ``connect_cell`` refuses that cell now (the
        test below), and the free desk rung survives one flag over: ``console_dummy`` is the profile
        whose gripper a dummy arm can carry (``gripper.vendor: dummy`` builds a real
        ``DummyGripper`` that clamps and echoes a width), so connect, three picks and teardown all
        still run here.
        """
        self.assertEqual(main(["--rehearse", "--runs", "3", "--profile", "console_dummy"]), 0)

    def test_a_rehearsal_of_the_shipped_tree_is_refused_at_the_connect(self) -> None:
        """MEASURED: ``--rehearse --runs 3`` printed ``RESULT: 3/3 succeeded`` and exited 0 on a
        build that logged ``Built a NullGripper (no real end-effector)`` one screen earlier. It
        exits 1 at the connect now, before the arm is commanded, and the config-class exit code is
        the honest one: the fix for this cell is a YAML line, not a trip to a bench."""
        self.assertEqual(main(["--rehearse", "--runs", "3"]), 1)

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
        from src.robot.grippers.dummy import DummyGripper

        # And the profile moved with the gripper class. The base tree rehearsal is refused at the
        # connect now (its Robotiq substitutes on a dummy arm), so an order this test can watch
        # needs a cell that is allowed to come up at all. console_dummy is that cell.
        with patch.object(DummyRobotArm, "connect", autospec=True,
                          side_effect=lambda self: order.append("arm")), \
             patch.object(DummyGripper, "connect", autospec=True,
                          side_effect=lambda self: order.append("gripper")):
            main(["--rehearse", "--runs", "1", "--profile", "console_dummy"])
        self.assertEqual(order[:2], ["arm", "gripper"])

    def test_a_gripper_that_refuses_does_not_leave_the_arm_connected(self) -> None:
        """Connect is a TRANSACTION: the cell comes up whole, or it does not come up.

        It did not. This branch returns before the pick block's ``finally``, so a gripper refusal left
        the arm connected -- holding the UR controller's one control script, with the payload already
        pushed -- until the process exited. A one-shot CLI survives that; the operator console shares
        this code path and cannot, so the rollback lives here where both get it.
        """
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.grippers.dummy import DummyGripper

        order: list[str] = []
        with patch.object(DummyRobotArm, "connect", autospec=True,
                          side_effect=lambda self: order.append("arm.connect")), \
             patch.object(DummyRobotArm, "disconnect", autospec=True,
                          side_effect=lambda self: order.append("arm.disconnect")), \
             patch.object(DummyGripper, "connect", autospec=True,
                          side_effect=RuntimeError("URCap socket 63352 refused")):
            code = main(["--rehearse", "--runs", "1", "--profile", "console_dummy"])

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


class CellChecksTests(unittest.TestCase):
    """The checks under `scripts/checks/` must keep RUNNING, and their verdict is data.

    ⛔ THESE REPLACE A CLASS THAT IMPORTED EXAMPLES AND CALLED `main([])`. That worked while the
    examples were tools with an argument surface. They are not any more: an example is straight-line
    code with no `main`, no flags and no exit code worth reading, and the file that had a verdict
    moved to `scripts/checks/cell_bringup.py` and kept it. Every example is executed instead by
    `tests/test_examples_run.py`, which runs the whole directory rather than four files by name.

    ⚠ AND THE OLD LOADER IS WHY THIS WAS RED FOR AN HOUR. It built its path as
    `"scripts" / "examples" / f"{name}.py"`, so the literal path existed nowhere in the source and a
    sweep for dead references after the move could not see it. A path assembled at runtime has no
    edge to a grep, which is the argument for running the files rather than searching for them.

    Run as a subprocess rather than imported. A check reads `sys.argv` and returns an exit code, and
    importing it to call `main` would test a function while the operator runs a program.
    """

    _CHECKS = _ROOT / "scripts" / "checks"

    def _run(self, name: str, *args: str, profile: str | None = None):
        import os
        import subprocess

        env = dict(os.environ)
        if profile is not None:
            # The caller sets the profile, which is the whole point: no check may write to its own
            # environment, and `tests/test_examples_run.py` asserts that none does.
            env["WILLY_PROFILE"] = profile
        return subprocess.run(
            [sys.executable, str(self._CHECKS / f"{name}.py"), *args],
            cwd=_ROOT, capture_output=True, text=True, timeout=300, env=env,
        )

    def test_the_bringup_check_refuses_to_connect_unasked(self) -> None:
        """A check that opens a socket because somebody typed its name is a hazard.

        Exit 2 is "nothing to check here", which is the honest answer for a connection nobody
        authorised, and it is distinct from exit 1, which would claim the cell is wrong.
        """
        result = self._run("cell_bringup")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--live", result.stdout)

    def test_the_safety_check_reads_this_cell_and_returns_a_verdict(self) -> None:
        """Its exit code is a finding about the config, never about the check.

        0 means every wired guard refused a violation of its own family, 1 means one accepted or
        refused for the wrong reason, 2 means this cell has no guard pipeline to interrogate (a
        dummy or sim arm states outright that nothing gates its motion). All three are legitimate
        answers about a tree, which is why this asserts the set and not a value.
        """
        result = self._run("safety_guards")
        self.assertIn(result.returncode, (0, 1, 2), result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_a_check_honours_a_profile_it_did_not_set(self) -> None:
        """Profile layering has to survive being layered, and the layering is the caller's.

        The previous generation took `--profile` and set `WILLY_PROFILE` itself without restoring
        it, and one run leaked into a safety test three files away that then failed in the suite
        while passing alone. No check takes that flag now.
        """
        for name in ("safety_guards", "camera_artifacts"):
            with self.subTest(check=name):
                result = self._run(name, profile="ur3e")
                self.assertIn(result.returncode, (0, 1, 2), result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_a_check_refuses_a_flag_it_does_not_have(self) -> None:
        """No argparse means no `--help`, so an unknown flag has to be answered by hand.

        Silently ignoring it would be worse than the argparse it replaced: an operator who mistypes
        a flag would get a run that looks like the one they asked for and is not.
        """
        result = self._run("cell_bringup", "--nonsense")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

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
