"""The cell routes every motion through cuRobo + the exact-mesh collision engine -- or it refuses.

Both fallbacks exist and both have been measured to MISLEAD rather than merely to degrade:

  * blind IK proposes self-colliding arm branches that the guard then (correctly) rejects, so the run
    fails for a reason that reads as bad grasping;
  * the capsule proxy's default 60 mm link radius over-rejects legitimate reach-down grasps, and was
    deceived three separate times where the exact-mesh check was not.

So a run without them does not produce a slightly-worse number for the configured system -- it produces
a number for a DIFFERENT system. A warning line was already there and was, in practice, invisible in a
several-thousand-line Isaac boot log. These tests pin the stronger contract: refuse by default, and make
the refusal impossible to miss.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest import mock

from src.willy_sim.harness.bootstrap import (
    ENV_ALLOW_DEGRADED,
    DegradedMotionStackError,
    motion_stack_banner,
    require_motion_stack,
)


@dataclass
class _Status:
    available: bool


@dataclass
class _Env:
    curobo: _Status
    collision: _Status


def _patch_env(*, curobo: bool, collision: bool):
    return mock.patch(
        "src.robot.safety.planning.environment.probe_planning_environment",
        return_value=_Env(curobo=_Status(curobo), collision=_Status(collision)),
    )


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._allow = mock.patch.dict("os.environ", {}, clear=False)
        self._allow.start()
        self.addCleanup(self._allow.stop)
        import os

        os.environ.pop(ENV_ALLOW_DEGRADED, None)

    def test_fully_anchored_cell_passes_silently(self) -> None:
        with _patch_env(curobo=True, collision=True):
            self.assertEqual(
                require_motion_stack("curobo", exact_mesh_collision=True), [],
            )

    def test_missing_curobo_refuses(self) -> None:
        with _patch_env(curobo=False, collision=True), self.assertRaises(DegradedMotionStackError) as ctx:
            require_motion_stack("curobo", exact_mesh_collision=True)
        self.assertIn("cuRobo", str(ctx.exception))

    def test_missing_exact_mesh_engine_refuses(self) -> None:
        with _patch_env(curobo=True, collision=False), self.assertRaises(DegradedMotionStackError) as ctx:
            require_motion_stack("curobo", exact_mesh_collision=True)
        self.assertIn("Coal", str(ctx.exception))

    def test_both_missing_are_reported_together(self) -> None:
        """One boot should tell the operator EVERYTHING to install, not send them round the loop twice."""
        with _patch_env(curobo=False, collision=False), self.assertRaises(DegradedMotionStackError) as ctx:
            require_motion_stack("curobo", exact_mesh_collision=True)
        self.assertIn("cuRobo", str(ctx.exception))
        self.assertIn("Coal", str(ctx.exception))

    def test_a_cell_that_never_asked_for_the_mesh_backend_is_not_held_to_it(self) -> None:
        """``self_collision.backend`` other than "fcl" means the capsule path is the CONFIGURED choice,
        not a silent degradation -- refusing there would be wrong."""
        with _patch_env(curobo=True, collision=False):
            self.assertEqual(require_motion_stack("curobo", exact_mesh_collision=False), [])

    def test_a_cell_that_never_asked_for_curobo_is_not_held_to_it(self) -> None:
        with _patch_env(curobo=False, collision=True):
            self.assertEqual(require_motion_stack("ik", exact_mesh_collision=True), [])


class ExplicitOverrideTests(unittest.TestCase):
    def test_the_override_permits_the_run_but_still_prints(self) -> None:
        """An operator may knowingly run degraded -- but never quietly. The banner prints EVERY time, and
        the missing engines are returned so the runner can stamp them onto its results."""
        with _patch_env(curobo=False, collision=False), \
                mock.patch.dict("os.environ", {ENV_ALLOW_DEGRADED: "1"}), \
                mock.patch("builtins.print") as printed:
            missing = require_motion_stack("curobo", exact_mesh_collision=True)
        self.assertEqual(len(missing), 2)
        self.assertTrue(printed.called, "a degraded run must still announce itself")


class BannerTests(unittest.TestCase):
    def test_it_names_the_variable_to_set_for_each_engine(self) -> None:
        """The point of the banner is that the reader can FIX it without going hunting."""
        text = motion_stack_banner(
            None, planner="curobo", missing=["cuRobo planner", "exact-mesh collision (Coal/fcl)"],
        )
        self.assertIn("WILLY_CUROBO_PYTHON", text)
        self.assertIn("WILLY_COAL_PREFIX", text)
        # ONE page now, not two: the Coal and cuRobo setup docs were merged into ext_deps/README.md
        # on 2026-08-26, so the banner names the single install guide once.
        self.assertEqual(text.count("ext_deps/README.md"), 2,
                         "each missing engine must still carry the install pointer")
        self.assertIn(ENV_ALLOW_DEGRADED, text)

    def test_it_states_the_honesty_consequence_not_just_the_defect(self) -> None:
        """A missing engine is not a performance note: numbers measured without it describe a different
        system and must not be reported as the configured one."""
        text = motion_stack_banner(None, planner="curobo", missing=["cuRobo planner"])
        self.assertIn("must not be reported", text)

    def test_it_is_a_box_that_cannot_be_skimmed_past(self) -> None:
        text = motion_stack_banner(None, planner="curobo", missing=["cuRobo planner"])
        lines = [line for line in text.splitlines() if line]
        self.assertGreater(len(lines), 6)
        self.assertEqual({len(line) for line in lines}, {96}, "the box must be square")
        self.assertTrue(all(line.startswith("#") and line.endswith("#") for line in lines))


if __name__ == "__main__":
    unittest.main()
