"""P1 — the demo-mode -> GraspMode + from_components sub-policy mapping (willy_sim/modes.py).

Pure-Python / mock-safe (no isaacsim). Pins that EASY/AUTO/CLOSED_LOOP resolve + wire the right
sub-policies, that the dense modes are refused (deferred to P2/P3), and that the per-runner refinement
tuning threads through.
"""

from __future__ import annotations

import unittest

from src.willy_sim.harness.modes import (
    DEMO_MODES,
    DENSE_MODES,
    mode_service_kwargs,
    resolve_demo_mode,
)
from src.robot.execution.autonomous_grasp.config import GraspMode


class ResolveDemoModeTests(unittest.TestCase):
    def test_string_aliases(self) -> None:
        self.assertIs(resolve_demo_mode("easy"), GraspMode.EASY)
        self.assertIs(resolve_demo_mode("auto"), GraspMode.AUTO)
        self.assertIs(resolve_demo_mode("closed_loop"), GraspMode.CLOSED_LOOP)
        self.assertIs(resolve_demo_mode("Closed-Loop"), GraspMode.CLOSED_LOOP)  # case/sep-insensitive
        self.assertIs(resolve_demo_mode("dense_clutter"), GraspMode.DENSE_CLUTTER)  # P2

    def test_enum_passthrough(self) -> None:
        self.assertIs(resolve_demo_mode(GraspMode.AUTO), GraspMode.AUTO)

    def test_dense_autonomous_resolves(self) -> None:
        # C2: dense_autonomous is now wired (the full perceive->refine->verify->recover loop).
        self.assertIs(resolve_demo_mode("dense_autonomous"), GraspMode.DENSE_AUTONOMOUS)
        self.assertIs(resolve_demo_mode("Dense-Autonomous"), GraspMode.DENSE_AUTONOMOUS)

    def test_unknown_mode_refused(self) -> None:
        with self.assertRaises(ValueError):
            resolve_demo_mode("turbo")

    def test_mode_constants(self) -> None:
        self.assertEqual(DEMO_MODES, ("easy", "auto", "closed_loop"))
        self.assertEqual(
            DENSE_MODES, ("easy", "auto", "closed_loop", "dense_clutter", "dense_autonomous")
        )


class ModeServiceKwargsTests(unittest.TestCase):
    def test_easy_has_no_extra_wiring(self) -> None:
        self.assertEqual(mode_service_kwargs("easy"), {})

    def test_dense_clutter_has_no_extra_wiring(self) -> None:
        # DENSE_CLUTTER only switches the sampler (no S3/S4); the profile + clutter signal do the rest.
        self.assertEqual(mode_service_kwargs("dense_clutter"), {})

    def test_auto_wires_a_decision_engine(self) -> None:
        kw = mode_service_kwargs("auto")
        self.assertIn("decision_engine", kw)
        self.assertEqual(set(kw), {"decision_engine"})

    def test_closed_loop_wires_refiner_and_verifier(self) -> None:
        kw = mode_service_kwargs("closed_loop")
        self.assertEqual(
            set(kw),
            {"refinement_policy", "refiner", "verification_policy", "verifier"},
        )
        # the verifier is fail-open in sim (advisory width signal)
        self.assertFalse(kw["verification_policy"].fail_closed)
        self.assertTrue(kw["refinement_policy"].enabled)

    def test_closed_loop_refinement_kwargs_thread_through(self) -> None:
        kw = mode_service_kwargs("closed_loop", refinement_kwargs={"target_match_iou_threshold": 0.1})
        self.assertEqual(kw["refinement_policy"].target_match_iou_threshold, 0.1)

    def test_dense_autonomous_wires_loop_with_reperceive_hold(self) -> None:
        # C2: dense_autonomous wires the full loop, with the refiner in HOLD mode (reperceive=False) — a static
        # overhead camera can't give a 2nd viewpoint, so the standoff+re-perceive is skipped.
        kw = mode_service_kwargs("dense_autonomous")
        self.assertEqual(
            set(kw),
            {"refinement_policy", "refiner", "verification_policy", "verifier"},
        )
        self.assertTrue(kw["refinement_policy"].enabled)
        self.assertFalse(kw["refinement_policy"].reperceive)  # HOLD, not two-scan
        self.assertFalse(kw["verification_policy"].fail_closed)  # advisory width in sim


if __name__ == "__main__":
    unittest.main()
