"""P1 — the demo-mode -> GraspMode + from_components sub-policy mapping (willy_sim/harness/modes.py).

Pure-Python / mock-safe (no isaacsim). Pins that EASY / AUTO / DENSE_CLUTTER resolve and wire the right
sub-policies, and that the two modes removed on 2026-09-29 (``closed_loop``, ``dense_autonomous``, with the
two-scan refinement they ran) are refused with the mode to name instead. Their wiring cases, the refiner and
the verifier each wired, left with them.
"""

from __future__ import annotations

import unittest

from src.config.schema._removed import REMOVED_GRASP_MODES
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
        self.assertIs(resolve_demo_mode("Dense-Clutter"), GraspMode.DENSE_CLUTTER)  # case/sep-insensitive

    def test_enum_passthrough(self) -> None:
        self.assertIs(resolve_demo_mode(GraspMode.AUTO), GraspMode.AUTO)

    def test_a_removed_mode_is_refused_with_its_replacement(self) -> None:
        for mode, replacement in (("closed_loop", "name auto"), ("Dense-Autonomous", "name dense_clutter")):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as caught:
                    resolve_demo_mode(mode)
                said = str(caught.exception)
                self.assertIn("removed on purpose", said)
                self.assertIn(replacement, said)
                self.assertIn(REMOVED_GRASP_MODES[mode.lower().replace("-", "_")], said)

    def test_unknown_mode_refused(self) -> None:
        with self.assertRaises(ValueError):
            resolve_demo_mode("turbo")

    def test_mode_constants(self) -> None:
        self.assertEqual(DEMO_MODES, ("easy", "auto"))
        self.assertEqual(DENSE_MODES, ("easy", "auto", "dense_clutter"))


class ModeServiceKwargsTests(unittest.TestCase):
    def test_easy_has_no_extra_wiring(self) -> None:
        self.assertEqual(mode_service_kwargs("easy"), {})

    def test_dense_clutter_has_no_extra_wiring(self) -> None:
        # DENSE_CLUTTER only switches the sampler; the profile + clutter signal do the rest.
        self.assertEqual(mode_service_kwargs("dense_clutter"), {})

    def test_auto_wires_a_decision_engine(self) -> None:
        kw = mode_service_kwargs("auto")
        self.assertIn("decision_engine", kw)
        self.assertEqual(set(kw), {"decision_engine"})


if __name__ == "__main__":
    unittest.main()
