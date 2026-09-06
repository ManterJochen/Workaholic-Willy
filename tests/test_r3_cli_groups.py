"""R3.2 CLI command-group contract — asserts the register-pattern decomposition kept every subcommand wired.

__main__'s god-CLI was split into _commands_dataset/train/eval/hardening (each owns its _cmd_* handlers + a
register_X_commands(sub)); __main__.build_parser now just calls the four registers. This pins that build_parser
still exposes ALL 13 subcommands AND each dispatches to a handler (the set_defaults wiring survived the move). The
the per-capability test suites exercise the real CLI behaviour via subprocess; this is the fast structural guard.
"""

from __future__ import annotations

import argparse
import unittest

from src.robot.grasping.rl.__main__ import build_parser

_EXPECTED_SUBCOMMANDS = {
    # Added 2026-08-19. Answers "is this record log trainable?" WITHOUT training: occupancy, variance
    # and success/fail pair counts, with a meaningful exit code. It exists because the two ways a
    # corpus is worthless -- no pairs, and features that read 0.0 because `_project_feature` maps a
    # missing key to zero silently -- both look identical to a healthy log until a training run says
    # "converged" over nothing.
    "check-dataset",
    "build-dataset",
    "audit-leakage",
    "replay-env-check",
    "train-candidate-policy",
    "train-ranking-policy",
    "train-sequencing-policy",
    "train-perception-budget-policy",
    "train-recovery-policy",
    "ope",
    "promote-policy",
    "paired-soak",
    "rollback-drill",
}


class CliGroupsContractTests(unittest.TestCase):
    def test_all_subcommands_registered_with_handlers(self) -> None:
        parser = build_parser()
        sub_actions = [
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        ]
        self.assertEqual(len(sub_actions), 1, "expected exactly one subparsers action")
        choices = sub_actions[0].choices
        self.assertEqual(set(choices), _EXPECTED_SUBCOMMANDS)
        for name, subparser in choices.items():
            with self.subTest(subcommand=name):
                # the register functions set_defaults(handler=_cmd_X) — the dispatch contract.
                self.assertIn("handler", subparser._defaults, f"{name} has no handler wired")
                self.assertTrue(callable(subparser._defaults["handler"]))


if __name__ == "__main__":
    unittest.main()
