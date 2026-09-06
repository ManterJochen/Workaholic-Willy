"""A cell about to collect untrainable records says so at boot, not months later.

THE FAILURE. The per-candidate feature rows -- ``extra.rl_candidate_features``, the only thing a
pairwise ranker can rank -- are written exclusively when the RANKING SHADOW ran, and the shadow is
built only for ``robot.rl.mode == "rl_shadow"``. A cell on any other mode logs records that look
completely healthy: the outcome is there, the KPIs roll up, the file grows every pick. The shortfall
surfaces only when somebody finally trains, and the trainer reports a converged fit over nothing.

That silence is the point. A customer collecting on their own cell can fill a log for months and
discover a config flag was off at the end of it, which is an expensive way to learn one line of YAML.

A WARNING, NEVER A REFUSAL, asserted below so a later pass does not "tighten" it into one. Record
logging also feeds the KPI roll-up, the failure taxonomy and every post-mortem a bring-up depends on;
a cell that never intends to train an RL policy has good reasons to log. Refusing here would be a
fail-closed gesture with no safety behind it, which is the opposite of what earns fail-closed its
place everywhere else in this stack.

MEASURED through the real composition root:

    profile console_dummy   rl.mode 'hybrid_ml'   -> warning fired
    profile rl_datagen      rl.mode 'rl_shadow'   -> silent
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.service import (
    _warn_if_records_will_not_be_trainable,
)

_LOGGER_NAME = "src.robot.execution.autonomous_grasp.service"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _warn_for(mode: str | None, path: str = "logs/cell/grasp_records.jsonl") -> list[str]:
    robot = SimpleNamespace(rl=SimpleNamespace(mode=mode) if mode is not None else None)
    capture = _Capture()
    logger = logging.getLogger(_LOGGER_NAME)
    previous = logger.level
    logger.addHandler(capture)
    logger.setLevel(logging.WARNING)
    try:
        _warn_if_records_will_not_be_trainable(robot, path)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous)
    return capture.messages


class ItFiresWhereItShouldTests(unittest.TestCase):
    def test_the_shipped_default_mode_warns(self) -> None:
        # `hybrid_ml` is the shipped default and runs no RL at all -- the most likely cell to be
        # quietly collecting a corpus nobody can train on.
        messages = _warn_for("hybrid_ml")
        self.assertEqual(len(messages), 1)
        self.assertIn("per-candidate features", messages[0])

    def test_every_non_shadow_mode_warns(self) -> None:
        for mode in ("hybrid_ml", "geometry_only", "", None):
            with self.subTest(mode=mode):
                self.assertEqual(len(_warn_for(mode)), 1)

    def test_the_message_names_the_flag_the_path_and_the_way_out(self) -> None:
        # A warning that says "something is wrong" without saying which line to change has only moved
        # the guessing, which is the same rule the preflight's `fix` strings follow.
        message = _warn_for("hybrid_ml", "logs/september/grasp_records.jsonl")[0]
        self.assertIn("robot.rl.mode", message)
        self.assertIn("rl_shadow", message)
        self.assertIn("policy_id", message)
        self.assertIn("logs/september/grasp_records.jsonl", message)
        self.assertIn("check-dataset", message)

    def test_it_says_the_records_are_still_worth_keeping(self) -> None:
        # Otherwise the honest reading is "turn logging off", which would cost the cell its KPIs and
        # every post-mortem a bring-up depends on.
        message = _warn_for("hybrid_ml")[0]
        self.assertIn("KPIs", message)


class ItStaysSilentWhereItShouldTests(unittest.TestCase):
    def test_rl_shadow_is_silent(self) -> None:
        self.assertEqual(_warn_for("rl_shadow"), [])

    def test_it_is_a_warning_and_never_raises(self) -> None:
        # Pinned deliberately: record logging serves KPIs and diagnosis, so refusing here would be a
        # fail-closed gesture with no safety reason behind it.
        for mode in ("hybrid_ml", "rl_shadow", None, "nonsense"):
            with self.subTest(mode=mode):
                _warn_if_records_will_not_be_trainable(
                    SimpleNamespace(rl=SimpleNamespace(mode=mode)), "logs/x.jsonl"
                )  # must not raise

    def test_a_config_without_an_rl_block_at_all_still_only_warns(self) -> None:
        messages = _warn_for(None)
        self.assertEqual(len(messages), 1)
        self.assertIn("per-candidate features", messages[0])


class WiredIntoTheCompositionRootTests(unittest.TestCase):
    """It has to run where record logging is actually switched on, or it warns nobody."""

    def test_from_robot_config_calls_it(self) -> None:
        from pathlib import Path

        source = Path(
            "src/robot/execution/autonomous_grasp/service.py"
        ).read_text(encoding="utf-8")
        # Immediately after `enable_record_logging` in the config-driven boot path: the warning is
        # about a cell that has just turned logging ON, so anywhere else would either miss cells or
        # warn ones that log nothing.
        self.assertIn("_warn_if_records_will_not_be_trainable(robot_cfg, record_log_path)", source)
        wiring = source.index("_warn_if_records_will_not_be_trainable(robot_cfg, record_log_path)")
        enable = source.index("service.enable_record_logging(\n                record_log_path,")
        self.assertLess(enable, wiring)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
