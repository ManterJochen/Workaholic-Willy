"""R4 K3 Seam-0 — pin the recovery dedups (the executed predicate + the serializer re-export).

The `_motion_executed` dedup of the two identical agitate/nudge call sites is, since R6.1a, an IDENTITY check
against `MotionStatus.EXECUTED` (the former str-match on the qualified "MotionStatus.EXECUTED" form was a side
effect of MotionStatus being a plain (str, Enum); now that it is a StrEnum, str() yields "executed" and the old
str-match would silently reject success — see test_motion_executed_real_enum_guard). And
`recovery_actions_from_trail`, relocated to recovery_trail_serialize, must stay re-exported from
recovery_orchestrator (the 4 by-name importers: service.py, run_dense_pick.py, 2 tests).
"""

from __future__ import annotations

import unittest


class RecoverySeam0Tests(unittest.TestCase):
    def test_motion_executed_predicate(self) -> None:
        # R6.1a: identity check (status is MotionStatus.EXECUTED), NOT a str-match. A raw string — even the
        # value "executed" or the old qualified form — is NOT the enum member and counts as not-executed.
        from src.robot.core import MotionStatus
        from src.robot.grasping.recovery.policy import _motion_executed

        self.assertFalse(_motion_executed(None))
        self.assertTrue(_motion_executed(MotionStatus.EXECUTED))
        self.assertFalse(_motion_executed(MotionStatus.IK_FAILED))
        self.assertFalse(_motion_executed("executed"))
        self.assertFalse(_motion_executed("MotionStatus.EXECUTED"))

    def test_motion_executed_real_enum_guard(self) -> None:
        # R6.0 — the REAL regression guard the literal-string cases above can NOT provide. At runtime
        # _motion_executed receives an actual MotionStatus enum member (from MotionResult.status), NOT a
        # string. Under today's (str,Enum) + str-predicate this passes (str(EXECUTED) ends with "EXECUTED").
        # The R6.1a StrEnum swap makes str(EXECUTED)=="executed" → the str-predicate would silently return
        # False for a SUCCESSFUL move (breaking the C2/C3 recovery loops); this assertion goes RED the instant
        # the base is swapped without the same-commit identity-check fix.
        from src.robot.core import MotionStatus
        from src.robot.grasping.recovery.policy import _motion_executed

        self.assertTrue(_motion_executed(MotionStatus.EXECUTED))
        self.assertFalse(_motion_executed(MotionStatus.WORKSPACE_REJECTED))

    def test_recovery_actions_from_trail_reexport(self) -> None:
        from src.robot.grasping.recovery import (
            orchestrator,
            trail_serialize,
        )

        self.assertIs(
            orchestrator.recovery_actions_from_trail,
            trail_serialize.recovery_actions_from_trail,
            "recovery_orchestrator no longer re-exports the relocated serializer",
        )


if __name__ == "__main__":
    unittest.main()
