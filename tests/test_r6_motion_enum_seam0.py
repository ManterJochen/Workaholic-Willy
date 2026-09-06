"""R6.0 Seam-0 — pin the byte-identical surface of MotionStatus/MotionCommand/MotionResult before R6.1.

R6.1a swaps ``class X(str, Enum)`` → ``StrEnum`` (MotionStatus/MotionCommand) and R6.1b adds ``slots=True``
to MotionResult/URPose. On Python 3.11 the ONLY behavioral delta of the StrEnum swap is ``__str__`` /
``__format__`` / ``%s`` (which flip from the qualified ``"MotionStatus.EXECUTED"`` to the value
``"executed"``); EVERYTHING that any consumer or persisted artifact actually relies on —
``repr`` / ``.value`` / ``.name`` / ``==`` (str compare) / ``is`` (identity) / ``json.dumps`` (str-subclass
fast path) — stays byte-identical. slots=True changes none of those either (no subclass, no dynamic attr,
dataclass repr unchanged).

This golden pins exactly that byte-identical surface so the swap is provably safe. The CHANGES rows (str / f-string
/ %s) are deliberately NOT asserted here — they are an intended, surfaced change documented below; the only
runtime LOGIC that consumed the qualified str() form (recovery._motion_executed + service.py legacy-move
predicate) is fixed to an identity check IN THE SAME COMMIT as the swap (R6.1a), and guarded by the hardened
real-enum assertion in test_r4_recovery_seam0.

CHANGES rows (captured pre-swap; flip to the .value form after R6.1a — NOT a regression):
    str(MotionStatus.EXECUTED)      "MotionStatus.EXECUTED"        -> "executed"
    f"{MotionStatus.EXECUTED}"      "MotionStatus.EXECUTED"        -> "executed"
    "%s" % MotionStatus.EXECUTED    "MotionStatus.EXECUTED"        -> "executed"
"""

from __future__ import annotations

import json
import unittest

from src.robot.core.motion_result import (
    MotionCommand,
    MotionResult,
    MotionStatus,
)


class MotionEnumSeam0Tests(unittest.TestCase):
    def test_motion_status_same_rows_byte_identical(self) -> None:
        m = MotionStatus.EXECUTED
        self.assertEqual(repr(m), "<MotionStatus.EXECUTED: 'executed'>")
        self.assertEqual(m.value, "executed")
        self.assertEqual(m.name, "EXECUTED")
        self.assertTrue(m == "executed")  # str comparison stays True
        self.assertIs(m, MotionStatus.EXECUTED)  # identity stays
        self.assertEqual(json.dumps(m), '"executed"')
        self.assertEqual(json.dumps({"k": m}), '{"k": "executed"}')

    def test_motion_status_multiword_same_rows(self) -> None:
        m = MotionStatus.SELF_COLLISION_REJECTED
        self.assertEqual(
            repr(m), "<MotionStatus.SELF_COLLISION_REJECTED: 'self_collision_rejected'>"
        )
        self.assertEqual(m.value, "self_collision_rejected")
        self.assertEqual(m.name, "SELF_COLLISION_REJECTED")
        self.assertTrue(m == "self_collision_rejected")
        self.assertEqual(json.dumps(m), '"self_collision_rejected"')
        self.assertEqual(json.dumps({"k": m}), '{"k": "self_collision_rejected"}')

    def test_motion_command_same_rows_byte_identical(self) -> None:
        c = MotionCommand.MOVE_TO
        self.assertEqual(repr(c), "<MotionCommand.MOVE_TO: 'move_to'>")
        self.assertEqual(c.value, "move_to")
        self.assertEqual(c.name, "MOVE_TO")
        self.assertTrue(c == "move_to")
        self.assertIs(c, MotionCommand.MOVE_TO)
        self.assertEqual(json.dumps(c), '"move_to"')

    def test_motion_result_repr_byte_identical(self) -> None:
        # The dataclass repr embeds the ENUM repr (<...: 'executed'>), which is unchanged by BOTH the
        # StrEnum swap (repr identical) and slots=True (dataclass repr generation identical). This single
        # assertion guards R6.1a + R6.1b at once — a regression in either flips it.
        self.assertEqual(
            repr(MotionResult.executed(MotionCommand.MOVE_TO, message="hi")),
            "MotionResult(status=<MotionStatus.EXECUTED: 'executed'>, "
            "command=<MotionCommand.MOVE_TO: 'move_to'>, target_pose=None, "
            "target_joints=None, message='hi')",
        )
        self.assertEqual(
            repr(MotionResult.failed(MotionStatus.IK_FAILED, MotionCommand.MOVE_TO, message="x")),
            "MotionResult(status=<MotionStatus.IK_FAILED: 'ik_failed'>, "
            "command=<MotionCommand.MOVE_TO: 'move_to'>, target_pose=None, "
            "target_joints=None, message='x')",
        )

    def test_json_safe_value_surface(self) -> None:
        # The persisted-record contract: a MotionStatus serializes to its VALUE through json (the str-subclass
        # fast path), not the qualified form — true under both (str,Enum) and StrEnum.
        self.assertEqual(
            json.dumps({"status": MotionStatus.EXECUTED, "command": MotionCommand.MOVE_TO}),
            '{"status": "executed", "command": "move_to"}',
        )


if __name__ == "__main__":
    unittest.main()
