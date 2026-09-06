"""Honest action-mask evaluator — Schicht 1 (structural correctness) + Schicht 2 (consistency metric).

Fake SafetyPreflight / IK services drive the per-candidate checks so the whole mask contract is
provable Isaac-free: each channel fires exactly when its check fails, an accepted candidate is never
masked, precedence order holds, a failing check is fail-closed, and the executed-candidate-masked
metric behaves. The real-data measurement is the on-box shadow soak.
"""

from __future__ import annotations

import unittest

from src.robot.execution.autonomous_grasp.action_mask_eval import (
    MaskCandidate,
    build_action_mask_context,
    executed_candidate_masked,
    feasibility_reject_from_ik,
    safety_reject_from_preflight,
)
from src.robot.grasping.rl.action_mask import apply_action_mask


class _Decision:
    def __init__(self, rejected: bool) -> None:
        self.rejected = rejected


class _FakePreflight:
    """Rejects any candidate whose id is in ``reject_ids``."""

    def __init__(self, reject_ids: set[str]) -> None:
        self._reject = reject_ids

    def evaluate(self, ctx: str) -> _Decision:
        return _Decision(ctx in self._reject)


def _cands(*ids: str) -> list[MaskCandidate]:
    # payload doubles as the candidate id for the fake services
    return [MaskCandidate(candidate_id=i, payload=i) for i in ids]


class ChannelCorrectnessTests(unittest.TestCase):
    def test_unsupplied_channels_stay_empty(self) -> None:
        ctx = build_action_mask_context(_cands("a", "b"))
        self.assertEqual(ctx.safety_rejected_ids, frozenset())
        self.assertEqual(ctx.feasibility_failed_ids, frozenset())
        self.assertFalse(ctx.degraded_mode_active)
        # nothing masked
        for row in apply_action_mask(("a", "b"), ctx):
            self.assertTrue(row.keep)

    def test_safety_predicate_populates_only_safety(self) -> None:
        pf = _FakePreflight({"b"})
        ctx = build_action_mask_context(
            _cands("a", "b", "c"),
            safety_reject=safety_reject_from_preflight(pf, lambda c: c.payload),
        )
        self.assertEqual(ctx.safety_rejected_ids, frozenset({"b"}))
        self.assertEqual(ctx.feasibility_failed_ids, frozenset())
        rows = {r.candidate_id: r for r in apply_action_mask(("a", "b", "c"), ctx)}
        self.assertEqual(rows["b"].masked_by, ("safety",))
        self.assertTrue(rows["a"].keep)
        self.assertTrue(rows["c"].keep)

    def test_feasibility_from_ik_no_solution(self) -> None:
        # ik returns None (no solution) for "c" only
        def ik(payload: str):
            return None if payload == "c" else object()

        ctx = build_action_mask_context(
            _cands("a", "b", "c"), feasibility_reject=feasibility_reject_from_ik(ik)
        )
        self.assertEqual(ctx.feasibility_failed_ids, frozenset({"c"}))

    def test_multiple_channels_and_precedence_order(self) -> None:
        pf = _FakePreflight({"x"})

        def ik(payload: str):
            return None if payload == "x" else object()

        ctx = build_action_mask_context(
            _cands("x", "y"),
            safety_reject=safety_reject_from_preflight(pf, lambda c: c.payload),
            feasibility_reject=feasibility_reject_from_ik(ik),
            degraded_mode_active=False,
        )
        row_x = apply_action_mask(("x", "y"), ctx)[0]
        # x fails BOTH safety + feasibility; precedence puts safety before feasibility
        self.assertEqual(row_x.masked_by, ("safety", "feasibility"))

    def test_degraded_mode_masks_everything(self) -> None:
        ctx = build_action_mask_context(_cands("a", "b"), degraded_mode_active=True)
        for row in apply_action_mask(("a", "b"), ctx):
            self.assertEqual(row.masked_by, ("degraded_mode",))

    def test_failing_check_is_fail_closed(self) -> None:
        def boom(_c: MaskCandidate) -> bool:
            raise RuntimeError("check exploded")

        ctx = build_action_mask_context(_cands("a"), safety_reject=boom)
        # an errored check masks the candidate (never a silent pass)
        self.assertEqual(ctx.safety_rejected_ids, frozenset({"a"}))


class ConsistencyMetricTests(unittest.TestCase):
    def test_executed_candidate_never_masked_when_accepted(self) -> None:
        pf = _FakePreflight({"b"})  # only b rejected; executed=a is accepted
        ctx = build_action_mask_context(
            _cands("a", "b"),
            safety_reject=safety_reject_from_preflight(pf, lambda c: c.payload),
        )
        self.assertFalse(executed_candidate_masked(ctx, "a"))
        # if the executed candidate WERE masked, the metric catches it (divergence signal)
        self.assertTrue(executed_candidate_masked(ctx, "b"))

    def test_none_executed_is_not_masked(self) -> None:
        ctx = build_action_mask_context(_cands("a"), degraded_mode_active=True)
        self.assertFalse(executed_candidate_masked(ctx, None))


if __name__ == "__main__":
    unittest.main()
