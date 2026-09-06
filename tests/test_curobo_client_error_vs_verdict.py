"""A failed planning CALL must never reach a caller looking like a planning VERDICT.

This test exists because the confusion cost three commits and two wrong conclusions in writing.

``plan_js`` is not a method on this cuRobo build's planner — it is ``plan_cspace``. Every joint-space
request therefore raised ``AttributeError`` inside the sidecar, the sidecar's broad ``except`` reported
it as a failure, and the client mapped any failure to ``None``. ``None`` is the client's documented way
of saying *"the planner searched and found no collision-free path"*, so the arm refused to move and the
refusal read as a statement about the robot. It was investigated as one: the scan pose was called
unsafe (twice, in commits), then the planner's collision world was blamed, then the robot descriptor —
and a search began for inflated collision spheres in ``ur5e.yml``, a file that had nothing to do with
it. The tell was there from the first run and went unnoticed: the planner also refused to plan from a
configuration **to itself**.

So the two outcomes are now distinguishable by construction — the sidecar labels every failure
``planner_error: true|false`` — and this asserts the client acts on that label. It needs no GPU, no
cuRobo and no sidecar: the whole point is that the distinction lives in the protocol, not in the
planner.
"""

from __future__ import annotations

import unittest

from src.robot.safety.planning.curobo_client import (
    CuroboUnavailableError,
    _raise_if_call_failed,
)


class ErrorVersusVerdictTests(unittest.TestCase):
    def test_a_failed_call_raises(self) -> None:
        with self.assertRaises(CuroboUnavailableError) as ctx:
            _raise_if_call_failed({
                "success": False,
                "planner_error": True,
                "reason": "AttributeError: 'MotionPlanner' object has no attribute 'plan_js'",
            })
        message = str(ctx.exception)
        self.assertIn("CALL failed", message)
        self.assertIn("AttributeError", message, "the raised error must carry the actual cause")

    def test_a_genuine_no_solution_does_not_raise(self) -> None:
        """The planner searched and found nothing: a verdict the caller must be able to act on."""
        _raise_if_call_failed({
            "success": False,
            "planner_error": False,
            "reason": "no collision-free joint-space plan",
        })

    def test_an_unlabelled_failure_does_not_raise(self) -> None:
        """Back-compat: an older sidecar sends no label, and its failures stay verdicts as before."""
        _raise_if_call_failed({"success": False, "reason": "something"})

    def test_the_sidecar_labels_both_outcomes(self) -> None:
        """Read the sidecar source: every failure emission must carry the label, or the split is fiction.

        A source check rather than a live one, because running the sidecar needs a GPU and the cuRobo
        env — but a missing label silently collapses the two outcomes back into one, which is exactly
        the bug this file is about. Cheap to assert, expensive to rediscover.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/robot/safety/planning/curobo_planner_server.py"
        ).read_text(encoding="utf-8")
        emissions = [
            line for line in source.splitlines()
            if '"success": False' in line
        ]
        self.assertGreaterEqual(len(emissions), 2, "expected both plan paths to emit failures")
        for line in emissions:
            with self.subTest(line=line.strip()):
                self.assertIn(
                    "planner_error", line,
                    "a failure emission without planner_error makes a broken call indistinguishable "
                    "from a planning verdict -- the exact bug this test exists for",
                )

    def test_the_sidecar_calls_a_method_that_exists(self) -> None:
        """``plan_js`` was invented. Assert the name in the source is the real one.

        cuRobo is not importable in CI, so this cannot introspect the class; it pins the name that was
        verified on-box against ``MotionPlanner`` (2026-08-11, curobo 0.8.0.post1.dev42). If a future
        cuRobo renames it, this fails and points at the probe rather than at a phantom robot descriptor.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/robot/safety/planning/curobo_planner_server.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_planner.plan_cspace(", source)
        self.assertNotIn("_planner.plan_js(", source)


if __name__ == "__main__":
    unittest.main()
