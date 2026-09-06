"""R10.1 — behavioral conformance of every SafetyGuard implementation to the fail-closed contract.

The RobotArm / Gripper conformance suites guard return *types*; this guards the SafetyGuard *behavioral*
contract documented in ``backend/src/robot/safety/guard.py``. Every guard wired into the canonical
preflight pipeline must:

* expose a non-empty, stable ``name`` that is its slot in the canonical guard order;
* return a :class:`SafetyDecision` tagged with that ``name`` (never a bare ``bool`` / ``None``);
* **NOT raise on an ordinary rejection** — the load-bearing fail-closed invariant. A guard that raised
  instead of returning :meth:`SafetyDecision.reject` would crash the preflight (and, depending on the
  caller, the whole motion) rather than *refusing* the move. This suite is the cross-guard proof that
  none of them does.
* never reject a benign, in-workspace command.

The pipeline is built from the real production wiring (``SafetyPreflight.from_safety_config`` on the
loaded config, which enforces all six guards), so this exercises the actual shipped guard set — not
hand-built stubs. ``WorkspaceGuard`` / ``SingularityGuard`` are intentionally absent: the former is
wrapped by ``WorkspaceSafetyGuard`` (the real SafetyGuard adapter) and the latter is not a SafetyGuard
(it has no ``evaluate(ctx) -> SafetyDecision``).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.config.loader import load_config
from src.geometry import Frame, Pose
from src.geometry.quaternion import IDENTITY_QUAT_XYZW
from src.robot.core import MotionCommand
from src.robot.safety import (
    SafetyContext,
    SafetyDecision,
    SafetyGuard,
    SafetyPreflight,
)

# The canonical pipeline order (pinned here so a reorder/rename in from_safety_config is caught).
_CANONICAL_ORDER = (
    "workspace",
    "joint_limit",
    "ik_quality",
    "self_collision",
    "payload",
    "motion_continuity",
)


def _pose(x_mm: float, y_mm: float, z_mm: float, label: str = "p") -> Pose:
    return Pose(
        position_mm=np.array([x_mm, y_mm, z_mm], dtype=np.float64),
        quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
        frame=Frame.BASE,
        label=label,
    )


def _guards() -> tuple[SafetyGuard, ...]:
    cfg = load_config()
    robot = cfg.robot
    assert robot is not None, "loaded config has no robot block"
    pf = SafetyPreflight.from_safety_config(robot.safety, robot.workspace_limits)
    return pf.guards


class SafetyGuardConformanceTests(unittest.TestCase):
    """Every production SafetyGuard honours the fail-closed Protocol contract."""

    def setUp(self) -> None:
        self._guards = _guards()
        # Full canonical pipeline expected on the default (all-enforced) config.
        self.assertEqual(
            tuple(g.name for g in self._guards),
            _CANONICAL_ORDER,
            "from_safety_config did not wire the full canonical guard pipeline",
        )

    def test_every_guard_satisfies_runtime_protocol(self) -> None:
        for g in self._guards:
            with self.subTest(guard=type(g).__name__):
                self.assertIsInstance(g, SafetyGuard)

    def test_name_is_nonempty_stable_and_canonical(self) -> None:
        for g in self._guards:
            with self.subTest(guard=type(g).__name__):
                self.assertIsInstance(g.name, str)
                self.assertTrue(g.name, "guard name must be non-empty")
                self.assertEqual(g.name, g.name, "guard name must be stable across reads")
                self.assertIn(g.name, _CANONICAL_ORDER)

    def test_evaluate_never_raises_and_tags_its_decision(self) -> None:
        # THE fail-closed invariant (guard.py): "Guards MUST NOT raise on ordinary rejection." For both
        # a benign in-workspace move and a wildly out-of-box one, every guard must RETURN a SafetyDecision
        # tagged with its own name (some accept, some fail-closed reject — but NONE raises). A guard that
        # raised here would crash the preflight instead of refusing the move.
        contexts = {
            "benign": SafetyContext(command=MotionCommand.MOVE_TO, target_pose=_pose(0.0, 0.0, 350.0)),
            "out_of_box": SafetyContext(
                command=MotionCommand.MOVE_TO, target_pose=_pose(1.0e4, 1.0e4, 1.0e4)
            ),
        }
        for label, ctx in contexts.items():
            for g in self._guards:
                with self.subTest(guard=g.name, context=label):
                    try:
                        decision = g.evaluate(ctx)
                    except Exception as exc:  # noqa: BLE001 - the whole point is to prove no guard raises
                        self.fail(
                            f"{g.name} raised {type(exc).__name__} on the {label} context instead of "
                            f"returning a SafetyDecision: {exc!r}"
                        )
                    self.assertIsInstance(decision, SafetyDecision)
                    self.assertEqual(decision.guard, g.name)

    def test_workspace_guard_accepts_in_box_and_rejects_out_of_box(self) -> None:
        # Non-vacuity anchor: the workspace guard (which needs only a pose) must ACCEPT a centred target
        # and REJECT a 10 m out-of-box one — proving both decision paths are genuinely exercised, so the
        # suite is not all-accept (or all-error-swallow) theatre.
        ws = next(g for g in self._guards if g.name == "workspace")
        accept = ws.evaluate(
            SafetyContext(command=MotionCommand.MOVE_TO, target_pose=_pose(0.0, 0.0, 350.0))
        )
        reject = ws.evaluate(
            SafetyContext(command=MotionCommand.MOVE_TO, target_pose=_pose(1.0e4, 1.0e4, 1.0e4))
        )
        self.assertFalse(accept.rejected, "workspace guard rejected a centred in-box target")
        self.assertTrue(reject.rejected, "workspace guard accepted a 10 m out-of-box target")
        self.assertEqual(reject.guard, "workspace")

    def test_data_starved_guards_fail_closed(self) -> None:
        # Documents + pins the measured fail-closed semantics: the data-dependent guards REJECT a pose-only
        # MOVE_TO (no resolved joints / IK metrics) rather than waving it through. Flipping either to an
        # accept-on-missing-data would be a real safety regression and must fail here.
        ctx = SafetyContext(command=MotionCommand.MOVE_TO, target_pose=_pose(0.0, 0.0, 350.0))
        by_name = {g.name: g for g in self._guards}
        for name in ("joint_limit", "ik_quality"):
            with self.subTest(guard=name):
                decision = by_name[name].evaluate(ctx)
                self.assertTrue(
                    decision.rejected,
                    f"{name} did NOT fail closed on a context with no joint / IK data",
                )


if __name__ == "__main__":
    unittest.main()
