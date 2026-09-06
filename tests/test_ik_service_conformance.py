"""R10.1 — behavioral conformance of every IKService implementation to the reachability contract.

The execution layer injects a vendor-neutral reachability oracle into the grasp calculator through the
``IKService`` Protocol (``backend/src/robot/grasping/planning/reachability.py``). Its documented contract
(see the Protocol + ``IKResult`` docstrings) is narrow but load-bearing: ``query(pose)`` must

* return an :class:`IKResult` — **never a bare ``bool`` / ``None``** (``filter_reachable_poses`` raises
  ``TypeError`` on anything else, so a wrong return type crashes the whole filter);
* be a *pure* oracle — no side effects on the pose / calculator;
* honour ``IKResult.reachable`` as a real ``bool`` decision, and on a rejection surface a non-empty
  ``reason`` string so the calculator can correlate the dropped candidate;
* **NOT raise on a well-formed base-frame pose** — every adapter wraps its underlying solver and maps
  failures into ``IKResult(reachable=False, reason=...)`` rather than propagating a vendor exception. This
  suite is the cross-adapter proof that none of them leaks an exception out of ``query``.

The suite parametrizes over every *constructible* production implementer, each paired with a well-formed
pose it should ACCEPT and one it should REJECT, so both decision branches are genuinely exercised (it is
not all-accept / all-reject theatre). The implementers are built with the exact fakes / construction
idioms the existing per-adapter suites use (``tests/test_execution_ik_service.py`` ``_FakeArm`` and
``tests/test_willy_sim_dense.py`` ``_IKArm``) so this exercises the shipped behaviour, not hand-built
stubs.

``URAnalyticIKService`` is intentionally absent: its constructor raises ``ImportError`` unless the
optional, undeclared ``ur_ikfast`` package is installed, so on a normal checkout it is not a constructible
DI seam and cannot be conformance-tested here (the existing suite only asserts that import-error). See the
module-level notes in the R10.1 report.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Frame
from src.robot.core import (
    JointPositions,
    RobotConnectionError,
    RobotKinematicsError,
)
from src.robot.execution.ik_service import (
    CachedIKService,
    RobotArmIKService,
)
from src.robot.grasping.planning import (
    IKResult,
    WorkspaceBoxIKService,
    filter_reachable_poses,
)
from src.robot.grasping.planning.grasp_pose import GraspPose


def _grasp(x: float = 0.0, *, frame: Frame = Frame.BASE) -> GraspPose:
    """Well-formed base-frame grasp pose (mirrors the existing IK-service suites)."""
    return GraspPose(
        position_mm=np.array([x, 0.0, 200.0], dtype=np.float64),
        rotation_matrix=np.eye(3, dtype=np.float64),
        grip_width_mm=40.0,
        score=0.5,
        confidence=0.5,
        contacts=(
            np.array([x, -20.0, 200.0], dtype=np.float64),
            np.array([x, 20.0, 200.0], dtype=np.float64),
        ),
        frame=frame,
        metadata={},
    )


class _FakeArm:
    """Minimal arm satisfying the subset of ``RobotArm`` the IK services touch.

    Mirrors ``tests/test_execution_ik_service.py`` so this exercises the real adapter
    code paths against a known-good arm idiom.
    """

    def __init__(self, *, connected: bool = True, always_raise_kinematics: bool = False) -> None:
        self.connected = connected
        self.always_raise_kinematics = always_raise_kinematics
        self.ik_calls: list[Any] = []

    def get_joint_positions(self) -> JointPositions:
        if not self.connected:
            raise RobotConnectionError("not connected")
        return JointPositions([0.0] * 6)

    def ik(self, pose: Any, *, seed: JointPositions | None = None) -> JointPositions:
        self.ik_calls.append(pose)
        if not self.connected:
            raise RobotConnectionError("not connected")
        if self.always_raise_kinematics:
            raise RobotKinematicsError("no solution")
        return JointPositions([0.1] * 6)


class _IKArm:
    """Arm idiom from ``tests/test_willy_sim_dense.py`` for ``ArmBackedIKService``.

    Has only ``ik`` (no ``fk``): a reachable ``ik`` then falls through the singularity
    probe gracefully (``analyze_joint_singularity`` fails -> reachable, probe_unavailable).
    """

    def __init__(self, *, raises: bool = False) -> None:
        self._raises = raises

    def ik(self, pose: Any) -> JointPositions:  # noqa: ANN401 - matches the sim idiom
        if self._raises:
            raise RuntimeError("unreachable")
        return JointPositions(np.zeros(6))


# Build the parametrisation explicitly so each impl's accept/reject pair is honest about which arm /
# pose produces which decision (some adapters reject by frame, some by a failing inner solver).
# ``ArmBackedIKService`` lives in the willy_sim composition layer; import it lazily so a checkout
# without that subtree still runs the rest of the matrix.
def _matrix() -> list[tuple[str, Any, GraspPose, Any, GraspPose]]:
    """(label, accept_service, accept_pose, reject_service, reject_pose)."""
    box = WorkspaceBoxIKService(
        min_corner_mm=np.array([-500.0, -500.0, 0.0]),
        max_corner_mm=np.array([500.0, 500.0, 800.0]),
    )
    matrix: list[tuple[str, Any, GraspPose, Any, GraspPose]] = [
        ("WorkspaceBoxIKService", box, _grasp(0.0), box, _grasp(10_000.0)),
        (
            # _FakeArm is the existing partial RobotArm stub from tests/test_execution_ik_service.py.
            "RobotArmIKService",
            RobotArmIKService(_FakeArm()),  # type: ignore[arg-type]
            _grasp(0.0),
            RobotArmIKService(_FakeArm(always_raise_kinematics=True)),  # type: ignore[arg-type]
            _grasp(0.0),
        ),
        (
            "CachedIKService",
            CachedIKService(RobotArmIKService(_FakeArm())),  # type: ignore[arg-type]
            _grasp(0.0),
            CachedIKService(RobotArmIKService(_FakeArm(always_raise_kinematics=True))),  # type: ignore[arg-type]
            _grasp(0.0),
        ),
    ]
    try:
        from src.willy_sim.harness.ik_service import ArmBackedIKService
    except Exception:  # noqa: BLE001 - optional sim layer
        pass
    else:
        matrix.append(
            (
                "ArmBackedIKService",
                ArmBackedIKService(_IKArm()),
                _grasp(0.0),
                ArmBackedIKService(_IKArm(raises=True)),
                _grasp(0.0),
            )
        )
    return matrix


class IKServiceConformanceTests(unittest.TestCase):
    """Every production IKService honours the reachability-oracle contract."""

    def setUp(self) -> None:
        self._matrix = _matrix()
        # Guard against a refactor silently dropping every adapter from the import surface.
        self.assertGreaterEqual(
            len(self._matrix), 3, "expected at least 3 constructible production IKService impls"
        )

    def _all_services(self) -> list[tuple[str, Any, GraspPose, bool]]:
        """Flatten the matrix into (label, service, pose, expected_reachable) cases."""
        flat: list[tuple[str, Any, GraspPose, bool]] = []
        for label, acc_svc, acc_pose, rej_svc, rej_pose in self._matrix:
            flat.append((f"{label}[accept]", acc_svc, acc_pose, True))
            flat.append((f"{label}[reject]", rej_svc, rej_pose, False))
        return flat

    def test_query_returns_ikresult_and_never_raises(self) -> None:
        # THE contract invariant: query() must RETURN an IKResult (never a bare bool / None / a raised
        # vendor exception) for both an accept-pose and a reject-pose. ``filter_reachable_poses`` raises
        # TypeError on a non-IKResult, so a wrong type would crash the whole reachability filter.
        for label, service, pose, _expected in self._all_services():
            with self.subTest(case=label):
                try:
                    result = service.query(pose)
                except Exception as exc:  # noqa: BLE001 - the whole point is to prove no adapter raises
                    self.fail(
                        f"{label}: query() raised {type(exc).__name__} on a well-formed pose instead of "
                        f"returning an IKResult: {exc!r}"
                    )
                self.assertIsInstance(result, IKResult, f"{label}: query() did not return an IKResult")
                self.assertIsInstance(result.reachable, bool, f"{label}: reachable must be a bool")

    def test_accept_and_reject_paths_are_genuinely_exercised(self) -> None:
        # Non-vacuity anchor: each adapter ACCEPTS its in-contract pose and REJECTS its out-of-contract
        # one. Proves both decision branches fire -> the suite is not all-accept (or all-reject) theatre.
        for label, acc_svc, acc_pose, rej_svc, rej_pose in self._matrix:
            with self.subTest(adapter=label):
                accept = acc_svc.query(acc_pose)
                reject = rej_svc.query(rej_pose)
                self.assertTrue(accept.reachable, f"{label}: rejected a pose it should accept")
                self.assertFalse(reject.reachable, f"{label}: accepted a pose it should reject")

    def test_rejection_carries_a_nonempty_reason(self) -> None:
        # IKResult.reason documents WHY a candidate was dropped; the calculator surfaces it as the
        # rejection diagnostic. A rejection with reason=None/"" would make a dropped candidate
        # undiagnosable, so every reject path must populate a non-empty reason string.
        for label, _acc_svc, _acc_pose, rej_svc, rej_pose in self._matrix:
            with self.subTest(adapter=label):
                reject = rej_svc.query(rej_pose)
                self.assertFalse(reject.reachable)
                self.assertIsInstance(reject.reason, str, f"{label}: reject reason must be a str")
                assert reject.reason is not None  # for mypy; asserted str just above
                self.assertTrue(reject.reason, f"{label}: reject reason must be non-empty")

    def test_query_is_pure_repeatable(self) -> None:
        # The Protocol mandates a *pure* oracle: querying the same pose twice must yield the same
        # reachability verdict (no hidden state flipping the decision between calls).
        for label, service, pose, expected in self._all_services():
            with self.subTest(case=label):
                first = service.query(pose).reachable
                second = service.query(pose).reachable
                self.assertEqual(first, second, f"{label}: query() is not repeatable")
                self.assertEqual(first, expected, f"{label}: verdict disagreed with the expected branch")

    def test_filter_reachable_poses_accepts_every_adapter(self) -> None:
        # End-to-end through the real consumer: filter_reachable_poses() runs each adapter and itself
        # TypeError-guards the return type, so a green run is independent proof the adapter satisfies the
        # type contract the grasp calculator depends on. The kept-pose / diagnostics lists stay parallel.
        for label, acc_svc, acc_pose, rej_svc, rej_pose in self._matrix:
            with self.subTest(adapter=label):
                kept, diags = filter_reachable_poses([acc_pose], acc_svc)
                self.assertEqual(len(diags), 1)
                self.assertEqual(kept, [acc_pose], f"{label}: in-contract pose was filtered out")
                kept_r, diags_r = filter_reachable_poses([rej_pose], rej_svc)
                self.assertEqual(len(diags_r), 1)
                self.assertEqual(kept_r, [], f"{label}: out-of-contract pose was kept")


if __name__ == "__main__":
    unittest.main()
