"""R10.1 — behavioral conformance of every ViewpointPlanner implementation to its Protocol contract.

The Protocol lives at ``backend.src.robot.grasping.loop.pick_loop.ViewpointPlanner`` (``@runtime_checkable``).
Its single method is::

    def next_viewpoint(self, *, current_tcp: Pose, history: Sequence[Pose]) -> Pose | None: ...

Every production planner wired into ``BinPickingOrchestrator``'s active-perception path must:

* satisfy the runtime-checkable Protocol (``isinstance(planner, ViewpointPlanner)``);
* return EITHER a :class:`Pose` OR ``None`` — never a bare ``bool`` / tuple / numpy array. ``None`` is the
  documented "no further viewpoint" signal the orchestrator maps to :attr:`PickOutcome.EXHAUSTED`;
* when it returns a Pose, that Pose is expressed in :attr:`Frame.BASE` (the contract pins ``current_tcp``
  and the returned next-best-view to the base frame) and carries a real numpy position vector;
* be **advisory / non-mutating** — calling :meth:`next_viewpoint` must NOT mutate the caller-owned
  ``current_tcp`` pose or the ``history`` sequence. This is the load-bearing invariant: the orchestrator
  hands the planner its live TCP and the running history, and a planner that wrote back into either would
  corrupt the loop's state rather than merely *proposing* a move;
* exhaust to ``None`` once its viewpoint budget is consumed (every shipped planner is budget-bounded), so
  the orchestrator's loop terminates.

The two implementers are constructed with the exact idioms used by the existing suites
(``tests/test_pick_loop.py`` for ``LateralOffsetViewpointPlanner`` and
``tests/test_grasp_active_perception.py`` for ``ScoringViewpointPlanner``) — both default-constructible
value objects, no fixtures required. The two-path (accept/exhaust) anchor proves the suite is not
all-accept theatre: a fresh planner MUST propose a Pose, and a budget-exhausted one MUST return ``None``.
"""

from __future__ import annotations

import copy
import unittest
from collections.abc import Sequence

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping.closed_loop.active_perception import (
    ScoringViewpointPlanner,
    ViewScoringPolicy,
)
from src.robot.grasping.loop.pick_loop import (
    LateralOffsetViewpointPlanner,
    ViewpointPlanner,
)


def _pose(xyz: tuple[float, float, float] = (100.0, 200.0, 350.0), label: str = "tcp") -> Pose:
    return Pose(
        position_mm=np.asarray(xyz, dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        frame=Frame.BASE,
        label=label,
    )


def _planners() -> tuple[ViewpointPlanner, ...]:
    """Every production ViewpointPlanner, built with its existing-test construction idiom.

    * ``LateralOffsetViewpointPlanner`` — the deterministic cardinal-offset stub from ``pick_loop``
      (constructed exactly as ``tests/test_pick_loop.py`` does).
    * ``ScoringViewpointPlanner`` — the score-and-filter planner from ``active_perception`` (constructed
      exactly as ``tests/test_grasp_active_perception.py`` does, default policy).
    """

    return (
        LateralOffsetViewpointPlanner(offset_mm=50.0, max_viewpoints=4),
        ScoringViewpointPlanner(policy=ViewScoringPolicy(max_viewpoints=4)),
    )


class ViewpointPlannerConformanceTests(unittest.TestCase):
    """Every production ViewpointPlanner honours the advisory, base-frame, exhausting Protocol contract."""

    def setUp(self) -> None:
        self._planners = _planners()
        # Guard against the suite silently degrading to a single impl after a refactor.
        self.assertGreaterEqual(
            len(self._planners), 2, "expected at least two production ViewpointPlanner impls"
        )

    def test_every_planner_satisfies_runtime_protocol(self) -> None:
        for p in self._planners:
            with self.subTest(planner=type(p).__name__):
                self.assertIsInstance(p, ViewpointPlanner)

    def test_next_viewpoint_returns_pose_or_none_in_base_frame(self) -> None:
        # Core type contract: the return is a Pose (in BASE frame, with a real 3-vector position) or None —
        # never a bare bool / array / tuple. A planner that returned, say, the raw position array or True
        # would fail here, so the suite is not satisfied by a degenerate non-Pose return.
        for p in self._planners:
            with self.subTest(planner=type(p).__name__):
                result = p.next_viewpoint(current_tcp=_pose(), history=())
                self.assertTrue(
                    result is None or isinstance(result, Pose),
                    f"{type(p).__name__}.next_viewpoint returned {type(result).__name__}, "
                    "not Pose | None",
                )
                if isinstance(result, Pose):
                    self.assertIs(
                        result.frame,
                        Frame.BASE,
                        "next-best-view pose must be expressed in Frame.BASE",
                    )
                    position = np.asarray(result.position_mm, dtype=np.float64)
                    self.assertEqual(position.shape, (3,))
                    self.assertTrue(np.all(np.isfinite(position)))

    def test_next_viewpoint_never_raises_on_well_formed_input(self) -> None:
        # The orchestrator calls next_viewpoint inside its recovery loop; a planner that raised on a
        # well-formed TCP + history would crash the pick instead of advising it. Exercise both the
        # empty-history (first call) and the non-empty-history (mid-session) shapes.
        contexts = {
            "empty_history": (_pose(), ()),
            "with_history": (
                _pose(),
                (_pose((150.0, 200.0, 350.0)), _pose((50.0, 200.0, 350.0))),
            ),
        }
        for label, (current_tcp, history) in contexts.items():
            for p in self._planners:
                with self.subTest(planner=type(p).__name__, context=label):
                    try:
                        result = p.next_viewpoint(current_tcp=current_tcp, history=history)
                    except Exception as exc:  # noqa: BLE001 - the point is to prove no planner raises
                        self.fail(
                            f"{type(p).__name__} raised {type(exc).__name__} on the {label} context "
                            f"instead of returning Pose | None: {exc!r}"
                        )
                    self.assertTrue(result is None or isinstance(result, Pose))

    def test_next_viewpoint_is_advisory_and_does_not_mutate_inputs(self) -> None:
        # THE advisory / non-mutating invariant. The planner is handed the caller's live TCP and the
        # running history; it must only PROPOSE. We snapshot the TCP's position/quaternion and a deep copy
        # of the history, call next_viewpoint, and assert the inputs are byte-for-byte unchanged. A planner
        # that wrote back into current_tcp or appended to history (corrupting the orchestrator's state)
        # would fail here.
        for p in self._planners:
            with self.subTest(planner=type(p).__name__):
                current_tcp = _pose()
                tcp_pos_before = np.array(current_tcp.position_mm, dtype=np.float64, copy=True)
                tcp_quat_before = np.array(current_tcp.quaternion_xyzw, dtype=np.float64, copy=True)
                history: list[Pose] = [_pose((150.0, 200.0, 350.0), label="v0")]
                history_before = copy.deepcopy(history)

                p.next_viewpoint(current_tcp=current_tcp, history=tuple(history))

                np.testing.assert_array_equal(
                    np.asarray(current_tcp.position_mm),
                    tcp_pos_before,
                    err_msg=f"{type(p).__name__} mutated current_tcp.position_mm",
                )
                np.testing.assert_array_equal(
                    np.asarray(current_tcp.quaternion_xyzw),
                    tcp_quat_before,
                    err_msg=f"{type(p).__name__} mutated current_tcp.quaternion_xyzw",
                )
                self.assertEqual(
                    len(history), len(history_before), f"{type(p).__name__} mutated the history list"
                )
                for got, want in zip(history, history_before, strict=True):
                    np.testing.assert_array_equal(
                        np.asarray(got.position_mm), np.asarray(want.position_mm)
                    )

    def test_proposes_a_pose_then_exhausts_to_none(self) -> None:
        # Non-vacuity anchor: BOTH decision paths are genuinely exercised per impl. A fresh planner with a
        # positive budget MUST propose a Pose (accept path); after the budget is consumed it MUST return
        # None (exhaust path). This rules out an all-accept (always returns a Pose) OR an all-refuse
        # (always returns None) impl passing the suite.
        for p in self._planners:
            with self.subTest(planner=type(p).__name__):
                history: list[Pose] = []
                first = p.next_viewpoint(current_tcp=_pose(), history=tuple(history))
                self.assertIsInstance(
                    first,
                    Pose,
                    f"{type(p).__name__} did not propose any viewpoint from a fresh budget",
                )
                assert isinstance(first, Pose)  # narrow for the static checker
                history.append(first)
                # Drain the remaining budget; both impls are bounded by max_viewpoints=4.
                exhausted: Pose | None = first
                for _ in range(8):
                    exhausted = p.next_viewpoint(current_tcp=_pose(), history=tuple(history))
                    if exhausted is None:
                        break
                    history.append(exhausted)
                self.assertIsNone(
                    exhausted,
                    f"{type(p).__name__} never exhausted to None within its budget — the loop "
                    "would not terminate",
                )

    def test_history_typing_contract(self) -> None:
        # The signature types ``history`` as a Sequence[Pose]; pin that the planners accept the two shapes
        # the orchestrator actually passes (an empty tuple and a populated tuple) without special-casing a
        # concrete container type.
        for p in self._planners:
            with self.subTest(planner=type(p).__name__):
                empty: Sequence[Pose] = ()
                populated: Sequence[Pose] = (_pose((150.0, 200.0, 350.0)),)
                self.assertTrue(
                    p.next_viewpoint(current_tcp=_pose(), history=empty) is None
                    or isinstance(p.next_viewpoint(current_tcp=_pose(), history=empty), Pose)
                )
                self.assertTrue(
                    p.next_viewpoint(current_tcp=_pose(), history=populated) is None
                    or isinstance(
                        p.next_viewpoint(current_tcp=_pose(), history=populated), Pose
                    )
                )


if __name__ == "__main__":
    unittest.main()
