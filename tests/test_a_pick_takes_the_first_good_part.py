"""A pick computes a look's parts one at a time and takes the first good one (the owner's "speed first", 2026-10-08).

The pick loop used to compute every part of a look before it chose one: on the owner's cell one tilted cube cost 24 s,
most of it SFE's fine search, and the part taken cost 3. With ``lazy_objects`` (``robot.grasping.first_good_part``) the
parts are computed in the order they stand apart, those the hand closes across first and the less crowded first, and
the first whose result is full, carries no rescan reason and scores ``accept_score`` (``good_part_score``) or more is
taken; the rest are not computed. None good, the best is taken. With ``fine_search_waits``
(``robot.grasping.fine_pass_waits``) a part is computed without SFE's fine search, but for the last where no part has a
full result yet; a part whose coarse grid found few grasps waits, and gets the fine search only where no part has a full
result with a grasp, when the choice is the one of before. A frame with no grasp is computed whole and fails as it
always did, whatever the order; a later look of a wrist pick computes only the part it keeps; the clutter selector still
computes every part. Both default off on the loop itself, so every direct construction ranks as before.

The calculators here answer each part by its name, as the real one answers it, and record what they were asked.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot.grasping_schema import GraspingSupportConfig
from src.geometry import Frame, Transform
from src.robot.grasping.generation._support_footprint_stage import FINE_SEARCH_DEFERRED, FINE_SEARCH_KEY
from src.robot.grasping.geometry.closing_axis import CLOSING_AXIS_REFUSED
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.loop.target_selector import BlockerGraphConfig, OrderingReason, TargetOrderingConfig
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame
from tests._helpers import _FakeArm, _FakePerception
from tests._wrist_views import CUBE, K, Box, render

NOTHING = (GraspFailureReason.NO_VALID_GRASP, GraspFailureReason.RESCAN_RECOMMENDED)


@dataclass(eq=False)
class _Part:
    mask: np.ndarray
    name: str
    label: str = "part"
    score: float = 0.9


def _frame(*names: str) -> PerceptionFrame:
    """One part per name, side by side in a 40 px square each, in the order the camera lists them."""
    depth = np.full((40, 40 * len(names)), 500.0)
    parts = []
    for number, name in enumerate(names):
        mask = np.zeros(depth.shape, dtype=bool)
        mask[10:30, 40 * number + 10:40 * number + 30] = True
        parts.append(_Part(mask=mask, name=name))
    return PerceptionFrame(depth_map=depth, intrinsics=K.copy(), segmentations=tuple(parts))


class _Parts:
    """Answers each part by its name: ``full`` is the best score its full search finds, ``None`` for no grasp; ``coarse``
    what its coarse grid alone finds where the fine search waits (a part not in it finds enough on its coarse grid);
    ``reasons`` what a part's result carries. ``calls`` keeps every part asked and whether its fine search ran."""

    render_debug_images = False

    def __init__(self, full: "dict[str, float | None]", coarse: "dict[str, float | None] | None" = None, *,
                 reasons: "dict[str, tuple[GraspFailureReason, ...]] | None" = None) -> None:
        self.full = full
        self.coarse = coarse or {}
        self.reasons = reasons or {}
        self.sfe_fine_pass = True
        self.calls: list[tuple[str, bool]] = []

    def compute_result(self, seg: Any, *_args: Any, **_kwargs: Any) -> GraspResult:
        self.calls.append((seg.name, self.sfe_fine_pass))
        waited = not self.sfe_fine_pass and seg.name in self.coarse
        score = self.coarse[seg.name] if waited else self.full[seg.name]
        telemetry: dict[str, Any] = {FINE_SEARCH_KEY: FINE_SEARCH_DEFERRED} if waited else {}
        if score is None:
            return GraspResult(reasons=self.reasons.get(seg.name, NOTHING), telemetry=telemetry)
        grasp = GraspPoint(position=np.array([0.0, 0.0, 400.0]), approach=np.array([0.0, 0.0, 1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=score, frame=GraspFrame.BASE,
                           label=seg.name)
        return GraspResult(candidates=(grasp,), reasons=self.reasons.get(seg.name, ()), top_score=score,
                           telemetry=telemetry)

    def asked(self) -> list[str]:
        return [name for name, _ in self.calls]


class _Policy:
    """Executes nothing and says it executed: which part's grasp it was handed is what these tests read."""

    def __init__(self) -> None:
        self.executed: list[GraspPoint] = []

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        self.executed.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


def _pick(calculator: Any, frame: PerceptionFrame, **switches: Any) -> "tuple[BinPickingOrchestrator, _Policy]":
    policy = _Policy()
    orchestrator = BinPickingOrchestrator(arm=_FakeArm(), calculator=calculator, perception=_FakePerception([frame]),
                                          policy=policy, max_attempts=1, **switches)  # type: ignore[arg-type]
    return orchestrator, policy


def _taken(policy: _Policy) -> str:
    (grasp,) = policy.executed
    return str(grasp.label)


class TheFirstGoodPartIsTakenTests(unittest.TestCase):
    def test_the_first_good_part_is_taken_and_the_others_are_not_computed(self) -> None:
        calculator = _Parts({"a": 0.9, "b": 0.95, "c": 0.8})
        orchestrator, policy = _pick(calculator, _frame("a", "b", "c"), lazy_objects=True)

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual("a", _taken(policy))
        self.assertEqual(["a"], calculator.asked())

    def test_below_the_bar_the_next_part_is_computed_and_the_best_is_taken(self) -> None:
        calculator = _Parts({"a": 0.6, "b": 0.7, "c": 0.5})
        orchestrator, policy = _pick(calculator, _frame("a", "b", "c"), lazy_objects=True)

        orchestrator.run()

        self.assertEqual(["a", "b", "c"], calculator.asked())
        self.assertEqual("b", _taken(policy))

    def test_a_part_with_a_rescan_reason_is_not_good_enough(self) -> None:
        calculator = _Parts({"a": 0.9, "b": 0.8}, reasons={"a": (GraspFailureReason.RESCAN_RECOMMENDED,)})
        orchestrator, policy = _pick(calculator, _frame("a", "b"), lazy_objects=True)

        orchestrator.run()

        self.assertEqual(["a", "b"], calculator.asked())
        self.assertEqual("b", _taken(policy))

    def test_the_bar_is_the_accept_score(self) -> None:
        calculator = _Parts({"a": 0.9, "b": 0.95})
        orchestrator, policy = _pick(calculator, _frame("a", "b"), lazy_objects=True, accept_score=0.92)

        orchestrator.run()

        self.assertEqual(["a", "b"], calculator.asked())
        self.assertEqual("b", _taken(policy))

    def test_off_every_part_is_computed_and_the_best_taken_as_before(self) -> None:
        calculator = _Parts({"a": 0.9, "b": 0.95, "c": 0.8})
        orchestrator, policy = _pick(calculator, _frame("a", "b", "c"))

        orchestrator.run()

        self.assertEqual(["a", "b", "c"], calculator.asked())
        self.assertEqual([("a", True), ("b", True), ("c", True)], calculator.calls)
        self.assertEqual("b", _taken(policy))
        self.assertEqual((False, 0.75, False),
                         (orchestrator.lazy_objects, orchestrator.accept_score, orchestrator.fine_search_waits))


class TheFineSearchWaitsTests(unittest.TestCase):
    def test_a_part_whose_coarse_grid_finds_few_waits_while_another_part_is_full(self) -> None:
        calculator = _Parts({"a": 0.7, "b": 0.8}, coarse={"a": None})
        orchestrator, policy = _pick(calculator, _frame("a", "b"), lazy_objects=True, fine_search_waits=True)

        orchestrator.run()

        self.assertEqual([("a", False), ("b", True)], calculator.calls)
        self.assertEqual("b", _taken(policy))

    def test_a_waiting_part_stays_waiting_where_a_full_part_is_below_the_bar(self) -> None:
        calculator = _Parts({"a": 0.7, "b": 0.6}, coarse={"a": 0.55})
        orchestrator, policy = _pick(calculator, _frame("a", "b"), lazy_objects=True, fine_search_waits=True)

        orchestrator.run()

        self.assertEqual([("a", False), ("b", True)], calculator.calls)
        self.assertEqual("b", _taken(policy), "a part's coarse grasps were taken over a full result")

    def test_with_no_full_part_the_parts_that_waited_get_the_fine_search_and_the_choice_of_before(self) -> None:
        for lazy in (False, True):
            with self.subTest(lazy_objects=lazy):
                calculator = _Parts({"a": 0.7, "b": None, "c": 0.65}, coarse={"a": None, "c": None})
                orchestrator, policy = _pick(calculator, _frame("a", "c", "b"), lazy_objects=lazy,
                                             fine_search_waits=True)

                orchestrator.run()

                self.assertEqual([("a", False), ("c", False), ("b", True), ("a", True), ("c", True)],
                                 calculator.calls)
                self.assertEqual("a", _taken(policy))

    def test_the_last_part_is_never_left_waiting(self) -> None:
        calculator = _Parts({"a": 0.7}, coarse={"a": None})
        orchestrator, policy = _pick(calculator, _frame("a"), lazy_objects=True, fine_search_waits=True)

        orchestrator.run()

        self.assertEqual([("a", True)], calculator.calls, "a single part did not get its full search at once")
        self.assertEqual("a", _taken(policy))

    def test_the_switch_is_given_back_when_the_calculator_raises(self) -> None:
        calculator = _Parts({"a": 0.9, "b": 0.9})

        def raises(*_args: Any, **_kwargs: Any) -> GraspResult:
            raise RuntimeError("the calculator fell over")

        calculator.compute_result = raises  # type: ignore[method-assign]
        orchestrator, _ = _pick(calculator, _frame("a", "b"), lazy_objects=True, fine_search_waits=True)

        with self.assertRaises(RuntimeError):
            orchestrator._best_result_over_segmentations(_frame("a", "b"))
        self.assertIs(True, calculator.sfe_fine_pass)

    def test_a_calculator_without_the_switch_computes_every_part_in_full(self) -> None:
        class NoSwitch:
            """A calculator with no fine search to hold back, a deep one say."""

            render_debug_images = False

            def __init__(self) -> None:
                self.asked: list[str] = []

            def compute_result(self, seg: Any, *args: Any, **kwargs: Any) -> GraspResult:
                self.asked.append(seg.name)
                return _Parts({"a": 0.6, "b": 0.7}).compute_result(seg, *args, **kwargs)

        calculator = NoSwitch()
        orchestrator, policy = _pick(calculator, _frame("a", "b"), lazy_objects=True, fine_search_waits=True)

        orchestrator.run()

        self.assertEqual(["a", "b"], calculator.asked)
        self.assertFalse(hasattr(calculator, "sfe_fine_pass"), "the loop gave a calculator a switch it does not have")
        self.assertEqual("b", _taken(policy))


class WithNoGraspEveryPartIsComputedTests(unittest.TestCase):
    @staticmethod
    def _failing() -> _Parts:
        """Part ``axis`` was left no grasp by the closing axis; part ``boxed`` collided with every grasp."""
        calculator = _Parts({"axis": None, "boxed": None}, reasons={
            "axis": (GraspFailureReason.NO_VALID_GRASP,),
            "boxed": (GraspFailureReason.ALL_COLLIDED, GraspFailureReason.RESCAN_RECOMMENDED)})
        plain = calculator.compute_result

        def answers(seg: Any, *args: Any, **kwargs: Any) -> GraspResult:
            result = plain(seg, *args, **kwargs)
            if seg.name == "axis":
                result.telemetry[CLOSING_AXIS_REFUSED] = "closing_axis -y: none of the 1 grasp candidate(s) closes"
            return result

        calculator.compute_result = answers  # type: ignore[method-assign]
        return calculator

    def test_the_failure_is_the_one_of_before_whatever_the_order(self) -> None:
        for names in (("axis", "boxed"), ("boxed", "axis")):
            for reverse in (False, True):
                with self.subTest(camera_order=names, computed_backwards=reverse):
                    before = self._failing()
                    orchestrator, _ = _pick(before, _frame(*names))
                    expected = orchestrator._best_result_over_segmentations(_frame(*names))
                    calculator = self._failing()
                    orchestrator, _ = _pick(calculator, _frame(*names), lazy_objects=True, fine_search_waits=True)
                    real = BinPickingOrchestrator._isolation_key

                    def key(self_: Any, ctx: Any, obj: Any, support: Any, surface: Any, idx: int) -> Any:
                        found = real(self_, ctx, obj, support, surface, idx)
                        return (found[0], found[1], -idx) if reverse else found

                    with mock.patch.object(BinPickingOrchestrator, "_isolation_key", key):
                        result, index, ordering = orchestrator._best_result_over_segmentations(_frame(*names))
                    self.assertEqual(sorted(names), sorted(calculator.asked()), "a part was left uncomputed")
                    self.assertEqual(expected[1], index)
                    self.assertIsNotNone(result)
                    assert result is not None and expected[0] is not None
                    self.assertEqual(expected[0].reasons, result.reasons)
                    self.assertEqual(names.index("boxed"), index, "the part the axis refused is the frame's failure")
                    self.assertIsNone(ordering)


class TheClutterSelectorTests(unittest.TestCase):
    def test_with_the_selector_on_every_part_is_computed(self) -> None:
        from tests.test_t4_pick_loop_ordering import _clutter_frame, _IndexedCalculator

        calculator = _IndexedCalculator([0.55, 0.6, 0.58])
        selector = TargetOrderingConfig(enabled=True, unlock_weight=0.5, max_local_score_drop=0.2,
                                        blocker_graph=BlockerGraphConfig(mask_adjacency_enabled=True,
                                                                         depth_only_enabled=True,
                                                                         adjacency_radius_px=3,
                                                                         depth_tolerance_mm=50.0))
        orchestrator = BinPickingOrchestrator(arm=_FakeArm(), calculator=calculator,  # type: ignore[arg-type]
                                              perception=_FakePerception([_clutter_frame()]), max_attempts=1,
                                              target_ordering=selector, lazy_objects=True, fine_search_waits=True)

        report = orchestrator.run()

        self.assertEqual(3, calculator.calls)
        decision = report.attempts[0].ordering_decision
        assert decision is not None
        self.assertEqual((0, OrderingReason.UNLOCK_SWAP), (report.attempts[0].target_index, decision.reason))


# ---------------------------------------------------------------------------------------------------------------------
# The order the parts stand apart in: a fixed camera over the bench, ray cast
# ---------------------------------------------------------------------------------------------------------------------


def _looking_down(x_mm: float, y_mm: float, height_mm: float = 700.0) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = (x_mm, y_mm, height_mm)
    return matrix


CAMERA = _looking_down(0.0, -700.0)
#: A 60 mm cube, wider both ways than the Hand-E closes on; a 30 mm part with a tall block 10 mm off it; a 30 mm part
#: standing alone. The camera lists them in that order, the block last.
WIDE = Box((100.0, -730.0, 0.0), (160.0, -670.0, 40.0), "part")
CROWDED = Box((-15.0, -715.0, 0.0), (15.0, -685.0, 40.0), "part")
BLOCK = Box((25.0, -740.0, 0.0), (65.0, -660.0, 60.0), "block")
ALONE = Box((-130.0, -715.0, 0.0), (-100.0, -685.0, 40.0), "part")


@dataclass(eq=False)
class _Seen:
    mask: np.ndarray
    label: str
    name: str
    score: float = 0.9


class _BenchCamera:
    def __init__(self, boxes: "dict[str, Box]") -> None:
        self.boxes = boxes

    def acquire(self) -> PerceptionFrame:
        depth, hit = render(CAMERA, tuple(self.boxes.values()))
        seen = tuple(_Seen(mask=hit == index, label=box.label, name=name)
                     for index, (name, box) in enumerate(self.boxes.items()) if np.any(hit == index))
        return PerceptionFrame(depth_map=depth, intrinsics=K.copy(), segmentations=seen, timestamp=1.0)


class ThePartsAreComputedInTheOrderTheyStandApartTests(unittest.TestCase):
    def test_a_part_wider_than_the_stroke_goes_last_and_the_less_crowded_part_first(self) -> None:
        calculator = _Parts({"wide": 0.55, "crowded": 0.6, "alone": 0.5})
        calculator.max_grip_mm = 49.99  # type: ignore[attr-defined]
        camera = _BenchCamera({"wide": WIDE, "crowded": CROWDED, "alone": ALONE, "block": BLOCK})
        policy = _Policy()
        orchestrator = BinPickingOrchestrator(
            arm=_FakeArm(), calculator=calculator, perception=camera, policy=policy,  # type: ignore[arg-type]
            max_attempts=1, target_label="part", lazy_objects=True,
            frame_resolver=StaticCameraToBaseResolver(transform=Transform.from_matrix(
                CAMERA, from_frame=Frame.CAMERA, to_frame=Frame.BASE)),
            support_config=GraspingSupportConfig(height_mm=0.0, refine_from_target=False))
        keys: dict[int, Any] = {}
        real = BinPickingOrchestrator._isolation_key

        def kept(self_: Any, ctx: Any, obj: Any, support: Any, surface: Any, idx: int) -> Any:
            keys[idx] = real(self_, ctx, obj, support, surface, idx)
            return keys[idx]

        with mock.patch.object(BinPickingOrchestrator, "_isolation_key", kept):
            orchestrator.run()

        self.assertEqual(["alone", "crowded", "wide"], calculator.asked(), keys)
        self.assertEqual("crowded", _taken(policy), "none was good: the best was not taken")
        wide, crowded, alone = keys[0], keys[1], keys[2]
        self.assertEqual((True, False, False), (wide[0], crowded[0], alone[0]), keys)
        self.assertEqual(0.0, alone[1], keys)
        self.assertGreater(crowded[1], 0.0, keys)

    def test_what_cannot_be_read_says_nothing_and_the_cameras_order_stands(self) -> None:
        calculator = _Parts({"a": 0.5, "b": 0.6, "c": 0.55})
        orchestrator, _ = _pick(calculator, _frame("c", "a", "b"), lazy_objects=True)

        orchestrator.run()

        self.assertEqual(["c", "a", "b"], calculator.asked())

    def test_an_order_that_raises_keeps_the_cameras_and_costs_no_pick(self) -> None:
        calculator = _Parts({"a": 0.5, "b": 0.6, "c": 0.55})
        orchestrator, policy = _pick(calculator, _frame("c", "a", "b"), lazy_objects=True)

        with mock.patch.object(BinPickingOrchestrator, "_fits_the_stroke", side_effect=RuntimeError("unreadable")):
            report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(["c", "a", "b"], calculator.asked())
        self.assertEqual("b", _taken(policy))


# ---------------------------------------------------------------------------------------------------------------------
# A later look of a wrist pick computes only the part it keeps
# ---------------------------------------------------------------------------------------------------------------------


class ALaterLookComputesOnlyThePinnedPartTests(unittest.TestCase):
    """The first look finds the part's grasp uncertain; the second judges that part alone (``_judge``)."""

    @staticmethod
    def _cell(looks: tuple[Any, ...], boxes: tuple[Box, ...], *, lazy: bool) -> Any:
        from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import _Cell

        cell = _Cell(looks=looks, boxes=boxes, uncertain_on=(0,))
        cell.orchestrator.lazy_objects = lazy
        cell.orchestrator.fine_search_waits = lazy
        return cell

    def test_a_later_look_computes_only_the_part_it_keeps(self) -> None:
        from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import LOOK_PLUS_X, LOOK_PLUS_Y

        beside = Box((60.0, -760.0, 0.0), (100.0, -720.0, 40.0), "block")
        computed: dict[bool, set[str]] = {}
        for lazy in (False, True):
            cell = self._cell((LOOK_PLUS_X, LOOK_PLUS_Y), (CUBE, beside), lazy=lazy)

            report = cell.run()

            self.assertIs(PickOutcome.EXECUTED, report.outcome)
            self.assertEqual(2, len(cell.camera.taken))
            computed[lazy] = {label for number, label, _ in cell.calculator.calls if number == 1}
        self.assertEqual({"part", "block"}, computed[False], "the second look was meant to see the block too")
        self.assertEqual({"part"}, computed[True])

    def test_a_later_look_that_does_not_see_the_kept_part_computes_nothing(self) -> None:
        from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import LOOK_ELSEWHERE, LOOK_PLUS_X

        elsewhere = Box((-20.0, -320.0, 0.0), (20.0, -280.0, 40.0), "block")
        computed: dict[bool, set[str]] = {}
        for lazy in (False, True):
            cell = self._cell((LOOK_PLUS_X, LOOK_ELSEWHERE), (CUBE, elsewhere), lazy=lazy)

            cell.run()

            computed[lazy] = {label for number, label, _ in cell.calculator.calls if number == 1}
        self.assertEqual({"block"}, computed[False], "the second look was meant to see the block alone")
        self.assertEqual(set(), computed[True])


if __name__ == "__main__":
    unittest.main()
