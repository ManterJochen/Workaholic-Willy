"""A sort's pick prompt grounds every rule's kind in one call, and the pick loop's label gate passes those kinds alone.

The owner's sort (2026-10-09): "Grüne Teile in die gelbe Kiste, rote in die blaue". The pick prompt is a class list,
``"each separate green part | each separate red part"`` (``class_list_prompt``); the camera source maps each box's
words onto the kinds (``object_labels=("green part", "red part")``); the pick loop takes any of them
(``target_labels``, with ``target_label`` ``None``). What no rule clearly claims is no target and stays a neighbour: an
object the model boxed under both descriptions, which the parser made ``ambiguous``, and a word no rule names ("orange
part"). The report says which kind the pick went for (``target_label``) and what no rule claimed at its first look
(``unclaimed_labels``), on the service's report too.

End to end: the real grounder and parser over a model double, the real two-stage backend over a box-cutting segmenter,
the real camera source with its colour check on painted parts, and the real pick loop and service, with a calculator
that records what it was asked to grasp. No GPU, no weights, no camera.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose
from src.models.perception_backend import TwoStageBackend
from src.models.vlm.parsing import AMBIGUOUS_LABEL
from src.models.vlm.qwen import class_list_prompt, classes_of
from src.robot.core import MotionCommand, MotionResult, RobotCapabilities
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome, AutonomousGraspService, GraspMode
from src.robot.execution.autonomous_grasp.prompt import PickPrompt
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.perception import RealSenseVisionPerceptionSource
from tests.test_a_locate_of_a_class_list_finds_every_bin_in_one_call import _BoxSegmenter, _grounder

H, W = 40, 60
_K = np.array([[400.0, 0.0, 30.0], [0.0, 400.0, 20.0], [0.0, 0.0, 1.0]])

KINDS = ("green part", "red part")
PROMPT = class_list_prompt(["each separate green part", "each separate red part"])

#: Where the parts lie in the image, pixel boxes (x0, y0, x1, y1), each painted its colour (BGR) on a dark mat.
GREEN = (4, 4, 18, 18)
RED = (24, 4, 38, 18)
BOTH = (4, 22, 18, 36)
ORANGE = (24, 22, 38, 36)
PAINT = {GREEN: (40, 180, 40), RED: (40, 40, 200), BOTH: (0, 200, 200), ORANGE: (0, 120, 255)}


def _grid_answer(*boxes: tuple[tuple[int, int, int, int], str]) -> str:
    """A grounding answer in Qwen's 0-1000 grid for pixel boxes of this frame."""
    items = []
    for (x0, y0, x1, y1), label in boxes:
        grid = [round(x0 / W * 1000), round(y0 / H * 1000), round(x1 / W * 1000), round(y1 / H * 1000)]
        items.append(f'{{"bbox_2d": {grid}, "label": "{label}"}}')
    return "[" + ", ".join(items) + "]"


#: What the model double answers: each kind under its description, one part boxed under both (a box a pixel shorter
#: under the second), and a part under a word no rule names.
ANSWER = _grid_answer((GREEN, "each separate green part"), (RED, "each separate red part"),
                      (BOTH, "each separate green part"), ((4, 23, 18, 36), "each separate red part"),
                      (ORANGE, "orange part"))


class _Streamer:
    """One fixed RGB-D frame: the mat at 500 mm, each part 40 mm up and painted."""

    def __init__(self) -> None:
        depth = np.full((H, W), 500, dtype=np.uint16)
        colour = np.full((H, W, 3), 30, dtype=np.uint8)
        for (x0, y0, x1, y1), bgr in PAINT.items():
            depth[y0:y1, x0:x1] = 460
            colour[y0:y1, x0:x1] = bgr
        self.frame = RGBDFrame(color=colour, depth=depth)

    def grab(self) -> RGBDFrame:
        return self.frame

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


def _source(prompt: str = PROMPT, object_labels: tuple[str, ...] = KINDS) -> tuple[RealSenseVisionPerceptionSource,
                                                                                 Any]:
    grounder, model = _grounder(ANSWER)
    backend = TwoStageBackend(detector=grounder, segmenter=_BoxSegmenter())
    source = RealSenseVisionPerceptionSource(streamer=_Streamer(), backend=backend, prompt=prompt,
                                             object_labels=object_labels, warmup_grabs=0)
    return source, model


class _Recording:
    """Grasps every part it is asked about, in BASE, and keeps the label of each, in order; the red part ranks
    higher, so a pick that may take either takes the red one."""

    render_debug_images = False

    def __init__(self) -> None:
        self.asked: list[str] = []

    def compute_result(self, seg: Any, *_args: Any, **_kwargs: Any) -> GraspResult:
        label = str(getattr(seg, "label", ""))
        self.asked.append(label)
        score = 0.9 if label == "red part" else 0.6
        grasp = GraspPoint(position=np.array([100.0, 50.0, 400.0]), approach=np.array([0.0, 0.0, 1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=score, frame=GraspFrame.BASE,
                           label=label)
        return GraspResult(candidates=(grasp,), top_score=score)


class _Policy:
    def __init__(self) -> None:
        self.executed: list[GraspPoint] = []

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        self.executed.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


class TheSourceLabelsEachBoxByItsKindTests(unittest.TestCase):
    def test_one_call_labels_the_kinds_and_what_no_rule_claims(self) -> None:
        source, model = _source()
        frame = source.acquire()
        self.assertEqual(1, len(model.calls), "every kind in one call")
        self.assertEqual(("each separate green part", "each separate red part"), classes_of(PROMPT))
        self.assertEqual(["green part", "red part", AMBIGUOUS_LABEL, AMBIGUOUS_LABEL, "orange part"],
                         [seg.label for seg in frame.segmentations],
                         "each kind mapped onto its rule, the doubly claimed part and the other word onto none")


class TheGatePassesOnlyTheKindsTests(unittest.TestCase):
    def _run(self) -> tuple[Any, _Recording, _Policy]:
        source, _ = _source()
        calculator, policy = _Recording(), _Policy()
        orchestrator = BinPickingOrchestrator(arm=MagicMock(), calculator=calculator, perception=source,
                                              policy=policy, max_attempts=1, target_labels=KINDS)  # type: ignore[arg-type]
        return orchestrator.run(), calculator, policy

    def test_both_kinds_are_targets_and_nothing_else_is(self) -> None:
        report, calculator, policy = self._run()
        self.assertEqual(["green part", "red part"], calculator.asked, "only the two kinds were asked to be grasped")
        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(["red part"], [grasp.label for grasp in policy.executed])

    def test_the_report_names_the_kind_it_took_and_what_no_rule_claimed(self) -> None:
        report, _, _ = self._run()
        self.assertEqual("red part", report.target_label)
        self.assertEqual((AMBIGUOUS_LABEL, "orange part"), report.unclaimed_labels,
                         "one entry per part no rule claims, in the camera's order: the part both descriptions "
                         "boxed is one part")


# ---------------------------------------------------------------------------------------------------------------------
# The service: set_prompt and its round trip, and the report a sort reads
# ---------------------------------------------------------------------------------------------------------------------

_CAPS = RobotCapabilities(
    vendor="ur", model="ur5e", dof=6, supports_joint_move=True, supports_linear_move=True,
    supports_async_move=False, has_native_fk=True, has_native_ik=True, has_force_control=False,
    is_simulated=False,
)


class _Arm:
    """Moves wherever it is told and says so."""

    def __init__(self) -> None:
        self._tcp = Pose(position_mm=np.array([0.0, 0.0, 500.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                         frame=Frame.BASE, label="home")

    @property
    def capabilities(self) -> RobotCapabilities:
        return _CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


def _service() -> tuple[AutonomousGraspService, RealSenseVisionPerceptionSource]:
    """A service over the live camera source, built with the console's phrase and no object labels."""
    source, _ = _source(prompt="object", object_labels=())
    service = AutonomousGraspService.from_components(
        arm=_Arm(), calculator=_Recording(), perception=source, mode=GraspMode.EASY,  # type: ignore[arg-type]
    )
    return service, source


SORT = PickPrompt(phrase=PROMPT, object_labels=KINDS, target_labels=KINDS)


class SetPromptSetsTheKindsTests(unittest.TestCase):
    def test_a_sort_sets_its_kinds_beside_the_phrase_and_the_labels(self) -> None:
        service, source = _service()
        orchestrator = service.runtime.orchestrator

        previous = service.set_prompt(SORT)

        self.assertEqual((PROMPT, KINDS), (source.prompt, source.object_labels))
        self.assertEqual((None, KINDS), (orchestrator.target_label, orchestrator.target_labels))
        self.assertEqual(PickPrompt(phrase="object", target_label=None, object_labels=(), target_labels=()), previous)

    def test_the_prompt_it_returned_puts_everything_back(self) -> None:
        service, source = _service()
        orchestrator = service.runtime.orchestrator

        replaced = service.set_prompt(service.set_prompt(SORT))

        self.assertEqual(SORT, replaced, "what the sort set is what passing the old prompt back replaced")
        self.assertEqual(("object", (), None, ()), (source.prompt, source.object_labels, orchestrator.target_label,
                                                    orchestrator.target_labels))

    def test_a_typed_prompt_after_a_sort_hands_the_kinds_back(self) -> None:
        """The previous prompt carries the kinds the pick loop took, so a campaign that types its own prompt during a
        sort and puts the old one back leaves the sort as it found it."""
        service, source = _service()
        orchestrator = service.runtime.orchestrator
        service.set_prompt(SORT)

        previous = service.set_prompt("the red part")

        self.assertEqual(KINDS, previous.target_labels)
        self.assertEqual(("the red part", ()), (orchestrator.target_label, orchestrator.target_labels),
                         "a typed prompt is one kind's, its label alone")
        service.set_prompt(previous)
        self.assertEqual((PROMPT, KINDS, None, KINDS), (source.prompt, source.object_labels,
                                                        orchestrator.target_label, orchestrator.target_labels))

    def test_set_target_label_leaves_the_kinds_alone(self) -> None:
        service, _ = _service()
        service.set_prompt(SORT)
        service.set_target_label(None)
        self.assertEqual(KINDS, service.runtime.orchestrator.target_labels)


class TheServicesReportCarriesTheKindTests(unittest.TestCase):
    def test_a_sorted_pick_names_its_kind_and_what_no_rule_claimed_on_the_service_report(self) -> None:
        service, _ = _service()
        service.set_prompt(SORT)

        report = service.pick()

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.render())
        assert report.pick_report is not None
        self.assertEqual("red part", report.pick_report.target_label)
        self.assertEqual((AMBIGUOUS_LABEL, "orange part"), report.pick_report.unclaimed_labels)
        wire = report.to_dict()
        self.assertEqual("red part", wire["target_label"])
        self.assertEqual([AMBIGUOUS_LABEL, "orange part"], wire["unclaimed_labels"])

    def test_a_one_kind_pick_names_its_label_and_nothing_unclaimed(self) -> None:
        service, _ = _service()
        service.set_prompt(PickPrompt(phrase="each separate red part", target_label="red part",
                                      object_labels=("red part",)))

        report = service.pick()

        assert report.pick_report is not None
        self.assertEqual("red part", report.pick_report.target_label)
        self.assertEqual((), report.pick_report.unclaimed_labels, "a pick that sorts nothing names nothing unclaimed")
        self.assertEqual([], report.to_dict()["unclaimed_labels"])


class OnePartIsCountedOnceTests(unittest.TestCase):
    def test_the_pick_loop_reads_two_masks_as_one_part_where_the_class_list_does(self) -> None:
        from src.models.vlm.parsing import AMBIGUOUS_IOU
        from src.robot.grasping.loop.pick_loop import _SAME_OBJECT_IOU

        self.assertEqual(AMBIGUOUS_IOU, _SAME_OBJECT_IOU)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
