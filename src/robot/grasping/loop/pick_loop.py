"""
Bin-picking orchestrator.

This module closes the loop that the calculator exposes via
:class:`GraspResult.reasons`. The calculator reports why a pick failed
(no candidates, every candidate rejected by collision, workspace or IK,
rescan recommended, ...); the orchestrator turns those codes into actions:

* A reason in ``_RESCAN_REASONS`` (``RESCAN_RECOMMENDED``,
  ``ACTIVE_PERCEPTION_RECOMMENDED``, ``ALL_OUT_OF_WORKSPACE`` and the
  no-candidate and weak-perception reasons): request a fresh perception
  frame from the perception source without moving the robot, within
  ``max_attempts``.
* Anything else: terminate with the corresponding outcome.

With an IK service wired into the calculator, its IK filter drops each
candidate the IK cannot reach before the result comes back; a result whose
every candidate failed IK carries ``IK_FAILED`` beside ``RESCAN_RECOMMENDED``
and is rescanned. A grasp a guard or the planner refused before anything was
sent is followed by the next grasp of the same look, up to
:data:`GRASPS_TRIED_PER_ATTEMPT` in all, while every motion the try sent kept
the open hand's jaw region out of the held target's keep-out box (the owner's
"try the next grasps", cell fixes Track C): a stop is asked for, the controller
read and a wrist pick driven back to its look on a judged move before each.
An attempt whose every try was so refused, none at the part, carries
``MOTION_PLAN_REFUSED``. A motion that fails once it is commanded ends the
pick with its typed outcome.

The orchestrator never plans a viewpoint. The viewpoint planners that
once did, and the relocate path that drove them, were removed on
2026-09-29: no config has built a planner since the multi-view commit
gate left (2026-09-28), and a cell that needs another view hands the
pick look poses (``src/robot/execution/looks.py``) or fuses a second
camera (``grasping.fusion``).

A camera on the wrist looks from the looks it is handed (``looks``), in
their order, before its first attempt (:meth:`BinPickingOrchestrator.look_around`).
The owner's rule (2026-09-29): the looks exist to find a safe grasp point,
never to scan the part whole. Each look is fused with the looks before it,
the grasp is computed again on what they saw together, and the looking
stops at the first look whose grasp is valid and carries none of the
rescan reasons. A grasp that is still uncertain, or no candidate at all,
sends the arm on to the next look. A fixed camera ignores ``looks`` and
picks exactly as it always did. A wrist pick handed looks never rescans
where the arm stands: its next view is the next look. One handed none has
no next look, and rescans where it stands as it always did.

Once every look a pick was handed was visited and its grasp is still not
safe enough, it generates one more view, the last resort (the owner,
2026-09-29): its judged look turned about the part to the contact face of
the chosen grasp no view showed, or to the side no look faced when there is
no grasp, screened by the arm before it moves and driven on the straight
joint line alone (``src/robot/execution/generated_view.py``). The arm then
goes back to the look that saw the part on that line, or approaches from
where it stands. The planner world holds every frame the pick took until the
pick ends, and is offered the target's cloud fused over the looks. With
``both_faces`` a grasp no view showed both contact faces of is not gripped,
on a fixed camera as on the wrist.

Every object's result passes the closing axis a program names
(``GraspMotion(closing_axis=...)``, the owner's "choose, don't twist" of
2026-09-30) right after the calculator ranked it, before anything reads its
candidates: only the grasps that close within 30 degrees of the axis are kept,
each turned the named way round. Where the program names none and the cell
names how its hand and camera naturally stand (``robot.natural_closing_axis``),
every grasp is turned the way round nearer that direction instead, none left
out (:meth:`BinPickingOrchestrator._closing_along`). So the target choice, the
looks' judgement, the reranks, the approach check and the push all read the
grasps the pick would grip.

Two recoveries run inside the attempt, where the part is known (the owner,
2026-09-29 and 2026-09-30). Where the service hands a pick a push gate
(``push_gate``, dense_clutter only), a wrist pick whose part every grasp
collided with something (``ALL_COLLIDED``), or whose every approach was
blocked where approach validation runs, and which a neighbour stands within
25 mm of in what the looks saw, pushes the part with the open jaws, goes back
to the look it stood at (to the view it generated on the straight joint line
alone), looks again and picks from that look
(:meth:`BinPickingOrchestrator._push_the_failed_part`). Before it pushes, it clears a blocker where it can (the owner,
2026-10-02, :meth:`BinPickingOrchestrator._clear_the_blockers`): the neighbour whose removal frees the most grasps of the
part, gripped like any part and set down on a free spot the camera saw or at the place the owner names
(:attr:`BinPickingOrchestrator.blocker_place`); it looks again, and the next attempt grasps the part. The push comes
only where no blocker could be cleared. And the parts
``next_target`` skips (``exclusion_zones``) are no targets, next to the label
gate; a pick that sees only such parts stops with their sentence.

The orchestrator is vendor-neutral: it depends only on the
:class:`RobotArm` Protocol, the calculator and a perception protocol.
A real SAM2 perception source plugs in without changes to this file.
"""

from __future__ import annotations

import logging
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Mapping,
    TypedDict,
)

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Transform
from src.robot.constants import create_robot_logger
from src.robot.core.keep_out import KeepOutBox, SegmentationOffer, holding_views, keeping_out
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    CameraWorldStamp,
    CameraWorldUse,
    Gripper,
    MotionStatus,
    RobotArm,
    SupportsRobotStatus,
)
from src.robot.core.arm_capabilities import halt_state_of, halted_refusal
from src.robot.core.gripper import why_not_known_open
from src.robot.grasping.loop._shadow_aggregator import (
    _ShadowTelemetryAggregator,
    _build_candidate_log as _build_candidate_log,  # re-exported under this module's name
)
from src.robot.grasping.motion.execution_policy import (
    GraspExecutionPolicy,
    PolicyOutcome,
    PolicyReport,
    closing_axis_twisted,
)
from src.robot.grasping.geometry.closing_axis import (
    CANDIDATES_THE_AXIS_CHOOSES_AMONG,
    ClosingAxis,
    closing_axis_of,
    closing_axis_refusal,
    result_closing_along,
    result_closing_toward,
)
from src.robot.grasping.telemetry.latency_tracker import LatencyStage
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.types.perception import (
    PerceptionFrame,
    PerceptionSource,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.modes import (
    GraspSamplingMode,
    resolve_grasp_sampling_mode,
)
from src.robot.grasping.scoring.success_probability import (
    RankingBlendConfig,
    RankingBlendTelemetry,
    SHADOW_METADATA_KEY_PROBABILITY,
    ShadowSuccessContext,
    ShadowSuccessTelemetry,
    UncertaintyRerankConfig,
    UncertaintyRerankTelemetry,
    annotate_grasp_points_with_shadow_probability,
    maybe_blend_rerank_candidates,
    maybe_uncertainty_rerank_candidates,
)
from src.robot.grasping.loop.progress import (
    PickProgressListener,
    PickStage,
    ShouldCancel,
    emit,
)
from src.robot.grasping.loop.target_selector import (
    OrderingDecision,
    TargetOrderingConfig,
    select_target,
)
from src.robot.grasping.motion.trajectory_safety import (
    ApproachPathPolicy,
    validate_approach_and_retreat,
)
from src.robot.grasping.loop._pick_helpers import (
    _build_target_candidates,
    _grasp_point_to_grasp_pose,
)

if TYPE_CHECKING:  # pragma: no cover (import only for typing)
    from src.config.schema.robot.grasping_schema import (
        FusionGeometryConfig,
        GraspingSupportConfig,
    )
    from src.geometry import Pose
    from src.robot.core import JointPositions
    from src.robot.grasping.recovery.blocker import BlockerRecord, Cluster
    from src.robot.execution.generated_view import GeneratedView, MovedBack, ViewAim
    from src.robot.execution.motion import MotionReport
    from src.robot.grasping.deep.ranker.context import DeepRankerContext
    from src.robot.grasping.multiview.scene_geometry import FusedSceneGeometry, ObservedView
    from src.robot.grasping.multiview.unseen_side import JawFacesSeen
    from src.robot.grasping.recovery.exclusion_zones import ExclusionZones
    from src.robot.grasping.recovery.push_gate import FailedPart, PickPush, PushGate
    from src.robot.grasping.recovery.push_motion import PushOutcome
    from src.robot.grasping.recovery.push_planner import PushPlan
    from src.robot.grasping.types.perception import MultiCameraPerceptionSource
    from src.robot.grasping.collision import GripperGeometryStrategy
    from src.robot.grasping.scoring.corridor import CorridorAnalysisConfig
    from src.robot.grasping.telemetry.latency_tracker import LatencyTracker
    from src.robot.grasping.scoring.feasibility_score import FeasibilityScoreConfig
    from src.robot.grasping.motion.frame_resolver import FrameResolver
    from src.robot.grasping.rl.router import (
        PerceptionShadowTelemetry,
        RankingShadowTelemetry,
        RecoveryShadowTelemetry,
        SequencingAttemptState,
        SequencingShadowTelemetry,
        ShadowRouter,
    )

__all__ = [
    "BinPickingOrchestrator",
    "LookedAround",
    "PerceptionFrame",
    "PerceptionSource",
    "PickAttempt",
    "PickOutcome",
    "PickReport",
]


# ``PerceptionFrame`` and ``PerceptionSource`` live in
# ``grasping.types.perception`` (imported above) so downstream modules can name
# them without importing this orchestrator. They stay re-exported here (see
# ``__all__``) for backward compatibility. ``SegmentationLike`` lives beside
# them there and is not re-exported here; import it from
# ``grasping.types.perception``, or from ``grasping``, which does re-export it.


#: The pick loop's own lines, to the robot log and ``pick_loop.log`` (RC6 of the cell-fix plan: they reached no file).
_LOG: logging.Logger = create_robot_logger(__name__, "pick_loop.log")

#: How many grasps of one look an attempt tries at most: the best, then up to three more of its candidates, each only
#: after the one before was refused before anything was sent with the hand kept out of the part's keep-out box (the
#: owner's "try the next grasps", cell fixes Track C).
GRASPS_TRIED_PER_ATTEMPT: int = 4
#: What a try was, on ``PickAttempt.tries`` (the fix plan's contract 3).
_TRY_KEYS = ("rank", "outcome", "motion_status", "motion_message", "sent", "reached_part")
#: The opening a hand that says none is taken to have when its jaw region is laid out at a pose a try sent, millimetres:
#: wider than any jaw the cells carry, so a region the pick cannot measure is never the smaller one.
_UNKNOWN_APERTURE_MM: float = 100.0
#: The solids under the push's sweep are read every this many millimetres for the finger's floor.
_FLOOR_GRID_MM: float = 5.0
#: A push whose finger floor stands within this of the part's top, millimetres, would pass over the part: refused.
_FLOOR_UNDER_THE_TOP_MM: float = 5.0
#: How far a hand that says none opens, when a blocker's width is judged against it, millimetres: the Hand-E's stroke.
_BLOCKER_OPEN_WIDTH_MM: float = 50.0
#: How far past the robot base's own radius nothing is taken for a blocker, millimetres: the camera world's margin.
_BASE_PADDING_MM: float = 15.0
#: A group of seen points this near where this pick set a blocker down, millimetres, is that blocker, never taken again.
_SET_DOWN_REACH_MM: float = 30.0
#: A blocker's mask is its points' pixels grown by this many: the calculator reads every second pixel.
_BLOCKER_MASK_GROWTH_PX: int = 2
#: The TCP that sets a blocker down stays this far inside the workspace's floor and ceiling, millimetres.
_SET_DOWN_WORKSPACE_MARGIN_MM: float = 20.0
#: The support's band where the live world says none, millimetres (``plane_clearance_mm``).
_SUPPORT_BAND_MM: float = 5.0
#: A support the camera world reads under a part this much over the declared one, millimetres, is said where it raises.
_SUPPORT_RAISE_SAID_MM: float = 0.5

#: How much vertical extent a single-view target cloud needs before it may be used to locate the
#: support plane, in millimetres along the support normal.
#:
#: Not a quality threshold. It separates two situations that a point cloud cannot otherwise tell
#: apart: seeing down the side of an object to what it rests on, and looking straight at one face.
#: Only the first has observed the support; the second, used as an observation, puts the plane on
#: top of the object. 25 mm is a jaw pad's own dimension: below that there is not enough geometry
#: for the lowest point to mean anything, and the declared height, which an operator can at least
#: verify, is the better answer.
_MIN_CLOUD_EXTENT_FOR_SUPPORT_MM: float = 25.0

#: The look of a wrist pick handed no looks: the pose the arm stands at, perceived without moving.
_HERE: str = "here"
#: Why a pick handed looks generated no view: its task switched multi-view off (``generated_view``).
_NO_VIEW_GENERATED_WITH_MULTI_VIEW_OFF = (
    "this pick looks from its first look only (multi-view is off for its task), so no view is generated after it")
#: How far, on every joint, the arm may stand from the one view its pick generated and still stand at it, in degrees:
#: what a read of the joints the arm was driven to comes back as.
_SAME_JOINTS_DEG: float = 1.0
#: What a halt left of a pick whose motion it ended, in the halt's sentence (``halted_refusal``).
_HALTED_WHERE_IT_STANDS = "it stands where the halt left it, and nothing more is commanded"


class PickOutcome(str, Enum):
    """Terminal status of a single :meth:`BinPickingOrchestrator.run` call.

    ``RELOCATED_EXHAUSTED`` (``relocated_exhausted``) is retired: the relocate path that ended a pick
    with it was removed on 2026-09-29. A record logged before then still audits, because the service
    reported it as ``no_valid_grasp`` and kept the low-level name only as free text in
    ``telemetry['low_level_outcome']``, which nothing parses back into this enum.
    """

    EXECUTED = "executed"
    RESCANNED_EXHAUSTED = "rescanned_exhausted"
    NO_PERCEPTION = "no_perception"
    ABORTED = "aborted"
    # An operator asked the run to stop between attempts. Deliberately not ``ABORTED``: that maps to
    # EXECUTION_FAILED, which would count every press of a stop button as a failed execution and quietly
    # poison the pick-rate KPI with operator decisions.
    CANCELLED = "cancelled"
    OBJECT_NOT_DETECTED = "object_not_detected"
    EXECUTION_FAILED = "execution_failed"
    # The winning grasp candidate is a camera-frame pose, and the execution
    # policy refuses to command it, fail-closed per the frame contract. Kept
    # distinct from EXECUTION_FAILED so a caller can detect missing
    # FrameResolver wiring.
    CAMERA_FRAME_REJECTED = "camera_frame_rejected"
    # Dense-mode approach validation. Every ranked candidate's straight-line
    # approach/retreat sweep collided with a neighbour in the scene cloud (the
    # calculator only checks the final pose), so the fall-back found no
    # sweep-clear candidate and refused rather than driving into an obstacle.
    APPROACH_PATH_BLOCKED = "approach_path_blocked"
    # The controller could not move: protective stop, emergency stop, a safety-mode stop, or simply
    # powered off. Deliberately not EXECUTION_FAILED, for the same reason CANCELLED is not ABORTED:
    # the grasp did not fail, the cell stopped, and folding the two together buries a machine state
    # an operator must act on inside the failure taxonomy.
    #
    # A stopped controller reports commanded motions as a driver failure ("Driver reported moveJ()
    # failure"), which reads exactly like a bad grasp unless the driver is asked: get_robot_status
    # reports the stop and raise_if_stopped raises on it.
    CONTROLLER_NOT_OPERATIONAL = "controller_not_operational"
    # The gripper raised while it was commanded, or a hand that toggles with no sensor would not start
    # the pick (``PolicyOutcome.GRIPPER_FAULT``): it believed its jaws closed and nobody at a terminal
    # said otherwise, or a person aborted. Also a push that found nobody can say where such a hand's
    # jaws stand, or a gripper that measures its width not connected or unreadable
    # (``refused_jaws_unknown``). Not EXECUTION_FAILED, for the reason above: the grasp did not fail, the
    # hand needs a person, and a campaign stops on it rather than perceiving again.
    GRIPPER_FAULT = "gripper_fault"


class _MotionFields(TypedDict):
    """The motion fields of a `PickAttempt`, as `_motion_fields` and its kin flatten a motion onto it: strings, or
    ``None`` where the motion said nothing.

    Typed by name, as `_FusionFields` is, so they spread into an attempt beside its fields of other types.
    """

    motion_status: str | None
    motion_message: str | None
    motion_error: str | None
    camera_world: str | None
    camera_world_reason: str | None


def _motion_fields(report: "PolicyReport | None") -> _MotionFields:
    """The motion explanations a ``PolicyReport`` carries, flattened onto an attempt.

    Best-effort and never raising: a report that predates these fields, or an object that is not a
    ``PolicyReport`` at all, yields ``None`` for each and the attempt is exactly as before.
    Stringified on the way out so the attempt stays JSON-safe for the telemetry tail.

    ``camera_world`` and ``camera_world_reason`` are the use and the reason of the report's weakest
    camera-world stamp. Both are ``None`` when no typed motion was commanded or that stamp is
    UNSTATED, so a driver that stamps nothing adds nothing to the attempt or to the event.
    """
    if report is None:
        return {"motion_status": None, "motion_message": None, "motion_error": None,
                "camera_world": None, "camera_world_reason": None}
    status = getattr(report, "motion_status", None)
    err = getattr(report, "error", None)
    stamp = getattr(report, "camera_world", None)
    said: CameraWorldStamp | None = (
        stamp
        if isinstance(stamp, CameraWorldStamp) and stamp.use is not CameraWorldUse.UNSTATED
        else None
    )
    return {
        "motion_status": getattr(status, "value", None) if status is not None else None,
        "motion_message": (
            str(getattr(report, "motion_message", None))
            if getattr(report, "motion_message", None) is not None
            else None
        ),
        "motion_error": f"{type(err).__name__}: {err}" if err is not None else None,
        "camera_world": said.use.value if said is not None else None,
        "camera_world_reason": said.reason if said is not None else None,
    }


def _sent_nothing(moved: "MotionReport") -> bool:
    """Whether a look's failed motion was refused by a guard or the planner before anything was sent.

    The one rule, ``generated_view.refused_before_sending`` (its ``REFUSED_BEFORE_SENDING``): only such a refusal lets a
    wrist pick skip a look, a generated view's turn or its move back and go on (the owner, 2026-09-29). Every other
    failed motion may have been commanded, so the arm may have moved part of the way or may still be moving, and the
    attempt ends there, as every failed motion of a pick does.
    """
    from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415

    return refused_before_sending(moved)


def judged_faces_turned_away(orchestrator: Any) -> str:
    """Why a pick that asks for both jaw contact faces (``both_faces``) cannot run on ``orchestrator``, or ``""``.

    ``both_faces`` judges the contact faces of the grasp the calculator chose, and grips that grasp only: the reranks
    stand down and the approach check falls back to no other candidate. A policy that yaws every grasp about base Z
    before it closes (``GraspExecutionPolicy.align_closing_to_base_x``, ``GraspMotion``'s option for symmetric parts)
    closes the jaws on another pair of faces, which nobody judged. On a hand that closes jaws on two faces the two
    together are the program's error, refused (``ValueError``) before anything moves: by the pick loop's
    :meth:`BinPickingOrchestrator.run` and :meth:`BinPickingOrchestrator.look_around`, and by the service's ``pick``
    before it asks the controller or the hand. A suction cup has no jaw faces to turn away from.
    """
    from src.robot.grasping.collision.gripper_model import SuctionCupGripperModel  # noqa: PLC0415

    if isinstance(getattr(orchestrator, "gripper_model", None), SuctionCupGripperModel):
        return ""
    if not getattr(getattr(orchestrator, "policy", None), "align_closing_to_base_x", False):  # as the twist reads it
        return ""
    return ("both_faces grips only the grasp whose jaw contact faces were judged, and this pick's motion turns every "
            "grasp about base Z to close along base X (GraspMotion(align_closing_to_base_x=True)), onto a pair of faces "
            "nobody judged: ask for one or the other; nothing was moved")


def _look_motion_fields(moved: "MotionReport") -> _MotionFields:
    """A refused look's motion, flattened onto the attempt the way :func:`_motion_fields` flattens a policy's motion."""
    error = moved.result.exception
    stamp = moved.camera_world
    said = stamp if isinstance(stamp, CameraWorldStamp) and stamp.use is not CameraWorldUse.UNSTATED else None
    return {
        "motion_status": moved.status.value,
        "motion_message": moved.message or None,
        "motion_error": f"{type(error).__name__}: {error}" if error is not None else None,
        "camera_world": said.use.value if said is not None else None,
        "camera_world_reason": said.reason if said is not None else None,
    }


def _push_motion_fields(outcome: "PushOutcome") -> _MotionFields:
    """A push's last motion, flattened onto its attempt the way :func:`_motion_fields` flattens a policy's motion."""
    last = outcome.results[-1] if outcome.results else None
    error = outcome.error
    stamp = getattr(last, "camera_world", None)
    said = stamp if isinstance(stamp, CameraWorldStamp) and stamp.use is not CameraWorldUse.UNSTATED else None
    return {
        "motion_status": outcome.motion_status.value if outcome.motion_status is not None else None,
        "motion_message": outcome.motion_message or None,
        "motion_error": f"{type(error).__name__}: {error}" if error is not None else None,
        "camera_world": said.use.value if said is not None else None,
        "camera_world_reason": said.reason if said is not None else None,
    }


def _leg_verdicts(outcome: "PushOutcome") -> tuple[str, ...]:
    """What the arm answered for every motion of a push, ``"<leg>: <status>"`` in order (P0, then the contact legs), the
    message added where a motion did not run (Track P's item 8)."""
    from src.robot.grasping.recovery.push_motion import APPROACH_LEG, CONTACT_LEGS  # noqa: PLC0415

    legs = (APPROACH_LEG, *CONTACT_LEGS)
    said = []
    for leg, result in zip(legs, outcome.results):
        status = getattr(getattr(result, "status", None), "value", "unknown")
        message = "" if getattr(result, "ok", False) else str(getattr(result, "message", "") or "")
        said.append(f"{leg}: {status}" + (f" ({message})" if message else ""))
    return tuple(said)


def _label_key(segmentation: Any) -> str:
    """The name a segmentation goes by at the label gate: its label, else its prim path, else ``""``."""
    return str(getattr(segmentation, "label", "") or getattr(segmentation, "prim_path", "") or "")


def _say_the_look(
    label: str, *, valid: bool, uncertain: bool, good: bool, reasons: tuple[GraspFailureReason, ...],
    faces: "JawFacesSeen | None", faces_asked: bool,
) -> None:
    """One line per look of a wrist pick: whether its grasp is safe enough to stop looking at, and if not, why."""
    said = ", ".join(str(reason) for reason in reasons) or "no reason given"
    if good:
        _LOG.info("look %s: a valid grasp with no rescan reason; the looking ends here", label)
    elif not valid:
        _LOG.info("look %s: no grasp candidate (%s); it is not safe to stop looking here", label, said)
    elif uncertain:
        _LOG.info("look %s: the grasp is uncertain (%s); it is not safe to stop looking here", label, said)
    elif faces_asked:
        missing = faces.missing if faces is not None else ()
        jaws = ", ".join(f"jaw {1 if face.side < 0 else 2}" for face in missing) or "either jaw"
        _LOG.info("look %s: the contact face at %s of the chosen grasp was not seen; both are asked for, so it is "
                  "not safe to stop looking here", label, jaws)


@dataclass(frozen=True, slots=True)
class PickAttempt:
    """One iteration of the pick loop, surfaced for debugging."""

    attempt_index: int
    target_index: int | None
    reasons: tuple[GraspFailureReason, ...]
    score: float
    # "executed", "object_not_detected", "camera_frame_rejected",
    # "approach_path_blocked", "gripper_fault" or "execution_failed" from _map_policy_outcome;
    # "rescan" or "exhausted" when no candidate was found (``_RESCAN_REASONS``; a wrist pick handed
    # looks never rescans where it stands, so its attempt is "exhausted");
    # "controller_not_operational" when the controller cannot move; "look_refused" when a wrist
    # pick reached none of its looks, the motion to one (its generated view and its move back
    # included) failed once it may have been commanded or was refused by its verb before any
    # command, or the controller stopped on the way to one, with nothing perceived there and
    # nothing else commanded; "faces_unseen" when a pick asked for both jaw contact faces
    # (``both_faces``) and no view showed both (no look of a wrist pick, generated or declared, nor
    # the cameras of a fixed one), so its grasp was not gripped. "relocate" is retired with the relocate path (2026-09-29) and appears only on
    # attempts recorded before then.
    action: str
    # When ``target_ordering`` is enabled this carries the typed
    # :class:`OrderingDecision` produced by the selector for this
    # attempt. Defaults to ``None`` so legacy consumers see no surface
    # change.
    ordering_decision: OrderingDecision | None = None
    # Why the motion failed, in the one place that survives the pick.
    #
    # ``_map_policy_outcome`` collapses a ``PolicyReport`` into the single word "execution_failed",
    # dropping the report's typed ``motion_status``, the driver's own ``motion_message`` and the
    # exception, so a cell that reports a failed motion otherwise keeps no explanation of it.
    #
    # Copied onto the attempt rather than referenced: the orchestrator keeps only the last policy
    # report (``_last_policy_report``), so on a multi-attempt pick every earlier explanation is gone
    # by the time anyone reads it. Optional and defaulted, so every existing construction site is
    # unchanged.
    motion_status: str | None = None
    motion_message: str | None = None
    motion_error: str | None = None
    # Which camera world stood behind the motions, copied for the same reason: the use and the
    # reason of the policy report's weakest stamp, both None when no typed motion was commanded or
    # the stamp said nothing.
    camera_world: str | None = None
    camera_world_reason: str | None = None
    # Which cameras' surfaces of this attempt's target were fused into the one cloud its grasps were
    # generated from, the camera the grasp is synthesised in first (`grasping.fusion.geometry`).
    # Empty whenever that cloud came from one camera: fusion off or standing down, no second camera
    # delivered, none identified the target, or an external target cloud replaced the fused one.
    # Never one camera alone, and never the same camera twice.
    #
    # Not the frame telemetry's `fused_views`, which lists every other camera that segmented
    # anything, matched or not. Copied onto the attempt for the reason the motion fields are: the
    # fused scene is rebuilt every frame, so an earlier attempt's answer is gone by the end.
    fused_views: tuple[str, ...] = ()
    # How many objects of the frame this attempt ranked gained a second camera's surface, the
    # target or not; 0 when none did or no frame was ranked.
    fused_objects: int = 0
    # Why the grasp this attempt ranked was not gripped, in the owner's words, on a "faces_unseen"
    # attempt: the pick asked for both jaw contact faces (``both_faces``) and this names the one no
    # view showed. Copied onto the attempt so the service's report can say it where no look result
    # carries it (a fixed camera). On an attempt that found no grasp to go on with, where the closing
    # axis the program named (``closing_axis``) left a part none, which axis and how far, whichever
    # part the frame failed on. ``None`` on every other attempt.
    withheld: str | None = None
    # What the push of a "push" attempt came to: ``pushed`` (the part was pushed, the arm went back to the look and
    # looked again; the next attempt picks from that look), ``unsafe_recovery_refused`` or
    # ``controller_not_operational`` (it stopped where the arm stands: a person decides),
    # ``refused_controller_stopped`` (the controller could not move, or could not be read, before anything moved: the
    # pick ends there, ``controller_not_operational``), or ``refused_jaws_unknown`` (nobody can say where a toggle
    # hand's jaws stand, or a gripper that measures its width is not connected or its width read failed, before
    # anything moved: the pick ends there, ``gripper_fault``). ``None`` on every other attempt. The push's own sentence
    # is on the pick loop's ``pushes``.
    push: str | None = None
    # Why no part was a target, on an attempt that saw only parts ``next_target`` skips: the exclusion zones' sentence
    # (``ExclusionZones.only_excluded_sentence``). The pick stops there rather than try one of them again or another
    # object. ``None`` on every other attempt.
    excluded: str | None = None
    # On such an attempt (``excluded``): whether every part it skipped stands in a region its task keeps out
    # (``ExclusionRegion``: the bin it places into, the circle about its drop), so the look saw nothing left for the task
    # to pick. ``False`` where a zone ``next_target`` made after a failed pick skipped one of them alone: that part still
    # stands there to be picked. ``False`` on every other attempt.
    excluded_by_regions: bool = False
    # Every grasp this attempt handed the policy, in order (the fix plan's contract 3, the owner's "try the next
    # grasps"): per try its ``rank`` among the look's candidates, the policy's ``outcome``, the ``motion_status`` and
    # ``motion_message`` it ended on, whether it ``sent`` any motion, and whether it ``reached_part``: a motion it sent
    # put the open hand's jaws into the held target's keep-out box, or it went to the part (where the pick cannot tell,
    # it did). Empty where no grasp was handed over.
    tries: tuple[dict[str, Any], ...] = ()
    # What clearing a blocker came to, on a "clear_blocker" attempt (``blocker.BlockerRecord.code``): ``set_aside``, or
    # why the clearing stopped once something moved. ``None`` on every other attempt.
    blocker: str | None = None


class _FusionFields(TypedDict):
    """The two fusion fields of a `PickAttempt`, as `BinPickingOrchestrator._fusion_of` gives them.

    Typed by name so they spread into an attempt beside `_motion_fields`, whose values are strings.
    """

    fused_views: tuple[str, ...]
    fused_objects: int


@dataclass(frozen=True, slots=True)
class PickReport:
    """Aggregated outcome of a :meth:`BinPickingOrchestrator.run` call."""

    outcome: PickOutcome
    attempts: tuple[PickAttempt, ...] = ()
    executed_grasp: GraspResult | None = None
    #: The telemetry of the grasp the loop chose, whatever happened next.
    #:
    #: `executed_grasp` is set only on `PickOutcome.EXECUTED`, which is right for what it names. But
    #: the calculator's telemetry, and the `deep_ranker_*` keys the shadow ranker stamps onto it, are
    #: about the candidates rather than about whether the motion succeeded. Read off `executed_grasp`
    #: they are absent from every `EXECUTION_FAILED` attempt: measured on the offline feature sweep,
    #: three of four records, each scored by the shadow whose opinion then reached no record.
    #:
    #: Empty when nothing was selected, so a run that generated no candidate is byte-identical.
    selected_telemetry: Mapping[str, Any] = field(default_factory=dict)
    # Optional shadow-mode telemetry for the winning candidate of the
    # final attempt. ``None`` whenever the orchestrator is built without
    # a :class:`ShadowSuccessContext`, which is the case when the
    # success model is disabled or failed to load. Populated for both
    # ``EXECUTED`` outcomes and the ``EXECUTION_FAILED`` /
    # ``CAMERA_FRAME_REJECTED`` branches when the model scored the
    # chosen grasp.
    shadow_success_telemetry: ShadowSuccessTelemetry | None = None
    # Audit telemetry from the bounded probability-blend reranker.
    # ``None`` when ``ranking_blend_config`` was not provided or
    # ``ranking_blend_config.enabled is False`` (the silent
    # byte-identical no-op). When non-``None``, the ``applied`` field
    # tells whether the rerank actually fired and ``skip_reason`` names
    # the gate when it did not.
    ranking_blend_telemetry: RankingBlendTelemetry | None = None
    # Per-candidate uncertainty re-rank audit. ``None`` when the rerank config is unset/disabled
    # (the silent byte-identical path); otherwise an UncertaintyRerankTelemetry whose ``applied`` tells
    # whether the rerank fired and ``skip_reason`` names the gate when it did not.
    uncertainty_rerank_telemetry: UncertaintyRerankTelemetry | None = None
    #: Where the object this pick went for was seen, in BASE millimetres: the median of its mask's surface, the pixels
    #: behind a depth step left out, as ``Locator`` places an object. ``None`` when no candidate reached the arm, when
    #: the cell resolves no CAMERA to BASE, or when the mask saw no depth.
    target_centre_mm: "tuple[float, float, float] | None" = None


# ---------------------------------------------------------------------------
# Looks of a wrist pick
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _LookView:
    """What one look of a wrist pick saw, kept for the looks after it and for whoever records the pick.

    ``frame`` is the look's own frame whole: its RGB, its depth, the tool pose stamped at its shutter and its
    intrinsics. ``surfaces`` holds, per segmentation of the frame, the mask a fusion may give full weight: the mask less
    the pixels behind a depth step (``pixels_behind_depth_steps``, the Locator's surface rule) and less the grazing ones
    (``scene_geometry.grazing_pixels``), on ``depth``, which is the measured surface where the frame carries one.
    ``clouds`` is each of those in BASE, placed by ``camera_to_base``. ``view`` is the same surfaces as the looks after
    this one fuse them, the empty ones left out, so a blob number in an association is a position in ``blob_objects``,
    which says whose segmentation it is; ``None`` when the look held nothing it could place. ``name`` is the view's
    identity in every association and report: the rig and the look, ``rig@look``, so two looks of one camera are two
    views; the rig alone for a pick handed no looks. ``joints`` is the configuration the arm stood at when the frame was
    taken, read off the arm, what the move back to this look returns to; ``None`` for a pick handed no looks, or where
    the arm could not say.
    """

    label: str
    pose: Any
    name: str
    frame: PerceptionFrame
    camera_to_base: "np.ndarray | None"
    depth: "np.ndarray | None" = None
    surfaces: "tuple[np.ndarray | None, ...]" = ()
    clouds: "tuple[np.ndarray, ...]" = ()
    view: "ObservedView | None" = None
    blob_objects: tuple[int, ...] = ()
    joints: "JointPositions | None" = None

    def blob_of(self, index: int) -> int | None:
        """The blob number segmentation ``index`` of this look has in :attr:`view`, or ``None`` when it has none."""
        return self.blob_objects.index(index) if index in self.blob_objects else None

    def label_of_blob(self, blob: int) -> str:
        """The label of the segmentation behind ``blob``, lower case, ``""`` when it carries none."""
        if self.view is None or not 0 <= blob < len(self.view.segmentations):
            return ""
        return str(getattr(self.view.segmentations[blob], "label", "") or "").strip().lower()

    def cloud_of_blob(self, blob: int) -> np.ndarray:
        if not 0 <= blob < len(self.blob_objects):
            return np.zeros((0, 3), dtype=np.float64)
        return self.clouds[self.blob_objects[blob]]


@dataclass(frozen=True, slots=True, eq=False)
class _Call:
    """What one object of a ranking was computed with, so the calculator can be asked again of the same frame: with a
    neighbour's pixels left out (clearing a blocker), or of a neighbour as the part (the blocker's own grasp)."""

    seg: Any
    depth: Any
    neighbours: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    camera_id: str


@dataclass(frozen=True, slots=True, eq=False)
class _RankedObject:
    """What the ranking of one look computed for one of its objects: the calculator's result, the height of the support
    it was computed against, BASE mm (``None`` when none was resolved), the cloud the generator was handed for it
    (``None`` when it derived its own from the mask), and the call that computed it."""

    result: GraspResult
    support_height_mm: float | None
    cloud_base_mm: "np.ndarray | None"
    call: "_Call | None" = None


@dataclass(frozen=True, slots=True, eq=False)
class _Tried:
    """What the grasps an attempt handed the policy came to (:meth:`BinPickingOrchestrator._execute`).

    ``report`` is the last try's. ``tries`` are the rows of ``PickAttempt.tries``. ``refused`` is whether every try was
    refused before anything was sent and none reached the part: the attempt is typed ``MOTION_PLAN_REFUSED``.
    ``cancelled`` is a stop asked for between two tries, ``controller`` why the controller cannot move where it said so
    between two tries, ``moved_back`` the move back to the look that failed once it was sent, and ``not_back`` why the
    arm could not be driven back there before anything was sent: each ends the attempt with nothing more commanded.
    """

    report: PolicyReport
    tries: tuple[dict[str, Any], ...] = ()
    refused: bool = False
    cancelled: bool = False
    controller: str | None = None
    moved_back: "MotionReport | None" = None
    not_back: str = ""


@dataclass(frozen=True, slots=True, eq=False)
class _Blocker:
    """A neighbour as the calculator is asked about it: its mask in the frame the part was judged on, its label."""

    mask: np.ndarray
    label: str = "blocker"
    score: float = 1.0


@dataclass(frozen=True, slots=True, eq=False)
class _BlockerChoice:
    """The blocker a clearing takes away (:meth:`BinPickingOrchestrator._choose_a_blocker`): the group of seen points,
    its mask, its grasp (BASE), the TCP pose that sets it down, what that place is and where its centre lands, how many
    grasps of the part its removal frees and how many of the part's refusals it spares, asked of the calculator."""

    cluster: "Cluster"
    mask: np.ndarray
    grasp: GraspPoint
    release: "Pose"
    where: str
    spot_xy: tuple[float, float]
    freed: int
    fewer: int


@dataclass(frozen=True, slots=True, eq=False)
class _Judgement:
    """How one look of a wrist pick answered: the grasp it ranked on the views fused so far, and whether it is safe.

    ``members`` names every view whose surface of the target is part of ``target_cloud_base_mm``, view identity to its
    blob there, this look's own included, so a later look finds the same part by association. ``member_looks`` are the
    looks among them, this one first. ``reasons`` are the result's, and ``RESCAN_RECOMMENDED`` when the looks disagree
    on what the target is (``labels_disagree``). ``faces`` is whether each jaw's contact face of the chosen grasp was
    seen, ``None`` when there is no grasp, a suction hand, or no BASE grasp to judge. ``good`` is the early stop: a
    valid grasp with none of the rescan reasons, and, with ``both_faces``, both contact faces seen.
    ``hand_eye_gap_mm`` is the hand-eye check, measured once, on the judgement the looking ends on (``None`` before).

    The rest is the orchestrator's per-frame state as the ranking of this look left it, restored by ``_adopt`` when
    this look's grasp is the one the pick goes on with.
    """

    look: _LookView
    result: "GraspResult | None"
    target_index: int | None
    ordering: "OrderingDecision | None"
    target_cloud_base_mm: "np.ndarray | None"
    members: Mapping[str, int]
    member_looks: tuple[str, ...]
    reasons: tuple[GraspFailureReason, ...]
    faces: "JawFacesSeen | None"
    good: bool
    labels_agreed: int
    labels_disagree: tuple[tuple[str, str], ...]
    #: How the looks that saw the part agreed on its label, as the pick's account says it (``label agreed in N of M
    #: looks``, ``association.label_agreement_said``); ``""`` when this look judged no object. Said, never acted on.
    labels_said: str
    hand_eye_gap_mm: float | None
    camera_to_base: "Transform | None"
    scene_objects: tuple[Any, ...]
    fused_views_by_object: Mapping[int, tuple[str, ...]]
    fused_objects_in_frame: int
    fusion_telemetry: Mapping[str, Any]
    support_telemetry: Mapping[str, Any]
    debug_image_png: "bytes | None"
    #: What the ranking of this look fused (its per-object clouds and the neighbours every other view saw), ``None``
    #: where it fused nothing; and the height of the support the target's grasp was computed against, BASE mm, ``None``
    #: where none was resolved. What a push of the target plans from.
    fused_scene: "FusedSceneGeometry | None" = None
    support_height_mm: float | None = None
    #: What the ranking computed the target with (its call, to ask the calculator again when a blocker is cleared), and
    #: the support model of the frames the look ranked on (``support_surfaces.SupportModel``, contract 7), ``None`` where
    #: the arm's live world reads no support surfaces.
    ranked: "_RankedObject | None" = None
    support_model: Any = None

    @property
    def valid(self) -> bool:
        """Whether this look ranked a grasp at all."""
        return self.result is not None and self.result.is_success

    @property
    def frame(self) -> PerceptionFrame:
        return self.look.frame


@dataclass(frozen=True, slots=True, eq=False)
class LookedAround:
    """What a wrist pick's looks came to: where it looked, which grasp it goes on with, and why it stopped.

    ``visited`` are the looks perceived from, in order (``here`` for a pick handed none), the generated view included
    and the move back not repeated. ``judged`` is the look whose grasp the attempt goes on with, ``None`` when no look
    saw anything. ``dropped`` are looks that did not see the part the grasp was found on, ``refused`` the declared
    looks a guard or the planner refused before anything was sent, skipped, each with the refusal. ``views`` keeps
    every look's frame and surfaces for the record. ``stopped`` is the motion that ended the looking with nothing else
    commanded after it: one that failed once it may have been commanded, one the controller stopped on, or the first
    refusal when no look was reached. ``stopped_look`` is the look it was going to and ``controller`` why the controller
    cannot move, when that is why; ``cancelled`` a stop asked for between looks.

    The one generated view (``src/robot/execution/generated_view.py``): ``generated`` is its label when the arm went
    there, ``orbit +140 deg (...) deg``, and ``generated_view_deg`` its turn about the part, both empty when it did not;
    ``generated_skipped`` why no view was generated where one was due, ``""`` when one was, or none was due. The move
    back: ``moved_back`` is the look the arm went back to on the straight joint line, ``move_back_refused`` why it did
    not, when the approach starts from where the arm stands instead. ``withheld`` says why the grasp is not gripped:
    ``both_faces`` was asked and no view showed both contact faces of it, naming the face that was not seen.
    """

    visited: tuple[str, ...] = ()
    judged: _Judgement | None = None
    dropped: tuple[str, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    views: tuple[_LookView, ...] = ()
    stopped: "MotionReport | None" = None
    stopped_look: str = ""
    controller: str | None = None
    cancelled: bool = False
    generated: str = ""
    generated_view_deg: float | None = None
    generated_skipped: str = ""
    moved_back: str = ""
    move_back_refused: str = ""
    withheld: str = ""

    @property
    def looks_fused(self) -> tuple[str, ...]:
        """The looks whose views make up the target's cloud, the one the grasp was ranked on first."""
        return self.judged.member_looks if self.judged is not None else ()

    @property
    def jaw_faces_seen(self) -> tuple[bool, bool] | None:
        """(jaw 1 face seen, jaw 2 face seen) for the chosen grasp, ``None`` when it was not judged."""
        faces = self.judged.faces if self.judged is not None else None
        return faces.as_pair() if faces is not None else None

    @property
    def hand_eye_gap_mm(self) -> float | None:
        """The median distance between the looks' surfaces of the target, ``None`` when no two looks shared one."""
        return self.judged.hand_eye_gap_mm if self.judged is not None else None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------



def _mean_winning_association_score(fused: "FusedSceneGeometry") -> float:
    """Mean score over the (object, view) pairs that won their assignment. ``0.0`` when none did.

    Reads the associations the fusion already produced rather than recomputing anything, and counts
    only winning pairs: a losing pair's score says how strongly some other object was rejected, which
    is not evidence about the object that was kept.

    Never raises: an association shape it does not recognise contributes nothing rather than ending
    a pick, because this is telemetry.
    """
    total = 0.0
    count = 0
    for association in getattr(fused, "associations", ()) or ():
        assignment = getattr(association, "assignment", ()) or ()
        scores = getattr(association, "scores", ()) or ()
        for primary_index, matched in enumerate(assignment):
            if matched is None or primary_index >= len(scores):
                continue
            row = scores[primary_index]
            if matched < len(row):
                total += float(row[matched])
                count += 1
    return round(total / count, 6) if count else 0.0


#: The failure reasons a no-candidate attempt is retried on, with a fresh frame and no motion; every
#: other reason ends the pick. ``ACTIVE_PERCEPTION_RECOMMENDED`` and ``ALL_OUT_OF_WORKSPACE`` asked for
#: the camera to be moved, and were rescanned already whenever no viewpoint planner was wired, as on
#: every cell built from config since 2026-09-28. The planners and that relocate path were removed on
#: 2026-09-29, so they are rescanned everywhere.
_RESCAN_REASONS = frozenset(
    {
        GraspFailureReason.RESCAN_RECOMMENDED,
        GraspFailureReason.NO_CANDIDATES_GENERATED,
        GraspFailureReason.LOW_DEPTH_CONFIDENCE,
        GraspFailureReason.LOW_MASK_CONFIDENCE,
        GraspFailureReason.NO_VALID_DEPTH,
        GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED,
        GraspFailureReason.ALL_OUT_OF_WORKSPACE,
    }
)

def segmentation_offer_from_frame(
    frame: "PerceptionFrame",
    target_index: "int | None",
    *,
    camera: "Maybe[str]",
    camera_to_base_mm: "np.ndarray | None",
    target_points_base_mm: "np.ndarray | None" = None,
) -> "SegmentationOffer | None":
    """What one ranked perception frame offers a live planner world, or ``None`` when it can offer nothing.

    Every segmentation's mask under its label (``object_<index>`` for an unnamed one) and the
    target's mask to exclude, for ``camera``; and the target's surface in BASE, from
    ``surface_depth_map`` where the frame carries one and ``depth_map`` otherwise, placed by
    ``camera_to_base_mm``. ``target_points_base_mm``, where given and not empty, is offered as the
    target's surface instead: the target's cloud fused over the looks of a wrist pick, which holds
    what every view of the pick saw of it, so its box leaves the part out of every frame the world
    holds. With no camera only the target's points are offered, and with neither a camera nor
    points, or a frame that carries no capture time, nothing is.
    """
    segmentations = tuple(frame.segmentations)
    labelled = tuple(
        (str(getattr(seg, "label", "") or f"object_{index}"), mask)
        for index, seg in enumerate(segmentations)
        if (mask := getattr(seg, "mask", None)) is not None
    )
    exclude: tuple = ()
    target_label = ""
    points = None
    if target_index is not None and 0 <= target_index < len(segmentations):
        target = segmentations[target_index]
        target_label = str(getattr(target, "label", "") or f"object_{target_index}")
        target_mask = getattr(target, "mask", None)
        if target_mask is not None:
            exclude = (target_mask,)
            if camera_to_base_mm is not None:
                from src.robot.grasping.multiview.scene_geometry import to_base_mm

                surface = getattr(frame, "surface_depth_map", None)
                depth = surface if surface is not None else frame.depth_map
                base = to_base_mm(np.asarray(target_mask).astype(bool), depth, frame.intrinsics, camera_to_base_mm)
                points = base if base.shape[0] else None
    if target_points_base_mm is not None and np.asarray(target_points_base_mm).size:
        points = np.asarray(target_points_base_mm, dtype=np.float64).reshape(-1, 3)
    stamp = getattr(frame, "timestamp", None)
    if stamp is None:
        return None
    if not chosen(camera):
        if points is None:
            return None
        return SegmentationOffer(captured_at_s=float(stamp), target_points_base_mm=points, target_label=target_label)
    return SegmentationOffer(
        captured_at_s=float(stamp), camera=camera, labelled_masks=labelled, exclude_masks=exclude,
        target_points_base_mm=points, target_label=target_label,
    )


@dataclass
class BinPickingOrchestrator:
    """Drive the perceive, grasp, execute loop with failure-aware retries.

    Parameters
    ----------
    arm:
        Any :class:`RobotArm` driver. Only :meth:`get_tcp_pose` and
        :meth:`move_to` are required for the default pipeline; a real
        retreat motion additionally calls :meth:`move_to` once more.
    calculator:
        Configured :class:`GraspCalculator` instance.
    perception:
        Source of :class:`PerceptionFrame` snapshots.
    max_attempts:
        Hard cap on the loop iterations.
    dense_sampling:
        Legacy bool surface forwarded to :meth:`GraspCalculator.compute`.
        Recommended ``True`` for cluttered bins. Kept for backward
        compatibility; new callers should use ``grasp_sampling_mode``.
    grasp_sampling_mode:
        Typed grasp sampling mode (:class:`GraspSamplingMode`, legacy
        bool, string alias, or ``None``). When provided it overrides
        ``dense_sampling``; otherwise the legacy bool drives behavior.
        Default ``None`` resolves to ``AUTO`` so the calculator picks
        between silhouette and dense sampling heuristically.
    """

    arm: RobotArm
    calculator: GraspCalculator
    perception: PerceptionSource
    max_attempts: int = 5
    dense_sampling: bool = True
    grasp_sampling_mode: GraspSamplingMode | bool | str | None = None
    standoff_mm: float = 80.0
    retreat_mm: float = 100.0
    gripper: Gripper | None = None
    policy: GraspExecutionPolicy | None = None
    #: How the cell's hand and camera naturally stand (``robot.natural_closing_axis``, the owner's decision of
    #: 2026-09-30), set from the robot config by ``RuntimePickService.from_robot_config``: every grasp this loop ranks is
    #: turned, of its two equivalent wrist turns, to the one whose tool +X lies nearer it (:meth:`_closing_along`), none
    #: left out or tilted. A policy's ``closing_axis`` takes precedence, and beside its ``align_closing_to_base_x`` none
    #: is turned. A name or an orientation set here by hand is read as the config's is, and one that names no axis is
    #: refused before anything moves. ``None``, the default, turns no grasp.
    natural_closing_axis: "ClosingAxis | None" = None
    # Optional resolver for ``T_cam_to_base`` per frame. When wired, the
    # orchestrator forwards the resolved transform into
    # ``calculator.compute_result(camera_to_base=...)`` so the returned
    # grasps are in base frame. ``None``, the default, forwards nothing:
    # the calculator falls back to its constructor-time
    # ``camera_matrix`` and the candidates stay in the frame the
    # calculator chooses, which is camera unless the constructor was
    # configured with a transform.
    frame_resolver: "FrameResolver | None" = None
    # Optional clutter-aware target selector. When ``None`` or when
    # ``target_ordering.enabled`` is False the orchestrator takes the
    # plain ``argmax(local_score)`` path. When enabled the
    # per-segmentation winners are passed through :func:`select_target`
    # and the chosen candidate plus decision telemetry are returned.
    target_ordering: TargetOrderingConfig | None = None
    # Shadow-mode success-probability model carrier. When ``None`` (the
    # default) the orchestrator behaves byte-identically: no
    # ``GraspPoint`` metadata is mutated and
    # ``PickReport.shadow_success_telemetry`` stays ``None``. When set,
    # the chosen result's candidates are annotated with calibrated
    # probabilities, read-only with respect to ranking, and the winner's
    # telemetry is surfaced on the report for downstream replay tooling.
    shadow_success_context: ShadowSuccessContext | None = None
    # Bounded probability-blend reranker config. ``None`` or
    # ``enabled=False`` is the silent byte-identical no-op:
    # :func:`maybe_blend_rerank_candidates` returns the original tuple
    # and ``None`` telemetry, and the
    # ``PickReport.ranking_blend_telemetry`` field stays ``None``. When
    # enabled, the reranker fires only in the dense mode
    # (``dense_clutter``) under a non-shadow lifecycle phase and never
    # mutates ``GraspPoint.score``.
    ranking_blend_config: RankingBlendConfig | None = None
    # Per-candidate uncertainty re-rank carrier. ``None`` (default) is a silent byte-identical
    # no-op. When set, the orchestrator re-sorts candidates by ``score - weight*corridor_risk`` after the
    # blend (dense-only, non-shadow, weight>0, every candidate carrying a corridor-risk signal). A
    # producer must run: either ``GraspCalculator.corridor_risk_per_candidate`` set at construction,
    # or a ``corridor_config`` forwarded to ``compute()``, which this orchestrator does whenever its
    # own ``corridor_config`` is set. With neither, no candidate carries
    # ``corridor_blockage_confidence`` and the rerank no-ops with ``skip_reason=missing_uncertainty``.
    uncertainty_rerank_config: UncertaintyRerankConfig | None = None
    # Resolved grasp-mode label (``"easy"``, ``"auto"`` or ``"dense_clutter"``),
    # set by ``from_robot_config``. The mode-gated
    # approach validation reads it (``_approach_path_modes``). Empty string
    # means the orchestrator was constructed without mode awareness, and then
    # that validation never runs; this preserves byte-identical behavior for
    # every caller that constructs the orchestrator directly instead of via
    # ``from_robot_config``.
    mode_label: str = ""
    # Optional hard label-target for a prompt-driven dense pick. ``None`` (default) is
    # byte-identical: every segmentation is a candidate target (legacy argmax / ordering). When set to an
    # object label, only the matching segmentation is considered as the executable target; the other
    # segmentations still feed ``other_object_masks`` (the neighbour clutter the DENSE_CLUTTER sampler +
    # corridor-risk + approach-validation consume), so the prompt picks that object and never a
    # distractor. If the target's grasp fails the pick refuses (an honest no-pick, not a wrong-object pick).
    target_label: str | None = None

    # Multi-view scene collision: an optional fused multi-view scene point cloud in the
    # robot BASE frame (mm), supplied by a multi-view runner from the fixed cameras (overhead + obliques)
    # that see the whole scene, including neighbour surfaces the single grasp-synthesis view (e.g. the
    # eye-in-hand wrist) cannot see. ``None`` (default) is byte-identical: nothing extra is fed to the
    # collision filter. When set, the orchestrator transforms it into the captured frame's CAMERA frame
    # (the frame the collision filter plans in) and passes it as ``scene_points_mm``, vstacked with any
    # same-view neighbours, so a grasp that would collide with a neighbour the synthesis view missed is
    # rejected and the neighbour pick survives. The grasp synthesis stays single-view.
    external_scene_points_base_mm: "np.ndarray | None" = None

    # Optional fused multi-view target cloud (BASE frame) fed to the calculator's geometry generator
    # so it finds real side-face antipodal contacts a single top-down view cannot (the measured
    # bottleneck). ``None`` (default) is byte-identical (key never added). Passed straight through as
    # ``geometry_points_base_mm`` (the calculator transforms BASE->CAMERA itself); only forwarded when a
    # ``target_label`` is set (so the target's fused cloud is never applied to a non-target segmentation).
    external_target_geometry_base_mm: "np.ndarray | None" = None

    # Multi-camera geometry fusion (``grasping.fusion.geometry``). The generalisation of the field
    # above: instead of one externally-supplied cloud for one labelled target, the rig is observed
    # every pick and every candidate object gets the surface every camera that can identify it sees.
    # That is what a fixed multi-camera cell can do and a single view cannot: a candidate is
    # accepted on its closing axis, the closing axis comes from the silhouette, and one view sees
    # one side while an antipodal grasp needs two.
    #
    # All three ``None`` (default) means the keys are never added and the single-view path is
    # byte-identical. ``multi_camera_perception`` supplies the other cameras' frames,
    # ``camera_frame_resolvers`` maps each camera id to its own CAMERA->BASE (built for each camera
    # ``grasping.fusion.cameras`` fuses, from the calibration declared on its rig,
    # ``camera.cameras.rigs[<id>].extrinsics``, one persisted calibration per camera), and
    # ``fusion_geometry_config`` carries the metric, the match threshold and the missing-camera
    # posture.
    multi_camera_perception: "MultiCameraPerceptionSource | None" = None
    #: The cameras a real cell opened for its live planner world alone, given back on teardown
    #: beside `perception` and `multi_camera_perception`. `None` on every other cell.
    planner_world_cameras: "object | None" = None
    camera_frame_resolvers: "dict[str, FrameResolver] | None" = None
    #: Which cameras the config asked for, which is not the same as which ones built.
    #:
    #: `on_camera_unavailable: refuse` needs an expectation that a camera failing to build cannot
    #: erase. `set(camera_frame_resolvers)` is not one: that map is `{}` whenever `fusion.enabled`
    #: is false, or the `cameras` map is empty, or `grasping_cfg` is None, and in each of those
    #: cases `expected` is empty, `missing` is empty, and the refusal cannot fire. An expectation
    #: derived from what succeeded cannot notice what did not. The schema says what that costs, in
    #: the option's own description: "a cell must never fall back to single-view silently, which is
    #: the whole failure this option exists to make visible."
    #:
    #: Empty means "nothing was configured", not "nothing arrived", and the two are different
    #: answers. A cell with no fusion block configured is single-view by design and must not be
    #: refused; a cell that named three cameras and got two must be.
    #: One calculator per camera, for a cell whose cameras may each introduce an object. Empty is
    #: the normal case and means every object is computed by `calculator`, which is what happened
    #: before this existed. A calculator is bound to one camera's intrinsics at construction, so
    #: this map is what keeps a promoted object out of the wrong lens.
    camera_calculators: "dict[str, Any] | None" = None
    configured_camera_ids: "tuple[str, ...]" = ()
    #: The camera `perception` streams from, as the rest of the rig names it. Only a label: nothing
    #: is looked up by it, and an empty string is the honest value for a cell whose camera has no id
    #: in `camera.cameras.rigs` (a simulator runner building its own source, mostly). It exists so
    #: that a scene object can say which camera it was seen by without that being inferred from a
    #: position in a list, which is the coupling this change removes.
    primary_camera_id: str = ""
    fusion_geometry_config: "FusionGeometryConfig | None" = None

    # What the gripper is, and what the parts stand on. Both are inputs ``compute()`` accepts and
    # this loop must supply. Unsupplied, the calculator filters candidates against a default gripper
    # envelope and skips the table check entirely: ``collision_checker.py`` runs that check only when
    # ``support_plane`` is not None, and with it skipped ``rejected_table`` measured 0 in 2128 of
    # 2128 telemetry lines while an independent reference rejected 37 % of the same candidates for
    # driving a finger under the support.
    #
    # ``gripper_model`` comes from ``grasping.gripper_geometry`` via ``build_gripper_geometry``.
    # ``support_config`` comes from ``grasping.support``; the plane is resolved per pick, because the
    # declared height cannot know that a part is standing on another part and the target's own lowest
    # point can (measured: the declared height is too low on 100 % of stacked parts, the footprint on
    # 0 % of all 1073). Both ``None`` (default) means the keys are never added and the path is
    # byte-identical.
    gripper_model: "GripperGeometryStrategy | None" = None
    support_config: "GraspingSupportConfig | None" = None
    # ``deep_ranker_context`` comes from ``grasping.deep_ranker`` via ``apply_orchestrator_overlays``.
    # None (default) means nothing is scored and the path is byte-identical. Set means every
    # attempt's candidates are scored by the learned ranker and the result is stamped into
    # telemetry, changing no ordering and no decision.
    #
    # Shadow, and for a measured reason rather than caution: the failure a ranker has to predict
    # here is `finger_collision`, and features that cannot see the surroundings cannot predict it.
    # A ranker scored against an analytic verdict rather than against physics earns the right to be
    # watched on real picks and nothing more.
    deep_ranker_context: "DeepRankerContext | None" = None
    # ``corridor_config`` comes from ``grasping.occlusion`` via ``apply_orchestrator_overlays``.
    # None (default) is not forwarded, so the calculator's per-candidate corridor producer keeps its
    # own geometry. Set forwards the operator's corridor geometry into every compute() call this
    # orchestrator makes, which is the point: unforwarded, the block's values reach the telemetry
    # snapshot and stop there.
    corridor_config: "CorridorAnalysisConfig | None" = None
    # Optional latency tracker, set per pick by the service. The single ranking recording site, and
    # the single fusion one (`_fused_scene`).
    #
    # The span belongs on the one call every path shares: the default open-loop pick and the decision
    # path both reach it (the two-scan closed-loop path did until it left on 2026-09-29), while a wrap
    # further out covers only some of them
    # and leaves the open-loop pick with no ranking span at all. Timing the open-loop path from
    # outside is not an option either: that path calls `runtime.run_attempt()`, whose elapsed time
    # includes perception and arm motion, and reporting that against an 80 ms ranking SLO would
    # breach instantly and mean nothing. A second wrap around a caller nests a span inside a span
    # and double-counts. The fusion span sits inside the ranking one by the same rule: the geometry
    # fusion runs inside the ranking of a frame, so the ranking span has always included it.
    latency_tracker: "LatencyTracker | None" = field(default=None, repr=False)
    # ``feasibility_*`` come from ``grasping.feasibility`` via ``apply_orchestrator_overlays``.
    # None (default) is not forwarded, so the calculator keeps its constructor values, i.e. off.
    # Enabling this is expected to cost picks. calculator.py carries an on-box measurement: the
    # ik_quality sub-signal took the UR5e overhead pick 10/10 -> 0/10 at every weight, because the
    # good top-down grasp is itself near-singular, so "demote high condition number" demotes
    # precisely the grasp the pick needs. Wired so the claim can be re-measured through config; it
    # stays default-off regardless of how the number comes out.
    feasibility_config: "FeasibilityScoreConfig | None" = None
    feasibility_weight: float | None = None

    # Optional swept-volume approach/retreat validator. ``None`` (default) is a strict no-op
    # (byte-identical for single-object and for every direct construction). When set and the
    # resolved mode is in ``_approach_path_modes`` (dense-only, set by builders), the orchestrator
    # validates each ranked candidate's approach/retreat sweep against the scene neighbour cloud and
    # executes the first sweep-clear candidate (fall-back), refusing with APPROACH_PATH_BLOCKED if all
    # are blocked. The calculator only checks the final grasp pose; this adds the intermediate sweep.
    approach_path_policy: "ApproachPathPolicy | None" = None
    _approach_path_modes: frozenset[str] = field(default_factory=frozenset, repr=False)

    # Optional :class:`ShadowRouter` carrying a ranking-shadow policy.
    # ``None`` is the silent byte-identical default. When set and the
    # router has a ``ranking_policy`` attached, the orchestrator runs the
    # ranker after the blend block and stashes the typed telemetry on
    # :attr:`_ranking_telemetry`. The ranker never mutates
    # ``best_result``, ``GraspPoint.score``, or any other field that
    # influences execution.
    shadow_router: "ShadowRouter | None" = None

    #: Where a camera on the wrist looks from before the first attempt, in order: each a
    #: :class:`~src.robot.core.JointPositions` or ``"home"`` (:mod:`src.robot.execution.looks`). Set per pick by
    #: whoever runs it. Empty means the pose the arm stands at is the one look. Read only when the frames are placed
    #: by the tool pose (an :class:`EyeInHandFrameResolver`): a fixed camera never moves to look, and its picks are the
    #: picks it always made.
    looks: tuple[Any, ...] = ()
    #: Require that both jaw contact faces of the chosen grasp were seen before gripping: the owner's switch for
    #: safety-critical processes (2026-09-29). Off, the default and the fast one, a look is good enough when its grasp
    #: is valid and carries none of the rescan reasons. On, a look is good enough only once both contact faces were
    #: seen as well, and when no view, declared or generated, showed both, nothing is gripped: the attempt ends
    #: ``faces_unseen``, ``RESCANNED_EXHAUSTED`` (``no_valid_grasp`` to the service), naming the face no view showed
    #: (:attr:`LookedAround.withheld`). A fixed camera never looks around, so on it the grasp its cameras ranked is
    #: gripped only where they showed both contact faces (fused, where the cell fuses its fixed cameras), and the
    #: attempt ends the same way where they did not. A suction hand has no jaw faces and always passes. Like
    #: ``looks``, it stays set on the orchestrator from one pick to the next until whoever set it resets it.
    both_faces: bool = False
    #: What lets this pick push its failed part (``nudge_target``, the owner's rules of 2026-09-29), handed per pick by
    #: the service from its recovery policy and taken back after it: ``None``, the default, and nothing is pushed. Read
    #: only on a wrist camera's pick, after its looks judged the part: every candidate collided (``ALL_COLLIDED``), or,
    #: where approach validation runs, every approach sweep was blocked, and a neighbour stands within 25 mm of the
    #: part in what the looks saw. See :meth:`_push_the_failed_part`.
    push_gate: "PushGate | None" = None
    #: The parts ``next_target`` skips (``ExclusionZones``), the campaign's, handed per pick by the service: a
    #: segmentation of the same label whose centre stands in a zone is not a target, next to the label gate, and a pick
    #: that sees only such parts stops with the zones' sentence rather than try one of them again. A region a task keeps
    #: out (``ExclusionRegion``: the bin it places into, the circle about its drop) applies the same way, to every label
    #: where it names none, the empty one included. ``None``: none.
    exclusion_zones: "ExclusionZones | None" = None
    #: Whether a wrist pick handed looks may generate its one view once they are used up (:meth:`_generated_view`, the
    #: last resort). ``False`` for a pick its task switched multi-view off for (the owner's Q11, 2026-09-30: the first
    #: look only and no generated view), handed per pick by the service and taken back after it, as ``looks`` are.
    generated_view: bool = True
    #: Where a blocker this pick clears is set down: a TCP pose in BASE the owner names, such as a pose taught as "Müll"
    #: over a bin (``blocker.taught_place("Müll")``), released through the place verb as a place always is. ``None``, the
    #: default, sets it down on a free spot of the support the camera saw, away from the part and everything else
    #: (:meth:`_clear_the_blockers`). Like ``looks``, it stays set until whoever set it resets it.
    blocker_place: "Pose | None" = None

    # ---- per pick, while a wrist pick looks around (``look_around``) ----
    #: The earlier looks of this pick, in the order they were visited: what every later look is fused with.
    _look_views: "list[_LookView]" = field(default_factory=list, init=False, repr=False)
    #: The fixed cameras' views, the cameras absent and those that grounded nothing, acquired once per pick at its
    #: first look; ``None`` until then.
    _fixed_views: "tuple[tuple[ObservedView, ...], list[str], list[str]] | None" = field(
        default=None, init=False, repr=False)
    #: The look being ranked, while it is: what ``_fused_scene`` fuses with the earlier looks.
    _current_look: "_LookView | None" = field(default=None, init=False, repr=False)
    #: What the ranking of the current look fused, for its judgement.
    _last_fused_scene: "FusedSceneGeometry | None" = field(default=None, init=False, repr=False)
    #: Per object of the last ranking: its result, its support and the cloud its generator was handed.
    _results_by_object: "dict[int, _RankedObject]" = field(default_factory=dict, init=False, repr=False)
    #: The look sequence whose looks are still held (:meth:`look_around` to :meth:`close_looks`): the only one
    #: :meth:`go_on_with` hands on.
    _open_looked: "LookedAround | None" = field(default=None, init=False, repr=False)
    #: A look sequence handed to the next :meth:`run` (:meth:`go_on_with`), which goes on with it rather than looking
    #: again; consumed by that run, dropped by :meth:`close_looks`.
    _pending_looked: "LookedAround | None" = field(default=None, init=False, repr=False)
    #: What the last wrist pick's looks came to; ``None`` for a fixed camera and before the first look.
    looked_around: "LookedAround | None" = field(default=None, init=False, repr=False)
    #: Holds what a pick keeps for as long as it looks and picks: the frames its wrist camera took at every pose, in the
    #: arm's live planner world (``keep_out.holding_views``), for a pick handed looks. Closed by :meth:`close_looks`.
    _views_scope: ExitStack = field(default_factory=ExitStack, init=False, repr=False)
    #: Whether the arm's live planner world holds the frames of the pick that is looking (what ``holding_views`` said).
    _holding: bool = field(default=False, init=False, repr=False)
    #: What every push this run considered came to, in order (:attr:`pushes`); emptied as each run begins.
    _pushes: "list[PickPush]" = field(default_factory=list, init=False, repr=False)
    #: The part the last run failed on (its label, BASE centre and footprint), what ``next_target`` skips; ``None`` where
    #: the run succeeded or failed on no part. Reset as each run begins, so a run never reports the one before it.
    failed_part: "FailedPart | None" = field(default=None, init=False, repr=False)
    #: Every part a push of this run moved: the points it swept, from where it stood to where it was pushed, and the BASE
    #: XY it was predicted to land at. A later offer of the run whose target stands within ``SAME_PART_RADIUS_MM`` of
    #: that landing is the pushed part, and keeps those points out of the planner world too, since the frames the world
    #: holds from before the push still show the part where it stood. Any other target leaves them in: there the pushed
    #: part is an obstacle like any other.
    _pushed_parts: "list[tuple[np.ndarray, tuple[float, float]]]" = field(default_factory=list, init=False, repr=False)
    #: What every clearing of a blocker this run considered came to, in order (:attr:`blockers`); why the clearing of
    #: this run stopped for good (``""`` while it may go on); and where it set blockers down, BASE x and y, which it never
    #: takes for a blocker again.
    _blockers: "list[BlockerRecord]" = field(default_factory=list, init=False, repr=False)
    _blockers_stopped: str = field(default="", init=False, repr=False)
    _set_down_at: "list[tuple[float, float]]" = field(default_factory=list, init=False, repr=False)
    #: Where a blocker stood whose grasp was refused once the arm left the look, the hand never closed: not tried again.
    _blockers_refused: "list[tuple[float, float]]" = field(default_factory=list, init=False, repr=False)
    #: The support model of the frame being ranked (``support_surfaces.SupportModel``), ``None`` where the arm's live world
    #: reads no support surfaces: what its objects' grasps and a judgement's push read.
    _last_support_model: Any = field(default=None, init=False, repr=False)

    _last_policy_report: PolicyReport | None = field(default=None, init=False, repr=False)
    #: What the grasps the last attempt handed the policy came to (:meth:`_try_the_grasps`), and what its tries read
    #: beside the result while it runs: the looks to come back to and the part's keep-out box.
    _last_tried: "_Tried | None" = field(default=None, init=False, repr=False)
    _tries_looked: "LookedAround | None" = field(default=None, init=False, repr=False)
    _tries_box: "KeepOutBox | None" = field(default=None, init=False, repr=False)
    #: The objects the last ranking iterated, for where the chosen one was seen (``PickReport.target_centre_mm``).
    _last_scene_objects: tuple[Any, ...] = field(default=(), init=False, repr=False)
    _resolved_sampling_mode: GraspSamplingMode = field(
        default=GraspSamplingMode.AUTO, init=False, repr=False
    )
    # ---- live progress + cancellation (both default-off, both byte-identical when unset) ----
    # A caller that wants to watch a pick attaches ``on_progress``; with it unset every emit is one
    # ``is None`` test and no payload is built. Without it a server subscribing to a live pick
    # receives exactly one message, after the pick is over.
    on_progress: "PickProgressListener | None" = None
    # Asked once at the top of each attempt. ``True`` means "do not begin another attempt"; it never
    # interrupts a motion already in flight, because there is no safe way to do that from here and the
    # physical stop is the one that can. Unset leaves the loop a bare ``for``.
    should_cancel: "ShouldCancel | None" = None
    # Per-pick CAMERA->BASE transform for the winning frame, stashed by
    # _best_result_over_segmentations so _execute can map the camera-frame neighbour cloud into the
    # candidate (BASE) frame before the swept-volume check. Reset per pick.
    _pending_camera_to_base: "Transform | None" = field(default=None, init=False, repr=False)
    #: The keep-out scope of the attempt that is running: the target held out of the arm's live
    #: world from the offer until the pick ends or the next frame's offer replaces it. Opened fresh
    #: by every run.
    _keep_out: ExitStack = field(default_factory=ExitStack, init=False, repr=False)
    #: Whether this run already said that its target stays an obstacle, so the line is written once
    #: per run.
    _keep_out_unplaced_said: bool = field(default=False, init=False, repr=False)
    # What the multi-camera geometry fusion actually did this frame (cameras that contributed,
    # objects fused). Empty while fusion is off, so the telemetry stays byte-identical.
    _fusion_geometry_telemetry: dict = field(default_factory=dict, init=False, repr=False)
    # Where the table ended up for the object being evaluated, and whether that height was
    # declared or observed. `SupportResolution.as_telemetry` produced exactly this and had no
    # caller in the tree. Kept on the orchestrator rather than pushed into the record, because
    # `GraspAttemptRecord` is a frozen contract with catalog, KPI and RL consumers and a plane
    # height is not worth opening it; the log line at the resolution site is what an operator
    # needs, and a later change can join this.
    _support_telemetry: dict = field(default_factory=dict, init=False, repr=False)
    #: The other cameras' views from this pick's `_fused_scene`, so the scene object list is built
    #: from the same observation rather than from a second one taken moments later.
    _last_other_views: tuple = field(default=(), init=False, repr=False)
    #: Per ranked frame: the cameras behind each object's fused cloud, by scene index, for the objects
    #: whose fused cloud reached the generator (`PickAttempt.fused_views`), and how many objects of the
    #: frame had one (`PickAttempt.fused_objects`). Emptied at the top of every pick and every ranking,
    #: so a frame that fused nothing never reports the frame before it.
    _fused_views_by_object: "dict[int, tuple[str, ...]]" = field(default_factory=dict, init=False, repr=False)
    _fused_objects_in_frame: int = field(default=0, init=False, repr=False)
    # (config key, BASE points). The walls are fixed geometry; rebuilding them every attempt would
    # be waste in the one path that runs on every pick.
    _container_wall_cache: "tuple[tuple, np.ndarray] | None" = field(
        default=None, init=False, repr=False
    )
    # Per-pick ranking-shadow telemetry. Reset to ``None`` at the top of
    # :meth:`run`; populated at most once (on the attempt that reaches
    # the blend seam). Read by ``AutonomousGraspService`` post-pick and
    # never fed back into the deterministic decision path.
    _ranking_telemetry: "RankingShadowTelemetry | None" = field(
        default=None, init=False, repr=False
    )
    # Per-pick per-candidate log. ``_candidate_log`` = the top-N candidates' 15-key feature
    # rows + a tail aggregate; ``_behavior_candidate_id`` = the executed (rank-0) candidate id. Persisted into
    # ``record.extra`` (rl_candidate_features / rl_behavior_candidate_id) by ``attach_shadow_router`` so the
    # offline RL + OPE/promotion can train + recover the behavior action from real logged picks. Reset
    # per pick; populated at most once in :meth:`_maybe_run_ranking_shadow`, which is shadow-gated, so
    # the default is off and byte-identical. Never fed back into the deterministic decision path.
    _candidate_log: "list[dict[str, object]] | None" = field(
        default=None, init=False, repr=False
    )
    _behavior_candidate_id: "str | None" = field(default=None, init=False, repr=False)
    # Per-pick sequencing-shadow accumulators. ``_sequencing_current_attempts``
    # is a stable reference to the live attempts list; ``_sequencing_states``
    # collects the per-failure :class:`SequencingAttemptState` rows; and
    # ``_sequencing_telemetry`` carries the finalised typed telemetry for the
    # post-pick reader. All three are reset/populated inside the pick loop.
    _sequencing_current_attempts: list[PickAttempt] = field(
        default_factory=list, init=False, repr=False
    )
    _sequencing_states: "list[SequencingAttemptState]" = field(
        default_factory=list, init=False, repr=False
    )
    _sequencing_telemetry: "SequencingShadowTelemetry | None" = field(
        default=None, init=False, repr=False
    )
    # Per-pick perception-budget + recovery shadow telemetry. Reset at the top of the
    # pick and populated at most once by _finalize_perception/recovery (logging-only carriers; never fed back into the
    # deterministic perceive, score, decide, commit, execute path). Read post-pick by attach_shadow_router.
    _perception_telemetry: "PerceptionShadowTelemetry | None" = field(
        default=None, init=False, repr=False
    )
    _recovery_telemetry: "RecoveryShadowTelemetry | None" = field(
        default=None, init=False, repr=False
    )
    # The stateless shadow-telemetry producer + finalizer logic. Owns no
    # per-pick state: the slots above stay here; the aggregator reads loop state via getter-closures
    # and returns telemetry, and the orchestrator's thin delegators assign their own slots
    # (return-and-assign). Built in __post_init__.
    _shadow: "_ShadowTelemetryAggregator" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # ---- resolve grasp sampling mode once per orchestrator ----
        # Precedence: explicit ``grasp_sampling_mode`` wins. When only
        # ``dense_sampling`` is provided the legacy bool meaning holds
        # (``True`` selects DENSE_CLUTTER, ``False`` SINGLE_OBJECT) so old
        # code does not change behavior. The resolved enum is what reaches
        # the calculator at every attempt.
        if self.grasp_sampling_mode is not None:
            self._resolved_sampling_mode = resolve_grasp_sampling_mode(
                self.grasp_sampling_mode
            )
        else:
            self._resolved_sampling_mode = resolve_grasp_sampling_mode(
                self.dense_sampling
            )

        # Build a default policy when none is provided. ``standoff_mm`` and
        # ``retreat_mm`` are forwarded so a caller that sets them on the
        # orchestrator gets them on the policy.
        if self.policy is None:
            self.policy = GraspExecutionPolicy(
                arm=self.arm,
                gripper=self.gripper,
                standoff_mm=self.standoff_mm,
                retreat_mm=self.retreat_mm,
            )

        # The shadow-telemetry aggregator (logic-only; reads loop state via getter-closures-over-self
        # so it always sees end-of-pick values: _sequencing_current_attempts is rebound per pick).
        self._shadow = _ShadowTelemetryAggregator(
            max_attempts_getter=lambda: self.max_attempts,
            shadow_router_getter=lambda: self.shadow_router,
            resolved_mode_getter=lambda: self._resolved_sampling_mode,
        )

    def _resolve_support(self, seg, frame, camera_to_base, looks_cloud=None, support_model=None):  # noqa: ANN001, ANN202
        """The support plane for this pick: the declared height, raised by what the cameras saw.

        Where the arm's live world finds what the parts stand on (``support_model``, contract 7 of the cell fixes), the
        declared height handed on is the higher of the config's and the model's reading under the part's foot
        (``SupportModel.height_under``): a part on the owner's 55 mm mat is planned on the mat, not on the bench under it,
        and a mat that is gone tomorrow is gone from the plan tomorrow. The target's own cloud may raise it further, as
        before.

        The observation is an externally supplied fused target cloud when one is available, because
        a single view of a bin sees the part over the wall and reports its rim as the base: measured
        p90 error 50.6 mm in the bin family against 13.3 mm fused. Otherwise the current view's own
        target cloud is used, which is still correct and only noisier.

        "Fused" here means `external_target_geometry_base_mm`, the cloud a simulator runner hands in
        for one labelled target, and not the multi-camera fusion under `grasping.fusion.geometry`
        that a real cell runs. The two are different seams and this one has never seen the other's
        output, so a cell with two calibrated cameras refines its support plane from a single view. ``support_config`` unset returns ``None``, the
        key is never added, and the path is byte-identical.

        The one exception is a wrist pick's looks: ``looks_cloud`` is the target's cloud fused over two or more views
        of the pick's looks, and it refines the plane in place of the current view's own, under the same extent gate
        in BASE (:data:`_MIN_CLOUD_EXTENT_FOR_SUPPORT_MM`). It holds this view's surface and what the other looks saw of
        the part besides, so the lowest point it reaches is the lowest any of them saw: one look that lost the foot of
        the part behind a neighbour no longer puts the table under what it could see.
        """
        if self.support_config is None:
            return None
        from src.robot.grasping.collision import resolve_support_plane  # noqa: PLC0415

        cfg = self.support_config
        clouds: list = []
        if cfg.refine_from_target:
            if self.external_target_geometry_base_mm is not None and self.target_label is not None:
                clouds.append(self.external_target_geometry_base_mm)
            elif looks_cloud is not None and self._spans_the_support(looks_cloud, "the looks' fused"):
                clouds.append(looks_cloud)
            elif camera_to_base is not None:
                cloud = self._target_cloud_base_mm(seg, frame, camera_to_base)
                if cloud is not None:
                    clouds.append(cloud)
        declared = float(cfg.height_mm)
        floor = cfg.container.floor_height_mm
        under = self._support_under_the_foot(seg, frame, camera_to_base, looks_cloud, support_model)
        if under is not None:
            stands = float(floor) if floor is not None else declared
            if under > stands:
                if under > stands + _SUPPORT_RAISE_SAID_MM:
                    _LOG.info("the camera world finds what the part stands on at %.1f mm under its foot, over the %.1f mm "
                              "declared: its grasps are planned on that", under, stands)
                if floor is not None:
                    floor = under
                else:
                    declared = under
        return resolve_support_plane(
            declared_height_mm=declared,
            container_floor_mm=floor,
            normal=cfg.normal,
            target_clouds_base_mm=clouds or None,
            refine_from_target=cfg.refine_from_target,
        )

    def _support_under_the_foot(  # noqa: ANN001
        self, seg, frame, camera_to_base, looks_cloud, support_model,
    ) -> float | None:
        """The support model's highest local reading under the part's foot (``SupportModel.height_under``), BASE z mm, or
        ``None``: no model, no surface under the part, or no foot to read it under. The foot is the looks' fused cloud
        where there is one, else the part's mask in this frame placed in BASE."""
        if support_model is None or camera_to_base is None:
            return None
        foot = None
        if looks_cloud is not None and np.asarray(looks_cloud).size:
            foot = np.asarray(looks_cloud, dtype=np.float64).reshape(-1, 3)
        else:
            from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

            mask = getattr(seg, "mask", None)
            surface = getattr(frame, "surface_depth_map", None)
            depth = np.asarray(surface if surface is not None else frame.depth_map, dtype=np.float64)
            matrix = getattr(camera_to_base, "to_matrix", None)
            if mask is None or np.asarray(mask).shape != depth.shape[:2] or not callable(matrix):
                return None
            try:
                foot = to_base_mm(np.asarray(mask).astype(bool), depth, np.asarray(frame.intrinsics, dtype=np.float64),
                                  np.asarray(matrix(), dtype=np.float64))
            except (TypeError, ValueError):
                return None
        if foot is None or not foot.shape[0]:
            return None
        value = support_model.height_under(foot[:, :2])
        return None if value is None else float(value)

    def _support_model_of_frame(self, frame: PerceptionFrame, camera_to_base: "Transform | None") -> Any:
        """What the parts of this look stand on, as the arm's live world finds it on the same frames: the frame being
        ranked and every earlier look of the pick, read by the world's own limits and tuning
        (``support_surfaces.find_supports``, contract 7 of the cell fixes). ``None`` where the arm has no live world, its
        world reads no support surfaces or declares no bench, or the frame has no placement: the pick then plans as it
        always did. The parts the pick may go for are held out, so a large flat part is never its own support."""
        world = getattr(self.arm, "live_planner_world", None)
        limits = getattr(world, "limits", None)
        tuning = getattr(world, "tuning", None)
        if camera_to_base is None or limits is None or tuning is None:
            return None
        if not bool(getattr(tuning, "support_surfaces", False)) or getattr(limits, "support_plane_top_mm", None) is None:
            return None
        from src.robot.safety.planning.perceived import DepthView  # noqa: PLC0415
        from src.robot.safety.planning.support_surfaces import find_supports  # noqa: PLC0415

        try:
            matrix = np.asarray(camera_to_base.to_matrix(), dtype=np.float64)
            surface = getattr(frame, "surface_depth_map", None)
            depth = np.asarray(surface if surface is not None else frame.depth_map, dtype=np.float64)
            intrinsics = np.asarray(frame.intrinsics, dtype=np.float64)
            views = [DepthView(surface_depth_mm=depth, intrinsics=intrinsics, camera_to_base=matrix, name="ranked")]
            views += [DepthView(surface_depth_mm=seen.depth, intrinsics=np.asarray(seen.frame.intrinsics, dtype=np.float64),
                                camera_to_base=seen.camera_to_base, name=seen.name)
                      for seen in self._look_views if seen.depth is not None and seen.camera_to_base is not None]
            keep_out = self._parts_held_out_of_the_supports(frame, depth, intrinsics, matrix,
                                                            float(getattr(tuning, "margin_mm", 15.0)))
            return find_supports(views, limits=limits, tuning=tuning, base=self._robot_base(), keep_out=keep_out)
        except Exception as exc:  # noqa: BLE001 (the declared support stands, and says so)
            _LOG.warning("the surfaces the parts stand on could not be found on this look (%s: %s): the declared support "
                         "stands", type(exc).__name__, exc)
            return None

    def _parts_held_out_of_the_supports(
        self, frame: PerceptionFrame, depth: np.ndarray, intrinsics: np.ndarray, matrix: np.ndarray, margin_mm: float,
    ) -> "tuple[KeepOutBox, ...]":
        """The boxes of the parts this look may go for (the prompted label's, else every segmentation), grown by the
        world's margin: a surface more than half of whose pixels lie in one is a part's own top, no support."""
        from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

        boxes: list[KeepOutBox] = []
        for index, seg in enumerate(frame.segmentations):
            if self.target_label is not None and _label_key(seg) != self.target_label:
                continue
            mask = getattr(seg, "mask", None)
            if mask is None or np.asarray(mask).shape != depth.shape[:2]:
                continue
            points = to_base_mm(np.asarray(mask).astype(bool), depth, intrinsics, matrix)
            if not points.shape[0]:
                continue
            low, high = points.min(axis=0), points.max(axis=0)
            place = np.eye(4)
            place[:3, 3] = (low + high) / 2.0
            boxes.append(KeepOutBox.from_matrix(f"part_{index}", place, (high - low) / 2.0 + margin_mm))
        return tuple(boxes)

    def _robot_base(self) -> Any:
        """The robot's base about the BASE axis (``self_envelope.ROBOT_BASES``) for the arm's model, ``None`` where it is
        not known: no support is found on it."""
        from src.robot.safety.planning.self_envelope import ROBOT_BASES  # noqa: PLC0415

        model = getattr(getattr(getattr(self.arm, "config", None), "ur", None), "model", "")
        return ROBOT_BASES.get(str(model or "").lower())

    def _target_cloud_base_mm(self, seg, frame, camera_to_base):  # noqa: ANN001, ANN202
        """This view's target surface in BASE mm, or None when it cannot be built.

        The frame is the whole point of this function, and two conversions can lose it silently:

        * ``camera_to_base`` arrives as a :class:`Transform` from the frame resolver, and
          ``np.asarray(Transform, dtype=float)`` raises rather than yielding a matrix, so a guard on
          ``transform.shape`` alone never sees a Transform: the call dies before it, outside the
          ``try``.
        * ``masked_point_cloud`` returns a :class:`MaskedPointCloud`, not an array. Treating it as
          one is how a camera-frame value reaches a BASE-tagged consumer.

        Both matter here more than anywhere else: the caller feeds this cloud's lowest point to
        ``resolve_support_plane`` under ``refine_from_target``, so a camera-frame slip does not
        produce a wrong number, it produces a depth where a height belongs: measured on the
        multiview gate as a support plane at -311.11 mm (the wrist-camera standoff) against a table
        at 0, which rejected 12 of 12 candidates. So: accept either shape explicitly, and say so
        when it cannot be built.
        """
        import numpy as _np  # noqa: PLC0415

        from src.robot.grasping.geometry.pointcloud import (  # noqa: PLC0415
            masked_point_cloud,
        )

        mask = getattr(seg, "mask", None)
        depth = getattr(frame, "depth_map", None)
        intrinsics = getattr(self.calculator, "camera_matrix", None)
        if mask is None or depth is None or intrinsics is None:
            return None
        try:
            cloud = masked_point_cloud(mask, _np.asarray(depth), _np.asarray(intrinsics))
        except (ValueError, TypeError):
            return None
        points_cam = getattr(cloud, "points_mm", cloud)
        points = _np.asarray(points_cam, dtype=float).reshape(-1, 3)
        if points.size == 0:
            return None
        # The cloud must have seen the support, or it must not be used to locate it.
        #
        # `resolve_support_plane` takes this cloud's lowest point as the plane. That is right when
        # the camera can see down the side of the object to whatever it rests on, and catastrophic
        # when it cannot: a top-down view returns the top face, the plane lands on top of the
        # object, and every candidate below it is rejected as under-the-table. A cloud whose z_min
        # and z_max are both 50.0 mm takes `real_cell --rehearse` from 3/3 to exit 2.
        #
        # Vertical extent is the cheap, honest test for whether the support was observed. It is
        # measured along the support normal rather than along z, so a tilted tray works the same
        # way.
        # And therefore it is measured in BASE, after the transform. `support_config.normal` is a
        # BASE vector and this cloud arrives in CAMERA, so the check used to project base-frame
        # coordinates onto a camera-frame axis: it was reading the camera's own depth span and
        # calling it height. Harmless while the producer flattened the depth, because the span was
        # exactly 0.0 either way and this branch could never fire. With a measured surface it decides
        # whether the support plane is refined, and on an obliquely mounted camera a flat horizontal
        # top face spans a large distance in depth and no height at all, so the guard would clear and
        # the plane would land on top of the object. That is the same family as the 577 mm defect.
        matrix = getattr(camera_to_base, "to_matrix", None)
        transform = _np.asarray(matrix() if callable(matrix) else camera_to_base, dtype=float)
        if transform.shape != (4, 4):
            _LOG.warning(
                "support refinement got a CAMERA->BASE of shape %s, not (4, 4); the target cloud is "
                "not refining the support plane this attempt", transform.shape)
            return None
        base_points = points @ transform[:3, :3].T + transform[:3, 3]
        if not self._spans_the_support(base_points, "single-view"):
            return None
        return base_points

    def _spans_the_support(self, base_points: np.ndarray, which: str) -> bool:
        """Whether a BASE target cloud reaches far enough along the support normal to have seen what it stands on.

        :data:`_MIN_CLOUD_EXTENT_FOR_SUPPORT_MM` along ``support_config.normal``, the gate
        :meth:`_target_cloud_base_mm` sets out; ``which`` names the cloud in the line that says it did not.
        """
        if self.support_config is None:
            return True
        normal = np.asarray(self.support_config.normal, dtype=float).reshape(3)
        length = float(np.linalg.norm(normal))
        points = np.asarray(base_points, dtype=float).reshape(-1, 3)
        if length <= 1e-12 or not points.size:
            return bool(points.size)
        along = points @ (normal / length)
        if float(along.max() - along.min()) < _MIN_CLOUD_EXTENT_FOR_SUPPORT_MM:
            _LOG.debug(
                "%s target cloud spans %.1f mm along the support normal (< %.1f); "
                "it has not observed the support, so the declared height stands",
                which, float(along.max() - along.min()), _MIN_CLOUD_EXTENT_FOR_SUPPORT_MM)
            return False
        return True

    def _target_centre_base_mm(
        self, frame: PerceptionFrame, target_index: int | None, *, fused_cloud_base_mm: "np.ndarray | None" = None,
    ) -> tuple[float, float, float] | None:
        """Where the chosen object was seen, BASE mm, or ``None``: the median of its mask's surface, as ``Locator`` says it.

        From the object the ranking chose (``_last_scene_objects``), with that camera's own depth, intrinsics and
        CAMERA to BASE. The primary camera's measured surface (``surface_depth_map``) is preferred over the grasp
        path's depth, which a producer may flatten to the object's top face plus a penetration. A cell with no CAMERA
        to BASE says nothing: its object would be placed in the camera's frame and called BASE.

        ``fused_cloud_base_mm``, where given and not empty, is the object's cloud fused over the views of a wrist pick,
        and its median is the centre: one view sees one side of the part, and its median leans that way.
        """
        if fused_cloud_base_mm is not None and np.asarray(fused_cloud_base_mm).size:
            middle = np.median(np.asarray(fused_cloud_base_mm, dtype=np.float64).reshape(-1, 3), axis=0)
            return float(middle[0]), float(middle[1]), float(middle[2])
        points = self._target_surface_base_mm(frame, target_index)
        if points is None:
            return None
        centre = np.median(points, axis=0)
        return float(centre[0]), float(centre[1]), float(centre[2])

    def _target_surface_base_mm(self, frame: PerceptionFrame, target_index: int | None) -> "np.ndarray | None":
        """The chosen object's surface in ``frame``, BASE mm, or ``None``: its mask less the pixels behind a depth step.

        From the object the ranking chose (``_last_scene_objects``), with that camera's own depth, intrinsics and
        CAMERA to BASE; the primary camera's measured surface (``surface_depth_map``) is preferred over the grasp path's
        depth. ``None`` with no CAMERA to BASE, no such object, or no depth under its mask.
        """
        from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps  # noqa: PLC0415
        from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

        scene = self._last_scene_objects
        if self._pending_camera_to_base is None or target_index is None or not 0 <= target_index < len(scene):
            return None
        chosen_object = scene[target_index]
        surface_depth = getattr(frame, "surface_depth_map", None)
        depth = chosen_object.depth_map if chosen_object.promoted or surface_depth is None else surface_depth
        try:
            mask = np.asarray(chosen_object.segmentation.mask).astype(bool)
            surface = mask & ~pixels_behind_depth_steps(mask, depth)
            points = to_base_mm(surface, depth, chosen_object.intrinsics, chosen_object.camera_to_base)
        except (AttributeError, TypeError, ValueError):
            return None
        return points if points.shape[0] else None

    def _fusion_of(self, target_index: int | None) -> _FusionFields:
        """What the last ranked frame fused, as a `PickAttempt` keeps it: the target's cameras and the frame's count.

        The views are the ones recorded where the target's fused cloud was handed to the generator, so a target whose
        cloud came from one camera, or never reached the generator, names none; the count is the frame's, whichever
        object the attempt went for.
        """
        views = self._fused_views_by_object.get(target_index, ()) if target_index is not None else ()
        return {"fused_views": views, "fused_objects": self._fused_objects_in_frame}

    # ------------------------------------------------------------------
    # Looks of a wrist pick
    # ------------------------------------------------------------------

    def _perceives_from_the_wrist(self) -> bool:
        """Whether this loop's frames are placed by the tool pose: a camera on the wrist, which looks around."""
        from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver  # noqa: PLC0415

        return isinstance(self.frame_resolver, EyeInHandFrameResolver)

    def look_around(self) -> LookedAround:
        """Look from each of the pick's :attr:`looks` in order, each fused with the ones before, until a grasp is safe.

        At each look the arm moves there through its own judged verb (``move_to_look``: ``move_to_joints``, or the
        gated home), the camera perceives, and the frame is ranked on this look's surfaces fused with every earlier
        look's and with the fixed cameras', acquired once for the pick (:meth:`_fused_looks`). The grasp is computed
        again on each look, so a better one found on more of the part is taken. The looking stops at the first look
        whose grasp is valid and carries none of the rescan reasons (``_RESCAN_REASONS``), and with :attr:`both_faces`
        only once both jaw contact faces of that grasp were seen. A grasp that is still uncertain, or no candidate,
        goes on to the next look. With no looks the pose the arm stands at is the one look.

        Once a look ranks a valid grasp, the part it is on is kept (the views its cloud is made of): a later look
        judges that part, found by association, and a look that does not see it is left out and said. A look that
        finds that part without a valid grasp lets it go.

        Once every declared look is used up, reached or refused before anything was sent, and the grasp is still not
        safe enough, one view is generated, the last resort (:meth:`_generated_view`,
        ``src/robot/execution/generated_view.py``): the judged look turned about the part to the contact face of the
        chosen grasp no view showed, or to the side no look faced when there is no grasp, every turn screened by the arm
        before it moves, the smallest first, and driven on the straight joint line alone; it is perceived and judged as
        a look. With :attr:`both_faces`, a grasp no view showed both contact faces of is then not gripped
        (:attr:`LookedAround.withheld`). Otherwise, where the grasp was ranked on a look other than the one the arm
        stands at, the arm goes back there on the straight joint line; where that line is not clear, would turn a joint
        past the generated view's travel cap, or would not be judged against the world holding the frames of the pick,
        the approach starts from where the arm stands, planned and judged as every approach is. Neither the generated
        view nor the move back runs unless that world holds them.

        The motions: nothing is said to the hand. A camera that cannot vouch for the cell on the way raises, a fault of
        the cell. A controller that stopped ends the looking where it is, with nothing else commanded. A look a guard
        or the planner refused before anything was sent is skipped and said, before a grasp was found as after, and the
        pick goes on with the next look and the views fused so far; a pick that reaches none of its looks ends there
        with nothing perceived, as a refused look always ended it. A motion that failed once it may have been
        commanded, to a look, to the generated view or back, ends the looking there, with nothing else commanded: the
        arm may have moved part of the way. A stop asked for before a look ends the looking there.

        A pick handed looks holds every frame its wrist camera takes in the arm's live planner world
        (``keep_out.holding_views``), from its first look until :meth:`close_looks`, which :meth:`run` calls when the
        pick ends, whether it gripped, failed or raised: every motion of the pick is judged against all of them. A look
        around that raises lets them go itself. What
        the looks came to is kept on :attr:`looked_around`. The next :meth:`run` goes on with it only when it is handed
        on (:meth:`go_on_with`); otherwise that run looks for itself. A camera that is not on the wrist never looks
        around, and asking it to raises ``ValueError`` with nothing moved; so does :attr:`both_faces` with a policy that
        turns every grasp off the faces that were judged (:func:`judged_faces_turned_away`).
        """
        if not self._perceives_from_the_wrist():
            raise ValueError(
                "look_around is a wrist camera's: this pick's frames are not placed by the tool pose, so moving "
                "the arm would not move its camera; a fixed camera perceives where it stands (run())")
        self._refuse_a_turned_grasp()
        try:
            return self._look_around(attempt_index=0)
        except BaseException:
            # A look around that raised hands nothing on, and its frames must not outlive it in the planner world:
            # the caller's next motion would be judged against a pick that is over. `run()` does the same in its
            # `finally`; a caller of this public entry may have no such block.
            self.close_looks()
            raise

    def _refuse_a_turned_grasp(self) -> None:
        """Raise ``ValueError`` before anything moves where :attr:`both_faces` is asked and the policy would close the
        jaws on faces nobody judged (:func:`judged_faces_turned_away`), where the policy names a closing axis and
        twists every grasp as well (``execution_policy.closing_axis_twisted``), and where the closing axis it names (set
        on it after it was built, say) names none (``closing_axis.closing_axis_of``). So is a :attr:`natural_closing_axis`
        set by hand that names none, with the reader's own exception, naming the field."""
        named = getattr(self.policy, "closing_axis", None)
        if named is not None:
            closing_axis_of(named)
        if self.natural_closing_axis is not None:
            try:
                closing_axis_of(self.natural_closing_axis)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"natural_closing_axis: {exc}") from None
        turned = closing_axis_twisted(self.policy) or (judged_faces_turned_away(self) if self.both_faces else "")
        if turned:
            raise ValueError(turned)

    def go_on_with(self, looked: LookedAround) -> None:
        """Hand ``looked`` to the next :meth:`run`, which goes on with its grasp rather than looking again.

        Only the look sequence :meth:`look_around` returned last, while its looks are still held: a judgement nobody
        handed on never reaches a later pick, and one whose looks were closed is not handed on. That run consumes it;
        :meth:`close_looks` drops it unconsumed.
        """
        if looked is not self._open_looked:
            raise ValueError(
                "only the look sequence look_around() returned last, before close_looks(), can be handed on to the "
                "next run")
        self._pending_looked = looked

    def _look_around(self, *, attempt_index: int) -> LookedAround:
        """:meth:`look_around` for attempt ``attempt_index``, the attempt its progress events carry."""
        from src.robot.core.errors import CameraWorldUnavailable  # noqa: PLC0415
        from src.robot.execution.generated_view import move_back_to_look  # noqa: PLC0415
        from src.robot.execution.looks import look_label, looks_of, move_to_look  # noqa: PLC0415
        from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

        self.close_looks()
        self.looked_around = None
        plan: list[tuple[str, Any]] = (
            [(look_label(pose), pose) for pose in looks_of(self.looks)] if self.looks else [(_HERE, None)])
        held = False
        visited: list[str] = []
        dropped: list[str] = []
        refused: list[tuple[str, str]] = []
        first_refusal: tuple[str, MotionReport] | None = None
        judged: _Judgement | None = None
        cancelled = False
        stopped: MotionReport | None = None
        stopped_look = ""
        controller: str | None = None
        for label, pose in plan:
            if self.should_cancel is not None and self.should_cancel():
                cancelled = True
                break
            if pose is not None:
                moved = move_to_look(self.arm, pose)
                if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE and isinstance(
                        moved.result.exception, CameraWorldUnavailable):
                    raise moved.result.exception
                if not moved.ok:
                    controller = self._controller_cannot_move()
                    if controller is not None or not _sent_nothing(moved):
                        # A stopped controller, or a motion that may have moved the arm: the looking ends here.
                        stopped, stopped_look = moved, label
                        break
                    why = f"{moved.status.value}: {moved.message}"
                    refused.append((label, why))
                    if first_refusal is None:
                        first_refusal = (label, moved)
                    # A refused look is used up as a visited one is: the one generated view may still come after it.
                    _LOG.warning(
                        "look %s was refused before anything was sent (%s): it is skipped, and the pick goes on %sto "
                        "its next look or, once none is left and the grasp is still not safe enough, to the one "
                        "generated view", label, why,
                        "with the views fused so far, " if judged is not None and judged.valid else "")
                    continue
                if not held:
                    # Every frame the wrist camera takes from here until the pick ends stays in the planner world,
                    # from the first look the arm stands at: not the frame of the pose the pick started from, which
                    # the motion here was planned on and which no look saw (owner, 2026-09-30).
                    self._holding = self._views_scope.enter_context(holding_views(self.arm))
                    held = True
            visited.append(label)
            judgement, missed = self._look(label, pose, judged, attempt_index=attempt_index)
            if missed:
                dropped.append(label)
            if judgement is not None:
                judged = judgement
                if judged.good:
                    break
        if not cancelled and stopped is None and not visited and first_refusal is not None:
            # Not one look reached: the attempt ends at the first refusal, with nothing perceived.
            stopped_look, stopped = first_refusal
        generated: GeneratedView | None = None
        generated_skipped = ""
        if self.looks and not cancelled and stopped is None and visited and (judged is None or not judged.good):
            # Every declared look is used up, reached or refused before anything was sent (addendum 1), and the grasp is
            # still not safe enough: the one generated view, unless this pick's task switched multi-view off (Q11).
            if not self.generated_view:
                generated_skipped = _NO_VIEW_GENERATED_WITH_MULTI_VIEW_OFF
                _LOG.info("no view is generated: %s", generated_skipped)
            elif self.should_cancel is not None and self.should_cancel():
                cancelled = True
            elif judged is None:
                generated_skipped = "no look saw anything to turn about"
            else:
                judgement, missed, generated = self._generated_view(judged, attempt_index=attempt_index)
                generated_skipped = generated.skipped
                if generated.stopped is not None:
                    stopped, stopped_look, controller = generated.stopped, generated.label, generated.controller
                elif generated.moved:
                    visited.append(generated.label)
                    if missed:
                        dropped.append(generated.label)
                    if judgement is not None:
                        judged = judgement
        withheld = ""
        if not cancelled and stopped is None and judged is not None and judged.valid and self._faces_unseen_by(judged):
            withheld = self._unseen_faces(judged.faces)
        moved_back = move_back_refused = ""
        if (not cancelled and stopped is None and not withheld and judged is not None and judged.valid
                and self._look_views and judged.look is not self._look_views[-1]):
            # The grasp was ranked on a look other than the one the arm stands at: back there, on the straight line.
            if self.should_cancel is not None and self.should_cancel():
                cancelled = True
            else:
                back = move_back_to_look(self.arm, judged.look.joints, look=judged.look.label,
                                         here=self._joints_here(), holding=self._holding,
                                         controller_stopped=self._controller_cannot_move)
                if back.stopped is not None:
                    stopped, stopped_look, controller = back.stopped, judged.look.label, back.controller
                elif back.done:
                    moved_back = judged.look.label
                else:
                    move_back_refused = back.refused
                    _LOG.warning(
                        "the move back to look %s, which saw the part, was refused (%s): the approach starts from where "
                        "the arm stands, at look %s, planned and judged as every approach is, against the world that "
                        "holds every frame of the pick", judged.look.label, back.refused, self._look_views[-1].label)
        if judged is not None:
            # Once, on the judgement the looking ends on, against the looks before it that its part's cloud holds.
            earlier = self._look_views[:self._look_views.index(judged.look)] if judged.look in self._look_views else []
            judged = replace(judged, hand_eye_gap_mm=self._hand_eye_gap_mm(
                judged.look, judged.target_index, [seen for seen in earlier if seen.view is not None], judged.members))
        looked = LookedAround(
            visited=tuple(visited), judged=judged, dropped=tuple(dropped), refused=tuple(refused),
            views=tuple(self._look_views), stopped=stopped, stopped_look=stopped_look, controller=controller,
            cancelled=cancelled,
            generated=generated.label if generated is not None and generated.moved else "",
            generated_view_deg=generated.azimuth_deg if generated is not None and generated.moved else None,
            generated_skipped=generated_skipped, moved_back=moved_back, move_back_refused=move_back_refused,
            withheld=withheld,
        )
        self._say_what_the_looks_came_to(looked)
        self.looked_around = looked
        self._open_looked = looked
        return looked

    def close_looks(self) -> "LookedAround | None":
        """End what a pick's looks hold, the frames held in the planner world among it, and drop a look sequence handed
        on and not picked. Idempotent: what the looks came to stays readable on :attr:`looked_around`."""
        self._views_scope.close()
        self._views_scope = ExitStack()
        self._holding = False
        self._open_looked = None
        self._pending_looked = None
        self._current_look = None
        self._look_views = []
        self._fixed_views = None
        self._last_fused_scene = None
        return self.looked_around

    def _look_name(self, label: str) -> str:
        """A look's view identity: ``rig@look``, the rig alone for a pick handed no looks."""
        rig = self.primary_camera_id or "primary"
        return rig if label == _HERE else f"{rig}@{label}"

    def _look_name_ranked(self) -> str | None:
        """The view identity of the look being ranked, ``None`` outside a look."""
        return self._current_look.name if self._current_look is not None else None

    def _look(
        self, label: str, pose: Any, judged: "_Judgement | None", *, attempt_index: int,
    ) -> "tuple[_Judgement | None, bool]":
        """Perceive where the arm stands, rank this look fused with the earlier ones, and judge it.

        Returns the judgement, or ``None`` when the look adds none, and whether the look missed the part a valid grasp
        was found on. A look that segmented nothing is kept for the record and ranks nothing. For a pick handed looks,
        the configuration the arm stands at is read first and kept with the look: the move back returns to it.
        """
        joints = self._joints_here() if self.looks else None
        frame = self.perception.acquire()
        route = getattr(self.perception, "last_route", None)
        emit(
            self.on_progress, PickStage.PERCEIVED,
            attempt=attempt_index, segmentation_count=len(frame.segmentations),
            route=route[0] if route else None, route_reason=route[1] if route else None,
            **({"extra": {"look": label}} if pose is not None else {}),
        )
        camera_to_base = (
            self.frame_resolver.camera_to_base_for_frame(frame, arm=self.arm)
            if self.frame_resolver is not None else None
        )
        view = self._look_view(label, pose, frame, camera_to_base)
        if joints is not None:
            view = replace(view, joints=joints)
        pinned = judged is not None and judged.valid
        if not frame.segmentations:
            self._look_views.append(view)
            _LOG.info("look %s: nothing was segmented there", label)
            return None, pinned
        self._current_look = view
        try:
            result, index, ordering = self._best_result_over_segmentations(frame)
            judgement = self._judge(view, result, index, ordering, judged)
        finally:
            self._current_look = None
        self._look_views.append(view)
        return judgement, judgement is None and pinned

    def _look_view(
        self, label: str, pose: Any, frame: PerceptionFrame, camera_to_base: "Transform | None",
    ) -> _LookView:
        """What a look keeps of its frame: every segmentation's surface a fusion may give full weight, in BASE.

        A mask less the pixels behind a depth step (``pixels_behind_depth_steps``, as the Locator places a part): at
        45 degrees the band of table the mask bleeds onto at the part's far edge would otherwise join the part's fused
        cloud, and the support-footprint generator reads that cloud as it comes. Less the pixels that see their surface
        at a grazing incidence (``grazing_pixels``), which say more about the edge they were seen past than about where
        the surface is. On the measured surface (``surface_depth_map``) where the frame carries one, because the grasp
        path's depth may be flattened under every mask.
        """
        from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps  # noqa: PLC0415
        from src.robot.grasping.multiview.scene_geometry import (  # noqa: PLC0415
            ObservedView,
            grazing_pixels,
            to_base_mm,
        )

        name = self._look_name(label)
        matrix = None if camera_to_base is None else np.asarray(camera_to_base.to_matrix(), dtype=np.float64)
        measured = getattr(frame, "surface_depth_map", None)
        depth = np.asarray(measured if measured is not None else frame.depth_map, dtype=np.float64)
        if matrix is None or not frame.segmentations:
            return _LookView(label=label, pose=pose, name=name, frame=frame, camera_to_base=matrix, depth=depth)
        intrinsics = np.asarray(frame.intrinsics, dtype=np.float64)
        surfaces: list[np.ndarray | None] = []
        clouds: list[np.ndarray] = []
        for seg in frame.segmentations:
            mask = getattr(seg, "mask", None)
            if mask is None:
                surfaces.append(None)
                clouds.append(np.zeros((0, 3), dtype=np.float64))
                continue
            kept = np.asarray(mask).astype(bool)
            kept &= ~pixels_behind_depth_steps(kept, depth)
            kept &= ~grazing_pixels(kept, depth, intrinsics)
            surfaces.append(kept)
            clouds.append(to_base_mm(kept, depth, intrinsics, matrix))
        blob_objects = tuple(index for index, cloud in enumerate(clouds) if cloud.shape[0])
        view = ObservedView(
            name=name,
            masks=tuple(kept for index, kept in enumerate(surfaces) if index in blob_objects and kept is not None),
            depth_map=depth,
            intrinsics=intrinsics,
            camera_to_base=matrix,
            segmentations=tuple(frame.segmentations[index] for index in blob_objects),
        ) if blob_objects else None
        return _LookView(
            label=label, pose=pose, name=name, frame=frame, camera_to_base=matrix, depth=depth,
            surfaces=tuple(surfaces), clouds=tuple(clouds), view=view, blob_objects=blob_objects,
        )

    def _judge(
        self,
        look: _LookView,
        result: "GraspResult | None",
        index: int | None,
        ordering: "OrderingDecision | None",
        judged: "_Judgement | None",
    ) -> "_Judgement | None":
        """Whether this look's grasp is safe enough to stop looking at, and on which part, from the views fused so far.

        Before any look found a valid grasp the ranking's choice is judged, and the latest look wins: it fused the
        most views. Once one did, the part it is on is kept: this look judges the object that associates with one of
        the views that part's cloud was made of, whatever the ranking chose, and ``None`` says this look did not see it.

        Two rules of the owner's (2026-09-29) read what the looks add. Two looks that call the associated part by
        different labels make the grasp uncertain (``RESCAN_RECOMMENDED``); looks that agree only add to the count
        that agreed. And the jaw contact faces of the chosen grasp are judged on the cloud its generator was handed.
        The hand-eye check is measured once, on the judgement the looking ends on (:meth:`_look_around`).
        """
        fused = self._last_fused_scene
        if judged is not None and judged.valid:
            target = self._pinned_object(fused, judged.members)
            found = self._results_by_object.get(target) if target is not None else None
            if target is None or found is None:
                _LOG.warning(
                    "look %s did not see the part the grasp of look %s is on: it is left out, and that grasp stands",
                    look.label, judged.look.label)
                return None
            if target != index:
                ordering = None
            index, result = target, found.result
        ranked = self._results_by_object.get(index) if index is not None else None
        earlier = [seen for seen in self._look_views if seen.view is not None]
        members: dict[str, int] = {}
        if index is not None:
            own = look.blob_of(index)
            if own is not None:
                members[look.name] = own
            for association in fused.associations if fused is not None else ():
                assigned = association.assignment[index] if index < len(association.assignment) else None
                if assigned is not None:
                    members[association.view] = assigned
            left_out = [seen.label for seen in earlier if seen.name not in members]
            if left_out:
                _LOG.info(
                    "look %s: what look(s) %s saw did not associate with its target and is left out of its cloud",
                    look.label, ", ".join(left_out))
        member_looks = ((look.label,) if index is not None else ()) + tuple(
            seen.label for seen in earlier if seen.name in members)
        own_cloud = look.clouds[index] if index is not None and 0 <= index < len(look.clouds) else None
        cloud = ranked.cloud_base_mm if ranked is not None and ranked.cloud_base_mm is not None else own_cloud
        if cloud is not None and not cloud.size:
            cloud = None

        reasons = tuple(result.reasons) if result is not None else ()
        label = self._label_of(index)
        from src.robot.grasping.multiview.association import label_agreement, label_agreement_said  # noqa: PLC0415

        agreed, disagree = label_agreement(label, [
            (seen.label, seen.label_of_blob(members[seen.name]) if seen.name in members else "") for seen in earlier])
        labels_said = label_agreement_said(label, [
            (seen.label, seen.label_of_blob(members[seen.name])) for seen in earlier if seen.name in members],
        ) if index is not None else ""
        if disagree:
            reasons = tuple(dict.fromkeys((*reasons, GraspFailureReason.RESCAN_RECOMMENDED)))
            _LOG.warning(
                "look %s calls its target %r and %s the same part otherwise (%s): the grasp is uncertain",
                look.label, label, "an earlier look calls" if len(disagree) == 1 else "earlier looks call",
                ", ".join(f"look {seen}: {other!r}" for seen, other in disagree))

        faces = self._jaw_faces(result, cloud, ranked.support_height_mm if ranked is not None else None)
        valid = result is not None and result.is_success
        uncertain = bool(set(reasons) & _RESCAN_REASONS)
        faces_ok = not self.both_faces or not self._hand_has_jaw_faces() or (faces is not None and faces.both)
        good = valid and not uncertain and faces_ok
        _say_the_look(look.label, valid=valid, uncertain=uncertain, good=good, reasons=reasons,
                      faces=faces if self.both_faces else None, faces_asked=self.both_faces)
        return _Judgement(
            look=look, result=result, target_index=index, ordering=ordering, target_cloud_base_mm=cloud,
            members=members, member_looks=member_looks, reasons=reasons, faces=faces, good=good,
            labels_agreed=agreed, labels_disagree=tuple(disagree), labels_said=labels_said, hand_eye_gap_mm=None,
            camera_to_base=self._pending_camera_to_base, scene_objects=self._last_scene_objects,
            fused_views_by_object=dict(self._fused_views_by_object),
            fused_objects_in_frame=self._fused_objects_in_frame,
            fusion_telemetry=dict(self._fusion_geometry_telemetry),
            support_telemetry=dict(self._support_telemetry),
            debug_image_png=getattr(self.calculator, "last_debug_image_png", None),
            fused_scene=fused,
            support_height_mm=ranked.support_height_mm if ranked is not None else None,
            ranked=ranked,
            support_model=self._last_support_model,
        )

    @staticmethod
    def _pinned_object(fused: "FusedSceneGeometry | None", members: Mapping[str, int]) -> int | None:
        """The object of this look that the kept part's views associate with, by most of them; ``None`` for none."""
        if fused is None:
            return None
        votes: dict[int, int] = {}
        for association in fused.associations:
            blob = members.get(association.view)
            if blob is None:
                continue
            for index, assigned in enumerate(association.assignment):
                if assigned == blob:
                    votes[index] = votes.get(index, 0) + 1
        return min(votes, key=lambda index: (-votes[index], index)) if votes else None

    def _label_of(self, index: int | None) -> str:
        """The label of object ``index`` of the last ranking, lower case, ``""`` when it carries none."""
        if index is None or not 0 <= index < len(self._last_scene_objects):
            return ""
        segmentation = getattr(self._last_scene_objects[index], "segmentation", None)
        return str(getattr(segmentation, "label", "") or "").strip().lower()

    def _hand_has_jaw_faces(self) -> bool:
        """Whether the hand closes jaws on two faces: every model but a suction cup's."""
        from src.robot.grasping.collision.gripper_model import SuctionCupGripperModel  # noqa: PLC0415

        return not isinstance(self.gripper_model, SuctionCupGripperModel)

    def _grips_only_the_judged_grasp(self) -> bool:
        """Whether the grasp gripped must be the one whose jaw contact faces were judged: :attr:`both_faces` asked on a
        hand that closes jaws on two faces. The reranks then stand down and the approach check falls back to no other
        candidate, since only the calculator's best had its faces judged."""
        return self.both_faces and self._hand_has_jaw_faces()

    def _jaw_faces(
        self, result: "GraspResult | None", cloud: "np.ndarray | None", support_height_mm: float | None,
    ) -> "JawFacesSeen | None":
        """Whether each jaw's contact face of the chosen grasp was seen in ``cloud`` (``unseen_side.jaw_faces_seen``).

        The contact patch is the hand's own, off its gripper model (the 2F-85's defaults when none is configured).
        ``None`` for no valid grasp, no cloud, a suction hand, a grasp not in BASE, or one the check cannot read.
        """
        from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel  # noqa: PLC0415
        from src.robot.grasping.multiview.unseen_side import jaw_faces_seen  # noqa: PLC0415

        best = result.best if result is not None else None
        if best is None or cloud is None or best.frame is not GraspFrame.BASE or not self._hand_has_jaw_faces():
            return None
        jaw = self.gripper_model if isinstance(self.gripper_model, ParallelJawGripperModel) \
            else ParallelJawGripperModel()
        try:
            return jaw_faces_seen(
                cloud, centre_mm=best.position, approach=best.approach, closing_axis=best.axis,
                grip_width_mm=float(best.grip_width_mm), pad_ahead_mm=float(jaw.pad_ahead_mm),
                pad_behind_mm=max(0.0, float(jaw.pad_length_mm) - float(jaw.pad_ahead_mm)),
                finger_width_mm=float(jaw.finger_width_mm), support_height_mm=support_height_mm,
            )
        except ValueError as exc:
            _LOG.info("the jaw contact faces of the chosen grasp could not be judged: %s", exc)
            return None

    @staticmethod
    def _hand_eye_gap_mm(
        look: _LookView, index: int | None, earlier: "list[_LookView]", members: Mapping[str, int],
    ) -> float | None:
        """The median distance between ``look``'s surface of object ``index`` and the earlier looks' (hand-eye check).

        Over the surface both saw (``association.nearest_surface_distances_mm``: each point's nearest point of the other
        look within reach, on a surface that faces the same way, each turned toward the camera that saw it), pooled
        over every earlier look whose surface of the part associated with this one's. ``None`` when fewer than
        ``association.HAND_EYE_MIN_POINTS`` distances were measured: two looks that shared no face say nothing about the
        calibration.
        """
        from src.robot.grasping.multiview.association import (  # noqa: PLC0415
            HAND_EYE_MIN_POINTS,
            nearest_surface_distances_mm,
        )

        own_cloud = look.clouds[index] if index is not None and 0 <= index < len(look.clouds) else None
        if own_cloud is None or not own_cloud.size or look.camera_to_base is None:
            return None
        found = [
            nearest_surface_distances_mm(
                own_cloud, seen.cloud_of_blob(members[seen.name]), seen_from_mm=look.camera_to_base[:3, 3],
                other_seen_from_mm=seen.camera_to_base[:3, 3])
            for seen in earlier if seen.name in members and seen.camera_to_base is not None
        ]
        pooled = np.concatenate(found) if found else np.zeros(0)
        return float(np.median(pooled)) if pooled.size >= HAND_EYE_MIN_POINTS else None

    def _say_what_the_looks_came_to(self, looked: LookedAround) -> None:
        """One account per pick of where the looks ended, in the owner's words: a safe grasp point, not a scan. It names
        how the looks that saw the part agreed on its label (``label agreed in N of M looks``, the owner's decision 4,
        2026-09-30), which changes nothing: only a disagreement acts, on the look that meets it."""
        from src.robot.grasping.multiview.association import HAND_EYE_DRIFT_WARN_MM  # noqa: PLC0415

        judged = looked.judged
        if judged is not None and judged.hand_eye_gap_mm is not None:
            if judged.hand_eye_gap_mm > HAND_EYE_DRIFT_WARN_MM:
                _LOG.warning(
                    "the looks measure the part's shared surface %.1f mm apart (median), more than %.1f mm: the "
                    "hand-eye calibration may have drifted (a camera loosened on the wrist?). Nothing is changed "
                    "for it; check the calibration", judged.hand_eye_gap_mm, HAND_EYE_DRIFT_WARN_MM)
            else:
                _LOG.info("hand-eye check: the looks measure the part's shared surface %.1f mm apart (median)",
                          judged.hand_eye_gap_mm)
        if not self.looks or looked.cancelled or looked.stopped is not None:
            return
        looked_from = ", ".join(looked.visited) or "no look"
        if judged is None:
            _LOG.info("no look saw anything to pick (looked from %s)", looked_from)
            return
        labels = f"; {judged.labels_said}" if judged.labels_said else ""
        if judged.good:
            _LOG.info("the looks stop at look %s, whose grasp is safe enough (looked from %s)%s", judged.look.label,
                      looked_from, labels)
            return
        if not judged.valid:
            _LOG.info("no look ranked a grasp (looked from %s); the attempt ends without one%s", looked_from, labels)
            return
        if looked.withheld:
            # Not gripped: the attempt says why (`_faces_unseen`).
            _LOG.info("the looks end on the grasp of look %s, which is not gripped (looked from %s)%s", judged.look.label,
                      looked_from, labels)
            return
        reached = len(looked.visited) - (1 if looked.generated else 0)
        views = ", ".join(judged.member_looks) or judged.look.label
        _LOG.info("the looks reached (%d of %d)%s gave no grasp safe enough to stop at; the pick goes on with the grasp "
                  "of look %s on the views of %s%s", reached, reached + len(looked.refused),
                  f" and the generated view {looked.generated}" if looked.generated else "", judged.look.label, views,
                  labels)

    def _joints_here(self) -> "JointPositions | None":
        """The configuration the arm stands at, read off it; ``None``, and said, where it cannot say."""
        from src.robot.core.errors import RobotError  # noqa: PLC0415

        try:
            return self.arm.get_joint_positions()
        except (RobotError, RuntimeError, OSError) as exc:
            _LOG.warning("the arm could not say where it stands (%s: %s)", type(exc).__name__, exc)
            return None

    def _generated_view(
        self, judged: _Judgement, *, attempt_index: int,
    ) -> "tuple[_Judgement | None, bool, GeneratedView]":
        """The one view a wrist pick generates once its declared looks are used up, then perceived and judged as a look.

        Aimed by :meth:`_aim`, turned from the judged look (its camera, the tool pose stamped at its shutter, its lens
        and image), and moved to by ``generated_view.go_to_generated_view`` from where the arm stands, with the arm's
        home for the cable window, whether the pick's frames are held, and this loop's controller diagnosis: every
        turn screened by the arm before it moves, the smallest first, the straight joint line alone. Returns the look's
        judgement and whether it missed the kept part (as :meth:`_look`), and what the view came to.
        """
        from src.robot.execution.generated_view import GeneratedView, go_to_generated_view  # noqa: PLC0415

        aim, why = self._aim(judged)
        tool = getattr(judged.frame, "tool_pose", None)
        here = self._joints_here() if aim is not None else None
        if aim is not None and (tool is None or judged.look.camera_to_base is None):
            why = f"the frame of look {judged.look.label} carries no tool pose or camera placement to turn"
        elif aim is not None and here is None:
            why = "the arm could not say where it stands"
        if why:
            _LOG.info("no view is generated: %s", why)
            return None, False, GeneratedView(skipped=why)
        assert aim is not None and tool is not None and here is not None  # narrowed above
        height, width = np.asarray(judged.frame.depth_map).shape[:2]
        generated = go_to_generated_view(
            self.arm, aim=aim, camera_to_base_mm=judged.look.camera_to_base, tool_to_base_mm=tool.to_matrix(),
            intrinsics=np.asarray(judged.frame.intrinsics, dtype=np.float64), image_size=(int(width), int(height)),
            here=here, home=getattr(self.arm, "home_joint_positions", None), holding=self._holding,
            controller_stopped=self._controller_cannot_move,
        )
        if not generated.moved:
            return None, False, generated
        assert generated.joints is not None  # moved
        judgement, missed = self._look(generated.label, generated.joints, judged, attempt_index=attempt_index)
        return judgement, missed, generated

    def _aim(self, judged: _Judgement) -> "tuple[ViewAim | None, str]":
        """What the generated view is to show, or ``None`` and why there is nothing to show (the owner, 2026-09-29).

        With a grasp, the jaw contact faces of the chosen grasp no view showed, turning about its centre; a grasp both
        of whose faces were seen, or whose faces could not be judged (a suction hand among them), leaves nothing to
        show. With no grasp at all, the side of the target no look faced, about the middle of its cloud.
        """
        from src.robot.execution.generated_view import aim_at_faces, aim_at_unseen_side  # noqa: PLC0415

        best = judged.result.best if judged.valid and judged.result is not None else None
        if best is not None:
            if judged.faces is None:
                return None, "the jaw contact faces of the chosen grasp could not be judged, so there is no face to show"
            if not judged.faces.missing:
                return None, "both jaw contact faces of the chosen grasp were seen, so there is no face to show"
            return aim_at_faces(judged.faces.missing, grasp_centre_mm=best.position), ""
        if judged.target_cloud_base_mm is None:
            return None, "the target has no cloud to turn about"
        centres = [seen.camera_to_base[:3, 3] for seen in self._look_views if seen.camera_to_base is not None]
        try:
            return aim_at_unseen_side(judged.target_cloud_base_mm, camera_centres_mm=centres), ""
        except ValueError as exc:
            return None, f"no side of the target can be told unseen ({exc})"

    def _faces_unseen_by(self, judged: _Judgement) -> bool:
        """Whether :attr:`both_faces` was asked and no view showed both jaw contact faces of the judged grasp."""
        return (self.both_faces and self._hand_has_jaw_faces()
                and not (judged.faces is not None and judged.faces.both))

    @staticmethod
    def _unseen_faces(faces: "JawFacesSeen | None") -> str:
        """Which jaw contact face of the chosen grasp no view showed, in the owner's words."""
        if faces is None:
            return "the jaw contact faces of the chosen grasp could not be judged on what the views saw"
        missing = faces.missing
        jaws = " and ".join(f"jaw {1 if face.side < 0 else 2}" for face in missing)
        if len(missing) == 1:
            return f"the contact face at {jaws} of the chosen grasp was not seen from any view"
        return f"the contact faces at {jaws} of the chosen grasp were not seen from any view"

    def _adopt(
        self, judged: _Judgement,
    ) -> "tuple[PerceptionFrame, GraspResult | None, int | None, OrderingDecision | None]":
        """Go on with the judged look's grasp: its frame, and the per-frame state its ranking left, restored.

        Every look ranked after it overwrote that state, and the rest of the attempt reads it: the CAMERA to BASE the
        planner world places the target by, the objects the target's centre is read from, what was fused for it, the
        support it was computed against, and the calculator's overlay of it.
        """
        self._pending_camera_to_base = judged.camera_to_base
        self._last_scene_objects = judged.scene_objects
        self._fused_views_by_object = dict(judged.fused_views_by_object)
        self._fused_objects_in_frame = judged.fused_objects_in_frame
        self._fusion_geometry_telemetry = dict(judged.fusion_telemetry)
        self._support_telemetry = dict(judged.support_telemetry)
        if judged.debug_image_png is not None and hasattr(self.calculator, "last_debug_image_png"):
            self.calculator.last_debug_image_png = judged.debug_image_png  # type: ignore[attr-defined]
        return judged.frame, judged.result, judged.target_index, judged.ordering

    def _look_refused(
        self, attempt_index: int, looked: LookedAround, attempts: list[PickAttempt],
    ) -> PickReport:
        """The attempt of a wrist pick whose looks ended on a motion to a look: nothing perceived there, nothing else
        commanded. ``CONTROLLER_NOT_OPERATIONAL`` when the controller cannot move, ``EXECUTION_FAILED`` otherwise: no
        look reached, a motion the verb refused before any command, or a look motion that failed once it may have been
        commanded."""
        from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

        assert looked.stopped is not None  # the caller's branch
        fields = _look_motion_fields(looked.stopped)
        if looked.controller is not None:
            _LOG.error("pick aborted on the way to look %s: %s", looked.stopped_look, looked.controller)
            outcome = PickOutcome.CONTROLLER_NOT_OPERATIONAL
        elif _sent_nothing(looked.stopped):
            _LOG.error("the pick reached none of its looks (%s); nothing else is commanded",
                       "; ".join(f"{look}: {why}" for look, why in looked.refused))
            outcome = PickOutcome.EXECUTION_FAILED
        elif looked.stopped.outcome is MotionOutcome.REFUSED:
            # Refused by the motion verb before it commanded anything (a link not open, a camera world the arm would
            # refuse, a route that does not run, an arm that did not come to rest): nothing was sent, and the cell is
            # not fit for the motion, so the pick still ends there.
            _LOG.error("the motion to look %s was refused before any command (%s: %s): nothing was sent, and the pick "
                       "ends there with nothing else commanded", looked.stopped_look,
                       looked.stopped.status.value, looked.stopped.message)
            outcome = PickOutcome.EXECUTION_FAILED
        else:
            _LOG.error("the motion to look %s failed once it may have been commanded (%s: %s): the arm may have moved "
                       "part of the way, so the pick ends there and nothing else is commanded", looked.stopped_look,
                       looked.stopped.status.value, looked.stopped.message)
            outcome = PickOutcome.EXECUTION_FAILED
        action = "look_refused"
        emit(
            self.on_progress, PickStage.ATTEMPT_FINISHED,
            attempt=attempt_index, action=action, outcome=str(outcome),
            extra={"look": looked.stopped_look},
            **{k: v for k, v in fields.items() if v is not None},
        )
        attempts.append(PickAttempt(
            attempt_index=attempt_index,
            target_index=None,
            reasons=(GraspFailureReason.CONTROLLER_NOT_OPERATIONAL,) if looked.controller is not None else (),
            score=0.0,
            action=action,
            ordering_decision=None,
            fused_views=(),
            fused_objects=0,
            **fields,
        ))
        return PickReport(outcome=outcome, attempts=tuple(attempts))

    def _faces_unseen(
        self, attempt_index: int, attempts: list[PickAttempt], *, withheld: str, result: GraspResult,
        target_index: int | None, ordering: "OrderingDecision | None", looked: "LookedAround | None" = None,
    ) -> PickReport:
        """The attempt of a pick that asked for both jaw contact faces (:attr:`both_faces`), when no view showed both.

        Nothing is gripped and nothing else is commanded: the owner's fail-closed ending for safety-critical processes
        (2026-09-29). ``RESCANNED_EXHAUSTED`` with ``NO_VALID_GRASP``, which the service reports as ``no_valid_grasp``,
        and ``withheld``, the reason naming the face no view showed, on the event (and, for a wrist pick, on
        :attr:`LookedAround.withheld`, its ``looked``). ``result`` is the grasp that was not gripped.
        """
        _LOG.error("%s: both_faces asks for both before gripping, so nothing is gripped and nothing else is commanded",
                   withheld)
        action = "faces_unseen"
        reasons = (GraspFailureReason.NO_VALID_GRASP,)
        emit(
            self.on_progress, PickStage.ATTEMPT_FINISHED,
            attempt=attempt_index, action=action, outcome=str(PickOutcome.RESCANNED_EXHAUSTED),
            target_index=target_index, reasons=tuple(str(reason) for reason in reasons),
            extra={"withheld": withheld, **self._looks_extra(looked)},
        )
        attempts.append(PickAttempt(
            attempt_index=attempt_index, target_index=target_index, reasons=reasons,
            score=float(result.top_score), action=action, ordering_decision=ordering,
            **self._fusion_of(target_index), withheld=withheld,
        ))
        return PickReport(
            outcome=PickOutcome.RESCANNED_EXHAUSTED, attempts=tuple(attempts),
            selected_telemetry=dict(getattr(result, "telemetry", None) or {}),
        )

    def _faces_the_fixed_cameras_did_not_show(
        self, frame: PerceptionFrame, result: GraspResult, target_index: int | None,
    ) -> str:
        """For a fixed-camera pick asked for both jaw contact faces (:attr:`both_faces`): which face of the chosen grasp
        its cameras did not show, in the owner's words; ``""`` when they showed both.

        Judged as a wrist look is (``unseen_side.jaw_faces_seen``), on the cloud the generator was handed for the
        target, fused over the fixed cameras where the cell fuses them, else on the target's surface in this frame, the
        pixels behind a depth step left out, against the support the grasp was computed on. Faces that cannot be judged
        are not seen: the switch fails closed.
        """
        ranked = self._results_by_object.get(target_index) if target_index is not None else None
        cloud = ranked.cloud_base_mm if ranked is not None else None
        if cloud is None:
            cloud = self._target_surface_base_mm(frame, target_index)
        faces = self._jaw_faces(result, cloud, ranked.support_height_mm if ranked is not None else None)
        return "" if faces is not None and faces.both else self._unseen_faces(faces)

    def _fused_target(self, judged: _Judgement) -> "np.ndarray | None":
        """The judged target's cloud where the pick was handed looks and more than one view's surface of it is in it,
        the one the planner world and the target's centre take; ``None`` otherwise, and one view's frame speaks for
        itself as it always did. A pick handed no looks keeps its own frame for both even where a fixed camera is fused
        into its grasp, as it did before a wrist pick could look around."""
        cloud = judged.target_cloud_base_mm
        if not self.looks or len(judged.members) < 2 or cloud is None or not cloud.size:
            return None
        return cloud

    def _looks_extra(self, looked: "LookedAround | None") -> dict[str, Any]:
        """What the looks came to, as a progress event carries it; empty unless the pick was handed looks."""
        if looked is None or not self.looks:
            return {}
        pair = looked.jaw_faces_seen
        return {
            "looks": list(looked.visited),
            "looks_fused": list(looked.looks_fused),
            "jaw_faces_seen": list(pair) if pair is not None else None,
        }

    def run(self) -> PickReport:
        """Execute the loop and return a :class:`PickReport`.

        Thin wrapper around :meth:`_execute_pick` that finalises the
        sequencing / perception / recovery shadow telemetry in a
        ``try/finally`` block so every terminal branch surfaces the
        typed carriers for the post-pick reader.

        :attr:`both_faces` with a policy that turns every grasp off the faces that were judged raises ``ValueError``
        first, before anything is perceived or moved (:func:`judged_faces_turned_away`).
        """

        self._refuse_a_turned_grasp()
        emit(self.on_progress, PickStage.PICK_STARTED, attempt_total=self.max_attempts)
        report: PickReport | None = None
        self._keep_out = ExitStack()
        self._keep_out_unplaced_said = False
        # What this run pushed, which part it failed on and what it swept out of the world are this run's alone.
        self._pushes = []
        self.failed_part = None
        self._pushed_parts = []
        # So are the blockers it cleared, and why their clearing stopped.
        self._blockers = []
        self._blockers_stopped = ""
        self._set_down_at = []
        self._blockers_refused = []
        if self._pending_looked is None:
            # A pick that was not handed its looks says nothing about the last pick's.
            self.looked_around = None
        try:
            report = self._execute_pick()
            return report
        finally:
            # First, so the target is back in the world before anything else this pick left behind
            # is read.
            self._keep_out.close()
            self.close_looks()
            self._finalize_sequencing_shadow()
            self._finalize_perception_shadow()
            self._finalize_recovery_shadow()
            # In the `finally` so a raising pick still closes its event stream: a console that only
            # ever saw PICK_STARTED would show a spinner forever, which reads as "the arm is still
            # moving", the single most misleading thing a robot UI can display.
            emit(
                self.on_progress, PickStage.PICK_FINISHED,
                outcome=str(report.outcome) if report is not None else "raised",
            )

    def _execute_pick(self) -> PickReport:
        """Run the deterministic pick loop."""
        attempts: list[PickAttempt] = []
        # The shadow finaliser reads the same list via this stable reference.
        self._sequencing_current_attempts = attempts
        # Reset the per-pick shadow accumulators (see the field
        # declarations for what each slot carries).
        self._pending_camera_to_base = None
        self._fused_views_by_object = {}
        self._fused_objects_in_frame = 0
        self._ranking_telemetry = None
        self._candidate_log = None
        self._behavior_candidate_id = None
        self._sequencing_states = []
        self._sequencing_telemetry = None
        self._perception_telemetry = None
        self._recovery_telemetry = None
        looking = self._perceives_from_the_wrist()
        for attempt_index in range(self.max_attempts):
            # "Stop" means do not begin another attempt. Asked here, before perception, so a cancel
            # costs at most the attempt already running; it never interrupts a motion in flight,
            # which this loop has no safe way to do and the physical stop button does.
            if self.should_cancel is not None and self.should_cancel():
                emit(self.on_progress, PickStage.CANCELLED, attempt=attempt_index)
                return PickReport(
                    outcome=PickOutcome.CANCELLED,
                    attempts=tuple(attempts),
                )
            emit(
                self.on_progress, PickStage.ATTEMPT_STARTED,
                attempt=attempt_index, attempt_total=self.max_attempts,
            )
            looked: LookedAround | None = None
            fused_target: np.ndarray | None = None
            if looking:
                # A camera on the wrist looks from every look of the pick, fused, before it ranks: the
                # attempt goes on with the grasp the looks judged, from the frame it was ranked on. A
                # pick handed looks never comes back here for a second attempt (its next view was its
                # next look); one handed none comes back to rescan where it stands, as it always did.
                # A look sequence is gone on with only when it was handed on (`go_on_with`).
                looked = self._pending_looked
                self._pending_looked = None
                if looked is None:
                    looked = self._look_around(attempt_index=attempt_index)
                if looked.cancelled:
                    emit(self.on_progress, PickStage.CANCELLED, attempt=attempt_index)
                    return PickReport(outcome=PickOutcome.CANCELLED, attempts=tuple(attempts))
                if looked.stopped is not None:
                    return self._look_refused(attempt_index, looked, attempts)
                if looked.judged is None:
                    attempts.append(PickAttempt(
                        attempt_index=attempt_index, target_index=None,
                        reasons=(GraspFailureReason.EMPTY_MASK,), score=0.0, action="exhausted",
                    ))
                    return PickReport(outcome=PickOutcome.NO_PERCEPTION, attempts=tuple(attempts))
                frame, best_result, target_index, ordering_decision = self._adopt(looked.judged)
                if looked.withheld:
                    # Both jaw contact faces were asked for and no view showed both: nothing is gripped.
                    assert best_result is not None  # a judgement that withholds a grasp ranked one
                    return self._faces_unseen(
                        attempt_index, attempts, withheld=looked.withheld, result=best_result,
                        target_index=target_index, ordering=ordering_decision, looked=looked)
                # The target's cloud fused over the views of the pick: what the planner world leaves out, and
                # where the part is said to be. The approach starts where the looks left the arm (the look the
                # grasp was ranked on, or where the move back to it was refused), judged against every frame.
                fused_target = self._fused_target(looked.judged)
            else:
                frame = self.perception.acquire()
                # Read after acquire, so it describes the frame that was just grounded rather than the
                # previous one. Duck-typed and optional: a source with no routed backend returns None and
                # the event is exactly the one this loop has always emitted.
                _route = getattr(self.perception, "last_route", None)
                emit(
                    self.on_progress, PickStage.PERCEIVED,
                    attempt=attempt_index, segmentation_count=len(frame.segmentations),
                    route=_route[0] if _route else None,
                    route_reason=_route[1] if _route else None,
                )
                if not frame.segmentations:
                    attempts.append(
                        PickAttempt(
                            attempt_index=attempt_index,
                            target_index=None,
                            reasons=(GraspFailureReason.EMPTY_MASK,),
                            score=0.0,
                            action="exhausted",
                        )
                    )
                    return PickReport(
                        outcome=PickOutcome.NO_PERCEPTION,
                        attempts=tuple(attempts),
                    )

                best_result, target_index, ordering_decision = (
                    self._best_result_over_segmentations(frame)
                )
            self._offer_masks_to_planner_world(frame, target_index, target_points_base_mm=fused_target)
            # What the looks came to, on the events a console reads. Empty unless the pick was
            # handed looks, so every other pick emits exactly the events it emitted before.
            looks_extra = self._looks_extra(looked)
            if best_result is None or not best_result.is_success:
                reasons = best_result.reasons if best_result is not None else ()
                # Every candidate of the part collided with something, after the looks judged it: where the service
                # handed this pick a push gate, a neighbour stands within 25 mm and every guard holds, the part is
                # pushed away from it, and the next attempt picks from the look taken again after the push.
                push_extra: dict[str, Any] = {}
                if looked is not None and GraspFailureReason.ALL_COLLIDED in reasons:
                    considered_before = len(self._pushes)
                    # A neighbour that can be taken away is taken away first; the push comes only where none can be
                    # (the owner, 2026-10-02). The next attempt goes on with the look taken again after it.
                    cleared = self._clear_the_blockers(
                        attempt_index, attempts, looked, frame, target_index, reasons=tuple(reasons),
                        fused_target=fused_target)
                    if isinstance(cleared, PickReport):
                        return cleared
                    if cleared:
                        continue
                    pushed = self._push_the_failed_part(
                        attempt_index, attempts, looked, frame, target_index, trigger="all_collided",
                        reasons=tuple(reasons), fused_target=fused_target)
                    if isinstance(pushed, PickReport):
                        return pushed
                    if pushed:
                        continue
                    # Only a push this attempt considered: an earlier attempt's refusal is not this one's.
                    if len(self._pushes) > considered_before:
                        push_extra = self._push_refused_extra()
                # A reason worth another look gets a fresh frame, without motion; any other ends the
                # pick. A wrist pick handed looks has had its other looks already.
                action = "rescan" if set(reasons) & _RESCAN_REASONS and not (looking and self.looks) else "exhausted"
                # A refusal an operator can act on carries what was seen, not just what failed.
                # Only the label gate populates these keys, so every other rejection path emits
                # exactly the event it emitted before.
                seen_extra = (
                    {
                        "target_label": best_result.telemetry["target_label"],
                        "labels_seen": list(best_result.telemetry["labels_seen"]),
                    }
                    if best_result is not None
                    and GraspFailureReason.TARGET_LABEL_NOT_FOUND in reasons
                    else {}
                )
                # Only parts next_target skips were seen: the pick stops, and says why.
                excluded = self._only_excluded_of(best_result)
                if excluded:
                    seen_extra["excluded"] = excluded
                    _LOG.warning("%s", excluded)
                emit(
                    self.on_progress, PickStage.NO_CANDIDATE,
                    attempt=attempt_index, target_index=target_index,
                    reasons=tuple(str(r) for r in reasons), action=action,
                    **({"extra": {**seen_extra, **looks_extra, **push_extra}}
                       if seen_extra or looks_extra or push_extra else {}),
                )
                attempts.append(
                    PickAttempt(
                        attempt_index=attempt_index,
                        target_index=target_index,
                        reasons=reasons,
                        score=0.0,
                        action=action,
                        withheld=self._closing_axis_withheld(best_result),
                        ordering_decision=ordering_decision,
                        **self._fusion_of(target_index),
                        excluded=excluded or None,
                        excluded_by_regions=bool(excluded) and self._excluded_by_regions_of(best_result),
                    )
                )
                # Where the part it went for stood, for next_target and for the report, on a failure as on a grasp.
                target_centre = self._remember_failed_part(frame, target_index, fused_target)
                if action == "rescan":
                    continue
                # No further recovery available.
                return PickReport(
                    outcome=PickOutcome.RESCANNED_EXHAUSTED,
                    attempts=tuple(attempts),
                    target_centre_mm=target_centre,
                )

            if not looking and self.both_faces and self._hand_has_jaw_faces():
                # Both jaw contact faces asked for on a fixed camera, which cannot look around: the grasp its cameras
                # ranked is gripped only where they showed both, and another frame from where they stand shows no more.
                withheld = self._faces_the_fixed_cameras_did_not_show(frame, best_result, target_index)
                if withheld:
                    return self._faces_unseen(
                        attempt_index, attempts, withheld=withheld, result=best_result, target_index=target_index,
                        ordering=ordering_decision)

            emit(
                self.on_progress, PickStage.RANKED,
                attempt=attempt_index, target_index=target_index,
                candidate_count=len(best_result.candidates),
                score=float(best_result.top_score),
                **({"extra": looks_extra} if looks_extra else {}),
            )
            # Success path: execute approach, grasp, retreat via the policy.
            # Annotate the chosen result's candidates with shadow
            # success probabilities before execution so the telemetry
            # is available regardless of whether the motion succeeds.
            # ``annotate_grasp_points_with_shadow_probability`` is a
            # no-op when ``shadow_success_context is None``, preserving
            # the legacy path byte-for-byte for callers that did not
            # opt in.
            shadow_telemetry = annotate_grasp_points_with_shadow_probability(
                best_result.candidates,
                ctx=self.shadow_success_context,
            )
            # ---- capture pre-blend snapshot ----
            # The ranker needs per-candidate features in their pre-blend
            # state (the annotator has run but the reranker has not).
            # The snapshot taken here reconstructs candidate identity
            # post-blend. Each candidate is keyed by its pre-blend index:
            # the same ``GraspPoint`` object survives the blend, so
            # post-blend ids map back via ``id(...)``.
            pre_blend_candidates = tuple(best_result.candidates)
            # With both jaw contact faces asked for (``both_faces``), the grasp gripped is the one whose faces were
            # judged: the calculator's best at ranking time. Neither rerank below may swap another candidate in
            # ahead of it, so both are handed no config and stand down (their telemetry stays ``None``), and
            # ``_execute`` falls back to no other candidate either.
            judged_only = self._grips_only_the_judged_grasp()
            if judged_only and (self.ranking_blend_config is not None or self.uncertainty_rerank_config is not None):
                _LOG.info("both_faces: the reranks stand down, so the grasp gripped is the one whose contact faces "
                          "were judged")
            # Bounded probability-blend reranker. No-op when:
            #   * ``ranking_blend_config`` is None / disabled (silent;
            #     returns ``blend_telemetry=None``, legacy parity).
            #   * The orchestrator has no shadow context.
            #   * The current grasp-mode is not in the dense-only
            #     allow-list (schema-locked).
            #   * The lifecycle phase is ``"shadow"``.
            #   * Any candidate is missing a shadow probability (the
            #     annotator must have populated each candidate first).
            # This never mutates ``GraspPoint.score``: only metadata and
            # the order of the returned tuple change. When the rerank
            # fires the top-1 candidate may change, and
            # ``shadow_telemetry`` is then recomputed from the new winner
            # so :attr:`PickReport.shadow_success_telemetry` always
            # describes the candidate that will be executed.
            blend_candidates, blend_telemetry = maybe_blend_rerank_candidates(
                best_result.candidates,
                ctx=self.shadow_success_context,
                config=None if judged_only else self.ranking_blend_config,
            )
            # Apply the blend result (swap candidates + re-derive shadow telemetry on a top-1
            # change) via the shared helper.
            best_result, shadow_telemetry = self._apply_rerank_result(
                best_result, shadow_telemetry, blend_candidates, blend_telemetry
            )
            # ---- per-candidate uncertainty re-rank ----
            # Runs after the blend (re-sorts the blended order), before the ranking shadow (so the
            # shadow observes the final executed order) and before execution (so the approach
            # validation walks the order that executes). Subtractive:
            # ``adjusted = score - weight*corridor_risk``.
            # Silent no-op unless wired + dense + non-shadow + weight>0 + every candidate carries a
            # corridor-risk signal. On a top-1 change ``shadow_telemetry`` is re-derived from the new
            # winner, as the blend block above.
            rerank_candidates, uncertainty_rerank_telemetry = (
                maybe_uncertainty_rerank_candidates(
                    best_result.candidates,
                    ctx=self.shadow_success_context,
                    config=None if judged_only else self.uncertainty_rerank_config,
                )
            )
            # Apply the uncertainty rerank result via the same shared helper.
            best_result, shadow_telemetry = self._apply_rerank_result(
                best_result,
                shadow_telemetry,
                rerank_candidates,
                uncertainty_rerank_telemetry,
            )
            # ---- ranking-shadow producer ----
            # Silent no-op unless ``shadow_router`` and its
            # ``ranking_policy`` are both wired. Telemetry stashed on
            # ``self._ranking_telemetry`` for the caller
            # (``AutonomousGraspService``) to fold into the
            # post-pick shadow-router telemetry block. Never mutates
            # ``best_result``, ``shadow_telemetry``, or scores.
            self._maybe_run_ranking_shadow(
                pre_blend_candidates=pre_blend_candidates,
                post_blend_candidates=best_result.candidates,
                attempt_index=attempt_index,
            )
            # The last event before the arm moves. Everything after this is motion, so a console that
            # stops updating here is telling an operator the truth: the stack is done deciding.
            emit(
                self.on_progress, PickStage.EXECUTING,
                attempt=attempt_index, target_index=target_index,
                score=float(best_result.top_score),
                position_mm=_grasp_position_mm(best_result),
            )
            # Read before the arm moves, off the frame the grasp was synthesised in.
            target_centre = self._target_centre_base_mm(frame, target_index, fused_cloud_base_mm=fused_target)
            fused_for_target = self._fusion_of(target_index)
            # What the tries read beside the result: the looks to come back to, and the box the world keeps out for the
            # part. Handed through the loop, so ``_execute`` keeps the one-argument seam a program or a test may stand
            # in for; what the tries came to is read back off ``_last_tried``.
            self._tries_looked = looked
            self._tries_box = self._held_target_box(frame, target_index, fused_target)
            self._last_tried = None
            try:
                policy_report = self._execute(best_result)
            finally:
                self._tries_looked, self._tries_box = None, None
            tried = (self._last_tried if self._last_tried is not None and self._last_tried.report is policy_report
                     else _Tried(report=policy_report))
            self._last_policy_report = policy_report
            if tried.cancelled:
                # A stop asked for between two grasps of the look: nothing more moves, as between two attempts.
                _LOG.info("a stop was asked for between two grasps of the look; nothing more is handed over")
                emit(self.on_progress, PickStage.CANCELLED, attempt=attempt_index)
                attempts.append(PickAttempt(
                    attempt_index=attempt_index, target_index=target_index, reasons=(), score=best_result.top_score,
                    action="cancelled", ordering_decision=ordering_decision, **_motion_fields(policy_report),
                    **fused_for_target, tries=tried.tries,
                ))
                return PickReport(outcome=PickOutcome.CANCELLED, attempts=tuple(attempts), target_centre_mm=target_centre)
            # Map the PolicyOutcome to its action label and PickOutcome via _map_policy_outcome.
            attempt_action, pick_outcome = self._map_policy_outcome(
                policy_report.outcome
            )
            # Why the motion failed can be the cell, not the grasp. Asked only here, on a path that
            # has already failed, so a healthy pick pays nothing. When the controller itself cannot
            # move, say so instead of reporting a failed execution: the reason is typed, the recovery
            # dispatcher maps it to no action, and the outcome stays out of the failure taxonomy.
            controller_reason = tried.controller or (
                self._controller_cannot_move()
                if pick_outcome is not PickOutcome.EXECUTED
                else None
            )
            if tried.moved_back is not None or tried.not_back:
                # The arm was to go back to the look before the next grasp, and did not: it may have moved part of the
                # way, or it stands at the last standoff. Nothing more is commanded; the attempt ends on that motion.
                _LOG.error("the move back to the look before the next grasp %s; nothing more is commanded",
                           f"failed ({tried.moved_back.status.value}: {tried.moved_back.message})"
                           if tried.moved_back is not None else f"was refused before anything was sent ({tried.not_back})")
                attempt_action, pick_outcome = "execution_failed", PickOutcome.EXECUTION_FAILED
            if controller_reason is not None:
                _LOG.error("pick aborted: %s", controller_reason)
                attempt_action = "controller_not_operational"
                pick_outcome = PickOutcome.CONTROLLER_NOT_OPERATIONAL
            if (pick_outcome is PickOutcome.APPROACH_PATH_BLOCKED and looked is not None
                    and self.push_gate is not None and self.push_gate.on_approach_blocked):
                # Every approach sweep was blocked and nothing moved: where approach validation runs, that is a trigger
                # of the push as well (owner, 2026-09-29), under the same neighbour evidence and every other guard.
                pushed = self._push_the_failed_part(
                    attempt_index, attempts, looked, frame, target_index, trigger="approach_path_blocked",
                    reasons=tuple(best_result.reasons), fused_target=fused_target)
                if isinstance(pushed, PickReport):
                    return pushed
                if pushed:
                    continue
            if pick_outcome is not PickOutcome.EXECUTED:
                self._remember_failed_part(frame, target_index, fused_target, centre=target_centre)
            # The motion the attempt ended on: the last try's, or the move back to the look that failed once sent.
            motion = (_look_motion_fields(tried.moved_back) if tried.moved_back is not None
                      else _motion_fields(policy_report))
            emit(
                self.on_progress, PickStage.ATTEMPT_FINISHED,
                attempt=attempt_index, action=attempt_action, outcome=str(pick_outcome),
                # Why the motion did not happen, on the wire.
                #
                # The progress event is the only thing a console sees while a run is happening, so
                # it carries the same three motion fields ``PickAttempt`` carries. Without them a
                # refusal as precise as ``workspace_rejected`` on an approach pose 33.6 mm below the
                # configured workspace floor reaches the operator as a bare ``execution_failed``,
                # and a fail-closed guard that cannot explain itself to the person at the cell is a
                # guard that gets switched off.
                #
                # Additive and empty on the healthy path: `_motion_fields` yields three ``None``s
                # for a report that carries none, and the keys are dropped below, so every existing
                # consumer sees the event it saw before.
                **{k: v for k, v in motion.items() if v is not None},
            )
            attempts.append(
                PickAttempt(
                    attempt_index=attempt_index,
                    target_index=target_index,
                    reasons=(
                        (GraspFailureReason.CONTROLLER_NOT_OPERATIONAL,)
                        if controller_reason is not None
                        # Every grasp tried was refused before anything was sent, none at the part: a typed refusal the
                        # recovery loop answers by looking again, never by a motion (RC4, the owner's P1-P4).
                        else (GraspFailureReason.MOTION_PLAN_REFUSED,)
                        if tried.refused and pick_outcome is not PickOutcome.EXECUTED
                        else best_result.reasons
                    ),
                    score=best_result.top_score,
                    action=attempt_action,
                    ordering_decision=ordering_decision,
                    **motion,
                    **fused_for_target,
                    tries=tried.tries,
                )
            )
            return PickReport(
                outcome=pick_outcome,
                attempts=tuple(attempts),
                executed_grasp=(
                    best_result if pick_outcome is PickOutcome.EXECUTED else None
                ),
                # The selected grasp's telemetry, not the executed one's. See the field's comment.
                selected_telemetry=dict(getattr(best_result, "telemetry", None) or {}),
                shadow_success_telemetry=shadow_telemetry,
                ranking_blend_telemetry=blend_telemetry,
                uncertainty_rerank_telemetry=uncertainty_rerank_telemetry,
                target_centre_mm=target_centre,
            )

        return PickReport(
            outcome=PickOutcome.RESCANNED_EXHAUSTED,
            attempts=tuple(attempts),
            target_centre_mm=self.failed_part.centre_mm if self.failed_part is not None else None,
        )

    # ------------------------------------------------------------------
    # The push of a failed part (nudge_target), and the part next_target skips
    # ------------------------------------------------------------------

    @property
    def pushes(self) -> "tuple[PickPush, ...]":
        """What every push the last run considered came to, in order: refused before anything moved, pushed, or
        stopped where the arm is. Empty where no push was considered."""
        return tuple(self._pushes)

    def _push_the_failed_part(
        self,
        attempt_index: int,
        attempts: list[PickAttempt],
        looked: LookedAround,
        frame: PerceptionFrame,
        target_index: int | None,
        *,
        trigger: str,
        reasons: tuple[GraspFailureReason, ...],
        fused_target: "np.ndarray | None",
    ) -> "PickReport | bool":
        """Push the part the looks judged, where :attr:`push_gate` lets this pick, and look again: the owner's
        ``nudge_target`` of 2026-09-29, run inside the pick attempt.

        Returns ``False`` where the push touched nothing and the attempt goes on as it would have, to its next action:
        no gate, no neighbour within 25 mm, the planner or the budgets refused, a stop asked for while the push was
        planned, a guard of the push refused before its first motion, or the down leg was refused before it was sent
        and the arm went back to the look. ``True`` where the part was pushed, the arm went back to the look it stood at
        and looked again: the next attempt goes on with that look. A :class:`PickReport` where the pick ends: the push
        found the controller unable to move before anything moved (``CONTROLLER_NOT_OPERATIONAL``, nothing commanded,
        as a stopped controller ends a pick anywhere), or nobody able to say where a toggle hand's jaws stand, or a
        gripper that measures its width not connected or whose width read failed (``GRIPPER_FAULT``, nothing commanded,
        as the toggle's own refusal ends a pick; a count or a width that says closed is a refusal the attempt goes on
        from), or it stopped once something may have moved (a toggle's count nobody can vouch for before a contact leg,
        or once the up leg ended, among it), or the move back to the look did not run, and then the arm stays where
        it is, nothing more is commanded and a person decides (``CONTROLLER_NOT_OPERATIONAL`` where the controller
        cannot move, ``ABORTED`` otherwise, which the service reports as ``unsafe_recovery_refused``).

        Every check before the first motion reads and never asks anybody (:meth:`_considered_push`, then
        :func:`~src.robot.grasping.recovery.push_motion.execute_push`'s own): the jaw count read off the hand and never
        switched, the controller, the live camera world, the plan's shape, the budgets (1 per part, 2 per pick, 5 per
        campaign), an attempt left to pick with, the joints to come back to, and a stop asked for. The push closes the
        way round nearer :attr:`natural_closing_axis` where the cell names one, as every grasp this loop ranks does,
        else nearer where the tool stands. The part and its
        swept path are kept out of the world for the push, and the pick's target is offered again once it ends, a
        refusal included, since the push's keep-out forgets every offer as it closes. Every push that commanded a
        motion counts against the budgets, P0 in the air above a part nothing touched included (``PushBudgets.record``).
        The move back runs like a grasp approach, except to the one view this pick generated, a configuration the pick
        made up, which it runs on the straight joint line alone (:meth:`_back_to_the_look`). After a push, what the
        pick's earlier looks saw of the part is stale, and it leaves the views the next look is fused with. A camera
        world that cannot vouch for the cell on the way raises, as on every motion of a pick, once the push is kept as
        one that stopped where the arm stands (:meth:`_push_stopped_by_the_camera_world`).
        """
        from src.robot.core.errors import CameraWorldUnavailable  # noqa: PLC0415
        from src.robot.core.gripper import toggle_without_sensor_of, width_is_measured_of  # noqa: PLC0415
        from src.robot.execution.looks import look_label  # noqa: PLC0415
        from src.robot.grasping.recovery.push_gate import REFUSED_STOP_REQUESTED  # noqa: PLC0415
        from src.robot.grasping.recovery.push_motion import PushOutcomeCode, execute_push  # noqa: PLC0415
        from src.robot.grasping.recovery.push_planner import swept_target_points_mm  # noqa: PLC0415

        gate = self.push_gate
        judged = looked.judged
        if gate is None or judged is None or target_index is None:
            return False
        considered = self._considered_push(gate, judged, target_index, attempt_index, trigger=trigger)
        if not isinstance(considered, tuple):
            self._pushes.append(considered)
            _LOG.info("no push of the part (%s): %s", considered.code, considered.sentence)
            return False
        plan, points, back_to = considered
        transform = self._pending_camera_to_base
        offer = segmentation_offer_from_frame(
            frame, target_index, camera=self._perception_camera_name(),
            camera_to_base_mm=None if transform is None else np.asarray(transform.to_matrix(), dtype=np.float64),
            target_points_base_mm=points,
        )
        if offer is None:
            refused = self._push_record(trigger, "refused_no_offer", "The frame the part was judged on carries no "
                                        "shutter time to offer the planner world, so no push is made.", plan=plan)
            self._pushes.append(refused)
            return False
        if self.should_cancel is not None and self.should_cancel():
            # A stop asked for while the push was ranked and planned, after the looks asked the last time: nothing is
            # commanded, and the next attempt's start ends the pick, as a stop between two attempts ends it.
            self._pushes.append(self._push_record(trigger, REFUSED_STOP_REQUESTED, (
                "A stop was asked for before the push began, so nothing was pushed and nothing moved."), plan=plan))
            _LOG.info("no push of the part: a stop was asked for before it began")
            return False
        (x0, y0), (x1, y1) = plan.landing_box_xy_mm
        _LOG.info("pushing the part %.0f mm along (%.2f, %.2f) to open room beside it (%s): the finger at %.1f mm over "
                  "the support, the landing box x %.0f to %.0f mm, y %.0f to %.0f mm", plan.push_distance_mm,
                  plan.direction[0], plan.direction[1], trigger, plan.finger_height_mm, x0, x1, y0, y1)
        natural = closing_axis_of(self.natural_closing_axis) if self.natural_closing_axis is not None else None
        try:
            outcome = execute_push(self.arm, plan, gripper=self.gripper, offer=offer, natural_closing_axis=natural,
                                   should_cancel=self.should_cancel)
        except CameraWorldUnavailable as exc:
            self._push_stopped_by_the_camera_world(gate, trigger, plan, exc)
            raise
        record = self._push_record(trigger, outcome.code.value, outcome.reason, plan=plan, outcome=outcome)
        _LOG.info("the push came to %s, leg by leg: %s", outcome.code.value, "; ".join(record.leg_verdicts) or "none ran")
        if record.arm_moved:
            # Counted as soon as the push commanded any motion, whether or not it finished (PushBudgets.record), P0 in
            # the air above a part nothing touched included: by the same centre the check was given, and by the landing
            # it predicted where the part may have moved (push_budget_key); a part nothing touched stands where it stood.
            assert outcome.budget_centre_mm is not None  # a push that moved the arm has its plan
            gate.budgets.record(centre_mm=outcome.budget_centre_mm,
                                landing_mm=outcome.budget_landing_mm if outcome.motion_started else None)
        if outcome.motion_started:
            swept = np.vstack([points, swept_target_points_mm(points, plan.direction, plan.push_distance_mm)])
            landing = outcome.budget_landing_mm or plan.predicted_landing_centre_mm
            self._pushed_parts.append((swept, (float(landing[0]), float(landing[1]))))
            # The push's keep-out forgot every offer as it closed (the live world's rule): the part and the path it was
            # pushed along go back out of the world before the next motion, the move back to the look, whose first
            # stretch starts right above the part.
            self._keep_out.close()
            self._keep_out.enter_context(keeping_out(self.arm, replace(offer, target_points_base_mm=swept)))
        else:
            # Refused before anything moved, the keep-out possibly entered and closed: the pick's own target goes back.
            self._offer_masks_to_planner_world(frame, target_index, target_points_base_mm=fused_target)
        if outcome.code is PushOutcomeCode.REFUSED_CONTROLLER_STOPPED:
            # The controller cannot move, or cannot be read, and nothing moved: the pick ends here, as a look the
            # controller stopped ends it, with nothing else commanded. Falling through would send the arm to its looks
            # again (next_target), on a controller that just said it cannot move.
            return self._push_found_the_controller_stopped(attempt_index, attempts, record, target_index,
                                                           frame=frame, fused_target=fused_target)
        if outcome.code is PushOutcomeCode.REFUSED_JAWS_UNKNOWN and (
                toggle_without_sensor_of(self.gripper) is not None or width_is_measured_of(self.gripper)):
            # Nobody can say where a toggle's jaws stand (DO0 switched at the pendant mid-pick, the count lost or
            # unreadable), or a gripper that measures its width is not connected or its width read failed (the Robotiq
            # socket driver counts itself connected until the program disconnects it, so a socket that stopped
            # answering shows as a read that fails), and nothing moved: the pick ends here as a gripper fault, as the
            # toggle's own refusal ends one (the owner, 2026-09-30). Falling through would drive the looks again with a
            # hand nobody vouches for.
            return self._push_found_the_jaws_unknown(attempt_index, attempts, record, target_index,
                                                     frame=frame, fused_target=fused_target)
        if outcome.refused_before_motion and not outcome.legs_done:
            self._pushes.append(record)
            _LOG.info("the push was refused before anything moved (%s): %s", outcome.code.value, outcome.reason)
            return False
        if outcome.stopped:
            controller = outcome.code is PushOutcomeCode.CONTROLLER_NOT_OPERATIONAL
            stopped = replace(record, stopped=True, controller_stopped=controller)
            return self._push_stopped(attempt_index, attempts, stopped, target_index, outcome=outcome,
                                      frame=frame, fused_target=fused_target)
        # The arm left the look: pushed, or at P0 in the air where the down leg was refused before it was sent. Back to
        # the look it stood at, unless a stop was asked for since the push's last leg (Track P's item 5): then the arm
        # stays where it stands and a person decides.
        if self.should_cancel is not None and self.should_cancel():
            first = ("The part was pushed" if outcome.motion_started
                     else "The arm went to P0 above the part and nothing touched it")
            stopped = replace(record, code=PushOutcomeCode.UNSAFE_RECOVERY_REFUSED.value, stopped=True,
                              controller_stopped=False, leg="back_to_the_look", sentence=(
                                  f"{first}, and a stop was asked for before the move back to the look: the arm stays "
                                  "where it stands, nothing more is commanded, and a person decides what happens next."))
            return self._push_stopped(attempt_index, attempts, stopped, target_index, outcome=outcome,
                                      frame=frame, fused_target=fused_target)
        try:
            back, generated = self._back_to_the_look(looked, back_to)
        except CameraWorldUnavailable as exc:
            self._push_stopped_by_the_camera_world(gate, trigger, plan, exc, outcome=outcome)
            raise
        if not back.done:
            controller_reason = back.controller
            where = (f"the straight joint line back to the view the pick generated, {look_label(back_to)}," if generated
                     else f"the move back to the look {look_label(back_to)}")
            what = (f"{back.stopped.status.value}: {back.stopped.message}" if back.stopped is not None
                    else f"refused before anything was sent: {back.refused}")
            made_up = ("; the pick made that view up, so nothing plans around its line (the owner, 2026-09-29)"
                       if generated else "")
            first = ("The part was pushed" if outcome.motion_started
                     else "The arm went to P0 above the part and nothing touched it")
            why = (f"{first}, and {where} did not run ({what}){made_up}: the arm stays where it stopped, nothing more "
                   "is commanded, and a person decides what happens next.")
            stopped = replace(record, code=(PushOutcomeCode.CONTROLLER_NOT_OPERATIONAL.value if controller_reason
                                            else PushOutcomeCode.UNSAFE_RECOVERY_REFUSED.value),
                              sentence=why + (f" {controller_reason}" if controller_reason else ""),
                              stopped=True, controller_stopped=controller_reason is not None, leg="back_to_the_look")
            return self._push_stopped(attempt_index, attempts, stopped, target_index, outcome=outcome,
                                      frame=frame, fused_target=fused_target, moved=back.stopped,
                                      refused=back.refused)
        if outcome.refused_before_motion:
            # The down leg was refused before anything was sent: the arm is back at the look and the part untouched, so
            # the looks still stand and the attempt goes on to its next action, as on any refusal before motion.
            self._pushes.append(replace(record, sentence=f"{outcome.reason} The arm went back to the look."))
            _LOG.info("the push was refused before it touched the part (%s); the arm went back to the look",
                      outcome.code.value)
            return False
        label = f"{look_label(back_to)} after the push"
        relooked = self._look_again_after_the_push(looked, judged, label, back_to, attempt_index=attempt_index)
        pushed = replace(record, looked_again=label)
        self._pushes.append(pushed)
        fields = _push_motion_fields(outcome)
        emit(
            self.on_progress, PickStage.ATTEMPT_FINISHED,
            attempt=attempt_index, action="push", outcome=outcome.code.value, target_index=target_index,
            reasons=tuple(str(reason) for reason in reasons),
            extra={"push": outcome.code.value, "push_mm": round(float(plan.push_distance_mm), 1),
                   "push_reason": outcome.reason, "looked_again": label},
            **{k: v for k, v in fields.items() if v is not None},
        )
        attempts.append(PickAttempt(
            attempt_index=attempt_index, target_index=target_index, reasons=reasons, score=0.0, action="push",
            ordering_decision=None, **fields, **self._fusion_of(target_index), push=outcome.code.value,
        ))
        self._pending_looked = relooked
        return True

    def _considered_push(
        self, gate: "PushGate", judged: _Judgement, target_index: int, attempt_index: int, *, trigger: str,
    ) -> "tuple[PushPlan, np.ndarray, JointPositions] | PickPush":
        """The plan, the part's points and the joints to come back to, or why no push is made; nothing moves here.

        In order, the cheapest first: an attempt left to pick with after the push; the part's points (its cloud fused
        over the looks, else what the judged look saw of it) and the support its grasp was computed against; a
        neighbour within 25 mm of it in what the looks saw (the trigger's evidence); the plan
        (:func:`~src.robot.grasping.recovery.push_planner.plan_push` with the table every look saw, the cell's hand,
        workspace and clearance, the campaign's distance, the fixture box); the budgets, on the plan's own centre; and
        where the arm stands, to come back to.
        """
        from src.robot.grasping.recovery.push_gate import support_points_of_views  # noqa: PLC0415
        from src.robot.grasping.recovery.push_motion import push_budget_key  # noqa: PLC0415
        from src.robot.grasping.recovery.push_planner import (  # noqa: PLC0415
            REFUSED_NO_BLOCKING_NEIGHBOUR,
            PushRefusal,
            neighbour_evidence,
            plan_push,
        )

        if attempt_index + 1 >= self.max_attempts:
            return self._push_record(trigger, "refused_no_attempt_left", (
                f"This pick has no attempt left after attempt {attempt_index + 1} of {self.max_attempts} to pick the "
                "part from after a push, so it is not pushed."))
        points = self._pushed_part_points(judged, target_index)
        if points is None:
            return self._push_record(trigger, "refused_no_target_points",
                                     "The looks placed no point of the part in BASE, so it is not pushed.")
        if self.support_config is None or judged.support_height_mm is None:
            return self._push_record(trigger, "refused_no_support", (
                "No support plane was resolved for the part (grasping.support), so how high a finger rides beside "
                "it is not known and it is not pushed."))
        normal = np.asarray(self.support_config.normal, dtype=np.float64).reshape(3)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        offset = float(judged.support_height_mm)
        # Where the camera world found the surface the part stands on (contract 7), the push plans on that surface's
        # plane through its own reading under the part, and bounds its landing by that surface's own pixels.
        model = judged.support_model
        surface = model.surface_under(points[:, :2]) if model is not None else None
        if surface is not None and surface.offset_mm is not None:
            normal = np.asarray(surface.normal, dtype=np.float64).reshape(3)
            offset = float(surface.offset_mm)
            _LOG.info("the push plans on %s, tilted %.1f deg, through its reading under the part at %.1f mm",
                      surface.name, float(surface.tilt_deg), float(surface.at_mm[2]) if surface.at_mm else offset)
        neighbours = self._neighbour_points_of(judged, target_index)
        try:
            evidence = neighbour_evidence(target_points_mm=points, neighbour_points_mm=neighbours,
                                          support_normal=normal, support_offset_mm=offset)
            if not evidence.found:
                seen = ("no neighbour was seen" if not np.isfinite(evidence.nearest_gap_mm)
                        else f"the nearest is {evidence.nearest_gap_mm:.0f} mm away")
                return self._push_record(trigger, REFUSED_NO_BLOCKING_NEIGHBOUR, (
                    f"No neighbour stands within {evidence.radius_mm:g} mm of the part in what the looks saw "
                    f"({seen}), so nothing says a neighbour left the fingers no room, and it is not pushed."))
            fixed = self._fixed_views[0] if self._fixed_views is not None else ()
            if surface is not None and model is not None:
                table = model.table_points(self._depth_views(fixed), points[:, :2])
            else:
                views = [(seen.depth, seen.frame.intrinsics, seen.camera_to_base) for seen in self._look_views]
                views += [(view.depth_map, view.intrinsics, view.camera_to_base) for view in fixed]
                table = support_points_of_views(views, support_normal=normal, support_offset_mm=offset)
            floor = self._finger_floor_mm(model, points, normal, offset, gate)
            floor_kwargs: dict[str, Any] = {} if floor is None else {"finger_floor_mm": floor}
            if floor is not None:
                heights = points @ normal - offset
                top = float(np.percentile(heights, 95.0)) if heights.size else 0.0
                if floor >= top - _FLOOR_UNDER_THE_TOP_MM:
                    return self._push_record(trigger, "refused_finger_floor_over_the_part", (
                        f"The finger would ride {floor:.0f} mm over the support, the guard's distance and 1 mm over the "
                        f"solid the camera world holds beside the part, and the part's top stands {top:.0f} mm over "
                        "it: the finger would pass over the part, so it is not pushed."))
            plan = plan_push(
                target_points_mm=points, neighbour_points_mm=neighbours, support_normal=normal,
                support_offset_mm=offset, workspace=gate.cell.workspace, hand=gate.cell.hand, table_points_mm=table,
                push_distance_mm=gate.distance_mm, container_interior=gate.cell.container_interior,
                operator_box=gate.operator_box, hand_clearance_mm=gate.cell.hand_clearance_mm,
                beside_part_clearance_mm=gate.cell.beside_part_clearance_mm, beside_part_mm=gate.cell.beside_part_mm,
                **floor_kwargs,
            )
        except ValueError as exc:
            return self._push_record(trigger, "refused_plan_malformed",
                                     f"The push could not be planned on what the looks saw ({exc}), so the part is "
                                     "not pushed.")
        if isinstance(plan, PushRefusal):
            return self._push_record(trigger, plan.code, plan.sentence)
        centre, _landing = push_budget_key(plan)
        verdict = gate.budgets.check(centre)
        if not verdict.allowed:
            return self._push_record(trigger, verdict.code, verdict.sentence, plan=plan)
        back_to = self._joints_here()
        if back_to is None:
            return self._push_record(trigger, "refused_joints_unknown", (
                "The arm could not say where it stands, so it could not come back to the look after the push, and "
                "the part is not pushed."), plan=plan)
        return plan, points, back_to

    def _depth_views(self, fixed: Any = ()) -> list[Any]:
        """Every look of the pick, and every fixed camera fused with it, as the camera world's ``DepthView``s."""
        from src.robot.safety.planning.perceived import DepthView  # noqa: PLC0415

        views = [DepthView(surface_depth_mm=seen.depth, intrinsics=np.asarray(seen.frame.intrinsics, dtype=np.float64),
                           camera_to_base=seen.camera_to_base, name=seen.name)
                 for seen in self._look_views if seen.depth is not None and seen.camera_to_base is not None]
        views += [DepthView(surface_depth_mm=np.asarray(view.depth_map, dtype=np.float64),
                            intrinsics=np.asarray(view.intrinsics, dtype=np.float64),
                            camera_to_base=np.asarray(view.camera_to_base, dtype=np.float64), name=str(view.name))
                  for view in fixed]
        return views

    @staticmethod
    def _finger_floor_mm(model: Any, points: np.ndarray, normal: np.ndarray, offset: float,
                         gate: "PushGate") -> float | None:
        """How high over the push's support the finger rides at least, millimetres: the highest top a solid the camera
        world holds stands at under anything the hand may sweep (the part's footprint grown by the hand's whole stroke and
        the push), above the push's plane, plus the guard's distance to a camera box and 1 mm (contract 7: the guard
        never refuses the push's own path at a solid). ``None`` where no model or no solid lies there."""
        from src.robot.grasping.recovery.push_planner import CONTACT_GAP_MM  # noqa: PLC0415

        if model is None or not getattr(model, "solids", ()):
            return None
        hand = gate.cell.hand
        reach = (CONTACT_GAP_MM + 2.0 * hand.half_outer_mm + float(gate.distance_mm)
                 + 0.5 * max(hand.palm_width_mm, hand.finger_width_mm))
        foot = np.asarray(points, dtype=np.float64)[:, :2]
        low, high = foot.min(axis=0) - reach, foot.max(axis=0) + reach
        xs, ys = np.meshgrid(np.arange(low[0], high[0] + 1e-9, _FLOOR_GRID_MM), np.arange(low[1], high[1] + 1e-9,
                                                                                       _FLOOR_GRID_MM))
        grid = np.column_stack([xs.ravel(), ys.ravel()])
        top = np.full(grid.shape[0], -np.inf)
        for solid in model.solids:
            covered = solid.covers_xy(grid)
            if covered.any():
                top[covered] = np.maximum(top[covered], solid.top_at(grid[covered]))
        held = np.isfinite(top)
        if not held.any():
            return None
        unit = np.asarray(normal, dtype=np.float64) / float(np.linalg.norm(normal))
        heights = np.column_stack([grid[held], top[held]]) @ unit - float(offset)
        return float(heights.max()) + float(gate.cell.guard_distance_mm) + 1.0

    @staticmethod
    def _push_record(
        trigger: str, code: str, sentence: str, *, plan: "PushPlan | None" = None, outcome: "PushOutcome | None" = None,
    ) -> "PickPush":
        """One push the pick considered, as :attr:`pushes` keeps it."""
        from src.robot.grasping.recovery.push_gate import PickPush  # noqa: PLC0415
        from src.robot.grasping.recovery.push_motion import push_budget_key  # noqa: PLC0415

        centre, landing = push_budget_key(plan) if plan is not None else (None, None)
        return PickPush(
            trigger=trigger, code=code, sentence=sentence,
            motion_started=bool(outcome is not None and outcome.motion_started),
            arm_moved=bool(outcome is not None and (outcome.motion_started or outcome.legs_done)),
            distance_mm=None if plan is None else float(plan.push_distance_mm),
            part_centre_mm=centre, landing_mm=landing,
            direction=None if plan is None else tuple(float(v) for v in plan.direction),  # type: ignore[arg-type]
            leg=None if outcome is None else outcome.leg,
            legs_done=() if outcome is None else tuple(outcome.legs_done),
            finger_height_mm=None if plan is None else float(plan.finger_height_mm),
            landing_box_xy_mm=None if plan is None else (
                (float(plan.landing_box_xy_mm[0][0]), float(plan.landing_box_xy_mm[0][1])),
                (float(plan.landing_box_xy_mm[1][0]), float(plan.landing_box_xy_mm[1][1]))),
            leg_verdicts=() if outcome is None else _leg_verdicts(outcome),
            outcome=outcome,
        )

    def _swept_by_a_push(self, offer: "SegmentationOffer | None") -> "np.ndarray | None":
        """The points a push of this run swept the offered target through, where the target is the pushed part, and
        ``None`` otherwise.

        The pushed part is the target whose BASE centre lies within ``SAME_PART_RADIUS_MM`` of where a push predicted
        it to land and whose footprint (BASE XY, grown by ``OFFER_FOOTPRINT_MARGIN_MM``) holds that landing: the rule
        ``push_motion`` holds an offer to before it keeps one out. A part of the label merely near the landing is
        another part, and its offer taking the pushed part's path would take the pushed part itself out of the world
        while the arm picks beside it. Size tells nothing here: parts of one label are alike."""
        if offer is None or not self._pushed_parts:
            return None
        own = offer.target_points_base_mm
        if own is None or not len(own):
            return None
        from src.robot.grasping.recovery.push_budgets import SAME_PART_RADIUS_MM  # noqa: PLC0415
        from src.robot.grasping.recovery.push_motion import OFFER_FOOTPRINT_MARGIN_MM  # noqa: PLC0415

        footprint = np.asarray(own, dtype=np.float64)[:, :2]
        centre = np.median(footprint, axis=0)
        low = footprint.min(axis=0) - OFFER_FOOTPRINT_MARGIN_MM
        high = footprint.max(axis=0) + OFFER_FOOTPRINT_MARGIN_MM
        swept = [points for points, landing in self._pushed_parts
                 if float(np.hypot(centre[0] - landing[0], centre[1] - landing[1])) <= SAME_PART_RADIUS_MM
                 and bool(np.all(np.asarray(landing) >= low) and np.all(np.asarray(landing) <= high))]
        return np.vstack(swept) if swept else None

    def _push_refused_extra(self) -> dict[str, Any]:
        """The last push this run considered and refused before anything moved, as a no-candidate event carries it."""
        if not self._pushes or self._pushes[-1].motion_started:
            return {}
        last = self._pushes[-1]
        return {"push": last.code, "push_reason": last.sentence}

    def _push_stopped(
        self,
        attempt_index: int,
        attempts: list[PickAttempt],
        stopped: "PickPush",
        target_index: int | None,
        *,
        outcome: "PushOutcome",
        frame: PerceptionFrame,
        fused_target: "np.ndarray | None",
        moved: "MotionReport | None" = None,
        refused: str = "",
    ) -> PickReport:
        """The attempt of a push that stopped where the arm stands: nothing more is commanded, a person decides.

        ``CONTROLLER_NOT_OPERATIONAL`` where the controller said it cannot move, ``ABORTED`` otherwise. The attempt
        carries no failure reason a recovery could act on, so the recovery loop stops on it, and so does the campaign.
        ``moved`` is the move back to the look that failed, ``refused`` why its straight joint line was refused before
        anything was sent; neither where the push itself stopped.
        """
        self._pushes.append(stopped)
        _LOG.error("the push stopped where the arm stands (%s): %s", stopped.code, stopped.sentence)
        fields: _MotionFields
        if moved is not None:
            fields = _look_motion_fields(moved)
        elif refused:
            fields = {"motion_status": None, "motion_message": refused, "motion_error": None, "camera_world": None,
                      "camera_world_reason": None}
        else:
            fields = _push_motion_fields(outcome)
        pick_outcome = (PickOutcome.CONTROLLER_NOT_OPERATIONAL if stopped.controller_stopped
                        else PickOutcome.ABORTED)
        emit(
            self.on_progress, PickStage.ATTEMPT_FINISHED,
            attempt=attempt_index, action="push", outcome=str(pick_outcome), target_index=target_index,
            extra={"push": stopped.code, "push_reason": stopped.sentence, "push_leg": stopped.leg},
            **{k: v for k, v in fields.items() if v is not None},
        )
        attempts.append(PickAttempt(
            attempt_index=attempt_index, target_index=target_index,
            reasons=(GraspFailureReason.CONTROLLER_NOT_OPERATIONAL,) if stopped.controller_stopped else (),
            score=0.0, action="push", ordering_decision=None, **fields, **self._fusion_of(target_index),
            push=stopped.code,
        ))
        target_centre = self._remember_failed_part(frame, target_index, fused_target)
        return PickReport(outcome=pick_outcome, attempts=tuple(attempts), target_centre_mm=target_centre)

    def _push_found_the_controller_stopped(
        self,
        attempt_index: int,
        attempts: list[PickAttempt],
        refused: "PickPush",
        target_index: int | None,
        *,
        frame: PerceptionFrame,
        fused_target: "np.ndarray | None",
    ) -> PickReport:
        """The attempt of a push refused because the controller cannot move, or cannot be read, before anything moved.

        Nothing was commanded and nothing more is: ``CONTROLLER_NOT_OPERATIONAL``, the attempt's reason the same, which
        no recovery action answers, as a look the controller stopped ends a pick and as the service's own check refuses
        one on a controller it cannot read (``Robot.pick``'s rule). The service reports it ``controller_stopped``, and
        the campaign stops on it; a person clears the stop where the arm is visible."""
        self._pushes.append(refused)
        _LOG.error("pick aborted before the push: %s", refused.sentence)
        outcome = PickOutcome.CONTROLLER_NOT_OPERATIONAL
        emit(
            self.on_progress, PickStage.ATTEMPT_FINISHED,
            attempt=attempt_index, action="push", outcome=str(outcome), target_index=target_index,
            extra={"push": refused.code, "push_reason": refused.sentence},
        )
        attempts.append(PickAttempt(
            attempt_index=attempt_index, target_index=target_index,
            reasons=(GraspFailureReason.CONTROLLER_NOT_OPERATIONAL,), score=0.0, action="push", ordering_decision=None,
            **self._fusion_of(target_index), push=refused.code,
        ))
        target_centre = self._remember_failed_part(frame, target_index, fused_target)
        return PickReport(outcome=outcome, attempts=tuple(attempts), target_centre_mm=target_centre)

    def _push_found_the_jaws_unknown(
        self,
        attempt_index: int,
        attempts: list[PickAttempt],
        refused: "PickPush",
        target_index: int | None,
        *,
        frame: PerceptionFrame,
        fused_target: "np.ndarray | None",
    ) -> PickReport:
        """The attempt of a push refused because nobody can say where a toggle hand's jaws stand
        (``refused_jaws_unknown``: DO0 switched at the pendant since the pick began, the count lost at a failed change
        or a disconnect, or unreadable), or because a gripper that measures its width is not connected or its width
        read failed, before anything moved.

        Nothing was commanded and nothing more is: ``GRIPPER_FAULT``, as the toggle's own refusal ends a pick
        (``PolicyOutcome.GRIPPER_FAULT``; the owner, 2026-09-30). The attempt carries no reason a recovery could act on,
        so no next_target drives the looks again with a hand nobody vouches for, and its ``motion_message`` says why,
        which the service's report reads as its ``gripper_fault``: ``PickRun`` and the console run stop on it. Nobody is
        asked here; a person looks at the jaws, and the next pick's start, or the next connect, asks where they stand.
        A count or a width that says closed is no fault: that push is refused and the pick falls through."""
        self._pushes.append(refused)
        _LOG.error("pick aborted before the push: %s", refused.sentence)
        outcome = PickOutcome.GRIPPER_FAULT
        said = f"the pick ends before the push, with nothing commanded: {refused.sentence}"
        emit(
            self.on_progress, PickStage.ATTEMPT_FINISHED,
            attempt=attempt_index, action="push", outcome=str(outcome), target_index=target_index,
            extra={"push": refused.code, "push_reason": refused.sentence},
        )
        attempts.append(PickAttempt(
            attempt_index=attempt_index, target_index=target_index, reasons=(), score=0.0, action="push",
            ordering_decision=None, motion_message=said, **self._fusion_of(target_index), push=refused.code,
        ))
        target_centre = self._remember_failed_part(frame, target_index, fused_target)
        return PickReport(outcome=outcome, attempts=tuple(attempts), target_centre_mm=target_centre)

    def _push_stopped_by_the_camera_world(
        self, gate: "PushGate", trigger: str, plan: "PushPlan", exc: BaseException, *,
        outcome: "PushOutcome | None" = None,
    ) -> None:
        """Keep a push the camera world stopped (``CameraWorldUnavailable``) as one that stopped where the arm stands,
        before the raise goes on, as on every motion of a pick, and ends the pick as a fault of the cell.

        Raised by the push itself (``outcome`` ``None``), the arm stands anywhere from the look to the lift point and
        the part may have moved, so the push counts against the budgets; raised on the move back to the look after it
        (``outcome``, what the push came to), it was counted already. The service puts it on the fault's report
        (``push_stopped``), which then needs a person, so no further pick moves the arm from where it stopped."""
        from src.robot.grasping.recovery.push_motion import PushOutcomeCode, push_budget_key  # noqa: PLC0415

        if outcome is None:
            centre, landing = push_budget_key(plan)
            gate.budgets.record(centre_mm=centre, landing_mm=landing)
            first = "During the push"
        elif outcome.motion_started:
            first = "The part was pushed, and on the move back to the look"
        else:
            first = "The arm went to P0 above the part and nothing touched it, and on the move back to the look"
        sentence = (f"{first} the camera world could not vouch for the cell ({type(exc).__name__}: {exc}): the arm "
                    "stays where it stopped, nothing more is commanded, and a person decides what happens next.")
        record = self._push_record(trigger, PushOutcomeCode.UNSAFE_RECOVERY_REFUSED.value, sentence, plan=plan,
                                   outcome=outcome)
        self._pushes.append(replace(record, motion_started=outcome is None or record.motion_started, arm_moved=True,
                                    stopped=True, leg=record.leg if outcome is None else "back_to_the_look"))
        _LOG.error("the push stopped where the arm stands (%s): %s", record.code, sentence)

    def _back_to_the_look(
        self, looked: LookedAround, back_to: "JointPositions",
    ) -> "tuple[MovedBack, bool]":
        """Send the arm back to the look it stood at before a push, and whether that look is the view the pick made up.

        A declared look, or where a pick handed none stood, is moved to like a grasp approach (``move_to_look``: its own
        judged joint move, the straight joint line first and the planner only around it, the camera world refreshed).
        The one view this pick generated is a configuration the pick made up: the arm goes back to it on the straight
        joint line alone (``generated_view.move_back_to_look``, the travel cap and the world holding every frame of the
        pick), with no cuRobo at any angle and no detour (the owner, 2026-09-29), after a push as before it. What came
        of it is a ``MovedBack``: done, refused before anything was sent, or stopped once sent, and what the controller
        said. Raises ``CameraWorldUnavailable``, as every motion of a pick does."""
        from src.robot.core.errors import CameraWorldUnavailable  # noqa: PLC0415
        from src.robot.execution.generated_view import MovedBack, move_back_to_look  # noqa: PLC0415
        from src.robot.execution.looks import move_to_look  # noqa: PLC0415
        from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

        if self._stands_at_the_generated_view(looked, back_to):
            back = move_back_to_look(self.arm, back_to, look=looked.generated, here=self._joints_here(),
                                     holding=self._holding, controller_stopped=self._controller_cannot_move)
            return back, True
        moved = move_to_look(self.arm, back_to)
        if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE and isinstance(
                moved.result.exception, CameraWorldUnavailable):
            raise moved.result.exception
        if moved.ok:
            return MovedBack(done=True), False
        return MovedBack(stopped=moved, controller=self._controller_cannot_move()), False

    @staticmethod
    def _stands_at_the_generated_view(looked: LookedAround, here: "JointPositions") -> bool:
        """Whether ``here`` is the one view the pick generated (``looked.generated``): the configuration it was driven
        to, within :data:`_SAME_JOINTS_DEG` on every joint. Where that configuration cannot be read, it is taken to be:
        the straight joint line is the stricter motion."""
        from src.robot.core import JointPositions  # noqa: PLC0415

        if not looked.generated:
            return False
        view = next((seen for seen in looked.views if seen.label == looked.generated), None)
        joints = getattr(view, "pose", None)
        if not isinstance(joints, JointPositions) or not isinstance(here, JointPositions) or joints.dof != here.dof:
            return True
        gap = np.abs(np.asarray(joints.degrees(), dtype=np.float64) - np.asarray(here.degrees(), dtype=np.float64))
        return bool(np.all(gap <= _SAME_JOINTS_DEG))

    def _pushed_part_points(self, judged: _Judgement, target_index: int) -> "np.ndarray | None":
        """The part's points in BASE for a push: its cloud fused over the looks, else what the judged look saw of it."""
        cloud = judged.target_cloud_base_mm
        if cloud is None or not np.asarray(cloud).size:
            clouds = judged.look.clouds
            cloud = clouds[target_index] if 0 <= target_index < len(clouds) else None
        if cloud is None or not np.asarray(cloud).size:
            return None
        return np.asarray(cloud, dtype=np.float64).reshape(-1, 3)

    @staticmethod
    def _neighbour_points_of(judged: _Judgement, target_index: int) -> np.ndarray:
        """Every other object the looks saw, in BASE: the judged look's own other objects, what every other view saw
        that is not the part (the fused neighbour cloud the candidate filter reads), and what the calculator saw beside
        the part that no prompt named (the fix plan's contract 2): Track A's obstacle points, and every point the camera
        world's rule keeps there with no voxel threshold, so a neighbour the voxel rule dropped still refuses the push's
        path. On the owner's cell every prompt grounded one box, and without those the push knew no neighbour at all."""
        parts = [cloud for index, cloud in enumerate(judged.look.clouds)
                 if index != target_index and np.asarray(cloud).size]
        fused = judged.fused_scene.neighbour_for(target_index) if judged.fused_scene is not None else None
        if fused is not None and np.asarray(fused).size:
            parts.append(np.asarray(fused, dtype=np.float64).reshape(-1, 3))
        metadata = getattr(getattr(judged, "result", None), "metadata", None)
        if isinstance(metadata, Mapping):
            for key in ("scene_obstacle_points_base_mm", "scene_points_world_rule_base_mm"):
                seen = metadata.get(key)
                if seen is not None and np.asarray(seen).size:
                    parts.append(np.asarray(seen, dtype=np.float64).reshape(-1, 3))
        return np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.float64)

    def _look_again_after_the_push(
        self, looked: LookedAround, judged: _Judgement, label: str, back_to: "JointPositions", *, attempt_index: int,
    ) -> LookedAround:
        """The look taken again from where the arm stood before the push, fused with the pick's earlier looks less the
        part they saw where it stood, and judged afresh: what the next attempt goes on with."""
        self._without_the_pushed_part(judged)
        judgement, _missed = self._look(label, back_to, None, attempt_index=attempt_index)
        withheld = ""
        if judgement is not None and judgement.valid and self._faces_unseen_by(judgement):
            withheld = self._unseen_faces(judgement.faces)
        relooked = LookedAround(
            visited=(*looked.visited, label), judged=judgement, dropped=looked.dropped, refused=looked.refused,
            views=tuple(self._look_views), generated=looked.generated, generated_view_deg=looked.generated_view_deg,
            generated_skipped=looked.generated_skipped, withheld=withheld,
        )
        self.looked_around = relooked
        self._open_looked = relooked
        return relooked

    # ------------------------------------------------------------------
    # Clear the blocker (the owner, 2026-10-02)
    # ------------------------------------------------------------------

    @property
    def blockers(self) -> "tuple[BlockerRecord, ...]":
        """What every clearing of a blocker the last run considered came to, in order (``blocker.BlockerRecord``): set
        aside, or why none was, or why the clearing stopped. Empty where none was considered."""
        return tuple(self._blockers)

    def _said_blocker(self, record: "BlockerRecord") -> None:
        self._blockers.append(record)
        _LOG.info("clear the blocker (%s): %s", record.code, record.sentence)

    def _clear_the_blockers(
        self,
        attempt_index: int,
        attempts: list[PickAttempt],
        looked: LookedAround,
        frame: PerceptionFrame,
        target_index: int | None,
        *,
        reasons: tuple[GraspFailureReason, ...],
        fused_target: "np.ndarray | None",
    ) -> "PickReport | bool":
        """Take away what stands in the way of the part's grasps before anything pushes it (the owner, 2026-10-02:
        "entweder das Objekt verschieben oder ein anderes Objekt nehmen, was im Weg liegt und dann das eigentliche Objekt
        aufheben"), where :attr:`push_gate` lets this pick change the scene.

        In turn, with no numeric budget: the neighbour whose removal frees the most grasps of the part
        (:meth:`_choose_a_blocker`) is gripped like any part with the part back in the planner world, set down on a free
        spot the camera saw or at :attr:`blocker_place` through the place verb, the arm goes back to the look, the
        pick's stale frames are let go, and the look is taken again (:meth:`_set_the_blocker_aside`). The clearing goes
        on while the part still has no grasp because of what the camera saw and each removal spared some of its
        refusals; it stops with a typed reason where a removal freed nothing (:data:`blocker.FREED_NOTHING`), and for
        good in this run.

        Returns ``False`` where nothing moved for it, and the push is considered next: no gate, no attempt left to grasp
        the part with, no separate blocker seen, none whose removal frees a grasp, none graspable, no free spot, a stop
        asked for, or the blocker's pick refused before anything was sent. ``True`` where a blocker was set aside: the
        next attempt goes on with the look taken again. A :class:`PickReport` where the pick ends: the controller cannot
        move, or the clearing stopped once something moved, where the arm stands (a person decides).
        """
        from src.robot.grasping.recovery import blocker as B  # noqa: PLC0415

        gate = self.push_gate
        judged = looked.judged
        if gate is None or judged is None or target_index is None:
            return False
        if self._blockers_stopped:
            _LOG.info("no blocker is cleared: the clearing of this pick stopped (%s)", self._blockers_stopped)
            return False
        if attempt_index + 1 >= self.max_attempts:
            self._said_blocker(B.BlockerRecord(code=B.REFUSED_NO_ATTEMPT_LEFT, sentence=(
                f"This pick has no attempt left after attempt {attempt_index + 1} of {self.max_attempts} to grasp the "
                "part with once a blocker is set aside, so none is taken away.")))
            return False
        current = looked
        before = _seen_refusals(judged.result)
        cleared = 0
        while True:
            assert current.judged is not None  # every look this loop goes on with judged the part
            chosen_one = self._choose_a_blocker(current.judged, gate, refusals=before)
            if isinstance(chosen_one, B.BlockerRecord):
                self._said_blocker(chosen_one)
                if not cleared:
                    return False
                self._blockers_stopped = chosen_one.code
                self._pending_looked = current
                return True
            moved = self._set_the_blocker_aside(attempt_index, attempts, current, chosen_one, reasons=reasons,
                                                fused_target=fused_target if not cleared else None)
            if isinstance(moved, PickReport):
                return moved
            if isinstance(moved, B.BlockerRecord):
                self._said_blocker(moved)
                if not cleared:
                    return False
                self._blockers_stopped = moved.code
                self._pending_looked = current
                return True
            cleared += 1
            current = moved
            now_judged = current.judged
            if now_judged is None or now_judged.valid or GraspFailureReason.ALL_COLLIDED not in now_judged.reasons:
                # The part has a grasp, or no longer meets what the camera saw: the next attempt goes on with this look.
                self._pending_looked = current
                return True
            now = _seen_refusals(now_judged.result)
            if now >= before:
                self._said_blocker(B.BlockerRecord(code=B.FREED_NOTHING, sentence=(
                    f"Setting the blocker aside freed no grasp of the part: {now} of its tries still meet what the camera "
                    f"saw beside it, against {before} before. No other blocker is taken away in this pick; the push is "
                    "considered next.")))
                self._blockers_stopped = B.FREED_NOTHING
                self._pending_looked = current
                return True
            before = now

    def _choose_a_blocker(self, judged: _Judgement, gate: "PushGate", *, refusals: int) -> Any:
        """The blocker to take away (a ``_BlockerChoice``), or why none (a ``blocker.BlockerRecord``); nothing moves.

        The candidates are the separate objects among what the calculator saw beside the part (Track A's obstacle
        points, ``blocker.clusters_of``) that may be gripped as a blocker (``blocker.not_a_blocker``: on what the part
        stands on, apart from it, away from the robot's base, narrow enough for the hand) and that this pick has not
        set down. For each, the calculator is asked again of the same frame with that object's pixels left out
        (:meth:`_ask_again`): how many grasps of the part its removal frees, and how many of the part's refusals by
        what the camera saw it spares. Those that free or spare any are taken in that order, the nearer the part first
        on a tie; the first the calculator grasps (:meth:`_blocker_grasp`) with somewhere to go
        (:meth:`_where_the_blocker_goes`) is the blocker."""
        from src.robot.grasping.recovery import blocker as B  # noqa: PLC0415
        from src.robot.grasping.recovery.blocker import clusters_of, mask_of, not_a_blocker  # noqa: PLC0415

        ranked = judged.ranked
        call = ranked.call if ranked is not None else None
        look = judged.look
        if call is None or look.camera_to_base is None or judged.target_index is None:
            return B.BlockerRecord(code=B.REFUSED_CANNOT_ASK_AGAIN, sentence=(
                "The look the part was judged on keeps no call of the calculator to ask again with a neighbour left out "
                "of its frame, so no blocker is taken away."))
        metadata = getattr(judged.result, "metadata", None)
        seen = metadata.get("scene_obstacle_points_base_mm") if isinstance(metadata, Mapping) else None
        if seen is None or not np.asarray(seen).size:
            return B.BlockerRecord(code=B.NO_BLOCKER_SEEN, sentence=(
                "The calculator saw nothing beside the part to take away, so no blocker is cleared."))
        target = self._pushed_part_points(judged, judged.target_index)
        target = target if target is not None else np.zeros((0, 3))
        model = judged.support_model
        width = self._open_width_mm() or _BLOCKER_OPEN_WIDTH_MM
        base = self._robot_base()
        base_reach = None if base is None else float(base.radius_mm) + _BASE_PADDING_MM
        depth = np.asarray(call.depth, dtype=np.float64)
        intrinsics = np.asarray(judged.frame.intrinsics, dtype=np.float64)
        target_mask = self._grown(np.asarray(getattr(call.seg, "mask", np.zeros(depth.shape[:2], bool))).astype(bool), 3)
        candidates: list[tuple[Any, float | None, np.ndarray]] = []
        said: list[str] = []
        for cluster in clusters_of(seen):
            centre = cluster.centre_mm
            if any(float(np.hypot(centre[0] - x, centre[1] - y)) <= _SET_DOWN_REACH_MM for x, y in self._set_down_at):
                said.append(f"({centre[0]:.0f}, {centre[1]:.0f}): a blocker this pick set down")
                continue
            if any(float(np.hypot(centre[0] - x, centre[1] - y)) <= _SET_DOWN_REACH_MM
                   for x, y in self._blockers_refused):
                said.append(f"({centre[0]:.0f}, {centre[1]:.0f}): a blocker whose grasp this pick saw refused")
                continue
            support = self._support_under_points(cluster.points_base_mm, model, judged)
            why = not_a_blocker(cluster, support_mm=support, target_points_mm=target, base_radius_mm=base_reach,
                                open_width_mm=width)
            if why:
                said.append(f"({centre[0]:.0f}, {centre[1]:.0f}): {why}")
                continue
            mask = mask_of(cluster.points_base_mm, shape=depth.shape[:2], intrinsics=intrinsics,
                           camera_to_base=look.camera_to_base, grow_px=_BLOCKER_MASK_GROWTH_PX) & ~target_mask
            if mask.any():
                candidates.append((cluster, support, mask))
        if not candidates:
            return B.BlockerRecord(code=B.NO_BLOCKER_SEEN, sentence=(
                "Nothing beside the part may be taken away as a blocker" + (": " + "; ".join(said) if said else ".")))
        frees: list[tuple[Any, float | None, np.ndarray, int, int]] = []
        for cluster, support, mask in candidates:
            blanked = depth.copy()
            blanked[mask] = 0.0
            again = self._ask_again(call, depth=blanked, drop_near=cluster.points_base_mm)
            freed = len(again.candidates) if again.is_success else 0
            fewer = refusals - _seen_refusals(again)
            centre = cluster.centre_mm
            _LOG.info("a blocker at (%.0f, %.0f, %.0f) mm, %s: taking it away frees %d grasp(s) of the part and spares %d "
                      "of its %d refusal(s) by what the camera saw", centre[0], centre[1], centre[2],
                      "x".join(f"{w:.0f}" for w in cluster.widths_mm) + " mm", freed, max(fewer, 0), refusals)
            if freed > 0 or fewer > 0:
                frees.append((cluster, support, mask, freed, max(fewer, 0)))
        if not frees:
            return B.BlockerRecord(code=B.NO_BLOCKER_FREES_A_GRASP, sentence=(
                f"Taking away none of the {len(candidates)} neighbour(s) beside the part frees a grasp of it or spares "
                "one of its refusals, asked of the calculator, so none is taken away."))
        centre_of_part = np.median(target[:, :2], axis=0) if target.shape[0] else np.zeros(2)
        frees.sort(key=lambda row: (-row[3], -row[4], float(np.hypot(*(row[0].centre_mm[:2] - centre_of_part)))))
        nowhere = 0
        too_long: list[float] = []
        carried = self._carried_length_mm()
        for cluster, support, mask, freed, fewer in frees:
            grasped = self._blocker_grasp(call, judged, cluster, mask)
            if grasped is None:
                continue
            # Between the close and the release only the planner holds what the hand carries, and it models a part this
            # far past the fingertips (the console's rule for every part it carries): a blocker reaching further is not
            # taken (the lead's review, 2026-10-02). An arm that models none carries it as it carries every part.
            reach = self._reach_past_the_fingertips(grasped, cluster, support)
            if carried is not None and reach > carried:
                centre = cluster.centre_mm
                _LOG.info("the blocker at (%.0f, %.0f) mm reaches %.0f mm past the fingertips, more than the %g mm the "
                          "planner models a carried part with: it is not taken", centre[0], centre[1], reach, carried)
                too_long.append(reach)
                continue
            place = self._where_the_blocker_goes(judged, cluster, support, grasped, gate)
            if place is None:
                nowhere += 1
                continue
            pose, where, spot_xy = place
            return _BlockerChoice(cluster=cluster, mask=mask, grasp=grasped, release=pose, where=where,
                                  spot_xy=spot_xy, freed=freed, fewer=fewer)
        if nowhere:
            return B.BlockerRecord(code=B.NO_FREE_SPOT, sentence=(
                f"{nowhere} blocker(s) could be gripped and no free spot the camera saw on what they stand on lies "
                f"{B.SPOT_FROM_THE_PART_MM:g} mm from the part with nothing within the hand's reach and "
                f"{B.SPOT_CLEARANCE_MM:g} mm of it, so none is taken away."))
        if too_long:
            return B.BlockerRecord(code=B.TOO_LONG_TO_CARRY, sentence=(
                f"{len(too_long)} blocker(s) could be gripped and reach {min(too_long):.0f} mm or more past the "
                f"fingertips, further than the {carried:g} mm the planner models a carried part with "
                "(safety.planning_world.payload.length_mm), so none is taken away."))
        return B.BlockerRecord(code=B.NO_GRASPABLE_BLOCKER, sentence=(
            f"The calculator finds no grasp on any of the {len(frees)} neighbour(s) whose removal would free the part, "
            "so none is taken away."))

    def _ask_again(
        self, call: _Call, *, depth: Any = None, seg: Any = None, drop_near: "np.ndarray | None" = None,
        kwargs: "Mapping[str, Any] | None" = None, neighbours: "tuple[Any, ...] | None" = None,
    ) -> GraspResult:
        """The calculator of ``call`` asked again of the same frame, with ``depth`` for its depth, ``seg`` for the part,
        and the fused neighbour points near ``drop_near`` left out; turned as every ranking is (:meth:`_closing_along`)
        and never drawn."""
        arguments = dict(call.kwargs if kwargs is None else kwargs)
        arguments.pop("rgb_image", None)
        if drop_near is not None and "scene_points_mm" in arguments and "camera_to_base" in arguments:
            arguments["scene_points_mm"] = _without_points_near(arguments["scene_points_mm"], drop_near,
                                                                arguments["camera_to_base"])
        calculator = self.calculator_for(call.camera_id)
        with self._asking_more_along_the_axis(calculator) as cap:
            result = calculator.compute_result(
                call.seg if seg is None else seg, call.depth if depth is None else depth, pixel_to_mm=None,
                dense_sampling=self.dense_sampling, grasp_sampling_mode=self._resolved_sampling_mode,
                other_object_masks=list(call.neighbours if neighbours is None else neighbours), **arguments)
        return self._closing_along(result, call.camera_id, cap)

    def _blocker_grasp(self, call: _Call, judged: _Judgement, cluster: Any, mask: np.ndarray) -> "GraspPoint | None":
        """The blocker's grasp, BASE, or ``None`` where the calculator finds none: the same frame and calculator as the
        part's, the blocker's mask for the part, the part and every other object as its neighbours, the support resolved
        under the blocker, and the part's own fused cloud left out."""
        seg = _Blocker(mask=mask)
        arguments = dict(call.kwargs)
        arguments.pop("geometry_points_base_mm", None)
        support = self._resolve_support(seg, judged.frame, judged.camera_to_base, None, judged.support_model)
        if support is not None and "support_plane" in arguments:
            arguments["support_plane"] = support.plane
        part_mask = getattr(call.seg, "mask", None)
        neighbours = tuple(m for m in (part_mask, *call.neighbours) if m is not None)
        result = self._ask_again(call, seg=seg, drop_near=cluster.points_base_mm, kwargs=arguments,
                                 neighbours=neighbours)
        best = result.best if result.is_success else None
        if best is None or best.frame is not GraspFrame.BASE:
            centre = cluster.centre_mm
            _LOG.info("the blocker at (%.0f, %.0f) mm gets no grasp (%s)", centre[0], centre[1],
                      ", ".join(str(reason) for reason in result.reasons) or "none in BASE")
            return None
        return best

    def _where_the_blocker_goes(
        self, judged: _Judgement, cluster: Any, support: float | None, grasp: GraspPoint, gate: "PushGate",
    ) -> "tuple[Pose, str, tuple[float, float]] | None":
        """The TCP pose that sets the blocker down, what it is, and where its centre lands; ``None`` where it may go
        nowhere. The owner's named place (:attr:`blocker_place`) where one is set, else a free spot of the support the
        camera saw (``blocker.free_spot``), the grasp moved there as it was taken (``blocker.release_pose``)."""
        from src.robot.grasping.recovery.blocker import SET_DOWN_AIR_MM, free_spot, release_pose  # noqa: PLC0415

        if self.blocker_place is not None:
            place = self.blocker_place
            xy = np.asarray(place.position_mm, dtype=np.float64)[:2]
            return place, f"the place the owner named at ({xy[0]:.0f}, {xy[1]:.0f}) mm", (float(xy[0]), float(xy[1]))
        table, occupied, reading = self._support_pixels(judged, cluster)
        target = self._pushed_part_points(judged, judged.target_index) if judged.target_index is not None else None
        target = target if target is not None else np.zeros((0, 3))
        jaw = self.gripper_model
        width = self._open_width_mm() or _BLOCKER_OPEN_WIDTH_MM
        reach = 0.5 * width + float(getattr(jaw, "finger_thickness_mm", 0.0) or 0.0)
        reach = max(reach, 0.5 * float(getattr(jaw, "palm_width_mm", 0.0) or 0.0))
        # The blocker comes down where its grasp puts it: the grasp's point lands on the spot's centre.
        held_at = np.asarray(grasp.position, dtype=np.float64)[:2]
        spot = free_spot(table_points_mm=table, occupied_points_mm=np.vstack([occupied, target, cluster.points_base_mm]),
                         blocker_points_mm=cluster.points_base_mm, target_points_mm=target,
                         workspace_xy=gate.cell.workspace.xy, hand_reach_mm=reach,
                         centre_xy=(float(held_at[0]), float(held_at[1])))
        if spot is None:
            return None
        under_spot = reading(np.array([spot.centre_xy]))
        rise = 0.0 if under_spot is None or support is None else float(under_spot) - float(support)
        shift = (spot.centre_xy[0] - float(held_at[0]), spot.centre_xy[1] - float(held_at[1]), rise + SET_DOWN_AIR_MM)
        pose = release_pose(grasp, shift_mm=shift)
        z = float(pose.position_mm[2])
        workspace = gate.cell.workspace
        if not workspace.min_mm[2] + _SET_DOWN_WORKSPACE_MARGIN_MM <= z <= workspace.max_mm[2] - _SET_DOWN_WORKSPACE_MARGIN_MM:
            return None
        return pose, (f"a free spot at ({spot.centre_xy[0]:.0f}, {spot.centre_xy[1]:.0f}) mm, "
                      f"{spot.distance_to_part_mm:.0f} mm from the part"), spot.centre_xy

    def _carried_length_mm(self) -> float | None:
        """How far past the fingertips the planner models a carried part, millimetres
        (``safety.planning_world.payload.length_mm``), where the arm says it models one; ``None`` where it models none,
        or cannot say."""
        declined = getattr(self.arm, "payload_declined_reason", None)
        if not callable(declined) or declined() is not None:
            return None
        payload = getattr(getattr(getattr(getattr(self.arm, "config", None), "safety", None), "planning_world", None),
                          "payload", None)
        length = getattr(payload, "length_mm", None)
        if isinstance(length, bool) or not isinstance(length, (int, float)) or not np.isfinite(float(length)):
            return None
        return float(length)

    def _reach_past_the_fingertips(self, grasp: GraspPoint, cluster: Any, support: float | None) -> float:
        """How far a blocker reaches past the fingertips along its grasp's approach, millimetres: its seen points, and
        their foot on the support under it, against the pads' reach ahead of the grasp centre. The planner's carried part
        starts at its hand spheres' tip, a little past the pads, so this reads the blocker no shorter than it models."""
        points = np.asarray(cluster.points_base_mm, dtype=np.float64).reshape(-1, 3)
        if support is not None and points.shape[0]:
            feet = points.copy()
            feet[:, 2] = float(support)
            points = np.vstack([points, feet])
        approach = np.asarray(grasp.approach, dtype=np.float64).reshape(3)
        approach = approach / float(np.linalg.norm(approach))
        reach = float(np.max((points - np.asarray(grasp.position, dtype=np.float64).reshape(3)) @ approach))
        ahead = getattr(self.gripper_model, "pad_ahead_mm", 0.0)
        return reach - (float(ahead) if isinstance(ahead, (int, float)) else 0.0)

    def _support_under_points(self, points: np.ndarray, model: Any, judged: _Judgement) -> float | None:
        """What the support reads under ``points``: the model's highest local reading under their foot, else the bench
        (the live world's declared plane, else ``grasping.support``'s height); ``None`` where nothing says."""
        if model is not None:
            under = model.height_under(np.asarray(points, dtype=np.float64)[:, :2])
            if under is not None:
                return float(under)
        return self._bench_top_mm()

    def _bench_top_mm(self) -> float | None:
        limits = getattr(getattr(self.arm, "live_planner_world", None), "limits", None)
        plane = getattr(limits, "support_plane_top_mm", None)
        if plane is not None:
            return float(plane)
        return float(self.support_config.height_mm) if self.support_config is not None else None

    def _support_pixels(self, judged: _Judgement, cluster: Any) -> "tuple[np.ndarray, np.ndarray, Any]":
        """What a free spot is read from, in the frame the part was judged on: the table there (the support surface's
        own pixels within its band where the camera world found one under the blocker, else the bench's band), every
        pixel standing over what it stands on by more than the band, and who reads the support under a point."""
        model = judged.support_model
        look = judged.look
        intrinsics = np.asarray(judged.frame.intrinsics, dtype=np.float64)
        depth = np.asarray(look.depth if look.depth is not None else judged.frame.depth_map, dtype=np.float64)
        matrix = np.asarray(look.camera_to_base, dtype=np.float64)
        stride = max(1, int(getattr(getattr(getattr(self.arm, "live_planner_world", None), "tuning", None),
                                    "pixel_stride", 2) or 2))
        rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[rows, cols]
        valid = np.isfinite(z) & (z > 0.0)
        rows, cols, z = rows[valid], cols[valid], z[valid]
        camera = np.column_stack([(cols - intrinsics[0, 2]) * z / intrinsics[0, 0],
                                  (rows - intrinsics[1, 2]) * z / intrinsics[1, 1], z])
        points = camera @ matrix[:3, :3].T + matrix[:3, 3]
        bench = self._bench_top_mm()
        band = float(getattr(getattr(getattr(self.arm, "live_planner_world", None), "limits", None),
                             "plane_clearance_mm", _SUPPORT_BAND_MM) or _SUPPORT_BAND_MM)
        reading = np.full(points.shape[0], np.nan if bench is None else float(bench))
        if model is not None and points.shape[0]:
            planes = model.local_plane_under(points[:, :2])
            if planes is not None:
                planes = np.asarray(planes, dtype=np.float64)
                covered = np.isfinite(planes[:, 0])
                nx, ny, nz, d = (planes[covered, i] for i in range(4))
                reading[covered] = (d - nx * points[covered, 0] - ny * points[covered, 1]) / nz
        known = np.isfinite(reading)
        occupied = points[known & (points[:, 2] > reading + band)]
        surface = model.surface_under(cluster.points_base_mm[:, :2]) if model is not None else None
        if surface is not None and model is not None:
            from src.robot.safety.planning.perceived import DepthView  # noqa: PLC0415

            table = model.table_points([DepthView(surface_depth_mm=depth, intrinsics=intrinsics, camera_to_base=matrix,
                                                  name="judged")], cluster.points_base_mm[:, :2])

            def read(xy: np.ndarray) -> float | None:
                return model.height_under(xy)
        else:
            table = points[known & (np.abs(points[:, 2] - reading) <= band)] if bench is not None else np.zeros((0, 3))

            def read(xy: np.ndarray) -> float | None:
                return bench
        return np.asarray(table, dtype=np.float64).reshape(-1, 3), occupied, read

    @staticmethod
    def _grown(mask: np.ndarray, pixels: int) -> np.ndarray:
        if pixels <= 0 or not mask.any():
            return mask
        from scipy import ndimage  # noqa: PLC0415

        return ndimage.binary_dilation(mask, iterations=int(pixels))

    def _set_the_blocker_aside(
        self, attempt_index: int, attempts: list[PickAttempt], looked: LookedAround, choice: "_BlockerChoice", *,
        reasons: tuple[GraspFailureReason, ...], fused_target: "np.ndarray | None",
    ) -> "LookedAround | PickReport | BlockerRecord":
        """Grip the blocker, set it down, go back to the look and look again: the look taken again, or why nothing moved
        (a ``BlockerRecord``), or the report of a pick that ends here.

        Before anything moves: a stop asked for, the controller, and where the arm stands, to come back to. Then the
        part goes back into the planner world (the blocker's pick meets it as any obstacle) and the blocker is held out
        of it while it is picked and set down. The pick is the policy's, as every grasp of the pick: the toggle is
        asked, never switched before the close, and closed once. A blocker pick refused before anything was sent ends the
        clearing with nothing moved. Once something moved, any failure, a stop asked for while the blocker is held, or a
        controller that stops, ends the pick where the arm stands: a person decides. The place is the place verb, its
        standoff the policy's. The move back to the look runs like a grasp approach (to a generated view on the straight
        joint line alone, :meth:`_back_to_the_look`). Then the pick's frames, which show the blocker where it stood, are
        let go, held again from the look taken now, and the earlier looks leave the views it is fused with."""
        from src.robot.core.errors import CameraWorldUnavailable  # noqa: PLC0415
        from src.robot.execution.looks import look_label  # noqa: PLC0415
        from src.robot.execution.robot import Robot  # noqa: PLC0415
        from src.robot.grasping.recovery import blocker as B  # noqa: PLC0415

        judged = looked.judged
        assert judged is not None  # the caller's
        cluster = choice.cluster
        centre = tuple(float(v) for v in cluster.centre_mm)
        widths = cluster.widths_mm
        said_of: dict[str, Any] = dict(centre_mm=centre, widths_mm=widths, freed=choice.freed,
                                       fewer_refusals=choice.fewer, place=choice.where)
        if self.should_cancel is not None and self.should_cancel():
            return B.BlockerRecord(code=B.REFUSED_STOP_REQUESTED, sentence=(
                "A stop was asked for before the blocker was taken, so nothing moved."), **said_of)
        controller = self._controller_cannot_move()
        if controller is not None:
            return self._blocker_stopped(attempt_index, attempts, judged, reasons=reasons, controller=controller,
                                         sentence=f"The controller cannot move, so no blocker is taken: {controller}",
                                         said_of=said_of, moved=False)
        back_to = self._joints_here()
        if back_to is None:
            return B.BlockerRecord(code=B.REFUSED_CANNOT_ASK_AGAIN, sentence=(
                "The arm could not say where it stands, so it could not come back to the look after the blocker, and "
                "none is taken."), **said_of)
        _LOG.info("taking away the blocker at (%.0f, %.0f, %.0f) mm (%s mm): it frees %d grasp(s) of the part, and goes "
                  "to %s", centre[0], centre[1], centre[2], "x".join(f"{w:.0f}" for w in widths), choice.freed,
                  choice.where)
        # The part back into the world: the blocker's pick meets it as any obstacle. The blocker out of it.
        self._keep_out.close()
        standoff = float(getattr(self.policy, "standoff_mm", 80.0))
        world = getattr(self.arm, "live_planner_world", None)
        with ExitStack() as scope:
            if world is not None:
                try:
                    scope.enter_context(keeping_out(self.arm, self._blocker_offer(judged, choice)))
                except CameraWorldUnavailable:
                    raise
                except Exception as exc:  # noqa: BLE001 (a world that refuses the offer refuses the clearing)
                    self._offer_masks_to_planner_world(judged.frame, judged.target_index,
                                                       target_points_base_mm=fused_target)
                    return B.BlockerRecord(code=B.BLOCKER_NOT_REACHED, sentence=(
                        f"The arm's camera world did not take the blocker to keep out ({type(exc).__name__}: {exc}), so "
                        "nothing moved."), **said_of)
            assert self.policy is not None
            try:
                picked = self.policy.execute(choice.grasp)
            except CameraWorldUnavailable as exc:
                self._blocker_stopped_by_the_camera_world(exc, said_of, "the blocker was picked")
                raise
            if picked.outcome is not PolicyOutcome.EXECUTED:
                sent, _reached, refused = self._what_the_try_did(picked, None)
                if refused and not sent:
                    scope.close()
                    self._offer_masks_to_planner_world(judged.frame, judged.target_index,
                                                       target_points_base_mm=fused_target)
                    return B.BlockerRecord(code=B.BLOCKER_NOT_REACHED, sentence=(
                        f"The blocker's grasp was refused before anything was sent ({getattr(picked.motion_status, 'value', None)}: "
                        f"{picked.motion_message or 'no reason was given'}), so nothing moved."), **said_of)
                # Refused once the arm stood over the blocker (its line in, or the lift judged as if carrying), the jaws
                # never closed: the hand is known empty and open, so the arm goes back to the look on a judged move and
                # this blocker is not tried again (the URSim gate of 2026-10-02 met it three times in eight). The move
                # is judged with the blocker still held out, as the line down to it was: the open jaws stand round it.
                in_the_air = ((picked.outcome is PolicyOutcome.MOTION_FAILED and refused)
                              or picked.outcome is PolicyOutcome.CARRIED_RETREAT_REFUSED)
                if in_the_air and not why_not_known_open(self.gripper):
                    return self._blocker_back_to_the_look(attempt_index, attempts, looked, judged, picked, back_to,
                                                          scope, reasons=reasons, said_of=said_of,
                                                          fused_target=fused_target)
                return self._blocker_stopped(
                    attempt_index, attempts, judged, reasons=reasons, controller=self._controller_cannot_move(),
                    sentence=(f"The blocker's pick ended {picked.outcome.value} once the arm had moved "
                              f"({getattr(picked.motion_status, 'value', None)}: {picked.motion_message or 'no reason was given'}): "
                              "the arm stays where it stopped, the blocker may be in the hand, and a person decides."),
                    said_of=said_of, fields=_motion_fields(picked),
                    gripper_fault=picked.outcome is PolicyOutcome.GRIPPER_FAULT)
            if self.should_cancel is not None and self.should_cancel():
                return self._blocker_stopped(
                    attempt_index, attempts, judged, reasons=reasons, controller=None, said_of=said_of,
                    sentence=("A stop was asked for with the blocker in the hand: the arm stays where it stands, nothing "
                              "more is commanded, and a person decides what happens to it."))
            controller = self._controller_cannot_move()
            if controller is not None:
                return self._blocker_stopped(attempt_index, attempts, judged, reasons=reasons, controller=controller,
                                             said_of=said_of, sentence=f"With the blocker in the hand: {controller}")
            robot = Robot.from_parts(arm=self.arm, gripper=self.gripper, lock_key=None)
            placed = robot.place(choice.release, standoff_mm=standoff)
            if not placed.ok:
                return self._blocker_stopped(
                    attempt_index, attempts, judged, reasons=reasons, controller=self._controller_cannot_move(),
                    said_of=said_of, fields=_handling_fields(placed),
                    sentence=(f"Setting the blocker down at {choice.where} ended {placed.outcome.value} "
                              f"({placed.message or 'no reason was given'}): the arm stays where it stopped, and a "
                              "person decides."))
        self._set_down_at.append(choice.spot_xy)
        fields = _handling_fields(placed)
        if self.should_cancel is not None and self.should_cancel():
            record = B.BlockerRecord(code=B.SET_ASIDE, sentence=(
                f"The blocker was set down at {choice.where}, and a stop was asked for: the arm stays over it."),
                moved=True, **said_of)
            self._said_blocker(record)
            attempts.append(PickAttempt(attempt_index=attempt_index, target_index=judged.target_index, reasons=reasons,
                                        score=0.0, action="clear_blocker", **fields, blocker=B.SET_ASIDE))
            emit(self.on_progress, PickStage.CANCELLED, attempt=attempt_index)
            return PickReport(outcome=PickOutcome.CANCELLED, attempts=tuple(attempts))
        try:
            back, generated = self._back_to_the_look(looked, back_to)
        except CameraWorldUnavailable as exc:
            self._blocker_stopped_by_the_camera_world(exc, said_of, "the arm went back to the look after the blocker")
            raise
        if not back.done:
            what = (f"{back.stopped.status.value}: {back.stopped.message}" if back.stopped is not None
                    else f"refused before anything was sent: {back.refused}")
            return self._blocker_stopped(
                attempt_index, attempts, judged, reasons=reasons, controller=back.controller, said_of=said_of,
                fields=_look_motion_fields(back.stopped) if back.stopped is not None else fields,
                sentence=(f"The blocker was set down at {choice.where}, and the move back to the look "
                          f"{look_label(back_to)} did not run ({what}): the arm stays where it stopped, nothing more is "
                          "commanded, and a person decides."))
        # The frames the pick holds show the blocker where it stood: they are let go, and held again from the look now.
        self._views_scope.close()
        self._views_scope = ExitStack()
        self._holding = False
        self._look_views = []
        self._fixed_views = None
        self._holding = self._views_scope.enter_context(holding_views(self.arm))
        label = f"{look_label(back_to)} after the blocker was set aside"
        judgement, _missed = self._look(label, back_to, None, attempt_index=attempt_index)
        withheld = ""
        if judgement is not None and judgement.valid and self._faces_unseen_by(judgement):
            withheld = self._unseen_faces(judgement.faces)
        relooked = LookedAround(
            visited=(*looked.visited, label), judged=judgement, dropped=looked.dropped, refused=looked.refused,
            views=tuple(self._look_views), generated=looked.generated if generated else "",
            generated_view_deg=looked.generated_view_deg if generated else None,
            generated_skipped=looked.generated_skipped, withheld=withheld,
        )
        self.looked_around = relooked
        self._open_looked = relooked
        record = B.BlockerRecord(code=B.SET_ASIDE, sentence=(
            f"The blocker was taken from ({centre[0]:.0f}, {centre[1]:.0f}) mm and set down at {choice.where}; the arm "
            f"went back to the look and looked again ({label})."), moved=True, **said_of)
        self._said_blocker(record)
        emit(
            self.on_progress, PickStage.ATTEMPT_FINISHED,
            attempt=attempt_index, action="clear_blocker", outcome=B.SET_ASIDE, target_index=judged.target_index,
            reasons=tuple(str(reason) for reason in reasons),
            extra={"blocker": B.SET_ASIDE, "blocker_at_mm": [round(v, 1) for v in centre], "set_down": choice.where,
                   "looked_again": label},
            **{k: v for k, v in fields.items() if v is not None},
        )
        attempts.append(PickAttempt(
            attempt_index=attempt_index, target_index=judged.target_index, reasons=reasons, score=0.0,
            action="clear_blocker", ordering_decision=None, **fields, **self._fusion_of(judged.target_index),
            blocker=B.SET_ASIDE,
        ))
        return relooked

    def _blocker_back_to_the_look(
        self, attempt_index: int, attempts: list[PickAttempt], looked: LookedAround, judged: _Judgement,
        picked: PolicyReport, back_to: "JointPositions", held: ExitStack, *, reasons: tuple[GraspFailureReason, ...],
        said_of: Mapping[str, Any], fused_target: "np.ndarray | None",
    ) -> "BlockerRecord | PickReport":
        """After a blocker's grasp refused in the air, the hand never closed: the move back to the look, and a
        ``BLOCKER_NOT_REACHED`` that lets the push be considered; or, where that move does not run, the stop where the
        arm stands, a person to decide.

        ``held`` holds the blocker out of the arm's live world. The move back is judged with it held, the world the
        line down to the blocker was judged in: the open jaws stand round the blocker, inside its box once it is back
        (the URSim re-run of 2026-10-02 met every move from there refused). It is let go once the move ended or
        stopped, and the part is offered again after that."""
        from src.robot.core.errors import CameraWorldUnavailable  # noqa: PLC0415
        from src.robot.execution.looks import look_label  # noqa: PLC0415
        from src.robot.grasping.recovery import blocker as B  # noqa: PLC0415

        why = (f"{getattr(picked.motion_status, 'value', None) or picked.outcome.value}: "
               f"{picked.motion_message or 'no reason was given'}")
        try:
            back, _generated = self._back_to_the_look(looked, back_to)
        except CameraWorldUnavailable as exc:
            self._blocker_stopped_by_the_camera_world(
                exc, said_of, "the arm went back to the look after the blocker's grasp was refused")
            raise
        finally:
            held.close()
        if not back.done:
            what = (f"{back.stopped.status.value}: {back.stopped.message}" if back.stopped is not None
                    else f"refused before anything was sent: {back.refused}")
            return self._blocker_stopped(
                attempt_index, attempts, judged, reasons=reasons, controller=back.controller, said_of=said_of,
                fields=_look_motion_fields(back.stopped) if back.stopped is not None else _motion_fields(picked),
                sentence=(f"The blocker's grasp was refused once the arm stood over it ({why}), and the move back to "
                          f"the look {look_label(back_to)} did not run ({what}): the arm stays where it stopped, the jaws "
                          "open and empty, nothing more is commanded, and a person decides."))
        centre = said_of.get("centre_mm")
        if centre is not None:
            self._blockers_refused.append((float(centre[0]), float(centre[1])))
        self._offer_masks_to_planner_world(judged.frame, judged.target_index, target_points_base_mm=fused_target)
        return B.BlockerRecord(code=B.BLOCKER_NOT_REACHED, sentence=(
            f"The blocker's grasp was refused once the arm stood over it ({why}); the jaws never closed, so the arm went "
            f"back to the look {look_label(back_to)}, and this blocker is not tried again."), moved=True, **said_of)

    def _blocker_stopped_by_the_camera_world(self, exc: BaseException, said_of: Mapping[str, Any], when: str) -> None:
        """Keep a clearing the camera world stopped (``CameraWorldUnavailable``) as one that stopped where the arm stands,
        before the raise goes on as on every motion of a pick: the service then reports it as a stopped push of trigger
        ``clear_the_blocker`` on the fault's report, which needs a person (the blocker may be in the hand)."""
        from src.robot.grasping.recovery import blocker as B  # noqa: PLC0415
        from src.robot.grasping.recovery.push_gate import PickPush  # noqa: PLC0415

        sentence = (f"While {when}, the camera world could not vouch for the cell ({type(exc).__name__}: {exc}): the arm "
                    "stays where it stopped, the blocker may be in the hand, nothing more is commanded, and a person "
                    "decides what happens next.")
        self._said_blocker(B.BlockerRecord(code=B.STOPPED_WHERE_THE_ARM_STANDS, sentence=sentence, moved=True, **said_of))
        self._pushes.append(PickPush(trigger="clear_the_blocker", code=B.STOPPED_WHERE_THE_ARM_STANDS, sentence=sentence,
                                     motion_started=True, arm_moved=True, stopped=True))
        _LOG.error("clearing the blocker stopped: %s", sentence)

    def _blocker_offer(self, judged: _Judgement, choice: "_BlockerChoice") -> SegmentationOffer:
        """What the arm's live world keeps out while the blocker is picked and set down: its points, as the target of
        those motions, and its mask under the camera that took the frame where the source names one."""
        stamp = getattr(judged.frame, "timestamp", None)
        points = np.asarray(choice.cluster.points_base_mm, dtype=np.float64)
        camera = self._perception_camera_name()
        when = float(stamp) if stamp is not None else time.time()
        if chosen(camera):
            return SegmentationOffer(captured_at_s=when, camera=camera, exclude_masks=(choice.mask,),
                                     target_points_base_mm=points, target_label="blocker")
        return SegmentationOffer(captured_at_s=when, target_points_base_mm=points, target_label="blocker")

    def _blocker_stopped(
        self, attempt_index: int, attempts: list[PickAttempt], judged: _Judgement, *,
        reasons: tuple[GraspFailureReason, ...], controller: str | None, sentence: str, said_of: Mapping[str, Any],
        fields: "_MotionFields | None" = None, moved: bool = True, gripper_fault: bool = False,
    ) -> PickReport:
        """The attempt of a clearing that ends the pick: nothing more is commanded. Once something moved it stopped where
        the arm stands and a person decides: it is kept as a stopped push of trigger ``clear_the_blocker``, which the
        service reports as one that stopped where the arm stands (``needs_person``). ``CONTROLLER_NOT_OPERATIONAL``
        where the controller cannot move, ``GRIPPER_FAULT`` for a hand that raised, ``ABORTED`` otherwise."""
        from src.robot.grasping.recovery import blocker as B  # noqa: PLC0415
        from src.robot.grasping.recovery.push_gate import PickPush  # noqa: PLC0415

        code = B.STOPPED_WHERE_THE_ARM_STANDS
        self._said_blocker(B.BlockerRecord(code=code, sentence=sentence, moved=moved, **said_of))
        if moved:
            self._pushes.append(PickPush(trigger="clear_the_blocker", code=code, sentence=sentence, motion_started=True,
                                         arm_moved=True, stopped=True, controller_stopped=controller is not None))
        _LOG.error("clearing the blocker stopped: %s", sentence)
        outcome = (PickOutcome.CONTROLLER_NOT_OPERATIONAL if controller is not None
                   else PickOutcome.GRIPPER_FAULT if gripper_fault else PickOutcome.ABORTED)
        motion: _MotionFields = fields if fields is not None else {
            "motion_status": None, "motion_message": sentence, "motion_error": None, "camera_world": None,
            "camera_world_reason": None}
        emit(
            self.on_progress, PickStage.ATTEMPT_FINISHED,
            attempt=attempt_index, action="clear_blocker", outcome=str(outcome), target_index=judged.target_index,
            extra={"blocker": code, "blocker_reason": sentence},
            **{k: v for k, v in motion.items() if v is not None},
        )
        attempts.append(PickAttempt(
            attempt_index=attempt_index, target_index=judged.target_index,
            reasons=(GraspFailureReason.CONTROLLER_NOT_OPERATIONAL,) if controller is not None else (),
            score=0.0, action="clear_blocker", ordering_decision=None,
            **motion, blocker=code,
        ))
        return PickReport(outcome=outcome, attempts=tuple(attempts))

    def _without_the_pushed_part(self, judged: _Judgement) -> None:
        """Take the pushed part out of every earlier look's view: those views saw it where it stood before the push,
        and fused with the next look they would put it there again. Its other objects stay. The fixed cameras' views
        were taken before the push as well, and are acquired again at the next look."""
        kept: list[_LookView] = []
        for seen in self._look_views:
            blob = judged.members.get(seen.name)
            if blob is None or seen.view is None or not 0 <= blob < len(seen.blob_objects):
                kept.append(seen)
                continue
            dropped = seen.blob_objects[blob]
            blobs = tuple(index for index in seen.blob_objects if index != dropped)
            view = replace(
                seen.view,
                masks=tuple(mask for number, mask in enumerate(seen.view.masks) if number != blob),
                segmentations=tuple(seg for number, seg in enumerate(seen.view.segmentations) if number != blob),
            ) if blobs else None
            kept.append(replace(
                seen, view=view, blob_objects=blobs,
                clouds=tuple(np.zeros((0, 3), dtype=np.float64) if index == dropped else cloud
                             for index, cloud in enumerate(seen.clouds)),
                surfaces=tuple(None if index == dropped else surface for index, surface in enumerate(seen.surfaces)),
            ))
        self._look_views = kept
        self._fixed_views = None

    def _remember_failed_part(
        self, frame: PerceptionFrame, target_index: int | None, fused_cloud: "np.ndarray | None", *,
        centre: "Maybe[tuple[float, float, float] | None]" = UNSET,
    ) -> "tuple[float, float, float] | None":
        """Keep the part this attempt failed on, as ``next_target`` skips it (:attr:`failed_part`), and return where it
        stood: its label, its BASE centre (``centre`` where the caller read it already) and its footprint."""
        from src.robot.grasping.recovery.exclusion_zones import footprint_diagonal_mm  # noqa: PLC0415
        from src.robot.grasping.recovery.push_gate import FailedPart  # noqa: PLC0415

        if chosen(centre):
            where = centre
        else:
            where = self._target_centre_base_mm(frame, target_index, fused_cloud_base_mm=fused_cloud)
        label = self._label_key_of(target_index)
        if where is None or target_index is None:
            self.failed_part = None
            return where
        points = fused_cloud if fused_cloud is not None and np.asarray(fused_cloud).size else \
            self._target_surface_base_mm(frame, target_index)
        diagonal = footprint_diagonal_mm(points) if points is not None and len(points) >= 2 else None
        self.failed_part = FailedPart(label=label, centre_mm=where, footprint_diagonal_mm=diagonal) if label else None
        return where

    def _label_key_of(self, index: int | None) -> str:
        """The name object ``index`` of the last ranking goes by at the label gate: its label, else its prim path."""
        if index is None or not 0 <= index < len(self._last_scene_objects):
            return ""
        return _label_key(getattr(self._last_scene_objects[index], "segmentation", None))

    def _exclusion_of(self, zones: "ExclusionZones", index: int, obj: Any, frame: PerceptionFrame) -> str:
        """What keeps object ``index`` of the frame being ranked out of the pick: ``"region"`` where it stands in a
        region its task keeps out (whatever zone holds it too), ``"zone"`` where only a zone ``next_target`` made for its
        label holds it, ``""`` where nothing does. The zones say whether anything applies to its label (an empty one
        included) before its centre is read."""
        label = _label_key(obj.segmentation)
        if not zones.applies(label):
            return ""
        cloud = getattr(obj, "fused_cloud_base_mm", None) if self._current_look is not None else None
        centre = self._target_centre_base_mm(frame, index, fused_cloud_base_mm=cloud)
        if centre is None:
            return ""
        if zones.kept_out_by_a_region(label=label, centre_mm=centre):
            return "region"
        return "zone" if zones.excludes(label=label, centre_mm=centre) else ""

    def _only_excluded(self, zones: "ExclusionZones", excluded: list[int], *, by_regions: bool) -> GraspResult:
        """The result of a frame whose every part of the label is one ``next_target`` skips or its task keeps out: no
        candidate, no reason a recovery could act on, the zones' sentence, and whether a region of the task held every
        one of them (``by_regions``)."""
        label = self.target_label or self._label_key_of(excluded[0])
        sentence = zones.only_excluded_sentence(label=label, count=len(excluded))
        return GraspResult(reasons=(), telemetry={"only_excluded": sentence, "excluded_parts": len(excluded),
                                                  "excluded_by_regions": bool(by_regions)})

    @staticmethod
    def _only_excluded_of(result: "GraspResult | None") -> str:
        """The sentence of a result that saw only parts ``next_target`` skips, ``""`` otherwise."""
        telemetry = getattr(result, "telemetry", None)
        said = telemetry.get("only_excluded") if isinstance(telemetry, Mapping) else None
        return said if isinstance(said, str) else ""

    @staticmethod
    def _excluded_by_regions_of(result: "GraspResult | None") -> bool:
        """Whether a result that saw only skipped parts saw each of them in a region its task keeps out."""
        telemetry = getattr(result, "telemetry", None)
        return isinstance(telemetry, Mapping) and telemetry.get("excluded_by_regions") is True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shadow_telemetry_from_winner(
        self, candidates: tuple[Any, ...]
    ) -> ShadowSuccessTelemetry:
        """Re-derive the shadow-success telemetry from the top-1 candidate's metadata.

        Shared by the blend and uncertainty reranks, which both re-point the telemetry at the
        executed winner when a rerank changes top-1.
        """
        new_winner_md = getattr(candidates[0], "metadata", None) or {}
        new_prob = new_winner_md.get(SHADOW_METADATA_KEY_PROBABILITY)
        ctx = self.shadow_success_context
        return ShadowSuccessTelemetry(
            predicted_success_probability=(
                float(new_prob)
                if isinstance(new_prob, (int, float)) and not isinstance(new_prob, bool)
                else None
            ),
            success_probability_model_version=(
                ctx.version_label if ctx is not None else None
            ),
            model_lifecycle_phase=(ctx.lifecycle_phase if ctx is not None else None),
        )

    def _apply_rerank_result(
        self,
        best_result: GraspResult,
        shadow_telemetry: ShadowSuccessTelemetry | None,
        candidates: tuple[Any, ...],
        telemetry: RankingBlendTelemetry | UncertaintyRerankTelemetry | None,
    ) -> tuple[GraspResult, ShadowSuccessTelemetry | None]:
        """Apply a rerank result to ``best_result``; shared by the blend and uncertainty reranks.
        When the rerank fired (``telemetry.applied``) the reranked candidates and the new top score
        are swapped in; when it also changed top-1, ``shadow_telemetry`` is re-pointed at the new
        winner so the executed candidate's probability is reported. Otherwise the inputs come back
        unchanged."""
        if telemetry is not None and telemetry.applied:
            new_top_score = float(candidates[0].score) if candidates else 0.0
            best_result = replace(
                best_result, candidates=candidates, top_score=new_top_score
            )
            if telemetry.top1_changed:
                shadow_telemetry = self._shadow_telemetry_from_winner(candidates)
        return best_result, shadow_telemetry

    def _stamp_deep_ranker(  # noqa: ANN001, ANN202
        self, result, extra_kwargs: dict, fused_scene, index: int,
    ) -> None:
        """Score this attempt's candidates with the learned ranker and stamp the result. Never acts.

        Off by default (``deep_ranker_context is None``) and then this is one attribute read, so the
        default path is byte-identical.

        Two frame checks guard two hazards that sit one line apart:

        * ``support_plane`` defaults to ``Frame.CAMERA``. Its ``offset_mm`` is a BASE height only when
          the plane says so; a camera-frame plane read as a height puts 577 mm where -7 mm belongs.
        * ``scene_points_mm`` in ``extra_kwargs`` is CAMERA-frame, so the obstacle cloud is built
          from ``fused_scene`` instead, which is BASE, because the features are BASE. The
          comprehension below reads ``fused_scene.indices``, which ``FusedSceneGeometry`` does not
          define, so ``obstacle_points_mm`` is always ``None`` and the ranker scores every candidate
          without neighbour geometry.

        Nothing here may raise: the scorer's opinion is optional, the pick is not.
        """
        context = self.deep_ranker_context
        if context is None:
            return
        try:
            from src.robot.grasping.deep.ranker.shadow import score_candidates  # noqa: PLC0415

            target = extra_kwargs.get("geometry_points_base_mm")
            support_z = 0.0
            plane = extra_kwargs.get("support_plane")
            if plane is not None:
                frame = getattr(plane, "frame", None)
                if getattr(frame, "value", frame) != "base":
                    telemetry = {"deep_ranker_scored": False,
                                 "deep_ranker_reason": f"support plane is in {frame}, not BASE"}
                    _stamp(result, telemetry)
                    return
                support_z = float(plane.offset_mm)

            obstacles = None
            obstacle_objects = 0
            if fused_scene is not None:
                # Every other object's fused cloud. Read from `clouds_base_mm`, which is the tuple
                # the fused scene actually carries, rather than from an attribute name looked up
                # with a default: `getattr(fused_scene, "indices", ())` named nothing on this type,
                # so it returned empty on every pick and the ranker scored every candidate against
                # no obstacle at all, with `jaw_clearance_mm` pinned at its no-obstacle value.
                others = [
                    cloud for other in range(len(fused_scene.clouds_base_mm))
                    if other != index and (cloud := fused_scene.cloud_for(other)) is not None
                    and cloud.size
                ]
                obstacle_objects = len(others)
                if others:
                    import numpy as _np  # noqa: PLC0415

                    obstacles = _np.vstack(others)

            outcome = score_candidates(
                context.ranker, context.spec, list(getattr(result, "candidates", ()) or ()),
                target_points_mm=target, obstacle_points_mm=obstacles, support_z_mm=support_z,
            )
            # How many objects the clearance was measured against. A zero here is what the defect
            # above looked like from the outside, and it looked like nothing at all.
            _stamp(result, {**outcome.as_dict(), "deep_ranker_obstacle_objects": obstacle_objects})
        except Exception:  # noqa: BLE001 (an optional observer may never cost an attempt)
            _LOG.exception("deep_ranker shadow scoring failed; the pick is unaffected")

    def _controller_cannot_move(self) -> str | None:
        """A one-line reason when the controller is unable to move, else ``None``.

        Asked only after a motion has already failed: a healthy run pays nothing, because a healthy
        run never calls this. ``get_robot_status()`` is four cheap enum reads plus one dashboard
        round trip, which is fine on a failure path and would not be fine per attempt.

        The criterion is ``not is_operational`` rather than ``is_stopped``: a controller powered off
        mid-session reports ``protective_stopped=False``, ``emergency_stopped=False`` and
        ``is_stopped=False``, correctly, since none of those flags describes a power-off, while
        being exactly as unable to move. ``is_operational`` catches it.

        A halted arm cannot move either, and is said in the halt's own words: the latch is read
        first, of any arm that carries one (``SupportsHalt``, the dummy included), then a status that
        says so (``RobotStatus.halted``). A halt is no controller stop, so it never reads "clear the
        stop": a person confirms the cell is clear, then Restart.

        Otherwise inert by construction for every arm that is not a real UR: only ``URRobotArm``
        implements ``SupportsRobotStatus``, so the sim arm, the dummy arm and every other arm that is
        not halted take the ``None`` path and behave byte-identically. No flag needed.
        """
        # No lazy import here: this module imports `src.robot.core` at module level, so deferring
        # `SupportsRobotStatus` defers 0 modules and 0.0 ms. The module is loaded long before this
        # function can be called.
        halted = halt_state_of(self.arm)
        if halted is not None:
            return halted_refusal(halted.reason, _HALTED_WHERE_IT_STANDS)
        if not isinstance(self.arm, SupportsRobotStatus):
            return None
        try:
            status = self.arm.get_robot_status()
        except Exception:  # noqa: BLE001 (a diagnosis must never replace the fault it explains)
            return None
        said = getattr(status, "halted", "")
        if isinstance(said, str) and said:
            return halted_refusal(said, _HALTED_WHERE_IT_STANDS)
        if status.is_operational:
            return None
        detail = f": {status.message}" if status.message else ""
        return (
            f"the controller cannot move: robot_mode={status.robot_mode.value}, "
            f"safety_mode={status.safety_mode.value}, protective_stop={status.protective_stopped}, "
            f"emergency_stop={status.emergency_stopped}{detail}. Nothing is wrong with the scene or "
            "the grasp, and no retry or re-perceive will change it; the cell needs a human. Clear "
            "the stop where the arm is visible, or power it back on, then run again."
        )

    def _map_policy_outcome(
        self, outcome: PolicyOutcome
    ) -> tuple[str, PickOutcome]:
        """Map a :class:`PolicyOutcome` to the pair the success path records.

        The pair is an attempt-action label and a :class:`PickOutcome`. The camera-frame and
        approach-blocked outcomes keep their own labels; any outcome not enumerated here,
        MOTION_FAILED among them, falls through to EXECUTION_FAILED.
        """
        if outcome is PolicyOutcome.EXECUTED:
            return "executed", PickOutcome.EXECUTED
        if outcome is PolicyOutcome.OBJECT_NOT_DETECTED:
            return "object_not_detected", PickOutcome.OBJECT_NOT_DETECTED
        if outcome is PolicyOutcome.CAMERA_FRAME_REJECTED:
            # The policy refused a camera-frame grasp; surface distinctly.
            return "camera_frame_rejected", PickOutcome.CAMERA_FRAME_REJECTED
        if outcome is PolicyOutcome.APPROACH_PATH_BLOCKED:
            # Every candidate's approach/retreat sweep hit a neighbour: refused, no motion.
            return "approach_path_blocked", PickOutcome.APPROACH_PATH_BLOCKED
        if outcome is PolicyOutcome.GRIPPER_FAULT:
            # The hand needs a person; the policy's report carries why, and the attempt copies it.
            return "gripper_fault", PickOutcome.GRIPPER_FAULT
        return "execution_failed", PickOutcome.EXECUTION_FAILED

    def _maybe_run_ranking_shadow(
        self,
        *,
        pre_blend_candidates: tuple,
        post_blend_candidates: tuple,
        attempt_index: int,
    ) -> None:
        """Invoke the ranking shadow router.

        Strict no-op when:
          * :attr:`shadow_router` is :data:`None`.
          * The router has no :attr:`ranking_policy`.
          * The pre-blend candidate set is empty.

        Builds 15-key feature rows per candidate from the ``shadow`` +
        ``rl_state_features`` metadata blocks plus the three
        ``geometric_score`` / ``feasibility_score`` /
        ``shadow_predicted_success_probability`` signals, and stashes the
        typed :class:`RankingShadowTelemetry` on
        :attr:`_ranking_telemetry`. Any exception is converted to a
        fallback row by the router itself, and anything raised while
        the rows are built or the router runs is caught inside
        :meth:`_ShadowTelemetryAggregator.run_ranking_shadow`, which
        returns ``(None, None, None)`` and leaves the three slots
        unset. This helper adds no handler of its own.
        """

        router = self.shadow_router
        if router is None or getattr(router, "ranking_policy", None) is None:
            return
        if not pre_blend_candidates:
            return
        # The wired path delegates to the aggregator; assign all 3 slots from the return. The
        # aggregator returns (None, None, None) on its broad except. The no-router / no-candidates
        # guards above return early without writing any slot, so a direct call leaves all three at
        # their construction-default None.
        (
            self._ranking_telemetry,
            self._candidate_log,
            self._behavior_candidate_id,
        ) = self._shadow.run_ranking_shadow(
            pre_blend_candidates=pre_blend_candidates,
            post_blend_candidates=post_blend_candidates,
            attempt_index=attempt_index,
        )

    # ------------------------------------------------------------------
    # Sequencing-shadow helpers
    # ------------------------------------------------------------------

    def _finalize_sequencing_shadow(self) -> None:
        """Delegate to the aggregator's sequencing shadow (called from :meth:`run`'s ``finally``).
        Assigns both _sequencing_telemetry and _sequencing_states from the return; on the unwired
        path the aggregator returns (None, None), so _sequencing_states keeps its per-pick reset
        value."""
        telemetry, states = self._shadow.finalize_sequencing(attempts=self._sequencing_current_attempts)
        self._sequencing_telemetry = telemetry
        if states is not None:
            self._sequencing_states = states

    def _finalize_perception_shadow(self) -> None:
        """Delegate to the aggregator's perception-budget shadow.

        Called from :meth:`run`'s ``finally``. Assigns the slot from the return; the aggregator
        returns None on the unwired and except paths.
        """
        self._perception_telemetry = self._shadow.finalize_perception(
            attempts=self._sequencing_current_attempts
        )

    def _finalize_recovery_shadow(self) -> None:
        """Delegate to the aggregator's recovery shadow.

        Called from :meth:`run`'s ``finally``. Assigns the slot from the return; the aggregator
        returns None on the unwired, success and except paths.
        """
        self._recovery_telemetry = self._shadow.finalize_recovery(
            attempts=self._sequencing_current_attempts
        )

    # ------------------------------------------------------------------

    def _external_scene_points_camera_frame(
        self, camera_to_base: "Transform | None"
    ) -> "np.ndarray | None":
        """The fused multi-view scene cloud expressed in the captured frame's CAMERA frame.

        The cloud is :attr:`external_scene_points_base_mm`, which is BASE, and the CAMERA frame is
        the frame the collision filter plans in. Returns ``None`` when no cloud is set or no
        resolver gave a transform. ``base->camera = inv(camera_to_base)``.
        """
        pts = self.external_scene_points_base_mm
        if pts is None or camera_to_base is None:
            return None
        return self._base_points_to_camera(pts, camera_to_base)

    @staticmethod
    def _base_points_to_camera(
        points_base_mm: "np.ndarray", camera_to_base: "Transform"
    ) -> "np.ndarray | None":
        """BASE mm to the captured frame's CAMERA mm, the frame the collision filter plans in."""

        base = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
        if base.size == 0:
            return None
        m = np.asarray(camera_to_base.to_matrix(), dtype=np.float64)  # CAMERA -> BASE
        inv = np.linalg.inv(m)  # BASE -> CAMERA
        return (inv[:3, :3] @ base.T).T + inv[:3, 3]

    def _container_wall_points_base_mm(self) -> "np.ndarray | None":
        """The declared container walls as BASE points, built once and reused.

        Without them the candidate filter has no wall geometry at all: a target cloud, the
        neighbours' clouds and a support plane, and a plane is a floor. Replaying the reference
        verdict on the rank-0 winners that lose the ranking gap on ``finger_collision`` puts 60.9 %
        of them against a bin wall rather than an object (in the ``bin`` family, 14 of 14).
        Perception cannot supply it: no detector segments a wall as an instance. So the wall is
        declared, and this turns the declaration into the points every existing check already reads.

        Cached because it depends only on config: rebuilding a fixed bin every pick would be pure
        waste in the one path that runs on every attempt.
        """

        support = self.support_config
        container = getattr(support, "container", None) if support is not None else None
        if container is None or not bool(getattr(container, "wall_collision_enabled", False)):
            return None
        low = getattr(container, "interior_min_mm", None)
        high = getattr(container, "interior_max_mm", None)
        if low is None or high is None:
            # The schema refuses this combination at load time; a hand-built config can still reach
            # here, and a silently wall-free cell is exactly what this seam exists to prevent.
            _LOG.warning(
                "grasping.support.container.wall_collision_enabled is set but the interior box is "
                "not declared; the walls are not collision geometry for this attempt")
            return None
        key = (tuple(low), tuple(high), float(container.wall_thickness_mm),
               float(container.wall_sample_mm))
        if self._container_wall_cache is None or self._container_wall_cache[0] != key:
            from src.robot.grasping.collision import (  # noqa: PLC0415
                container_wall_points_base_mm,
            )

            points = container_wall_points_base_mm(
                low, high,
                thickness_mm=float(container.wall_thickness_mm),
                sample_mm=float(container.wall_sample_mm),
            )
            self._container_wall_cache = (key, points)
        return self._container_wall_cache[1]

    def _fused_scene(
        self,
        frame: PerceptionFrame,
        camera_to_base: "Transform | None",
    ) -> "FusedSceneGeometry | None":
        """What the whole rig sees, per segmentation of ``frame``, or ``None`` to stay single-view.

        The one place the multi-camera policy lives, because it is the one place that holds both the
        config and the rig. Everything geometric is delegated to
        :mod:`~src.robot.grasping.multiview.scene_geometry`.

        A cell configured for three cameras that quietly runs on one (a dropped trigger, a dirty
        lens, a calibration artifact that stopped loading) looks exactly like a healthy cell from
        the outside, just worse at grasping. So every stand-down reason is either a warning or a
        refusal, never silence, and the telemetry records which cameras actually contributed rather
        than which were configured.

        A look of a wrist pick is fused elsewhere (:meth:`_fused_looks`): with the pick's earlier looks, and with the
        fixed cameras acquired once for the pick.
        """

        # The counters describe THIS pick. Cleared before any stand-down below, or a pick that stands
        # down early keeps the dict of the last pick that fused, and the service turns it into this
        # record's `fused_view_count` (review of 2026-09-24). The other cameras' views go the same
        # way: the scene builder promotes objects from them (`promote_unmatched`), and views kept
        # from the last pick that fused would bring an object seen at another moment into this scene.
        self._fusion_geometry_telemetry = {}
        self._last_other_views = ()
        if self._current_look is not None:
            return self._fused_looks(frame, camera_to_base)
        config = self.fusion_geometry_config
        if config is None or not bool(config.enabled):
            return None
        if not self._fixed_cameras_placeable(camera_to_base):
            return None
        assert camera_to_base is not None  # narrowed by _fixed_cameras_placeable

        from src.robot.grasping.multiview.association import AssociationMetric
        from src.robot.grasping.multiview.scene_geometry import fuse_scene_geometry

        # The fusion clock (`LatencyStage.FUSION`) starts once the other cameras' frames are in hand,
        # before they are placed (`_fixed_camera_views` reads it). Their capture, detection and
        # segmentation are perception, and a fusion budget read against them would breach for a
        # reason that is not fusion. It stops when the fused scene is built.
        views, missing, grounded_nothing, fusion_t0_ns = self._fixed_camera_views(config)

        # One entry per segmentation, in order, and not compacted. `cloud_for(idx)` is addressed by
        # segmentation index at the candidate loop, so dropping a mask-less segmentation here would
        # shift every later object down one and hand the calculator a different object's surface.
        # Silently: the lookup is bounds-safe and returns a cloud, just the wrong one. Unreachable
        # today because `SegmentationResult.mask` is not optional, which is exactly why it is worth
        # closing now rather than when someone makes it so. An empty mask stands in for a
        # segmentation that has none, and `fuse_scene_geometry` associates nothing to it.
        #
        # The stand-in has the frame's shape, because the back-projection validates mask against
        # depth and a token-sized array would raise where the real one only selects nothing.
        _empty = np.zeros(np.asarray(frame.depth_map).shape[:2], dtype=bool)
        primary_masks = tuple(
            _empty if mask is None else mask
            for mask in (getattr(seg, "mask", None) for seg in frame.segmentations)
        )
        if not any(mask is not _empty for mask in primary_masks):
            primary_masks = ()
        if not views or not primary_masks:
            # Still stamped, and with the counters: "no camera contributed" and "two cameras
            # looked and found nothing" are different answers, and the difference is the whole
            # reason a second camera was bought.
            self._fusion_geometry_telemetry = {
                "fused_views_used": 0,
                "fused_objects": 0,
                "cameras_grounded_nothing": len(grounded_nothing),
                "cameras_absent": len(missing),
            }
            return None

        # Kept for the scene builder, which needs the same views and must not re-acquire them: a
        # second `acquire_all` would be a second detect and segment pass per camera, and worse, a
        # second moment, so the objects would be grouped across frames taken at different times.
        self._last_other_views = tuple(views)

        with_neighbours = bool(getattr(config, "neighbour_scene_enabled", False))
        fused = fuse_scene_geometry(
            primary_masks,
            frame.depth_map,
            frame.intrinsics,
            np.asarray(camera_to_base.to_matrix(), dtype=np.float64),
            views,
            metric=AssociationMetric(str(config.metric)),
            min_score=float(config.min_score),
            neighbour_mm=float(config.neighbour_mm),
            max_centroid_mm=float(config.max_centroid_mm),
            with_neighbours=with_neighbours,
            neighbour_voxel_mm=float(getattr(config, "neighbour_voxel_mm", 8.0)),
            # Scoring-only decimation. `getattr` keeps a config object without the field on the
            # untouched path, and 0.0 is off, so this line changes nothing anywhere it is not set.
            score_voxel_mm=float(getattr(config, "score_voxel_mm", 0.0)),
        )
        self._stamp_fusion(fused, fusion_t0_ns, missing=missing, grounded_nothing=grounded_nothing,
                           with_neighbours=with_neighbours)
        return fused

    def _fixed_cameras_placeable(self, camera_to_base: "Transform | None") -> bool:
        """Whether an enabled `grasping.fusion.geometry` has what it needs to place the other cameras' views.

        Enabled but unbuildable is a deployment error, not a runtime hiccup, so it is said plainly every time rather
        than folded into the missing-camera policy of :meth:`_fixed_camera_views`.
        """
        if self.multi_camera_perception is not None and camera_to_base is not None:
            return True
        _LOG.warning(
            "grasping.fusion.geometry is enabled but %s; the cell is running single-view",
            "no multi-camera perception source is wired"
            if self.multi_camera_perception is None
            else "no CAMERA->BASE transform is available",
        )
        return False

    def _stamp_fusion(
        self, fused: "FusedSceneGeometry", fusion_t0_ns: int, *, missing: list[str], grounded_nothing: list[str],
        with_neighbours: bool,
    ) -> None:
        """Record the fusion span and stamp what the fusion did, as :meth:`_fused_scene` has always stamped it."""
        fusion_ms = (time.monotonic_ns() - fusion_t0_ns) / 1_000_000.0
        # `fusion_latency_ms`, recorded only where another camera's view entered the fusion
        # (`views_used`, the record's `fused_view_count`), so a pick that stood down to one view
        # reports no fusion time rather than a zero, and the SLO gate skips it. A later attempt of
        # the same pick that fuses replaces the span, as every stage does.
        if fused.views_used and self.latency_tracker is not None:
            self.latency_tracker.record(LatencyStage.FUSION, fusion_ms)
        # Stamp what ran, not what was configured: the same rule the geometry stage follows.
        self._fusion_geometry_telemetry = {
            "fused_views_used": len(fused.views_used),
            "fused_views": list(fused.views_used),
            "fused_objects": fused.objects_fused,
            "fused_objects_total": len(fused.clouds_base_mm),
            # Delivered a frame, grounded nothing on it. Not a fault, and not the same as absent.
            "cameras_grounded_nothing": len(grounded_nothing),
            # Configured, delivered no frame at all. What `on_camera_unavailable` acts on.
            "cameras_absent": len(missing),
            # How well the cameras agreed, not just how many answered: the mean association score
            # over the pairs that won their assignment. The telemetry catalog declares
            # `fusion_evidence_quality` as a unit interval without saying what it means; this is the
            # association's own confidence, the only quantity in the fusion that is one. Bounded
            # [0, 1] for every metric (overlap is a fraction, centroid is clipped, box-IoU is a
            # ratio), so the catalog's validator holds whichever is configured.
            "fused_evidence_quality": _mean_winning_association_score(fused),
        }
        if with_neighbours:
            self._fusion_geometry_telemetry["fused_neighbour_points"] = fused.neighbour_points

    def _fixed_camera_views(
        self, config: "FusionGeometryConfig",
    ) -> "tuple[list[ObservedView], list[str], list[str], int]":
        """Every other camera's segmented view placed in BASE, with the cameras absent and those that grounded nothing.

        Acquires every other camera once (``multi_camera_perception.acquire_all()``) and applies the missing-camera
        policy (``on_camera_unavailable``: refuse raises, anything else warns). A camera with no resolver, or none that
        placed this frame, is dropped with a warning; one that grounded nothing is counted, not dropped silently. The
        last element is ``time.monotonic_ns()`` read as the frames came in, where the fusion clock starts.
        """
        from src.robot.grasping.multiview.scene_geometry import ObservedView

        assert self.multi_camera_perception is not None  # narrowed by _fixed_cameras_placeable
        resolvers = self.camera_frame_resolvers or {}
        observations = tuple(self.multi_camera_perception.acquire_all())
        fusion_t0_ns = time.monotonic_ns()
        delivered = {observation.camera_id for observation in observations}
        # The configured list first, the resolver map only as a fallback for a caller that wired
        # this by hand. An expectation derived from the resolvers that built cannot notice a camera
        # whose resolver never built, so that camera is one nobody waits for.
        expected = set(self.configured_camera_ids) or set(resolvers)
        missing = sorted(expected - delivered)
        # Why each absent camera was absent, when the rig recorded it. Absence is the protocol's only
        # error channel, so without this the policy below can say that a camera did not deliver and
        # not whether the device is unplugged, the pipeline never started, or a driver raised.
        why_absent = dict(getattr(self.multi_camera_perception, "last_failures", None) or {})
        if missing:
            reasons = "; ".join(f"{cam}: {why_absent[cam]}" for cam in missing if cam in why_absent)
            if str(config.on_camera_unavailable) == "refuse":
                raise RuntimeError(
                    f"grasping.fusion.geometry: configured camera(s) {missing} delivered no frame "
                    "and on_camera_unavailable='refuse'"
                    + (f" ({reasons})" if reasons else "")
                )
            _LOG.warning(
                "grasping.fusion.geometry: camera(s) %s delivered no frame%s; continuing with %s",
                missing,
                f" ({reasons})" if reasons else "",
                sorted(delivered & expected) or "no other camera (single-view)",
            )

        views: list[ObservedView] = []
        #: Cameras that answered with a frame and found nothing on it. Counted so the difference
        #: between "no camera looked" and "a camera looked and the bin was empty there" survives
        #: into the telemetry.
        grounded_nothing: list[str] = []
        for observation in observations:
            resolver = resolvers.get(observation.camera_id)
            if resolver is None:
                # An id with no calibration entry cannot be placed in BASE. Guessing a default
                # extrinsic would fuse a real surface into the wrong place, which is worse than
                # not fusing it at all.
                _LOG.warning(
                    "grasping.fusion.geometry: camera %r has no CAMERA->BASE resolver, so its view "
                    "is dropped: a fused camera needs grasping.fusion.enabled, an enabled entry in "
                    "fusion.cameras and its calibration declared on its rig, "
                    "camera.cameras.rigs[<id>].extrinsics",
                    observation.camera_id,
                )
                continue
            other_to_base = resolver.camera_to_base_for_frame(observation.frame, arm=self.arm)
            if other_to_base is None:
                _LOG.warning(
                    "grasping.fusion.geometry: camera %r resolved no CAMERA->BASE this frame; its "
                    "view is dropped",
                    observation.camera_id,
                )
                continue
            # The segmentation, not just its mask. A mask is enough to fuse a surface onto an
            # object the primary already found, and not enough to grasp from: the calculator takes a
            # segmentation, this camera's depth and this camera's intrinsics, and a bare mask is one
            # of the three. Carrying the pair keeps them in step, so an object promoted from this
            # camera cannot end up with one camera's mask and another's depth.
            segmented = tuple(
                (seg, mask)
                for seg, mask in ((s, getattr(s, "mask", None))
                                  for s in observation.frame.segmentations)
                if mask is not None
            )
            masks = tuple(mask for _seg, mask in segmented)
            if not masks:
                # A camera that delivered a frame and grounded nothing on it. That is a legitimate
                # answer, not a fault: a camera looking at an empty stretch of bin has done its job,
                # so it counts as delivered and the missing-camera policy above stays out of it.
                # It used to be a bare `continue`. The option's own description promises that a cell
                # must never fall back to single-view silently, and this was the way it did.
                grounded_nothing.append(observation.camera_id)
                _LOG.info(
                    "grasping.fusion.geometry: camera %r delivered a frame and grounded nothing on "
                    "it; it contributes no surface this pick",
                    observation.camera_id,
                )
                continue
            views.append(
                ObservedView(
                    name=observation.camera_id,
                    masks=masks,
                    segmentations=tuple(seg for seg, _mask in segmented),
                    depth_map=observation.frame.depth_map,
                    intrinsics=observation.frame.intrinsics,
                    camera_to_base=np.asarray(other_to_base.to_matrix(), dtype=np.float64),
                )
            )

        return views, missing, grounded_nothing, fusion_t0_ns

    def _fixed_views_of_the_pick(
        self, camera_to_base: "Transform | None",
    ) -> "tuple[tuple[ObservedView, ...], list[str], list[str]]":
        """The fixed cameras' views for every look of a wrist pick: acquired once, at its first look, and kept.

        A fixed camera sees the same cell from every look of the wrist, so acquiring it again per look would be a
        second detect and segment pass per camera for the same answer. Only with ``grasping.fusion.geometry`` enabled,
        under its missing-camera policy. A rig that lists the wrist camera itself hands back a copy of the first look,
        placed at the first look's pose for every later one: it is left out, because the looks already fuse that
        camera, each at the pose it was taken from.
        """
        if self._fixed_views is None:
            config = self.fusion_geometry_config
            views: list[ObservedView] = []
            missing: list[str] = []
            grounded_nothing: list[str] = []
            if config is not None and bool(config.enabled) and self._fixed_cameras_placeable(camera_to_base):
                views, missing, grounded_nothing, _placed_ns = self._fixed_camera_views(config)
                views = [view for view in views if not (self.primary_camera_id and view.name == self.primary_camera_id)]
            self._fixed_views = (tuple(views), missing, grounded_nothing)
        return self._fixed_views

    def _fused_looks(
        self, frame: PerceptionFrame, camera_to_base: "Transform | None",
    ) -> "FusedSceneGeometry | None":
        """The look being ranked, fused with every earlier look of the pick and with the fixed cameras.

        The looks' surfaces are the ones :meth:`_look_view` kept: each mask less the pixels behind a depth step and
        the grazing ones, on the measured surface. Two looks of one wrist camera disagree by more than two fixed cameras
        do, the more the wrist turned between them, so while an earlier look takes part the association runs at the
        wrist tolerances (``association.WRIST_VIEW_*``) unless the fusion config asks for wider ones. The neighbours
        every look saw are returned beside each object's cloud (``with_neighbours``), the seam the candidate filter
        reads its obstacles from, so a part only an earlier look saw stays an obstacle to this look's grasps. With no
        earlier look the fixed cameras are fused as :meth:`_fused_scene` fuses them.

        ``None`` when there is nothing to fuse with, or this look placed nothing; the counters are stamped as the fixed
        cameras' fusion stamps them, and only where that fusion is enabled, so a wrist cell without it keeps the
        telemetry it had.
        """
        from src.robot.grasping.multiview.association import (
            DEFAULT_NEIGHBOUR_MM,
            WRIST_VIEW_MIN_SCORE,
            WRIST_VIEW_NEIGHBOUR_MM,
            WRIST_VIEW_SCORE_VOXEL_MM,
            AssociationMetric,
        )
        from src.robot.grasping.multiview.scene_geometry import fuse_scene_geometry

        look = self._current_look
        assert look is not None  # the caller's branch
        self._last_fused_scene = None
        config = self.fusion_geometry_config
        fixed_on = config is not None and bool(config.enabled)
        fixed, missing, grounded_nothing = self._fixed_views_of_the_pick(camera_to_base)
        earlier = [seen.view for seen in self._look_views if seen.view is not None]
        views = [*earlier, *fixed]
        if look.camera_to_base is None or look.view is None or look.depth is None or not views:
            if fixed_on:
                self._fusion_geometry_telemetry = {
                    "fused_views_used": 0,
                    "fused_objects": 0,
                    "cameras_grounded_nothing": len(grounded_nothing),
                    "cameras_absent": len(missing),
                }
            return None
        self._last_other_views = tuple(fixed)
        # One mask per segmentation, not compacted, as `_fused_scene` passes them: the clouds come back by
        # segmentation index.
        empty = np.zeros(look.depth.shape[:2], dtype=bool)
        primary_masks = tuple(empty if surface is None else surface for surface in look.surfaces)
        if earlier:
            metric = AssociationMetric(str(config.metric)) if config is not None else AssociationMetric.OVERLAP
            min_score = float(config.min_score) if config is not None else WRIST_VIEW_MIN_SCORE
            neighbour_mm = max(float(config.neighbour_mm) if config is not None else DEFAULT_NEIGHBOUR_MM,
                               WRIST_VIEW_NEIGHBOUR_MM)
            score_voxel_mm = max(float(getattr(config, "score_voxel_mm", 0.0)) if config is not None else 0.0,
                                 WRIST_VIEW_SCORE_VOXEL_MM)
        else:
            assert config is not None  # the fixed views exist only where their fusion is enabled
            metric = AssociationMetric(str(config.metric))
            min_score = float(config.min_score)
            neighbour_mm = float(config.neighbour_mm)
            score_voxel_mm = float(getattr(config, "score_voxel_mm", 0.0))
        with_neighbours = bool(earlier) or bool(getattr(config, "neighbour_scene_enabled", False))
        fusion_t0_ns = time.monotonic_ns()
        fused = fuse_scene_geometry(
            primary_masks,
            look.depth,
            np.asarray(frame.intrinsics, dtype=np.float64),
            look.camera_to_base,
            views,
            metric=metric,
            min_score=min_score,
            neighbour_mm=neighbour_mm,
            max_centroid_mm=float(config.max_centroid_mm) if config is not None else 150.0,
            with_neighbours=with_neighbours,
            neighbour_voxel_mm=float(getattr(config, "neighbour_voxel_mm", 8.0)) if config is not None else 8.0,
            score_voxel_mm=score_voxel_mm,
        )
        self._stamp_fusion(fused, fusion_t0_ns, missing=missing, grounded_nothing=grounded_nothing,
                           with_neighbours=with_neighbours)
        self._last_fused_scene = fused
        return fused

    def calculator_for(self, camera_id: str):  # noqa: ANN201
        """The calculator bound to ``camera_id``'s intrinsics, or the cell's own.

        The fallback is not a convenience: a cell with one camera has one calculator and no map, and
        every object it ever sees is the primary's, so the lookup must answer without a map existing.
        A miss with a map present is also the cell's own calculator rather than a refusal, because
        that combination means promotion produced an object from a camera nobody built a calculator
        for, and refusing the pick over it would be a larger failure than computing it in the primary
        lens and saying so.
        """
        table = self.camera_calculators
        if not table:
            return self.calculator
        found = table.get(camera_id)
        if found is None:
            _LOG.warning(
                "no calculator built for camera %r; computing its object in the primary camera's "
                "lens, which places the grasp with the wrong intrinsics", camera_id)
            return self.calculator
        return found

    def _scene_objects(self, frame, camera_to_base, fused_scene):  # noqa: ANN001, ANN202
        """The objects this pick may attempt, and the camera each one is grasped from.

        Default-off means the same list, not a similar one. With `promote_unmatched` false this
        returns exactly one entry per segmentation of the primary frame, in the primary's own order,
        carrying the primary's depth and intrinsics. Position for position it is the list the loop
        iterated before this method existed, so nothing measured against that loop moves.

        With it on, an object no primary segmentation corresponds to survives instead of being
        dropped. Detection and segmentation already ran on that camera and cost a full pass; what was
        missing was anywhere for the result to go.
        """
        from src.robot.grasping.multiview.scene_geometry import (  # noqa: PLC0415
            ObservedView,
            build_scene_objects,
        )

        segmented = tuple(
            (seg, mask)
            for seg, mask in ((s, getattr(s, "mask", None)) for s in frame.segmentations)
            if mask is not None
        )
        primary = ObservedView(
            name=self.primary_camera_id,
            masks=tuple(mask for _s, mask in segmented),
            depth_map=frame.depth_map,
            intrinsics=frame.intrinsics,
            camera_to_base=(np.asarray(camera_to_base.to_matrix(), dtype=np.float64)
                            if camera_to_base is not None else np.eye(4)),
            segmentations=tuple(seg for seg, _m in segmented),
        )
        clouds = tuple(
            fused_scene.cloud_for(index) for index in range(len(primary.masks))
        ) if fused_scene is not None else None

        config = self.fusion_geometry_config
        if config is None or not bool(getattr(config, "promote_unmatched", False)):
            return build_scene_objects([primary], (), fused_clouds=clouds)

        from src.robot.grasping.multiview.association import (  # noqa: PLC0415
            AssociationMetric,
            ViewCandidates,
            cluster_views,
        )
        from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

        views = [primary, *self._last_other_views]
        candidates = [
            ViewCandidates(
                name=view.name,
                clouds_base_mm=tuple(
                    to_base_mm(mask, view.depth_map, view.intrinsics, view.camera_to_base)
                    for mask in view.masks
                ),
            )
            for view in views
        ]
        clusters = cluster_views(
            candidates,
            metric=AssociationMetric(str(config.metric)),
            min_score=float(config.min_score),
            neighbour_mm=float(config.neighbour_mm),
            max_centroid_mm=float(config.max_centroid_mm),
            score_voxel_mm=float(getattr(config, "score_voxel_mm", 0.0)),
        )
        objects = build_scene_objects(
            views, clusters, fused_clouds=clouds,
            # The same clouds the clustering scored, so a promoted object seen by two cameras
            # is grasped from both. Reused rather than rebuilt: back-projecting them a second
            # time would be the same arithmetic on the same frames for the same answer.
            view_clouds=[view.clouds_base_mm for view in candidates],
        )
        promoted = sum(1 for obj in objects if obj.promoted)
        if promoted:
            # Said once per pick, because an object appearing that the primary camera cannot see is
            # the whole point of the feature and also the thing an operator will want to check first
            # when a cell reaches for something they did not expect it to know about.
            _LOG.info(
                "grasping.fusion.geometry: %d object(s) promoted from a camera other than %r; "
                "%d object(s) in the scene",
                promoted, self.primary_camera_id, len(objects),
            )
        self._fusion_geometry_telemetry["objects_promoted"] = promoted
        self._fusion_geometry_telemetry["objects_in_scene"] = len(objects)
        return objects

    def _offer_masks_to_planner_world(
        self, frame: "PerceptionFrame", target_index: "int | None", *,
        target_points_base_mm: "np.ndarray | None" = None,
    ) -> None:
        """Tell the planner world which object this attempt is reaching for, and hold it out for the attempt.

        Two things come out of this and only one is optional. The names let a refusal say
        which object refused instead of printing a number. The target has to be left out of
        the world entirely: it is the one obstacle the arm is deliberately driving into, and
        a goal inside an obstacle has no plan at all.

        Duck-typed through the arm, because a cell without a live planner world has nothing
        to offer and this loop has no business knowing what one is. An arm that answers
        `None` costs one attribute lookup per attempt.

        The offer (``segmentation_offer_from_frame``) carries the frame's masks under the
        perception source's camera name and the target's BASE points, placed by the CAMERA to BASE
        this frame was ranked with, never a second resolve. It is held through ``keeping_out`` for
        every motion of the attempt, closed when the pick ends or the next frame is offered. A
        source that names no camera offers the target's box alone; one that names no camera and has
        no transform offers nothing, and the target stays an obstacle, which this says once per run.
        ``target_points_base_mm`` is the target's cloud fused over the views of a wrist pick, offered
        in place of this frame's surface of it, so the box leaves the part out of every frame the
        world holds for the pick, not only out of this one.
        """
        world = getattr(self.arm, "live_planner_world", None)
        if world is None:
            return
        transform = self._pending_camera_to_base
        offer = segmentation_offer_from_frame(
            frame, target_index, camera=self._perception_camera_name(),
            camera_to_base_mm=None if transform is None else np.asarray(transform.to_matrix(), dtype=np.float64),
            target_points_base_mm=target_points_base_mm,
        )
        swept = self._swept_by_a_push(offer)
        if offer is not None and swept is not None:
            # The target is the part a push of this run moved: the frames the world holds from before the push still
            # show it where it stood, and a finger has to go there now, so that space and its path stay out as well.
            own = offer.target_points_base_mm
            offer = replace(offer, target_points_base_mm=swept if own is None or not len(own)
                            else np.vstack([own, swept]))
        self._keep_out.close()
        if offer is None:
            if not self._keep_out_unplaced_said:
                _LOG.warning(
                    "the target stays an obstacle in the planner world: the perception source names no camera the "
                    "masks belong to and the frame has no CAMERA to BASE to place the target's box"
                )
                self._keep_out_unplaced_said = True
            return
        self._keep_out.enter_context(keeping_out(self.arm, offer))

    def _perception_camera_name(self) -> "Maybe[str]":
        """The camera the perception source's frames come from, or unset when it names none.

        A camera's rig id where the source streams from a rig (``streamer.rig_id``), else the name a
        source states (``camera_name``). A stated name that is not this loop's ``primary_camera_id``
        is a misbuilt cell: the masks would be offered under a camera whose image they are not from.
        """
        rig_id = getattr(getattr(self.perception, "streamer", None), "rig_id", None)
        named = getattr(self.perception, "camera_name", UNSET)
        name: Maybe[str] = UNSET
        if isinstance(rig_id, str) and rig_id.strip():
            name = rig_id
        elif chosen(named) and isinstance(named, str) and named.strip():
            name = named
        if chosen(name) and self.primary_camera_id and name != self.primary_camera_id:
            raise ValueError(
                f"the perception source names camera {name!r} and this loop's primary_camera_id is "
                f"{self.primary_camera_id!r}: its masks would be offered under a camera whose image they are not from"
            )
        return name

    def _best_result_over_segmentations(
        self, frame: PerceptionFrame
    ) -> "tuple[GraspResult | None, int | None, OrderingDecision | None]":
        """Thin wrapper: records the ranking latency span, then delegates.

        A wrapper rather than a span inside the body because this is the one call every pick path
        makes, open-loop and decision alike, so this single site covers both.
        """
        tracker = self.latency_tracker
        if tracker is None:
            return self._best_result_over_segmentations_inner(frame)
        with tracker.span(LatencyStage.RANKING):
            return self._best_result_over_segmentations_inner(frame)

    def _best_result_over_segmentations_inner(
        self,
        frame: PerceptionFrame,
    ) -> tuple[GraspResult | None, int | None, OrderingDecision | None]:
        """Compute a grasp for each segmentation; return the best success.

        Falls back to the last non-success result (with its reasons)
        when no segmentation yields candidates, so the caller can route
        on those reasons.

        When :attr:`frame_resolver` is wired, the resolved
        ``T_cam_to_base`` for the captured frame is forwarded to the
        calculator so the candidates come back in base frame. The
        resolver is invoked once per ``run`` iteration, not once per
        segmentation, because all segmentations in a single
        :class:`PerceptionFrame` were captured at the same TCP pose.

        When :attr:`target_ordering` is set and enabled the
        per-segmentation winners are passed through
        :func:`select_target` and the chosen candidate plus the
        :class:`OrderingDecision` are returned. Otherwise the legacy
        ``argmax(top_score)`` path runs verbatim and the returned
        decision is ``None``.
        """
        # What this frame fuses, said by this frame alone: emptied before anything below can raise or
        # stand down, and filled where a fused cloud is handed to the generator.
        self._fused_views_by_object = {}
        self._fused_objects_in_frame = 0
        self._results_by_object = {}
        camera_to_base: Transform | None = None
        if self.frame_resolver is not None:
            camera_to_base = self.frame_resolver.camera_to_base_for_frame(
                frame, arm=self.arm
            )
        # Stash this frame's CAMERA->BASE so _execute can map the camera-frame neighbour cloud
        # (on result.metadata) into the candidate (BASE) frame before the swept-volume validation.
        self._pending_camera_to_base = camera_to_base
        # Transform the fused multi-view scene cloud (BASE) into this frame's CAMERA frame once
        # (the same camera_to_base for every segmentation of the frame). None when not supplied.
        external_scene_cam = self._external_scene_points_camera_frame(camera_to_base)
        # Observe the rest of the rig once per frame and fuse every candidate object's surface.
        # ``None`` (default, or any stand-down) leaves the single-view path below untouched.
        fused_scene = self._fused_scene(frame, camera_to_base)
        # What the parts of this frame stand on, as the arm's live world finds it on the same frames (contract 7): read
        # once per frame, by every object's support and calculator. ``None`` plans as before.
        support_model = self._support_model_of_frame(frame, camera_to_base)
        self._last_support_model = support_model
        # The declared container walls, in this frame's CAMERA frame. Frame-level like the cloud
        # above (the bin does not move between segmentations), and additive below rather than an
        # alternative: walls and neighbours are different obstacles and a cell that has both
        # declared must get both.
        wall_points_base = self._container_wall_points_base_mm()
        wall_cam = (
            self._base_points_to_camera(wall_points_base, camera_to_base)
            if wall_points_base is not None and camera_to_base is not None
            else None
        )
        if wall_points_base is not None and camera_to_base is None:
            _LOG.warning(
                "container walls are configured but no CAMERA->BASE transform is available; the "
                "wall-collision check is not running for this attempt")
        # The objects this pick may attempt. With `promote_unmatched` off this is one entry per
        # segmentation of the primary frame, in the primary's order, so the loop below runs on
        # exactly the list it ran on before scene objects existed.
        scene = self._scene_objects(frame, camera_to_base, fused_scene)
        self._last_scene_objects = tuple(scene)
        if fused_scene is not None:
            # Objects whose cloud really holds two cameras, which is not `fused_scene.objects_fused`: that
            # counts the primary's objects alone and would count a camera fused with its own copy.
            self._fused_objects_in_frame = sum(
                1 for index, obj in enumerate(scene)
                if _views_behind(obj, index, fused_scene, primary_name=self._look_name_ranked()))
        # Neighbour masks stay the primary frame's, deliberately. They are the clutter around a
        # target in the frame the grasp is synthesised in, and a mask from another camera is in
        # another camera's pixels: handing it to the dense sampler would place clutter by index into
        # an image it does not belong to. A promoted object's neighbours are a separate question and
        # this change does not answer it.
        other_masks = [getattr(seg, "mask", None) for seg in frame.segmentations]
        best_success: GraspResult | None = None
        best_index: int | None = None
        last_failure: GraspResult | None = None
        last_failure_index: int | None = None
        # Collect successful per-segmentation results so the target
        # selector can rank them after all segmentations have been
        # scored. Both arms of the conditional below allocate the
        # list; what ``ordering_active`` gates is the append, so with
        # ordering off the list stays empty and the selector below
        # short-circuits.
        ordering_active = (
            self.target_ordering is not None and self.target_ordering.enabled
        )
        successes: list[tuple[int, GraspResult]] = [] if ordering_active else []  # type: ignore[assignment]
        # How many segmentations the hard label gate below let through. Zero of them, with a label
        # set, is a nameable failure rather than an empty reason tuple.
        label_matches = 0
        # The parts next_target skips (the campaign's exclusion zones), by segmentation index, and of them those no region
        # of the task holds: a part skipped after a failed pick, which still stands there to be picked.
        zones = self.exclusion_zones
        excluded: list[int] = []
        zoned: list[int] = []
        for idx, obj in enumerate(scene):
            seg = obj.segmentation
            if seg is None:
                # A view that carried masks and no segmentations. It can contribute surface to
                # another object and it cannot be grasped from, and saying so beats failing three
                # calls later inside the calculator with a `None` nobody expected.
                continue
            # When a hard label-target is set, only the matching segmentation is an executable
            # target. The non-target segs are skipped here but remain in ``other_masks`` above, so they
            # still feed ``neighbours`` of the chosen target (the dense sampler + corridor-risk +
            # approach-validation see the clutter).
            if self.target_label is not None and (
                (getattr(seg, "label", "") or getattr(seg, "prim_path", "")) != self.target_label
            ):
                continue
            label_matches += 1
            # Next to the label gate: a part of the same label that next_target skips is no target either.
            held = self._exclusion_of(zones, idx, obj, frame) if zones is not None else ""
            if held:
                excluded.append(idx)
                if held != "region":
                    zoned.append(idx)
                continue
            # A promoted object is not in `other_masks` at all, so `j != idx` would exclude an
            # unrelated primary object instead of itself. Excluding nothing is right for it: every
            # primary mask genuinely is clutter around it.
            neighbours = [
                m for j, m in enumerate(other_masks)
                if m is not None and not (not obj.promoted and j == idx)
            ]
            extra_kwargs: dict = {}
            if camera_to_base is not None:
                extra_kwargs["camera_to_base"] = camera_to_base
            # Forward the frame rgb only when the calculator opted into rendering the grasp-point
            # overlay through its ``render_debug_images`` switch. Default off means
            # ``rgb_image=None``, no render, byte-identical and zero cost. ``getattr`` keeps a
            # calculator that does not declare the switch on the off path.
            if getattr(self.calculator, "render_debug_images", False):
                extra_kwargs["rgb_image"] = getattr(frame, "rgb", None)
            # Feed the fused multi-view scene cloud (camera frame) so the collision filter
            # sees neighbour geometry this single synthesis view cannot. The calculator vstacks it
            # with the same-view ``other_object_masks`` neighbours. None keeps the key absent and
            # the path byte-identical.
            if external_scene_cam is not None:
                extra_kwargs["scene_points_mm"] = external_scene_cam
            # The multi-camera generalisation of the line above: after target fusion, 60.0 % of
            # every remaining mis-rank is a finger collision, and the neighbour that the fingers
            # hit is one the synthesis view never saw. This hands this object the rest of the rig's
            # view of everything that is not it. Same precedence rule as the target cloud: an
            # explicit external cloud still wins, so enabling fusion cannot change a runner that
            # supplies one.
            elif fused_scene is not None and camera_to_base is not None:
                neighbour_base = (None if obj.promoted else fused_scene.neighbour_for(idx))
                if neighbour_base is not None and neighbour_base.size:
                    neighbour_cam = self._base_points_to_camera(neighbour_base, camera_to_base)
                    if neighbour_cam is not None:
                        extra_kwargs["scene_points_mm"] = neighbour_cam
            # Walls travel on their own kwarg, alongside whichever cloud won above rather than
            # instead of it: they are a different obstacle from a neighbouring part and answer a
            # different share of the ranking gap, so a cell that declared both must get both.
            # ``rigid_obstacle_points_mm`` rather than ``scene_points_mm`` because the generator
            # dilates observed points by 12 mm to cover the gaps between depth samples, and a
            # declared wall has no gaps: dilating it forbids a band of the bin a gripper can reach.
            if wall_cam is not None:
                extra_kwargs["rigid_obstacle_points_mm"] = wall_cam
            # Forward the fused multi-view target cloud (BASE) so the generator searches side-face
            # contacts. Only for the target segmentation (target_label set), so the target's cloud
            # is never applied to a neighbour seg. None keeps the key absent and byte-identical.
            if self.external_target_geometry_base_mm is not None and self.target_label is not None:
                extra_kwargs["geometry_points_base_mm"] = self.external_target_geometry_base_mm
            # The multi-camera generalisation of the line above: this object's own fused cloud,
            # matched across the rig by surface overlap. Indexed by segmentation, so every candidate
            # object gets its own, not just a labelled target. An explicit external cloud (the sim
            # runner's ground-truth path) still wins, so enabling fusion cannot change a runner that
            # already supplies one.
            elif fused_scene is not None:
                object_cloud = obj.fused_cloud_base_mm
                if object_cloud is not None and object_cloud.size:
                    extra_kwargs["geometry_points_base_mm"] = object_cloud
                    # Named here and nowhere else: this is the line the fused cloud reaches the
                    # generator on, so an object is said to be planned on cameras only when it was.
                    self._fused_views_by_object[idx] = _views_behind(
                        obj, idx, fused_scene, primary_name=self._look_name_ranked())
            if self.gripper_model is not None:
                extra_kwargs["gripper_model"] = self.gripper_model
            if self.corridor_config is not None:
                extra_kwargs["corridor_config"] = self.corridor_config
            if self.feasibility_config is not None:
                extra_kwargs["feasibility_config"] = self.feasibility_config
                extra_kwargs["feasibility_weight"] = self.feasibility_weight
            # The plane is BASE-framed, and the calculator's collision filter runs on CAMERA-frame
            # poses, so it needs the transform to place the table at all: without one it refuses
            # rather than silently skipping the check. The calculator is never handed an
            # unanswerable question: a cell with no CAMERA->BASE transform cannot drive to a
            # base-frame grasp either, so the plane is withheld and the warning says so instead of
            # the check being dropped quietly (a silently-skipped table check is the exact defect
            # this whole seam exists to close).
            support_cfg = self.support_config
            # A look of a wrist pick refines the support from the target's cloud fused over its looks, when there is one.
            looks_cloud = (
                obj.fused_cloud_base_mm
                if self._current_look is not None and not obj.promoted and obj.fused_cloud_base_mm is not None
                and obj.fused_cloud_base_mm.size
                else None
            )
            support = self._resolve_support(seg, frame, camera_to_base, looks_cloud, support_model)
            if support_model is not None:
                # The model rides beside the plane (contract 2): the calculator reads its surfaces for what is support and
                # what an obstacle, for SFE's support under the part, and for the solids the open hand keeps off.
                extra_kwargs["support_model"] = support_model
            if support is not None and support_cfg is not None and camera_to_base is not None:
                extra_kwargs["support_plane"] = support.plane
                extra_kwargs["min_table_clearance_mm"] = support_cfg.min_clearance_mm
                # Where the table ended up, and whether it was declared or observed.
                # `SupportResolution.as_telemetry` was written for this and had no caller anywhere in
                # the tree; the only plane number reaching a log was a camera depth from the
                # collision filter, which is not a height. That cost nothing while single-view
                # refinement was structurally inert: the target cloud's extent along the support
                # normal was exactly 0.0 against a 25 mm gate, because the producer flattened the
                # depth under every mask, so the plane was always the declared one. It refines now.
                # `resolve_support_plane` only ever raises the plane and every millimetre it rises
                # eats into a 5 mm clearance budget, so a plane walking up onto the object has to be
                # visible while it happens.
                self._support_telemetry = support.as_telemetry()
                if support.source == "observed":
                    _LOG.info(
                        "support plane observed at %.1f mm, %.1f mm above the declared %.1f mm, "
                        "from this object's own cloud",
                        support.height_mm, support.height_mm - support.declared_mm,
                        support.declared_mm)
            elif support is not None:
                _LOG.warning(
                    "support plane configured but no CAMERA->BASE transform is available; "
                    "the table-clearance check is not running for this attempt")
            # The calculator for the camera this object was seen by, and its depth. A calculator
            # is bound to one camera's intrinsics at construction, so computing a promoted object
            # with the primary's calculator would synthesise a grasp using the wrong focal length
            # and the wrong principal point on the right pixels: plausible numbers, wrong place.
            # `calculator_for` returns the cell's own calculator for every object the primary saw,
            # which is every object unless promotion is on, so this line is the line it replaced.
            calculator = self.calculator_for(obj.camera_id)
            with self._asking_more_along_the_axis(calculator) as cap:
                result = calculator.compute_result(
                    seg,
                    obj.depth_map,
                    pixel_to_mm=None,
                    dense_sampling=self.dense_sampling,
                    grasp_sampling_mode=self._resolved_sampling_mode,
                    other_object_masks=neighbours,
                    **extra_kwargs,
                )
            result = self._closing_along(result, obj.camera_id, cap)
            self._stamp_deep_ranker(result, extra_kwargs, fused_scene, idx)
            self._results_by_object[idx] = _RankedObject(
                result=result,
                support_height_mm=float(support.height_mm) if support is not None else None,
                cloud_base_mm=extra_kwargs.get("geometry_points_base_mm"),
                call=_Call(seg=seg, depth=obj.depth_map, neighbours=tuple(neighbours), kwargs=dict(extra_kwargs),
                           camera_id=str(obj.camera_id)),
            )
            if result.is_success:
                if ordering_active:
                    successes.append((idx, result))
                if best_success is None or result.top_score > best_success.top_score:
                    best_success = result
                    best_index = idx
            elif last_failure is None or not closing_axis_refusal(result) or closing_axis_refusal(last_failure):
                # A part the closing axis left no grasp is the frame's failure only where no other part failed for its
                # own reason, whichever order the camera lists them in: that failure is the one a push or a recovery can
                # act on (a boxed-in part's ALL_COLLIDED), and the attempt still names the axis (``withheld``).
                last_failure = result
                last_failure_index = idx
        if best_success is None:
            if last_failure is None and self.target_label is not None and label_matches == 0:
                # Every segmentation was skipped at the label gate, so no calculator ran and there
                # is no result to inherit reasons from. Without a named reason the caller gets
                # ``None`` and the attempt records ``reasons=()``, so an operator prompt that
                # matched nothing surfaces as a bare "no valid grasp", indistinguishable from a
                # scene the cell genuinely could not grasp.
                #
                # This names the failure and changes nothing else: the reason is absent from
                # ``_RESCAN_REASONS``, so the attempt's action is still "exhausted", as it is for
                # the empty tuple. ``labels_seen`` is the actionable half: what the operator asked
                # for versus what perception actually returned.
                labels_seen = tuple(
                    str(getattr(seg, "label", "") or getattr(seg, "prim_path", "") or "")
                    for seg in frame.segmentations
                )
                return (
                    GraspResult(
                        reasons=(GraspFailureReason.TARGET_LABEL_NOT_FOUND,),
                        telemetry={
                            "target_label": self.target_label,
                            "labels_seen": labels_seen,
                            "segmentations": len(frame.segmentations),
                        },
                    ),
                    None,
                    None,
                )
            if last_failure is None and excluded and len(excluded) == label_matches and zones is not None:
                # Every part the label gate let through is one next_target skips: stop, rather than try one of them
                # again or another object.
                return self._only_excluded(zones, excluded, by_regions=not zoned), None, None
            return last_failure, last_failure_index, None

        if not ordering_active or len(successes) <= 1:
            return best_success, best_index, None

        # ---- clutter-aware ordering ----
        candidates = _build_target_candidates(
            successes, frame.depth_map, frame.segmentations
        )
        assert self.target_ordering is not None  # narrow for the checker
        decision = select_target(
            candidates=candidates, config=self.target_ordering
        )
        # Map the selector's choice (index into ``candidates``) back
        # to the originating segmentation index + cached GraspResult.
        if decision.chosen_index is None:
            return best_success, best_index, decision
        chosen_seg_idx, chosen_result = successes[decision.chosen_index]
        return chosen_result, chosen_seg_idx, decision

    def _closing_along(self, result: GraspResult, camera_id: str, cap: "int | None" = None) -> GraspResult:
        """``result`` with only the grasps that close along the closing axis the policy names (``closing_axis``), each
        turned half a turn about its approach where that puts it the named way round: the owner's "choose, don't twist"
        (2026-09-30). Read on every object's result before anything reads its candidates, so the target choice, a look's
        judgement and its jaw faces, the reranks and the approach check see only these, and the grasp judged is the grasp
        executed. None left is ``NO_VALID_GRASP``, which ends the attempt, with the sentence that names the axis and the
        tolerance (``closing_axis.closing_axis_refusal``). Of more than ``cap``, the calculator's own cap, which
        :meth:`_asking_more_along_the_axis` raised for this ranking, the best ``cap`` are kept.

        Where the policy names no axis and the cell names its hand and camera's natural orientation
        (:attr:`natural_closing_axis`), every grasp is turned, of its two equivalent wrist turns, to the one whose tool +X
        lies nearer it (``closing_axis.result_closing_toward``), at the same point and for the same reason; none is left
        out. Not beside the simulator's twist (``align_closing_to_base_x``, read as the twist reads it), which chooses the
        way round itself, so that a turn would change only the side a leaning approach leans to: the twist executes what
        it executes without a natural orientation. ``result`` itself where neither is named, it has no candidate, or
        nothing changed. Where the result changed, the overlay of the calculator of ``camera_id``, which computed it, is
        drawn again over what it kept (:meth:`_overlay_the_kept`)."""
        wanted = getattr(self.policy, "closing_axis", None)
        if not result.is_success:
            return result
        if wanted is not None:
            kept = result_closing_along(result, closing_axis_of(wanted))
            if cap is not None and len(kept.candidates) > cap:
                kept = replace(kept, candidates=kept.candidates[:cap])
            refused = closing_axis_refusal(kept)
            if refused:
                _LOG.warning("%s", refused)
        elif self.natural_closing_axis is not None and not getattr(self.policy, "align_closing_to_base_x", False):
            kept = result_closing_toward(result, closing_axis_of(self.natural_closing_axis))
        else:
            return result
        if kept is not result:
            self._overlay_the_kept(self.calculator_for(camera_id), kept)
        return kept

    @contextmanager
    def _asking_more_along_the_axis(self, calculator: Any) -> Iterator["int | None"]:
        """The ranking of one object by ``calculator``, which is asked for ``CANDIDATES_THE_AXIS_CHOOSES_AMONG`` (36)
        where the policy names a closing axis, so the axis chooses among as many as ``Scene.grasps`` does (the owner,
        2026-09-30). Yields the calculator's own cap (``max_candidates``, 12 by default), which :meth:`_closing_along`
        keeps of those along the axis and which the calculator has back once it has ranked, raise or not. ``None``, and
        nothing raised, where no axis is named or the calculator has no cap to raise: it ranks as it always did."""
        cap = getattr(calculator, "max_candidates", None)
        if getattr(self.policy, "closing_axis", None) is None or isinstance(cap, bool) or not isinstance(cap, int):
            yield None
            return
        calculator.max_candidates = max(CANDIDATES_THE_AXIS_CHOOSES_AMONG, cap)
        try:
            yield cap
        finally:
            calculator.max_candidates = cap

    @staticmethod
    def _overlay_the_kept(calculator: Any, kept: GraspResult) -> None:
        """Draw ``calculator``'s grasp overlay again over the grasps the closing axis kept, best first, each the way round
        it closes, so the picture a campaign pins and example 12 writes shows the grasps the pick chooses among. Where
        the calculator drew none, nothing; where it cannot draw one again, its overlay is dropped rather than shown with
        a grasp the axis left out ranked first."""
        if getattr(calculator, "last_debug_image_png", None) is None:
            return
        redraw = getattr(calculator, "redraw_debug_image", None)
        if callable(redraw):
            redraw(kept.candidates)
        else:
            calculator.last_debug_image_png = None

    def _closing_axis_withheld(self, result: "GraspResult | None") -> str | None:
        """Why the closing axis the program named left grasps out, for an attempt that found none to go on with, or
        ``None``: ``result``'s own sentence, then that of every other object of the last ranking the axis left no
        grasp, so a frame that failed on another object, for that object's own reason, still says the part was refused
        for its axis (``PickAttempt.withheld``, which the report's line names)."""
        said = [closing_axis_refusal(result), *(closing_axis_refusal(ranked.result)
                                                for ranked in self._results_by_object.values())]
        return "; ".join(dict.fromkeys(sentence for sentence in said if sentence)) or None

    def _execute(self, result: GraspResult) -> PolicyReport:
        """Delegate motion and gripper choreography to the policy: the grasps of :meth:`_try_the_grasps`, its last
        try's report returned and what every try came to kept on ``_last_tried``. The looks to come back to and the
        part's keep-out box are the ones the attempt set (``_tries_looked``, ``_tries_box``), none outside an attempt."""
        tried = self._try_the_grasps(result, looked=self._tries_looked, target_box=self._tries_box)
        self._last_tried = tried
        return tried.report

    def _try_the_grasps(
        self, result: GraspResult, *, looked: "LookedAround | None" = None, target_box: "KeepOutBox | None" = None,
    ) -> _Tried:
        """Hand the policy the best grasp, and the next ones of the same look while each before it was refused before
        anything was sent with the hand kept out of the part's keep-out box (the owner's "try the next grasps").

        The grasps, best first, are ``result.best`` and then the rest of ``result.candidates`` in rank order, at most
        :data:`GRASPS_TRIED_PER_ATTEMPT`. Where approach validation is wired (``approach_path_policy`` set and this
        orchestrator's ``mode_label`` in ``_approach_path_modes``) only the candidates whose approach and retreat sweep
        is clear of the scene's neighbour cloud are tried, with the cell's own hand, and where none is the answer is
        ``APPROACH_PATH_BLOCKED`` with nothing moved. With both jaw contact faces asked for (``both_faces``, a jaw hand)
        only ``result.best`` is tried: its faces are the ones that were judged.

        A next grasp follows only where the try before failed on a motion a guard or the planner refused before it was
        sent (``generated_view.REFUSED_BEFORE_SENDING``, or the planner's own no-plan sentence) and every motion it did
        send kept the open hand's jaw region out of ``target_box``, the box the arm's live world keeps out for the part
        (:meth:`_held_target_box`); where the pick cannot tell, the hand counts as having entered it. Anything else ends
        the tries: a grasp executed, a close, a gripper fault, a carried lift the arm would refuse, a motion that was sent
        and failed, a camera-frame refusal. A camera world that cannot vouch for the cell raises, as on every motion.

        Before each next grasp a stop asked for, and the controller, end the tries; a wrist pick (``looked``) whose try
        sent a motion is driven back to where it stood before the first on a judged move (:meth:`_back_to_the_look`),
        and both are asked again right before the policy. The toggle is asked by the policy itself, and never switched.
        Each try is said at INFO, each refused one at WARNING with its status and the driver's sentence.
        """
        grasp = result.best
        if grasp is None:
            # Defensive: callers should only invoke _execute on successful
            # results, but guard rather than IndexError on the policy.
            return _Tried(report=PolicyReport(outcome=PolicyOutcome.EXECUTED))
        assert self.policy is not None  # ensured by __post_init__
        pool: tuple[GraspPoint, ...] = (grasp,) if self._grips_only_the_judged_grasp() else (
            grasp, *(candidate for candidate in result.candidates if candidate is not grasp))
        if self.approach_path_policy is not None and self.mode_label in self._approach_path_modes:
            cloud = self._obstacle_cloud_in_candidate_frame(result)
            pool = tuple(candidate for candidate in pool if self._approach_sweep_is_clear(candidate, cloud))
            if not pool:
                # Every ranked candidate's sweep is blocked: refuse rather than drive into an obstacle.
                return _Tried(report=PolicyReport(outcome=PolicyOutcome.APPROACH_PATH_BLOCKED))
        order = pool[:GRASPS_TRIED_PER_ATTEMPT]
        ranks = {id(candidate): rank for rank, candidate in enumerate(result.candidates)}
        start = self._joints_here() if looked is not None and len(order) > 1 else None
        tries: list[dict[str, Any]] = []
        refused_all = True
        sent_last = False
        report: PolicyReport | None = None
        for number, candidate in enumerate(order, start=1):
            rank = ranks.get(id(candidate), number - 1)
            if number > 1:
                assert report is not None  # the try before
                ended = self._before_the_next_grasp(looked, start, sent_last)
                if ended is not None:
                    # A move back that failed once it was sent may have moved the arm: that is no refusal before sending.
                    return replace(ended, report=report, tries=tuple(tries),
                                   refused=refused_all and ended.moved_back is None)
            _LOG.info("try %d of %d: %s (rank %d of the look's grasps)", number, len(order),
                      "the grasp the look ranked first" if number == 1 else "the next grasp of the same look", rank)
            report = self.policy.execute(candidate)
            sent, reached, refused = self._what_the_try_did(report, target_box)
            status = getattr(report.motion_status, "value", None)
            tries.append(dict(zip(_TRY_KEYS, (rank, report.outcome.value, status, report.motion_message or None,
                                              sent, reached))))
            refused_all = refused_all and refused and not reached
            sent_last = sent
            more = refused and not reached and number < len(order)
            if report.outcome is not PolicyOutcome.EXECUTED:
                then = ("; the next grasp of the same look follows" if more else
                        "; the hand stood in the part's keep-out box, so no other grasp follows" if refused and reached
                        else "")
                _LOG.warning("try %d of %d (rank %d) %s: %s: %s%s", number, len(order), rank, report.outcome.value,
                             status or "no status", report.motion_message or "no reason was given", then)
            if not more:
                break
        assert report is not None  # at least the best was tried
        return _Tried(report=report, tries=tuple(tries), refused=refused_all and report.outcome is not
                      PolicyOutcome.EXECUTED)

    def _before_the_next_grasp(
        self, looked: "LookedAround | None", start: "JointPositions | None", sent: bool,
    ) -> "_Tried | None":
        """What ends the tries before the next grasp is handed over, or ``None`` to hand it over: a stop asked for, a
        controller that cannot move, and for a wrist pick whose try sent a motion, the move back to where it stood before
        the first grasp, refused or failed; both asked again once the arm is back. The returned ``_Tried`` carries only
        why (its report and tries are the caller's)."""
        stand_in = PolicyReport(outcome=PolicyOutcome.MOTION_FAILED)
        if self.should_cancel is not None and self.should_cancel():
            return _Tried(report=stand_in, cancelled=True)
        controller = self._controller_cannot_move()
        if controller is not None:
            return _Tried(report=stand_in, controller=controller)
        if not sent or looked is None:
            return None
        if start is None:
            return _Tried(report=stand_in, not_back=(
                "the arm could not say where it stood before the first grasp, so it cannot be driven back there"))
        back, _generated = self._back_to_the_look(looked, start)
        if not back.done:
            if back.stopped is not None:
                return _Tried(report=stand_in, moved_back=back.stopped, controller=back.controller)
            return _Tried(report=stand_in, not_back=back.refused or "the move back was refused")
        if self.should_cancel is not None and self.should_cancel():
            return _Tried(report=stand_in, cancelled=True)
        controller = self._controller_cannot_move()
        if controller is not None:
            return _Tried(report=stand_in, controller=controller)
        return None

    def _what_the_try_did(self, report: PolicyReport, target_box: "KeepOutBox | None") -> tuple[bool, bool, bool]:
        """``(sent, reached_part, refused_before_sending)`` of one try.

        Refused before sending: it failed on a motion a guard or the planner refused before anything was sent. Reached
        the part: it closed, went to the part, or a motion it sent put the open hand's jaw region into ``target_box``;
        where the pick cannot tell (no box, a hand that is no parallel jaw, a pose not in BASE), a motion it sent counts
        as having reached it, and so does a motion that was sent and failed."""
        from src.robot.execution.generated_view import REFUSED_BEFORE_SENDING  # noqa: PLC0415

        waypoints = tuple(report.waypoints or ())
        outcome = report.outcome
        if outcome is PolicyOutcome.MOTION_FAILED:
            status = report.motion_status
            refused = ((status is MotionStatus.TIMEOUT and report.motion_message == NO_PLAN_FAIL_SAFE_MESSAGE)
                       or status in REFUSED_BEFORE_SENDING)
            if not refused:
                return True, True, False
            entered = any(self._jaws_in_the_box(pose, target_box) is not False for pose in waypoints)
            return bool(waypoints), entered, True
        if outcome in (PolicyOutcome.CAMERA_FRAME_REJECTED, PolicyOutcome.APPROACH_PATH_BLOCKED):
            return False, False, False
        if outcome is PolicyOutcome.GRIPPER_FAULT and not waypoints:
            return False, False, False
        return True, True, False

    def _jaws_in_the_box(self, pose: Any, box: "KeepOutBox | None") -> bool | None:
        """Whether the open hand's jaw region at ``pose`` (``KeepOutBox.from_jaw``, the space between its pads) meets
        ``box``; ``None`` where the pick cannot tell."""
        from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel  # noqa: PLC0415
        from src.robot.grasping.generation.scene_obstacles import box_distances_mm  # noqa: PLC0415

        jaw = self.gripper_model
        if box is None or not isinstance(jaw, ParallelJawGripperModel) or getattr(pose, "frame", None) is not Frame.BASE:
            return None
        try:
            region = KeepOutBox.from_jaw(
                aperture_mm=self._open_width_mm() or _UNKNOWN_APERTURE_MM, finger_width_mm=float(jaw.finger_width_mm),
                pad_ahead_mm=float(jaw.pad_ahead_mm),
                pad_behind_mm=max(0.0, float(jaw.pad_length_mm) - float(jaw.pad_ahead_mm)),
                tcp_to_base_mm=pose.to_matrix())
        except (AttributeError, TypeError, ValueError):
            return None
        a, b = region.matrix(), box.matrix()
        gap = box_distances_mm(a[:3, 3], a[:3, :3], np.asarray(region.half_extents_mm), b[:3, 3], b[:3, :3],
                               np.asarray(box.half_extents_mm))
        return bool(float(gap[0]) <= 0.0)

    def _open_width_mm(self) -> float | None:
        """How far the hand opens, millimetres, as the gripper says; ``None`` where it says nothing."""
        value = getattr(self.gripper, "max_width_mm", None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0:
            return float(value)
        return None

    def _held_target_box(
        self, frame: PerceptionFrame, target_index: int | None, fused_target: "np.ndarray | None",
    ) -> "KeepOutBox | None":
        """The box the arm's live world keeps out for the part this attempt goes for: the box ``target_keep_out_box``
        fits round the points the pick offers it (its fused cloud, else its mask in this frame, and a push's swept
        path where this part was pushed), by the world's own limits and tuning. ``None`` where the arm has no such
        world or the part has no point."""
        world = getattr(self.arm, "live_planner_world", None)
        limits, tuning = getattr(world, "limits", None), getattr(world, "tuning", None)
        if limits is None or tuning is None:
            return None
        transform = self._pending_camera_to_base
        offer = segmentation_offer_from_frame(
            frame, target_index, camera=UNSET,
            camera_to_base_mm=None if transform is None else np.asarray(transform.to_matrix(), dtype=np.float64),
            target_points_base_mm=fused_target,
        )
        points = offer.target_points_base_mm if offer is not None else None
        swept = self._swept_by_a_push(offer)
        if swept is not None:
            points = swept if points is None or not len(points) else np.vstack([points, swept])
        if points is None or not len(points):
            return None
        from src.robot.safety.planning.perceived import target_keep_out_box  # noqa: PLC0415

        try:
            return target_keep_out_box(points, name="target", limits=limits, tuning=tuning)
        except (TypeError, ValueError):
            return None

    def _obstacle_cloud_in_candidate_frame(self, result: GraspResult) -> "np.ndarray | None":
        """The scene neighbour cloud in the winning candidate's frame, or ``None`` to skip the check.

        The calculator surfaces a CAMERA-frame cloud on ``result.metadata``. Candidates are
        BASE-frame when a frame_resolver is wired (the production path), so transform the cloud with
        the stashed CAMERA->BASE. Fail-safe: a BASE candidate with no transform available returns
        ``None``, skipping the check and proceeding, rather than validate a base grasp against a
        camera cloud (a mis-framed verdict).
        """
        meta = result.metadata or {}
        cloud = meta.get("scene_points_mm")
        if cloud is None:
            return None
        cloud = np.asarray(cloud, dtype=np.float64)
        if cloud.size == 0:
            return None
        best = result.best
        if best is not None and best.frame is GraspFrame.BASE:
            transform = self._pending_camera_to_base
            if transform is None:
                return None  # skip: do not validate a base grasp against a camera-frame cloud
            mat = np.asarray(transform.to_matrix(), dtype=np.float64)
            return (cloud @ mat[:3, :3].T) + mat[:3, 3]
        return cloud  # candidate is camera-frame, so the cloud is already camera-frame

    def _approach_sweep_is_clear(
        self, candidate: GraspPoint, cloud: "np.ndarray | None"
    ) -> bool:
        """True iff the candidate's approach and retreat sweeps are clear of the obstacle cloud."""
        try:
            grasp_pose = _grasp_point_to_grasp_pose(candidate)
        except ValueError:
            # Degenerate axes (approach ~parallel to closing) cannot build a sweep; the calculator
            # already accepted this candidate's final pose, so proceed rather than spuriously refuse.
            return True
        # The cell's own hand, not the 2F-85's defaults the validator falls back to (E2-9 of the cell analysis).
        approach_report, retreat_report = validate_approach_and_retreat(
            grasp_pose, obstacle_points_mm=cloud, policy=self.approach_path_policy, gripper_model=self.gripper_model,
        )
        return approach_report.is_clear and retreat_report.is_clear


def _seen_refusals(result: Any) -> int:
    """How many of SFE's tries for a part met a neighbour the camera saw (``support_footprint_refused``, B.1's
    ``seen_fingers`` and ``seen_corridor``); 0 where the result says nothing of them."""
    telemetry = getattr(result, "telemetry", None)
    refused = telemetry.get("support_footprint_refused") if isinstance(telemetry, Mapping) else None
    if not isinstance(refused, Mapping):
        return 0
    return int(sum(int(refused.get(key, 0) or 0) for key in ("seen_fingers", "seen_corridor")))


def _without_points_near(points_camera_mm: Any, near_base_mm: np.ndarray, camera_to_base: Any,
                         reach_mm: float = 10.0) -> Any:
    """``points_camera_mm`` (CAMERA) less those within ``reach_mm`` of ``near_base_mm`` (BASE), placed by
    ``camera_to_base``: a neighbour left out of the fused neighbours a call was handed, as its pixels are left out of
    the frame. ``None`` where none is left."""
    if points_camera_mm is None:
        return None
    points = np.asarray(points_camera_mm, dtype=np.float64).reshape(-1, 3)
    near = np.asarray(near_base_mm, dtype=np.float64).reshape(-1, 3)
    if not points.shape[0] or not near.shape[0]:
        return points_camera_mm
    matrix = getattr(camera_to_base, "to_matrix", None)
    transform = np.asarray(matrix() if callable(matrix) else camera_to_base, dtype=np.float64).reshape(4, 4)
    base = points @ transform[:3, :3].T + transform[:3, 3]
    from scipy.spatial import cKDTree  # noqa: PLC0415

    distance, _ = cKDTree(near).query(base, k=1, distance_upper_bound=float(reach_mm))
    kept = points[~np.isfinite(distance)]
    return kept if kept.shape[0] else None


def _handling_fields(report: Any) -> _MotionFields:
    """A place verb's report (``HandlingReport``), flattened onto an attempt the way :func:`_motion_fields` flattens a
    policy's: its last status and its sentence."""
    statuses = tuple(getattr(report, "statuses", ()) or ())
    status = statuses[-1] if statuses else None
    message = getattr(report, "message", "") or ""
    outcome = getattr(getattr(report, "outcome", None), "value", None)
    return {
        "motion_status": getattr(status, "value", None) if status is not None else None,
        "motion_message": message or (f"place {outcome}" if outcome else None),
        "motion_error": None, "camera_world": None, "camera_world_reason": None,
    }


def _stamp(result: "GraspResult", payload: dict) -> None:
    """Merge shadow telemetry into a result's own dict, without disturbing what is already there.

    `GraspResult.telemetry` is a plain mutable dict carrying `GraspCalculator.last_telemetry`, so
    this adds keys beside it rather than replacing anything. Every key is `deep_ranker_`-prefixed,
    so a bare name such as `top1` cannot collide with another subsystem's telemetry.
    """
    existing = getattr(result, "telemetry", None)
    if isinstance(existing, dict):
        existing.update(payload)


def _views_behind(obj: Any, index: int, fused_scene: Any, primary_name: str | None = None) -> tuple[str, ...]:
    """The cameras whose surfaces make up ``obj``'s fused cloud, the camera it is grasped from first; ``()`` for one.

    Read the way each cloud was built. A primary object's cloud is its own surface plus every other camera's blob the
    association assigned to it (`fuse_scene_clouds`), so those cameras are read off the associations, in view order.
    A promoted object's cloud is its cluster's members (`build_scene_objects`), whose cameras
    `SceneObject.views_used` already names, the owner first. A camera named twice counts once, so a rig that lists the
    primary fuses nothing onto it; fewer than two cameras is one view and reads as empty. A primary with no id in the
    rig is named ``primary``. ``primary_name`` names the primary instead: a look of a wrist pick is ``rig@look``, so two
    looks of one camera are two views.

    Never raises: this is what a report says, and it runs in the ranking of every fused frame, so a shape it does not
    know names nothing rather than ending a pick.
    """
    cloud = getattr(obj, "fused_cloud_base_mm", None)
    if cloud is None or not np.asarray(cloud).size:
        return ()
    if getattr(obj, "promoted", False):
        named = getattr(obj, "views_used", ())
        views = [str(view) for view in named] if isinstance(named, (tuple, list)) else []
    else:
        views = [primary_name or str(getattr(obj, "camera_id", "") or "primary")]
        associations = getattr(fused_scene, "associations", ())
        for association in associations if isinstance(associations, (tuple, list)) else ():
            assignment = getattr(association, "assignment", ())
            if isinstance(assignment, (tuple, list)) and index < len(assignment) and assignment[index] is not None:
                views.append(str(getattr(association, "view", "")))
    distinct = tuple(dict.fromkeys(view for view in views if view))
    return distinct if len(distinct) > 1 else ()


def _grasp_position_mm(result: "GraspResult") -> tuple[float, float, float] | None:
    """The winning candidate's BASE-frame position, for a progress event. ``None`` if unavailable.

    Best-effort by design: this feeds a display, and a malformed candidate must not be able to raise
    on the line before the arm moves. Everything that decides anything reads the candidate itself.
    """
    candidates = getattr(result, "candidates", ())
    if not candidates:
        return None
    position = getattr(candidates[0], "position", None)
    if position is None:
        return None
    try:
        return (float(position[0]), float(position[1]), float(position[2]))
    except (TypeError, ValueError, IndexError):  # pragma: no cover (defensive, display-only)
        return None
