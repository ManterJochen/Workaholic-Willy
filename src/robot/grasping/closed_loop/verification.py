"""Post-grasp verification.

This module ships the typed contract and a small family of default
implementations for the locked "verify a pick actually grabbed
something" workflow:

    1. The execution policy approaches, closes the gripper, and
       retreats to a lift pose.
    2. The high-level :class:`AutonomousGraspService` hands the
       executed grasp, the gripper, snapshots of jaw widths before
       and after close, an optional post-lift perception frame, and
       the :class:`TargetIdentity` that drove the pick to a
       :class:`GraspVerifier`.
    3. The verifier returns a :class:`GraspVerificationReport`.
    4. The service maps :attr:`VerificationOutcome.FAILED` (or, when
       the operator configured :attr:`GraspVerificationPolicy.fail_closed`,
       :attr:`VerificationOutcome.INCONCLUSIVE`) to
       :attr:`AutonomousGraspOutcome.VERIFICATION_FAILED`: no silent
       success.

Scope and non-goals
-------------------

* Verification runs after :class:`GraspExecutionPolicy.execute` has
  returned :attr:`PolicyOutcome.EXECUTED`. The execution policy keeps
  its :class:`ObjectDetectingGripper` close-time check untouched:
  that is the locked "legacy trust path", and it must keep working
  for callers who do not wire a verifier.
* This module owns no motion. Acquiring the post-lift frame (when
  configured) is the service's responsibility; the verifier consumes
  it through :class:`GraspVerificationContext`.
* Robotiq-specific boolean object detection is not reimplemented
  here. :class:`ObjectDetectingGripperVerifier` only delegates to
  whatever the configured :class:`Gripper` advertises via the
  :class:`ObjectDetectingGripper` Protocol; if a vendor driver lies
  about that bit, the lie surfaces through ``VERIFICATION_FAILED``
  rather than being hidden by this module.
* No state is stored across calls. A verifier is a pure function from
  :class:`GraspVerificationContext` to :class:`GraspVerificationReport`.

Public surface
--------------

* :class:`VerificationOutcome`: typed terminal status.
* :class:`GraspVerificationPolicy`: operator-bounded configuration.
* :class:`GraspVerificationContext`: frozen aggregate of inputs.
* :class:`GraspVerificationReport`: frozen aggregate of outputs.
* :class:`GraspVerifier`: Protocol.
* :class:`NoOpVerifier`: always :attr:`VerificationOutcome.PASSED`,
  the wrapper for the locked "legacy trust path".
* :class:`ObjectDetectingGripperVerifier`: delegates to
  :class:`ObjectDetectingGripper`.
* :class:`WidthDeltaGripperVerifier`: checks the post-close jaw
  opening sits between configured bounds.
* :class:`VisionTargetDisplacementVerifier`: re-acquires the scene
  and asserts the previously identified target is no longer present
  (i.e. the gripper is carrying it).
* :class:`CompositeGraspVerifier`: combines several verifiers under
  configurable aggregation rules.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, runtime_checkable

from src.robot.core import Gripper, ObjectDetectingGripper
from src.robot.grasping.types.grasp_point import GraspPoint
from src.robot.grasping.closed_loop.refinement import (
    IoUCentroidTargetTracker,
    TargetIdentity,
    TargetTracker,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.types.perception import PerceptionFrame


__all__ = [
    "CompositeGraspVerifier",
    "GraspVerificationContext",
    "GraspVerificationPolicy",
    "GraspVerificationReport",
    "GraspVerifier",
    "NoOpVerifier",
    "ObjectDetectingGripperVerifier",
    "VerificationOutcome",
    "VisionTargetDisplacementVerifier",
    "WidthDeltaGripperVerifier",
]


class VerificationOutcome(StrEnum):
    """Terminal status of a single :meth:`GraspVerifier.verify` call.

    The taxonomy has four values:

    * :attr:`PASSED`: the verifier observed positive evidence the
      grasp succeeded (object detected, jaw width plausible, target
      missing from the scene, ...).
    * :attr:`FAILED`: the verifier observed positive evidence the
      grasp failed (empty jaws, full close, target still visible).
    * :attr:`INCONCLUSIVE`: the verifier could not collect the
      evidence it needs to decide (no gripper, no detection
      capability, no perception frame, ...). The operator chooses how
      this maps to the high-level outcome via
      :attr:`GraspVerificationPolicy.fail_closed`.
    * :attr:`SKIPPED`: the policy itself is disabled. The
      :class:`AutonomousGraspService` never invokes a verifier whose
      policy is disabled; this value exists so a verifier can
      short-circuit cleanly inside :class:`CompositeGraspVerifier`.
    """

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"



#: Named so an operator can silence or raise this one voice alone; the warning below is
#: the only thing that distinguishes "verified" from "nobody could measure".
_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class GraspVerificationPolicy:
    """Bounded operator configuration for post-grasp verification.

    All fields have defaults sized for typical bin-picking with a
    parallel-jaw gripper. Values are conservative: a verifier that
    cannot reach a conclusion is reported honestly via
    :attr:`VerificationOutcome.INCONCLUSIVE`, and the operator decides
    via :attr:`fail_closed` whether that becomes a successful pick or
    a verification failure.

    Attributes
    ----------
    enabled
        Master switch. When :data:`False` the service never invokes
        the verifier and falls back to the locked "legacy trust
        path" (the close-time :class:`ObjectDetectingGripper` check
        inside :class:`GraspExecutionPolicy`).
    require_object_detected
        When :data:`True` and the configured gripper advertises
        :class:`ObjectDetectingGripper`, an
        ``is_object_detected() is False`` flips
        :class:`ObjectDetectingGripperVerifier` to
        :attr:`VerificationOutcome.FAILED`. When :data:`False` the
        same situation yields :attr:`VerificationOutcome.INCONCLUSIVE`,
        which suits grippers whose object-detection bit is unreliable.
    width_delta_min_mm
        Minimum acceptable post-close jaw width above the gripper's
        physical closed width. When the jaws collapse to within
        ``closed_width_mm + width_delta_min_mm`` they are considered
        fully closed (empty grasp). Defaults to ``2.0`` mm to allow
        for small objects without losing the empty-grasp signal.
    width_delta_max_mm
        Optional maximum acceptable post-close jaw width below the
        commanded close width. When set, a post-close width that
        exceeds ``commanded + width_delta_max_mm`` (e.g. the gripper
        never actually closed) is flagged as failure. :data:`None`
        disables the upper-bound check.
    post_lift_vision_check
        When :data:`True` the service acquires a fresh perception
        frame after retreat and supplies it to the verifier through
        :attr:`GraspVerificationContext.post_lift_frame`. The default
        :class:`VisionTargetDisplacementVerifier` then asserts the
        previously identified target is no longer present at the
        pick location.
    vision_displacement_iou_max
        Upper bound on the IoU a candidate segmentation can score
        against the original :class:`TargetIdentity` in the post-lift
        frame. Above this the verifier reports
        :attr:`VerificationOutcome.FAILED` (the target is still on
        the table). Defaults to ``0.2``.
    fail_closed
        When :data:`True` (default) a
        :attr:`VerificationOutcome.INCONCLUSIVE` outcome is mapped to
        :attr:`VerificationOutcome.FAILED` for the high-level service
        outcome. Set :data:`False` only when the operator explicitly
        accepts the risk of unverified picks (e.g. a gripper with no
        feedback and no vision check).
    """

    enabled: bool = False
    require_object_detected: bool = False
    width_delta_min_mm: float = 2.0
    width_delta_max_mm: Optional[float] = None
    post_lift_vision_check: bool = False
    vision_displacement_iou_max: float = 0.2
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if self.width_delta_min_mm < 0.0:
            raise ValueError(
                "width_delta_min_mm must be non-negative; "
                f"got {self.width_delta_min_mm}"
            )
        if (
            self.width_delta_max_mm is not None
            and self.width_delta_max_mm < 0.0
        ):
            raise ValueError(
                "width_delta_max_mm must be non-negative when set; "
                f"got {self.width_delta_max_mm}"
            )
        if not 0.0 <= self.vision_displacement_iou_max <= 1.0:
            raise ValueError(
                "vision_displacement_iou_max must be in [0, 1]; "
                f"got {self.vision_displacement_iou_max}"
            )


@dataclass(frozen=True, slots=True)
class GraspVerificationContext:
    """Frozen aggregate of everything a verifier needs.

    Fields are :data:`None` when the upstream service was unable (or
    unwilling) to collect that signal. Built-in verifiers all degrade
    gracefully to :attr:`VerificationOutcome.INCONCLUSIVE` when their
    required inputs are missing; they never crash.
    """

    grasp: GraspPoint
    policy: GraspVerificationPolicy
    gripper: Optional[Gripper] = None
    pre_close_width_mm: Optional[float] = None
    post_close_width_mm: Optional[float] = None
    commanded_close_width_mm: Optional[float] = None
    post_lift_frame: Optional["PerceptionFrame"] = None
    target_identity: Optional[TargetIdentity] = None


@dataclass(frozen=True, slots=True)
class GraspVerificationReport:
    """Frozen aggregate result of a single verifier call.

    Attributes
    ----------
    outcome
        Terminal :class:`VerificationOutcome`.
    reason
        Short machine-readable string explaining the decision. Stable
        keys; safe to log to JSONL.
    telemetry
        Free-form key/value bag. JSON-serializable values only.
    """

    outcome: VerificationOutcome
    reason: str = ""
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class GraspVerifier(Protocol):
    """Vendor-neutral post-grasp verifier."""

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        ...


@dataclass(frozen=True, slots=True)
class NoOpVerifier:
    """Always reports :attr:`VerificationOutcome.PASSED`.

    The wrapper for the locked "legacy trust path": an operator who
    wires verification on but supplies a :class:`NoOpVerifier` is
    explicitly opting out of post-grasp evidence. The service still
    runs the verifier so telemetry stays uniform.
    """

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        return GraspVerificationReport(
            outcome=VerificationOutcome.PASSED,
            reason="noop",
            telemetry={"verifier": "noop"},
        )


@dataclass(frozen=True, slots=True)
class ObjectDetectingGripperVerifier:
    """Delegate to :meth:`ObjectDetectingGripper.is_object_detected`.

    * Gripper missing, or not advertising the capability:
      :attr:`VerificationOutcome.INCONCLUSIVE`.
    * Capability returns :data:`True`: :attr:`VerificationOutcome.PASSED`.
    * Capability returns :data:`False`:
      * with :attr:`GraspVerificationPolicy.require_object_detected`,
        :attr:`VerificationOutcome.FAILED`,
      * without it, :attr:`VerificationOutcome.INCONCLUSIVE` so the
        operator can pair this verifier with a vision check that has
        the final say.
    """

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        gripper = context.gripper
        if gripper is None or not isinstance(gripper, ObjectDetectingGripper):
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="no_object_detection_capability",
                telemetry={"verifier": "object_detecting_gripper"},
            )
        try:
            detected = bool(gripper.is_object_detected())
        except Exception as exc:  # noqa: BLE001 (keep verifier total)
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="object_detection_query_raised",
                telemetry={
                    "verifier": "object_detecting_gripper",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        if detected:
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="gripper_object_detected",
                telemetry={
                    "verifier": "object_detecting_gripper",
                    "object_detected": True,
                },
            )
        if context.policy.require_object_detected:
            return GraspVerificationReport(
                outcome=VerificationOutcome.FAILED,
                reason="gripper_object_not_detected",
                telemetry={
                    "verifier": "object_detecting_gripper",
                    "object_detected": False,
                },
            )
        return GraspVerificationReport(
            outcome=VerificationOutcome.INCONCLUSIVE,
            reason="gripper_reports_empty_but_policy_does_not_require",
            telemetry={
                "verifier": "object_detecting_gripper",
                "object_detected": False,
            },
        )


@dataclass(frozen=True, slots=True)
class WidthDeltaGripperVerifier:
    """Verify a grasp by inspecting the post-close jaw width.

    An engaged gripper holds an object whose cross-section is
    somewhere between its commanded close width and its mechanical
    minimum. A jaw width too close to the minimum means the jaws
    fully collapsed (empty grasp); a jaw width that barely moved from
    the commanded close width means the jaws never actually clamped
    down (when an upper bound is configured).

    Inputs required:

    * a :class:`Gripper` on the context (to read ``closed_width_mm``,
      falling back to ``min_width_mm`` when the gripper has no such
      field),
    * :attr:`GraspVerificationContext.post_close_width_mm`.

    Either of those missing yields
    :attr:`VerificationOutcome.INCONCLUSIVE`.

    A jawless gripper is refused before any of that arithmetic runs, and it is refused by name.
    Measured on this tree: ``from_robot_config`` answers four impossible gripper configurations with
    a working ``NullGripper`` (``gripper.vendor: robotiq`` on a non-UR arm is one, and it is what
    ``Cell.rehearsal`` boots). That object takes ``set_width_mm(5.0)`` and answers
    ``get_width_mm() -> 85.0``, the configured maximum, forever. 85 mm sits far above
    ``closed + width_delta_min_mm``, so this verifier read it as jaws holding 85 mm of something and
    the pick reported SUCCEEDED with nothing on the flange.

    Three repairs were weighed and two were rejected:

    (a) Have the substituted gripper echo the command back. Physically it is the closest model of a
        position-controlled jaw travelling on air, and it is still a lie here, a better-dressed one.
        The close width this stack commands is ``grip_width_mm - close_squeeze_mm``
        (``execution_policy.py`` ``_resolve_close_width``), which is the predicted cross-section of
        the object minus about a millimetre. An echo therefore answers "39 mm, exactly as asked",
        clears the collapse threshold, clears the optional upper bound, and now varies with the
        object, so the fabricated evidence tracks the scene and reads more convincingly than the
        constant it replaced. It is also already built: ``DummyGripper`` is exactly a gripper that
        clamps and echoes, and it exists to be a plausible one. A ``NullGripper`` exists to be
        distinguishable from a gripper, and (a) would delete that difference.

    (b) Report the physically closed width after a close. It reaches the right verdict and it
        reaches it by coincidence. A ``NullGripper`` carries no ``closed_width_mm``, so the only
        closed-ish value it could report is ``min_width_mm``, which is the very number this verifier
        falls back to as its threshold base; give the class a ``closed_width_mm`` one day and the
        coincidence moves. It also still hands a number to anything that measures jaw travel (85 mm
        to 5 mm, an 80 mm sweep that no motor made), and it names nothing: the operator is sent to
        the width thresholds to explain a cell that has no gripper.

    (c) Refuse on what the object says about itself, which is what runs. There are no jaws, so
        nothing was held, and that is true regardless of whether anybody wanted a gripper: it is
        positive evidence of failure rather than absent evidence, hence FAILED and not INCONCLUSIVE,
        and an operator's ``fail_closed: false`` cannot turn it back into a success.

    Two attributes are read, in that order, and both by ``getattr`` so this module keeps knowing
    nothing about the gripper package. ``substitution`` (a ``GripperSubstitution``, already attached
    by the build path, already read by the operator console which refuses to connect and by the sim
    runners which warn) means a gripper was asked for and could not be built, giving reason
    ``no_end_effector_built``, carrying what was requested. ``holds_nothing`` means the object grips
    nothing whatever the config wanted, giving reason ``no_end_effector_configured``, which is what
    ``gripper.vendor: none`` reaches. Decided 2026-09-09, inverting a pinned test: a deliberately
    gripper-less cell must not verify a grasp either. The reasons stay two strings because the two
    repairs are different, and a rehearsal that wants motion without a grasp verdict turns
    verification off rather than collecting a fabricated pass.
    """

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        gripper = context.gripper
        if gripper is None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="no_gripper",
                telemetry={"verifier": "width_delta"},
            )
        # Before the width sample is even looked for: an empty flange is knowable without a
        # readback, and a verifier that waited for one would degrade to INCONCLUSIVE on the gripper
        # that most needs a verdict. ``getattr`` rather than an isinstance check on ``NullGripper``,
        # so this reads attributes and not the class: any driver that grows the same fields
        # participates, and this module keeps knowing nothing about the gripper package.
        substitution = getattr(gripper, "substitution", None)
        if substitution is not None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.FAILED,
                reason="no_end_effector_built",
                telemetry={
                    "verifier": "width_delta",
                    "substitution_reason": str(getattr(substitution, "reason", "")),
                    "requested_gripper": str(getattr(substitution, "requested", "")),
                    "substitution_detail": str(getattr(substitution, "detail", "")),
                    "substitution_fix": str(getattr(substitution, "fix", "")),
                    # Kept so the record still shows what the absent gripper claimed. On the
                    # shipped config that is 85.0 mm after a commanded 5.0 mm close.
                    "post_close_width_mm": context.post_close_width_mm,
                },
            )
        # The same verdict for a cell that was never given an end-effector, and a second reason
        # string rather than a shared one: an operator whose Robotiq could not be built edits the
        # arm/gripper pairing, an operator on ``gripper.vendor: none`` fits a gripper or stops
        # verifying, and a records rollup has to be able to count those two populations apart.
        if bool(getattr(gripper, "holds_nothing", False)):
            return GraspVerificationReport(
                outcome=VerificationOutcome.FAILED,
                reason="no_end_effector_configured",
                telemetry={
                    "verifier": "width_delta",
                    "post_close_width_mm": context.post_close_width_mm,
                },
            )
        if context.post_close_width_mm is None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="no_post_close_width_sample",
                telemetry={"verifier": "width_delta"},
            )
        # The physical closed width, not the policy floor. "Did the jaws collapse on nothing?" is a
        # question about the mechanism, and ``min_width_mm`` is a policy value (the shipped 5.0 is the
        # smallest meaningful grip, while a 2F-85 physically shuts to 0). Reading the floor puts this
        # threshold at 7 mm and reports a genuinely held 6 mm part as an empty grasp. The ``getattr``
        # fallback keeps a gripper without the field working unchanged.
        _closed = getattr(gripper, "closed_width_mm", None)
        min_width = float(_closed) if _closed is not None else float(
            getattr(gripper, "min_width_mm", 0.0) or 0.0
        )
        post = float(context.post_close_width_mm)
        min_engaged_width = min_width + float(
            context.policy.width_delta_min_mm
        )
        telemetry: dict[str, Any] = {
            "verifier": "width_delta",
            "post_close_width_mm": post,
            "min_width_mm": min_width,
            "min_engaged_width_mm": min_engaged_width,
        }
        if post <= min_engaged_width:
            return GraspVerificationReport(
                outcome=VerificationOutcome.FAILED,
                reason="jaws_collapsed_to_minimum",
                telemetry=telemetry,
            )
        upper_bound = context.policy.width_delta_max_mm
        commanded = context.commanded_close_width_mm
        # ⛔ **THE COLLAPSE TEST ABOVE CAN BE UNREACHABLE ARITHMETIC, AND IT USED TO REPORT A PASS.**
        # It asks whether the jaws shut to the PHYSICAL closed width. A jaw gripper stops where it was
        # commanded, so that can only happen when the command itself was at or below the threshold.
        # MEASURED 2026-09-10 on the shipped Hand-E profile: the threshold is
        # closed_width_mm (0.0) + width_delta_min_mm (2.0) = 2.0 mm, while the pick path clamps every
        # close to min_width_mm = 5.0 mm, so an empty close lands at 5.0 and `post <= 2.0` is false by
        # construction. The verifier then returned PASSED, which reads as "the grasp was confirmed"
        # and meant "the question was never asked".
        #
        # Recorded on every attempt, so an operator reading telemetry can see which half of this
        # verifier was live.
        # ⚠ RECORDED, NOT ESCALATED, AND THE DIFFERENCE IS A RETRACTION. On 2026-09-10 this returned
        # INCONCLUSIVE when the collapse half could not fire and no upper bound was configured. Two
        # standing tests refused it, and they were right: a plausible width is documented as a pass,
        # twice, and a new claim against a standing artifact loses first. It also bought nothing
        # where the defect lives, because the shipped config always sets `width_delta_max_mm`, so the
        # branch could only ever fire in policies built WITHOUT one, which is the test contexts and
        # the sim runners rather than a real cell.
        #
        # The stamp stays because it is true everywhere and costs nothing: a reader can see which
        # half of this verifier was live on a given attempt. The protection against an empty close is
        # `ObjectDetectingGripperVerifier` reading gOBJ, which now exists on the gripper that ships.
        telemetry["collapse_test_reachable"] = bool(
            commanded is None or float(commanded) <= min_engaged_width
        )
        if upper_bound is not None and commanded is not None:
            ceiling = float(commanded) + float(upper_bound)
            telemetry["upper_bound_mm"] = ceiling
            if post > ceiling:
                return GraspVerificationReport(
                    outcome=VerificationOutcome.FAILED,
                    reason="jaws_did_not_close_enough",
                    telemetry=telemetry,
                )
        return GraspVerificationReport(
            outcome=VerificationOutcome.PASSED,
            reason="width_within_bounds",
            telemetry=telemetry,
        )


@dataclass(frozen=True, slots=True)
class VisionTargetDisplacementVerifier:
    """Verify a grasp by checking the target left the workspace.

    Uses the same :class:`TargetTracker` machinery that drives the
    two-scan refinement. The post-lift frame is scanned for any segmentation
    whose IoU against the original :class:`TargetIdentity` exceeds
    :attr:`GraspVerificationPolicy.vision_displacement_iou_max`. If
    one is found, the target is still on the table and verification
    fails. If none is found the outcome is PASSED; when no perception
    frame was supplied it is INCONCLUSIVE.

    The tracker is injectable, so a caller can replace the default
    IoU matcher without touching the verifier.
    """

    tracker: TargetTracker = field(default_factory=IoUCentroidTargetTracker)

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        frame = context.post_lift_frame
        identity = context.target_identity
        if frame is None or identity is None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="no_post_lift_frame_or_target_identity",
                telemetry={"verifier": "vision_target_displacement"},
            )
        if not frame.segmentations:
            # No candidates at all means the target is gone.
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="post_lift_frame_has_no_segmentations",
                telemetry={
                    "verifier": "vision_target_displacement",
                    "match_iou": 0.0,
                },
            )
        # The tracker is expected to fail to find the target. The
        # threshold handed to it is the maximum tolerable IoU:
        # anything at or above that means the target is still
        # there.
        match = self.tracker.match(
            identity,
            frame,
            iou_threshold=context.policy.vision_displacement_iou_max,
        )
        if match is None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="target_no_longer_visible",
                telemetry={
                    "verifier": "vision_target_displacement",
                    "iou_threshold": context.policy.vision_displacement_iou_max,
                },
            )
        idx, iou = match
        return GraspVerificationReport(
            outcome=VerificationOutcome.FAILED,
            reason="target_still_visible",
            telemetry={
                "verifier": "vision_target_displacement",
                "match_segmentation_index": int(idx),
                "match_iou": float(iou),
                "iou_threshold": context.policy.vision_displacement_iou_max,
            },
        )


@dataclass(frozen=True, slots=True)
class CompositeGraspVerifier:
    """Combine several verifiers under a typed aggregation rule.

    Aggregation rules
    -----------------
    * ``"all_must_pass"`` (default): the composite reports
      :attr:`VerificationOutcome.PASSED` only when every child returns
      :attr:`VerificationOutcome.PASSED`. The first
      :attr:`VerificationOutcome.FAILED` short-circuits the chain.
      :attr:`VerificationOutcome.INCONCLUSIVE` children are treated as
      passes unless :attr:`require_all_conclusive` is :data:`True`, in
      which case they are treated as failures.
    * ``"any_pass"``: PASSED is reported as soon as any child returns
      :attr:`VerificationOutcome.PASSED`. If every child is
      :attr:`VerificationOutcome.INCONCLUSIVE` the composite reports
      :attr:`VerificationOutcome.INCONCLUSIVE`; otherwise it reports
      :attr:`VerificationOutcome.FAILED`.

    Both rules surface the per-child telemetry under
    ``telemetry["children"]`` so an operator can tell which check
    rejected the pick.
    """

    verifiers: tuple[GraspVerifier, ...]
    rule: str = "all_must_pass"
    require_all_conclusive: bool = False

    def __post_init__(self) -> None:
        if not self.verifiers:
            raise ValueError(
                "CompositeGraspVerifier requires at least one child verifier"
            )
        if self.rule not in {"all_must_pass", "any_pass"}:
            raise ValueError(
                "CompositeGraspVerifier.rule must be 'all_must_pass' or "
                f"'any_pass'; got {self.rule!r}"
            )

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        child_reports: list[GraspVerificationReport] = [
            child.verify(context) for child in self.verifiers
        ]
        child_telemetry = tuple(
            {
                "outcome": str(r.outcome),
                "reason": r.reason,
                "telemetry": dict(r.telemetry),
            }
            for r in child_reports
        )
        if self.rule == "all_must_pass":
            for report in child_reports:
                if report.outcome is VerificationOutcome.FAILED:
                    return GraspVerificationReport(
                        outcome=VerificationOutcome.FAILED,
                        reason=f"child_failed:{report.reason}",
                        telemetry={
                            "verifier": "composite",
                            "rule": self.rule,
                            "children": child_telemetry,
                        },
                    )
                if (
                    self.require_all_conclusive
                    and report.outcome is VerificationOutcome.INCONCLUSIVE
                ):
                    return GraspVerificationReport(
                        outcome=VerificationOutcome.FAILED,
                        reason=f"child_inconclusive:{report.reason}",
                        telemetry={
                            "verifier": "composite",
                            "rule": self.rule,
                            "children": child_telemetry,
                        },
                    )
            # ⛔ **"PASSED" HERE CAN MEAN "NOBODY MEASURED ANYTHING", AND IT SAID SO NOWHERE.**
            # `require_all_conclusive` is documented as counting an inconclusive child as a pass
            # "and logs a WARNING plus a telemetry stamp saying verification ran and learned
            # nothing". MEASURED 2026-09-10: neither existed. In `verification_outcome`, the field a
            # KPI roll-up and an operator both read, an attempt where every verifier shrugged was
            # identical to one where the grasp was confirmed.
            learned_nothing = all(
                r.outcome is VerificationOutcome.INCONCLUSIVE for r in child_reports
            ) and bool(child_reports)
            if learned_nothing:
                _LOGGER.warning(
                    "grasp verification ran and learned nothing: every verifier returned "
                    "INCONCLUSIVE (%s). This is reported as a PASS because "
                    "require_all_conclusive is false; set it true once the gripper reports a real "
                    "measurement.",
                    ", ".join(r.reason for r in child_reports),
                )
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason=("no_verifier_could_measure" if learned_nothing
                        else "all_children_passed_or_inconclusive"),
                telemetry={
                    "verifier": "composite",
                    "rule": self.rule,
                    "children": child_telemetry,
                    #: ⭐ THE STAMP THE SCHEMA PROMISED. False means this PASS is evidence.
                    "measured_something": not learned_nothing,
                },
            )
        # "any_pass"
        any_pass = any(
            r.outcome is VerificationOutcome.PASSED for r in child_reports
        )
        if any_pass:
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="at_least_one_child_passed",
                telemetry={
                    "verifier": "composite",
                    "rule": self.rule,
                    "children": child_telemetry,
                },
            )
        all_inconclusive = all(
            r.outcome is VerificationOutcome.INCONCLUSIVE
            for r in child_reports
        )
        return GraspVerificationReport(
            outcome=(
                VerificationOutcome.INCONCLUSIVE
                if all_inconclusive
                else VerificationOutcome.FAILED
            ),
            reason=(
                "all_children_inconclusive"
                if all_inconclusive
                else "no_child_passed"
            ),
            telemetry={
                "verifier": "composite",
                "rule": self.rule,
                "children": child_telemetry,
            },
        )
