"""
Runtime pick facade: one pick attempt behind a single synchronous call.

This module sits between any future API or UI layer and the grasping
orchestrator. Its job is narrow:

1. Own one full perception, grasp and execution attempt synchronously.
2. Carry vendor identity (real or simulated) onto the result so a
   caller can tell which backend produced the report.
3. Aggregate the typed :class:`MotionStatus` chain, plus coarse timing
   markers, into a single :class:`PickSessionReport`.

Design notes
------------
* The facade depends only on the vendor-neutral surfaces
  (:class:`RobotArm`, :class:`Gripper`, :class:`PerceptionSource`,
  :class:`GraspCalculator`, :class:`BinPickingOrchestrator`).
* No vendor SDK import lives here. Construction takes already-built
  components, or a validated ``RobotConfig`` tree through
  :meth:`RuntimePickService.from_robot_config`.
* The facade is synchronous and in-process. It is importable on macOS
  without Isaac installed because it touches no simulator code path
  directly; when a caller hands it an :class:`IsaacRobotArm`, that
  driver's lazy SDK import decides what happens.
* The session report surfaces the typed motion status chain, so
  orchestration and debugging tooling need not scrape logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TYPE_CHECKING

from src.robot.core import (
    CameraWorldStamp,
    Gripper,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
)
from src.robot.execution.robot_parts import build_gripper, resolve_arm
from src.robot.grasping.motion.execution_policy import (
    GraspExecutionPolicy,
    PolicyReport,
    weakest_camera_world,
)
from src.robot.grasping.geometry.closing_axis import natural_closing_axis_of
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PerceptionSource,
    PickAttempt,
    PickOutcome,
    PickReport,
)

if TYPE_CHECKING:  # pragma: no cover (import only for typing)
    from src.geometry import Pose

    # Telemetry carriers, referenced only from string-form field
    # annotations (PEP 563 ``from __future__ import annotations`` is in
    # effect at module top). Keeping the import behind ``TYPE_CHECKING``
    # means ``runtime_pick`` never grows a runtime dependency on the
    # predictor module: the wiring surface stays ``pick_loop``
    # (consumer) and ``autonomous_grasp`` (loader).
    from src.robot.grasping.scoring.success_probability import (
        RankingBlendTelemetry,
        ShadowSuccessTelemetry,
        UncertaintyRerankTelemetry,
    )


__all__ = [
    "PickSessionReport",
    "PickTimings",
    "RuntimePickService",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PickTimings:
    """Coarse wall-clock markers for a single :meth:`RuntimePickService.run_attempt` call.

    All values are in seconds. ``total_s`` covers the entire call duration; ``orchestrator_s`` is the
    ``BinPickingOrchestrator.run()`` span, and today the two are equal. There is no separate
    ``perception_s`` field: perception time is part of ``orchestrator_s`` because the orchestrator
    owns perception acquisition, and surfacing it separately needs a per-stage sub-timer on
    ``PickReport``.
    """

    total_s: float = 0.0
    orchestrator_s: float = 0.0


@dataclass(frozen=True, slots=True)
class PickSessionReport:
    """Typed aggregate result of one :meth:`RuntimePickService.run_attempt` call.

    Attributes
    ----------
    outcome
        Terminal :class:`PickOutcome` from the orchestrator.
    robot_vendor
        Vendor label as reported by :attr:`RobotCapabilities.vendor`
        (e.g. ``"ur"``, ``"kuka"``, ``"sim"``).
    is_simulated
        Mirror of :attr:`RobotCapabilities.is_simulated`. Lets callers
        gate "real hardware only" actions without re-checking the
        vendor string.
    gripper_present
        :data:`True` iff a gripper was wired into the policy. Object
        detection on grippers without that capability is reported as
        :data:`None` in :attr:`object_detected`.
    candidate_count
        Number of grasp candidates evaluated on the last perception
        frame seen by the orchestrator. ``0`` when no candidates were
        generated or perception was empty.
    target_index
        Segmentation index of the grasp chosen for execution, or
        :data:`None` if the loop never reached execution.
    selected_score
        Score of the executed grasp, or ``0.0`` when none ran.
    motion_status_chain
        Ordered tuple of every typed :class:`MotionStatus` observed
        along the attempt. Empty when the underlying driver only
        exposed the bool ``move_to`` path, which stays supported: an
        empty chain means the driver predates the typed contract, not
        that the attempt failed.
    motion_message
        Free-text detail from the last :class:`MotionResult`, useful
        for surfacing controller-side rejection messages.
    camera_worlds
        One :class:`CameraWorldStamp` per typed motion of the last
        policy call, in command order, threaded from
        :attr:`PolicyReport.camera_worlds` and read weakest first through
        :attr:`camera_world`. Stamps from an earlier attempt of the same
        session survive only on :attr:`attempts` and in the progress
        events. Empty for a legacy ``move_to`` driver and when no motion
        was commanded.
    object_detected
        Threaded from :attr:`PolicyReport.object_detected`. :data:`None`
        when the gripper has no detection capability.
    attempts
        Full attempt history forwarded from the orchestrator, so
        debugging tools can see every rescan without re-running the
        loop. The loop never switches target on its own: a
        ``next_target`` is a recovery action, which lives on the
        service's recovery trail where the mode's profile lists it
        (``auto`` and ``dense_clutter`` do).
    executed_grasp
        The :class:`GraspResult` that was actually executed, or
        :data:`None` when execution did not happen.
    timings
        Wall-clock markers; see :class:`PickTimings`.
    error
        Last unexpected exception observed by the execution policy, if
        any. Typed motion failures do not populate this field: they
        live in :attr:`motion_status_chain`, so a non-``None`` value
        always indicates an out-of-band failure.
    object_centre_mm
        Where the object the pick went for was seen, in BASE
        millimetres (:attr:`PickReport.target_centre_mm`). ``None`` when
        no candidate reached the arm or the cell resolves no BASE.
    grasp_pose
        The pose the tool closed at, in BASE, on a pick that executed:
        the grasp waypoint the execution policy judged and drove, read
        off its commanded waypoints rather than rebuilt. ``None`` on
        every other outcome.
    target_label
        The label of the segmentation the pick went for
        (:attr:`PickReport.target_label`): a sort reads it to tell which
        rule's place the part goes to. ``""`` where no grasp was chosen.
    unclaimed_labels
        On a sort, the labels the label gate turned away at the pick's
        first look, the parts no rule claims
        (:attr:`PickReport.unclaimed_labels`); ``()`` otherwise.
    stands_at
        Where a try left the arm that was due back at its look and was not
        driven back there (:attr:`PickReport.stands_at`, "standoff of grasp
        N", S2, 2026-10-08); ``""`` where the arm stands where the pick's
        tries started. The service's report says a person is needed on it,
        and a task stops rather than pick again from there.
    """

    outcome: PickOutcome
    robot_vendor: str
    is_simulated: bool
    gripper_present: bool
    candidate_count: int = 0
    #: The executed grasp's own telemetry, taken from the pick loop.
    #:
    #: This is what fills `GraspAttemptRecord.initial_telemetry`. Without it the calculator's
    #: telemetry stops at `GraspResult.telemetry` inside the pick loop and reaches no record, so
    #: the `deep_ranker_*` keys the shadow ranker stamps onto this dict cannot be read back from a
    #: single record however many picks run.
    #:
    #: Empty when nothing executed, so a run that reaches no grasp is byte-identical.
    calculator_telemetry: Mapping[str, Any] = field(default_factory=dict)
    target_index: Optional[int] = None
    selected_score: float = 0.0
    motion_status_chain: tuple[MotionStatus, ...] = ()
    motion_message: str = ""
    camera_worlds: tuple[CameraWorldStamp, ...] = ()
    object_detected: Optional[bool] = None
    attempts: tuple[PickAttempt, ...] = ()
    executed_grasp: Optional[GraspResult] = None
    timings: PickTimings = field(default_factory=PickTimings)
    error: Optional[BaseException] = None
    # Shadow-mode success-probability telemetry, forwarded verbatim
    # from :class:`PickReport`. ``None`` whenever the orchestrator was
    # built without a :class:`ShadowSuccessContext` or no candidate was
    # scored.
    shadow_success_telemetry: "ShadowSuccessTelemetry | None" = None
    # Bounded probability-blend reranker audit telemetry, forwarded
    # verbatim from :class:`PickReport`. ``None`` whenever the
    # orchestrator was built without a ``RankingBlendConfig`` (or
    # ``ranking_blend_config.enabled is False``); a non-``None`` record
    # describes either the applied rerank or the reason it was skipped.
    ranking_blend_telemetry: "RankingBlendTelemetry | None" = None
    # Per-candidate uncertainty re-rank telemetry, forwarded verbatim from the PickReport.
    uncertainty_rerank_telemetry: "UncertaintyRerankTelemetry | None" = None
    object_centre_mm: Optional[tuple[float, float, float]] = None
    grasp_pose: "Pose | None" = None
    target_label: str = ""
    unclaimed_labels: tuple[str, ...] = ()
    stands_at: str = ""

    @property
    def succeeded(self) -> bool:
        """Convenience flag: :data:`True` iff the grasp was executed."""

        return self.outcome is PickOutcome.EXECUTED

    @property
    def camera_world(self) -> CameraWorldStamp | None:
        """The weakest stamp of :attr:`camera_worlds` (:func:`weakest_camera_world`)."""
        return weakest_camera_world(self.camera_worlds)


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


@dataclass
class RuntimePickService:
    """Single-attempt pick service used as the internal runtime entry point.

    The facade is minimal: it wraps a fully-built
    :class:`BinPickingOrchestrator` and produces a typed
    :class:`PickSessionReport`. Construction can:

    * take a pre-built orchestrator via the ``orchestrator`` field,
    * accept the orchestrator's ingredients and build one internally
      via :meth:`from_components`, or
    * read a validated ``RobotConfig`` tree via
      :meth:`from_robot_config`.
    """

    orchestrator: BinPickingOrchestrator

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_components(
        cls,
        *,
        arm: RobotArm,
        calculator: GraspCalculator,
        perception: PerceptionSource,
        gripper: Gripper | None = None,
        max_attempts: int = 5,
        dense_sampling: bool = True,
        grasp_sampling_mode: GraspSamplingMode | bool | str | None = None,
        standoff_mm: float = 80.0,
        retreat_mm: float = 100.0,
        policy: GraspExecutionPolicy | None = None,
    ) -> "RuntimePickService":
        """Build a :class:`RuntimePickService` from raw components.

        Mirrors :class:`BinPickingOrchestrator`'s public constructor so
        callers do not need to import the orchestrator class directly
        once they have a facade in hand. The optional ``policy``
        argument is honoured if supplied; otherwise the orchestrator
        builds its own default. ``grasp_sampling_mode`` is the
        production-preferred knob; when omitted the ``dense_sampling``
        bool still drives behavior.
        """

        orchestrator = BinPickingOrchestrator(
            arm=arm,
            calculator=calculator,
            perception=perception,
            max_attempts=max_attempts,
            dense_sampling=dense_sampling,
            grasp_sampling_mode=grasp_sampling_mode,
            standoff_mm=standoff_mm,
            retreat_mm=retreat_mm,
            gripper=gripper,
            policy=policy,
        )
        return cls(orchestrator=orchestrator)

    # ------------------------------------------------------------------
    # Config-driven construction
    # ------------------------------------------------------------------

    @classmethod
    def from_robot_config(
        cls,
        robot_cfg,
        *,
        calculator: GraspCalculator,
        grasp_sampling_mode: GraspSamplingMode | bool | str | None = None,
        perception: PerceptionSource,
        max_attempts: int = 5,
        dense_sampling: bool = True,
        standoff_mm: float = 80.0,
        retreat_mm: float = 100.0,
        policy: GraspExecutionPolicy | None = None,
        arm: RobotArm | None = None,
        gripper: Gripper | None = None,
    ) -> "RuntimePickService":
        """Build a :class:`RuntimePickService` straight from a validated
        ``RobotConfig`` tree.

        ``arm`` and ``gripper`` are live-device escape hatches. Some end-effectors are not
        describable by config because they are not data: an ``IsaacGripper`` needs the simulator
        session its arm already lives on, exactly as the multi-camera rig needs a live camera
        handle. Passing the handle here lets everything else come from config. The alternative is
        ``from_components``, which builds no ``effective_config``, so the sim would measure a
        hand-wired stack while a real cell ran the config-driven one.

        A supplied handle replaces only its own construction step; nothing else is skipped. The
        arm-vendor readiness gate is skipped only when the handle advertises a different vendor than
        the config names, because such an arm is a stand-in that drives no device of the configured
        vendor. A handle that advertises the configured vendor is that vendor's driver and still
        faces the gate, since constructing one proves nothing about the SDK (see the comment at the
        gate in :func:`~src.robot.execution.robot_parts.resolve_arm`).

        A Robotiq is the one end-effector whose construction reads the arm rather than only the tree:
        it lives on the UR controller's tool I/O, so it is built only when the arm in hand reports
        vendor ``ur``, and against the controller address that arm was built with
        (``arm.config.ur.ip``) rather than against ``robot.ur.ip`` of the tree being picked with.
        Whenever this method built the arm the two are the same object. See the comment on the branch
        in :func:`~src.robot.execution.robot_parts.build_gripper`.

        The vendor block in the schema is the single
        source of truth: this method reads ``robot_cfg.vendor`` and,
        through :func:`~src.robot.execution.robot_parts.resolve_arm` and
        :func:`~src.robot.execution.robot_parts.build_gripper`,
        dispatches to the matching driver factory via
        :func:`src.robot.drivers.create_arm` /
        :func:`src.robot.grippers.create_gripper`. For
        ``RobotVendor.SIM`` the schema ``SimConfig`` is converted into
        the driver-side ``SimRobotConfig`` dataclass so the existing
        ``create_arm(RobotVendor.SIM, config=...)`` contract is honoured
        without leaking Pydantic types into the driver layer.

        Parameters
        ----------
        robot_cfg
            Validated ``RobotConfig`` instance from
            :mod:`src.config.schema.robot`.
        calculator
            Already-constructed :class:`GraspCalculator`.
        perception
            Already-constructed :class:`PerceptionSource`.
        max_attempts, dense_sampling, standoff_mm, retreat_mm, policy
            Forwarded verbatim to
            :class:`BinPickingOrchestrator` (see
            :meth:`from_components`).

        Returns
        -------
        RuntimePickService
            A ready-to-run facade. ``arm.connect()`` is not called
            here; callers stay in charge of the connect and disconnect
            lifecycle.
        """

        arm = resolve_arm(robot_cfg, arm=arm)
        # A supplied gripper is a live end-effector the config cannot describe (the Isaac gripper
        # on a shared session). Skipping the builder also skips its NullGripper substitution,
        # which is the point: the sim's `gripper.vendor: robotiq` on a `sim` arm would otherwise
        # substitute a gripper that closes on nothing and reports success.
        if gripper is None:
            gripper = build_gripper(robot_cfg, arm=arm)
        runtime = cls.from_components(
            arm=arm,
            calculator=calculator,
            perception=perception,
            gripper=gripper,
            max_attempts=max_attempts,
            grasp_sampling_mode=grasp_sampling_mode,
            dense_sampling=dense_sampling,
            standoff_mm=standoff_mm,
            retreat_mm=retreat_mm,
            policy=policy,
        )
        # How the hand and camera naturally stand (robot.natural_closing_axis): every grasp the pick loop ranks is turned
        # the way round nearer it. Unset turns none.
        runtime.orchestrator.natural_closing_axis = natural_closing_axis_of(robot_cfg)
        # How a ranking spends its time (the owner's "speed first", 2026-10-08): the first good part taken without
        # computing the rest (robot.grasping.first_good_part, good_part_score), and SFE's fine search waiting while
        # another part may have a full result (fine_pass_waits). A config that carries no such keys ranks as before.
        grasping = getattr(robot_cfg, "grasping", None)
        runtime.orchestrator.lazy_objects = bool(getattr(grasping, "first_good_part", False))
        runtime.orchestrator.accept_score = float(getattr(grasping, "good_part_score", 0.75))
        runtime.orchestrator.fine_search_waits = bool(getattr(grasping, "fine_pass_waits", False))
        return runtime

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_attempt(self) -> PickSessionReport:
        """Run one full pick attempt and return a typed session report.

        The call is synchronous. Any exception raised by the
        orchestrator propagates: unexpected runtime faults are still
        raised, while ordinary motion outcomes are typed.
        """

        start = time.perf_counter()
        pick_report = self.orchestrator.run()
        orchestrator_s = time.perf_counter() - start
        return self._compose(pick_report, orchestrator_s=orchestrator_s)

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    def _compose(
        self,
        pick_report: PickReport,
        *,
        orchestrator_s: float,
    ) -> PickSessionReport:
        capabilities = self._capabilities()
        policy_report = self.orchestrator._last_policy_report  # noqa: SLF001
        chain, message, object_detected, error, camera_worlds = self._extract_policy_fields(
            policy_report,
        )

        target_index, selected_score, candidate_count = self._executed_target_metrics(
            pick_report,
        )

        return PickSessionReport(
            outcome=pick_report.outcome,
            robot_vendor=capabilities.vendor if capabilities else "unknown",
            is_simulated=bool(capabilities.is_simulated) if capabilities else False,
            gripper_present=self.orchestrator.gripper is not None,
            candidate_count=candidate_count,
            # `selected_telemetry` first. `executed_grasp` exists only on a successful pick, and the
            # shadow ranker's opinion is about the candidates rather than about the motion, so a
            # failed execution must still carry it.
            calculator_telemetry=dict(
                getattr(pick_report, "selected_telemetry", None)
                or getattr(getattr(pick_report, "executed_grasp", None), "telemetry", None) or {}),
            target_index=target_index,
            selected_score=selected_score,
            motion_status_chain=chain,
            motion_message=message,
            camera_worlds=camera_worlds,
            object_detected=object_detected,
            attempts=pick_report.attempts,
            executed_grasp=pick_report.executed_grasp,
            timings=PickTimings(
                total_s=orchestrator_s,
                orchestrator_s=orchestrator_s,
            ),
            error=error,
            shadow_success_telemetry=pick_report.shadow_success_telemetry,
            ranking_blend_telemetry=pick_report.ranking_blend_telemetry,
            uncertainty_rerank_telemetry=pick_report.uncertainty_rerank_telemetry,
            object_centre_mm=getattr(pick_report, "target_centre_mm", None),
            grasp_pose=_closed_at(policy_report) if pick_report.outcome is PickOutcome.EXECUTED else None,
            # Which kind of part the pick went for, and what no rule claimed: a sort's (the owner, 2026-10-09).
            target_label=str(getattr(pick_report, "target_label", "") or ""),
            unclaimed_labels=tuple(getattr(pick_report, "unclaimed_labels", ()) or ()),
            # Where a try left the arm when its move back to the look did not run: the service's report and the task
            # read it here, the one report they are handed (S2, 2026-10-08).
            stands_at=str(getattr(pick_report, "stands_at", "") or ""),
        )

    def _capabilities(self) -> RobotCapabilities | None:
        """Return :class:`RobotCapabilities` if the arm exposes them.

        A driver that does not implement the property yields
        :data:`None`, and the facade falls back to ``"unknown"`` and
        ``is_simulated=False``.
        """

        caps = getattr(self.orchestrator.arm, "capabilities", None)
        if isinstance(caps, RobotCapabilities):
            return caps
        return None

    @staticmethod
    def _extract_policy_fields(
        policy_report: PolicyReport | None,
    ) -> tuple[
        tuple[MotionStatus, ...], str, bool | None, BaseException | None,
        tuple[CameraWorldStamp, ...],
    ]:
        """Pull the typed motion fields off the orchestrator's last policy report.

        The orchestrator stores only the most recent policy report,
        because each pick attempt fires a single policy call (the loop
        retries at the orchestrator level by re-running perception, not
        by replaying execution). The contract therefore yields at most
        one :class:`PolicyReport` per session, and its
        :attr:`PolicyReport.motion_status` is the canonical chain
        entry.

        The camera-world stamps share that limit: they are the last
        policy call's. An earlier attempt's weakest stamp survives on its
        :class:`PickAttempt` and in its progress event.
        """

        if policy_report is None:
            return (), "", None, None, ()

        chain: tuple[MotionStatus, ...]
        if policy_report.motion_status is None:
            chain = ()
        else:
            chain = (policy_report.motion_status,)

        return (
            chain,
            policy_report.motion_message,
            policy_report.object_detected,
            policy_report.error,
            tuple(policy_report.camera_worlds),
        )

    @staticmethod
    def _executed_target_metrics(
        pick_report: PickReport,
    ) -> tuple[Optional[int], float, int]:
        """Derive ``target_index``, ``selected_score``, ``candidate_count``.

        ``candidate_count`` is the length of the executed grasp's ranked candidate list, and
        ``0`` whenever no grasp executed. ``target_index`` comes from the last attempt in either
        case, and the executed grasp's own ``top_score`` is used as ``selected_score`` when there
        is one.
        """

        if pick_report.executed_grasp is not None:
            last = pick_report.attempts[-1] if pick_report.attempts else None
            target_index = last.target_index if last is not None else None
            return (
                target_index,
                float(pick_report.executed_grasp.top_score),
                len(pick_report.executed_grasp.candidates)
                if hasattr(pick_report.executed_grasp, "candidates")
                else 0,
            )

        if not pick_report.attempts:
            return None, 0.0, 0
        last = pick_report.attempts[-1]
        return last.target_index, float(last.score), 0


def _closed_at(policy_report: PolicyReport | None) -> "Pose | None":
    """The pose the tool closed at: the last commanded waypoint before the lift, on a policy report that executed.

    The policy labels its lift ``retreat`` (or ``retreat_NN``) and everything before it ``approach_NN``, the grasp
    last, whether it drove the interpolated approach or a planned standoff and one line. So the pose is read off what
    was commanded, the move that was judged and ran, rather than rebuilt from the grasp point, which a yaw alignment
    (``align_closing_to_base_x``) would get wrong.
    """
    if policy_report is None:
        return None
    before_lift = [pose for pose in getattr(policy_report, "waypoints", ()) or ()
                   if not str(getattr(pose, "label", "") or "").startswith("retreat")]
    return before_lift[-1] if before_lift else None
