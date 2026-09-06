"""The seam a grasp calculator has to satisfy, measured from what the runtime actually touches.

The runtime contract: the members the orchestrator, the closed-loop refiner and the operator
service genuinely read or write, found by parsing every access on the live paths rather than by
reading the class and copying its surface. That surface is deliberately tiny: `GraspCalculator`
has ~40 public members, and a protocol demanding them all would make a second implementation
impossible for no reason.

    compute_result(...)      the one ranking call site (pick_loop, refinement)   required
    render_debug_images      read and written; the operator console flips it       required
    last_debug_image_png     read after a compute when debug rendering is on       required
    camera_matrix            read by the single-view support-plane refinement      optional, below

The argument list is deliberately untyped. `compute()` takes thirty-odd keyword arguments and the
orchestrator assembles them conditionally: `camera_to_base`, `scene_points_mm`,
`rigid_obstacle_points_mm`, `geometry_points_base_mm`, `gripper_model`, `corridor_config`,
`feasibility_config`, `support_plane`, `min_table_clearance_mm` and more appear only when their
config block is enabled. Operator config reaches the calculator as per-call parameters, so that
list grows: an implementation must accept `**kwargs` and ignore what it does not understand, and
what it must not do is crash on an argument a future config block adds.

`camera_matrix` is read by the runtime and is deliberately not required here.
`pick_loop._target_surface_base_mm` reads it duck-typed, `getattr(self.calculator,
"camera_matrix", None)`, to back-project a single-view target cloud, and treats absence as "stand
down", so it is an optional capability and a protocol member would misdescribe it as mandatory.

Optional in the contract and optional in fact. `GraspCalculator` exposes `camera_matrix` as a
read-only property returning the calibrated K, never the `focal = width / 2` fallback that
`intrinsics()` synthesizes, so a cell built without real intrinsics reads `None` here and keeps
the single-view branch stood down. `support.refine_from_target` defaults to True, so wherever the
K is real that branch runs. What a wrong frame costs is recorded on `_target_surface_base_mm`
itself: a support plane placed at the wrist-camera standoff instead of at the table rejects every
candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.types.feedback import GraspResult

__all__ = ["GraspCandidateGenerator"]


@runtime_checkable
class GraspCandidateGenerator(Protocol):
    """Anything the pick loop can rank grasps with.

    `runtime_checkable` follows the precedent of `MultiContactGraspPlanner`, which is
    isinstance-checked at its injection point: a seam worth naming is a seam worth checking. A
    runtime `isinstance` against a Protocol checks that the members exist, not that their
    signatures match; it catches the wrong object, not the wrong arguments.
    """

    #: Whether this generator should render a debug overlay for the last computed grasp. Written by
    #: the operator service when the console turns previews on, so it is not read-only state.
    render_debug_images: bool

    #: PNG bytes of that overlay, or `None`. Read after `compute_result` when the flag above is set.
    last_debug_image_png: Optional[bytes]

    def compute_result(self, *args: Any, **kwargs: Any) -> "GraspResult":
        """Rank grasp candidates for one segmentation and return the structured result.

        Contract beyond the signature, every clause of it binding:

        * Never raise on "no grasp". An empty candidate list plus a typed `GraspFailureReason`
          is the answer; the retry / rescan / relocate / recovery routing runs entirely off
          those reasons, and an exception here bypasses all of it.
        * Frames. When the caller supplies `camera_to_base`, the returned candidates must be in
          the BASE frame; `GraspExecutionPolicy.require_base_frame_grasp` refuses camera-frame
          grasps on every resolver-wired path.
        * Units. Millimetres, XYZW quaternions, every pose frame-tagged, per the repo-wide rule.
        * Metadata. The per-candidate keys downstream consumers read (`total`, `geometric`,
          `stability`, `reachability`, `feasibility_score`, `pose_confidence`, `score_components`,
          ...) are a de-facto contract for the success model, the RL candidate rows and the shadow
          aggregator. A generator that omits them does not crash anything; it silently empties those
          features, which is worse.
        """
        ...


@runtime_checkable
class PreloadableGenerator(Protocol):
    """A generator that can be asked to load its weights before the first pick.

    A second protocol rather than a fourth member on `GraspCandidateGenerator`. `preload`
    declared there would stop the analytic calculator satisfying that protocol, and the analytic
    calculator is the default, so the seam would declare that the shipping generator is not a
    generator.

    The operator service's cell builder reads the name duck-typed, `getattr(calculator,
    "preload", None)`, and calls it when it is callable; that builder is not in this tree, so
    `preload` ships here with an implementation and no caller. Declaring it keeps the declared
    surface equal to what the runtime reads.

    `compute_result` may not raise, because a cell must not go down over one frame. A mismatched
    artifact therefore surfaces as an empty candidate list on every object rather than as a
    refusal, and the operator sees a model that finds nothing instead of a setup error. `preload`
    is the one moment a caller can still refuse.
    """

    def preload(self) -> None:
        """Load whatever the first pick will need, raising if it cannot be loaded."""
        ...

