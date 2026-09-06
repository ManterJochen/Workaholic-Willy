"""Grasp-mode selection for the Isaac sim runners.

One place that maps a demo-mode string to the typed :class:`GraspMode` and to the
``AutonomousGraspService.from_components`` sub-policy kwargs that mode needs, so every pick runner
and the mode-matrix harness wire the modes the same way:

* ``easy``: no extra wiring, the deterministic trust path.
* ``auto``: a real ``DecisionEngine``, so a pick perceives, ranks, decides and then grasps.
* ``closed_loop``: the pre-grasp refiner and the verifier. The verifier is fail-open here because
  the 2F-85 width signal is advisory. Refiner tuning differs per scene, so each runner passes its
  own ``refinement_kwargs``: an image-space IoU threshold for the vision scene, a standoff equal to
  the wrist view height for eye-in-hand.
* ``dense_clutter``: no extra sub-policy wiring. The profile switches the sampler to the dense
  point-cloud one and enables itself on a multi-object frame. Meaningful only on a clutter scene.
* ``dense_autonomous``: the full autonomous loop, with dense sampling, refine and verify. Recovery
  rides the existing recovery wiring.

Every heavy import is local, so this module imports where ``isaacsim`` is absent.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

#: The modes the single-object scenes can meaningfully run.
DEMO_MODES: tuple[str, ...] = ("easy", "auto", "closed_loop")
#: Modes the dense-clutter scene can run. ``dense_autonomous`` is the full loop on the static
#: overhead camera: perceive, refine, verify and, where wired, recover.
DENSE_MODES: tuple[str, ...] = ("easy", "auto", "closed_loop", "dense_clutter", "dense_autonomous")


def resolve_demo_mode(mode: Any) -> Any:
    """Map a demo-mode string (or a ``GraspMode``) to the typed :class:`GraspMode`.

    Accepts ``easy`` / ``auto`` / ``closed_loop`` / ``dense_clutter`` / ``dense_autonomous`` (and ``-``
    separators); raises ``ValueError`` for anything else so a typo never silently runs the wrong sampler.
    """

    from src.robot.execution.autonomous_grasp.config import GraspMode

    if isinstance(mode, GraspMode):
        return mode
    key = str(mode).strip().lower().replace("-", "_")
    table = {
        "easy": GraspMode.EASY,
        "auto": GraspMode.AUTO,
        "closed_loop": GraspMode.CLOSED_LOOP,
        "dense_clutter": GraspMode.DENSE_CLUTTER,
        "dense_autonomous": GraspMode.DENSE_AUTONOMOUS,
    }
    if key not in table:
        raise ValueError(
            f"unsupported demo mode {mode!r}; supported: {DENSE_MODES}"
        )
    return table[key]


def mode_service_kwargs(
    mode: Any,
    *,
    refinement_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return the ``from_components`` sub-policy kwargs that wire ``mode``.

    ``refinement_kwargs`` is merged into the CLOSED_LOOP ``RefinementPolicy``, so each runner passes
    its scene-specific tuning: ``target_match_iou_threshold`` for the vision scene, ``standoff_mm``
    for the eye-in-hand one.
    """

    from src.robot.execution.autonomous_grasp.config import GraspMode

    gm = resolve_demo_mode(mode)
    if gm is GraspMode.AUTO:
        from src.robot.grasping.decision import DecisionEngine, DecisionPolicy

        return {"decision_engine": DecisionEngine(policy=DecisionPolicy())}
    if gm is GraspMode.CLOSED_LOOP:
        from src.robot.grasping import (
            CompositeGraspVerifier,
            DefaultPreGraspRefiner,
            GraspVerificationPolicy,
            IoUCentroidTargetTracker,
            RefinementPolicy,
            WidthDeltaGripperVerifier,
            WorldSpacePoseTracker,
        )

        refine_policy = RefinementPolicy(enabled=True, **dict(refinement_kwargs or {}))
        verify_policy = GraspVerificationPolicy(enabled=True, fail_closed=False)
        # Select the viewpoint-invariant tracker when the policy opts in, through
        # ``refinement_kwargs`` ``use_world_space_tracker=True``. The default is the image-space one.
        tracker = (
            WorldSpacePoseTracker(
                max_pose_distance_mm=refine_policy.world_space_pose_distance_mm_threshold
            )
            if refine_policy.use_world_space_tracker
            else IoUCentroidTargetTracker()
        )
        return {
            "refinement_policy": refine_policy,
            "refiner": DefaultPreGraspRefiner(policy=refine_policy, tracker=tracker),
            "verification_policy": verify_policy,
            "verifier": CompositeGraspVerifier(verifiers=(WidthDeltaGripperVerifier(),)),
        }
    if gm is GraspMode.DENSE_AUTONOMOUS:
        # The full autonomous loop: dense-clutter sampling from the locked profile, then refine and
        # verify, with recovery riding the existing ``--recovery`` wiring. The overhead camera is
        # static, so the image-space ``IoUCentroidTargetTracker`` is the right tracker here and
        # ``WorldSpacePoseTracker`` belongs to the moving wrist camera. The verifier is fail-open
        # because the 2F-85 width signal is advisory.
        from src.robot.grasping import (
            CompositeGraspVerifier,
            DefaultPreGraspRefiner,
            GraspVerificationPolicy,
            IoUCentroidTargetTracker,
            RefinementPolicy,
            WidthDeltaGripperVerifier,
        )

        # reperceive=False holds the grasp on the initial overhead frame: a static camera cannot give
        # a second viewpoint, and the standoff move would occlude it or time out. The loop is pick,
        # refine-hold, verify, recover. A caller may override this through ``refinement_kwargs``.
        refine_policy = RefinementPolicy(
            enabled=True, reperceive=False, **dict(refinement_kwargs or {})
        )
        verify_policy = GraspVerificationPolicy(enabled=True, fail_closed=False)
        return {
            "refinement_policy": refine_policy,
            "refiner": DefaultPreGraspRefiner(policy=refine_policy, tracker=IoUCentroidTargetTracker()),
            "verification_policy": verify_policy,
            "verifier": CompositeGraspVerifier(verifiers=(WidthDeltaGripperVerifier(),)),
        }
    return {}


__all__ = ["DEMO_MODES", "resolve_demo_mode", "mode_service_kwargs"]
