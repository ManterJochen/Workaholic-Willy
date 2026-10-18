"""Operator-facing grasp-mode enum, per-mode behavior profiles, and the
effective-config snapshot.

The configuration value objects live apart from the service that consumes them.
Leaf of the package dependency graph: depends only on stdlib + ``grasping.types.modes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from src.robot.grasping.types.modes import GraspSamplingMode


class GraspMode(StrEnum):
    """The grasp mode a cell is built in: which sampler runs and which recovery actions are allowed.

    Attributes:
        EASY: One object, no recovery at all: a guarantee rather than a default.
        AUTO: The automatic sampler, with ``rescan``; ``robot.yaml``'s default.
        DENSE_CLUTTER: The dense sampler for a pile, with ``rescan`` and ``nudge_target`` (the push).
    """

    EASY = "easy"
    AUTO = "auto"
    DENSE_CLUTTER = "dense_clutter"


GraspModeInput = "GraspMode | str | None"

_GRASP_MODE_ALIASES: dict[str, GraspMode] = {
    "easy": GraspMode.EASY,
    "single": GraspMode.EASY,
    "single_object": GraspMode.EASY,
    "auto": GraspMode.AUTO,
    "dense": GraspMode.DENSE_CLUTTER,
    "dense_clutter": GraspMode.DENSE_CLUTTER,
}


def resolve_grasp_mode(value: GraspMode | str | None) -> GraspMode:
    """Coerce any accepted input form into a :class:`GraspMode`.

    Accepted inputs:

    * an existing :class:`GraspMode` (returned as-is),
    * :data:`None`, which maps to :attr:`GraspMode.AUTO`,
    * a string from the case-insensitive alias table above.

    A mode removed on purpose (``closed_loop``, ``dense_autonomous`` and the aliases they answered
    to) raises :class:`ValueError` with the sentence that names the mode to use instead. Anything
    else raises :class:`ValueError` listing the allowed forms.
    """

    if isinstance(value, GraspMode):
        return value
    if value is None:
        return GraspMode.AUTO
    if isinstance(value, str):
        key = value.strip().lower()
        mapped = _GRASP_MODE_ALIASES.get(key)
        if mapped is not None:
            return mapped
        # Imported here, on the refusal only: this module stays a leaf of the package graph.
        from src.config.schema._removed import REMOVED_GRASP_MODES  # noqa: PLC0415

        removed = REMOVED_GRASP_MODES.get(key)
        if removed is not None:
            raise ValueError(f"grasp mode {value!r} is removed on purpose: {removed}")
    allowed = "GraspMode enum, None, or one of " + ", ".join(
        sorted(set(_GRASP_MODE_ALIASES))
    )
    raise ValueError(
        f"Invalid grasp_mode value {value!r}; expected {allowed}."
    )


@dataclass(frozen=True, slots=True)
class GraspBehaviorProfile:
    """Locked behavior knobs for a given :class:`GraspMode`.

    All fields are read-only. Profiles are snapshot into
    :class:`AutonomousGraspReport` so a caller can always tell which
    flags were in effect for a specific pick, regardless of any later
    config changes.

    Attributes
    ----------
    mode
        The :class:`GraspMode` this profile represents.
    sampling_mode
        The :class:`GraspSamplingMode` label this profile's picks carry.
        The calculator runs one search whatever it says (2026-10-16).
    recovery_allowed_actions
        Allow-list of :class:`SceneRecoveryAction` names, held as
        strings so this module does not couple to a Protocol that does
        not exist yet. Empty for :attr:`GraspMode.EASY`: EASY must
        never produce recovery motion.

    ``refinement_enabled`` left on 2026-09-29 with the two-scan refinement, and
    ``verification_enabled`` the same day with the post-grasp verification stage;
    a record's profile block still carries both keys, written ``False``.
    """

    mode: GraspMode
    sampling_mode: GraspSamplingMode
    recovery_allowed_actions: tuple[str, ...] = ()


# Per-mode defaults. ``EASY`` always has zero recovery motion: that is a
# safety guarantee, not a knob. ``next_viewpoint`` left ``auto`` and ``dense_clutter`` on
# 2026-09-29, merged into ``rescan``, which both already allowed.
#
# ``nudge_target``, the push, stays in ``dense_clutter`` only, so auto recovery stays motion-free.
# ``next_target`` (a rescan that skipped the part that failed) and ``container_agitate`` left on
# 2026-10-16 (the owner's cleanup).
_PROFILES: Mapping[GraspMode, GraspBehaviorProfile] = {
    GraspMode.EASY: GraspBehaviorProfile(
        mode=GraspMode.EASY,
        sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        recovery_allowed_actions=(),
    ),
    GraspMode.AUTO: GraspBehaviorProfile(
        mode=GraspMode.AUTO,
        sampling_mode=GraspSamplingMode.AUTO,
        recovery_allowed_actions=("rescan",),
    ),
    # ``nudge_target`` moved here from ``dense_autonomous`` when that mode left on 2026-09-29 (owner's
    # decision), so the one physical recovery a built-in profile allowed stays reachable. The profile
    # is only the outer gate: the action still needs the cell's or the campaign's actions to name it.
    GraspMode.DENSE_CLUTTER: GraspBehaviorProfile(
        mode=GraspMode.DENSE_CLUTTER,
        sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        recovery_allowed_actions=("rescan", "nudge_target"),
    ),
}


#: The attempts a pick makes at most, each re-perceiving and trying again within one ``pick()`` call: the owner's
#: cell's, ``robot.grasping.max_attempts`` until 2026-10-17 (the owner's cleanup). A service built in code may ask
#: fewer (``AutonomousGraspService.from_robot_config(max_attempts=...)``).
MAX_ATTEMPTS = 5


def _profile_for(mode: GraspMode) -> GraspBehaviorProfile:
    """Return the locked profile for ``mode`` (a seam for future operator-supplied overrides)."""

    return _PROFILES[mode]


# The resolved ``robot.grasping`` overlay a cell was built with. ``decision``, ``feasibility``, ``occlusion``,
# ``ordering``, ``uncertainty``, ``success_model``, ``watchdog``, ``performance`` and ``approach_validation`` left on
# 2026-10-16 with their code (the owner's cleanup): switched on alone, none of them changed a pick of the replay bank's
# 479. :meth:`EffectiveGraspingConfig.to_dict` lost their flat keys with them.


@dataclass(frozen=True, slots=True)
class EffectiveRecoveryOrchestratorConfig:
    """Recovery-orchestrator overlay. All flags default off, which is what a service built from components carries.
    EASY mode is permanently excluded from the default ``apply_modes``. A cell built from config carries
    :data:`CELL_RECOVERY`: since 2026-10-16 no key sets it (the owner's cleanup)."""

    enabled: bool = False
    max_actions: int = 2
    apply_modes: tuple[str, ...] = (
        "auto",
        "dense_clutter",
    )
    allowed_actions: tuple[str, ...] = ()
    per_action_budget: tuple[tuple[str, int], ...] = ()
    #: How many of a blocker's grasps the clearing tries. Not in :meth:`EffectiveGraspingConfig.to_dict`.
    blocker_grasp_tries: int = 3
    #: Whether the parts are critical (``runtime.task.critical_parts``): a blocker is cleared and nothing is pushed. Not
    #: in :meth:`EffectiveGraspingConfig.to_dict`.
    critical_parts: bool = False
    #: Whether a blocker of a task that takes every part goes where the parts go. Not in
    #: :meth:`EffectiveGraspingConfig.to_dict`.
    blocker_into_the_place: bool = True


#: The recovery a cell built from config runs, fixed at the values the owner's cell ran (the owner, 2026-10-16): a
#: rescan once and a push twice a pick, at most two actions, three tries at a blocker's grasps and the blocker into the
#: place, in ``auto`` and ``dense_clutter``. The mode's own profile is the outer gate: ``easy`` allows no action, and
#: only ``dense_clutter`` the push. Whether the parts are critical is the cell's ``runtime.task.critical_parts``.
CELL_RECOVERY = EffectiveRecoveryOrchestratorConfig(
    enabled=True,
    max_actions=2,
    apply_modes=("auto", "dense_clutter"),
    allowed_actions=("rescan", "nudge_target"),
    per_action_budget=(("rescan", 1), ("nudge_target", 2)),
    blocker_grasp_tries=3,
    blocker_into_the_place=True,
)


@dataclass(frozen=True, slots=True)
class EffectiveGraspingConfig:
    """Snapshot of the ``robot.grasping`` block resolved at construction.

    The snapshot captures exactly the values the runtime used to wire
    the service, after merging:

    * the operator-supplied ``RobotConfig.grasping`` sub-tree, and
    * any per-call kwargs that overrode the config (``mode``,
      ``max_attempts``).

    Stored on :class:`AutonomousGraspService` (as
    :attr:`AutonomousGraspService.effective_config`) and threaded into
    every :class:`AutonomousGraspReport` so a downstream debugger can
    correlate behaviour to configuration without re-loading YAML.
    The :class:`GraspBehaviorProfile` already captures the locked
    per-mode toggles; this snapshot adds the config-derived overlay
    on top.

    Attributes
    ----------
    default_mode
        The :class:`GraspMode` that the service will use when
        :meth:`AutonomousGraspService.pick` is called without an
        explicit ``mode`` override.
    max_attempts
        Resolved upper bound on autonomous grasp attempts per
        :meth:`AutonomousGraspService.pick` call.

    ``closed_loop_enabled`` left on 2026-09-29 with the two-scan refinement, and
    ``verification_enabled``, ``dense_recovery_enabled`` and
    ``dense_recovery_allowed_actions`` the same day with the
    ``robot.grasping.verification`` and ``robot.grasping.dense_recovery`` blocks they
    reported, each with its flat key.
    """

    default_mode: GraspMode
    max_attempts: int
    #: Whether the cell fuses its other cameras' views: its camera section declares two or more enabled, calibrated
    #: RGB-D rigs (``camera_fusion.fused_cameras``). Whether another camera's view was fused depends on the rig
    #: delivering, a runtime fact this snapshot cannot see.
    fusion_enabled: bool = False
    recovery_orchestrator: EffectiveRecoveryOrchestratorConfig = field(
        default_factory=EffectiveRecoveryOrchestratorConfig
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict copy of the snapshot."""

        return {
            "default_mode": self.default_mode.value,
            "max_attempts": int(self.max_attempts),
            "recovery_orchestrator_enabled": bool(self.recovery_orchestrator.enabled),
            "recovery_orchestrator_max_actions": int(self.recovery_orchestrator.max_actions),
            "recovery_orchestrator_apply_modes": list(self.recovery_orchestrator.apply_modes),
            "recovery_orchestrator_allowed_actions": list(self.recovery_orchestrator.allowed_actions),
            "recovery_orchestrator_per_action_budget": [
                [str(action), int(count)]
                for action, count in self.recovery_orchestrator.per_action_budget
            ],
            "fusion_enabled": bool(self.fusion_enabled),
        }
