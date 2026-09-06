"""RL optimisation extension layer (mode-contract package).

This package hosts the offline RL stack: the dataset builder, the five
policy trainers, OPE, the promotion gate, the paired-soak and rollback
harnesses, the shadow and canary routers, and the five policy loaders.
Together they optionally optimise candidate selection, ranking,
sequencing, active perception budget, and recovery selection on top of
the deterministic grasping stack. This module is the contract surface:

* the public mode contract (re-exported from
  :mod:`src.config.schema.robot.robot_schema`),
* :class:`RLModeNotImplementedError`, the typed runtime rejection
  raised when an RL-active mode is selected before its producing
  policy lands (candidate-selection onward),
* :func:`assert_rl_mode_supported`, the runtime gate that consumes
  the schema-validated mode value and decides whether to admit it.

The runtime admits ``geometry_only``, ``hybrid_ml`` and ``rl_shadow``
only; ``rl_active`` and ``rl_experimental`` have no shipped producer
and are rejected (see :data:`RL_SUPPORTED_MODES`).

Authority hierarchy (RL-layer hard lock, do not weaken):

1. ``robot.safety``
2. hardware / runtime constraints
3. deterministic geometry pipeline
4. deterministic recovery constraints
5. ML / RL optimisation layers (this package)

The RL layer must never override a higher-authority layer.

Production learning restrictions (RL-layer hard lock, do not weaken):

* no online policy-weight updates,
* no in-process parameter mutation,
* no reward self-modification,
* no safety / runtime bypass,
* no telemetry suppression,
* no fallback suppression,
* no action-bound widening without explicit operator approval.

All policy updates must travel one path, in this order: offline
training, artifact promotion, shadow, canary, active.
"""

from __future__ import annotations

from src.config.schema.robot.robot_schema import (
    RL_ACTIVE_MODES,
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_HYBRID_ML,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
    RL_MODE_RL_SHADOW,
    RL_MODE_VALUES,
)


#: The newest mode the runtime admits. It admits the deterministic modes plus this one; anything
#: past it (``rl_active`` / ``rl_experimental``) is offline / not-yet-shipped and is rejected by
#: :func:`assert_rl_mode_supported`. An introspectable label with no runtime consumer: gating uses
#: :data:`RL_SUPPORTED_MODES`, and the admission error builds its message from that set.
RL_NEWEST_RUNTIME_MODE: str = RL_MODE_RL_SHADOW

#: Version of the RL offline tooling (dataset builder, trainers, OPE, the promotion gate, and the
#: soak + rollback harnesses) shipped in this package. Stamped into dataset manifests as the tooling
#: version that produced them; bump it when a new tooling capability lands.
RL_TOOLING_VERSION: str = "1.0"

#: The deterministic-only modes: no RL policy object is ever constructed for these. A named subset
#: so the deterministic contract is introspectable on its own; combined with ``rl_shadow`` it forms
#: :data:`RL_SUPPORTED_MODES_WITH_SHADOW`.
RL_SUPPORTED_MODES_DETERMINISTIC: frozenset[str] = frozenset(
    {RL_MODE_GEOMETRY_ONLY, RL_MODE_HYBRID_ML}
)

#: The deterministic modes plus ``rl_shadow`` (log-only shadow routing; the deterministic stack is
#: unchanged when it is selected). This is the full set the runtime admits.
RL_SUPPORTED_MODES_WITH_SHADOW: frozenset[str] = (
    RL_SUPPORTED_MODES_DETERMINISTIC | {RL_MODE_RL_SHADOW}
)

#: The canonical supported set (the same object as :data:`RL_SUPPORTED_MODES_WITH_SHADOW`).
#: Downstream code reads this one rather than the named subsets unless it needs the
#: deterministic-only surface.
RL_SUPPORTED_MODES: frozenset[str] = RL_SUPPORTED_MODES_WITH_SHADOW


class RLModeNotImplementedError(RuntimeError):
    """Raised when an RL-active mode is selected before its producer ships.

    Carries the requested mode and a plain-language description of the producer that will introduce
    it, so operator log messages are self-explanatory.
    """

    def __init__(self, mode: str, *, producer: str) -> None:
        self.mode = mode
        self.producer = producer
        super().__init__(
            f"robot.rl.mode={mode!r} is schema-valid but its producer ({producer}) has not "
            f"shipped; the runtime admits {sorted(RL_SUPPORTED_MODES)!r}."
        )


#: What each RL-active mode with no shipped producer is waiting on, in plain language, for the
#: actionable admission-error message. Consumed by :func:`assert_rl_mode_supported`, which returns
#: for a mode already in :data:`RL_SUPPORTED_MODES` before reaching this lookup, so only the
#: ``rl_active`` and ``rl_experimental`` entries are reachable.
RL_MODE_PRODUCER: dict[str, str] = {
    RL_MODE_RL_SHADOW: "the log-only shadow router",
    RL_MODE_RL_ACTIVE: "bounded online->canary active control",
    RL_MODE_RL_EXPERIMENTAL: "the experimental policy lane",
}


def assert_rl_mode_supported(mode: str) -> None:
    """Admission check for a configured ``robot.rl.mode`` (the single typed gate).

    Called by ``execution.autonomous_grasp.shadow.maybe_build_shadow_router``
    at boot, so a config that requests an RL-active mode whose producer has not
    shipped (``rl_active`` / ``rl_experimental``) fails loudly here instead of
    degrading to a silent no-op.

    Schema validation guarantees ``mode`` is one of the five locked
    enum values. This helper additionally enforces that the runtime
    has a shipped producer for the requested mode (see
    :data:`RL_SUPPORTED_MODES`).

    Raises:
        RLModeNotImplementedError: when ``mode`` is RL-active but its
            producer has not yet shipped.
        ValueError: when ``mode`` is not one of the locked enum
            values (defence-in-depth; should be unreachable when the
            value comes from :class:`RobotRLConfig`).
    """

    if mode not in RL_MODE_VALUES:
        raise ValueError(
            f"unknown robot.rl.mode={mode!r}; "
            f"expected one of {sorted(RL_MODE_VALUES)!r}"
        )
    if mode in RL_SUPPORTED_MODES:
        return
    producer = RL_MODE_PRODUCER.get(mode, "an unshipped producer")
    raise RLModeNotImplementedError(mode, producer=producer)


__all__ = (
    "RL_ACTIVE_MODES",
    "RL_MODE_GEOMETRY_ONLY",
    "RL_MODE_HYBRID_ML",
    "RL_MODE_PRODUCER",
    "RL_MODE_RL_ACTIVE",
    "RL_MODE_RL_EXPERIMENTAL",
    "RL_MODE_RL_SHADOW",
    "RL_MODE_VALUES",
    "RL_NEWEST_RUNTIME_MODE",
    "RL_SUPPORTED_MODES",
    "RL_SUPPORTED_MODES_DETERMINISTIC",
    "RL_SUPPORTED_MODES_WITH_SHADOW",
    "RL_TOOLING_VERSION",
    "RLModeNotImplementedError",
    "assert_rl_mode_supported",
)
