"""How much a config field is anyone's business: derived, never hand-maintained.

The config accepts 545 fields. Almost nobody needs almost any of them, and a listing that treats them
all alike buries the handful that matter: ``where gripper`` returns 34 hits with no hint that two are
cell facts and thirty are tuning behind a switch nobody turned on.

So each field carries a tier. Two rules bound what a tier is allowed to be:

  1. A tier is a display filter, never a gate. Every field stays settable, ``extra='forbid'`` keeps
     accepting it, and hiding it from a listing removes nothing. Dropping a value from a file removes
     nothing; dropping a field removes a capability, and a tier must never be mistaken for that.
  2. A required field is never demoted. 63 fields have no default at all, and treating one as long
     tail invites defaulting it later, which trades a hard load failure for a silent value in the
     fail-open direction.

A tier is derived from what the schema and the loaded cell already say, not declared per field: a
hand-maintained table of 545 entries drifts the moment a block's ``enabled`` flips, and would have
to be re-blessed into the schema golden. Derivation cannot drift, and costs the goldens nothing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TIERS", "tier_for"]

#: Ordered widest-first: a listing shows everything up to and including the requested tier.
TIERS: tuple[str, ...] = ("safety", "site", "tuned", "advanced")

#: Fail-closed switches that sit outside ``robot.safety`` and still decide whether a guard refuses
#: on doubt. Named one by one rather than pattern-matched: a rule like "anything called fail_closed"
#: stops matching after a rename, without saying so.
_SAFETY_ELSEWHERE = frozenset({
    "robot.grasping.decision.fail_closed_on_real_hardware",
    "robot.grasping.uncertainty.fail_closed_threshold",
})


def tier_for(
    path: str,
    *,
    required: bool = False,
    decided: bool = False,
    gated_off: bool = False,
) -> str:
    """The tier of one field.

    ``decided`` means a shipped YAML sets the field to something other than its default, so someone
    made a choice about this cell. ``gated_off`` means it sits under a block whose ``enabled`` is
    false in the configuration being viewed, so it cannot affect anything until that switch is
    thrown.

    The order is deliberate: safety outranks everything, because a safety bound behind a disabled
    block is still a safety bound and must never be hidden; then anything that must be stated or was
    decided; then the switched-off long tail; then the rest.
    """
    if path.startswith("robot.safety.") or path in _SAFETY_ELSEWHERE:
        return "safety"
    if required or decided:
        return "site"
    if gated_off:
        return "advanced"
    return "tuned"


def gate_state(path: str, index: dict[str, Any], values: dict[str, Any] | None) -> bool:
    """Is ``path`` under a block whose ``enabled`` switch is off?

    Walks the ancestors and asks each for its ``enabled`` sibling. ``values`` is the flattened loaded
    config when there is one, so the answer follows the chain actually being viewed: the sim block
    defaults to ``enabled: false`` and the ``sim`` layer turns it on, so calling its fields
    "advanced" while looking at a sim cell would be backwards. Without a loaded config, as in the
    schema-only ``where``, the schema default decides, which reads as advanced unless a cell enables
    it.
    """
    parts = path.split(".")
    for i in range(1, len(parts)):
        gate = ".".join(parts[:i]) + ".enabled"
        if gate == path or gate not in index:
            continue
        if values is not None and gate in values:
            if values[gate] is False:
                return True
            continue
        if index[gate].default is False:
            return True
    return False
