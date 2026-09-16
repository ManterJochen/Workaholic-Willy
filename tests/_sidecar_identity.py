"""What a sidecar started on an arm descriptor, with the hand added as a body link, says of itself (UM lane S12).

For tests whose cuRobo client is a stub. A planner checks this identity when it starts: a descriptor built for this arm
that carries no hand, and a body link named ``hand``. A stub that said nothing would be refused before its world was
ever asked, so every stub reports this, and a test that wants a refusal changes exactly one field of it.
"""

from __future__ import annotations

from src.robot.safety.planning.curobo_client import SidecarIdentity


def arm_identity(arm: str = "ur5e", *, bodies: tuple[str, ...] = ("hand",)) -> SidecarIdentity:
    """The identity of ``willy_{arm}.yml`` with ``bodies`` added, hashes unreported."""
    return SidecarIdentity(
        provenance={"arm": arm, "carries_hand": False, "generated_by": "scripts/curobo/build_ur_config.py"},
        bodies=tuple({"link": name, "parent": "tool0"} for name in bodies),
    )
