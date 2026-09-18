"""What a sidecar started on an arm descriptor, with the hand added as a body link, says of itself (UM lane S12, S22).

For tests whose cuRobo client is a stub. A planner checks this identity when it starts: a descriptor built for this arm
that carries no hand, a body link named ``hand``, and since S22 the three hashes a committed evidence file measured for
the combination. A stub that said nothing would be refused before its world was ever asked, so every stub reports this,
and a test that wants a refusal changes exactly one field of it.

⛔ The hashes are READ from the committed file for the combination, never typed here. A stub carrying invented hashes
would pass the evidence check on a tree where the measurement was never run, which is the one thing the check exists
to catch.
"""

from __future__ import annotations

import json

from src.contracts import UNSET
from src.robot.safety.planning.curobo_client import SidecarIdentity
from src.robot.safety.planning.evidence import evidence_path


def arm_identity(
    arm: str = "ur5e",
    *,
    hand: str = "robotiq_2f85",
    coupling_mm: float = 0.0,
    approach: str = "+Y",
    closing: str = "+X",
    planner_margin_mm: float = 4.0,
    attach_spheres: int = 0,
    bodies: tuple[str, ...] = ("hand",),
) -> SidecarIdentity:
    """The identity of ``willy_{arm}.yml`` with ``bodies`` added, carrying the hashes the committed file measured.

    Where no file measured the combination the hashes stay UNSET, which is what a sidecar nobody measured would give an
    evidence check to refuse.
    """
    path = evidence_path(arm=arm, hand=hand, coupling_mm=coupling_mm, approach=approach, closing=closing,
                         planner_margin_mm=planner_margin_mm, attach_spheres=attach_spheres)
    held = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return SidecarIdentity(
        provenance={"arm": arm, "carries_hand": False, "generated_by": "scripts/curobo/build_ur_config.py"},
        arm_descriptor_sha256=held.get("arm_descriptor_sha256", UNSET),
        urdf_sha256=held.get("urdf_sha256", UNSET),
        composed_sha256=held.get("composed_sha256", UNSET),
        bodies=tuple({"link": name, "parent": "tool0"} for name in bodies),
    )
