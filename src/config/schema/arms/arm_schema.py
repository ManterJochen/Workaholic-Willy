"""One arm, described: the joints it has, what each one may do, and what a simulator needs to stand it up.

The third hardware registry beside ``grippers/`` and ``cameras/`` (aurora's new-vendors plan, section 4.2), which
replaced ten hand-kept tables in six modules (guide 04, section 2.4, lists them).

One row per joint, not four parallel lists. The tables this replaces were parallel by construction: a DH
table with six rows beside a limit table with two six-tuples beside a name tuple, and nothing but a test holding
their lengths together. Here a joint is one object carrying its geometry, its envelope and its home angle, so a
model with five names and six DH rows cannot be written down at all, and ``dof`` is ``len(joints)`` rather than
a second number that can disagree with the first.

A description is not an admission: which of the eight a cell may name is each one's ``admitted`` field, and
``src/config/arms.py`` says why two are described and not admitted.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from .._base import StrictModel
from ..grippers import MODEL_NAME_PATTERN

__all__ = ["ArmBaseSpec", "ArmFlangeSpec", "ArmJointSpec", "ArmPlannerSpec", "ArmSimSpec", "ArmSpec",
           "ArmUpstreamSpec"]

#: A bundle part: ``__`` separates a part from its array (``<part>__v``) in a bundle, so a part may not hold one.
_PART_PATTERN = r"^[a-z0-9]+(?:_[a-z0-9]+)*$"
#: The parts every arm bundle also carries for the hand (``safety/planning/_hand_bundle.HAND_PARTS``, which this
#: layer cannot import; ``tests/repo/test_inlined_link_tables.py`` ties the two).
_HAND_PARTS = ("gripper", "lfinger", "rfinger")

#: Nothing on any arm this stack drives is longer than this, in metres. A bound, not a measurement: it catches a
#: millimetre written where a metre belongs, which is the one DH typo that still builds a plausible robot.
_MAX_LINK_M = 2.0


class ArmJointSpec(StrictModel):
    """One joint: where it puts the link after it, how far it may turn, and where it rests.

    Metres, not millimetres: the millimetre round trip is not exact for one committed length, so the unit is in
    every field name (guide 04, section 2.4, has the measurement).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    #: The joint's name in the description this arm came from, in the order a joint vector arrives in. A
    #: remapped joint vector is still a valid joint vector, so this is the one field whose error is silent.
    name: str = Field(min_length=1)
    #: The link this joint moves, its child link as the arm's description spells it. It is rigid with this joint's
    #: frame, which is the row's number, 1 to ``dof``: the frame its bundle arrays are baked in. So no frame is
    #: written, and none can disagree with the row (P0f F6c).
    link: str = Field(pattern=MODEL_NAME_PATTERN)
    #: The bundle part that link's arrays are baked under: ``<part>__v``, ``__f`` and ``__frame`` in
    #: ``safety/data/{model}_collision_meshes.npz``, and the body a sphere map fits.
    part: str = Field(pattern=_PART_PATTERN)
    #: Standard (Craig-style) DH: the link length, the offset along the previous z, and the twist.
    a_m: float = Field(ge=-_MAX_LINK_M, le=_MAX_LINK_M)
    d_m: float = Field(ge=-_MAX_LINK_M, le=_MAX_LINK_M)
    alpha_rad: float = Field(ge=-2.0 * math.pi, le=2.0 * math.pi)
    #: The envelope the self-collision guard enforces, degrees, low then high: the arm's own, not UR's planning
    #: file, which is ``planner_limit_deg`` (guide 04, section 2.4).
    limit_deg: tuple[float, float] = Field()
    #: The envelope the trajectory planner is given, degrees, before its clip (P0f F6f). Inside ``limit_deg``, or the
    #: planner would propose what the guard refuses; UR's elbow is +/-180 here, its own planning file's number.
    planner_limit_deg: tuple[float, float] = Field()
    #: Where this joint rests when a cell is sent home, in radians.
    home_rad: float = Field(ge=-4.0 * math.pi, le=4.0 * math.pi)
    #: Where the retract search starts for this joint, radians: Isaac's Lula ``default_q``. All joints or none, with
    #: ``retract_anchor_source`` on the arm (P0f F6g); compared by ``float.hex``, so ``0.0`` and never ``-0.0``.
    retract_anchor_rad: float | None = Field(default=None, ge=-4.0 * math.pi, le=4.0 * math.pi)

    @model_validator(mode="after")
    def _home_is_inside_the_envelope(self) -> "ArmJointSpec":
        low, high = self.limit_deg
        if low > high:
            raise ValueError(
                f"joint {self.name!r} has limit_deg {self.limit_deg}, which is empty: the low end comes first"
            )
        home_deg = math.degrees(self.home_rad)
        if not low <= home_deg <= high:
            raise ValueError(
                f"joint {self.name!r} rests at {home_deg:g} degrees, outside its own envelope {self.limit_deg}: "
                f"a cell sent home would be refused by the guard before it moved"
            )
        anchor = self.retract_anchor_rad
        if anchor is not None and not low <= math.degrees(anchor) <= high:
            raise ValueError(f"joint {self.name!r} has its retract anchor at {math.degrees(anchor):g} degrees, "
                             f"outside its own envelope {self.limit_deg}")
        return self

    @model_validator(mode="after")
    def _the_planner_stays_inside_the_guard(self) -> "ArmJointSpec":
        """Defined after the envelope's own check, so an empty ``limit_deg`` is refused in that one's words first."""
        low, high = self.planner_limit_deg
        if not low < high:
            raise ValueError(f"joint {self.name!r} has planner_limit_deg {self.planner_limit_deg}, which is empty")
        if low < self.limit_deg[0] or high > self.limit_deg[1]:
            raise ValueError(
                f"joint {self.name!r} gives the planner {self.planner_limit_deg}, wider than the guard's "
                f"{self.limit_deg}: the planner would propose what the guard refuses"
            )
        return self


class ArmPlannerSpec(StrictModel):
    """What the trajectory planner does to this arm's joints on top of ``planner_limit_deg`` (P0f F6f)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    #: cuRobo narrows each end of every joint's envelope by this when it loads a descriptor, radians, under its own
    #: key's name. In radians because a clip stored in degrees does not come back to 0.1 exactly.
    position_limit_clip_rad: float = Field(ge=0.0, le=math.pi / 2)


class ArmBaseSpec(StrictModel):
    """The end the arm stands on, as ``flange`` is the end a hand hangs from (P0f F6d).

    ``link`` is the description's root, where the URDF and the planner's descriptor start. ``chain_yaw_deg`` turns
    the frame the joints start in about z from it, ``controller_yaw_deg`` the frame the controller reports poses in:
    UR's ``ur_macro.xacro`` puts both ``base_link_inertia`` and ``base`` half a turn from ``base_link``. Two facts
    that agree on UR and need not elsewhere. 0 or 180 and nothing else: each reader has its own exact expression.
    """

    link: str = Field(pattern=MODEL_NAME_PATTERN)
    chain_yaw_deg: Literal[0, 180]
    controller_yaw_deg: Literal[0, 180]


class ArmUpstreamSpec(StrictModel):
    """The pin the derived half of this description was read from and is checked against (P0f F6b).

    The build adapter owns the bytes: ``scripts/arms/fetch.py`` fetches the repository at the commit and its manifest
    holds every file's digest. This states the same pin as data, so a test holds every description to the adapter and
    a pin bump that forgets one arm fails by name. It stores no path, because the adapter owns the paths, and it
    counts no joints, so a description from another vendor's package fills the same five fields.
    """

    #: The build adapter that reads this upstream: ``scripts/arms/_urdf_source.SOURCES``.
    adapter: Literal["ur"]
    #: What git fetches, over https.
    repo: str = Field(pattern=r"^https://\S+$")
    #: The full commit, because a branch or a tag moves.
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    #: One SPDX identifier, not prose.
    licence: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
    #: sha256 over the adapter manifest's rows, ``<digest>  <path>\n`` sorted by path, the header left out, so a
    #: reworded header or a CRLF checkout moves nothing while any pinned file's digest does.
    manifest_rows_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArmSimSpec(StrictModel):
    """What a simulator needs to stand this arm up. Absent on an arm no simulator here can build.

    Separate from the arm itself because it is the one part that is about an asset on a particular box rather
    than about the robot: ur16e has a full set of real-cell geometry and no Isaac asset at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    #: Isaac's own name for the arm, used to find its Lula kinematics config.
    lula_key: str = Field(min_length=1)
    #: The asset, relative to the Isaac assets root.
    usd_relpath: str = Field(min_length=1)
    #: The link the wrist camera and the tool hang from, relative to the robot root prim. A field rather than a
    #: constant because it is the first thing a non-UR arm changes, and a wrong guess mounts the camera on the
    #: wrong body without erroring: the image just moves with something else.
    wrist_link: str = Field(min_length=1)
    #: The ``Gripper`` USD variant to select on this asset, and the registry name of the hand it selects. Both
    #: or neither: an asset that offers no Robotiq bakes no hand, and the sim mounts the cell's hand standalone.
    baked_gripper_variant: str | None = None
    baked_hand: str | None = None
    #: The patch of table a scene generator may place objects on, in mm, for a cell standing this arm.
    workspace_center_mm: tuple[float, float]
    workspace_half_extents_mm: tuple[float, float]

    @model_validator(mode="after")
    def _a_baked_variant_names_the_hand_it_bakes(self) -> "ArmSimSpec":
        if (self.baked_gripper_variant is None) != (self.baked_hand is None):
            raise ValueError(
                f"baked_gripper_variant={self.baked_gripper_variant!r} and baked_hand={self.baked_hand!r}: an "
                f"asset that offers a variant names the hand that variant is, and one that offers none names "
                f"no hand, because the sim decides from the pair whether to select the variant or mount a hand "
                f"standalone"
            )
        if any(extent <= 0.0 for extent in self.workspace_half_extents_mm):
            raise ValueError(
                f"workspace_half_extents_mm={self.workspace_half_extents_mm} has no area, so no object could be "
                f"placed on this arm's patch"
            )
        return self


class ArmFlangeSpec(StrictModel):
    """Where a hand hangs on this arm, and which way a hand bolted there approaches (P0e E12, aurora's owner decision 7).

    ``link`` is the frame the planner hangs the hand from and the DH chain ends in, ``tool0`` on a UR. ``approach``
    is the axis a hand bolted there approaches along, UR's ``Z+ front``; the desk warns on a tool frame that differs.
    """

    link: str = Field(pattern=MODEL_NAME_PATTERN)
    approach: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


class ArmSpec(StrictModel):
    """One arm description, as ``config/arms/<model>.yaml`` holds it."""

    #: The driver family that drives this arm. Read with the model, because a limit table is resolved by the
    #: pair (``safety/joint_limits.py``) and a capability record carries both.
    vendor: str = Field(min_length=1, pattern=MODEL_NAME_PATTERN)
    #: The registry name, which is also the file's stem, so a name always finds exactly one file.
    model: str = Field(pattern=MODEL_NAME_PATTERN)
    #: Where these numbers came from, as a sentence a person reads. Free prose, as the hand and camera registries
    #: write it: the pinned upstream for the derived half, and who measured what for the rest.
    source: str = Field(min_length=1)
    #: The pin the derived half was read from, as data. Required, like ``admitted`` and ``flange``: a description
    #: that says nothing about where its joints came from cannot be held to it.
    upstream: ArmUpstreamSpec
    #: Whether a cell may name this arm. Required, a safety identity, and a decision per arm rather than derived:
    #: ur16e carries every artefact a real cell reads and is still not admitted (the owner, in aurora, 2026-09-22).
    admitted: bool
    #: Base to flange, one row per joint, in the order a joint vector arrives in.
    joints: tuple[ArmJointSpec, ...] = Field(min_length=1)
    #: The root the joints hang from and the half turn to where the DH chain starts. Required: a default of 0 would
    #: read every older copy and every new vendor as unturned, the mirrored plan Step 8f fixed.
    base: ArmBaseSpec
    #: The link a hand hangs from and the axis it approaches along. Required: a guess here places every hand.
    flange: ArmFlangeSpec
    #: UR's published working radius in mm, the number on the datasheet, not the sum of every link length and offset
    #: (each arm file says why). ``None`` where this repository has recorded none: UR's pinned files hold neither.
    reach_mm: float | None = Field(default=None, gt=0.0, le=5000.0)
    #: The rated payload in kg, or ``None`` on the same terms.
    payload_kg: float | None = Field(default=None, gt=0.0, le=1000.0)
    #: What a simulator needs, or ``None`` for an arm no simulator here can build.
    sim: ArmSimSpec | None = None
    #: The planner's clip. Required, like the envelope it narrows.
    planner: ArmPlannerSpec
    #: Where the joints' ``retract_anchor_rad`` came from, word for word as every retract row of this arm records it.
    retract_anchor_source: str | None = Field(default=None, min_length=1, pattern=r"^[^\n]+$")

    @property
    def dof(self) -> int:
        """How many joints a vector for this arm carries. Derived, so it cannot disagree with the joints."""
        return len(self.joints)

    @property
    def collision_links(self) -> tuple[tuple[str, str, int], ...]:
        """``(part, link, frame)`` base to flange. The frame is the row's number, and the flange frame is ``dof``."""
        return tuple((joint.part, joint.link, frame) for frame, joint in enumerate(self.joints, start=1))

    @model_validator(mode="after")
    def _an_arm_a_simulator_can_stand_up_has_a_reach(self) -> "ArmSpec":
        """The sim's workspace patch is derived from the reach, so an arm the sim stands up states its reach."""
        if self.sim is not None and (self.reach_mm is None or self.payload_kg is None):
            raise ValueError(
                f"{self.model} carries a sim block, so it stands up in a simulator, but "
                f"reach_mm={self.reach_mm!r} and payload_kg={self.payload_kg!r}: a scene generator sizes its "
                f"patch against the reach, and a cell refuses a part heavier than the payload"
            )
        return self

    @model_validator(mode="after")
    def _the_adapter_reads_this_vendor(self) -> "ArmSpec":
        """The ``ur`` adapter reads Universal Robots' package layout, which no other vendor's description has."""
        if self.upstream.adapter == "ur" and self.vendor != "ur":
            raise ValueError(
                f"{self.model} is a {self.vendor!r} arm and names upstream adapter 'ur', which reads Universal Robots' "
                f"description package and nothing else"
            )
        return self

    @model_validator(mode="after")
    def _an_anchor_is_whole_and_says_where_it_came_from(self) -> "ArmSpec":
        missing = [joint.name for joint in self.joints if joint.retract_anchor_rad is None]
        if missing and len(missing) < len(self.joints):
            raise ValueError(f"{self.model} anchors its retract search on some joints and not on {', '.join(missing)}: "
                             f"an anchor is a whole pose")
        if (not missing) != (self.retract_anchor_source is not None):
            raise ValueError(f"{self.model}: retract_anchor_source is stated exactly when every joint has an anchor")
        return self

    @model_validator(mode="after")
    def _the_clip_leaves_every_joint_a_range(self) -> "ArmSpec":
        clip = self.planner.position_limit_clip_rad
        shut = [joint.name for joint in self.joints
                if math.radians(joint.planner_limit_deg[1]) - math.radians(joint.planner_limit_deg[0]) <= 2.0 * clip]
        if shut:
            raise ValueError(f"{self.model}'s planner clip of {clip} rad closes the range of {', '.join(shut)}: "
                             f"cuRobo narrows each end of every joint by it")
        return self

    @model_validator(mode="after")
    def _the_joints_are_distinct(self) -> "ArmSpec":
        names = [joint.name for joint in self.joints]
        if len(set(names)) != len(names):
            raise ValueError(
                f"{self.model} names a joint twice ({', '.join(names)}): a joint vector is matched to a "
                f"controller's ordering by these names, and a repeated one matches the wrong axis in silence"
            )
        return self

    @model_validator(mode="after")
    def _each_joint_moves_its_own_link(self) -> "ArmSpec":
        """Defined after ``_the_joints_are_distinct``, so a repeated joint is refused in that one's words first."""
        links = [joint.link for joint in self.joints]
        parts = [joint.part for joint in self.joints]
        if len(set(links)) != len(links) or len(set(parts)) != len(parts):
            raise ValueError(
                f"{self.model} names a link or a part twice (links {', '.join(links)}; parts {', '.join(parts)}): "
                f"each joint moves its own link, and a bundle holds one set of arrays per part"
            )
        hand = sorted(set(parts) & set(_HAND_PARTS))
        if hand:
            raise ValueError(
                f"{self.model} names the arm part {', '.join(hand)}, which every bundle keeps for the hand "
                f"({', '.join(_HAND_PARTS)})"
            )
        if self.base.link in links or self.base.link == self.flange.link:
            raise ValueError(
                f"{self.model}'s base link {self.base.link!r} is a joint's link or the flange: the base is the root "
                f"the joints hang from"
            )
        if self.flange.link in links[:-1]:
            raise ValueError(
                f"{self.model}'s flange link {self.flange.link!r} is a joint's link short of the last: the flange is "
                f"frame {self.dof}, so only the last joint's link may be it"
            )
        return self
