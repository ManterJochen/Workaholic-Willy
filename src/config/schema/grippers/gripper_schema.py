"""One hand, described once.

A registry file holds, for one hand, the numbers the labeller's ``JawModel``
(``datagen/grasps/verdict.py``), the network's ``JAW_GEOMETRY``
(``src/robot/grasping/deep/net/gripper.py``), the runtime envelope
``grasping.gripper_geometry.parallel_jaw`` and the actuation widths in ``robot.gripper`` state, and
``robot.gripper.model`` names the file. The loader fills the widths and the envelope from it
(``src/config/hand_numbers.py``), and the guard, the planner bundles, the deep calculator and the sim's
mount take the hand from it. The sim gripper profile does not: it opens the 2F-85 to 87.1 mm where the
registry file says 85.0.

The vocabulary is the grasp frame the envelope uses: X closes between the contacts, Y is the binormal, Z
is the approach, and z = 0 is the grasp centre.

Two reach numbers stay two numbers. ``finger_behind_mm`` is the measured reach of the finger material
back toward the wrist, which the labeller and the network use; ``finger_length_mm`` is the conservative
length the collision envelope sweeps behind the centre. For the 2F-85 they are 33.37 and 39.98 mm, and
folding them into one would move either the labels or the envelope.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .._base import StrictModel

__all__ = ["MODEL_NAME_PATTERN", "GripperSpec", "ParallelJawSpec"]

#: A registry name: lower case letters, digits and underscores. No dot, so it can never be read as a
#: profile overlay, and no separator, so it can never name a file outside the registry directory.
MODEL_NAME_PATTERN = r"^[a-z0-9_]+$"

#: How far the three pad lengths may disagree. They are measurements written as decimals, so they have
#: to add up to the written precision, not to float equality.
_PAD_TOLERANCE_MM = 1e-6


class ParallelJawSpec(StrictModel):
    """A two-finger parallel jaw, in millimetres, in the grasp frame.

    The upper bounds on the fields the runtime collision envelope carries are that schema's own
    (``GraspingParallelJawGeometryConfig``), and the widths relate to the aperture the way the actuation
    block (``robot.gripper``) requires, so a registry file cannot hold a number a consumer built from it
    would refuse. The relations below say that the numbers describe one finger: the pad sits on it, and
    the envelope's length is at least the reach it was measured to have.

    Every number is finite. A field without an upper bound would otherwise take ``.inf``: an infinite
    aperture passes every width relation, and an infinite friction passes every antipodal check a
    labeller reading this file makes.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    #: How far the grasp centre sits from the hand's own origin along the approach: the flange for a
    #: hand baked off a mounted asset, the hand's own mounting face for a standalone one, the same
    #: origin the collision bundle declares. Every other number in this block is relative to that
    #: centre, so without it the block describes a shape and not where the shape sits, and a hand
    #: nobody baked cannot be placed on an arm at all. Measured off the bundle by
    #: scripts/grippers/measure_jaw_from_bundle.py. The desk reads it too: the ``grasp centre``
    #: preflight row holds a real cell's declared ``tool_frame.offset_mm`` along the approach to this
    #: number plus the coupling plates.
    grasp_centre_mm: float = Field(gt=0.0, le=1000.0)
    #: The physical open width.
    aperture_mm: float = Field(gt=0.0)
    #: The smallest grip that counts as a grip. A policy floor about the part, not the tool.
    min_width_mm: float = Field(ge=0.0)
    #: What the jaws read when shut on nothing.
    closed_width_mm: float = Field(ge=0.0)
    #: Finger material past the grasp centre, toward the part.
    finger_ahead_mm: float = Field(gt=0.0, le=200.0)
    #: The measured finger reach back toward the wrist.
    finger_behind_mm: float = Field(gt=0.0)
    #: The collision envelope's finger length behind the centre, conservative.
    finger_length_mm: float = Field(gt=0.0, le=500.0)
    finger_thickness_mm: float = Field(gt=0.0, le=200.0)
    finger_width_mm: float = Field(gt=0.0, le=200.0)
    finger_pad_overlap_mm: float = Field(ge=0.0, le=100.0)
    pad_length_mm: float = Field(gt=0.0, le=300.0)
    pad_ahead_mm: float = Field(gt=0.0, le=300.0)
    pad_behind_mm: float = Field(ge=0.0)
    palm_depth_mm: float = Field(gt=0.0, le=500.0)
    palm_width_mm: float = Field(gt=0.0, le=500.0)
    #: The housing's full extent along the closing axis: twice its furthest reach from the grasp axis,
    #: measured off the bundle by scripts/grippers/measure_jaw_from_bundle.py. Optional, because only
    #: an envelope written from these numbers reads it. That writer refuses a hand that leaves it
    #: unset, since the fingers' outer face does not bound the housing: the EGU-50's housing reaches
    #: 2.25 mm past it.
    palm_thickness_mm: float | None = Field(default=None, gt=0.0, le=500.0)
    #: False when the palm numbers are an estimate. A consumer that checks the housing says so.
    palm_measured: bool
    friction_coefficient: float = Field(gt=0.0)
    #: True when the friction is the default rather than a measurement of these pads.
    friction_is_default: bool

    @model_validator(mode="after")
    def _the_numbers_describe_one_jaw(self) -> "ParallelJawSpec":
        if abs(self.pad_ahead_mm + self.pad_behind_mm - self.pad_length_mm) > _PAD_TOLERANCE_MM:
            raise ValueError(
                f"the pad does not add up: pad_ahead_mm {self.pad_ahead_mm} + pad_behind_mm "
                f"{self.pad_behind_mm} is not pad_length_mm {self.pad_length_mm}"
            )
        if self.min_width_mm >= self.aperture_mm or self.closed_width_mm >= self.aperture_mm:
            raise ValueError(
                f"min_width_mm {self.min_width_mm} and closed_width_mm {self.closed_width_mm} must "
                f"both be smaller than aperture_mm {self.aperture_mm}"
            )
        if self.finger_length_mm < self.finger_behind_mm:
            raise ValueError(
                f"finger_length_mm {self.finger_length_mm} is shorter than the measured reach "
                f"finger_behind_mm {self.finger_behind_mm}; the envelope's length is the conservative one"
            )
        if self.pad_ahead_mm > self.finger_ahead_mm:
            raise ValueError(
                f"pad_ahead_mm {self.pad_ahead_mm} reaches past the finger it sits on "
                f"(finger_ahead_mm {self.finger_ahead_mm})"
            )
        if self.pad_behind_mm > self.finger_behind_mm:
            raise ValueError(
                f"pad_behind_mm {self.pad_behind_mm} reaches past the finger it sits on "
                f"(finger_behind_mm {self.finger_behind_mm})"
            )
        return self


class GripperSpec(StrictModel):
    """One registry file: who the hand is, where the numbers came from, and the numbers."""

    model: str = Field(pattern=MODEL_NAME_PATTERN)
    #: Other names the same hand is stamped with, for example the ``2f85`` existing corpora carry.
    aliases: tuple[str, ...] = ()
    kind: Literal["parallel_jaw"]
    #: Where the numbers came from, in words a person can check.
    source: str = Field(min_length=1)
    jaw: ParallelJawSpec
    #: The ``robot.gripper.vendor`` drivers that can actuate this hand, or ``None`` where the file says
    #: nothing. A real UR profile naming the hand with a driver not listed is refused at load. ``none``
    #: and ``dummy``, a flange with nothing actuated yet and a desk stand-in, are always admitted and
    #: are not listed.
    drivers: tuple[str, ...] | None = None

    @field_validator("drivers")
    @classmethod
    def _drivers_are_gripper_vendors(cls, drivers: "tuple[str, ...] | None") -> "tuple[str, ...] | None":
        if drivers is None:
            return None
        # Lazy, as GripperConfig's vendor check: the enum is a pure StrEnum and the config layer
        # imports no runtime.
        from src.robot.core.gripper_vendor import GripperVendor

        if not drivers:
            raise ValueError("drivers is empty: leave it out when the file says nothing about which driver actuates it")
        canonical = tuple(GripperVendor.from_string(name).value for name in drivers)
        always = {GripperVendor.NONE.value, GripperVendor.DUMMY.value}
        if always & set(canonical):
            raise ValueError(f"drivers lists {sorted(always & set(canonical))}, which are always admitted: leave them out")
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"drivers repeats a name: {list(drivers)}")
        return canonical

    @field_validator("aliases")
    @classmethod
    def _aliases_are_registry_names(cls, aliases: tuple[str, ...]) -> tuple[str, ...]:
        import re

        for alias in aliases:
            if not re.fullmatch(MODEL_NAME_PATTERN, alias):
                raise ValueError(f"alias {alias!r} is not a registry name ({MODEL_NAME_PATTERN})")
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"aliases repeat a name: {list(aliases)}")
        return aliases

    @model_validator(mode="after")
    def _no_alias_is_the_model(self) -> "GripperSpec":
        if self.model in self.aliases:
            raise ValueError(f"{self.model!r} is listed as an alias of itself")
        return self
