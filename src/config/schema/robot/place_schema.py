"""How a task sets its parts down (the owner, 2026-10-08 night): parts are set down, not dropped, and never piled.

Three choices, each a cell's own, every default what a task did before the block existed:

* **Into a box** (``release_in_a_box``). ``over_the_rim``: the jaws open over the rim, the part's bottom the task's rim
  air above it. ``below_the_rim``: they open ``below_the_rim_mm`` under it ("bei so kleinen Kisten ... 1-5 cm unter dem
  Kistenrand"), never lower than 10 mm over what the camera reads inside the box (its floor, or the parts already in
  it); where the opening is too narrow for the open hand and the part, or the camera could not read the inside, the
  part is let go over the rim, as before.
* **On a flat place** (``side_by_side``): a taught pose, or a target the camera found with no inside. Off, every part is
  let go at the one spot. On, the parts lie side by side ("sonst tuermen wir die Bauteile"): a spot the camera reads
  free first, else the next spot of a grid about the taught pose (``grid``), each part's footprint (the open hand's
  where that is wider) and ``spacing_margin_mm`` apart.
* **The carry** (``carry``). ``via_the_look``: the part is carried to the look the bin was found from, and the bin is
  looked at there. ``over_the_rim``: where one of the pick's looks was the bin's, the bin is checked on that look's
  frame and the part goes straight over to it, its bottom kept ``rim_floor_margin_mm`` and the rim air over the rim
  and over everything the pick's looks saw on the way; a carry that cannot be so goes via the look.

``part_bottom`` says what a part's hang below the tool is measured from: ``declared_support``, the support the cell
declares (a lower bound, so the error goes toward more air: on the owner's cell the drop stood 85 to 92 mm over the
rim, the part's 55 mm mat under it), or ``measured``, what the pick's own looks read under the part.

``relocate`` says what a task does with a bin the check before a drop lost (not seen, moved too far, another size where
it stood), on as shipped (the owner, 2026-10-09: "wenn sie dies nicht mehr tut, dann kann er seine Ablage nochmal neu
errechnen"): it looks for the bin again from its looks, the part in the jaws, keeps a bin of the size the survey found
and the colour it followed, standing in no other place, in place of the old one, and plans the drop anew over it, once
per part; found nowhere, the part goes back where it was gripped and the task asks. Off, every lost bin puts the part
back and asks, as before.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .._base import StrictModel

__all__ = ["PlaceGridConfig", "RobotPlaceConfig"]


class PlaceGridConfig(StrictModel):
    """The spots a flat place lays its parts out on where no camera reads it: ``rows`` x ``columns`` about the taught
    pose, along the taught tool's heading, the nearest the taught pose first."""

    #: How many spots along the taught tool's +X.
    rows: int = Field(default=3, ge=1, le=20)
    #: How many spots along the taught tool's +Y.
    columns: int = Field(default=3, ge=1, le=20)
    #: How far apart two spots stand, mm. ``None``: each part's own footprint, the open hand's where that is wider, and
    #: ``robot.place.spacing_margin_mm``, so a larger part takes a coarser grid.
    spacing_mm: float | None = Field(default=None, gt=0.0, le=1000.0)


class RobotPlaceConfig(StrictModel):
    """How a task sets its parts down: into a box, on a flat place, and the carry to a box. See the module docstring."""

    #: Where the jaws open on a box: ``over_the_rim`` (the task's rim air over it, as before) or ``below_the_rim``.
    release_in_a_box: Literal["over_the_rim", "below_the_rim"] = "over_the_rim"
    #: How far under the rim the part's bottom stands when the jaws open below it, mm: the owner's 10 to 50 mm.
    below_the_rim_mm: float = Field(default=30.0, ge=10.0, le=50.0)
    #: How far inside a box's opening the part and the open hand keep, on every side, before the part goes below the
    #: rim, mm. The guard judges the hand on the line in; the part is not modelled, so this keeps it off the walls.
    opening_margin_mm: float = Field(default=10.0, ge=0.0, le=50.0)
    #: What a part's hang below the tool is measured from: ``declared_support`` (as before) or ``measured``.
    part_bottom: Literal["declared_support", "measured"] = "declared_support"
    #: Whether the parts on a flat place lie side by side rather than at the one spot.
    side_by_side: bool = False
    #: How far apart two parts lie side by side beyond their footprints, mm.
    spacing_margin_mm: float = Field(default=20.0, ge=0.0, le=200.0)
    #: The grid about a taught pose where no camera reads its spots.
    grid: PlaceGridConfig = Field(default_factory=PlaceGridConfig)
    #: How a part is carried to a box: ``via_the_look`` (as before) or ``over_the_rim``.
    carry: Literal["via_the_look", "over_the_rim"] = "via_the_look"
    #: How far over the rim, and over what the pick's looks saw on the way, beyond the rim air, the part's bottom stays
    #: on a carry over the rim, mm.
    rim_floor_margin_mm: float = Field(default=15.0, ge=0.0, le=100.0)
    #: Whether a bin the check before a drop lost is looked for again and the drop planned anew over it (the owner,
    #: 2026-10-09), or the part goes back where it was gripped and the task asks (off, as before).
    relocate: bool = True
