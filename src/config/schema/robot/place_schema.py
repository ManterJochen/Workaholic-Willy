"""How a task sets its parts down (the owner, 2026-10-08 night): parts are set down, not dropped, and never piled.

Three choices, each a cell's own, every default what a task did before the block existed:

* **Into a box** (``release_in_a_box``). ``over_the_rim``: the jaws open over the rim, the part's bottom the task's rim
  air above it. ``below_the_rim``: they open 30 mm under it ("bei so kleinen Kisten ... 1-5 cm unter dem
  Kistenrand"), never lower than 10 mm over what the camera reads inside the box (its floor, or the parts already in
  it); where the opening is too narrow for the open hand and the part, or the camera could not read the inside, the
  part is let go over the rim, as before.
* **On a flat place** (``side_by_side``): a taught pose, or a target the camera found with no inside. Off, every part is
  let go at the one spot. On, the parts lie side by side ("sonst tuermen wir die Bauteile"): a spot the camera reads
  free first, else the next spot of a grid about the taught pose (``grid``), each part's footprint (the open hand's
  where that is wider) and 20 mm apart.
* **The carry** (``carry``). ``via_the_look``: the part is carried to the look the bin was found from, and the bin is
  looked at there. ``over_the_rim``: where one of the pick's looks was the bin's, the bin is checked on that look's
  frame and the part goes straight over to it, its bottom kept 15 mm and the rim air over the rim
  and over everything the pick's looks saw on the way; a carry that cannot be so goes via the look.

The jaws open 20 mm over a box's rim (the owner, 2026-10-14: "genau auf Höhe der Kante" is 0, "20 mm über der
Kante" 20): the part's bottom stands that far over it where the cell declares how far its parts hang
(``part_hang_mm``), the hang measured part by part from what the pick's looks read under the part; the tool where it
declares none. A task's own air (``PlaceAt.air_mm``) goes before it.

A bin the check before a drop lost (not seen, moved too far, another size where it stood) is looked for again (the
owner, 2026-10-09: "wenn sie dies nicht mehr tut, dann kann er seine Ablage nochmal neu errechnen"): from its looks,
the part in the jaws, a bin of the size the survey found and the colour it followed, standing in no other place,
takes the old one's place, and the drop is planned anew over it, once per part; found nowhere, the part goes back where
it was gripped and the task asks.

The numbers are the owner's cell's since 2026-10-17, fixed in ``src/robot/execution/place_target.py``: 30 mm under the
rim, 10 mm inside a box's opening, 20 mm of rim air, 20 mm between parts side by side, 15 mm over the rim on a carry;
and a lost bin is always looked for again. Their keys and ``relocate`` were refused at load from then on.
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
    #: the 20 mm a place keeps between parts, so a larger part takes a coarser grid.
    spacing_mm: float | None = Field(default=None, gt=0.0, le=1000.0)


class RobotPlaceConfig(StrictModel):
    """How a task sets its parts down: into a box, on a flat place, and the carry to a box. See the module docstring."""

    #: Where the jaws open on a box: ``over_the_rim`` (the task's rim air over it, as before) or ``below_the_rim``.
    release_in_a_box: Literal["over_the_rim", "below_the_rim"] = "over_the_rim"
    #: Whether the parts on a flat place lie side by side rather than at the one spot.
    side_by_side: bool = False
    #: The grid about a taught pose where no camera reads its spots.
    grid: PlaceGridConfig = Field(default_factory=PlaceGridConfig)
    #: How a part is carried to a box: ``via_the_look`` (as before) or ``over_the_rim``.
    carry: Literal["via_the_look", "over_the_rim"] = "via_the_look"
    #: How far the longest part this cell carries hangs past the fingertips along the approach, mm: a declared worst
    #: case rather than a measurement, since the grasp knows the part at the grasp line and nothing about the rest of it.
    #: A place sets the part's bottom down where the taught fingertips stood, raised by this, and keeps it over a box's
    #: rim (by the rim air); the pick's looks measure the hang part by part where they read under the part. ``None``:
    #: the part hangs from the TCP (the owner, 2026-10-14: "Wenn kein Payloadlength angegeben ist, dann soll es am TCP
    #: hängen"), the default; the owner's cell sets 1. It was ``robot.safety.planning_world.payload.length_mm`` until
    #: 2026-10-17, when the carried part's spheres left the planner (no cell modelled them: a closed hand is judged as an
    #: empty one), and a person sets it in the console as before.
    part_hang_mm: float | None = Field(default=None, gt=0.0, le=2000.0)
