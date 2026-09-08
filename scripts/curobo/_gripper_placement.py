"""Where a gripper's tool0 spheres sit on the flange. Stdlib only, on purpose.

``build_ur_config.py`` runs under the cuRobo sidecar's interpreter, where ``src.robot...`` is
un-importable, so this cannot live beside the fit in
``src/robot/safety/planning/robot/gripper_spheres.py``. It is a separate FILE rather than a
second copy: the script imports it from beside itself and the test loads the same file by path, so
there is one implementation and two callers. The sphere fit itself was written twice once already,
one copy calling itself a mirror of the other with nothing checking, and that cost a cell a planner
that modelled the wrong hand.

Nothing here fits anything. The arithmetic is a translation along the approach, and it is here so it
can be tested at all: a refusal proves the path is not taken silently, and it does not prove the
plate is added where it is supplied.
"""

from __future__ import annotations

#: The map already sits where the hand is bolted, so nothing is added.
FLANGE = "flange"

#: The map starts at the hand's own mounting face, so the coupling plate between that face and the
#: flange still has to be added.
MOUNTING_FACE = "mounting_face"

#: Index of the approach axis in a tool0 sphere centre. The frame is X closing, Y approach, Z
#: binormal, read out of the committed 2F-85 bundle rather than assumed.
_APPROACH = 1


class PlacementError(ValueError):
    """The spheres cannot be placed as described."""


def place_tool0_spheres(
    spheres: list[dict], *, origin: str, coupling_mm: float | None
) -> list[dict]:
    """Return the spheres where the hand actually is, in metres, or refuse.

    ``origin`` comes from the sphere map's own provenance:

    * :data:`FLANGE`: a map fitted from a COMPOSED arm asset, where the arm had already placed the
      hand. The numbers go in as they are, and supplying a coupling is a contradiction rather than a
      refinement, so it is refused instead of quietly added.
    * :data:`MOUNTING_FACE`: a map fitted from a standalone vendor asset, which has no idea what it
      will be bolted to. ``coupling_mm`` is required. Assuming zero would put every sphere one plate
      too close to the flange, which is optimistic in the one direction a planner must not be, and
      the resulting file looks entirely reasonable.

    Millimetres in, because that is how a plate is measured, and metres out, because that is what
    cuRobo reads. The conversion is here rather than at the call site so there is one of it.
    """
    if origin not in (FLANGE, MOUNTING_FACE):
        raise PlacementError(
            f"origin must be {FLANGE!r} or {MOUNTING_FACE!r}, got {origin!r}. The on-box builder "
            "decides whether to add a coupling from this, so an unknown value is not something to "
            "guess past."
        )
    if origin == FLANGE:
        if coupling_mm not in (None, 0.0):
            raise PlacementError(
                f"this map already sits at the flange, so a coupling of {coupling_mm} mm would move "
                "the hand away from where it was measured. Drop --coupling-mm, or use a map whose "
                f"origin is {MOUNTING_FACE!r}."
            )
        return [dict(s) for s in spheres]

    if coupling_mm is None:
        raise PlacementError(
            "this map holds the hand measured from its own mounting face, so where it sits on the "
            "arm depends on the coupling plate between them, and this will not guess it.\n"
            "\n"
            "  Measure flange face to gripper mounting face, in millimetres, and pass it:\n"
            "    --coupling-mm <thickness>\n"
            "\n"
            "  It is the same measurement robot.gripper.tool_frame.offset_mm needs, so take both in "
            "one bench session. Passing 0 is a legitimate answer for a hand bolted straight to the "
            "flange, and saying 0 out loud is the point."
        )
    if coupling_mm < 0.0:
        raise PlacementError(
            f"a coupling of {coupling_mm} mm would put the hand behind the flange, inside the wrist."
        )

    shift = float(coupling_mm) / 1000.0
    out = []
    for sphere in spheres:
        centre = [float(v) for v in sphere["center"]]
        centre[_APPROACH] += shift
        out.append({**sphere, "center": centre})
    return out
