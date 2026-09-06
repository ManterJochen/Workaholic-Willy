"""What is this part standing on? A declared height, raised by what the cameras can see.

The parallel-jaw calculator accepts a ``support_plane``, and when it is given none
``collision_checker.validate_grasp_collision`` skips the clearance check entirely. What that costs is
measured: with no plane, ``rejected_table`` stands at 0 in 2128 of 2128 telemetry lines while the
independent datagen reference rejects 37 % of those same candidates for putting a finger under the
support. This module decides which plane to hand over.

The rule is one line and it is measured rather than chosen: take the higher of the declared height and
the target's own lowest point.

Over 1073 reference objects, split by whether the part rests on the table or on another part:

===========================  =========================  ============================
estimator                    on the table (n=1052)      standing on something (n=21)
===========================  =========================  ============================
declared height alone        median +0.00, 0 % too low  median -14.08, **100 % too low**
target footprint             median +1.81, 0 % too low  median +0.89, **0 % too low**
plane fit around the object  median +19.97, 6 % too low  median -1.19, **33 % too low**
plane fit over the scene     median +14.75, 2 % too low  median +9.41, **33 % too low**
===========================  =========================  ============================

A declared constant cannot know that a part is standing on another part, and there it is too low every
single time. Both plane fits are worse: they read too low on a third of the stacked cases, and too low
is the direction that drives a finger through a surface. The target's own lowest point was never too
low, in any of the 1073 cases, in either situation, so raising the declared height to it can only make
a grasp more conservative, never bolder.

The footprint is fused across cameras. A single view of a bin sees the part over the wall, so its
lowest visible point is the rim and not the base: p90 error of 50.6 mm in the ``bin`` family, with 38 %
of parts more than 10 mm too high. The lowest point over every camera that identified the part cuts
that to 13.3 mm and 10.6 %, and stays at 0 % too low everywhere.

Pure and deterministic: numpy only, no vendor SDKs, no perception models. BASE millimetres throughout.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.geometry import Frame

from .table_collision import SupportPlane

__all__ = ["SupportResolution", "resolve_support_plane"]


class SupportResolution:
    """The plane, and why it is where it is. The reason travels with the number, deliberately.

    A clearance rejection that a reader cannot trace back to a height is indistinguishable from a bad
    grasp, and a check nobody can read is a check nobody keeps switched on.
    """

    __slots__ = ("plane", "declared_mm", "observed_mm", "source")

    def __init__(self, plane: SupportPlane, declared_mm: float,
                 observed_mm: float | None, source: str) -> None:
        self.plane = plane
        self.declared_mm = declared_mm
        self.observed_mm = observed_mm
        self.source = source

    @property
    def height_mm(self) -> float:
        return float(self.plane.offset_mm)

    def as_telemetry(self) -> dict:
        return {"support_height_mm": round(self.height_mm, 2),
                "support_declared_mm": round(self.declared_mm, 2),
                "support_observed_mm": (None if self.observed_mm is None
                                        else round(self.observed_mm, 2)),
                "support_source": self.source}


def resolve_support_plane(
    *,
    declared_height_mm: float = 0.0,
    container_floor_mm: float | None = None,
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    target_clouds_base_mm: Sequence[np.ndarray] | None = None,
    refine_from_target: bool = True,
) -> SupportResolution:
    """The support plane for one grasp target.

    ``container_floor_mm`` is the optional inside floor of a bin or tray. Give it and it replaces the
    workspace height, because a KLT standing on the table raises what its contents rest on by the
    thickness of its own floor. It does not fix a wall-occluded estimate, since the floor is the lower
    of the two and the observation still wins.

    ``target_clouds_base_mm`` is one cloud per camera that identified this target; see
    :mod:`~src.robot.grasping.multiview.association` for how a camera decides that it did. Passing
    several is what turns a bin's 50.6 mm p90 error into 13.3 mm. Passing one is still correct, only
    noisier, and passing none falls back to the declared height alone.

    The plane's offset is a height along ``normal``, so a tilted tray works: the offset is the declared
    surface projected onto the normal, and the observation is the lowest point measured along it.
    """
    unit = np.asarray(normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(unit))
    if length < 1e-12:
        raise ValueError("support normal must not be zero")
    unit = unit / length

    declared = float(container_floor_mm if container_floor_mm is not None else declared_height_mm)
    observed: float | None = None
    if refine_from_target and target_clouds_base_mm:
        lows = [float((np.asarray(c, dtype=np.float64).reshape(-1, 3) @ unit).min())
                for c in target_clouds_base_mm
                if np.asarray(c).size]
        if lows:
            # The lowest over the cameras, not an average: a camera that cannot see the base reports a
            # height that is too high, and averaging a wrong number in spreads the error instead of
            # discarding it. Whichever camera saw furthest down is the one that saw the base.
            observed = min(lows)

    if observed is None:
        return SupportResolution(SupportPlane(normal=unit, offset_mm=declared, frame=Frame.BASE),
                                 declared, None, "declared")
    # The higher wins. Never lower: the declared surface is the floor of what is possible, and an
    # observation below it is depth noise looking through the table.
    if observed > declared:
        return SupportResolution(SupportPlane(normal=unit, offset_mm=observed, frame=Frame.BASE),
                                 declared, observed, "observed")
    return SupportResolution(SupportPlane(normal=unit, offset_mm=declared, frame=Frame.BASE),
                             declared, observed, "declared")
