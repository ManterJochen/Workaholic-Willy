"""Objects made of parts: a mug with a handle, a jug with a neck, a tool with a grip.

`procedural.py` states the limit these families lift: sixteen kind words render as three solids, so
a "cup" is a cylinder and a "bowl" is a cuboid, and such an object has no graspable feature distinct
from its body. A network trained on that can learn where to grip a convex blob; it cannot learn that
a mug is grasped by its handle.

The design constraint that shapes everything here. `datagen.grasps` computes exact analytic contacts
from the same primitives the renderer authors. If the two ever disagreed, the labels would describe
a shape that was never rendered and every recall number built on them would be wrong. So a part is
not a new primitive: it is a box, a cylinder or a sphere at a pose inside the object. A handle is an
arch of three boxes, not a torus. That is a shape a real mug can have, it renders exactly, and its
grasp labels are the same closed-form contacts the rest of the library produces.

Parts fall out of the existing machinery. `label_jaw_grasps` works against one target solid plus
arbitrarily many obstacle solids, and `SceneGeometry.obstacles_for` assembles those. A composite is
labelled part by part, with the object's other parts passed as obstacles: a grip on the handle that
passes through the mug body is not a grasp, and the collision check that rejects it is already
written.

What a part label buys: the same object yields grasps at the handle and grasps on the body,
distinguishable by `part_role`, so a model can learn which one a prompt is asking for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from datagen.assets.manifest import AssetRecord

from datagen.assets.procedural import MAX_JAW_APERTURE_MM

__all__ = [
    "COMPOSITE_KINDS",
    "CompositeAsset",
    "CompositePart",
    "PartRole",
    "build_composite",
    "sample_composite_asset",
]


class PartRole:
    """What a part is, semantically. The channel a model learns to condition on.

    Strings rather than an enum: they travel into JSON labels and a schema field, and the set grows
    as families are added. The values are the vocabulary a prompt would use.
    """

    BODY: Final[str] = "body"
    HANDLE: Final[str] = "handle"
    NECK: Final[str] = "neck"
    RIM: Final[str] = "rim"
    GRIP: Final[str] = "grip"
    HEAD: Final[str] = "head"
    BASE: Final[str] = "base"


#: Composite families, and which part roles each contributes. `sample_composite_asset` draws from
#: these; the geometry per family lives in the builders below.
COMPOSITE_KINDS: Final[dict[str, tuple[str, ...]]] = {
    "mug": (PartRole.BODY, PartRole.HANDLE),
    "jug": (PartRole.BODY, PartRole.NECK, PartRole.HANDLE),
    "pan": (PartRole.BODY, PartRole.GRIP),
    "hammer": (PartRole.GRIP, PartRole.HEAD),
    "bucket": (PartRole.BODY, PartRole.HANDLE),
}


@dataclass(frozen=True, slots=True)
class CompositePart:
    """One primitive inside a composite object, in the object's own frame.

    `primitive` is one of ``box`` / ``cylinder`` / ``sphere``: the three the renderer can author and
    the label machinery can intersect. Anything else breaks the coupling the module docstring
    describes.
    """

    role: str
    primitive: str
    #: Full size in millimetres, as the renderer's `extent_mm` convention (not half-extents).
    extent_mm: tuple[float, float, float]
    #: Centre offset from the object origin, in the object's own frame, millimetres.
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Rotation about +Z in degrees. Enough for every family here; a part needing full 3-DoF
    #: orientation would take a quaternion, and none does yet.
    yaw_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.primitive not in ("box", "cylinder", "sphere"):
            raise ValueError(
                f"part primitive must be box/cylinder/sphere (the three the renderer authors and "
                f"the labeller intersects), got {self.primitive!r}"
            )
        if min(self.extent_mm) <= 0.0:
            raise ValueError(f"part extent must be positive, got {self.extent_mm!r}")

    @property
    def jaw_graspable(self) -> bool:
        """Could a 2F-85 close on this part?

        A mug body is often too wide for the jaw while its handle is not, which is the population a
        whole-object graspability test cannot reach.
        """
        return min(self.extent_mm) <= MAX_JAW_APERTURE_MM


@dataclass(frozen=True, slots=True)
class CompositeAsset:
    """An object built from parts. Satisfies the same `SceneAsset` contract as the other two kinds."""

    asset_id: str
    composite_kind: str
    parts: tuple[CompositePart, ...]
    mass_kg: float
    source: str = "procedural"
    license: str = "own"
    attribution: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def family(self) -> str:
        return "composite"

    @property
    def kind(self) -> str:
        """``composite`` selects the multi-primitive branch in the renderer and the labeller."""
        return "composite"

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        """The axis-aligned bound of every part: what placement fits and spaces objects by."""
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for part in self.parts:
            half = np.asarray(part.extent_mm, dtype=np.float64) / 2.0
            # A yawed part's AABB grows; rotate its half-extents rather than under-reporting.
            if abs(part.yaw_deg) > 1e-9:
                angle = math.radians(part.yaw_deg)
                cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
                half = np.array([
                    half[0] * cos + half[1] * sin,
                    half[0] * sin + half[1] * cos,
                    half[2],
                ])
            centre = np.asarray(part.offset_mm, dtype=np.float64)
            lows.append(centre - half)
            highs.append(centre + half)
        low = np.min(np.vstack(lows), axis=0)
        high = np.max(np.vstack(highs), axis=0)
        span = high - low
        return (float(span[0]), float(span[1]), float(span[2]))

    @property
    def jaw_graspable(self) -> bool:
        """True when any part fits the jaw.

        Not the whole object's shortest axis: a 100 mm mug body with a 14 mm handle is graspable,
        and reporting it as not graspable would hide the part that is.
        """
        return any(part.jaw_graspable for part in self.parts)

    def part_roles(self) -> tuple[str, ...]:
        return tuple(part.role for part in self.parts)

    def as_record(self) -> "AssetRecord":
        """This asset as the manifest stores it, parts included.

        The parts have to travel: `scene_geometry` derives an object's shape from the tail of its
        asset id, so a composite that reached the labeller without them would be labelled as the
        cuboid `primitive_for_kind` returns for "mug". See `SceneAsset.as_record`.
        """
        return AssetRecord(
            asset_id=self.asset_id, source=self.source, license=self.license,
            kind=self.kind, extent_mm=self.extent_mm, mass_kg=self.mass_kg,
            attribution=self.attribution, parts=tuple(self.parts),
            tags=(self.family,) + tuple(self.tags),
        )


# --------------------------------------------------------------------- builders
#
# Each builder returns parts in the object's own frame; `_centred` then shifts them so the origin
# sits at the centre of the axis-aligned bound, which is where the renderer places the rigid body.
# Dimensions are drawn from bands that produce recognisable objects at the scale this cell picks
# (25-180 mm).


def _handle_arch(
    *, body_radius_mm: float, height_mm: float, thickness_mm: float, reach_mm: float,
) -> tuple[CompositePart, ...]:
    """A C-shaped handle as three boxes: two stubs off the body and a bar joining them.

    Three boxes rather than a torus because a torus is a primitive this stack can neither author nor
    intersect analytically. The arch is a shape real mugs have, and every contact on it is a
    closed-form box contact.
    """
    stub = thickness_mm
    outer = body_radius_mm + reach_mm
    return (
        CompositePart(PartRole.HANDLE, "box", (reach_mm, stub, stub),
                      (body_radius_mm + reach_mm / 2.0, 0.0, height_mm / 2.0 - stub)),
        CompositePart(PartRole.HANDLE, "box", (reach_mm, stub, stub),
                      (body_radius_mm + reach_mm / 2.0, 0.0, -height_mm / 2.0 + stub)),
        CompositePart(PartRole.HANDLE, "box", (stub, stub, height_mm - 2.0 * stub),
                      (outer - stub / 2.0, 0.0, 0.0)),
    )


def _mug(rng: np.random.Generator) -> tuple[CompositePart, ...]:
    diameter = float(rng.uniform(60.0, 95.0))
    height = float(rng.uniform(80.0, 110.0))
    thickness = float(rng.uniform(8.0, 14.0))
    # Straddles _FINGER_CLEARANCE_MM: the low end is a handle pressed close to the body, which no
    # jaw can reach; the high end stands clear.
    reach = float(rng.uniform(20.0, 48.0))
    body = CompositePart(PartRole.BODY, "cylinder", (diameter, diameter, height))
    return (body, *_handle_arch(
        body_radius_mm=diameter / 2.0, height_mm=height * 0.62,
        thickness_mm=thickness, reach_mm=reach,
    ))


def _jug(rng: np.random.Generator) -> tuple[CompositePart, ...]:
    diameter = float(rng.uniform(75.0, 110.0))
    height = float(rng.uniform(110.0, 150.0))
    neck_d = diameter * float(rng.uniform(0.35, 0.55))
    neck_h = height * float(rng.uniform(0.15, 0.25))
    thickness = float(rng.uniform(8.0, 14.0))
    reach = float(rng.uniform(22.0, 50.0))
    body = CompositePart(PartRole.BODY, "cylinder", (diameter, diameter, height))
    neck = CompositePart(PartRole.NECK, "cylinder", (neck_d, neck_d, neck_h),
                         (0.0, 0.0, height / 2.0 + neck_h / 2.0))
    # 0.95 rather than 0.5: a jug's handle runs nearly its full height, which is both realistic and
    # what puts the free span near the ~100 mm the jaw needs. Some draws clear it, most do not.
    return (body, neck, *_handle_arch(
        body_radius_mm=diameter / 2.0, height_mm=height * 0.95,
        thickness_mm=thickness, reach_mm=reach,
    ))


def _pan(rng: np.random.Generator) -> tuple[CompositePart, ...]:
    diameter = float(rng.uniform(90.0, 140.0))
    depth = float(rng.uniform(30.0, 50.0))
    grip_len = float(rng.uniform(60.0, 95.0))
    grip_thick = float(rng.uniform(14.0, 22.0))
    body = CompositePart(PartRole.BODY, "cylinder", (diameter, diameter, depth))
    grip = CompositePart(PartRole.GRIP, "box", (grip_len, grip_thick, grip_thick),
                         (diameter / 2.0 + grip_len / 2.0, 0.0, depth / 4.0))
    return (body, grip)


def _hammer(rng: np.random.Generator) -> tuple[CompositePart, ...]:
    shaft_len = float(rng.uniform(110.0, 170.0))
    shaft_d = float(rng.uniform(16.0, 26.0))
    head_len = float(rng.uniform(45.0, 70.0))
    head_side = float(rng.uniform(24.0, 36.0))
    grip = CompositePart(PartRole.GRIP, "cylinder", (shaft_d, shaft_d, shaft_len))
    head = CompositePart(PartRole.HEAD, "box", (head_len, head_side, head_side),
                         (0.0, 0.0, shaft_len / 2.0 + head_side / 2.0))
    return (grip, head)


def _bucket(rng: np.random.Generator) -> tuple[CompositePart, ...]:
    diameter = float(rng.uniform(110.0, 160.0))
    height = float(rng.uniform(90.0, 130.0))
    thickness = float(rng.uniform(10.0, 15.0))
    # How far the bail rises above the rim. Straddles the finger clearance for the same reason.
    bail_rise = float(rng.uniform(22.0, 55.0))
    body = CompositePart(PartRole.BODY, "cylinder", (diameter, diameter, height))
    # A swing handle over the top: one bar plus two uprights.
    bar = CompositePart(PartRole.HANDLE, "box", (diameter, thickness, thickness),
                        (0.0, 0.0, height / 2.0 + bail_rise))
    left = CompositePart(PartRole.HANDLE, "box",
                         (thickness, thickness, bail_rise),
                         (-diameter / 2.0 + thickness / 2.0, 0.0, height / 2.0 + bail_rise / 2.0))
    right = CompositePart(PartRole.HANDLE, "box",
                          (thickness, thickness, bail_rise),
                          (diameter / 2.0 - thickness / 2.0, 0.0, height / 2.0 + bail_rise / 2.0))
    return (body, bar, left, right)


_BUILDERS = {
    "mug": _mug, "jug": _jug, "pan": _pan, "hammer": _hammer, "bucket": _bucket,
}

#: A handle is jaw-graspable only when its free span, the run of bar between the stubs that join it
#: to the body, exceeds roughly 100 mm. The reason is the gripper, not the geometry: the 2F-85's
#: inner finger is 62.3 mm long with a 41.5 mm pad, so it needs more clear run than a mug handle
#: has. A 2F-85 cannot grasp a typical mug by its handle.
#:
#: So these families are a deliberate mix, and both halves are the point. Buckets, pans and hammers
#: give a part-aware model positive examples of "grip the feature, not the body". Mugs and jugs give
#: it the negative: a handle it must not reach for, because this gripper cannot. A dataset holding
#: only the positives teaches a model that handles are always graspable, which fails on the first
#: real mug. A suction end effector is what picks those up.

#: How far a graspable feature must stand clear of the body, in millimetres.
#:
#: Measured, not chosen: `JawModel.finger_ahead_mm = 28.72` is how far the 2F-85's inner finger
#: reaches past the contact point, taken off the gripper's own collision shapes. A handle closer to
#: the body than that cannot be gripped however thin it is: the finger arrives before the pad does
#: and hits the body, and `FINGER_COLLISION` is the rejection that follows.
#:
#: The bands in the builders above deliberately straddle it. A dataset where every handle is
#: graspable would teach a model that handles always are, which is false and would cost it on the
#: first real mug.
_FINGER_CLEARANCE_MM: Final[float] = 28.72

#: Rough polymer/ceramic density for mass from part volume, kg/mm^3. Same reasoning as the mesh
#: side: the physics label needs a plausible, consistent mass, not a correct one.
_DENSITY_KG_PER_MM3: Final[float] = 0.6e-6



def _centred(parts: tuple[CompositePart, ...]) -> tuple[CompositePart, ...]:
    """Shift every part so the object's bounding box is centred on its origin.

    Placement puts an object down with `z = table + extent_mm[2] / 2` and spaces neighbours by
    `extent_mm`, both of which assume the origin sits at the centre of the bound. A pan whose grip
    juts to one side breaks that assumption: the object ends up floating or sunk into the table with
    no error anywhere, and the labels then describe contacts at heights the renderer never produced.
    """
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for part in parts:
        half = np.asarray(part.extent_mm, dtype=np.float64) / 2.0
        if abs(part.yaw_deg) > 1e-9:
            angle = math.radians(part.yaw_deg)
            cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
            half = np.array([half[0] * cos + half[1] * sin, half[0] * sin + half[1] * cos, half[2]])
        centre = np.asarray(part.offset_mm, dtype=np.float64)
        lows.append(centre - half)
        highs.append(centre + half)
    shift = (np.min(np.vstack(lows), axis=0) + np.max(np.vstack(highs), axis=0)) / 2.0
    if float(np.abs(shift).max()) < 1e-9:
        return parts
    return tuple(
        CompositePart(
            role=part.role,
            primitive=part.primitive,
            extent_mm=part.extent_mm,
            offset_mm=(
                float(part.offset_mm[0] - shift[0]),
                float(part.offset_mm[1] - shift[1]),
                float(part.offset_mm[2] - shift[2]),
            ),
            yaw_deg=part.yaw_deg,
        )
        for part in parts
    )


def build_composite(kind: str, rng: np.random.Generator, *, index: int,
                    unique_id: bool = False) -> CompositeAsset:
    """One composite object of ``kind``. Deterministic given ``rng``.

    ``unique_id`` makes the asset id name the shape rather than the scene slot. Default False; the
    comment beside the id says what that costs.
    """
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unknown composite kind {kind!r}; known: {sorted(_BUILDERS)}")
    parts = _centred(builder(rng))
    volume_mm3 = sum(
        float(np.prod(np.asarray(p.extent_mm, dtype=np.float64))) * (0.52 if p.primitive == "sphere"
                                                                     else 0.78 if p.primitive == "cylinder"
                                                                     else 1.0)
        for p in parts
    )
    # Hollow: a mug is a shell, and billing it as solid ceramic would make every physics label wrong
    # in the same direction.
    mass = max(0.010, volume_mm3 * 0.30 * _DENSITY_KG_PER_MM3)
    # The id names a slot, not a shape, and that has a cost. `index` is the object's position within
    # its scene, so every scene's slot 0 mug is `comp_00000_mug` while its dimensions are drawn
    # fresh. `build_manifest` dedups first-wins, and both the renderer and the labeller resolve
    # geometry through that one record, so every later mug in the same slot is placed, settled and
    # labelled as the first one: a different shape and a different mass wearing one name.
    #
    # It also caps diversity. Slot indices run over the objects in a scene, so a corpus holds at most
    # (objects per scene) x (kinds) distinct composites however many scenes are rendered.
    #
    # `unique_id` is the repair and it is off by default: turning it on changes every id and
    # therefore the manifest hash, so an existing corpus would no longer rebuild from its own
    # provenance, which the evaluation refuses. A new render can opt in.
    identity = (f"{index:05d}" if not unique_id
                else f"{index:05d}{_shape_key(parts, mass)}")
    return CompositeAsset(
        asset_id=f"comp_{identity}_{kind}",
        composite_kind=kind,
        parts=parts,
        mass_kg=round(mass, 5),
        tags=(kind, *sorted(set(p.role for p in parts))),
    )


def _shape_key(parts: "tuple[CompositePart, ...]", mass: float) -> str:
    """A short digest of the geometry, so two different shapes cannot share a name.

    Content-based rather than a counter: a counter depends on draw order, so the same scene rendered
    by two shards would name the same shape differently, and a fold split keyed on the id would then
    separate an object from itself.
    """
    import hashlib  # noqa: PLC0415 (only this path needs it)

    blob = "|".join(f"{p.primitive}:{p.role}:{p.extent_mm}:{p.offset_mm}" for p in parts)
    digest = hashlib.sha1(f"{blob}|{mass:.6f}".encode()).hexdigest()
    # Digits only. `asset_group` folds a generated id to its family with
    # `^(proc|comp)_\d+_(?P<family>.+)$`; a hex digest does not match, every composite becomes its
    # own fold group, and a held-out split then leaks because two mugs land on opposite sides. The
    # id must stay `comp_<digits>_<kind>`.
    return f"{int(digest[:12], 16) % 10**10:010d}"


def sample_composite_asset(
    rng: np.random.Generator, kinds: "tuple[str, ...] | None" = None, *, index: int,
    unique_id: bool = False,
) -> CompositeAsset:
    """Draw one composite object. ``kinds`` defaults to every family."""
    available = tuple(kinds) if kinds else tuple(sorted(_BUILDERS))
    unknown = [k for k in available if k not in _BUILDERS]
    if unknown:
        raise ValueError(f"unknown composite kinds {unknown}; known: {sorted(_BUILDERS)}")
    return build_composite(str(rng.choice(list(available))), rng, index=index,
                           unique_id=unique_id)
