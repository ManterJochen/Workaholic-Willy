"""The procedural asset library: the only source with no third-party licence anywhere in the chain.

Four families, all parametric, all licensed ``own``:

* primitive: boxes, cylinders, spheres, capsules. The colour, shape and size axes the perception
  routes are separated on, and the only family whose geometry is exactly known.
* industrial: tube sections, angle brackets, flanges, plates, fasteners. The hard case for depth
  sensing (specular metal) and for grasp planning.
* packaging: cartons, blisters, pouches. The bin-picking and suction domain. A pouch is approximated
  as a flattened, rounded box: soft-body physics is out of scope, and a rigid approximation labelled
  as one is better than a soft body that is secretly rigid.
* vessel: bottles, cups, cans, bowls. Overlaps with scanned household sets, but parametric, so
  proportions vary continuously instead of coming in a thousand fixed flavours.

Every entry is a description, not a mesh. The renderer authors geometry from ``kind`` and
``extent_mm``; nothing here imports a mesh library, so the library is importable and testable on any
machine. Sizes reach past the 2F-85's 85 mm aperture on purpose, because an object too big for the
jaw is a real object and a real suction target, and each asset carries
:attr:`ProceduralAsset.jaw_graspable` so the fact is labelled rather than filtered away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from datagen.assets.manifest import AssetRecord

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence

__all__ = [
    "FAMILY_KINDS", "MAX_JAW_APERTURE_MM", "PRIMITIVE_FOR_KIND", "ProceduralAsset",
    "primitive_for_kind", "sample_procedural_asset",
]

#: The shape vocabulary per family. ``kind`` is what the renderer switches on, so it stays a small,
#: closed set of things a renderer can actually author from parameters.
FAMILY_KINDS: dict[str, tuple[str, ...]] = {
    "primitive": ("box", "cylinder", "sphere", "capsule"),
    "industrial": ("tube", "bracket", "flange", "plate", "fastener"),
    "packaging": ("carton", "blister", "pouch"),
    "vessel": ("bottle", "cup", "can", "bowl"),
}

#: Sixteen kind words, three actual solids. The renderer authors a USD sphere, a USD cylinder or a
#: scaled USD cube, so a "flange" is a cuboid and a "bowl" is a cuboid, however the word reads. That
#: limit is written here rather than buried in the renderer, because the grasp labels in
#: :mod:`datagen.grasps` compute exact contacts from these solids. If this table and the renderer's
#: authoring ever disagreed, the labels would describe a shape that was never rendered and every
#: recall number built on them would be wrong. Both sides import from here, and the renderer must
#: switch on nothing else.
PRIMITIVE_FOR_KIND: dict[str, str] = {
    "sphere": "sphere",
    "cylinder": "cylinder", "can": "cylinder", "cup": "cylinder",
    "tube": "cylinder", "bottle": "cylinder", "capsule": "cylinder",
}


def primitive_for_kind(kind: str) -> str:
    """Which solid the renderer actually authors for ``kind``; ``box`` for everything unlisted."""
    return PRIMITIVE_FOR_KIND.get(kind, "box")


#: Size band. The lower bound keeps objects above the depth camera's usable detail. The upper bound
#: is deliberately above the 2F-85's 85 mm aperture: a big carton is a real object, a real
#: distractor and a real suction target, and a library that could not produce one would quietly
#: teach every downstream model that the world is jaw-sized. What must not happen is pretending such
#: an object is jaw-pickable, hence :attr:`ProceduralAsset.jaw_graspable`, a label not a filter.
_MIN_EXTENT_MM = 25.0
_MAX_EXTENT_MM = 120.0

#: A Robotiq 2F-85 opens to 85 mm. An object whose smallest extent exceeds that cannot be jaw-grasped
#: from any direction, whatever the grasp planner thinks.
MAX_JAW_APERTURE_MM = 85.0

#: Rough density per family in kg/mm^3, from g/cm^3: plastic-ish, steel-ish, cardboard-ish.
_DENSITY_KG_PER_MM3: dict[str, float] = {
    "primitive": 0.9e-6,
    "industrial": 4.0e-6,
    "packaging": 0.25e-6,
    "vessel": 0.6e-6,
}


@dataclass(frozen=True, slots=True)
class ProceduralAsset:
    """One generated object description. Manifest-shaped, so it audits like any other asset row."""

    asset_id: str
    family: str
    kind: str
    extent_mm: tuple[float, float, float]
    mass_kg: float

    @property
    def jaw_graspable(self) -> bool:
        """Could a 2F-85 close on this at all? False for the deliberately oversized suction objects.

        A label, not a filter; see the note on the size band. It is also what ``GraspCalculator`` can
        be checked against: proposing a jaw grasp on an object whose every axis exceeds the aperture
        is a definite error, independent of any scoring opinion.
        """
        return min(self.extent_mm) <= MAX_JAW_APERTURE_MM

    @property
    def source(self) -> str:
        return "procedural"

    @property
    def license(self) -> str:
        return "own"

    def as_record(self) -> "AssetRecord":
        """This asset as the manifest stores it. See `SceneAsset.as_record` for why it lives here."""
        return AssetRecord(
            asset_id=self.asset_id, source=self.source, license=self.license,
            kind=self.kind, extent_mm=self.extent_mm, mass_kg=self.mass_kg,
            attribution="", tags=(self.family,),
        )

    def as_manifest_row(self) -> dict[str, str]:
        """The audit's view of this asset: the same shape as a row from any other source."""
        return {
            "id": self.asset_id,
            "source": self.source,
            "license": self.license,
            "attribution": "",
            "kind": self.kind,
        }


def _extent_for(kind: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """Dimensions that make the shape recognisable as that shape, rather than a random blob.

    A 'tube' whose length equals its diameter is a puck, and a 'plate' as thick as it is wide is a
    block. The proportions are what carry the shape word a referring expression will use, so they are
    constrained per kind rather than drawn independently on three axes.
    """
    base = float(rng.uniform(_MIN_EXTENT_MM, _MAX_EXTENT_MM))
    if kind in ("sphere",):
        return (base, base, base)
    if kind in ("cylinder", "can", "cup"):
        return (base * 0.5, base * 0.5, base)
    if kind in ("capsule", "tube", "bottle"):
        return (base * 0.35, base * 0.35, base)
    if kind in ("plate", "blister"):
        return (base, base * 0.75, max(_MIN_EXTENT_MM * 0.25, base * 0.12))
    if kind in ("pouch",):
        return (base, base * 0.7, max(_MIN_EXTENT_MM * 0.3, base * 0.2))
    if kind in ("fastener",):
        small = max(_MIN_EXTENT_MM * 0.4, base * 0.25)
        return (small, small, base * 0.6)
    if kind in ("bracket", "flange"):
        return (base, base * 0.8, base * 0.5)
    if kind in ("bowl",):
        return (base, base, base * 0.45)
    # box, carton and anything new: a general cuboid with independent but bounded sides
    return (
        base,
        float(np.clip(base * rng.uniform(0.5, 1.0), _MIN_EXTENT_MM, _MAX_EXTENT_MM)),
        float(np.clip(base * rng.uniform(0.4, 1.0), _MIN_EXTENT_MM, _MAX_EXTENT_MM)),
    )


def sample_procedural_asset(
    rng: np.random.Generator, families: "Sequence[str]", *, index: int,
) -> ProceduralAsset:
    """Draw one asset from ``families``. Deterministic given ``rng`` and ``index``.

    ``index`` only names the asset (``proc_0007_tube``); it never influences the geometry, so a scene
    can be re-sampled and the ids stay readable without becoming load-bearing.
    """
    unknown = [family for family in families if family not in FAMILY_KINDS]
    if unknown:
        raise ValueError(
            f"unknown procedural families {unknown}; known: {sorted(FAMILY_KINDS)}"
        )
    family = str(rng.choice(list(families)))
    kind = str(rng.choice(list(FAMILY_KINDS[family])))
    extent = _extent_for(kind, rng)
    volume_mm3 = float(extent[0] * extent[1] * extent[2])
    # A shape factor, so a sphere is not billed as the cuboid that bounds it.
    fill = 0.52 if kind in ("sphere", "capsule") else 0.78 if kind in ("cylinder", "can", "tube") else 1.0
    mass = max(0.005, volume_mm3 * fill * _DENSITY_KG_PER_MM3[family])
    return ProceduralAsset(
        asset_id=f"proc_{index:05d}_{kind}",
        family=family,
        kind=kind,
        extent_mm=(round(extent[0], 3), round(extent[1], 3), round(extent[2], 3)),
        mass_kg=round(mass, 5),
    )
