"""Real meshes as scene assets, measured rather than authored.

A procedural asset is authored: the generator picks an extent inside a band and the renderer builds
that solid. A mesh asset is the opposite. Its size is a fact about the object, and the generator's
job is to read it and decide whether the thing fits in the workspace at all.

Why a size limit exists. The placement area is 300 x 300 mm (`workspace.half_extents_mm` 150 x 150),
and a mesh collection holds objects longer than the whole area. `DEFAULT_MAX_MESH_EXTENT_MM` is
180 mm: that still fits two or three objects side by side, and it is a config knob rather than a
constant, because a cell with a bigger table has a different answer.

"Jaw-graspable" here is the same label the procedural side uses: the shortest axis fits the 85 mm
aperture (`MAX_JAW_APERTURE_MM`). It stays a label rather than a filter, for the reason
`procedural.py` gives, that a library which could not produce an oversized object would quietly
teach every downstream model that the world is jaw-sized.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Iterable, Mapping

from datagen.assets.library import ATTRIBUTIONS, LICENSES, MeshLibrary, declared_license
from datagen.assets.mesh_geometry import load_mesh_shape
from datagen.assets.manifest import AssetRecord
from datagen.assets.procedural import MAX_JAW_APERTURE_MM

__all__ = [
    "DEFAULT_MAX_MESH_EXTENT_MM",
    "MeshAsset",
    "MeshAssetBank",
    "measure_mesh",
]

#: Longest-axis limit for a real mesh to be placeable, in millimetres. Raising it keeps more of a
#: collection and lowers the share of what is kept that is jaw-graspable.
DEFAULT_MAX_MESH_EXTENT_MM: Final[float] = 180.0

#: Density used to estimate a mesh's mass from its volume, kg/mm^3. Roughly a light polymer: these
#: are household objects, and the physics label only needs the mass to be plausible and consistent,
#: not to match the real article. Aletheia's shake overrides mass outright for the same reason.
_DENSITY_KG_PER_MM3: Final[float] = 0.35e-6


@dataclass(frozen=True, slots=True)
class MeshAsset:
    """One real mesh the generator may place.

    The same shape as `ProceduralAsset` (`asset_id`, `kind`, `extent_mm`, `mass_kg`,
    `jaw_graspable`), so layout and rendering treat the two alike and only the renderer's geometry
    branch has to know the difference.
    """

    asset_id: str
    source: str
    mesh_path: str
    extent_mm: tuple[float, float, float]
    mass_kg: float
    #: Always ``"mesh"``. The procedural side puts a solid name here (``cylinder``, ``can``, ...) and
    #: the renderer switches on it; this is the value that selects the mesh branch.
    kind: str = "mesh"

    @property
    def family(self) -> str:
        """The collection this mesh came from, ``gso`` or ``ycb``.

        The procedural side uses `family` for its shape family (``vessel``, ``packaging``, ...) and
        the manifest records it as a tag either way. For a real mesh the collection is the most
        honest grouping available: nothing here authored its shape, so nothing here knows it.
        """
        return self.source

    @property
    def license(self) -> str:
        """What this mesh is licensed under.

        For `gso` and `ycb` it is the map, because those datasets publish collection-wide terms. For
        `custom` it is whatever the operator declared at import time, read back from beside the
        meshes: a map entry would be a guessed licence string that the fail-closed audit then treats
        as checked, which is the one failure mode a licence gate must not have.
        """
        if self.source in LICENSES:
            return LICENSES[self.source]
        return declared_license(Path(self.mesh_path).parent)[0]

    @property
    def attribution(self) -> str:
        if self.source in ATTRIBUTIONS:
            return ATTRIBUTIONS[self.source]
        declared, credit = declared_license(Path(self.mesh_path).parent)
        if not (declared or credit):
            return "Custom meshes supplied by the operator; no licence was declared at import."
        return f"Custom meshes supplied by the operator ({credit or 'no attribution'}), {declared}."

    @property
    def jaw_graspable(self) -> bool:
        """Could a 2F-85 close on this at all? A label, not a filter; see the module docstring."""
        return min(self.extent_mm) <= MAX_JAW_APERTURE_MM

    def as_record(self) -> AssetRecord:
        return AssetRecord(
            asset_id=self.asset_id,
            source=self.source,
            license=self.license,
            kind=self.kind,
            extent_mm=self.extent_mm,
            mass_kg=self.mass_kg,
            attribution=self.attribution,
            mesh_path=self.mesh_path,
            tags=(self.family,),
        )


@lru_cache(maxsize=2048)
def _bounds_mm(path: str) -> tuple[float, float, float] | None:
    """A mesh's axis-aligned extent in millimetres, or ``None`` when it cannot be read.

    Cached because the bank is measured once per run and a scene may draw the same object
    repeatedly: a trimesh load costs tens of milliseconds, which is nothing once and minutes across
    ten thousand scenes.
    """
    import trimesh  # noqa: PLC0415 (heavy, and only the real-mesh path needs it)

    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception:  # noqa: BLE001 (a corrupt mesh must skip the asset, never kill the run)
        return None
    bounds = getattr(mesh, "bounds", None)
    if bounds is None or len(getattr(mesh, "faces", ())) == 0:
        return None
    # Metres are assumed, and that assumption is the whole reason for the refusal below. The factor
    # is right for a mesh authored in metres and is applied unconditionally, so the moment a
    # customer brings a part authored in millimetres it becomes a silent factor of a million in
    # volume.
    extent = (np.asarray(bounds[1]) - np.asarray(bounds[0])) * 1000.0
    if not np.all(np.isfinite(extent)) or float(extent.min()) <= 0.0:
        return None
    return (float(extent[0]), float(extent[1]), float(extent[2]))


#: Above this the mesh cannot be what its units claim. A grasped object is a household or industrial
#: part; a hundred kilograms is not one, and a part that reads as heavier than that is a unit error
#: rather than a heavy part. Deliberately generous: the failure it catches is a factor of a million
#: in volume, not a factor of two.
_IMPOSSIBLE_MASS_KG: Final[float] = 100.0


class MeshUnitsError(ValueError):
    """A mesh whose implied mass says its units cannot be metres."""


def _refuse_impossible_units(asset_id: str, path, extent_mm, density_kg_per_mm3: float) -> None:
    """Refuse a mesh whose units cannot be what the loader assumed.

    Mass is the detector, on purpose. `_bounds_mm` multiplies every mesh by 1000, so a part authored
    in millimetres measures a thousand times too large in each axis, enters a physics engine a
    billion times too heavy, and every check downstream still passes.

    A longest axis in the hundreds of thousands of millimetres invites a shrug; a mass in billions
    of kilograms does not, which is why the refusal is phrased in that quantity.
    """
    volume = float(extent_mm[0] * extent_mm[1] * extent_mm[2]) * 0.4
    mass = volume * float(density_kg_per_mm3)
    if mass <= _IMPOSSIBLE_MASS_KG:
        return
    longest = max(float(v) for v in extent_mm)
    raise MeshUnitsError(
        f"{asset_id}: this mesh implies a mass of {mass:,.0f} kg ({longest:,.0f} mm on its longest "
        f"axis). Meshes are read as METRES and scaled by 1000; a part authored in MILLIMETRES lands a "
        f"thousand times too large in every axis and a billion times too heavy, and every check "
        f"downstream still passes. Re-export {Path(path).name} in metres, or scale it before import.")


def measure_mesh(
    asset_id: str, source: str, path: Path | str, *, density_kg_per_mm3: float = _DENSITY_KG_PER_MM3
) -> MeshAsset | None:
    """Read one mesh into a `MeshAsset`, or ``None`` if it is unreadable.

    Returning ``None`` rather than raising: one unreadable file must cost that file, not the
    dataset. The caller counts what it skipped, so a silently shrinking bank is still visible. A
    mesh whose units are impossible is the other case, and raises `MeshUnitsError`.
    """
    extent = _bounds_mm(str(path))
    if extent is None:
        return None
    _refuse_impossible_units(asset_id, path, extent, density_kg_per_mm3)
    volume_mm3 = float(extent[0] * extent[1] * extent[2])
    # 0.4 fill: an axis-aligned box overstates a real object's volume badly, and a household item is
    # mostly not solid. Plausible and consistent beats precise here.
    mass = max(0.005, volume_mm3 * 0.4 * density_kg_per_mm3)
    return MeshAsset(
        asset_id=asset_id,
        source=source,
        mesh_path=str(path),
        extent_mm=(round(extent[0], 3), round(extent[1], 3), round(extent[2], 3)),
        mass_kg=round(mass, 5),
    )


class MeshAssetBank:
    """The placeable real meshes, measured once and drawn from deterministically."""

    def __init__(
        self,
        library: MeshLibrary | None = None,
        *,
        max_extent_mm: float = DEFAULT_MAX_MESH_EXTENT_MM,
        require_closed: bool = True,
        only_asset_ids: "Iterable[str] | None" = None,
        max_jaw_span_mm: float | None = None,
        jaw_screen: "Mapping[str, int] | None" = None,
    ) -> None:
        self.library = library if library is not None else MeshLibrary()
        self.max_extent_mm = float(max_extent_mm)
        #: The other axis, and it is the one a jaw actually closes on. `max_extent_mm` bounds the
        #: longest axis, which is about fitting the workspace; this bounds the shortest, which is
        #: about fitting the gripper. They are not substitutes: a 250 mm broom handle is easy to
        #: grasp and a 90 mm cube is impossible, and only this filter tells them apart. The share of
        #: objects carrying a jaw label falls off sharply above the 85 mm aperture.
        #:
        #: It is not a better `max_extent_mm`, it is a different question, so it is a separate knob,
        #: and `None` (the default) keeps every existing bank byte-identical. Reaching for the
        #: length limit instead costs variety at the same expected yield.
        self.max_jaw_span_mm = None if max_jaw_span_mm is None else float(max_jaw_span_mm)
        #: The only way to build a held-out dataset, which is why this exists at all. A reference
        #: dataset whose fold groups all appear in the training corpus can never report a held-out
        #: number, whatever it says; restricting the draw to meshes the model never saw is what
        #: turns a recall test into a measurement.
        #:
        #: `None` means no restriction, and that is the default: an existing config draws exactly
        #: the bank it always drew.
        self.only_asset_ids = None if only_asset_ids is None else frozenset(only_asset_ids)
        #: Drop meshes that cannot be closed into a solid. On by default because an unclosable object
        #: cannot be labelled or collided; off only for callers that are measuring the bank itself and
        #: want to see what is in it before the filter.
        self.require_closed = bool(require_closed)
        #: The labeller's own verdict, keyed by asset id, replacing `max_jaw_span_mm` where it is
        #: given. `min(extent_mm) <= max_jaw_span_mm` asks whether the object's hull fits between
        #: the fingers; a jaw closes on a line through the object, so a round pot 86 mm across has
        #: plenty of chords under 85 and a 60 mm sphere has none longer than 60. The proxy is wrong
        #: in both directions: it rejects meshes the labeller finds grasps on, and accepts meshes it
        #: finds none on. Written by `screen-meshes`; `None` keeps the shipped proxy, so every
        #: existing bank is byte-identical.
        self.jaw_screen = None if jaw_screen is None else dict(jaw_screen)
        self._by_source: dict[str, list[MeshAsset]] = {}
        self.skipped: dict[str, int] = {}
        #: The `skipped` breakdown's most interesting term, kept separately: a mesh lost to an open
        #: surface is a different fact about the library than one lost for being too big.
        self.unclosable: dict[str, int] = {}
        #: Per source, how many meshes the `only_asset_ids` subset excluded. Reported separately so
        #: a held-out run does not read as a library full of broken files, and so a subset that
        #: matched nothing is visible as a number rather than as an empty bank nobody can explain.
        self.restricted: dict[str, int] = {}

    def load(self, source: str) -> list[MeshAsset]:
        """Measure every mesh of one source and keep those that fit. Idempotent."""
        if source in self._by_source:
            return self._by_source[source]

        kept: list[MeshAsset] = []
        unreadable = oversized = unclosable = restricted = 0
        for path in self.library.available(source):
            asset_id = f"{source}_{path.stem}"
            if self.only_asset_ids is not None and asset_id not in self.only_asset_ids:
                # Counted apart from the quality filters below: "not in the requested subset" is
                # not a defect in the mesh, and `skipped` is for defects.
                restricted += 1
                continue
            asset = measure_mesh(asset_id, source, path)
            if asset is None:
                unreadable += 1
                continue
            if max(asset.extent_mm) > self.max_extent_mm:
                oversized += 1
                continue
            if self.jaw_screen is not None:
                # The measured answer wins. A mesh absent from the screen has not been judged, and an
                # unjudged mesh is not a rejected one: it falls through to the proxy below rather
                # than being silently dropped, so a stale screen shrinks the bank visibly instead of
                # quietly.
                verdict = self.jaw_screen.get(asset.asset_id)
                if verdict is not None:
                    if verdict <= 0:
                        oversized += 1
                        continue
                elif self.max_jaw_span_mm is not None and min(asset.extent_mm) > self.max_jaw_span_mm:
                    oversized += 1
                    continue
            elif self.max_jaw_span_mm is not None and min(asset.extent_mm) > self.max_jaw_span_mm:
                # Counted with the oversized: both are "the cell cannot use this", and splitting them
                # would imply the mesh is defective, which it is not.
                oversized += 1
                continue
            if self.require_closed and load_mesh_shape(str(path)) is None:
                # Dropped here, where the bank is built, rather than later. An object that cannot be
                # closed into a solid cannot be labelled and cannot be collided, so leaving it in
                # the bank would place it in scenes and then omit it from everything downstream: a
                # scene whose picture contains an object its labels do not.
                #
                # Costs about 200 ms per mesh, once per process (`load_mesh_shape` is cached and the
                # labeller reuses the same shapes), and only for meshes that already passed the size
                # filter.
                unclosable += 1
                continue
            kept.append(asset)

        self._by_source[source] = kept
        self.skipped[source] = unreadable + oversized + unclosable
        self.restricted[source] = restricted
        self.unclosable[source] = unclosable
        return kept

    def available(self, source: str) -> list[MeshAsset]:
        return self.load(source)

    def is_empty(self, source: str) -> bool:
        return not self.load(source)

    def draw(self, rng: "np.random.Generator", source: str) -> MeshAsset | None:
        """One asset, uniformly. ``None`` when this source has nothing placeable."""
        assets = self.load(source)
        if not assets:
            return None
        return assets[int(rng.integers(len(assets)))]
