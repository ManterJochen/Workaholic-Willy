"""What a scene is, before anything renders it: frozen, unit-tagged, and Isaac-free.

A :class:`SceneSpec` is the complete description of one scene: which assets, where they start, which
cameras look at it, how the lighting was randomised. It is produced by :mod:`..scenes.layout` from a
seed alone, and consumed by the renderer. Nothing here imports Isaac, torch or a mesh library, so the
whole layout half of the generator is testable on a laptop, which is the point: a layout bug that
only shows up after a night of path tracing costs a night of path tracing.

Units follow the repository, not the renderer. Millimetres and XYZW quaternions everywhere; the
metres-and-WXYZ conversion happens inside the render module, at the boundary, the same way vendor
unit conventions stay inside a driver. A generator that spoke Isaac's units would quietly disagree
with every other measurement in this repository.

Spawn poses, not settled poses. These are where objects start, before physics runs. The settled
poses are a rendering output and are recorded separately: a spec plus a physics version reproduces
the scene, a spec alone does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # a manifest record is what this returns; importing it here would cycle.
    from datagen.assets.manifest import AssetRecord

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "BinWall",
    "CameraMount",
    "CameraPlacement",
    "DomainKind",
    "DomainRandomization",
    "ObjectPlacement",
    "SceneFamily",
    "SceneSpec",
]



class SceneFamily(StrEnum):
    """How the objects are arranged. Each is a different physics regime, not a different density."""

    #: 2 to 5 objects, well separated. The baseline: nothing touches, nothing occludes.
    SPARSE = "sparse"
    #: 4 to 8 objects touching in a single layer. Where segmentation merges masks and quietly fails.
    PACKED = "packed"
    #: 5 to 12 objects dropped from height. Real occlusion and stacking, expensive to settle.
    PILE = "pile"
    #: Objects inside a KLT bin with walls. Wall occlusion, depth shadows, the industrial domain.
    BIN = "bin"


@runtime_checkable
class SceneAsset(Protocol):
    """What scene layout and the manifest need from an object, whichever kind it is.

    Every member is a read-only property, and that is not cosmetic: a Protocol member written as
    `x: str` demands a settable attribute, so every frozen dataclass in this package would fail the
    check for a reason unrelated to the contract. These objects are immutable by design.

    `kind` is here because the manifest records it and the renderer switches on it: it is the one
    member where the two implementations genuinely differ (`cylinder` or `can` against `mesh`).
    """

    @property
    def asset_id(self) -> str:
        """Names the object in the scene record. Never load-bearing for geometry."""
        ...

    @property
    def source(self) -> str:
        """``procedural`` / ``gso`` / ``ycb``: what the licence audit keys on."""
        ...

    @property
    def license(self) -> str:
        """SPDX-ish identifier. ``own`` for procedural, ``CC-BY-4.0`` for the real collections."""
        ...

    @property
    def family(self) -> str:
        """Shape family for authored objects, collection for real ones. Recorded as a manifest tag."""
        ...

    @property
    def kind(self) -> str:
        """The solid the renderer authors, or ``mesh`` to load geometry from a file."""
        ...

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        """Axis-aligned size. Placement fits, stacks and spaces objects by this."""
        ...

    @property
    def mass_kg(self) -> float:
        """Reaches the physics as a mass attribute."""
        ...

    @property
    def jaw_graspable(self) -> bool:
        """Could a 2F-85 close on this at all? A label, never a filter."""
        ...

    def as_record(self) -> "AssetRecord":
        """This asset as the `AssetRecord` the manifest stores. Every field it has, not a chosen few.

        On the Protocol rather than in `build_manifest`, because a manifest assembled field by field
        at the call site drops whatever the call site forgets. ``mesh_path`` and ``attribution`` are
        the two that cost most: without ``mesh_path`` the labeller cannot find the geometry, and
        without ``attribution`` the licence audit refuses every row of a mesh dataset.

        Asking the asset for its own record makes the next field arrive by construction, and this
        Protocol makes a type that forgets fail the check rather than the dataset.
        """
        ...



class DomainKind(StrEnum):
    """Whose workspace this is."""

    #: The real cell: the arm's reachable workspace, the cell's cameras, the cell's table.
    CELL = "cell"
    #: A generic tabletop with free camera placement and no robot: broader, less in-domain.
    TABLETOP = "tabletop"


class CameraMount(StrEnum):
    """How a camera gets to its pose, which decides whether the renderer can simply place it."""

    #: Authored directly at the given pose (the oblique pair, an overhead rig).
    FIXED = "fixed"
    #: Carried by the robot wrist: the renderer must drive the arm so the camera reaches this pose,
    #: and may fail to. Such a view is recorded as unreachable rather than silently placed anyway.
    WRIST = "wrist"


@dataclass(frozen=True, slots=True)
class ObjectPlacement:
    """One object's spawn state. ``asset_id`` indexes the manifest; geometry lives there, not here."""

    asset_id: str
    position_mm: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    scale: float = 1.0
    #: Base colour, RGB in [0, 1]. Procedural assets are tinted at author time; a scanned mesh keeps
    #: its own texture and ignores this.
    color_rgb: tuple[float, float, float] = (0.7, 0.7, 0.7)
    mass_kg: float = 0.2


@dataclass(frozen=True, slots=True)
class CameraPlacement:
    """One view. ``look_at_mm`` rather than a quaternion: aiming is what a rig actually specifies."""

    name: str
    mount: CameraMount
    position_mm: tuple[float, float, float]
    look_at_mm: tuple[float, float, float]
    resolution: tuple[int, int] = (640, 480)
    horizontal_fov_deg: float = 47.0


@dataclass(frozen=True, slots=True)
class BinWall:
    """A static wall, in the same form the safety layer's fixture boxes take (centre + half extents).

    One description feeds both the rendered prim and the collision fixture: a wall the camera sees
    but the planner does not is worse than no wall at all.
    """

    name: str
    center_mm: tuple[float, float, float]
    half_extents_mm: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DomainRandomization:
    """Every randomised appearance value, resolved at layout time so the scene is reproducible.

    Sampled here rather than at render time on purpose: a renderer that rolls its own dice makes the
    seed a half-truth, and two runs of "the same" dataset then differ in ways nobody can diff.
    """

    light_elevation_deg: float
    light_azimuth_deg: float
    light_intensity: float
    dome_intensity: float
    table_color_rgb: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """One complete, reproducible scene description. Seed in, this out, byte-for-byte."""

    scene_id: str
    seed: int
    family: SceneFamily
    domain: DomainKind
    objects: tuple[ObjectPlacement, ...]
    cameras: tuple[CameraPlacement, ...]
    randomization: DomainRandomization
    bin_walls: tuple[BinWall, ...] = field(default_factory=tuple)
    #: How far above the table objects are released from. Zero for analytically-placed families;
    #: non-zero for the pile, whose whole character comes from the drop.
    drop_height_mm: float = 0.0

    def asset_ids(self) -> tuple[str, ...]:
        """Every asset this scene needs: what the manifest resolves and attribution must list."""
        return tuple(dict.fromkeys(placement.asset_id for placement in self.objects))
