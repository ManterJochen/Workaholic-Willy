"""What the calculator sees beside the part it grasps: the scene's obstacles, under the support the parts stand on.

A prompt names one part, and the calculator used to see that part alone. On the owner's cell (2026-10-01) every prompt
grounded exactly one box, so SFE's obstacle grid was empty: it offered grasps whose open fingers stood inside the parts
beside the Zollstock, and the exact guard refused every one (fix plan RC2). Handed the neighbours, SFE refuses every
candidate on all five recorded looks.

So the calculator reads the depth the camera saw around the part, by the planner world's own rules
(:class:`SceneObstacleRules`, from the cell's tree):

* every pixel outside the part's mask grown by about 5 mm, the pixels behind a depth step left out;
* within 250 mm of the part, in BASE;
* not the support: where a support surface of the camera world's model covers a point (``support_surfaces``), it is an
  obstacle only above that surface's local reading plus its band; elsewhere only above the higher of the declared
  bench and the support the part stands on, plus the band;
* not within a declared body's band (``perceived.DECLARED_SURFACE_MM``);
* in a 25 mm voxel that holds at least ``min_points`` (12) points. A pre-filter may drop a speck; the guard holds it.

The same frame gives the push its evidence (the fix plan's contract 2): every point the camera world's rule keeps there,
with no voxel threshold, so a neighbour the voxel rule dropped still refuses a push's path.

And it says why a part got no grasp (:func:`why_no_grasp`): a neighbour the camera saw, a declared body, the support it
stands on, or a part too short for this hand. Only a neighbour reaches the push (``ALL_COLLIDED``).

Everything here is a pre-filter. The exact-mesh guard judges every sample of every path before anything is sent: a
wrong refusal here costs a pick, never a collision.

Pure: numpy, and SciPy's image tools for the mask growth and the cluster count. BASE millimetres throughout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.types.feedback import GraspFailureReason

__all__ = [
    "MASK_GROWTH_MM",
    "NO_GRASP_SAID",
    "REACH_MM",
    "SEEN_TOLERANCE_MM",
    "SOFT_SLACK_MM",
    "WHOLE_SLACK_MM",
    "DeclaredBox",
    "EnvelopeVerdict",
    "SceneObstacleRules",
    "SceneObstacles",
    "SeenEnvelope",
    "SeenInTheFrame",
    "corridor_seen_in",
    "envelope_verdicts",
    "least_part_height_mm",
    "no_grasp_said",
    "scene_obstacle_points",
    "seen_boxes",
    "why_no_grasp",
]

#: How far from the part, in BASE x and y, the camera's depth is read for obstacles, millimetres: past the hand's reach
#: at any approach the calculator offers (the open Hand-E spans 50 + 2 x 30 mm, its corridor runs 150 mm up).
REACH_MM: Final[float] = 250.0
#: How far the part's mask is grown before its neighbours are read, millimetres: the rim of a mask is the part's own
#: edge, read through depth-to-colour misregistration. Turned into pixels at the part's depth.
MASK_GROWTH_MM: Final[float] = 5.0
#: How much nearer than a point its pixel's measured depth may lie for the point to count as seen, millimetres (the fix
#: plan's contract 5): a depth ray that stops 5 mm short of a point has still looked at the space it stands in.
SEEN_TOLERANCE_MM: Final[float] = 5.0
#: The key of ``GraspResult.telemetry`` under which a result with no candidate says why, in one sentence
#: (:func:`no_grasp_said`).
NO_GRASP_SAID: Final[str] = "no_grasp_said"
#: SFE's tilt ladder and its height solve (``support_footprint.generate_support_footprint_grasps``), restated for the
#: least part height a hand grips (:func:`least_part_height_mm`); ``tests/test_the_support_is_no_neighbour.py`` holds
#: the two equal on the generator itself.
_SFE_TILTS_DEG: Final[tuple[float, ...]] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
#: The solve adds a height only where it lies this far under the part's top, millimetres (``need <= prism.z1 - 2``).
_SFE_TOP_SLACK_MM: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class DeclaredBox:
    """A box the guard holds as declared (``safety.self_collision.fixtures``), axis-aligned in BASE millimetres."""

    name: str
    centre_mm: tuple[float, float, float]
    half_extents_mm: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SceneObstacleRules:
    """The planner world's rules for what the camera saw beside a part, read once from the cell's tree.

    Build it with :meth:`from_robot_config`. Every number is the camera world's own, so the calculator and the planner
    world say the same thing about the same pixel.
    """

    #: The declared bench's top, BASE z millimetres; ``None`` where the cell declares none.
    bench_top_mm: float | None = None
    #: How far over the bench, or over a support's local reading, a point is still that surface, millimetres
    #: (``planning_world.perceived.plane_clearance_mm``).
    band_mm: float = 5.0
    #: The declared bodies the camera sees again (the bench slab, the fixtures the planning world includes, its meshes):
    #: a point within :attr:`declared_band_mm` of one is that body, not a neighbour. ``perceived.DeclaredBody``.
    declared: tuple[Any, ...] = ()
    #: ``perceived.DECLARED_SURFACE_MM``.
    declared_band_mm: float = 5.0
    #: A voxel needs this many points to be an obstacle (``perceived.min_points``).
    min_points: int = 12
    #: The voxel the points are counted in, millimetres (``perceived.cluster_voxel_mm``).
    cluster_voxel_mm: float = 25.0
    #: What the camera world grows every box of a neighbour by, millimetres (``perceived.margin_mm``).
    margin_mm: float = 15.0
    #: The height step a box of a neighbour's height map holds, millimetres (``perceived.voxel_size_mm``).
    voxel_size_mm: float = 10.0
    #: Whether the camera world carries a neighbour's boxes down to the bench (``perceived.floor_to_plane``).
    floor_to_plane: bool = True
    #: Every how many pixels the frame is read along each axis (``perceived.pixel_stride``), as the world reads it.
    pixel_stride: int = 2
    reach_mm: float = REACH_MM
    mask_growth_mm: float = MASK_GROWTH_MM
    #: What the guard keeps from a support's solid, a camera box (``self_collision.perceived_min_distance_mm``).
    support_distance_mm: float = 5.0
    #: The boxes the guard holds as declared, and what it keeps from them (``self_collision.min_distance_mm``).
    declared_boxes: tuple[DeclaredBox, ...] = ()
    declared_distance_mm: float = 10.0
    #: How far into a box of a named neighbour part a finger may come, as the camera world says
    #: (``WorldBuildTuning.part_soft_mm``): the margin and the finger contact where the fingers may touch parts
    #: (``perceived.fingers_touch_parts``), 0 where they may not.
    part_soft_mm: float = 0.0
    #: Whether the fingers keep the guard's distance from a support's reading and excess rather than from its solid
    #: (``perceived.fingers_to_the_support_reading``), as the guard holds them.
    fingers_to_the_reading: bool = False
    #: How much nearer than ``support_distance_mm`` a finger comes to what a support reads (``finger_floor_drop_mm``).
    finger_floor_drop_mm: float = 0.0
    #: Declared meshes that could not be read, each said: what the camera sees of them stays an obstacle.
    unread: tuple[str, ...] = field(default=())

    @classmethod
    def from_robot_config(cls, robot_config: Any) -> "SceneObstacleRules":
        """The rules a cell's tree states: the bench and its band, the declared bodies, the voxel rule, the distances.

        Read the way the camera world reads them (``camera_world_wiring._live_planner_world``): the bench is
        ``safety.planning_world.support_plane``'s top where the planning world is on, the declared bodies its cuboids
        and meshes, the voxel rule and the band ``planning_world.perceived``. A tree with no planning world has no
        bench: the support the part stands on bounds the rule alone.
        """
        from src.robot.safety.planning.live_world import _declared_bodies  # noqa: PLC0415 (the world's own reader)
        from src.robot.safety.planning.perceived import DECLARED_SURFACE_MM  # noqa: PLC0415
        from src.robot.safety.planning.world import build_planner_cuboids, build_planner_meshes  # noqa: PLC0415

        safety = robot_config.safety
        world = getattr(safety, "planning_world", None)
        perceived = getattr(world, "perceived", None)
        guard = safety.self_collision
        on = world is not None and bool(getattr(world, "enabled", False))
        plane = getattr(world, "support_plane", None) if on else None
        bodies: tuple[Any, ...] = ()
        unread: tuple[str, ...] = ()
        if on:
            # Every declared box, whatever the planner's slot count: a box left out of the planner is still declared.
            bodies, unread = _declared_bodies(build_planner_cuboids(world, guard.fixtures, max_cuboids=1_000_000),
                                              build_planner_meshes(world))
        boxes = tuple(
            DeclaredBox(str(fixture.name), tuple(float(v) for v in fixture.center_mm),  # type: ignore[arg-type]
                        tuple(float(v) for v in fixture.half_extents_mm))  # type: ignore[arg-type]
            for fixture in guard.fixtures if min(float(v) for v in fixture.half_extents_mm) > 0.0
        )
        return cls(
            bench_top_mm=None if plane is None else float(plane.height_mm),
            band_mm=float(getattr(perceived, "plane_clearance_mm", 5.0)),
            declared=tuple(bodies),
            declared_band_mm=float(DECLARED_SURFACE_MM),
            min_points=int(getattr(perceived, "min_points", 12)),
            cluster_voxel_mm=float(getattr(perceived, "cluster_voxel_mm", 25.0)),
            margin_mm=float(getattr(perceived, "margin_mm", 15.0)),
            voxel_size_mm=float(getattr(perceived, "voxel_size_mm", 10.0)),
            floor_to_plane=bool(getattr(perceived, "floor_to_plane", True)),
            pixel_stride=int(getattr(perceived, "pixel_stride", 2)),
            support_distance_mm=float(guard.perceived_min_distance_mm),
            declared_boxes=boxes,
            declared_distance_mm=float(guard.min_distance_mm),
            part_soft_mm=(float(getattr(perceived, "margin_mm", 15.0)) + float(getattr(perceived, "finger_contact_mm", 0.0))
                          if bool(getattr(perceived, "fingers_touch_parts", False)) else 0.0),
            fingers_to_the_reading=bool(getattr(perceived, "fingers_to_the_support_reading", False)),
            finger_floor_drop_mm=finger_floor_drop_mm(robot_config),
            unread=tuple(unread),
        )


@dataclass(frozen=True, slots=True, eq=False)
class SceneObstacles:
    """What one frame shows beside a part, by :class:`SceneObstacleRules`."""

    #: The obstacles the calculator plans against, ``(N, 3)`` BASE millimetres: the voxel rule applied.
    points_base_mm: np.ndarray
    #: Every point the camera world's rule keeps within the reach, ``(M, 3)`` BASE millimetres: no voxel rule, no
    #: depth-step trim (the fix plan's contract 2, for the push).
    world_rule_base_mm: np.ndarray
    #: How many separate groups the obstacles form, voxels touching at a face, an edge or a corner joined.
    clusters: int
    #: What became of the pixels read: ``read`` within the reach, then ``support`` (a support surface's own),
    #: ``below`` (under the bench or the part's support plus the band), ``declared`` (a declared body's band),
    #: ``speck`` (a voxel under ``min_points``), ``kept``.
    counts: Mapping[str, int]
    #: The pixels the part's mask was grown by.
    grown_px: int = 0
    #: The boxes the camera world builds of :attr:`world_rule_base_mm` (``perceived.SeenBox``, its margin in them):
    #: what the guard holds for these neighbours and keeps ``support_distance_mm`` from (:class:`SeenEnvelope`).
    boxes: tuple[Any, ...] = ()
    #: Which of :attr:`points_base_mm` are a named neighbour part's, a finger may come to (``rules.part_soft_mm``);
    #: ``None`` where none are.
    part_points: "np.ndarray | None" = None
    #: Which of :attr:`world_rule_base_mm` are a named neighbour part's, as :attr:`part_points`; ``None`` where none are.
    world_rule_part: "np.ndarray | None" = None

    @classmethod
    def none(cls) -> "SceneObstacles":
        empty = np.zeros((0, 3), dtype=np.float64)
        return cls(points_base_mm=empty, world_rule_base_mm=empty.copy(), clusters=0, counts={})

    def render(self) -> str:
        """One clause, for a log line: how many points, in how many groups, and what was not an obstacle."""
        c = dict(self.counts)
        return (f"{int(self.points_base_mm.shape[0])} obstacle point(s) in {self.clusters} cluster(s) beside the part "
                f"(read {c.get('read', 0)} within reach; support {c.get('support', 0)}, below the support "
                f"{c.get('below', 0)}, declared {c.get('declared', 0)}, specks {c.get('speck', 0)}; "
                f"{int(self.world_rule_base_mm.shape[0])} kept by the world's rule)")


def _matrix_of(intrinsics: Any) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` of a 3x3 or of anything with those four attributes."""
    if all(hasattr(intrinsics, name) for name in ("fx", "fy", "cx", "cy")):
        return float(intrinsics.fx), float(intrinsics.fy), float(intrinsics.cx), float(intrinsics.cy)
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def _grown(mask: np.ndarray, pixels: int) -> np.ndarray:
    """``mask`` grown by a disc of ``pixels``, worked on the mask's own window."""
    from scipy import ndimage  # noqa: PLC0415 (only the calculator's scene path pays for it)

    rows, cols = np.nonzero(mask)
    out = np.zeros(mask.shape, dtype=bool)
    if rows.size == 0:
        return out
    r0, r1 = max(int(rows.min()) - pixels, 0), min(int(rows.max()) + pixels + 1, mask.shape[0])
    c0, c1 = max(int(cols.min()) - pixels, 0), min(int(cols.max()) + pixels + 1, mask.shape[1])
    yy, xx = np.mgrid[-pixels:pixels + 1, -pixels:pixels + 1]
    disc = (xx * xx + yy * yy) <= pixels * pixels
    out[r0:r1, c0:c1] = ndimage.binary_dilation(mask[r0:r1, c0:c1], structure=disc)
    return out


def _voxel_groups(points: np.ndarray, voxel_mm: float, min_points: int) -> tuple[np.ndarray, int]:
    """Which points stand in a voxel of at least ``min_points``, and how many groups those voxels form."""
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool), 0
    keys = np.floor(points / float(voxel_mm)).astype(np.int64)
    cells, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    full = counts >= int(min_points)
    keep = full[inverse]
    if not full.any():
        return keep, 0
    from scipy import ndimage  # noqa: PLC0415

    kept = cells[full]
    low = kept.min(axis=0)
    grid = np.zeros(tuple(int(v) for v in kept.max(axis=0) - low + 1), dtype=bool)
    grid[tuple((kept - low).T)] = True
    _, groups = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    return keep, int(groups)


def scene_obstacle_points(
    depth_mm: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: Any,
    camera_to_base: np.ndarray,
    *,
    rules: SceneObstacleRules,
    support_height_mm: float | None,
    support_model: Any = None,
    part_mask: "np.ndarray | None" = None,
) -> SceneObstacles:
    """The obstacles one frame shows beside the part ``target_mask`` covers, by ``rules``.

    ``depth_mm`` is the frame's depth, CAMERA millimetres; ``camera_to_base`` the 4x4 that places it.
    ``support_height_mm`` is the support the part stands on as the pick resolved it (BASE z); ``support_model`` the
    camera world's support model of the same frames (``support_surfaces.SupportModel``) or ``None``: where one of its
    surfaces covers a point, that surface's local reading decides (``is_support_point``, never "inside a solid").

    An empty answer where the mask holds no measured pixel: there is nothing to grow it from, and a frame read without
    the part left out would take the part for its own neighbour.

    ``part_mask`` is the pixels of the other parts the detector named in the frame: their points are a neighbour part's,
    which a finger may come to (``rules.part_soft_mm``), and their boxes are built apart, soft (``SeenBox.soft_mm``).
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    mask = np.asarray(target_mask).astype(bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(f"the depth and the part's mask must be one 2-D shape, got {depth.shape} and {mask.shape}")
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    under = mask & valid
    if not under.any():
        return SceneObstacles.none()
    fx, fy, cx, cy = _matrix_of(intrinsics)
    transform = np.asarray(camera_to_base, dtype=np.float64).reshape(4, 4)
    turn, shift = transform[:3, :3], transform[:3, 3]
    part_depth = float(np.median(depth[under]))
    grow = max(1, int(math.ceil(float(rules.mask_growth_mm) * fx / max(part_depth, 1.0))))
    around = valid & ~_grown(mask, grow)
    behind = pixels_behind_depth_steps(around, depth)

    # Where the part is: the median of its own pixels in BASE.
    rows_t, cols_t = np.nonzero(under)
    z_t = depth[rows_t, cols_t]
    part = np.column_stack(((cols_t - cx) * z_t / fx, (rows_t - cy) * z_t / fy, z_t)) @ turn.T + shift
    centre = np.median(part[:, :2], axis=0)

    stride = max(1, int(rules.pixel_stride))
    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    take = around[rows, cols]
    rows, cols = rows[take], cols[take]
    z = depth[rows, cols]
    camera = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z))
    points = camera @ turn.T + shift
    near = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) < float(rules.reach_mm)
    points, trimmed, camera = points[near], behind[rows[near], cols[near]], camera[near]
    on_a_part = (np.zeros(points.shape[0], dtype=bool) if part_mask is None or float(rules.part_soft_mm) <= 0.0
                 else np.asarray(part_mask).astype(bool)[rows[near], cols[near]])
    counts: dict[str, int] = {"read": int(points.shape[0])}

    # The support rule (contract 7): a surface the model holds decides over its own pixels, the bench and the part's
    # support decide elsewhere.
    band = float(rules.band_mm)
    covered = np.zeros(points.shape[0], dtype=bool)
    surface = np.zeros(points.shape[0], dtype=bool)
    if support_model is not None and points.shape[0]:
        planes = support_model.local_plane_under(points[:, :2])
        if planes is not None:
            planes = np.asarray(planes, dtype=np.float64)
            covered = np.isfinite(planes[:, 0])
            reading = np.full(points.shape[0], np.inf)
            nx, ny, nz, d = (planes[covered, i] for i in range(4))
            reading[covered] = (d - nx * points[covered, 0] - ny * points[covered, 1]) / nz
            own = np.asarray(support_model.is_support_point(points), dtype=bool)
            surface = covered & (own | (points[:, 2] <= reading))
    floors = [float(v) for v in (rules.bench_top_mm, support_height_mm) if v is not None]
    floor = (max(floors) if floors else -math.inf) + band
    below = ~covered & (points[:, 2] <= floor)
    bench_floor = (float(rules.bench_top_mm) if rules.bench_top_mm is not None
                   else (float(support_height_mm) if support_height_mm is not None else -math.inf)) + band
    below_bench = ~covered & (points[:, 2] <= bench_floor)
    declared = np.zeros(points.shape[0], dtype=bool)
    for body in rules.declared:
        declared |= np.asarray(body.distance_mm(points), dtype=np.float64) <= float(rules.declared_band_mm)
    counts["support"] = int(surface.sum())
    counts["below"] = int((below & ~surface).sum())
    counts["declared"] = int((declared & ~surface & ~below).sum())

    world_rule = points[~surface & ~below_bench & ~declared]
    world_rule_part = on_a_part[~surface & ~below_bench & ~declared]
    # The boxes of the neighbours: what stands over the support the part stands on and over what its solids erase, the
    # pixels behind a depth step left out as the obstacles leave them (a mixed pixel at the part's own rim is no
    # neighbour), thinned to one point a voxel
    # as the world thins what it reads before it builds a box (``perceived._base_points``): the first pixel read in
    # each cell of the camera's own frame.
    beside = ~surface & ~below & ~declared & ~trimmed
    if support_model is not None and hasattr(support_model, "erased_by_support") and points.shape[0]:
        # What a support's solid holds up to its erasure top leaves the world as that support's, its noise with it.
        beside &= ~np.asarray(support_model.erased_by_support(points), dtype=bool).reshape(-1)
    first = _first_in_each_voxel(camera[beside], float(rules.voxel_size_mm))
    thinned, thinned_part = points[beside][first], on_a_part[beside][first]
    wanted = ~surface & ~below & ~declared & ~trimmed
    candidates, candidates_part = points[wanted], on_a_part[wanted]
    keep, groups = _voxel_groups(candidates, float(rules.cluster_voxel_mm), int(rules.min_points))
    kept, kept_part = candidates[keep], candidates_part[keep]
    counts["speck"] = int(candidates.shape[0] - kept.shape[0])
    counts["kept"] = int(kept.shape[0])
    # A named neighbour's points make its own boxes, soft for the fingers; everything else makes the rest, whole.
    boxes = seen_boxes(thinned[~thinned_part], rules)
    if bool(thinned_part.any()):
        boxes = boxes + soft_boxes(seen_boxes(thinned[thinned_part], rules), float(rules.part_soft_mm))
    return SceneObstacles(points_base_mm=kept, world_rule_base_mm=world_rule, clusters=groups, counts=counts,
                          grown_px=grow, boxes=boxes, part_points=kept_part,
                          world_rule_part=world_rule_part if bool(world_rule_part.any()) else None)


def _first_in_each_voxel(camera_mm: np.ndarray, voxel_mm: float) -> np.ndarray:
    """The indices of the first point in each ``voxel_mm`` cell of the camera's frame, in the order they were read,
    as ``perceived._base_points`` thins a frame: cells whole multiples of the voxel, so the same pixels stand for the
    same cells here and there."""
    if camera_mm.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    cells = np.floor(camera_mm / float(voxel_mm)).astype(np.int64)
    cells -= cells.min(axis=0)
    span = cells.max(axis=0) + 1
    keys = (cells[:, 0] * span[1] + cells[:, 1]) * span[2] + cells[:, 2]
    _, first = np.unique(keys, return_index=True)
    return np.sort(first)


def seen_boxes(points_base_mm: np.ndarray, rules: SceneObstacleRules) -> tuple[Any, ...]:
    """The boxes the camera world builds of the neighbours' points by ``rules`` (``perceived.seen_part_boxes``): each
    cluster of at least ``min_points`` points a height map of columns, grown by ``margin_mm``, carried down to the bench
    where the world carries its boxes down. The calculator keeps the guard's distance from them, so it offers no grasp
    the guard refuses beside a neighbour once the arm has moved (the owner, 2026-10-03)."""
    from src.robot.safety.planning.perceived import seen_part_boxes  # noqa: PLC0415

    floor = rules.bench_top_mm if rules.floor_to_plane else None
    return seen_part_boxes(points_base_mm, margin_mm=float(rules.margin_mm),
                           cluster_voxel_mm=float(rules.cluster_voxel_mm), voxel_size_mm=float(rules.voxel_size_mm),
                           floor_mm=None if floor is None else float(floor), min_points=int(rules.min_points))


def soft_boxes(boxes: Sequence[Any], soft_mm: float) -> tuple[Any, ...]:
    """``boxes`` of a named neighbour part, each soft by ``soft_mm`` for the fingers where it is no larger than a part
    (``perceived.PART_SIZED_MM``), as the camera world marks them (``PerceivedBox.soft_mm``)."""
    from dataclasses import replace as _replace  # noqa: PLC0415

    from src.robot.safety.planning.perceived import PART_SIZED_MM  # noqa: PLC0415

    across, tall = PART_SIZED_MM
    out = []
    for box in boxes:
        half = np.asarray(box.half_extents_mm, dtype=np.float64)
        sized = 2.0 * max(float(half[0]), float(half[1])) <= across and 2.0 * float(half[2]) <= tall
        out.append(_replace(box, soft_mm=float(soft_mm)) if soft_mm > 0.0 and sized else box)
    return tuple(out)


#: How far off a named part's measured surface the calculator plans a finger, millimetres, where the guard holds it to
#: the surface at no distance (``SeenBox.soft_mm``, the owner, 2026-10-05: "Finger dürfen streifen"). The guard's boxes
#: are built from every frame the pick kept and the calculator's from the one it judged, so they stand a little apart:
#: planned at no distance, 11 of 11 grasps in a pile were refused 0.02 to 0.21 mm inside a neighbour's surface (the
#: grasp bench, 2026-10-06). The finger still comes to within this of the part.
SOFT_SLACK_MM = 1.0
#: How far past the guard's distance the calculator plans the open hand from a whole box, millimetres, for the same
#: reason: the guard's boxes, built from every frame the pick kept and merged to fit its slots, stand a little apart
#: from the calculator's. On a tray the guard refused the line down to four grasps in a row 0.18 to 0.68 mm short of
#: its 3 mm (the grasp bench, 2026-10-06), and at 1 mm of slack still lines down 2.82 to 2.99 mm from a box, the hand
#: planned 4 mm off it (the bench on the cell's tree, 2026-10-07), each a try. 2 mm was measured against it on the ten
#: tightest scenes with the cell's own depth (2026-10-07): 4 picked against 5, the tray's 30 mm cube lost, so 1 mm
#: stays: a refused try costs a quarter of a second now, a grasp never offered costs the pick.
WHOLE_SLACK_MM = 1.0
#: How far past the distance a pair's bounding spheres have to stand before :meth:`SeenEnvelope.refuses` leaves the
#: pair unmeasured, millimetres. The spheres' gap is a lower bound on the boxes' distance, so any pair beyond the
#: distance could be left out; this much more keeps a rounding in either sum from ever deciding a verdict.
_SPHERE_SLACK_MM: Final[float] = 1e-6
#: How near its limit a number :meth:`SeenEnvelope.refuses_many` or :meth:`SeenInTheFrame.unseen_many` reads may lie
#: before its answer for that hand is "unsure" (2), millimetres or pixels. They answer for many hands at once, and a
#: product over many rounds its last bit, about 1e-13 mm here, as it likes; within this much of a limit the caller asks
#: the one-hand call (:meth:`SeenEnvelope.refuses`, :meth:`SeenInTheFrame.__call__`), which decides as it always did.
_MANY_SURE: Final[float] = 1e-6


@dataclass(frozen=True, slots=True, eq=False)
class SeenEnvelope:
    """The open hand against the boxes the camera world holds for the neighbours one frame shows.

    The guard keeps ``distance_mm`` (``perceived_min_distance_mm``) from every camera box. A grasp whose open hand comes
    nearer to one of these boxes, at the grasp or anywhere on its way in along the approach (from the nearest to the
    farthest of ``way_in_mm`` back from it, every place between), would be refused there once the arm has moved: the
    calculator does not offer it. The hand is ``gripper_model.collision_boxes(open_width_mm)``, which holds the hand's
    meshes, as :func:`envelope_verdicts` places it: the rotation's columns are the closing axis, the binormal and the
    approach.
    """

    boxes: tuple[Any, ...]
    hand_centres: np.ndarray
    hand_halves: np.ndarray
    distance_mm: float
    way_in_mm: tuple[float, ...] = (0.0, 40.0, 80.0)
    #: Which of the hand's boxes are a finger, which may come to a soft box's measured surface (``SeenBox.soft_mm``).
    hand_fingers: "np.ndarray | None" = None
    #: What :meth:`refuses` reads, worked out once from the fields above (``__post_init__``) rather than on every build
    #: SFE asks about: the hand's boxes swept along the way in, their bounding spheres' radii and which are a finger;
    #: every box whole, its sphere and its soft; and the soft boxes as a finger meets them, their soft taken off.
    _swept_centres: np.ndarray = field(init=False, repr=False)
    _swept_halves: np.ndarray = field(init=False, repr=False)
    _swept_radii: np.ndarray = field(init=False, repr=False)
    _fingers: np.ndarray = field(init=False, repr=False)
    _box_centres: np.ndarray = field(init=False, repr=False)
    _box_turns: np.ndarray = field(init=False, repr=False)
    _box_halves: np.ndarray = field(init=False, repr=False)
    _box_radii: np.ndarray = field(init=False, repr=False)
    _whole_pairs: np.ndarray = field(init=False, repr=False)
    _soft_centres: np.ndarray = field(init=False, repr=False)
    _soft_turns: np.ndarray = field(init=False, repr=False)
    _soft_halves: np.ndarray = field(init=False, repr=False)
    _soft_radii: np.ndarray = field(init=False, repr=False)
    #: Whether every box's rotation is a rotation, its axes square and of unit length: only then is a box's gap along a
    #: face normal a bound on its distance, which :meth:`refuses_many` measures by.
    _square: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Work out once what :meth:`refuses` reads, by the arithmetic :meth:`_least` works it out with per call."""
        fingers = (np.asarray(self.hand_fingers, dtype=bool).reshape(-1) if self.hand_fingers is not None
                   else np.zeros(np.asarray(self.hand_centres).shape[0], dtype=bool))
        front, back = (min(self.way_in_mm), max(self.way_in_mm)) if self.way_in_mm else (0.0, 0.0)
        swept_centres = (np.asarray(self.hand_centres, dtype=np.float64).reshape(-1, 3)
                         - np.array([0.0, 0.0, (front + back) / 2.0]))
        swept_halves = (np.asarray(self.hand_halves, dtype=np.float64).reshape(-1, 3)
                        + np.array([0.0, 0.0, (back - front) / 2.0]))
        softs = np.array([float(getattr(b, "soft_mm", 0.0) or 0.0) for b in self.boxes], dtype=np.float64)
        centres = np.array([b.centre_mm for b in self.boxes], dtype=np.float64).reshape(-1, 3)
        turns = np.array([b.rotation for b in self.boxes], dtype=np.float64).reshape(-1, 3, 3)
        halves = np.array([b.half_extents_mm for b in self.boxes], dtype=np.float64).reshape(-1, 3)
        soft = softs > 0.0
        # A soft box as a finger meets it: its soft taken off its sides and its top, as :meth:`_least` takes it off.
        lowered = np.minimum(softs, np.maximum(halves[:, 2] - 0.5, 0.0) * 2.0)
        soft_halves = halves.copy()
        soft_halves[:, :2] = np.maximum(soft_halves[:, :2] - softs[:, None], 0.5)
        soft_halves[:, 2] -= lowered / 2.0
        soft_centres = centres - turns[:, :, 2] * (lowered / 2.0)[:, None]
        values = {
            "_swept_centres": swept_centres, "_swept_halves": swept_halves,
            "_swept_radii": np.linalg.norm(swept_halves, axis=1), "_fingers": fingers,
            "_box_centres": centres, "_box_turns": turns, "_box_halves": halves,
            "_box_radii": np.linalg.norm(halves, axis=1),
            # A finger against a soft box is the soft test's, never the whole one's.
            "_whole_pairs": ~(fingers[:, None] & soft[None, :]),
            "_soft_centres": soft_centres[soft], "_soft_turns": turns[soft], "_soft_halves": soft_halves[soft],
            "_soft_radii": np.linalg.norm(soft_halves[soft], axis=1),
            "_square": _all_square(turns),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def of(cls, boxes: Sequence[Any], *, gripper_model: Any, open_width_mm: float, distance_mm: float,
           way_in_mm: Sequence[float] = (0.0, 40.0, 80.0)) -> "SeenEnvelope | None":
        """The envelope over ``boxes``, or ``None`` where there is none to keep off."""
        if not boxes:
            return None
        hand = gripper_model.collision_boxes(float(open_width_mm))
        centres = np.array([(np.asarray(b.min_corner_mm) + np.asarray(b.max_corner_mm)) / 2.0 for b in hand])
        halves = np.array([(np.asarray(b.max_corner_mm) - np.asarray(b.min_corner_mm)) / 2.0 for b in hand])
        fingers = np.array([str(getattr(b, "label", "")).startswith("finger") for b in hand], dtype=bool)
        return cls(boxes=tuple(boxes), hand_centres=centres, hand_halves=halves, distance_mm=float(distance_mm),
                   way_in_mm=tuple(float(v) for v in way_in_mm), hand_fingers=fingers)

    def least_mm(self, position: np.ndarray, rotation: np.ndarray) -> float:
        """The open hand's least distance to the boxes, at ``position`` and back along the approach, millimetres: every
        box whole for every part of the hand but a finger against a soft box, which :meth:`into_parts_mm` asks."""
        return self._least(position, rotation, soft=False)

    def into_parts_mm(self, position: np.ndarray, rotation: np.ndarray) -> float:
        """The open fingers' least distance to the measured surface of every soft box (a named neighbour part, its
        ``soft_mm`` taken off its sides and its top), at ``position`` and back along the approach; ``inf`` with none."""
        return self._least(position, rotation, soft=True)

    def _least(self, position: np.ndarray, rotation: np.ndarray, *, soft: bool) -> float:
        turn = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        at = np.asarray(position, dtype=np.float64).reshape(3)
        softs = np.array([float(getattr(b, "soft_mm", 0.0) or 0.0) for b in self.boxes], dtype=np.float64)
        fingers = (self.hand_fingers if self.hand_fingers is not None
                   else np.zeros(self.hand_centres.shape[0], dtype=bool))
        centres_b = np.array([b.centre_mm for b in self.boxes], dtype=np.float64)
        turns_b = np.array([b.rotation for b in self.boxes], dtype=np.float64)
        halves_b = np.array([b.half_extents_mm for b in self.boxes], dtype=np.float64)
        if soft:
            # The fingers alone, against the soft boxes alone, each less its soft on its sides and its top.
            keep_b, keep_h = softs > 0.0, fingers
            lowered = np.minimum(softs, np.maximum(halves_b[:, 2] - 0.5, 0.0) * 2.0)
            halves_b = halves_b.copy()
            halves_b[:, :2] = np.maximum(halves_b[:, :2] - softs[:, None], 0.5)
            halves_b[:, 2] -= lowered / 2.0
            centres_b = centres_b - turns_b[:, :, 2] * (lowered / 2.0)[:, None]
            reach = 0.0
        else:
            keep_b, keep_h = np.ones(len(self.boxes), dtype=bool), np.ones(fingers.shape[0], dtype=bool)
            reach = self.distance_mm
        least = math.inf
        if not keep_b.any() or not keep_h.any():
            return least
        hand_centres, hand_halves = self.hand_centres[keep_h], self.hand_halves[keep_h]
        hand_fingers = fingers[keep_h]
        # The hand at every place of its way in, from the nearest to the farthest back along the approach: each of its
        # boxes stands square to the approach, so the way one sweeps is a box again, as long as the box and the way
        # together, and one distance answers for every place.
        front, back = (min(self.way_in_mm), max(self.way_in_mm)) if self.way_in_mm else (0.0, 0.0)
        hand_centres = hand_centres - np.array([0.0, 0.0, (front + back) / 2.0])
        hand_halves = hand_halves + np.array([0.0, 0.0, (back - front) / 2.0])
        reach_hand = float(np.max(np.linalg.norm(hand_centres, axis=1) + np.linalg.norm(hand_halves, axis=1)))
        reach_b = np.linalg.norm(halves_b, axis=1)
        near = keep_b & (np.linalg.norm(centres_b - at, axis=1) <= reach_hand + reach_b + reach + WHOLE_SLACK_MM + 1.0)
        if not near.any():
            return least
        hand_c = at + hand_centres @ turn.T
        n_hand, n_box = hand_c.shape[0], int(near.sum())
        distances = box_distances_mm(
            np.repeat(hand_c, n_box, axis=0), np.repeat(turn[None, :, :], n_hand * n_box, axis=0),
            np.repeat(hand_halves, n_box, axis=0), np.tile(centres_b[near], (n_hand, 1)),
            np.tile(turns_b[near], (n_hand, 1, 1)), np.tile(halves_b[near], (n_hand, 1))).reshape(n_hand, n_box)
        if not soft:
            # A finger against a soft box is :meth:`into_parts_mm`'s, not this one's.
            distances = np.where(hand_fingers[:, None] & (softs[near] > 0.0)[None, :], np.inf, distances)
        return min(least, float(distances.min()))

    def refuses(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        """Whether the open hand comes nearer the boxes than the guard keeps, at the grasp or on the way in: every part
        of it the guard's distance and :data:`WHOLE_SLACK_MM` from every whole box, and a finger no closer than
        :data:`SOFT_SLACK_MM` to a soft box's measured surface.

        The verdict of ``least_mm < distance_mm + WHOLE_SLACK_MM or into_parts_mm < SOFT_SLACK_MM``, measured only where
        it can be decided: a pair of a hand's box and a box whose bounding spheres stand apart by the limit and more
        cannot come nearer than the limit, so it is not measured. SFE asks this of every build it gets that far with. A
        part on the owner's cell had 38 boxes beside it (2026-10-08), and measuring the hand's 3 boxes against all 38 was
        95 % of a call's 1.7 to 2.0 ms on synthetic scenes of the Hand-E; measured only where it can refuse, a call takes
        0.6 to 0.7 ms, and the verdict is the same on every pose of the seeded scenes
        (``tests/test_the_seen_boxes_are_measured_only_where_they_could_refuse.py``).
        """
        turn = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        at = np.asarray(position, dtype=np.float64).reshape(3)
        hand = at + self._swept_centres @ turn.T
        if _any_nearer(hand, turn, self._swept_halves, self._swept_radii, self._box_centres, self._box_turns,
                       self._box_halves, self._box_radii, self._whole_pairs, self.distance_mm + WHOLE_SLACK_MM):
            return True
        fingers = self._fingers
        if not self._soft_centres.shape[0] or not fingers.any():
            return False
        return _any_nearer(hand[fingers], turn, self._swept_halves[fingers], self._swept_radii[fingers],
                           self._soft_centres, self._soft_turns, self._soft_halves, self._soft_radii, None,
                           SOFT_SLACK_MM)

    def refuses_many(self, positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        """:meth:`refuses` for many hands at once: ``(N, 3)`` positions and ``(N, 3, 3)`` rotations in, one ``int8`` per
        hand out: 0 where it keeps its distance, 1 where it does not, 2 where a distance it measured lies within
        :data:`_MANY_SURE` of its limit, which only :meth:`refuses` decides.

        Each hand is placed as :meth:`refuses` places it, by the same product one hand at a time (a stacked
        ``np.matmul``), so its boxes stand where they stand there to the bit. Of the pairs :meth:`refuses` measures it
        leaves out those whose gap along one of the two boxes' six face normals already stands at the limit and
        :data:`_MANY_SURE` or more (a projection never lengthens a distance, so such a pair is not nearer; the bound's
        own rounding stayed under 1.1e-13 mm on 200,000 random pairs, 2026-10-08), and it refuses a hand at once where
        the point of one of its boxes nearest a box's centre already stands nearer to that box than the limit less the
        band. The distances it measures many at once may round their last bit another way than one hand's call does,
        which the band covers. SFE asks this of every build of a closing line at once (``support_footprint._build_many``):
        on a boxed-in cube's search (2026-10-09) 183 of the 25,843 pairs the spheres leave were measured, and the search
        took 0.15 instead of 0.56 s.
        """
        turns = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
        at = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        hands = at[:, None, :] + np.matmul(self._swept_centres[None], turns.transpose(0, 2, 1))
        # The face normals bound a distance only where both boxes' axes are square: the camera's boxes and each hand.
        square = (np.abs(np.matmul(turns.transpose(0, 2, 1), turns) - np.eye(3)).max(axis=(1, 2)) <= 1e-9
                  if self._square else np.zeros(at.shape[0], dtype=bool))
        verdict = _nearer_many(hands, turns, self._swept_halves, self._swept_radii, self._box_centres,
                               self._box_turns, self._box_halves, self._box_radii, self._whole_pairs,
                               self.distance_mm + WHOLE_SLACK_MM, square)
        fingers = self._fingers
        if not self._soft_centres.shape[0] or not fingers.any():
            return verdict
        soft = _nearer_many(hands[:, fingers], turns, self._swept_halves[fingers], self._swept_radii[fingers],
                            self._soft_centres, self._soft_turns, self._soft_halves, self._soft_radii, None,
                            SOFT_SLACK_MM, square)
        # Refused by either test is refused; unsure in either, and refused by neither, is unsure.
        return np.where((verdict == 1) | (soft == 1), 1, np.maximum(verdict, soft)).astype(np.int8)


def _all_square(turns: np.ndarray) -> bool:
    """Whether every one of ``(N, 3, 3)`` rotations is one, its columns square and of unit length to 1e-9."""
    turns = np.asarray(turns, dtype=np.float64).reshape(-1, 3, 3)
    return bool(not turns.shape[0]
                or np.abs(np.matmul(turns.transpose(0, 2, 1), turns) - np.eye(3)).max() <= 1e-9)


def _face_gaps(centres_a: np.ndarray, turns_a: np.ndarray, halves_a: np.ndarray, centres_b: np.ndarray,
               turns_b: np.ndarray, halves_b: np.ndarray) -> np.ndarray:
    """The largest gap between each pair of oriented boxes along the six normals of their faces, millimetres: a lower
    bound on their distance (a projection never lengthens one) where both boxes' axes are square (:func:`_all_square`),
    negative where no face separates them."""
    normals = np.concatenate([turns_a, turns_b], axis=2)
    reach = ((halves_a[:, :, None] * np.abs(np.einsum("pki,pkl->pil", turns_a, normals))).sum(axis=1)
             + (halves_b[:, :, None] * np.abs(np.einsum("pki,pkl->pil", turns_b, normals))).sum(axis=1))
    return (np.abs(np.einsum("pk,pkl->pl", centres_b - centres_a, normals)) - reach).max(axis=1)


def _nearest_point_distances(centres_a: np.ndarray, turns_a: np.ndarray, halves_a: np.ndarray, centres_b: np.ndarray,
                             turns_b: np.ndarray, halves_b: np.ndarray) -> np.ndarray:
    """For each pair of oriented boxes, how far the first's point nearest the second's centre stands from the second,
    millimetres: an upper bound on their distance, the least over every point of the first, where the axes are square."""
    local = np.einsum("pk,pki->pi", centres_b - centres_a, turns_a)
    point = centres_a + np.einsum("pij,pj->pi", turns_a, np.clip(local, -halves_a, halves_a))
    return _point_box_distance(point[:, None, :], centres_b, turns_b, halves_b)[:, 0]


def _nearer_many(hands: np.ndarray, turns: np.ndarray, hand_halves: np.ndarray, hand_radii: np.ndarray,
                 box_centres: np.ndarray, box_turns: np.ndarray, box_halves: np.ndarray, box_radii: np.ndarray,
                 pairs: "np.ndarray | None", limit_mm: float, square: np.ndarray) -> np.ndarray:
    """:func:`_any_nearer` for many hands at once (``hands`` ``(N, H, 3)``, a turn each): 0 no box nearer than
    ``limit_mm``, 1 one surely is, 2 one measured within :data:`_MANY_SURE` of it. Where the hand's turn is ``square``
    (and the boxes' are), the pairs the spheres leave are culled once more by the gap along the two boxes' six face
    normals, a lower bound on their distance (:func:`_face_gaps`), and a hand one of whose pairs is surely nearer by the
    distance of its point nearest the box's centre, an upper bound (:func:`_nearest_point_distances`), is refused
    without measuring the rest: only the pairs neither bound decides are measured (``box_distances_mm``)."""
    out = np.zeros(hands.shape[0], dtype=np.int8)
    if not hands.shape[0] or not hands.shape[1] or not box_centres.shape[0]:
        return out
    gaps = (np.linalg.norm(hands[:, :, None, :] - box_centres[None, None, :, :], axis=3)
            - hand_radii[None, :, None] - box_radii[None, None, :])
    near = gaps < float(limit_mm) + _SPHERE_SLACK_MM
    if pairs is not None:
        near &= pairs[None, :, :]
    which, hand, box = np.nonzero(near)
    if not which.size:
        return out
    culled = square[which]
    if culled.any():
        keep = np.ones(which.size, dtype=bool)
        keep[np.nonzero(culled)[0]] = _face_gaps(
            hands[which[culled], hand[culled]], turns[which[culled]], hand_halves[hand[culled]],
            box_centres[box[culled]], box_turns[box[culled]], box_halves[box[culled]]) < float(limit_mm) + _MANY_SURE
        which, hand, box = which[keep], hand[keep], box[keep]
        if not which.size:
            return out
    # A pair surely nearer than the limit decides its hand without its exact distance: the distance of the hand's box's
    # point nearest the other box's centre from that box is never less than the two boxes' distance.
    within = square[which] & (_nearest_point_distances(
        hands[which, hand], turns[which], hand_halves[hand], box_centres[box], box_turns[box],
        box_halves[box]) < float(limit_mm) - _MANY_SURE)
    out[which[within]] = 1
    open_hand = out[which] == 0
    which, hand, box = which[open_hand], hand[open_hand], box[open_hand]
    if not which.size:
        return out
    distances = box_distances_mm(hands[which, hand], turns[which], hand_halves[hand], box_centres[box],
                                 box_turns[box], box_halves[box])
    unsure = np.abs(distances - float(limit_mm)) < _MANY_SURE
    out[which[unsure]] = 2
    out[which[(distances < float(limit_mm)) & ~unsure]] = 1
    return out


def _any_nearer(hand_centres: np.ndarray, turn: np.ndarray, hand_halves: np.ndarray, hand_radii: np.ndarray,
                box_centres: np.ndarray, box_turns: np.ndarray, box_halves: np.ndarray, box_radii: np.ndarray,
                pairs: "np.ndarray | None", limit_mm: float) -> bool:
    """Whether any of the hand's boxes (BASE centres, one ``turn``) comes nearer than ``limit_mm`` to any of the boxes,
    among the pairs ``pairs`` marks (every pair where ``None``). Only the pairs whose bounding spheres come nearer than
    the limit and :data:`_SPHERE_SLACK_MM` are measured (``box_distances_mm``); every other pair is that far apart at
    least."""
    if not hand_centres.shape[0] or not box_centres.shape[0]:
        return False
    gaps = (np.linalg.norm(hand_centres[:, None, :] - box_centres[None, :, :], axis=2)
            - hand_radii[:, None] - box_radii[None, :])
    near = gaps < float(limit_mm) + _SPHERE_SLACK_MM
    if pairs is not None:
        near &= pairs
    hands, boxes = np.nonzero(near)
    if not hands.size:
        return False
    distances = box_distances_mm(hand_centres[hands], np.repeat(turn[None, :, :], hands.size, axis=0),
                                 hand_halves[hands], box_centres[boxes], box_turns[boxes], box_halves[boxes])
    return bool((distances < float(limit_mm)).any())


@dataclass(frozen=True, eq=False)
class SeenInTheFrame:
    """Who says whether a depth ray of one frame reached a point: what :func:`corridor_seen_in` answers, asked with
    ``(N, 3)`` BASE millimetres and answering one ``bool`` per point.

    A value rather than a closure, so it pickles and a worker process (``src.robot.grasping.workers``) asks the same
    question of the same numbers. The turn and the shift are read off :attr:`to_camera` at every call, the views the
    closure held, so the arithmetic is the closure's to the bit.
    """

    #: The frame's depth along the optical axis, millimetres, ``(H, W)``; 0 or not finite where nothing was measured.
    depth_mm: np.ndarray
    #: BASE to CAMERA, 4x4, millimetres.
    to_camera: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    #: How far short of a point its pixel's measured depth may stop and the point still count as seen, millimetres.
    tolerance_mm: float = SEEN_TOLERANCE_MM

    def __call__(self, points_base_mm: np.ndarray) -> np.ndarray:
        depth = self.depth_mm
        turn, shift = self.to_camera[:3, :3], self.to_camera[:3, 3]
        height, width = depth.shape
        slack = float(self.tolerance_mm)
        points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(points.shape[0], dtype=bool)
        if points.shape[0] == 0:
            return out
        camera = points @ turn.T + shift
        z = camera[:, 2]
        front = np.isfinite(z) & (z > 1e-6)
        safe_z = np.where(front, z, 1.0)
        u = np.rint(self.fx * camera[:, 0] / safe_z + self.cx)
        v = np.rint(self.fy * camera[:, 1] / safe_z + self.cy)
        inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        index = np.nonzero(inside)[0]
        measured = depth[v[index].astype(np.int64), u[index].astype(np.int64)]
        with np.errstate(invalid="ignore"):
            out[index] = np.isfinite(measured) & (measured > 0.0) & (measured >= z[index] - slack)
        return out

    def unseen_many(self, points_base_mm: np.ndarray, asked: np.ndarray) -> np.ndarray:
        """Whether a depth ray missed one of a build's points, for many builds at once: ``(N, P, 3)`` BASE millimetres
        and which of them are asked (``asked``, ``(N, P)``) in, one ``int8`` per build out: 0 where :meth:`__call__`
        sees every point asked, 1 where it misses one, 2 where a point's pixel lies within :data:`_MANY_SURE` of a
        rounding edge, its depth within it of the measured depth less the tolerance, or the point within it of the
        camera's plane, which only :meth:`__call__` decides.

        The arithmetic is :meth:`__call__`'s, on every point asked at once, whose product may round its last bit
        another way than one build's call does; the band covers that. SFE asks this of every tilted build of a closing
        line at once (``support_footprint._build_many``).
        """
        depth = self.depth_mm
        turn, shift = self.to_camera[:3, :3], self.to_camera[:3, 3]
        height, width = depth.shape
        slack = float(self.tolerance_mm)
        points = np.asarray(points_base_mm, dtype=np.float64)
        out = np.zeros(points.shape[0], dtype=np.int8)
        owner = np.nonzero(asked)[0]
        flat = points[asked]
        if not flat.shape[0]:
            return out
        camera = flat @ turn.T + shift
        z = camera[:, 2]
        front = np.isfinite(z) & (z > 1e-6)
        safe_z = np.where(front, z, 1.0)
        u_at = self.fx * camera[:, 0] / safe_z + self.cx
        v_at = self.fy * camera[:, 1] / safe_z + self.cy
        u, v = np.rint(u_at), np.rint(v_at)
        # A pixel rounds another way only near its half, a point leaves the camera's front only at its plane.
        unsure = ((np.abs(u_at - np.floor(u_at) - 0.5) < _MANY_SURE) | (np.abs(v_at - np.floor(v_at) - 0.5) < _MANY_SURE)
                  | (np.abs(z - 1e-6) < _MANY_SURE))
        inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        index = np.nonzero(inside)[0]
        measured = depth[v[index].astype(np.int64), u[index].astype(np.int64)]
        seen = np.zeros(flat.shape[0], dtype=bool)
        with np.errstate(invalid="ignore"):
            read = np.isfinite(measured) & (measured > 0.0)
            seen[index] = read & (measured >= z[index] - slack)
            unsure[index] |= read & (np.abs(measured - (z[index] - slack)) < _MANY_SURE)
        out[owner[unsure]] = 2
        out[owner[~seen & ~unsure]] = 1
        return out


def corridor_seen_in(
    depth_mm: np.ndarray, intrinsics: Any, camera_to_base: np.ndarray, *, tolerance_mm: float = SEEN_TOLERANCE_MM,
) -> SeenInTheFrame:
    """Who says whether a depth ray of this frame reached a point (SFE's ``CorridorSeen``, the fix plan's contract 5).

    A BASE point is seen where its pixel's measured depth reaches at least to it, less ``tolerance_mm``: the ray went
    through the space the point stands in, or stopped on it. Behind the camera, outside the image, or on a pixel with no
    measured depth it is not seen. The answer is a :class:`SeenInTheFrame`, which a worker process can be handed.
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"the depth must be 2-D, got {depth.shape}")
    fx, fy, cx, cy = _matrix_of(intrinsics)
    to_camera = np.linalg.inv(np.asarray(camera_to_base, dtype=np.float64).reshape(4, 4))
    return SeenInTheFrame(depth_mm=depth, to_camera=to_camera, fx=fx, fy=fy, cx=cx, cy=cy,
                          tolerance_mm=float(tolerance_mm))


# --------------------------------------------------------------------------------------------------------------------
# The hand's envelope against the solids the guard holds
# --------------------------------------------------------------------------------------------------------------------

#: The eight corners of a unit box, and its twelve edges as corner pairs.
_SIGNS: Final[np.ndarray] = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
_EDGES: Final[np.ndarray] = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)
                                      if int(np.abs(_SIGNS[i] - _SIGNS[j]).sum()) == 2])


def _overlap(ca: np.ndarray, ra: np.ndarray, ha: np.ndarray, cb: np.ndarray, rb: np.ndarray,
             hb: np.ndarray) -> np.ndarray:
    """Whether each pair of oriented boxes overlaps, by the separating axis test over its fifteen axes."""
    t = cb - ca
    axes = [ra[:, :, i] for i in range(3)] + [rb[:, :, i] for i in range(3)]
    axes += [np.cross(ra[:, :, i], rb[:, :, j]) for i in range(3) for j in range(3)]
    separated = np.zeros(ca.shape[0], dtype=bool)
    for axis in axes:
        length = np.linalg.norm(axis, axis=1)
        usable = length > 1e-9
        unit = axis / np.where(usable, length, 1.0)[:, None]
        reach_a = sum(ha[:, i] * np.abs(np.einsum("nk,nk->n", ra[:, :, i], unit)) for i in range(3))
        reach_b = sum(hb[:, i] * np.abs(np.einsum("nk,nk->n", rb[:, :, i], unit)) for i in range(3))
        separated |= usable & (np.abs(np.einsum("nk,nk->n", t, unit)) > reach_a + reach_b + 1e-9)
    return ~separated


def _point_box_distance(points: np.ndarray, centre: np.ndarray, turn: np.ndarray, half: np.ndarray) -> np.ndarray:
    """``(N, K)`` distances of ``(N, K, 3)`` points to ``N`` oriented boxes; 0 inside."""
    local = np.einsum("nkj,nji->nki", points - centre[:, None, :], turn)
    outside = np.maximum(np.abs(local) - half[:, None, :], 0.0)
    return np.linalg.norm(outside, axis=2)


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """The least distance between segments ``p1 q1`` and ``p2 q2``, elementwise over leading axes (Ericson 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("...k,...k->...", d1, d1)
    e = np.einsum("...k,...k->...", d2, d2)
    f = np.einsum("...k,...k->...", d2, r)
    c = np.einsum("...k,...k->...", d1, r)
    b = np.einsum("...k,...k->...", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    gap = (p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None])
    return np.linalg.norm(gap, axis=-1)


def box_distances_mm(ca: np.ndarray, ra: np.ndarray, ha: np.ndarray, cb: np.ndarray, rb: np.ndarray,
                     hb: np.ndarray) -> np.ndarray:
    """The least distance between each pair of oriented boxes, millimetres; 0 where they overlap.

    Each box is a centre ``(N, 3)``, a rotation ``(N, 3, 3)`` whose columns are its axes in BASE, and half extents
    ``(N, 3)``. Exact: two boxes apart are nearest at a corner of one against the other, or between two of their edges.
    """
    ca, cb = np.asarray(ca, dtype=np.float64).reshape(-1, 3), np.asarray(cb, dtype=np.float64).reshape(-1, 3)
    ra, rb = np.asarray(ra, dtype=np.float64).reshape(-1, 3, 3), np.asarray(rb, dtype=np.float64).reshape(-1, 3, 3)
    ha, hb = np.asarray(ha, dtype=np.float64).reshape(-1, 3), np.asarray(hb, dtype=np.float64).reshape(-1, 3)
    corners_a = ca[:, None, :] + np.einsum("nij,nkj->nki", ra, _SIGNS[None, :, :] * ha[:, None, :])
    corners_b = cb[:, None, :] + np.einsum("nij,nkj->nki", rb, _SIGNS[None, :, :] * hb[:, None, :])
    least = np.minimum(_point_box_distance(corners_a, cb, rb, hb).min(axis=1),
                       _point_box_distance(corners_b, ca, ra, ha).min(axis=1))
    ea0, ea1 = corners_a[:, _EDGES[:, 0], :], corners_a[:, _EDGES[:, 1], :]
    eb0, eb1 = corners_b[:, _EDGES[:, 0], :], corners_b[:, _EDGES[:, 1], :]
    edges = _segment_distance(ea0[:, :, None, :], ea1[:, :, None, :], eb0[:, None, :, :], eb1[:, None, :, :])
    least = np.minimum(least, edges.reshape(edges.shape[0], -1).min(axis=1))
    return np.where(_overlap(ca, ra, ha, cb, rb, hb), 0.0, least)


@dataclass(frozen=True, slots=True)
class EnvelopeVerdict:
    """Why one candidate's open hand stands too near what the guard holds, or that it does not."""

    #: ``""`` where it keeps its distance, else ``support`` or ``declared``.
    refused: str
    #: The nearest held solid and the hand's least distance to it, millimetres; ``""`` and ``inf`` where none is held.
    nearest: str
    distance_mm: float


def finger_floor_drop_mm(robot: Any) -> float:
    """How much nearer than the guard's distance from a camera box a finger comes to what a support reads, millimetres:
    ``self_collision.perceived_min_distance_mm`` less ``planning_world.perceived.finger_floor_mm`` (the owner's 1 mm,
    2026-10-06), 0 where the fingers keep their distance from the whole solid or the floor is no nearer."""
    safety = getattr(robot, "safety", None)
    perceived = getattr(getattr(safety, "planning_world", None), "perceived", None)
    if perceived is None or not bool(getattr(perceived, "fingers_to_the_support_reading", False)):
        return 0.0
    guard = float(getattr(getattr(safety, "self_collision", None), "perceived_min_distance_mm", 0.0) or 0.0)
    return max(0.0, guard - float(getattr(perceived, "finger_floor_mm", guard)))


def envelope_verdicts(
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    gripper_model: Any,
    open_width_mm: float,
    solids: Sequence[Any] = (),
    support_distance_mm: float = 5.0,
    declared_boxes: Sequence[DeclaredBox] = (),
    declared_distance_mm: float = 10.0,
    seen_boxes: Sequence[Any] = (),
    seen_distance_mm: float = 5.0,
    fingers_to_the_reading: bool = False,
    finger_floor_drop_mm: float = 0.0,
) -> list[EnvelopeVerdict]:
    """For each BASE grasp ``(position, rotation)``, whether the open hand there keeps the guard's distances.

    The hand is ``gripper_model.collision_boxes(open_width_mm)`` at the pose (the rotation's columns are the closing
    axis, the binormal and the approach, as a ``GraspPose`` holds them): it arrives open. A support's solid
    (``SupportSolid``: ``centre_mm``, ``rotation``, ``half_extents_mm``, the allowance included) is a camera box the
    guard keeps ``support_distance_mm`` from; a declared box keeps ``declared_distance_mm``; a box the camera world
    builds of a neighbour (``perceived.SeenBox``, :func:`seen_boxes`) keeps ``seen_distance_mm``; one of a named
    neighbour part (``SeenBox.soft_mm``) holds a finger to its measured surface instead, at no distance, every other
    part of the hand to the whole box (the owner, 2026-10-05). With ``fingers_to_the_reading`` a support's solid holds
    a finger at its distance from the solid's top less its band and its allowance (``band_mm``, ``allowance_mm``), as
    the guard does (``PerceivedBox.finger_top_mm``). Mirrors the guard over the hand's box envelope, which holds the
    hand's meshes: it refuses at least what the guard would refuse at the hand.
    """
    boxes = gripper_model.collision_boxes(float(open_width_mm))
    local_centres = np.array([(np.asarray(b.min_corner_mm) + np.asarray(b.max_corner_mm)) / 2.0 for b in boxes])
    local_halves = np.array([(np.asarray(b.max_corner_mm) - np.asarray(b.min_corner_mm)) / 2.0 for b in boxes])
    fingers = np.array([str(getattr(b, "label", "")).startswith("finger") for b in boxes], dtype=bool)
    held: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, float, float, float]] = []
    for solid in solids:
        drop = (float(getattr(solid, "band_mm", 0.0) or 0.0) + float(getattr(solid, "allowance_mm", 0.0) or 0.0)
                + max(0.0, float(finger_floor_drop_mm)) if fingers_to_the_reading else 0.0)
        held.append(("support", str(solid.name), np.asarray(solid.centre_mm, dtype=np.float64),
                     np.asarray(solid.rotation, dtype=np.float64).reshape(3, 3),
                     np.asarray(solid.half_extents_mm, dtype=np.float64), float(support_distance_mm), 0.0, drop))
    for box in declared_boxes:
        held.append(("declared", box.name, np.asarray(box.centre_mm, dtype=np.float64), np.eye(3),
                     np.asarray(box.half_extents_mm, dtype=np.float64), float(declared_distance_mm), 0.0, 0.0))
    for number, box in enumerate(seen_boxes):
        held.append(("seen", f"neighbour box {number}", np.asarray(box.centre_mm, dtype=np.float64),
                     np.asarray(box.rotation, dtype=np.float64).reshape(3, 3),
                     np.asarray(box.half_extents_mm, dtype=np.float64), float(seen_distance_mm),
                     float(getattr(box, "soft_mm", 0.0) or 0.0), 0.0))
    out: list[EnvelopeVerdict] = []
    if not held:
        return [EnvelopeVerdict("", "", math.inf) for _ in poses]
    n_hand, n_held = len(boxes), len(held)
    for position, rotation in poses:
        turn = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        centres = np.asarray(position, dtype=np.float64).reshape(3) + local_centres @ turn.T
        ca = np.repeat(centres, n_held, axis=0)
        ra = np.repeat(turn[None, :, :], n_hand * n_held, axis=0)
        ha = np.repeat(local_halves, n_held, axis=0)
        cb = np.tile(np.array([h[2] for h in held]), (n_hand, 1))
        rb = np.tile(np.array([h[3] for h in held]), (n_hand, 1, 1))
        hb = np.tile(np.array([h[4] for h in held]), (n_hand, 1))
        matrix = box_distances_mm(ca, ra, ha, cb, rb, hb).reshape(n_hand, n_held)
        limits = np.array([h[5] for h in held])
        softs = np.array([h[6] for h in held])
        soft = softs > 0.0
        distances = matrix.min(axis=0)
        if soft.any() and fingers.any():
            # A finger against a named part's box: to its measured surface, the box less its soft, kept
            # :data:`SOFT_SLACK_MM` off it. The rest of the hand against the whole box at the guard's; the box is short
            # where either comes nearer.
            whole = np.where(fingers[:, None] & soft[None, :], np.inf, matrix).min(axis=0)
            surface = _surface_distances(centres[fingers], turn, local_halves[fingers], held, soft)
            distances = np.where(soft, np.minimum(whole, surface - SOFT_SLACK_MM + limits), distances)
        drops = np.array([h[7] for h in held])
        dropped = drops > 0.0
        if dropped.any() and fingers.any():
            # A finger against a support's solid: to its top less its band and its allowance, at the distance; the rest
            # of the hand against the whole solid.
            whole = np.where(fingers[:, None] & dropped[None, :], np.inf, matrix).min(axis=0)
            reading = _lowered_distances(centres[fingers], turn, local_halves[fingers], held, dropped)
            distances = np.where(dropped, np.minimum(whole, reading), distances)
        short = distances < limits
        nearest = int(np.argmin(distances - limits))
        refused = ""
        if short.any():
            # A declared box outranks a camera's solid: it was measured, and its refusal says what to move. The support
            # outranks a neighbour: taking a neighbour away frees no grasp the support refuses.
            kinds = [held[i][0] for i in np.nonzero(short)[0]]
            refused = "declared" if "declared" in kinds else "support" if "support" in kinds else "seen"
            nearest = int(np.nonzero(short)[0][np.argmin((distances - limits)[short])])
        out.append(EnvelopeVerdict(refused, held[nearest][1], float(distances[nearest])))
    return out


def _lowered_distances(finger_centres: np.ndarray, turn: np.ndarray, finger_halves: np.ndarray,
                       held: Sequence[tuple[Any, ...]], dropped: np.ndarray) -> np.ndarray:
    """Per held box, the fingers' least distance to it with its top lowered by its drop along its own up axis, where it
    has one (a support's band and allowance), ``inf`` where it has none."""
    out = np.full(len(held), np.inf)
    indices = np.nonzero(dropped)[0]
    if indices.size == 0 or finger_centres.shape[0] == 0:
        return out
    centres = np.array([held[int(i)][2] for i in indices], dtype=np.float64)
    turns = np.array([held[int(i)][3] for i in indices], dtype=np.float64)
    halves = np.array([held[int(i)][4] for i in indices], dtype=np.float64).copy()
    drops = np.array([held[int(i)][7] for i in indices], dtype=np.float64)
    lowered = np.minimum(drops, np.maximum(halves[:, 2] - 0.5, 0.0) * 2.0)
    halves[:, 2] -= lowered / 2.0
    centres = centres - turns[:, :, 2] * (lowered / 2.0)[:, None]
    n_f, n_b = finger_centres.shape[0], indices.size
    found = box_distances_mm(
        np.repeat(finger_centres, n_b, axis=0), np.repeat(turn[None, :, :], n_f * n_b, axis=0),
        np.repeat(finger_halves, n_b, axis=0), np.tile(centres, (n_f, 1)), np.tile(turns, (n_f, 1, 1)),
        np.tile(halves, (n_f, 1))).reshape(n_f, n_b).min(axis=0)
    out[indices] = found
    return out


def _surface_distances(finger_centres: np.ndarray, turn: np.ndarray, finger_halves: np.ndarray,
                       held: Sequence[tuple[Any, ...]], soft: np.ndarray) -> np.ndarray:
    """Per held box, the fingers' least distance to its measured surface where it is soft (the box less its soft on its
    sides and its top, its foot where it stands), ``inf`` where it is not."""
    out = np.full(len(held), np.inf)
    indices = np.nonzero(soft)[0]
    if indices.size == 0 or finger_centres.shape[0] == 0:
        return out
    centres = np.array([held[int(i)][2] for i in indices], dtype=np.float64)
    turns = np.array([held[int(i)][3] for i in indices], dtype=np.float64)
    halves = np.array([held[int(i)][4] for i in indices], dtype=np.float64)
    softs = np.array([held[int(i)][6] for i in indices], dtype=np.float64)
    lowered = np.minimum(softs, np.maximum(halves[:, 2] - 0.5, 0.0) * 2.0)
    halves = halves.copy()
    halves[:, :2] = np.maximum(halves[:, :2] - softs[:, None], 0.5)
    halves[:, 2] -= lowered / 2.0
    centres = centres - turns[:, :, 2] * (lowered / 2.0)[:, None]
    n_f, n_b = finger_centres.shape[0], indices.size
    found = box_distances_mm(
        np.repeat(finger_centres, n_b, axis=0), np.repeat(turn[None, :, :], n_f * n_b, axis=0),
        np.repeat(finger_halves, n_b, axis=0), np.tile(centres, (n_f, 1)), np.tile(turns, (n_f, 1, 1)),
        np.tile(halves, (n_f, 1))).reshape(n_f, n_b).min(axis=0)
    out[indices] = found
    return out


# --------------------------------------------------------------------------------------------------------------------
# Why a part got no grasp
# --------------------------------------------------------------------------------------------------------------------


def least_part_height_mm(jaw: Any) -> float:
    """How far over its support a part has to stand for SFE to plan this hand on it at all, millimetres.

    SFE solves, per tilt of its ladder, the anchor height at which the fingers clear the support by the table
    clearance, and tries that height only where it lies 2 mm under the part's top (``generate_support_footprint_grasps``).
    A part lower than the least of those over the ladder gets no height to try: it is too short for this hand.
    """
    tilts = np.radians(np.asarray(_SFE_TILTS_DEG))
    need = float(jaw.finger_ahead_mm) * np.cos(tilts) + (float(jaw.finger_width_mm) / 2.0) * np.sin(tilts)
    return float(jaw.table_clearance_mm + need.min() + _SFE_TOP_SLACK_MM)


def why_no_grasp(telemetry: Mapping[str, Any]) -> "tuple[tuple[GraspFailureReason, ...], str]":
    """The reasons and the sentence for a calculator result with no candidate, read off its telemetry.

    In this order, the fix plan's (Track A):

    * the post-hoc filters emptied what SFE offered: a support's solid or a declared box too near the open hand
      (``ALL_TABLE_CONFLICT``), the scene's points inside the hand (``ALL_COLLIDED``), the workspace, the reach;
    * SFE offered nothing, and a build met a neighbour the camera saw (``seen_*``): ``ALL_COLLIDED``, the only reason a
      push answers;
    * it met a declared body (``declared_*``): ``ALL_TABLE_CONFLICT``;
    * the support refused it (``table``): ``ALL_TABLE_CONFLICT``, and where the part stands lower than the hand grips at
      any tilt, that it is too short for this hand;
    * the part's own low fragments alone: ``NO_VALID_GRASP``.

    ``RESCAN_RECOMMENDED`` comes last on every one, as before.
    """
    from src.robot.grasping.types.feedback import GraspFailureReason as R  # noqa: PLC0415, N817

    refused = telemetry.get("support_footprint_refused")
    counts: Mapping[str, Any] = refused if isinstance(refused, Mapping) else {}

    def count(*names: str) -> int:
        return int(sum(int(counts.get(name, 0) or 0) for name in names))

    def stamp(key: str) -> int:
        return int(telemetry.get(key, 0) or 0)

    obstacles = int(telemetry.get("scene_obstacle_points", 0) or 0)
    groups = int(telemetry.get("scene_obstacle_clusters", 0) or 0)
    beside = (f" ({obstacles} point(s) in {groups} cluster(s) within {REACH_MM:.0f} mm)" if obstacles else "")
    offered = int(telemetry.get("support_footprint_kept", 0) or 0)
    stage = str(telemetry.get("geometry_stage", ""))
    reasons: list[R] = []
    said = ""
    if stage != "support_footprint" or offered > 0:
        # Something was offered and a filter after it emptied the list.
        if stamp("rejected_workspace"):
            reasons.append(R.ALL_OUT_OF_WORKSPACE)
            said = said or "every grasp lies outside the workspace box"
        if stamp("rejected_declared"):
            reasons.append(R.ALL_TABLE_CONFLICT)
            said = said or (f"every grasp brings the hand within {_number(telemetry, 'scene_declared_distance_mm'):g} "
                            f"mm of the declared {telemetry.get('scene_declared_nearest') or 'body'}")
        if stamp("rejected_support") or stamp("rejected_table"):
            if R.ALL_TABLE_CONFLICT not in reasons:
                reasons.append(R.ALL_TABLE_CONFLICT)
            if stamp("rejected_support"):
                said = said or (f"every grasp brings the hand within {_number(telemetry, 'scene_support_distance_mm'):g}"
                                " mm of the support surface the part stands on")
            else:
                clearance = (_number(telemetry, "scene_table_clearance_mm")
                             or _number(telemetry, "table_clearance_required_mm"))
                said = said or f"every grasp brings the hand within {clearance:g} mm of the support the part stands on"
        if stamp("rejected_collision"):
            reasons.append(R.ALL_COLLIDED)
            said = said or f"every grasp puts the hand into a neighbour the camera saw beside the part{beside}"
        if stamp("rejected_ik"):
            reasons.append(R.IK_FAILED)
            said = said or "no grasp is within the arm's reach"
        if not reasons:
            reasons.append(R.NO_VALID_GRASP)
    elif count("seen_fingers", "seen_corridor"):
        reasons.append(R.ALL_COLLIDED)
        said = f"every grasp the hand fits meets a neighbour the camera saw beside the part{beside}"
    elif count("declared_fingers", "declared_corridor"):
        reasons.append(R.ALL_TABLE_CONFLICT)
        said = "every grasp the hand fits meets the declared walls beside the part"
    elif count("table"):
        reasons.append(R.ALL_TABLE_CONFLICT)
        height = telemetry.get("scene_part_height_mm")
        least = telemetry.get("scene_least_part_height_mm")
        if isinstance(height, (int, float)) and isinstance(least, (int, float)) and float(height) < float(least):
            said = (f"the part stands less than about {float(least):.0f} mm above its support ({float(height):.0f} mm), "
                    "too short for this hand")
        else:
            said = (f"every grasp brings the fingertips within {_number(telemetry, 'scene_table_clearance_mm'):g} mm of "
                    "the support the part stands on")
    elif count("own_fragments"):
        reasons.append(R.NO_VALID_GRASP)
        said = "every grasp the hand fits meets the part's own low fragments"
    else:
        reasons.append(R.NO_VALID_GRASP)
        if count("aperture"):
            said = "the part is wider than the hand opens, or thinner than it closes"
        elif int(telemetry.get("support_footprint_points", 0) or 0) and not any(int(v or 0) for v in counts.values()):
            said = "too few of the part's points stand over its support to plan a grasp on"
    reasons.append(R.RESCAN_RECOMMENDED)
    return tuple(dict.fromkeys(reasons)), said


def _number(telemetry: Mapping[str, Any], key: str) -> float:
    """A telemetry number, 0 where it is missing or not a number."""
    value = telemetry.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def no_grasp_said(result: object) -> str:
    """Why ``result`` has no candidate, in one sentence, or ``""``: what :func:`why_no_grasp` left in its telemetry."""
    telemetry = getattr(result, "telemetry", None)
    said = telemetry.get(NO_GRASP_SAID) if isinstance(telemetry, Mapping) else None
    return said if isinstance(said, str) else ""
