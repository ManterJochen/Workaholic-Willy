"""Exact mesh-against-mesh self-collision backend, for the ``backend='fcl'`` path of
:class:`SelfCollisionGuard`.

It supersedes the capsule proxy, which is measured wrong in both directions: it misses
a 25 mm ``forearm|wrist_3`` approach because ``_should_skip_pair`` skips that pair, and
it invents a ``link_2|link_5`` collision from the fat default radius. This measures
exact distance on the real meshes instead.

The engine is Coal where available and python-fcl otherwise, resolved once and
centrally through
:func:`src.robot.safety.planning.environment.import_collision_engine`. Coal is the
maintained successor to hpp-fcl, installed with ``conda install coal -c conda-forge``,
and is tried first; where it does not import the backend falls back to ``python-fcl``.
The two are proven byte-identical at the distance level on the values the guard
triggers on, which are 19.64 mm natural, 0.0 mm flipped and 19.224 mm park, and Coal
is about 1.5x faster on this workload, which is headroom toward a real-time budget on
hardware. It is an engine swap and nothing more: the same BVH meshes, pairs,
thresholds and ``distance`` query. The extra Coal features, swept-sphere, contact
patches and Nesterov among them, are deliberately unused here.

This path is reached only when ``self_collision.backend='fcl'``, so it is default-off
and byte-identical otherwise. Its design:

  * Per-link collision meshes are bundled as ``data/{model}_collision_meshes.npz``,
    pre-baked into the bundled-DH link frame by
    ``M_dh = inv(T_dh[frame](q0)) @ inv(R_base) @ world_usd(q0)``, validated
    vertex-exact to under 0.15 mm against the Isaac USD at six configurations. One BVH
    model is built per link once, and each evaluate updates only the CollisionObject
    transform rather than rebuilding.
  * The world transform of link L at joints q is ``R_base(yaw) @ T_dh[L.frame](q)``,
    the same ``kinematics_base_yaw_deg`` reconcile the capsule path uses, so meshes,
    fixtures and tool agree.
  * Pairwise self-collision skips only links within one DH frame of each other, which
    are the adjacent ones and the rigid wrist cluster that always touch by
    construction. Every farther pair is checked exactly, which covers the wrist pairs
    the capsule path has to skip and does so without its false positives.
  * Each link is also checked against the fixtures, which are boxes: axis-aligned where
    declared, and turned about base Z as the planner holds them where a camera saw them.

Neither engine is a hard dependency. Where neither Coal nor python-fcl imports, or the
mesh bundle is absent, :func:`make_backend` returns ``None`` and the guard falls back
to the capsule path rather than crashing. On Windows Coal has no pip wheel: point
``WILLY_COAL_PREFIX`` at a conda environment that has it, such as a micromamba
``coal`` environment, and the shared engine loader injects its DLL directory and
site-packages. ``python -m src.robot.safety.planning --check`` reports which engine
resolves on this box.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._ur_kinematics import UR_DH_TABLES_M
from .planning._declared_body import Box
from .planning._hand_bundle import HAND_PARTS, hand_bundle_refusal
from .planning._hand_placement import HandPlacement
from .planning.environment import (
    COLLISION_MESH_DIR,
    hand_writers_sentence,
    compose_collision_meshes,
    hand_mesh_bundle,
    import_collision_engine,
)


_LOGGER = logging.getLogger(__name__)

#: Mirrors ``self_collision._DEFAULT_LINK_RADIUS_MM``. The fallback warning quotes it,
#: so an operator sees how much coarser the capsule guard is without reading the other
#: module.
_CAPSULE_FALLBACK_RADIUS_HINT_MM = 60.0

_STATUS_HINTS = {
    "unknown_model": "No bundled DH chain for this model; add it to safety/_ur_kinematics.UR_DH_TABLES_M.",
    "no_bundle": (
        "Bake the per-link collision meshes into safety/data/{model}_collision_meshes.npz "
        "(same DH-frame recipe as the ur5e bundle; the 2F-85 tool0 meshes are model-independent and can be "
        "copied verbatim). Until then this cell has no exact-mesh self-collision authority."
    ),
    "no_engine": "Install Coal (WILLY_COAL_PREFIX) or python-fcl; expected/accepted on macOS + CI.",
    "no_hand_bundle": (
        "the hand robot.gripper.model names has no bundle of its own, safety/data/{hand}_hand_meshes.npz, to "
        "compose onto this arm: "
        + hand_writers_sentence("{hand}")
        + ". Until then this cell has no exact-mesh self-collision authority."
    ),
    "hand_bundle_refused": (
        "the hand bundle safety/data/{hand}_hand_meshes.npz is not a hand the guard can place: it holds a part other "
        "than gripper, lfinger and rfinger, misses one, carries a malformed array, or holds its fingers off the model's "
        "+Y. Write it again with its axes stated. Until then this cell has no exact-mesh self-collision authority."
    ),
}




#: The hand's parts that may come to a named part's measured surface (``AxisAlignedBox.soft_mm``): its two fingers. The
#: housing, ``gripper``, keeps the whole box and the distance, as every other link does.
FINGER_PARTS: frozenset[str] = frozenset({"lfinger", "rfinger"})


def _surface_of(adapter: Any, fixture: Any, soft_mm: float) -> Any:
    """``fixture`` less ``soft_mm`` on its sides and its top, its foot where it stands: the measured surface of a part
    the camera world grew by its margin. Never thinner than a millimetre."""
    turned = getattr(fixture, "turned", None)
    centre = np.asarray(fixture.center_mm, dtype=np.float64).copy()
    half = np.asarray(fixture.half_extents_mm if turned is None else turned.half_extents_mm, dtype=np.float64).copy()
    lowered = min(float(soft_mm), max(0.0, float(half[2]) - 0.5) * 2.0)
    half[:2] = np.maximum(half[:2] - float(soft_mm), 0.5)
    half[2] -= lowered / 2.0
    centre[2] -= lowered / 2.0
    if turned is None:
        return adapter.box_object(half, centre)
    return adapter.box_object(half, centre, float(turned.yaw_rad), getattr(turned, "rotation", None))


def _finger_top_of(adapter: Any, fixture: Any, drop_mm: float) -> Any:
    """``fixture`` with its top lowered by ``drop_mm`` along its own up axis, its sides and its foot where they stand: a
    support's solid as a finger keeps the distance from it, the reading and its excess (``finger_top_mm``). Never
    thinner than a millimetre."""
    turned = getattr(fixture, "turned", None)
    centre = np.asarray(fixture.center_mm, dtype=np.float64).copy()
    half = np.asarray(fixture.half_extents_mm if turned is None else turned.half_extents_mm, dtype=np.float64).copy()
    lowered = min(float(drop_mm), max(0.0, float(half[2]) - 0.5) * 2.0)
    rotation = None if turned is None else getattr(turned, "rotation", None)
    up = (np.array([0.0, 0.0, 1.0]) if rotation is None
          else np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, 2])
    half[2] -= lowered / 2.0
    centre = centre - up * (lowered / 2.0)
    if turned is None:
        return adapter.box_object(half, centre)
    return adapter.box_object(half, centre, float(turned.yaw_rad), rotation)


def _yaw_matrix(yaw_deg: float) -> np.ndarray:
    """The 3x3 rotation about +Z by ``yaw_deg``, which is the kinematics_base_yaw_deg reconcile."""
    import math
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class _EngineAdapter:
    """A uniform wrapper over Coal or python-fcl.

    The two APIs differ in a few names and types only, and the BVH build and
    closest-distance queries are proven byte-identical.
    """

    def __init__(self, mod: Any, kind: str) -> None:
        self._m = mod
        self.kind = kind

    def build_object(self, verts: np.ndarray, faces: np.ndarray) -> Any:
        m = self._m
        if self.kind == "coal":
            model = m.BVHModelOBBRSS()
            vv = m.StdVec_Vec3s()
            for v in np.asarray(verts, dtype=np.float64):
                vv.append(v)
            tt = m.StdVec_Triangle()
            for f in faces:
                tt.append(m.Triangle(int(f[0]), int(f[1]), int(f[2])))
            model.beginModel(len(tt), len(vv))
            model.addSubModel(vv, tt)
            model.endModel()
            return m.CollisionObject(model, m.Transform3s())
        model = m.BVHModel()
        model.beginModel(len(verts), len(faces))
        model.addSubModel(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64))
        model.endModel()
        return m.CollisionObject(model, m.Transform())

    def _transform(self, R: np.ndarray, t: np.ndarray) -> Any:
        m = self._m
        R = np.ascontiguousarray(R, dtype=np.float64)
        t = np.ascontiguousarray(t, dtype=np.float64)
        return m.Transform3s(R, t) if self.kind == "coal" else m.Transform(R, t)

    def set_transform(self, obj: Any, R: np.ndarray, t: np.ndarray) -> None:
        obj.setTransform(self._transform(R, t))

    def box_object(
        self, half_extents: np.ndarray, center: np.ndarray, yaw_rad: float = 0.0, rotation: "np.ndarray | None" = None,
    ) -> Any:
        """A box of ``half_extents`` about ``center``, turned about base Z by ``yaw_rad``, or by the whole ``rotation``
        (a 3x3 whose columns are its axes in BASE) where one is given: a support surface's solid tilts."""
        m = self._m
        size = 2.0 * np.asarray(half_extents, dtype=np.float64)
        if rotation is not None:
            turn = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        else:
            c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
            turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return m.CollisionObject(m.Box(*size), self._transform(turn, np.asarray(center, dtype=np.float64)))

    def cylinder_object(self, radius: float, length: float, center: np.ndarray) -> Any:
        """A solid cylinder about base Z, ``length`` long, centred on ``center``: a robot's base (``base_near_mm``)."""
        m = self._m
        return m.CollisionObject(m.Cylinder(float(radius), float(length)),
                                 self._transform(np.eye(3), np.asarray(center, dtype=np.float64)))

    def distance(self, a: Any, b: Any) -> float:
        m = self._m
        return float(m.distance(a, b, m.DistanceRequest(), m.DistanceResult()))

    def convex_object(self, vertices: np.ndarray) -> Any:
        """The convex hull of ``vertices`` as an engine object, or ``None`` where the engine builds none.

        Coal's ``Convex.convexHull``, for the whole-path judge's hull layer (:meth:`MeshSelfCollisionBackend.
        first_suspect`): the hull holds every triangle of the part, so its distance is never more than the mesh's.
        python-fcl computes no hull, and there every distance the layer would bound is measured on the meshes.
        """
        m = self._m
        hull_of = getattr(getattr(m, "Convex", None), "convexHull", None) if self.kind == "coal" else None
        if hull_of is None:
            return None
        points = m.StdVec_Vec3s()
        for v in np.asarray(vertices, dtype=np.float64):
            points.append(v)
        return m.CollisionObject(hull_of(points, False, None), m.Transform3s())


#: How many fixture tuples' engine objects a backend keeps: the declared fixtures, the boxes a camera saw, and the sets a
#: judgement asks beside them.
_PROBE_TUPLES_KEPT = 4


#: Below a measured distance less how far a part moved since, how far the bound stays from the limit before an exact query
#: is skipped, millimetres: the engine's own rounding, nothing more.
_ADVANCE_EPS_MM = 1e-6

#: How far over its limit a distance the whole-path judge measured has to keep before its sample is passed, millimetres
#: (``MeshSelfCollisionBackend.first_suspect``): the rounding between a path's parts placed at once and one sample's
#: placed at a time, and the engine's own, nothing more. A sample nearer than this is left to :meth:`evaluate`.
_SUSPECT_SLACK_MM = 1e-6

#: How far a part's convex hull is taken to lie nearer than it measures, millimetres, before its distance stands for the
#: part's mesh unmeasured. The hull holds the mesh, so its distance is never more than the mesh's; GJK on two convex
#: shapes stops within a millionth of the distance it measures (Coal's default relative tolerance), so a hull measured at
#: the limit plus this proves the mesh keeps the limit for every limit up to the schema's 500 mm.
_HULL_SLACK_MM = 1e-3

#: How many hull vertices times configurations one step of :meth:`MeshSelfCollisionBackend.lowest_mm_many` multiplies at
#: once: about 8 MB of heights, whatever the length of the path.
_LOWEST_CHUNK = 1_000_000

#: How many later samples the whole-path walk bounds at once after a measurement (``_first_flagged``).
_WALK_WINDOW = 512

#: One over the square root of two, and what the chord of a turn is padded by for rotations held in floats, which are
#: rotations to about 1e-15 (``_chord``).
_HALF_SQRT2 = 1.0 / math.sqrt(2.0)
_CHORD_PAD = 1e-12


@dataclass(frozen=True)
class _Probes:
    """The engine objects one fixture tuple is judged as, column by column (``MeshSelfCollisionBackend._probes``), and
    what the last exact query of each part and column measured: the distance (``nan`` where none was asked) and the
    part's turn and sphere centre at that query, which bound how near it can have come since."""

    boxes: tuple[Any, ...]
    surfaces: tuple[Any, ...]
    readings: tuple[Any, ...]
    centres: np.ndarray
    radii: np.ndarray
    soft: np.ndarray
    measured: np.ndarray
    turns: np.ndarray
    at: np.ndarray


@dataclass(frozen=True)
class _Placements:
    """Where every part of a backend stands at every sample of a path, in the system base frame and in the order of its
    parts: the turn ``(N, P, 3, 3)``, the frame's origin ``(N, P, 3)`` and the bounding sphere's centre ``(N, P, 3)``,
    each the product :meth:`MeshSelfCollisionBackend.evaluate` forms for one sample."""

    turns: np.ndarray
    origins: np.ndarray
    centres: np.ndarray


def _chord(turn: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """``|R1 - R0|``, the largest distance the turn from ``turn`` to each of ``turns`` ``(M, 3, 3)`` moves a point a
    unit from its centre: the chord ``2 sin(angle / 2)`` of the angle between.

    The difference of two rotations has the chord twice among its singular values and 0 once, so the chord is its
    Frobenius norm over the square root of two; ``_CHORD_PAD`` covers how far rotations in floats are from rotations.
    Taken from the difference rather than as ``sqrt(3 - trace(R0^T R1))``, which is the same number in exact arithmetic
    and nothing at all in floats where a part barely turns: below a few 1e-8 rad the trace rounds to 3 and the chord to
    0, which on the forearm's 382 mm sphere leaves about 1e-5 mm of travel unbounded, ten times ``_ADVANCE_EPS_MM``."""
    return np.linalg.norm((turns - turn).reshape(len(turns), 9), axis=1) * _HALF_SQRT2 + _CHORD_PAD


def _moved_mm(centre: np.ndarray, turn: np.ndarray, centres: np.ndarray, turns: np.ndarray, radius: float) -> np.ndarray:
    """How far any point of a part can have moved from where it stood, its sphere's centre ``centre`` and its turn
    ``turn``, to each later placement, ``centres`` ``(M, 3)`` and ``turns`` ``(M, 3, 3)``: the centre's travel plus the
    sphere's radius times the chord of the turn between (:func:`_chord`), the rule :meth:`MeshSelfCollisionBackend.
    evaluate` skips a box by. A point ``r`` from the centre moves by the centre's travel plus at most ``r`` times
    ``|R1 - R0|``."""
    return np.linalg.norm(centres - centre, axis=1) + float(radius) * _chord(turn, turns)


def _first_flagged(
    near: np.ndarray, until: int, limit: float, measure: Any, moved: Any,
) -> int:
    """The first sample before ``until`` at which one pair may come under ``limit``, walking only the samples whose
    spheres do not clear it (``near``), or ``until`` where there is none.

    The pair is measured at the first of them (``measure(sample)``, a lower bound on its exact distance there), which is
    flagged where that comes under the limit plus ``_SUSPECT_SLACK_MM``. Otherwise every later sample the distance less
    how far the pair can have moved since (``moved(sample, later)``) keeps at the limit is passed, and the next one it
    does not keep is measured. A flagged sample is only ever early: :meth:`MeshSelfCollisionBackend.evaluate` decides it.
    """
    open_ = np.flatnonzero(near[:until])
    if not len(open_):
        return until
    sample = int(open_[0])
    while True:
        bound = float(measure(sample))
        if not bound >= limit + _SUSPECT_SLACK_MM:
            return sample
        # The later samples a window at a time, so a pair that stays near the limit costs no more than the samples it
        # walks over, however long the path.
        start = sample + 1
        while True:
            stop = min(until, start + _WALK_WINDOW)
            later = start + np.flatnonzero(near[start:stop])
            unproven = (np.flatnonzero(~(bound - moved(sample, later) - _ADVANCE_EPS_MM >= limit)) if len(later)
                        else later)
            if len(unproven):
                break
            if stop >= until:
                return until
            start = stop
        sample = int(later[unproven[0]])


class MeshSelfCollisionBackend:
    """Holds the per-link BVH models and runs the exact pairwise and fixture distance checks."""

    def __init__(
        self, adapter: _EngineAdapter, meshes: dict[str, tuple[np.ndarray, np.ndarray, int]],
        wrist_parts: "frozenset[str]" = frozenset(),
    ) -> None:
        self._a = adapter
        #: Parts a wrist camera adds on frame 6. They are checked against wrist_2 on frame 5, which the
        #: pair rule skips for everything else on the flange, because a housing sticks out sideways and
        #: can reach it.
        self._wrist = frozenset(wrist_parts)
        self.engine = adapter.kind  # 'coal' or 'fcl', surfaced for telemetry
        self._models: dict[str, Any] = {}
        self._frame: dict[str, int] = {}
        # A per-link bounding sphere, centroid and radius in the link frame, for the
        # optional broadphase cull. It is conservative, because the sphere bounds the
        # mesh, so it never changes a verdict.
        self._sph_c: dict[str, np.ndarray] = {}
        self._sph_r: dict[str, float] = {}
        # The vertices that hold a part's lowest point however it is placed, for :meth:`lowest_mm`.
        self._hull: dict[str, np.ndarray] = {}
        for name, (verts, faces, frame) in meshes.items():
            self._models[name] = adapter.build_object(verts, faces)
            self._frame[name] = int(frame)
            v = np.asarray(verts, dtype=np.float64)
            c = v.mean(axis=0)
            self._sph_c[name] = c
            self._sph_r[name] = float(np.linalg.norm(v - c, axis=1).max())
            self._hull[name] = _hull_vertices(v)
        self._names = list(self._models)
        # The broadphase's spheres as arrays, in the order of ``_names``; the arm's own pairs the rule checks, listed once
        # (the rule reads frames, which never change); the finger rows, which a named part's surface holds at no distance.
        self._radius_row = np.asarray([self._sph_r[name] for name in self._names], dtype=np.float64)
        self._centre_rows = np.asarray([self._sph_c[name] for name in self._names], dtype=np.float64).reshape(-1, 3)
        self._finger_rows = np.asarray([name in FINGER_PARTS for name in self._names], dtype=bool)
        self._pairs = [(i, j) for i in range(len(self._names)) for j in range(i + 1, len(self._names))
                       if self.checks(self._names[i], self._names[j])]
        # The engine objects of the fixture tuples judged last, by the tuple itself (``_probes``).
        self._probe_cache: dict[int, tuple[tuple, _Probes]] = {}
        # A robot base's solid, by its radius and top (``base_near_mm``).
        self._bases: dict[tuple[float, float], Any] = {}
        # For a path judged whole (``first_suspect``): the DH frame of each part and the pairs, as index arrays; each
        # part's convex hull as an engine object, built the first time a path is judged whole.
        self._frame_rows = np.asarray([self._frame[name] for name in self._names], dtype=np.int64)
        self._pair_first = np.asarray([i for i, _ in self._pairs], dtype=np.int64)
        self._pair_second = np.asarray([j for _, j in self._pairs], dtype=np.int64)
        self._hulls: "dict[str, Any] | None" = None

    def checks(self, part_a: str, part_b: str) -> bool:
        """Whether :meth:`evaluate` judges ``part_a`` against ``part_b``: its pair rule, asked of one pair.

        Two different parts this backend holds, more than one DH frame apart, or one frame apart where either is a
        wrist camera's part, which stays checked against wrist_2. It is the rule :meth:`evaluate` skips by, so the
        planner's self pairs left to this guard (``planning.band``) are pairs it judged.
        """
        if part_a == part_b or part_a not in self._frame or part_b not in self._frame:
            return False
        gap = abs(self._frame[part_a] - self._frame[part_b])
        return not (gap == 0 or (gap == 1 and part_a not in self._wrist and part_b not in self._wrist))

    @property
    def part_frames(self) -> dict[str, int]:
        """The DH frame every part this backend holds hangs from, by name."""
        return dict(self._frame)

    def distance_mm(self, transforms_dh_mm: list[np.ndarray], yaw_deg: float, part_a: str, part_b: str) -> float:
        """The exact distance of two held parts at one configuration, in millimetres: for a sentence, never a verdict.

        :meth:`evaluate` names only the first pair closer than its limit; a person told that the planner refused a pose
        the meshes accept wants the distance the meshes keep on the pair the planner named.
        """
        rotation = _yaw_matrix(yaw_deg)
        for name in (part_a, part_b):
            frame = transforms_dh_mm[self._frame[name]]
            self._a.set_transform(self._models[name], rotation @ frame[:3, :3], rotation @ frame[:3, 3])
        return self._a.distance(self._models[part_a], self._models[part_b])

    def lowest_mm(self, transforms_dh_mm: list[np.ndarray], yaw_deg: float, *, from_frame: int = 2) -> float:
        """The lowest point of every part hanging from DH frame ``from_frame`` or past it, at one placement: z in the
        system base frame, mm, exact (the lowest vertex of each part's hull; ``inf`` where no part hangs there).

        The default leaves out the shoulder, which turns about Z on the base and stands where it stands whatever the
        joints do. Read beside a support plane only the planner holds (``mesh_first``), never as a verdict of its own.
        """
        rotation = _yaw_matrix(yaw_deg)
        low = float("inf")
        for name in self._names:
            frame = self._frame[name]
            if frame < from_frame:
                continue
            placed = transforms_dh_mm[frame]
            row = (rotation @ placed[:3, :3])[2]
            low = min(low, float((self._hull[name] @ row).min() + (rotation @ placed[:3, 3])[2]))
        return low

    def base_near_mm(
        self, transforms_dh_mm: list[np.ndarray], yaw_deg: float, *, radius_mm: float, top_mm: float, within_mm: float,
        from_frame: int = 2,
    ) -> tuple[str, float]:
        """The part hanging from DH frame ``from_frame`` or past it that comes nearest the robot's base, and how near:
        the base a solid cylinder about base Z, ``radius_mm`` round, from the mounting face (base z 0) up to ``top_mm``.

        Exact for every part within ``within_mm`` of the base: the engine's distance from the part's mesh to the solid,
        0 where they touch or overlap. A part whose bounding sphere keeps more than ``within_mm`` from the cylinder is
        passed over, which can never pass over one within it; where every part is passed over, ``("", inf)``. The
        default leaves out the shoulder, which stands on the base. Read beside a base only the planner holds
        (``mesh_first``), never as a verdict of its own.
        """
        radius, top = float(radius_mm), float(top_mm)
        base = self._bases.get((radius, top))
        if base is None:
            base = self._bases[(radius, top)] = self._a.cylinder_object(radius, top, np.array([0.0, 0.0, top / 2.0]))
        rotation = _yaw_matrix(yaw_deg)
        nearest, near = "", float("inf")
        for row, name in enumerate(self._names):
            frame = self._frame[name]
            if frame < from_frame:
                continue
            placed = transforms_dh_mm[frame]
            turn, at = rotation @ placed[:3, :3], rotation @ placed[:3, 3]
            centre = turn @ self._centre_rows[row] + at
            beside = max(0.0, math.hypot(float(centre[0]), float(centre[1])) - radius)
            above = max(0.0, float(centre[2]) - top, -float(centre[2]))
            if math.hypot(beside, above) - float(self._radius_row[row]) > float(within_mm):
                continue
            self._a.set_transform(self._models[name], turn, at)
            gap = max(0.0, self._a.distance(self._models[name], base))
            if gap < near:
                nearest, near = name, gap
        return nearest, near

    def evaluate(
        self,
        transforms_dh_mm: list[np.ndarray],
        yaw_deg: float,
        fixtures: tuple,
        min_distance_mm: float,
        broadphase: bool = False,
        *,
        arm_pairs: bool = True,
    ) -> tuple[str, float] | None:
        """Return ``(pair, signed_distance_mm)`` for the first violating pair, else ``None``.

        ``transforms_dh_mm`` is the full per-frame DH FK from ``ur_link_transforms_mm``.
        ``fixtures`` are :class:`AxisAlignedBox` values, centre and half-extents in mm,
        in the system base frame; one that carries its ``turned`` box is judged as that
        box, turned, which the enclosure holds. ``broadphase``, which the continuous monitor uses,
        drops a pair on its bounding spheres before the exact query where the two
        cannot be within ``min_distance_mm``. The cull is conservative, because the
        spheres bound the meshes, so the verdict matches the brute path exactly, and
        the default of ``False`` leaves the one-shot guard unchanged. ``arm_pairs`` false
        judges the fixtures alone, for a caller that judged the arm against itself already.
        """
        a = self._a
        Rb = _yaw_matrix(yaw_deg)
        centres = np.empty((len(self._names), 3), dtype=np.float64)  # world-frame sphere centroids, broadphase only
        turns = np.empty((len(self._names), 3, 3), dtype=np.float64)  # world-frame turns, broadphase only
        # Place every link mesh in the system base frame: R_base @ T_dh[frame].
        for row, name in enumerate(self._names):
            T = transforms_dh_mm[self._frame[name]]
            R = Rb @ T[:3, :3]
            t = Rb @ T[:3, 3]
            a.set_transform(self._models[name], R, t)
            if broadphase:
                centres[row] = R @ self._centre_rows[row] + t
                turns[row] = R
        # ---- link against link, skipping frames within 1 of each other, which are the
        # adjacent joints and the rigid wrist and gripper cluster ----
        # A wrist camera's part is skipped only on its own frame, so against wrist_2, one frame in, it
        # stays checked.
        if arm_pairs and self._pairs:
            near_pairs: "list[bool] | None" = None
            if broadphase:
                first = np.fromiter((i for i, _ in self._pairs), dtype=np.int64, count=len(self._pairs))
                second = np.fromiter((j for _, j in self._pairs), dtype=np.int64, count=len(self._pairs))
                gaps = (np.linalg.norm(centres[first] - centres[second], axis=1)
                        - self._radius_row[first] - self._radius_row[second])
                near_pairs = (gaps <= min_distance_mm).tolist()
            for index, (i, j) in enumerate(self._pairs):
                if near_pairs is not None and not near_pairs[index]:
                    continue  # the spheres are too far apart to violate, so skip the exact query
                ni, nj = self._names[i], self._names[j]
                d = a.distance(self._models[ni], self._models[nj])
                if d < min_distance_mm:
                    return (f"{ni}|{nj}", d)
        # ---- link against fixture: the box as it stands, turned or tilted where it is; the sphere about its
        # enclosure, which holds the turned or tilted box too, so the broadphase never culls a box the exact query
        # would refuse ----
        if not fixtures:
            return None
        probes = self._probes(fixtures)
        near: "np.ndarray | None" = None
        if broadphase:
            # A finger against a named part's measured surface keeps no distance, every other pair the limit.
            limits = np.where(self._finger_rows[:, None] & probes.soft[None, :], 0.0, float(min_distance_mm))
            gaps = (np.linalg.norm(centres[:, None, :] - probes.centres[None, :, :], axis=2)
                    - self._radius_row[:, None] - probes.radii[None, :])
            near = gaps <= limits
            # Where a part was measured against a box before: it can have come no nearer than that distance less how
            # far any point of it moved since, its sphere's centre plus its radius times the chord of its turn.
            known = ~np.isnan(probes.measured)
            if near.any() and known.any():
                # The chord from the difference of the turns, as _chord takes it: the trace form rounds to 0 where a
                # part barely turns and left up to about 1e-5 mm of travel unbounded (2026-10-09).
                chord = (np.linalg.norm((probes.turns - turns[:, None]).reshape(*probes.turns.shape[:2], 9), axis=2)
                         * _HALF_SQRT2 + _CHORD_PAD)
                moved = (np.linalg.norm(centres[:, None, :] - probes.at, axis=2)
                         + self._radius_row[:, None] * chord)
                near &= ~(known & (probes.measured - moved - _ADVANCE_EPS_MM >= limits))
            if not near.any():
                return None
        columns = range(len(fixtures)) if near is None else np.flatnonzero(near.any(axis=0)).tolist()
        for column in columns:
            fx = fixtures[column]
            box, surface, reading = probes.boxes[column], probes.surfaces[column], probes.readings[column]
            for row, name in enumerate(self._names):
                if near is not None and not near[row, column]:
                    continue
                if surface is not None and name in FINGER_PARTS:
                    probe, limit = surface, 0.0
                elif reading is not None and name in FINGER_PARTS:
                    probe, limit = reading, min_distance_mm
                else:
                    probe, limit = box, min_distance_mm
                d = a.distance(self._models[name], probe)
                if broadphase:
                    probes.measured[row, column] = d
                    probes.turns[row, column] = turns[row]
                    probes.at[row, column] = centres[row]
                if d < limit:
                    fname = getattr(fx, "name", "") or "fixture"
                    return (f"{name}|fixture:{fname}", d)
        return None

    def _probes(self, fixtures: tuple) -> "_Probes":
        """The engine objects ``fixtures`` are judged as, and the spheres about their enclosures, built once per tuple.

        The box as it stands, turned or tilted where it is; for the fingers, a part the detector named as its measured
        surface (the box less ``soft_mm`` on its sides and its top, its foot where it stands, at no distance) and a
        support's solid at its reading and excess (``finger_top_mm``), the owner's rules of 2026-10-05. The guard hands
        the same tuple for every sample until its world changes, so the objects of the last few tuples are kept, by the
        tuple itself; any other sequence is built for the one call.
        """
        if isinstance(fixtures, tuple):
            cached = self._probe_cache.get(id(fixtures))
            if cached is not None and cached[0] is fixtures:
                return cached[1]
        a = self._a
        boxes: list[Any] = []
        surfaces: list[Any] = []
        readings: list[Any] = []
        centres = np.zeros((len(fixtures), 3), dtype=np.float64)
        radii = np.zeros(len(fixtures), dtype=np.float64)
        soft_columns = np.zeros(len(fixtures), dtype=bool)
        for column, fx in enumerate(fixtures):
            fc = np.asarray(fx.center_mm, dtype=np.float64)
            centres[column] = fc
            radii[column] = float(np.linalg.norm(np.asarray(fx.half_extents_mm, dtype=np.float64)))
            turned = getattr(fx, "turned", None)
            if turned is None:
                boxes.append(a.box_object(np.asarray(fx.half_extents_mm, dtype=np.float64), fc))
            else:
                boxes.append(a.box_object(np.asarray(turned.half_extents_mm, dtype=np.float64), fc,
                                          float(turned.yaw_rad), getattr(turned, "rotation", None)))
            soft = float(getattr(fx, "soft_mm", 0.0) or 0.0)
            surface = _surface_of(a, fx, soft) if soft > 0.0 else None
            surfaces.append(surface)
            soft_columns[column] = surface is not None
            drop = float(getattr(fx, "finger_top_mm", 0.0) or 0.0)
            readings.append(_finger_top_of(a, fx, drop) if drop > 0.0 and surface is None else None)
        shape = (len(self._names), len(fixtures))
        probes = _Probes(boxes=tuple(boxes), surfaces=tuple(surfaces), readings=tuple(readings), centres=centres,
                         radii=radii, soft=soft_columns, measured=np.full(shape, np.nan),
                         turns=np.zeros((*shape, 3, 3)), at=np.zeros((*shape, 3)))
        if isinstance(fixtures, tuple):
            while len(self._probe_cache) >= _PROBE_TUPLES_KEPT:
                self._probe_cache.pop(next(iter(self._probe_cache)))
            self._probe_cache[id(fixtures)] = (fixtures, probes)
        return probes

    # ------------------------------------------------------------------
    # A whole path at once (safety.self_collision.whole_path_judge)
    # ------------------------------------------------------------------

    def first_suspect(
        self,
        transforms_dh_mm: np.ndarray,
        yaw_deg: float,
        *,
        declared: tuple,
        min_distance_mm: float,
        seen: tuple = (),
        seen_min_distance_mm: "float | None" = None,
        until: "int | None" = None,
    ) -> int:
        """The first sample of a path :meth:`evaluate` might refuse, never later than the first it does refuse; the
        path's length, or ``until``, where it refuses none before.

        ``transforms_dh_mm`` is every sample's DH chain at once, ``(N, 7, 4, 4)`` (``ur_link_transforms_mm_many``), and
        the rest is what the guard hands :meth:`evaluate` at every sample: the arm against itself and the ``declared``
        fixtures at ``min_distance_mm``, the boxes a camera ``seen`` at ``seen_min_distance_mm``, each part against each
        box as the probe and the limit :meth:`evaluate` takes for it. It never refuses and never names a pair: the guard
        judges from the sample it returns on with :meth:`evaluate`, one sample at a time, and says what refused.

        A sample before the one it returns is passed only on a proof that every distance :meth:`evaluate` would measure
        there keeps its limit: the bounding spheres :meth:`evaluate` culls by, asked of the whole path at once; the
        parts' convex hulls (a hull holds its mesh, so a hull that keeps the limit plus ``_HULL_SLACK_MM`` proves the
        mesh does; Coal only); or a distance measured at an earlier sample less how far the pair can have moved since
        (:func:`_moved_mm`; an arm pair in either part's frame, whichever bounds tighter). Where none proves it, the pair
        is measured as :meth:`evaluate` measures it, and a distance under the limit plus ``_SUSPECT_SLACK_MM`` flags the
        sample. ``until`` stops the walk where the caller knows of an earlier sample to start from already.
        """
        frames = np.asarray(transforms_dh_mm, dtype=np.float64)
        total = int(frames.shape[0])
        first = total if until is None else max(0, min(int(until), total))
        if first == 0 or not self._names:
            return first
        placed = self._placements(frames[:first], yaw_deg)
        hulls = self._hull_objects()
        first = self._first_among_pairs(placed, hulls, float(min_distance_mm), first)
        seen_limit = float(min_distance_mm if seen_min_distance_mm is None else seen_min_distance_mm)
        for fixtures, limit in ((declared, float(min_distance_mm)), (seen, seen_limit)):
            if fixtures and first > 0:
                first = self._first_against(fixtures, placed, hulls, limit, first)
        return first

    def first_near_base(
        self, transforms_dh_mm: np.ndarray, yaw_deg: float, *, radius_mm: float, top_mm: float, within_mm: float,
        from_frame: int = 2,
    ) -> int:
        """The first configuration of a path where a part hanging from DH frame ``from_frame`` or past it may come within
        ``within_mm`` of the robot's base, :meth:`base_near_mm`'s solid cylinder; never later than the first where
        :meth:`base_near_mm` says one does, and the path's length where none may.

        :meth:`base_near_mm`'s question asked of a whole path at once (the UR driver's mesh first,
        ``planning.band.ExactPairs.first_near_base``): a part is passed where its sphere keeps more than ``within_mm``
        from the cylinder, as :meth:`base_near_mm` passes it over, where its hull keeps that much (Coal), or where a
        distance measured at an earlier configuration less how far the part moved since does; else it is measured, and
        a distance under ``within_mm`` plus ``_SUSPECT_SLACK_MM`` flags the configuration. :meth:`base_near_mm` asked of
        that one says which part comes how near.
        """
        frames = np.asarray(transforms_dh_mm, dtype=np.float64)
        total = int(frames.shape[0])
        if not total or not self._names:
            return total
        radius, top, within = float(radius_mm), float(top_mm), float(within_mm)
        base = self._bases.get((radius, top))
        if base is None:
            base = self._bases[(radius, top)] = self._a.cylinder_object(radius, top, np.array([0.0, 0.0, top / 2.0]))
        placed = self._placements(frames, yaw_deg)
        hulls = self._hull_objects()
        centres = placed.centres
        beside = np.maximum(0.0, np.hypot(centres[..., 0], centres[..., 1]) - radius)
        above = np.maximum(0.0, np.maximum(centres[..., 2] - top, -centres[..., 2]))
        near = np.hypot(beside, above) - self._radius_row[None, :] <= within + _SUSPECT_SLACK_MM
        first = total
        for row in np.flatnonzero(near.any(axis=0)).tolist():
            if self._frame[self._names[row]] < from_frame:
                continue

            def measure(sample: int, row: int = row) -> float:
                self._place(row, sample, placed, hulls)
                return max(0.0, self._bound_mm(row, base, base, within, hulls))

            def moved(sample: int, later: np.ndarray, row: int = row) -> np.ndarray:
                return _moved_mm(centres[sample, row], placed.turns[sample, row], centres[later, row],
                                 placed.turns[later, row], float(self._radius_row[row]))

            first = _first_flagged(near[:, row], first, within, measure, moved)
            if first == 0:
                return 0
        return first

    def lowest_mm_many(self, transforms_dh_mm: np.ndarray, yaw_deg: float, *, from_frame: int = 2) -> np.ndarray:
        """:meth:`lowest_mm` of every configuration of a path at once, ``(N,)``: the same hull vertices turned the same
        way, so each equals what :meth:`lowest_mm` says of that configuration within float rounding (1e-9 mm; the UR
        driver's mesh first asks :meth:`lowest_mm` of any it would refuse, ``planning.band.ExactPairs.first_low``)."""
        frames = np.asarray(transforms_dh_mm, dtype=np.float64)
        rotation = _yaw_matrix(yaw_deg)
        low = np.full(frames.shape[0], np.inf)
        for name in self._names:
            frame = self._frame[name]
            if frame < from_frame:
                continue
            rows = np.matmul(rotation, frames[:, frame, :3, :3])[:, 2]
            heights = np.matmul(rotation, frames[:, frame, :3, 3:4])[:, 2, 0]
            hull = self._hull[name]
            step = max(1, _LOWEST_CHUNK // max(1, len(hull)))
            for start in range(0, len(rows), step):
                stop = start + step
                reached = (hull @ rows[start:stop].T).min(axis=0) + heights[start:stop]
                low[start:stop] = np.minimum(low[start:stop], reached)
        return low

    def _placements(self, transforms_dh_mm: np.ndarray, yaw_deg: float) -> _Placements:
        """Every part at every sample of ``transforms_dh_mm`` ``(N, 7, 4, 4)``: ``R_base(yaw) @ T_dh[frame]``, the
        product :meth:`evaluate` forms for one sample, and each part's sphere centre."""
        rotation = _yaw_matrix(yaw_deg)
        held = np.asarray(transforms_dh_mm, dtype=np.float64)[:, self._frame_rows]
        turns = np.matmul(rotation, held[..., :3, :3])
        origins = np.matmul(rotation, held[..., :3, 3:4])[..., 0]
        centres = np.matmul(turns, self._centre_rows[:, :, None])[..., 0] + origins
        return _Placements(turns=turns, origins=origins, centres=centres)

    def _hull_objects(self) -> "dict[str, Any]":
        """Each part's convex hull as an engine object (``_EngineAdapter.convex_object``), built from the hull vertices
        :meth:`lowest_mm` reads, once, the first time a path is judged whole. None where the engine builds no hull
        (python-fcl); a part whose hull the engine cannot build has none, and its mesh is measured."""
        if self._hulls is None:
            build = getattr(self._a, "convex_object", None)
            hulls: dict[str, Any] = {}
            if callable(build):
                for name in self._names:
                    try:
                        hull = build(self._hull[name])
                    except Exception:  # noqa: BLE001 - a part the engine cannot hull is measured on its mesh
                        hull = None
                    if hull is not None:
                        hulls[name] = hull
            self._hulls = hulls
            if hulls:
                _LOGGER.info("whole-path judge: the convex hulls of %d of %d parts (%s) clear what they can before a "
                             "mesh is measured", len(hulls), len(self._names), self.engine)
            else:
                _LOGGER.info("whole-path judge: the %s engine builds no convex hull, so every distance the spheres and "
                             "the motion bound leave open is measured on the meshes", self.engine)
        return self._hulls

    def _place(self, row: int, sample: int, placed: _Placements, hulls: "dict[str, Any]") -> None:
        """Stand part ``row``'s mesh, and its hull where it has one, where the part stands at ``sample``."""
        name = self._names[row]
        turn, origin = placed.turns[sample, row], placed.origins[sample, row]
        self._a.set_transform(self._models[name], turn, origin)
        hull = hulls.get(name)
        if hull is not None:
            self._a.set_transform(hull, turn, origin)

    def _bound_mm(self, row: int, other: Any, other_hull: Any, limit: float, hulls: "dict[str, Any]") -> float:
        """A lower bound on the exact distance from part ``row`` to ``other``, both placed.

        Where the part has a hull and ``other_hull`` stands for ``other`` (the other part's hull, or ``other`` itself
        where it is a box or the base, convex already): the hulls' distance less ``_HULL_SLACK_MM``, if that alone keeps
        ``limit`` plus ``_SUSPECT_SLACK_MM``. Otherwise the distance :meth:`evaluate` measures, the same query.
        """
        name = self._names[row]
        hull = hulls.get(name)
        if hull is not None and other_hull is not None:
            bound = self._a.distance(hull, other_hull) - _HULL_SLACK_MM
            if bound >= limit + _SUSPECT_SLACK_MM:
                return bound
        return self._a.distance(self._models[name], other)

    def _pair_motion(self, placed: _Placements, first: int, second: int) -> Any:
        """``moved(sample, later)``: how much nearer one another the parts ``first`` and ``second`` can have come since
        ``sample``, at each of ``later``. In one part's own frame only the other moves, its sphere's centre and its turn
        there; the smaller of the two ways round bounds it (the turn between is the same both ways)."""
        turn_first, turn_second = placed.turns[:, first], placed.turns[:, second]
        turns = np.einsum("nki,nkj->nij", turn_first, turn_second)  # the second's turn in the first's frame
        second_seen = np.einsum("nki,nk->ni", turn_first, placed.centres[:, second] - placed.origins[:, first])
        first_seen = np.einsum("nki,nk->ni", turn_second, placed.centres[:, first] - placed.origins[:, second])
        radius_first, radius_second = float(self._radius_row[first]), float(self._radius_row[second])

        def moved(sample: int, later: np.ndarray) -> np.ndarray:
            chord = _chord(turns[sample], turns[later])
            return np.minimum(np.linalg.norm(second_seen[later] - second_seen[sample], axis=1) + radius_second * chord,
                              np.linalg.norm(first_seen[later] - first_seen[sample], axis=1) + radius_first * chord)

        return moved

    def _first_among_pairs(self, placed: _Placements, hulls: "dict[str, Any]", limit: float, until: int) -> int:
        """The first sample before ``until`` where two of the arm's own parts may come under ``limit``
        (:meth:`first_suspect`), else ``until``."""
        if not self._pairs:
            return until
        centres = placed.centres
        gaps = (np.linalg.norm(centres[:, self._pair_first] - centres[:, self._pair_second], axis=2)
                - self._radius_row[self._pair_first] - self._radius_row[self._pair_second])
        near = gaps <= limit + _SUSPECT_SLACK_MM
        for index in np.flatnonzero(near.any(axis=0)).tolist():
            if not near[:until, index].any():
                continue
            first, second = self._pairs[index]
            other = self._models[self._names[second]]
            other_hull = hulls.get(self._names[second])

            def measure(sample: int, first: int = first, second: int = second, other: Any = other,
                        other_hull: Any = other_hull) -> float:
                self._place(first, sample, placed, hulls)
                self._place(second, sample, placed, hulls)
                return self._bound_mm(first, other, other_hull, limit, hulls)

            until = _first_flagged(near[:, index], until, limit, measure, self._pair_motion(placed, first, second))
            if until == 0:
                return 0
        return until

    def _first_against(
        self, fixtures: tuple, placed: _Placements, hulls: "dict[str, Any]", limit: float, until: int,
    ) -> int:
        """The first sample before ``until`` where a part may come under its limit from one of ``fixtures``
        (:meth:`first_suspect`), else ``until``: each part against each box with the probe and the limit
        :meth:`evaluate` takes for it, a finger at no distance from a named part's surface."""
        probes = self._probes(fixtures)
        limits = np.where(self._finger_rows[:, None] & probes.soft[None, :], 0.0, float(limit))
        centres = placed.centres[:until]
        radius = self._radius_row
        # Each part's sphere swept over the path, a box about its centres grown by its radius, against each box's sphere:
        # a box no part's sweep comes near is not looked at again. Never more than the spheres pass.
        low = centres.min(axis=0) - radius[:, None]
        high = centres.max(axis=0) + radius[:, None]
        outside = np.maximum(0.0, np.maximum(low[:, None, :] - probes.centres[None, :, :],
                                             probes.centres[None, :, :] - high[:, None, :]))
        maybe = np.linalg.norm(outside, axis=2) <= probes.radii[None, :] + limits + _SUSPECT_SLACK_MM
        for column in np.flatnonzero(maybe.any(axis=0)).tolist():
            rows = np.flatnonzero(maybe[:, column])
            gaps = (np.linalg.norm(centres[:, rows] - probes.centres[column][None, None, :], axis=2)
                    - radius[rows][None, :] - probes.radii[column])
            near = gaps <= limits[rows, column][None, :] + _SUSPECT_SLACK_MM
            box, surface, reading = probes.boxes[column], probes.surfaces[column], probes.readings[column]
            for slot, row in enumerate(rows.tolist()):
                if not near[:until, slot].any():
                    continue
                name = self._names[row]
                if surface is not None and name in FINGER_PARTS:
                    probe, probe_limit = surface, 0.0
                elif reading is not None and name in FINGER_PARTS:
                    probe, probe_limit = reading, float(limit)
                else:
                    probe, probe_limit = box, float(limit)

                def measure(sample: int, row: int = row, probe: Any = probe, probe_limit: float = probe_limit) -> float:
                    self._place(row, sample, placed, hulls)
                    return self._bound_mm(row, probe, probe, probe_limit, hulls)

                def moved(sample: int, later: np.ndarray, row: int = row) -> np.ndarray:
                    return _moved_mm(placed.centres[sample, row], placed.turns[sample, row], placed.centres[later, row],
                                     placed.turns[later, row], float(radius[row]))

                until = _first_flagged(near[:, slot], until, probe_limit, measure, moved)
                if until == 0:
                    return 0
        return until


def _hull_vertices(vertices: np.ndarray) -> np.ndarray:
    """The vertices of the convex hull of ``vertices``, which hold the part's lowest point however it is placed; every
    vertex where no hull is computed (no scipy, a flat part), which is as exact and slower."""
    try:
        from scipy.spatial import ConvexHull  # noqa: PLC0415

        return np.asarray(vertices[ConvexHull(vertices).vertices], dtype=np.float64)
    except Exception:  # noqa: BLE001 - a part qhull cannot hull is kept whole
        return np.asarray(vertices, dtype=np.float64)


def mesh_backend_status(
    model: str, mesh_dir: str | None = None, mesh_name: str | None = None
) -> str:
    """Why the exact-mesh backend can or cannot run for ``model``, as one stable token.

    It returns ``"ok"``; ``"unknown_model"``, where there is no bundled DH chain and the
    link meshes cannot be placed; ``"no_bundle"``, where there is no
    ``{model}_collision_meshes.npz``; ``"no_hand_bundle"``, where the named hand has no
    ``{hand}_hand_meshes.npz`` to compose onto the arm; ``"hand_bundle_refused"``, where it
    has one and ``_hand_bundle.hand_bundle_refusal`` refuses it; or ``"no_engine"``, where
    neither Coal nor python-fcl imports, which is the accepted condition on a host
    without either.

    There is no ``variant_model_mismatch`` token. Which arms a hand may be composed onto is
    measured, one evidence file per combination (``planning/evidence.py``), and a guard that
    also asked a list of arms recorded in the bundle would be a second admission beside the
    evidence, answering for arms nobody measured in either direction.

    ``mesh_name`` names the hand whose own bundle is composed onto the arm at load, and
    ``None`` is the arm bundle as it stands, which carries the 2F-85.

    There is deliberately no ``primitive_colliders`` token. The one arm whose USD collides
    with primitives is baked from its URDF package instead, so nothing reaches that state
    and a rule for it could never fire.

    The model and bundle checks come first and need no collision engine, which makes
    the answer deterministic and lets the caller tell a host that simply has no engine,
    which is expected, from a cell configured for a robot with no collision geometry
    here, which is a misconfiguration that quietly weakens the guard.
    """
    if model.lower() not in UR_DH_TABLES_M:
        return "unknown_model"
    folder = Path(mesh_dir) if mesh_dir else COLLISION_MESH_DIR
    if not (folder / f"{model.lower()}_collision_meshes.npz").exists():
        return "no_bundle"
    if mesh_name:
        hand_path = folder / hand_mesh_bundle(mesh_name).name
        if not hand_path.exists():
            return "no_hand_bundle"
        with np.load(hand_path, allow_pickle=True) as data:
            if hand_bundle_refusal({key: data[key] for key in data.files}, name=hand_path.name) is not None:
                return "hand_bundle_refused"
    mod, kind = import_collision_engine()
    if mod is None or kind is None:
        return "no_engine"
    return "ok"


#: The bundle arrays that belong to the hand rather than the arm. `wrist_3` shares their frame and
#: is arm, which is why this is a list of names and not "everything at frame 6". Owned by
#: ``planning/_hand_bundle``.
_HAND_PARTS = HAND_PARTS

#: The npz key a standalone-asset bake stamps, and the value that obliges a reader to add a plate.
_ORIGIN_KEY = "gripper__origin"
_MOUNTING_FACE = "mounting_face"

#: The tool approach axis in the wrist_3 frame, along which the coupling plate stacks. Measured off
#: the committed bundles: the fingers sit at y in [84.3, 146.2] and the palm at y in [-4.9, 99.2].
_APPROACH_AXIS = 1


def make_backend(
    model: str,
    mesh_dir: str | None = None,
    mesh_name: str | None = None,
    coupling_mm: float = 0.0,
    *,
    placement: HandPlacement | None = None,
    coupling_boxes: "tuple[Box, ...] | list[Box]" = (),
    wrist_parts: "dict[str, np.ndarray] | None" = None,
) -> MeshSelfCollisionBackend | None:
    """Build the mesh backend for ``model``, Coal where available and python-fcl otherwise, or ``None``.

    ``None`` is the documented signal for the guard to fall back to the capsule path,
    and it is never silent: every ``None`` is logged with the
    :func:`mesh_backend_status` token. The capsule fallback is markedly weaker, since
    its default 60 mm link radius already over-rejects legitimate UR5e reach-down
    grasps, which the ``self_collision`` block in ``robot.sim.yaml`` sets out, and a
    cell that swaps an exact-mesh guard for it quietly is a safety-relevant regression
    nobody would notice.

    Any UR model with a bundled DH chain and a committed mesh bundle is accepted, so a
    model starts using exact meshes as soon as its
    ``{model}_collision_meshes.npz`` lands, with no code change. ``mesh_name`` names the
    hand whose own ``{hand}_hand_meshes.npz`` is composed onto the arm, and ``None``
    keeps the arm bundle as it stands, which carries the 2F-85.

    ``placement`` is where the resolved hand sits on the flange, as ``PlannerHand``
    resolved it, and it is applied to the hand parts after the plate. ``None`` is a guard
    built without a hand, on the hand models' own axes. A refused placement never reaches
    here, because the cell refuses to build first. ``coupling_boxes`` are the plates the cell
    declared with a cross section, added as parts of their own (:func:`composed_parts`).

    ``wrist_parts`` are a wrist camera's parts as ``body_link.WristBody.guard_parts`` gives them
    (``<name>__v``, ``__f``, ``__frame``), added beside what :func:`composed_parts` holds and
    checked against wrist_2. A part whose name is already held is refused before any engine is
    built, so nothing is overwritten in silence.
    """
    wrist = _wrist_meshes(wrist_parts or {})
    status = mesh_backend_status(model, mesh_dir, mesh_name)
    if status != "ok":
        _LOGGER.warning(
            "exact-mesh self-collision unavailable for model %r (%s): falling back to the capsule guard, "
            "which is coarser and over-rejects (default %.0f mm link radius). %s",
            model, status, _CAPSULE_FALLBACK_RADIUS_HINT_MM,
            _STATUS_HINTS.get(status, ""),
        )
        return None
    mod, kind = import_collision_engine()
    if mod is None or kind is None:  # narrowing only; status "ok" already proved the import
        return None
    meshes = composed_parts(model, mesh_dir, mesh_name, coupling_mm,
                            placement=placement, coupling_boxes=coupling_boxes)
    clash = sorted(set(meshes) & set(wrist))
    if clash:
        raise ValueError(f"the wrist camera part(s) {clash} are already held by the guard's arm, hand or plates")
    meshes.update(wrist)
    # A cell without a camera hands the backend no wrist argument at all; only a camera's parts are named.
    marked: dict[str, frozenset[str]] = {"wrist_parts": frozenset(wrist)} if wrist else {}
    try:
        return MeshSelfCollisionBackend(_EngineAdapter(mod, kind), meshes, **marked)
    except Exception:  # noqa: BLE001 (any engine construction failure falls back to capsules)
        return None


def _wrist_meshes(parts: "dict[str, np.ndarray]") -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """``<name>__v``/``__f``/``__frame`` arrays as the guard's ``(vertices, faces, frame)`` parts."""
    names = sorted({key.rsplit("__", 1)[0] for key in parts})
    return {name: (np.asarray(parts[f"{name}__v"], dtype=np.float64), np.asarray(parts[f"{name}__f"]),
                   int(np.asarray(parts[f"{name}__frame"]).reshape(-1)[0]))
            for name in names}


def composed_parts(
    model: str,
    mesh_dir: str | None = None,
    mesh_name: str | None = None,
    coupling_mm: float = 0.0,
    *,
    placement: HandPlacement | None = None,
    coupling_boxes: "tuple[Box, ...] | list[Box]" = (),
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """The arrays the exact-mesh guard judges, placed, before any engine is asked to hold them.

    Separate from :func:`make_backend` so the combination evidence hashes what the guard holds
    (``planning.evidence.guard_sha256``) rather than a second assembly of the same idea. A hash
    with two producers that can drift apart proves nothing about the cell it admits. A wrist
    camera's parts, which the guard also holds, are not here: the evidence is keyed without the
    camera, and the planner start proves the camera's cover instead
    (``body_link.wrist_body_refusal``).
    """
    data = compose_collision_meshes(model, mesh_name, mesh_dir)
    names = sorted({k.split("__")[0] for k in data if not k.endswith("__origin")})
    # A mounting-face bundle is not where the hand is. It starts at the gripper's own mounting
    # face, so the coupling plate between that face and the flange is added here, which is what
    # the bake module stamps the origin for.
    origin = ""
    if _ORIGIN_KEY in data:
        origin = str(np.asarray(data[_ORIGIN_KEY]).reshape(-1)[0])
    if origin == _MOUNTING_FACE and float(coupling_mm) == 0.0:
        _LOGGER.warning(
            "%s is stamped origin=%r, so its gripper meshes start at the hand's MOUNTING FACE and a "
            "coupling plate has to be added before they are where the hand is. "
            "The coupling is 0.0 (robot.gripper.coupling_plates sums to nothing, or the guard was "
            "built without a hand), so nothing was added and this guard models "
            "the hand one plate closer to the flange than it is. That is the conservative direction "
            "for arm-versus-hand, and the planner's hand link reads the same plates, so it models the "
            "hand there too. Measure the plate once and set robot.gripper.coupling_plates and "
            "robot.gripper.tool_frame.offset_mm from it.",
            f"{mesh_name}_hand_meshes.npz", origin,
        )
    meshes: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for n in names:
        verts = place_hand_vertices(n, data[f"{n}__v"], origin=origin, coupling_mm=coupling_mm, placement=placement)
        meshes[n] = (verts, data[f"{n}__f"], int(data[f"{n}__frame"][0]))
    # The plates a cell measured across, as bodies. They sit between the flange and the hand's
    # mounting face, nearer the wrist than the hand is. A plate that declared only a thickness is
    # not here: it still places the hand, and the cell names it instead.
    for part, arrays in _coupling_parts(coupling_boxes, placement).items():
        meshes[part] = arrays
    return meshes


#: The frame a coupling body sits in: tool0, which is DH frame 6 and the hand's own. A plate is
#: bolted to the flange, so the guard's pair rule skips it against wrist_3 and wrist_2 exactly as it
#: skips the hand.
_COUPLING_FRAME = 6


def _coupling_parts(
    boxes: "tuple[Box, ...] | list[Box]", placement: HandPlacement | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """The declared plates as guard parts, turned by the placement like the hand they carry.

    The boxes are written in the hand model's own axes, where the stack grows along the approach, so
    a declared tool frame turns them with everything else. There is no plate shift: the boxes already
    sit between the flange at zero and the hand's mounting face one stack out.
    """
    if not len(boxes):
        return {}
    from .planning._declared_body import boxes_to_parts

    arrays = boxes_to_parts(list(boxes), frame=_COUPLING_FRAME)
    turn = placement is not None and not placement.is_identity
    parts: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for key in [k for k in arrays if k.endswith("__v")]:
        name = key[: -len("__v")]
        vertices = np.asarray(arrays[key], dtype=np.float64)
        if turn:
            assert placement is not None  # narrowing only
            vertices = vertices @ np.asarray(placement.rotation, dtype=np.float64).T
        parts[name] = (vertices, np.asarray(arrays[f"{name}__f"]), _COUPLING_FRAME)
    return parts


def place_hand_vertices(
    name: str, vertices: np.ndarray, *, origin: str, coupling_mm: float, placement: HandPlacement | None,
) -> np.ndarray:
    """Where one bundle part sits on the flange.

    A hand part is shifted one plate along its model's approach where its bundle starts at
    the mounting face, and is then turned by the placement. The plate goes on in model axes
    and the rotation after it, because the plate stacks along the hand's approach wherever
    the declared frame points that approach. An arm part comes back untouched, decided by
    name and not by frame, because ``wrist_3`` shares frame 6 with the hand. A part that
    needs neither a plate nor a turn comes back as the very array it was given, so a cell
    that runs today keeps its bytes.
    """
    if name not in _HAND_PARTS:
        return vertices
    shift = float(coupling_mm) if origin == _MOUNTING_FACE else 0.0
    turn = placement is not None and not placement.is_identity
    if not shift and not turn:
        return vertices
    out = vertices.copy()
    if shift:
        out[:, _APPROACH_AXIS] += shift
    if turn:
        assert placement is not None  # narrowing only: turn is False without one
        out = out @ np.asarray(placement.rotation, dtype=np.float64).T
    return out
