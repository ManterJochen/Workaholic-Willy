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
  * Each link is also checked against the fixtures, which are axis-aligned boxes.

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

    def box_object(self, half_extents: np.ndarray, center: np.ndarray) -> Any:
        m = self._m
        size = 2.0 * np.asarray(half_extents, dtype=np.float64)
        return m.CollisionObject(m.Box(*size), self._transform(np.eye(3), np.asarray(center, dtype=np.float64)))

    def distance(self, a: Any, b: Any) -> float:
        m = self._m
        return float(m.distance(a, b, m.DistanceRequest(), m.DistanceResult()))


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
        for name, (verts, faces, frame) in meshes.items():
            self._models[name] = adapter.build_object(verts, faces)
            self._frame[name] = int(frame)
            v = np.asarray(verts, dtype=np.float64)
            c = v.mean(axis=0)
            self._sph_c[name] = c
            self._sph_r[name] = float(np.linalg.norm(v - c, axis=1).max())
        self._names = list(self._models)

    def evaluate(
        self,
        transforms_dh_mm: list[np.ndarray],
        yaw_deg: float,
        fixtures: tuple,
        min_distance_mm: float,
        broadphase: bool = False,
    ) -> tuple[str, float] | None:
        """Return ``(pair, signed_distance_mm)`` for the first violating pair, else ``None``.

        ``transforms_dh_mm`` is the full per-frame DH FK from ``ur_link_transforms_mm``.
        ``fixtures`` are :class:`AxisAlignedBox` values, centre and half-extents in mm,
        in the system base frame. ``broadphase``, which the continuous monitor uses,
        drops a pair on its bounding spheres before the exact query where the two
        cannot be within ``min_distance_mm``. The cull is conservative, because the
        spheres bound the meshes, so the verdict matches the brute path exactly, and
        the default of ``False`` leaves the one-shot guard unchanged.
        """
        a = self._a
        Rb = _yaw_matrix(yaw_deg)
        wc: dict[str, np.ndarray] = {}  # world-frame sphere centroids, broadphase only
        # Place every link mesh in the system base frame: R_base @ T_dh[frame].
        for name in self._names:
            T = transforms_dh_mm[self._frame[name]]
            R = Rb @ T[:3, :3]
            t = Rb @ T[:3, 3]
            a.set_transform(self._models[name], R, t)
            if broadphase:
                wc[name] = R @ self._sph_c[name] + t
        # ---- link against link, skipping frames within 1 of each other, which are the
        # adjacent joints and the rigid wrist and gripper cluster ----
        # A wrist camera's part is skipped only on its own frame, so against wrist_2, one frame in, it
        # stays checked.
        for i in range(len(self._names)):
            ni = self._names[i]
            for j in range(i + 1, len(self._names)):
                nj = self._names[j]
                gap = abs(self._frame[ni] - self._frame[nj])
                if gap == 0 or (gap == 1 and ni not in self._wrist and nj not in self._wrist):
                    continue
                if broadphase and (float(np.linalg.norm(wc[ni] - wc[nj]))
                                   - self._sph_r[ni] - self._sph_r[nj] > min_distance_mm):
                    continue  # the spheres are too far apart to violate, so skip the exact query
                d = a.distance(self._models[ni], self._models[nj])
                if d < min_distance_mm:
                    return (f"{ni}|{nj}", d)
        # ---- link against fixture, the axis-aligned boxes ----
        for fx in fixtures:
            fc = np.asarray(fx.center_mm, dtype=np.float64)
            fr = float(np.linalg.norm(np.asarray(fx.half_extents_mm, dtype=np.float64)))
            box = a.box_object(np.asarray(fx.half_extents_mm, dtype=np.float64), fc)
            for name in self._names:
                if broadphase and (float(np.linalg.norm(wc[name] - fc))
                                   - self._sph_r[name] - fr > min_distance_mm):
                    continue
                d = a.distance(self._models[name], box)
                if d < min_distance_mm:
                    fname = getattr(fx, "name", "") or "fixture"
                    return (f"{name}|fixture:{fname}", d)
        return None


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
