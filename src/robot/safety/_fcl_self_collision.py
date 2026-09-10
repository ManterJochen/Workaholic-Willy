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
from .planning.environment import (
    collision_mesh_bundle,
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
    "variant_model_mismatch": (
        "safety.self_collision.collision_mesh_variant names a bundle baked from a different robot: "
        "its arm meshes belong to that arm, and placing them on this one puts every link somewhere "
        "it is not. Bake the variant for this model, or drop the variant and lose only the gripper "
        "geometry."
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

    def __init__(self, adapter: _EngineAdapter, meshes: dict[str, tuple[np.ndarray, np.ndarray, int]]) -> None:
        self._a = adapter
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
        for i in range(len(self._names)):
            ni = self._names[i]
            for j in range(i + 1, len(self._names)):
                nj = self._names[j]
                if abs(self._frame[ni] - self._frame[nj]) <= 1:
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
    ``{model}_collision_meshes.npz``; ``"variant_model_mismatch"``, where the named
    gripper variant was baked from a different robot so its arm meshes belong to
    another arm; or ``"no_engine"``, where neither Coal nor python-fcl imports, which
    is the accepted condition on a host without either.

    ⚠ A ``primitive_colliders`` token lived here for one day, for an arm whose USD collides
    with primitives and was believed unbakeable. It was retracted when that arm was baked
    from its URDF package instead: a state nothing can reach is a rule that can never fire.

    The model and bundle checks come first and need no collision engine, which makes
    the answer deterministic and lets the caller tell a host that simply has no engine,
    which is expected, from a cell configured for a robot with no collision geometry
    here, which is a misconfiguration that quietly weakens the guard.
    """
    if model.lower() not in UR_DH_TABLES_M:
        return "unknown_model"
    default = collision_mesh_bundle(model, mesh_name)
    path = (Path(mesh_dir) / default.name) if mesh_dir else default
    if not path.exists():
        return "no_bundle"
    if mesh_name and _variant_is_for_another_model(model, path, mesh_dir):
        return "variant_model_mismatch"
    mod, kind = import_collision_engine()
    if mod is None or kind is None:
        return "no_engine"
    return "ok"


#: One arm link is enough to tell two robots apart and is cheap to compare. The forearm
#: differs most between UR sizes: the measured z spans are -19.6 to 77.3 mm on a ur3e
#: and -49.5 to 60.4 mm on a ur5e.
_VARIANT_PROBE_LINK = "forearm__v"


def _variant_is_for_another_model(model: str, variant_path: "Path", mesh_dir: str | None) -> bool:
    """Was this variant bundle baked from a robot other than ``model``?

    A variant carries the arm meshes of its source robot and swaps the gripper alone,
    so comparing one arm link against the model own bundle answers the question. An
    unverifiable case returns ``False``: with no bundle for the model there is nothing
    to compare against, and refusing on an unanswered question would send every cell to
    the capsule proxy over a bundle that may be perfectly correct.
    """
    own = collision_mesh_bundle(model)
    own_path = (Path(mesh_dir) / own.name) if mesh_dir else own
    if not own_path.exists() or own_path == variant_path:
        return False
    try:
        import numpy as np

        with np.load(variant_path) as variant, np.load(own_path) as reference:
            if _VARIANT_PROBE_LINK not in variant.files or _VARIANT_PROBE_LINK not in reference.files:
                return False
            a, b = variant[_VARIANT_PROBE_LINK], reference[_VARIANT_PROBE_LINK]
            return a.shape != b.shape or not bool(np.array_equal(a, b))
    except Exception:  # noqa: BLE001 (an unreadable bundle is `no_bundle`'s problem, not this check's)
        return False


def make_backend(
    model: str, mesh_dir: str | None = None, mesh_name: str | None = None
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
    ``{model}_collision_meshes.npz`` lands, with no code change. ``mesh_name`` selects
    a per-mounted-gripper variant bundle, ``{mesh_name}_collision_meshes.npz``, which
    carries the arm meshes of the model it was baked from, so a variant is paired only
    with that same model.
    """
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
    default = collision_mesh_bundle(model, mesh_name)
    fname = default.name
    path = (Path(mesh_dir) / fname) if mesh_dir else default
    data = np.load(path)
    names = sorted({k.split("__")[0] for k in data.files})
    meshes: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for n in names:
        meshes[n] = (data[f"{n}__v"], data[f"{n}__f"], int(data[f"{n}__frame"][0]))
    try:
        return MeshSelfCollisionBackend(_EngineAdapter(mod, kind), meshes)
    except Exception:  # noqa: BLE001 (any engine construction failure falls back to capsules)
        return None
