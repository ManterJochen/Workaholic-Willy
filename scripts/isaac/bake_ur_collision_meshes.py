"""Bake ``{model}_collision_meshes.npz``, the per-link collision geometry the fail-closed Coal or fcl
self-collision guard checks against, into ``src/robot/safety/data/``.

This is the generator for that safety artifact. It reads the collision meshes out of the Isaac UR USD,
where they sit behind instance proxies, so ``Usd.PrimRange(prim, Usd.TraverseInstanceProxies())`` is
required and a plain ``PrimRange`` finds nothing. Each link's vertices are then re-expressed in the
bundled-DH link frame, which is the frame ``_fcl_self_collision.MeshSelfCollisionBackend`` places them
from::

    M_dh = inv(T_dh[frame](q0)) @ inv(R_base) @ world_usd(q0)

``R_base`` is the ``kinematics_base_yaw_deg`` reconcile, 180 deg, and ``q0`` is the USD's rest configuration,
meaning all joints at zero, the pose the asset composes to before any articulation is applied.

The ur5e run is self-validating: it diffs the freshly computed links against the committed ur5e bundle
and prints the largest per-vertex deviation, which has to come out at about 0 mm. That is the gate, because a wrong frame here corrupts a
safety guard without any other symptom. A deviation at or above the gate refuses the write.

The Robotiq 2F-85 tool0 meshes, ``gripper``, ``lfinger`` and ``rfinger`` at DH frame 6, are
model-independent and are copied verbatim from the ur5e bundle. The URDF joints from ``wrist_3``
through ``flange`` to ``tool0`` are identical across the UR e-series, so the transform from DH frame 6
to ``tool0`` is identity for every model here.

A hand other than that 2F-85 is a bundle of its own, ``{hand}_hand_meshes.npz``, written by
``scripts/grippers/bake_gripper_variant.py`` and composed onto whichever arm the cell carries when the
guard loads. A hand is therefore baked once rather than once per arm, and nothing here is re-run when a
cell changes hands.

Re-baking ``ur5e`` rewrites the reference every other model copies its tool0 arrays from, so the arms
already committed keep the copies they were baked with until they are re-baked as well.

Usage, on the box that has Isaac, under Isaac's own python::

    python.bat scripts/isaac/bake_ur_collision_meshes.py ur5e            # validate, write nothing
    python.bat scripts/isaac/bake_ur_collision_meshes.py ur3e --write

The exit code carries an answer only for the argument check, which runs before Isaac boots. Past that
point it carries nothing: ``SimulationApp.close()`` releases the Kit framework and ends the process
itself, and Isaac's launcher collapses what is left to zero or one. So a refusal here is enforced by
not writing the file and by saying why, and the last lines of the run are what a caller reads.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: scripts/isaac/this.py, then scripts, then the repository root. The layout is the same in a clone.
REPO = Path(__file__).resolve().parents[2]

#: Where the guard reads its bundles from. The library's own answer for a single bundle path is
#: ``src/robot/safety/planning/environment.collision_mesh_bundle(model)``, which anchors the
#: same directory from inside the package. It is not imported here: this file runs under Isaac's
#: interpreter, whose site-packages is not the project environment, so importing the package, which
#: pulls in the whole safety guard stack and from there src.config and pydantic, is not something a
#: bake can rely on. Nothing else about the repository is needed, so the path is anchored instead.
DATA_DIR = REPO / "src" / "robot" / "safety" / "data"

#: Mirrors ``safety.self_collision.kinematics_base_yaw_deg`` in ``config/robot/robot.sim.yaml``. The
#: USD arm faces the other way round from the bundled DH chain, and this is what turns it back.
BASE_YAW_DEG = 180.0
#: The largest per-vertex deviation, in mm, still counted as reproducing the committed ur5e bundle.
GATE_MM = 0.15

_ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
MODEL = (_ARGS[0] if _ARGS else "ur5e").lower()
WRITE = "--write" in sys.argv

# bundle name: (USD link prim, DH frame index)
LINKS = {
    "shoulder": ("shoulder_link", 1),
    "upper_arm": ("upper_arm_link", 2),
    "forearm": ("forearm_link", 3),
    "wrist_1": ("wrist_1_link", 4),
    "wrist_2": ("wrist_2_link", 5),
    "wrist_3": ("wrist_3_link", 6),
}
GRIPPER_KEYS = ("gripper", "lfinger", "rfinger")  # model-independent, tool0 is DH frame 6

# Standard UR DH (a, d, alpha), mirroring ``src/robot/safety/_ur_kinematics.UR_DH_TABLES_M``. Inlined
# for the same reason DATA_DIR is anchored: the bake must not depend on the project package being
# importable from Isaac's interpreter.
UR_DH = {
    "ur3": ((0.0, 0.1519, 1.570796327), (-0.24365, 0.0, 0.0), (-0.21325, 0.0, 0.0),
            (0.0, 0.11235, 1.570796327), (0.0, 0.08535, -1.570796327), (0.0, 0.0819, 0.0)),
    "ur3e": ((0.0, 0.15185, 1.570796327), (-0.24355, 0.0, 0.0), (-0.2132, 0.0, 0.0),
             (0.0, 0.13105, 1.570796327), (0.0, 0.08535, -1.570796327), (0.0, 0.0921, 0.0)),
    "ur5": ((0.0, 0.089159, 1.570796327), (-0.425, 0.0, 0.0), (-0.39225, 0.0, 0.0),
            (0.0, 0.10915, 1.570796327), (0.0, 0.09465, -1.570796327), (0.0, 0.0823, 0.0)),
    "ur5e": ((0.0, 0.1625, 1.570796327), (-0.425, 0.0, 0.0), (-0.3922, 0.0, 0.0),
             (0.0, 0.1333, 1.570796327), (0.0, 0.0997, -1.570796327), (0.0, 0.0996, 0.0)),
    "ur10": ((0.0, 0.1273, 1.570796327), (-0.612, 0.0, 0.0), (-0.5723, 0.0, 0.0),
             (0.0, 0.163941, 1.570796327), (0.0, 0.1157, -1.570796327), (0.0, 0.0922, 0.0)),
    "ur10e": ((0.0, 0.1807, 1.570796327), (-0.6127, 0.0, 0.0), (-0.57155, 0.0, 0.0),
              (0.0, 0.17415, 1.570796327), (0.0, 0.11985, -1.570796327), (0.0, 0.11655, 0.0)),
    # ur16e has no Isaac asset here, so this baker cannot produce it. The row is still required:
    # every inlined copy of the table must be able to place every bundle this repository ships, and
    # the ur16e bundle is baked from Universal Robots' own collision STLs by
    # scripts/isaac/bake_ur_meshes_from_urdf.py.
    "ur16e": ((0.0, 0.1807, 1.570796327), (-0.4784, 0.0, 0.0), (-0.36, 0.0, 0.0),
              (0.0, 0.17415, 1.570796327), (0.0, 0.11985, -1.570796327), (0.0, 0.11655, 0.0)),
}

# Checked before Isaac boots, so a typo costs a message and not a headless startup.
if MODEL not in UR_DH:
    raise SystemExit(f"unsupported model {MODEL!r}; known: {sorted(UR_DH)}")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


#: The Isaac sample assets. ``WILLY_ISAAC_ASSETS_ROOT`` is the override ``config/robot/robot.sim.yaml``
#: documents for the same question; otherwise Isaac answers it, which is right on a normal install.
_assets_root = os.environ.get("WILLY_ISAAC_ASSETS_ROOT") or get_assets_root_path()
if not _assets_root:
    raise SystemExit("no Isaac asset root; set WILLY_ISAAC_ASSETS_ROOT to the Assets/Isaac/<version> dir")
ASSETS = str(_assets_root).rstrip("/")


def dh_T(a: float, d: float, alpha: float, theta: float = 0.0) -> np.ndarray:
    """The standard 4x4 DH transform, in metres."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def dh_frames(model: str) -> list[np.ndarray]:
    """Per-DH-frame pose at q=0, in metres. Index 0 is the base."""
    T = np.eye(4)
    out = [T.copy()]
    for a, d, al in UR_DH[model]:
        T = T @ dh_T(a, d, al, 0.0)
        out.append(T.copy())
    return out


def world_T(prim: Usd.Prim) -> np.ndarray:
    """The prim's local-to-world transform as a 4x4 array."""
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64).T


def collision_mesh(link_prim: Usd.Prim) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The link's collision mesh, points Nx3 in world metres and faces Mx3, read through instance
    proxies. Returns a pair of ``None`` where the link carries no collision mesh."""
    for sub in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies()):
        if not sub.IsA(UsdGeom.Mesh) or not sub.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = UsdGeom.Mesh(sub)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not np.all(counts == 3):
            raise SystemExit(f"non-triangular collision mesh under {link_prim.GetPath()}")
        Tw = world_T(sub)
        world = (Tw[:3, :3] @ pts.T).T + Tw[:3, 3]
        return world, idx.reshape(-1, 3)
    return None, None


stage = omni.usd.get_context().get_stage()
PRIM = f"/World/{MODEL}"
add_reference_to_stage(f"{ASSETS}/Isaac/Robots/UniversalRobots/{MODEL}/{MODEL}.usd", PRIM)
stage.Load(Sdf.Path(PRIM))

c, s = np.cos(np.radians(BASE_YAW_DEG)), np.sin(np.radians(BASE_YAW_DEG))
R_base = np.eye(4)
R_base[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
R_base_inv = np.linalg.inv(R_base)
Tdh = dh_frames(MODEL)

out: dict[str, np.ndarray] = {}
print(f"\n[bake] model={MODEL} base_yaw={BASE_YAW_DEG} deg", flush=True)
for name, (link, frame) in LINKS.items():
    lp = stage.GetPrimAtPath(f"{PRIM}/{link}")
    if not lp:
        raise SystemExit(f"missing link prim {PRIM}/{link}")
    world_m, faces = collision_mesh(lp)
    if world_m is None or faces is None:
        raise SystemExit(f"no collision mesh under {PRIM}/{link}")
    # World metres, un-yawed into the base frame, then into the DH link frame, then millimetres.
    M = np.linalg.inv(Tdh[frame]) @ R_base_inv
    local_m = (M[:3, :3] @ world_m.T).T + M[:3, 3]
    v_mm = local_m * 1000.0
    out[f"{name}__v"] = v_mm.astype(np.float64)
    out[f"{name}__f"] = faces.astype(np.int64)
    out[f"{name}__frame"] = np.array([frame], dtype=np.int64)
    print(f"  {name:10s} v={v_mm.shape} f={faces.shape} frame={frame} "
          f"bbox_mm={np.round(v_mm.max(0) - v_mm.min(0), 1).tolist()}", flush=True)

# tool0 at frame 6, the Robotiq 2F-85 geometry: model-independent, so copied from the ur5e bundle.
REFERENCE = DATA_DIR / "ur5e_collision_meshes.npz"
if not REFERENCE.is_file():
    raise SystemExit(f"missing reference bundle {REFERENCE}; it carries the 2F-85 tool0 meshes reused "
                     f"by every model")
ref = np.load(REFERENCE)
for g in GRIPPER_KEYS:
    for suf in ("v", "f", "frame"):
        out[f"{g}__{suf}"] = ref[f"{g}__{suf}"]
print(f"  + copied the 2F-85 tool0 meshes {GRIPPER_KEYS} verbatim (frame 6 is identical across the "
      f"UR e-series)", flush=True)

gate_failed = False

# --- the ur5e gate: reproduce the committed bundle -------------------------------------------------
if MODEL == "ur5e":
    print("\n[validate] diffing against the committed ur5e bundle, which is the gate:", flush=True)
    worst = 0.0
    for name in LINKS:
        a = out[f"{name}__v"]
        b = ref[f"{name}__v"]
        if a.shape != b.shape:
            print(f"  {name:10s} SHAPE MISMATCH {a.shape} vs {b.shape}", flush=True)
            worst = float("inf")
            continue
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        print(f"  {name:10s} max|dv| = {d:.6f} mm", flush=True)
    passed = worst < GATE_MM
    print(f"[validate] worst = {worst:.6f} mm against a {GATE_MM} mm gate "
          f"({'PASS' if passed else 'FAIL, do not trust this recipe'})", flush=True)
    gate_failed = not passed

# Released before any write: a re-bake of the reference writes the file this handle holds open.
ref.close()

target = DATA_DIR / f"{MODEL}_collision_meshes.npz"
if WRITE and gate_failed:
    print("\n[write] refused: the validation gate failed, so this recipe must not become the "
          "reference", flush=True)
elif WRITE:
    # The ignore is the numpy stub: `allow_pickle` is a keyword of the same call, so a mapping
    # unpacked into it cannot be typed as arrays only. Every value here is an array.
    np.savez_compressed(target, **out)  # type: ignore[arg-type]
    print(f"\n[write] {target}", flush=True)
    print("[next] a hand bundle needs no Isaac, and none is per arm: it is baked once and composed "
          "onto every arm\n"
          "  python scripts/grippers/bake_gripper_variant.py robotiq_hande --write", flush=True)
else:
    print("\n[dry-run] pass --write to save the bundle", flush=True)

# Printed before the shutdown, because close() does not come back.
print("[done]", flush=True)
app.close()
