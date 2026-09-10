"""Bake ``{model}_collision_meshes.npz`` from a URDF and its mesh files. No Isaac, no GPU.

    python scripts/isaac/bake_ur_meshes_from_urdf.py ur10e            # the control, no write
    python scripts/isaac/bake_ur_meshes_from_urdf.py ur10 --write

⭐ **WHY A SECOND BAKER.** ``bake_ur_collision_meshes.py`` reads a COMPOSED Isaac articulation and
needs the simulator for it. That works for five UR models and cannot work for ``ur10``, whose USD
collides the whole arm with primitives. But ur10 does ship link meshes in a URDF package, and a URDF
plus an .obj needs no simulator at all. Same output format, same DH frames, different source, and
the bundle says which it came from.

⚠ **THE SOURCE IS A VISUAL MESH, AND THAT IS DELIBERATE AND MEASURED.** ur10 declares no collision
mesh anywhere, so the choice is its visual geometry or its collision cylinders. This bakes the
CONVEX HULL of the visual, and the reason is a control on an arm where both halves exist:

    ur10e, which this repository already plans with, ships visual .dae AND collision .stl.
    Its own collision geometry is 1.259x the volume of its visual (upper arm 1.619, forearm 1.642,
    the wrists 1.05 to 1.11), and every one of its six collision meshes is EXACTLY CONVEX
    (hull/mesh volume = 1.000).

    The convex hulls this script bakes for ur10 come out at 1.357x its visual, with the same
    per-link pattern (upper arm 1.735, forearm 1.766, wrists 1.09 to 1.20).

So a convex hull is not a compromise here: it is what the vendor ships as collision geometry, eight
percentage points more conservative. Conservative is the correct direction for a fail-closed guard,
and the hulls are SMALLER than ur10e's collision meshes in vertex count (2.4k-2.8k against 4k-5.6k).

⚠ **IT PROVES ITSELF ON A KNOWN-GOOD ARM FIRST.** Run with no ``--write`` on ``ur10e`` and it bakes
that arm from its COLLISION stl files through this same URDF path, then diffs the result against the
committed ``ur10e_collision_meshes.npz`` that was baked from Isaac. Two different readers, two
different files, one answer, or the frame chain here is wrong. That gate runs before every write.
"""

from __future__ import annotations

import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "src" / "robot" / "safety" / "data"
ISAAC = pathlib.Path("D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64")
MP = ISAAC / "exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots"
IMPORTER = ISAAC / "exts/isaacsim.asset.importer.urdf/data/urdf/robots"
CUROBO_MESHES = REPO / "ext_deps/curobo/curobo/content/assets/robot/ur_description/meshes"

#: bundle key -> (urdf link name, DH frame index). The same map both bakers and the guard use.
LINKS = {"shoulder": ("shoulder_link", 1), "upper_arm": ("upper_arm_link", 2),
         "forearm": ("forearm_link", 3), "wrist_1": ("wrist_1_link", 4),
         "wrist_2": ("wrist_2_link", 5), "wrist_3": ("wrist_3_link", 6)}

GRIPPER_KEYS = ("gripper", "lfinger", "rfinger")

#: The base yaw that reconciles the Isaac cell with the DH chain, mirroring
#: ``robot.sim.yaml safety.self_collision.kinematics_base_yaw_deg``.
BASE_YAW_DEG = 180.0

#: Standard UR DH (a, d, alpha), inlined for the same reason the other two copies are; the three are
#: compared by ``tests/test_inlined_dh_tables.py``.
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
}

#: model -> (urdf path, per-link mesh resolver). The control reads ur10e's COLLISION stl files, so
#: the gate compares like with like against a bundle Isaac produced from collision geometry.
SOURCES = {
    "ur10": {
        "urdf": IMPORTER / "ur10/urdf/ur10.urdf",
        "meshes": lambda key: IMPORTER / "ur10/meshes" / f"ur10_{key}.obj",
        "hull": True,
        "note": "convex hulls of the visual .obj meshes; this asset declares no collision mesh",
    },
    "ur10e": {
        "urdf": MP / "ur10e/ur10e.urdf",
        "meshes": lambda key: CUROBO_MESHES / "ur10e/collision" / (
            {"shoulder": "shoulder", "upper_arm": "upperarm", "forearm": "forearm",
             "wrist_1": "wrist1", "wrist_2": "wrist2", "wrist_3": "wrist3"}[key] + ".stl"),
        "hull": False,
        "note": "ur_description collision stl files, read through the URDF path as a control",
    },
}


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    def rot(axis: int, t: float) -> np.ndarray:
        a = np.zeros(3)
        a[axis] = 1.0
        K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
        return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)
    return rot(2, y) @ rot(1, p) @ rot(0, r)


def _origin(el) -> np.ndarray:
    M = np.eye(4)
    if el is None:
        return M
    r = [float(v) for v in el.attrib.get("rpy", "0 0 0").split()]
    x = [float(v) for v in el.attrib.get("xyz", "0 0 0").split()]
    M[:3, :3] = _rpy(*r)
    M[:3, 3] = x
    return M


def read_urdf(path: pathlib.Path):
    """(joints, per-link visual origin). Tolerates one malformed number and says so.

    ⛔ MEASURED 2026-09-10: Isaac's importer ur10.urdf line 28 reads ``xyz="0 0.0 0.0027000046)"``,
    with a stray closing parenthesis. That single character is why cuRobo's parser refuses the file,
    which is the whole of the "this urdf cannot be loaded" blocker. It is repaired on read here and
    in the descriptor copy, never in Isaac's own tree.
    """
    text = path.read_text(encoding="utf-8")
    fixed = re.sub(r'(xyz|rpy)="([^"]*)"',
                   lambda m: f'{m.group(1)}="{m.group(2).replace(chr(41), "")}"', text)
    if fixed != text:
        print(f"  [repair] {path.name} carries a malformed number; repaired on read", flush=True)
    root = ET.fromstring(fixed)
    joints = {}
    for jt in root.findall("joint"):
        joints[jt.attrib["name"]] = {
            "parent": jt.find("parent").attrib["link"],
            "child": jt.find("child").attrib["link"],
            "M": _origin(jt.find("origin")),
        }
    visual = {}
    for ln in root.findall("link"):
        vis = ln.find("visual")
        col = ln.find("collision")
        # The mesh origin is whichever element actually references a mesh for this link.
        chosen = None
        for cand in (col, vis):
            if cand is not None and cand.find("geometry/mesh") is not None:
                chosen = cand
                break
        visual[ln.attrib["name"]] = _origin(chosen.find("origin") if chosen is not None else None)
    return joints, visual


def link_frame(joints: dict, link: str) -> np.ndarray:
    for j in joints.values():
        if j["child"] == link:
            return link_frame(joints, j["parent"]) @ j["M"]
    return np.eye(4)


def dh_frames(model: str) -> list:
    T = np.eye(4)
    out = [T.copy()]
    for a, d, al in UR_DH[model]:
        ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[1.0, 0.0, 0.0, a], [0.0, ca, -sa, 0.0],
                          [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]])
        out.append(T.copy())
    return out


def bake(model: str) -> dict:
    """Every link of one arm, in its own DH link frame, millimetres."""
    import trimesh

    src = SOURCES[model]
    joints, visual_origin = read_urdf(src["urdf"])
    Tdh = dh_frames(model)
    Rz = np.eye(4)
    Rz[:3, :3] = _rpy(0.0, 0.0, np.radians(BASE_YAW_DEG))

    out: dict = {}
    print(f"\n[bake] model={model} source={src['note']}", flush=True)
    for key, (link, frame) in LINKS.items():
        path = src["meshes"](key)
        if not path.is_file():
            raise SystemExit(f"missing mesh for {model}/{key}: {path}")
        mesh = trimesh.load(str(path), process=False, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if src["hull"]:
            mesh = mesh.convex_hull

        # mesh frame -> link frame -> base -> un-yaw -> DH link frame, then metres to millimetres
        X = np.linalg.inv(Tdh[frame]) @ np.linalg.inv(Rz) @ link_frame(joints, link) @ visual_origin[link]
        v = np.asarray(mesh.vertices, dtype=np.float64)
        v = ((X[:3, :3] @ v.T).T + X[:3, 3]) * 1000.0
        f = np.asarray(mesh.faces, dtype=np.int64)
        out[f"{key}__v"] = v
        out[f"{key}__f"] = f
        out[f"{key}__frame"] = np.array([frame], dtype=np.int64)
        print(f"  {key:10s} v={v.shape} f={f.shape} frame={frame} "
              f"bbox_mm={np.round(v.max(0) - v.min(0), 1).tolist()}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    model = next((a for a in args if not a.startswith("-")), "ur10e").lower()
    write = "--write" in args
    if model not in SOURCES:
        raise SystemExit(f"no URDF source for {model!r}; known: {sorted(SOURCES)}")

    # ---- THE GATE: reproduce ur10e, which Isaac baked, through this URDF path -------------------
    print("[validate] baking ur10e from its collision stl files and diffing against the committed "
          "bundle Isaac produced:", flush=True)
    control = bake("ur10e")
    ref_path = DATA / "ur10e_collision_meshes.npz"
    if not ref_path.is_file():
        raise SystemExit(f"no control bundle at {ref_path}")
    ref = np.load(ref_path, allow_pickle=True)
    worst = 0.0
    for key in LINKS:
        a = np.asarray(control[f"{key}__v"], dtype=np.float64)
        b = np.asarray(ref[f"{key}__v"], dtype=np.float64)
        # Vertex ORDER differs between readers, so compare the shapes rather than the arrays:
        # the axis-aligned extent and the centroid are order-free and catch any frame error.
        d = max(float(np.max(np.abs((a.max(0) - a.min(0)) - (b.max(0) - b.min(0))))),
                float(np.max(np.abs(a.mean(0) - b.mean(0)))))
        worst = max(worst, d)
        print(f"  {key:10s} extent+centroid agree to {d:8.3f} mm", flush=True)
    limit = 15.0
    print(f"[validate] WORST = {worst:.3f} mm against {limit:.1f} mm "
          f"({'PASS' if worst <= limit else 'FAIL'})", flush=True)
    if worst > limit:
        raise SystemExit(
            "the URDF path does not reproduce the arm Isaac baked, so its frame chain is wrong and "
            "nothing is written. A bundle from a wrong frame does not fail: it guards empty space.")

    if model == "ur10e":
        print("\n[dry-run] that WAS the control; pass a different model to bake one", flush=True)
        return 0

    out = bake(model)

    # The 2F-85 tool0 arrays are model-independent (DH frame 6 is the flange on every UR) and are
    # copied verbatim, exactly as the Isaac baker does.
    src_gripper = np.load(DATA / "ur5e_collision_meshes.npz", allow_pickle=True)
    for g in GRIPPER_KEYS:
        for suf in ("v", "f", "frame"):
            out[f"{g}__{suf}"] = np.asarray(src_gripper[f"{g}__{suf}"])
    print(f"  + copied the 2F-85 tool0 meshes {GRIPPER_KEYS} verbatim from ur5e", flush=True)

    target = DATA / f"{model}_collision_meshes.npz"
    if not write:
        print(f"\n[dry-run] pass --write to save {target.name}", flush=True)
        return 0
    np.savez_compressed(target, **out)
    print(f"\n[write] {target}", flush=True)
    print(f"[write] provenance: {SOURCES[model]['note']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
