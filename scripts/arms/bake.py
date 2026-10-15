"""Bake ``{model}_collision_meshes.npz`` from Universal Robots' own description. No Isaac, no GPU, no simulator.

    python scripts/arms/bake.py ur10e            # bake and compare, no write
    python scripts/arms/bake.py ur10 --write     # write, when the comparison agrees

What a bundle is: the exact mesh self collision guard's whole authority, every arm link's collision geometry, in
that link's own DH frame, in millimetres. A bundle from a wrong frame does not fail loudly. It guards empty space
and passes every pose, while everything upstream reports healthy.

Every input is pinned, so a bundle is a function rather than an artefact. The meshes are Universal Robots' own
collision STL files, held to one upstream commit by ``scripts/arms/ur_meshes.sha256``. The config that places
them is vendored under ``scripts/arms/ur_ros2_description`` and pinned the same way by
``scripts/arms/ur_ros2_description.sha256``. The URDF is rendered from that config by
``_ur_description.render_urdf`` rather than shipped by anyone. The suite rebuilds every committed bundle through
this code and compares it array for array, so a bundle that drifted from its own inputs fails a check rather than
a pick.

A simulator's copy of an arm is not the same geometry. Measured against UR's own description, 26 of the 30 links
read out of a composed Isaac articulation agree to under a thousandth of a millimetre, four differ by exactly the
mesh origin offsets Isaac's URDFs omit (ur3 wrist_2 and wrist_3 by 2.000 mm, ur5e and ur10e wrist_3 by 0.5 mm),
and the links of the Isaac importer's ur10 asset sit 1.8 to 65.0 mm away, its upper arm worst at 52.0 mm. Every
arm bundle this repository ships is baked here, from the vendor's own files.

The gate before every write: a bake is compared with the bundle already committed for the same model, through the
order free and mirror aware comparator in ``_bundle_compare.py``: centroid, extent, and the distance from every
vertex of one to the nearest point of the other, in both directions. A difference beyond the ceilings refuses the
write unless ``--expect-change=<reason>`` names what moved and why it is right. A mirror is the reason the
comparison is not an extent check: reflecting a link keeps every extent, every centroid and every volume, and
leaves the guard watching a left handed arm.

It lived under ``scripts/isaac/``, beside the Isaac baker it replaced, until P0f F4 moved it here with the rest of
the arm build. Nothing in it imports Isaac.
"""

from __future__ import annotations

import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "src" / "robot" / "safety" / "data"

# UR's own description, vendored and pinned, and the meshes it names. Neither Isaac nor a GPU is on this path:
# the URDF is rendered from UR's config (_ur_description.py) and the meshes are the ones fetch.py pins to one
# upstream commit, under whichever asset root that module resolves. Both sit beside this script.
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _arm_file  # noqa: E402  (beside this script)
import _bundle_compare  # noqa: E402  (beside this script)
import _ur_description  # noqa: E402  (beside this script)
import fetch  # noqa: E402  (beside this script)

GRIPPER_KEYS = ("gripper", "lfinger", "rfinger")

# The arm's links, the half turn to where its DH chain starts, and its DH rows are the arm description's
# (config/arms/<model>.yaml), read through _arm_file beside this script since P0f F7. They were three
# inlined copies here until then, held to the description by a test.


def asset_root() -> pathlib.Path:
    """Where the pinned meshes live: whatever ``fetch`` resolves, so both read one root."""
    return fetch._default_dest()  # noqa: SLF001 (one resolver, deliberately shared)


def sources(model: str) -> dict:
    """The rendered URDF for ``model`` and the collision mesh each bundle link is baked from.

    Derived, not listed: the URDF names its own collision meshes, so a model is baked from what UR says it is
    made of. Anything that is not an ``.stl`` under ``meshes/<owner>/collision/`` is refused rather than baked,
    because the one thing this script must never do is write a bundle from geometry nobody checked.
    """
    urdf_text, names = _collision_meshes(model)
    unpinned = sorted(set(names.values()) - set(fetch.read_manifest(fetch.MANIFEST)))
    if unpinned:
        raise SystemExit(
            f"{model}: {fetch.MANIFEST.name} pins none of {', '.join(unpinned)}, so there is no checked geometry to "
            f"bake it from"
        )
    root = asset_root()
    return {
        "urdf_text": urdf_text,
        "meshes": {key: root / name for key, name in names.items()},
        "note": f"UR's own collision stl files, through the URDF rendered from its pinned description ({root.name})",
    }


def _collision_meshes(model: str) -> "tuple[str, dict[str, str]]":
    """The rendered URDF for ``model`` and, per bundle part, the collision mesh UR's description names for its link."""
    urdf_text = _ur_description.render_urdf(model)
    _, _, collision = read_urdf_text(urdf_text)
    names: dict[str, str] = {}
    for key, link, _frame in _arm_file.collision_links(model):
        name = collision.get(link)
        if not name:
            raise SystemExit(f"{model}/{link} declares no collision mesh in UR's description")
        parts = name.split(chr(47))
        if len(parts) != 4 or parts[0] != "meshes" or parts[2] != "collision" or not parts[3].endswith(".stl"):
            raise SystemExit(
                f"{model}/{link} names {name!r}, which is not a collision stl under meshes/<owner>/collision/: "
                "a bundle is only worth its file if the geometry in it is the vendor's own"
            )
        names[key] = name
    return urdf_text, names


def bakeable() -> list[str]:
    """The described arms whose every collision mesh the mesh manifest pins: the arms a bundle can be baked for."""
    pinned = set(fetch.read_manifest(fetch.MANIFEST))
    return [model for model in _arm_file.described() if set(_collision_meshes(model)[1].values()) <= pinned]


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

    Isaac's importer ``ur10.urdf`` carries ``xyz="0 0.0 0.0027000046)"``, with a stray closing parenthesis.
    That single character is why cuRobo's parser refuses the file. It is repaired on read here and in the
    descriptor copy, never in Isaac's own tree.
    """
    return read_urdf_text(path.read_text(encoding="utf-8"), name=path.name)[:2]


def read_urdf_text(text: str, *, name: str = "<rendered>"):
    """(joints, per-link mesh origin, per-link collision mesh filename) from URDF text."""
    fixed = re.sub(r'(xyz|rpy)="([^"]*)"',
                   lambda m: f'{m.group(1)}="{m.group(2).replace(chr(41), "")}"', text)
    if fixed != text:
        print(f"  [repair] {name} carries a malformed number; repaired on read", flush=True)
    root = ET.fromstring(fixed)
    joints = {}
    for jt in root.findall("joint"):
        parent, child = jt.find("parent"), jt.find("child")
        if parent is None or child is None:
            raise ValueError(f"{name}: joint {jt.attrib.get('name')!r} names no parent or no child link")
        joints[jt.attrib["name"]] = {
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "M": _origin(jt.find("origin")),
        }
    visual = {}
    collision_mesh = {}
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
        mesh = col.find("geometry/mesh") if col is not None else None
        if mesh is not None and mesh.attrib.get("filename"):
            collision_mesh[ln.attrib["name"]] = mesh.attrib["filename"]
    return joints, visual, collision_mesh


def link_frame(joints: dict, link: str) -> np.ndarray:
    for j in joints.values():
        if j["child"] == link:
            return link_frame(joints, j["parent"]) @ j["M"]
    return np.eye(4)


def dh_frames(model: str) -> list:
    T = np.eye(4)
    out = [T.copy()]
    for a, d, al in _arm_file.dh_rows(model):
        ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[1.0, 0.0, 0.0, a], [0.0, ca, -sa, 0.0],
                          [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]])
        out.append(T.copy())
    return out


def to_dh_frame(vertices_m, faces, model: str, key: str, urdf_text: str):
    """One mesh, from its own file frame into the link's DH frame, in millimetres. Pure arithmetic.

    The one place the chain lives, so the round trip against the descriptor builder's placement can be checked
    without a mesh file, Isaac or a GPU: mesh frame, link frame, base, un-yaw, DH link frame.

    Two things are refused rather than written. A DH table that has drifted from UR's own description places
    every vertex of every link, so it is compared before a single mesh is touched. And a transform whose
    determinant is not positive is a mirror: it keeps every distance, every bounding box and every extent
    check, so nothing downstream could see it, and the bundle would guard a left handed arm.
    """
    rows = _ur_description.dh_rows(model)
    described = _arm_file.dh_rows(model)
    assert np.allclose(np.asarray(described, dtype=np.float64), np.asarray(rows, dtype=np.float64),
                       rtol=0.0, atol=1e-12), (
        f"the arm description's DH rows disagree with UR's own description for {model}: {described} against {rows}"
    )
    link, frame = {part: (link, frame) for part, link, frame in _arm_file.collision_links(model)}[key]
    joints, mesh_origin, _ = read_urdf_text(urdf_text)
    rz = np.eye(4)
    # Rodrigues' form, as the bake always wrote it: the committed arrays were cut through it, last bits included.
    rz[:3, :3] = _rpy(0.0, 0.0, np.radians(float(_arm_file.chain_yaw_deg(model))))
    X = (np.linalg.inv(dh_frames(model)[frame]) @ np.linalg.inv(rz)
         @ link_frame(joints, link) @ mesh_origin[link])
    if float(np.linalg.det(X[:3, :3])) <= 0.0:
        raise ValueError(
            f"the chain for {model}/{key} composes to a reflection (det {np.linalg.det(X[:3, :3]):.6f}), "
            f"so this bundle would hold a mirrored link: every distance in it would look right"
        )
    v = np.asarray(vertices_m, dtype=np.float64)
    v_mm = ((X[:3, :3] @ v.T).T + X[:3, 3]) * 1000.0
    return v_mm, np.asarray(faces, dtype=np.int64), frame


def bake(model: str) -> dict:
    """Every link of one arm, in its own DH link frame, millimetres."""
    import trimesh

    src = sources(model)
    out: dict = {}
    print(f"\n[bake] model={model} source={src['note']}", flush=True)
    for key, _link, _frame in _arm_file.collision_links(model):
        path = src["meshes"][key]
        if not path.is_file():
            raise SystemExit(
                f"missing mesh for {model}/{key}: {path}. Fetch the pinned meshes first: "
                f"python scripts/arms/fetch.py"
            )
        mesh = trimesh.load(str(path), process=False, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise SystemExit(f"{path} loads as a {type(mesh).__name__}, not a triangle mesh: nothing to bake "
                             f"for {model}/{key}")

        v, f, frame = to_dh_frame(np.asarray(mesh.vertices, dtype=np.float64), mesh.faces, model, key,
                                  src["urdf_text"])
        out[f"{key}__v"] = v
        out[f"{key}__f"] = f
        out[f"{key}__frame"] = np.array([frame], dtype=np.int64)
        print(f"  {key:10s} v={v.shape} f={f.shape} frame={frame} "
              f"bbox_mm={np.round(v.max(0) - v.min(0), 1).tolist()}", flush=True)
    # float64, which is what the arithmetic above produces and what five of the six committed bundles already
    # held. The sixth was float32 from an older run, where one ulp at 400 mm is about 3e-5 mm. Nothing is cast
    # on the way out, so the arrays a bundle holds are the numbers that were computed.
    return out


def _same_arrays(path: pathlib.Path, arrays: dict) -> bool:
    """Whether ``path`` holds exactly ``arrays``: the same keys, and each the same dtype, shape and bytes."""
    with np.load(path, allow_pickle=False) as held:
        if set(held.files) != set(arrays):
            return False
        for key in held.files:
            mine, theirs = np.asarray(arrays[key]), held[key]
            if mine.dtype != theirs.dtype or mine.shape != theirs.shape:
                return False
            if np.ascontiguousarray(mine).tobytes() != np.ascontiguousarray(theirs).tobytes():
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    model = next((a for a in args if not a.startswith("-")), "ur10e").lower()
    write = "--write" in args
    expect = next((a.split("=", 1)[1] for a in args if a.startswith("--expect-change=")), "")
    if model not in bakeable():
        why = "no arm description" if model not in _arm_file.described() else "no pinned collision mesh"
        raise SystemExit(f"{model!r} cannot be baked: {why}. Bakeable: {', '.join(bakeable())}")

    out = bake(model)

    # ---- the gate: this has to be the arm already committed, or the change has to be stated ---------
    # Every model is compared with its own committed bundle, through the order free, mirror aware
    # comparator, because a bundle from a wrong frame does not fail: it guards empty space.
    committed = DATA / f"{model}_collision_meshes.npz"
    if committed.is_file():
        with np.load(committed, allow_pickle=True) as _held:
            held = {name: np.array(_held[name]) for name in _held.files}
        difference = _bundle_compare.compare(
            held, out, links=tuple(part for part, _link, _frame in _arm_file.collision_links(model)))
        print(f"\n[gate] against the committed {committed.name}:", flush=True)
        print(difference.render(), flush=True)
        if difference.agrees(_bundle_compare.CEILINGS):
            print("[gate] the same arm", flush=True)
        elif expect:
            print(f"[gate] DIFFERENT, and the caller says why: {expect}", flush=True)
        else:
            raise SystemExit(
                "this bake is not the arm that is committed, and nothing said it would be. Rerun with "
                "--expect-change=<the reason> once you can name what moved and why it is right."
            )
    else:
        print(f"\n[gate] no committed {committed.name} to compare with: this is a new arm", flush=True)

    # The 2F-85 tool0 arrays are model-independent (DH frame 6 is the flange on every UR) and are copied
    # verbatim, as the retired Isaac baker did. Read before anything is written, so re-baking ur5e itself
    # copies the committed arrays rather than the ones this run is about to produce.
    with np.load(DATA / "ur5e_collision_meshes.npz", allow_pickle=True) as src_gripper:
        for g in GRIPPER_KEYS:
            for suf in ("v", "f", "frame"):
                out[f"{g}__{suf}"] = np.array(src_gripper[f"{g}__{suf}"])
    print(f"  + copied the 2F-85 tool0 meshes {GRIPPER_KEYS} verbatim from ur5e", flush=True)

    if not write:
        print(f"\n[dry-run] pass --write to save {committed.name}", flush=True)
        return 0
    # The same arrays are not written again (P0f F3). The retract table hashes this file, so a rewrite of equal arrays
    # still invalidates it wherever the bytes move: another key order, another memory layout, another zlib. Measured
    # 2026-09-23, today's order and zlib happen to write the committed bytes back; nothing makes that hold.
    if committed.is_file() and _same_arrays(committed, out):
        print(f"\n[write] {committed.name} holds these arrays already and is left as it is, so the file the retract "
              f"table hashes does not move", flush=True)
        return 0
    np.savez_compressed(committed, **out)
    print(f"\n[write] {committed}", flush=True)
    print(f"[write] provenance: {sources(model)['note']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
