"""Assemble a cuRobo robot config for one bundled UR model from the ingredients already on the box.

Both drivers ask the planner for ``{model}.yml``, the name
``src/robot/drivers/sim/robot_models.curobo_robot_yml`` derives from the configured ``robot_model``.
That file is not shipped with the repository: it is assembled here, once per box and per model, and
written into the gitignored cuRobo content directory. A fresh cuRobo clone always needs this step.

Ingredients, all of them already installed on a cell that can plan:

  * kinematics: Isaac's canonical ``{model}.urdf``, the same description the sim's Lula solver uses,
    copied into the cuRobo content with the ``package://ur_description/`` prefix stripped so the mesh
    references resolve under the cuRobo asset root.
  * collision spheres and ``default_q``: Isaac's model-tuned Lula ``{model}_robot_description.yaml``,
    converted into the cuRobo sphere-map format.
  * tool0 gripper spheres: read from the committed sphere map for the gripper this cell runs,
    ``src/robot/safety/planning/robot/{gripper}_gripper_spheres.yml``, which is produced by
    ``build_gripper_spheres.py`` in the project venv from a baked bundle or from a vendor mesh.
    Independent of the arm model, because the bundle's frame 6 is cuRobo's ``tool0`` on every UR
    flange; not independent of the gripper, which is what ``--gripper`` selects. A cell running
    something other than the 2F-85 and planning with the 2F-85 map is planning against a different
    robot.
  * everything else: cuRobo's own ``ur10e.yml``, which carries the joint names, link names and
    end-effector link that every UR e-series model shares.

The arm-link surface augmentation is ur5e only. For ``ur5e`` each Lula arm link gains cuRobo
``SphereFitType.SURFACE`` spheres fitted to ``ur5e_collision_meshes.npz`` and placed through the UR5e
DH chain. Those meshes and that chain are ur5e-specific: a UR3e has different link lengths, so
applying them to another model would author collision geometry in the wrong place. Every other model
keeps its own Lula arm spheres, which are model-tuned and correct for its link lengths, and the run
says so.

What a re-run reproduces: the URDF half is byte-identical, the YAML half is not. cuRobo's
``sphere_fit`` samples the mesh surface and is not deterministic, so two runs of the same recipe
differ in the sphere centres and by a sphere or two on one link. Where a measured planner result
depends on a particular ``{model}.yml``, keep that file rather than expecting a rebuild to reproduce
it. ``src/robot/safety/planning/robot/PROVENANCE.md`` records that comparison.

This is a command line and not a library capability, and the reason is the interpreter. It reads
``curobo.content`` and ``curobo.sphere_fit``, which exist only inside ``ext_deps/curobo_env``, and
that environment is Python 3.10, while every module under ``src`` uses ``enum.StrEnum`` and needs
3.11. A twin in ``src`` would be un-importable at the one place this work can run, so the DH table
below is inlined on purpose rather than imported.

``src/robot/safety/planning/robot/build_gripper_spheres.py`` is the Isaac-free twin of the tool0 half
below: the same mesh bundle, the same grid fit and the same four constants, run under the project
environment so the repository carries a labelled copy of the gripper spheres. Changing the fit here
means changing it there, or the committed sphere map stops matching the on-box one.

Usage, on the box that owns the cell::

    python scripts/curobo/build_ur_config.py ur3e     # writes ur3e.urdf and ur3e.yml
    python scripts/curobo/build_ur_config.py ur5e
    python scripts/curobo/build_ur_config.py ur5e --gripper schunk_egu50    # the same arm, another hand     # the ur5e recipe, including arm augmentation

Run it with the cuRobo environment's interpreter. cuRobo then answers where its own content directory
is and nothing has to be guessed::

    ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur5e

Paths are auto-detected and overridable, so no machine-specific path is baked in:

  * ``WILLY_ISAAC_ROOT``       the Isaac Sim install directory
  * ``WILLY_ISAAC_MP_CONFIGS`` the Isaac ``motion_policy_configs/universal_robots`` directory
  * ``WILLY_CUROBO_CONTENT``   the cuRobo ``content`` directory, holding ``configs/robot`` and ``assets/robot``

These three are build-time only and are read here and nowhere else. The variables the planner reads at
runtime are the ``WILLY_CUROBO_*`` set declared in ``src/robot/safety/planning/environment.py``.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]  # scripts/curobo/this.py, then scripts, then the repository root

# --- 0) model + path resolution -------------------------------------------------------------------------
MODEL = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WILLY_UR_MODEL", "ur3e")).lower()
# Which hand is on the flange. The arm model does not decide it: the same UR5e takes a Robotiq 2F-85
# or a Schunk EGU-50, and the planner has to model the one that is actually there. Mirrors the safety
# guard's `collision_mesh_variant`, which selects the same bundle for the exact-mesh check.
GRIPPER = "ur5e"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gripper" and _i + 1 < len(sys.argv):
        GRIPPER = sys.argv[_i + 1]

#: Relative to an Isaac install root: where the per-model Lula descriptions live. Stable across Isaac
#: versions; the *install root* is what differs from box to box, so only that is searched for.
_ISAAC_MP_SUFFIX = (
    "exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots"
)

#: Isaac Sim is a multi-GB standalone installer that stays where NVIDIA puts it, see ext_deps/README.md,
#: so it is the one dependency this repo cannot relocate. These are the conventional install roots on
#: each platform; ``WILLY_ISAAC_ROOT`` and ``WILLY_ISAAC_MP_CONFIGS`` override them on any box.
_ISAAC_ROOT_HINTS = (
    "C:/isaacsim", "D:/isaacsim", "~/isaacsim", "~/.local/share/ov/pkg", "/opt/isaacsim",
)


def _isaac_roots() -> list[Path]:
    """Candidate Isaac install roots, the environment override first. Machine paths live only here."""
    roots: list[Path] = []
    override = os.environ.get("WILLY_ISAAC_ROOT")
    if override:
        roots.append(Path(override).expanduser())
    roots += [Path(hint).expanduser() for hint in _ISAAC_ROOT_HINTS]
    return [root for root in roots if root.is_dir()]


def _resolve_isaac_mp() -> Path:
    """The Isaac ``motion_policy_configs/universal_robots`` dir that has a description for ``MODEL``."""
    explicit = os.environ.get("WILLY_ISAAC_MP_CONFIGS")
    if explicit and (Path(explicit) / MODEL).is_dir():
        return Path(explicit)
    for root in _isaac_roots():
        direct = root / _ISAAC_MP_SUFFIX
        if (direct / MODEL).is_dir():
            return direct
        # Isaac unpacks into a versioned subdirectory whose name we should not have to know.
        for hit in root.glob(f"*/{_ISAAC_MP_SUFFIX}"):
            if (hit / MODEL).is_dir():
                return hit
        for hit in root.glob("**/motion_policy_configs/universal_robots"):
            if (hit / MODEL).is_dir():
                return hit
    raise SystemExit(
        f"could not locate Isaac motion_policy_configs/universal_robots/{MODEL}.\n"
        f"  searched: {', '.join(str(r) for r in _isaac_roots()) or '(no Isaac root found)'}\n"
        f"  set WILLY_ISAAC_ROOT=<isaac install dir>, or WILLY_ISAAC_MP_CONFIGS=<.../universal_robots>."
    )


def _resolve_curobo_content() -> Path:
    """The cuRobo ``content`` dir, the one holding ``configs/robot`` with a ur10e.yml template.

    The order is: the explicit environment override, then cuRobo's own authority
    (``curobo.content.get_content_root()``, correct on any box and importable when this runs under the
    cuRobo environment), then the in-repo install. Run this with the cuRobo environment's interpreter
    and the second entry answers it exactly, on every machine.
    """
    env = os.environ.get("WILLY_CUROBO_CONTENT")
    cands = [Path(env)] if env else []
    try:  # cuRobo's own answer: authoritative and machine-independent, only importable in its env
        from curobo.content import get_content_root  # type: ignore[import-not-found]

        cands.append(Path(str(get_content_root())))
    except Exception:  # noqa: BLE001 (not the cuRobo env; fall through to the clone layout)
        pass
    cands += [
        REPO / "ext_deps/curobo/curobo/content",
        REPO / "ext_deps/curobo/src/curobo/content",
    ]
    ext_deps = REPO / "ext_deps"
    if ext_deps.is_dir():
        cands += list(ext_deps.glob("**/curobo/content"))
    for c in cands:
        if c and (c / "configs/robot").is_dir():
            return c
    raise SystemExit(
        "could not locate the cuRobo content dir (needs configs/robot with the ur10e.yml template). "
        "Set WILLY_CUROBO_CONTENT=<...>/curobo/content. Searched: "
        + ", ".join(str(c) for c in cands if c)
    )


ISAAC_MODEL_DIR = _resolve_isaac_mp() / MODEL
CUROBO = _resolve_curobo_content()
URDF_OUT = CUROBO / f"assets/robot/ur_description/{MODEL}.urdf"
YML_OUT = CUROBO / f"configs/robot/{MODEL}.yml"
print(f"[paths] model={MODEL}\n  isaac  = {ISAAC_MODEL_DIR}\n  curobo = {CUROBO}")

# --- 1) URDF: Isaac's {model}.urdf with cuRobo-relative mesh paths ---------------------------------------
_urdf_src = ISAAC_MODEL_DIR / f"{MODEL}.urdf"
if not _urdf_src.is_file():
    raise SystemExit(f"missing Isaac URDF: {_urdf_src}")
urdf = _urdf_src.read_text(encoding="utf-8")
urdf = urdf.replace("package://ur_description/", "")  # meshes/<model>/... relative to asset_root_path
URDF_OUT.parent.mkdir(parents=True, exist_ok=True)
URDF_OUT.write_text(urdf, encoding="utf-8")
# Verify the referenced meshes actually exist under the cuRobo asset root. A missing mesh dir makes
# cuRobo fail to build the kinematics, and the planner server then never reports ready.
_missing: list[str] = []
for _tok in {t for t in urdf.split('"') if t.endswith((".dae", ".stl", ".obj", ".DAE", ".STL"))}:
    if not (URDF_OUT.parent / _tok).is_file() and not (CUROBO / "assets/robot/ur_description" / _tok).is_file():
        _missing.append(_tok)
print(f"wrote {URDF_OUT}")
if _missing:
    print(f"  !! {len(_missing)} referenced mesh file(s) not found under the cuRobo asset root, e.g. "
          f"{sorted(_missing)[:3]}. Copy the {MODEL} meshes into "
          f"{CUROBO / 'assets/robot/ur_description'} (see the Isaac ur_description meshes) before planning.")
else:
    print("  mesh references resolve OK")

# --- 2) Lula {model} spheres into the cuRobo map format, plus default_q ---------------------------------------------
_lula_src = ISAAC_MODEL_DIR / f"rmpflow/{MODEL}_robot_description.yaml"
if not _lula_src.is_file():
    raise SystemExit(f"missing Isaac Lula robot description: {_lula_src}")
lula = yaml.safe_load(_lula_src.read_text(encoding="utf-8"))
default_q = list(lula["default_q"])
sphere_map: dict[str, list[dict[str, object]]] = {}
for entry in lula["collision_spheres"]:  # Lula = list of single-key {link: [ {center,radius}, ... ]}
    for link, spheres in entry.items():
        sphere_map[link] = [{"center": [float(c) for c in s["center"]], "radius": float(s["radius"])}
                            for s in spheres]

# --- 2a) tool0 gripper spheres: the committed map for the gripper this cell runs --------------------------
# Read rather than fitted here. The fit lives in one place,
# `src/robot/safety/planning/robot/gripper_spheres.py`, and produces a committed file per gripper; a
# test compares that file against the fit so the two cannot drift. This script used to carry its own
# copy of the same arithmetic and always fitted the Robotiq 2F-85, under a comment calling it model
# independent. It is independent of the arm and not of the hand: with a Schunk EGU-50 bolted on, the
# guard read that bundle through `collision_mesh_variant` and the planner still modelled a Robotiq.
_SPHERE_MAP = REPO / f"src/robot/safety/planning/robot/{GRIPPER}_gripper_spheres.yml"
if not _SPHERE_MAP.is_file():
    raise SystemExit(
        f"no committed sphere map at {_SPHERE_MAP}. Write one in the project venv first:\n"
        f"  .venv/Scripts/python.exe -m src.robot.safety.planning.robot.build_gripper_spheres "
        f"--variant {GRIPPER}\n"
        "or, for a gripper with no baked bundle, fit it from the vendor mesh with --mesh."
    )
_gripper_cfg = yaml.safe_load(_SPHERE_MAP.read_text(encoding="utf-8"))
sphere_map["tool0"] = _gripper_cfg["collision_spheres"]["tool0"]
print(f"tool0 spheres: {len(sphere_map['tool0'])} from {_SPHERE_MAP.name} "
      f"({_gripper_cfg.get('_provenance', {}).get('gripper', GRIPPER)})")

# --- 2b) ARM-link surface augmentation: ur5e only ---------------------------------------------------------
# The bundle's arm meshes and the DH chain used to place them are ur5e-specific. Applying them to
# another model would author arm geometry in the wrong place, so every other model keeps its own
# model-tuned Lula arm spheres.
if MODEL == "ur5e":
    try:
        # The arm bundle, which is a different thing from the gripper map read above: these
        # are the ur5e's own link meshes, and this branch is ur5e only for exactly that reason.
        _ARM_NPZ = REPO / "src/robot/safety/data/ur5e_collision_meshes.npz"
        _MESH = np.load(_ARM_NPZ, allow_pickle=True)
        import trimesh  # type: ignore[import-not-found]
        from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh  # type: ignore[import-not-found]

        # UR5e DH (a, d, alpha), inlined rather than imported: this branch only runs in the cuRobo
        # sidecar environment, where the repository package cannot be imported. The values match
        # ``src/robot/safety/_ur_kinematics`` ur5e row for row, and a change to one is a change to both.
        _DH: list[tuple[float, float, float]] = [
            (0.0, 0.1625, 1.570796327), (-0.425, 0.0, 0.0), (-0.3922, 0.0, 0.0),
            (0.0, 0.1333, 1.570796327), (0.0, 0.0997, -1.570796327), (0.0, 0.0996, 0.0),
        ]

        def _dh_T(row: tuple[float, float, float]) -> np.ndarray:
            a, d, al = row
            ca, sa = np.cos(al), np.sin(al)
            return np.array([[1.0, 0.0, 0.0, a], [0.0, ca, -sa, 0.0], [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]])

        def _dh_frames_m() -> list[np.ndarray]:
            T = np.eye(4)
            out = [T.copy()]
            for row in _DH:
                T = T @ _dh_T(row)
                out.append(T.copy())
            return out

        def _aa(axis: tuple[float, float, float], th: float) -> np.ndarray:
            a = np.asarray(axis, float)
            a = a / (np.linalg.norm(a) or 1.0)
            c, s = np.cos(th), np.sin(th)
            x, y, z = a
            return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                             [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                             [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])

        def _urdf_link_frame() -> Callable[[str], np.ndarray]:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(urdf)
            kids: dict[str, tuple[str, np.ndarray]] = {}
            for jt in root.findall("joint"):
                o = jt.find("origin")
                rpy = [float(x) for x in o.attrib.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
                xyz = [float(x) for x in o.attrib.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
                M = np.eye(4)
                M[:3, :3] = _aa((0, 0, 1), rpy[2]) @ _aa((0, 1, 0), rpy[1]) @ _aa((1, 0, 0), rpy[0])
                M[:3, 3] = xyz
                child, parent = jt.find("child"), jt.find("parent")
                if child is None or parent is None:
                    raise SystemExit(f"URDF joint {jt.attrib.get('name', '?')} has no child or parent")
                kids[child.attrib["link"]] = (parent.attrib["link"], M)
            base_link = next(iter({v[0] for v in kids.values()} - set(kids)))

            def frame(link: str) -> np.ndarray:
                if link == base_link:
                    return np.eye(4)
                p, M = kids[link]
                return frame(p) @ M
            return frame

        _urdf_frame = _urdf_link_frame()
        _Tdh = _dh_frames_m()
        _Rz_pi = np.eye(4)
        _Rz_pi[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
        # bundle mesh name: (URDF link, DH frame index, sphere budget)
        _ARM_FIT = {"shoulder": ("shoulder_link", 1, 12), "upper_arm": ("upper_arm_link", 2, 22),
                    "forearm": ("forearm_link", 3, 22), "wrist_1": ("wrist_1_link", 4, 10),
                    "wrist_2": ("wrist_2_link", 5, 10), "wrist_3": ("wrist_3_link", 6, 6)}
        _added = 0
        for _mesh, (_L, _f, _n) in _ARM_FIT.items():
            _v = np.asarray(_MESH[f"{_mesh}__v"], float) / 1000.0
            _faces = np.asarray(_MESH[f"{_mesh}__f"], np.int64)
            _tm = trimesh.Trimesh(vertices=_v, faces=_faces, process=False)
            _res = fit_spheres_to_mesh(_tm, num_spheres=_n, surface_radius=0.010, fit_type=SphereFitType.SURFACE)
            _c = _res.centers.detach().cpu().numpy() if hasattr(_res.centers, "detach") else np.asarray(_res.centers)
            _r = (_res.radii.detach().cpu().numpy() if hasattr(_res.radii, "detach")
                  else np.asarray(_res.radii)).reshape(-1)
            _X = np.linalg.inv(_urdf_frame(_L)) @ _Rz_pi @ _Tdh[_f]
            for _ci, _ri in zip(_c, _r):
                if _ri <= 0.004:  # a degenerate fit sphere, smaller than the mesh detail it stands for
                    continue
                _cl = _X[:3, :3] @ _ci + _X[:3, 3]
                sphere_map[_L].append({"center": [round(float(x), 4) for x in _cl], "radius": round(float(_ri), 4)})
                _added += 1
        print(f"arm-link surface augmentation: +{_added} cuRobo surface spheres (ur5e only)")
    except Exception as exc:  # noqa: BLE001 (no cuRobo or trimesh here: Lula-only arm spheres, still valid)
        # This prints and keeps going, so the run still writes a yml. The banner is the whole
        # enforcement: nothing downstream can tell an augmented ur5e.yml from a Lula-only one.
        print("\n" + "!" * 100)
        print(f"!! ur5e arm-link surface augmentation SKIPPED ({type(exc).__name__}: {exc})")
        print("!! The emitted ur5e.yml is Lula-only. The augmentation adds surface spheres the Lula map")
        print("!! does not have, so a Lula-only ur5e.yml clears arm poses the augmented one refuses.")
        print("!! Re-run under the cuRobo env python (needs curobo.sphere_fit + trimesh) before shipping it.")
        print("!" * 100 + "\n")
else:
    print(f"arm-link surface augmentation SKIPPED for {MODEL}: the mesh bundle and DH placement are "
          f"ur5e-specific; {MODEL} keeps its own model-tuned Lula arm spheres, correct for its link lengths.")

print(f"converted spheres for {len(sphere_map)} links: {list(sphere_map)}  default_q={default_q}")

# --- 3) cuRobo ur10e.yml as the template; swap urdf + spheres + default config ----------------------------
_tpl = CUROBO / "configs/robot/ur10e.yml"
if not _tpl.is_file():
    raise SystemExit(f"missing cuRobo template {_tpl} (needed for the shared UR joint/link/ee scaffolding)")
cfg = yaml.safe_load(_tpl.read_text(encoding="utf-8"))
kin = cfg["robot_cfg"]["kinematics"]
kin["urdf_path"] = f"robot/ur_description/{MODEL}.urdf"
kin["collision_spheres"] = sphere_map
kin.pop("usd_path", None)  # the ur10e usd is wrong for this model
if "dynamics" in cfg["robot_cfg"]:
    cfg["robot_cfg"]["dynamics"].pop("neural_inverse_dynamics_state_dict", None)
# Fix the upstream ur10e typo: self_collision_ignore has "forarm_link", which should be
# "forearm_link". With the dense Lula spheres the misspelled key never ignores the forearm to wrist_1
# pair, so every configuration self-collides and plan_pose returns None.
sci = kin.get("self_collision_ignore", {})
if "forarm_link" in sci:
    sci["forearm_link"] = sorted(set(sci.get("forearm_link", []) + sci.pop("forarm_link")))
    print(f"  fixed the self_collision_ignore typo; forearm_link is now {sci['forearm_link']}")
cs = kin.get("cspace", {})
if "default_joint_position" in cs:
    print(f"  default_joint_position {cs['default_joint_position']} becomes {default_q}")
    cs["default_joint_position"] = default_q

YML_OUT.parent.mkdir(parents=True, exist_ok=True)
YML_OUT.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
print(f"wrote {YML_OUT}")
print(f"  joint order check: cspace.joint_names={cs.get('joint_names')}")
print(f"\nNext: point the planner at it with WILLY_CUROBO_ROBOT={MODEL}.yml, or set robot_model: {MODEL} "
      f"in the sim config, which derives the name automatically.")
print(f"Then confirm the box agrees: python -m src.robot.safety.planning --check --model {MODEL}")
