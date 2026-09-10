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

The arm-link surface augmentation runs for every model with a baked bundle. Each Lula arm link
gains cuRobo ``SphereFitType.SURFACE`` spheres fitted to that model's OWN
``{model}_collision_meshes.npz`` and placed through its OWN DH chain; both halves are per-model,
so the only thing that ever limited this was which bundles existed. A model with no bundle keeps
its own Lula arm spheres, which are model-tuned and correct for its link lengths, and the run
says so.

Two things the vendor map can be wrong about, and what happens then. A link it SKIPS is authored
from the collision primitives the description declares, and the build refuses to write if that
authored link covers less of its own surface than the weakest vendor-mapped link on the same
arm. A sphere whose centre lies further outside its own link mesh than its own RADIUS does not
touch that body, so it is dropped: measured on ur10, whose vendor map puts all three wrists
about 61 mm from where their geometry is, and keeping both sets made every configuration
collide so the planner returned nothing at all while the descriptor loaded perfectly.

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
import re
import shutil
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

# How far the hand sits in front of the flange, along the approach, because of whatever plate
# is bolted between them. Only consulted when the sphere map says its numbers start at the
# hand's own mounting face; see the refusal below. It is the same bench measurement
# `robot.gripper.tool_frame.offset_mm` needs, taken once.
COUPLING_MM = None
for _i, _arg in enumerate(sys.argv):
    if _arg == "--coupling-mm" and _i + 1 < len(sys.argv):
        COUPLING_MM = float(sys.argv[_i + 1])

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

# --- 1) URDF: a description that actually DESCRIBES A BODY ------------------------------------------------
# Isaac names every UR description `{model}.urdf` except one: `ur10` ships `ur10_robot.urdf`. That is
# the small half. The large half, measured 2026-09-09: Isaac ships ur10 in TWO INCOMPATIBLE
# LINK-FRAME FAMILIES, and the description sitting beside its Lula sphere map belongs to the wrong
# one. Pairing them put 23 of 30 sphere centres off the arm, worst 341.5 mm, with nothing raising --
# the planner would model the arm where it is not and leave unguarded the space where it is.
#
# ⭐ TWO GENERAL RULES DECIDE IT, AND NEITHER NAMES A MODEL:
#   1. A description with no geometry describes no body. `ur10_robot.urdf` is 74 lines of pure
#      kinematics: 0 <collision>, 0 <visual>, 0 mesh references. Every description that plans today
#      carries 7 / 7 / 14. That one rule excludes the wrong-family file and changes nothing for the
#      five that already work.
#   2. Its mesh references have to resolve. `ur10_robot_suction.urdf` has geometry and all 14 of its
#      mesh files are missing from disk, so it can supply frames but not a body.
#
# For ur10 those select the URDF-importer package copy, which is the +z-family description whose
# frames match Isaac's own sphere map. `tests/test_lula_and_bundle_agree.py` is the edge that keeps
# this honest: it fails if the description written here and the sphere map disagree about a link.


def _importer_root():
    """The URDF-importer package, which ships a SECOND copy of some robots, or None."""
    for _root in _isaac_roots():
        for _hit in _root.glob("**/isaacsim.asset.importer.urdf/data/urdf/robots"):
            return _hit
    return None


def _describes_a_body(path: Path) -> tuple[bool, str]:
    """(usable, why). A description needs geometry, and its meshes have to be findable."""
    text = path.read_text(encoding="utf-8")
    refs = re.findall(r'filename="([^"]+)"', text)
    if not refs and "<collision>" not in text:
        return False, "no geometry at all: 0 mesh references and no <collision>"
    found = sum(1 for r in refs if (path.parent / r).is_file()
                or (path.parent.parent / r.lstrip("./")).is_file())
    if refs and found == 0:
        return False, f"all {len(refs)} mesh references are missing from disk"
    return True, f"{len(refs)} mesh refs ({found} resolvable), {text.count('<collision>')} <collision>"


_candidates = [ISAAC_MODEL_DIR / f"{MODEL}.urdf", ISAAC_MODEL_DIR / f"{MODEL}_robot.urdf"]
_imp = _importer_root()
if _imp is not None:
    _candidates.append(_imp / f"{MODEL}/urdf/{MODEL}.urdf")
_urdf_src = None
for _cand in _candidates:
    if not _cand.is_file():
        continue
    _ok, _why = _describes_a_body(_cand)
    print(f"urdf candidate {_cand.name:26s} {'USABLE ' if _ok else 'skipped'} {_why}")
    if _ok and _urdf_src is None:
        _urdf_src = _cand
if _urdf_src is None:
    raise SystemExit(
        f"no usable description for {MODEL}. Looked at: "
        f"{[str(c) for c in _candidates if c.is_file()] or 'nothing on disk'}. A description with no "
        f"geometry cannot supply collision meshes, and one whose meshes are missing cannot either."
    )
print(f"urdf: using {_urdf_src}")

urdf = _urdf_src.read_text(encoding="utf-8")

# ⛔ ONE MALFORMED NUMBER IS THE WHOLE "cannot be loaded" BLOCKER. Measured 2026-09-10: the importer
# ur10.urdf carries `xyz="0 0.0 0.0027000046)"` on line 28, with a stray closing parenthesis, and
# cuRobo's parser (yourdfpy) raises ValueError on it. Repaired in the copy written here, never in
# Isaac's own tree.
_repaired = re.sub(r'(xyz|rpy)="([^"]*)"',
                         lambda m: f'{m.group(1)}="{m.group(2).replace(chr(41), "")}"', urdf)
if _repaired != urdf:
    print(f"urdf: {_urdf_src.name} carries a malformed number (a stray parenthesis); repaired in the "
          f"copy written here, not in Isaac's tree")
    urdf = _repaired

urdf = urdf.replace("package://ur_description/", "")  # -> meshes/<model>/... relative to asset_root_path

# THE MESHES TRAVEL WITH THE DESCRIPTION. A urdf referencing files that are not under the cuRobo
# asset root produces a config that loads and then has no body to check against, and `ur3`, `ur3e`
# and `ur5` have shipped in exactly that state. Anything resolvable beside the source is copied and
# the reference is rewritten to where it landed.
_mesh_dir = URDF_OUT.parent / "meshes" / MODEL
for _ref in sorted(set(re.findall(r'filename="([^"]+)"', urdf))):
    _src_mesh = None
    for _base in (_urdf_src.parent, _urdf_src.parent.parent):
        _cand_mesh = (_base / _ref).resolve()
        if _cand_mesh.is_file():
            _src_mesh = _cand_mesh
            break
    if _src_mesh is None:
        continue
    _mesh_dir.mkdir(parents=True, exist_ok=True)
    _dst_mesh = _mesh_dir / _src_mesh.name
    if not _dst_mesh.is_file() or _dst_mesh.stat().st_size != _src_mesh.stat().st_size:
        shutil.copyfile(_src_mesh, _dst_mesh)
    urdf = urdf.replace(f'filename="{_ref}"', f'filename="meshes/{MODEL}/{_src_mesh.name}"')
if _mesh_dir.is_dir():
    print(f"meshes: {len(list(_mesh_dir.iterdir()))} file(s) copied beside the description "
          f"-> {_mesh_dir.relative_to(URDF_OUT.parent)}")
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

# ⚠ AND GATED. The transplant is only meaningful if ``wrist_3_link`` is the SAME frame on both robots.
# That is asserted here rather than assumed: base -> wrist_3 at q=0 is computed from each description and
# the ROTATIONS must agree. Measured identical on all six, which is why this is a transcription.
def _urdf_frames(text: str):
    """base -> link at q=0, for every link in a URDF, as 4x4 metres."""
    import xml.etree.ElementTree as ET

    def _rpy(r, p, y):
        def _R(axis: int, t: float):
            a = np.zeros(3)
            a[axis] = 1.0
            K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
            return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)
        return _R(2, y) @ _R(1, p) @ _R(0, r)

    kids: dict = {}
    for jt in ET.fromstring(text).findall("joint"):
        child, parent = jt.find("child"), jt.find("parent")
        if child is None or parent is None:
            continue
        o = jt.find("origin")
        r = [float(v) for v in o.attrib.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
        x = [float(v) for v in o.attrib.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
        M = np.eye(4)
        M[:3, :3] = _rpy(*r)
        M[:3, 3] = x
        kids[child.attrib["link"]] = (parent.attrib["link"], M)

    def frame(link: str) -> np.ndarray:
        if link not in kids:
            return np.eye(4)
        parent, M = kids[link]
        return frame(parent) @ M
    return frame


# The donor supplies whatever is MISSING, not only tool0. A description can be the right robot
# in the right frame family and still be incomplete in a way that only shows at load.
def _inertias_are_complete(text: str) -> bool:
    """Every link with an <inertial> also carries an <inertia> tensor.

    ⛔ MEASURED 2026-09-10: the importer ur10.urdf gives every link a <mass> and no tensor.
    cuRobo reads inertia_matrix[0, 0], gets None, and dies with a TypeError naming no link at
    all -- after the descriptor has been written and reported as a success.
    """
    for _m in re.finditer(r"<inertial>(.*?)</inertial>", text, re.S):
        if "<inertia " not in _m.group(1):
            return False
    return True


_needs_tool0 = '<link name="tool0"' not in urdf
_needs_inertia = not _inertias_are_complete(urdf)
if _needs_tool0 or _needs_inertia:
    # The donor may be a SIBLING DESCRIPTION OF THE SAME ROBOT rather than another model. For ur10
    # the frame-identical `ur10_robot_suction.urdf` carries tool0 92.2 mm out and rotated, where a
    # ur10e donor carries an identity chain -- so a ur10e graft would put tool0 92.2 mm short and
    # 90 degrees off. The rotation gate below is what tells them apart; this list only has to offer
    # both, and offering more costs nothing because the gate refuses the wrong ones.
    _donor_dirs = [URDF_OUT.parent, ISAAC_MODEL_DIR]
    _donors = sorted(
        {p for d in _donor_dirs for p in d.glob("*.urdf")
         if p.name != URDF_OUT.name and '<link name="tool0"' in p.read_text(encoding="utf-8")},
        key=lambda p: (p.parent != ISAAC_MODEL_DIR, p.name),  # same-robot siblings first
    )
    if not _donors:
        raise SystemExit(
            f"{MODEL} has no tool0 and no sibling description in {URDF_OUT.parent} has one to copy from. "
            f"Build any other UR model first (its urdf lands in the same directory), then this one."
        )
    # THE GATE PICKS THE DONOR. Same wrist frame, or the transplant authors a tool frame pointing
    # somewhere else -- which is the mirrored-gripper defect in another costume. Every donor is
    # measured and the first that agrees is used; if none agrees, nothing is written.
    _mine = _urdf_frames(urdf)("wrist_3_link")[:3, :3]
    _donor, _donor_text, _worst = None, None, float("inf")
    for _cand_donor in _donors:
        _text = _cand_donor.read_text(encoding="utf-8")
        _d = float(np.max(np.abs(_mine - _urdf_frames(_text)("wrist_3_link")[:3, :3])))
        print(f"tool0 donor {_cand_donor.name:26s} wrist_3 rotation differs by {_d:.6f}"
              f"{'  <- taken' if _d <= 1e-6 and _donor is None else ''}")
        _worst = min(_worst, _d)
        if _d <= 1e-6 and _donor is None:
            _donor, _donor_text = _cand_donor, _text
    if _donor is None:
        raise SystemExit(
            f"{MODEL} wrist_3_link is not the frame ANY available donor uses (closest differs by "
            f"{_worst:.6f}), so the fixed wrist_3 -> flange -> tool0 chain cannot be transcribed onto "
            f"it. Read the tool frame off this description instead of copying one; a tool frame "
            f"written from memory passes every extent check while pointing the hand elsewhere."
        )

    import re as _re
    # `flange` is an INTERMEDIATE on one generation and absent on the other, so only `tool0` is
    # required. Measured 2026-09-10: the modern chain is wrist_3 -> flange -> tool0 with tool0 landing
    # exactly on wrist_3_link; the older one goes wrist_3 -> tool0 directly, 92.2 mm out and rotated
    # -90 degrees about x, with no flange at all. Requiring both made this refuse the one donor whose
    # wrist frame agrees, which is the donor the gate above had just chosen.
    _grafts = []
    for _name, _required in (("flange", False), ("tool0", _needs_tool0)):
        _lk = _re.search(rf'<link name="{_name}"\s*/>|<link name="{_name}">.*?</link>', _donor_text, _re.S)
        _jt = _re.search(rf'<joint name="[^"]*"[^>]*>(?:(?!</joint>).)*?'
                         rf'<child link="{_name}"[^>]*/>(?:(?!</joint>).)*?</joint>', _donor_text, _re.S)
        if _lk is None or _jt is None:
            if _required:
                raise SystemExit(
                    f"{_donor.name} has no complete {_name} link+joint to copy, and {_name} is what "
                    f"the cuRobo scaffolding names as its end effector."
                )
            continue
        _grafts += [_lk.group(0), _jt.group(0)]
    if _needs_tool0:
        urdf = urdf.replace("</robot>", "\n  " + "\n  ".join(_grafts) + "\n</robot>")

    if _needs_inertia:
        # Whole blocks, mass included: the importer file says the base weighs 200 kg where the donor
        # says 4, and half of one dynamics model plus half of another is worse than either.
        _donor_inertials = {
            _m.group(1): _re.search(r"<inertial>.*?</inertial>", _m.group(2), _re.S)
            for _m in _re.finditer(r'<link name="([^"]+)">(.*?)</link>', _donor_text, _re.S)
        }
        _swapped = []
        for _m in list(_re.finditer(r'<link name="([^"]+)">(.*?)</link>', urdf, _re.S)):
            _link, _body = _m.group(1), _m.group(2)
            _mine = _re.search(r"<inertial>.*?</inertial>", _body, _re.S)
            _theirs = _donor_inertials.get(_link)
            if _mine is None or _theirs is None or "<inertia " in _mine.group(0):
                continue
            urdf = urdf.replace(_m.group(0), _m.group(0).replace(_mine.group(0), _theirs.group(0)), 1)
            _swapped.append(_link)
        if not _swapped:
            raise SystemExit(
                f"{MODEL} has incomplete inertias and {_donor.name} has none to replace them with. "
                f"cuRobo reads inertia_matrix[0, 0] and would die at load with a TypeError naming no "
                f"link, so nothing is written."
            )
        print(f"inertias: {len(_swapped)} link(s) carried a mass with no tensor; transcribed complete "
              f"blocks from {_donor.name} ({', '.join(_swapped)})")

    URDF_OUT.write_text(urdf, encoding="utf-8")
    if _needs_tool0:
        print(f"tool0: {MODEL} is the older ur_description generation and has none. Transcribed "
              f"{len(_grafts) // 2} fixed link(s) from {_donor.name}, whose wrist_3 frame is this "
              f"one to {_worst:.2e} -- a copy rather than an invention.")


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
_gripper_prov = _gripper_cfg.get("_provenance", {})
_gripper_name = _gripper_prov.get("gripper", GRIPPER)
_origin = _gripper_prov.get("origin", "flange")

# --- the coupling -----------------------------------------------------------------------
# A map fitted from a composed arm asset was already placed by the arm: its numbers start at
# the flange and go in as they are. A map fitted from a standalone vendor asset starts at the
# hand's own mounting face, and the plate between that face and the flange is not in it.
#
# The arithmetic lives in `_gripper_placement.py` beside this script rather than inline, so a
# test can prove the plate is added where it is supplied. A refusal only proves the path is
# not taken silently. It is a separate file rather than a repo import because this runs under
# the cuRobo sidecar interpreter, where `src.robot...` is un-importable.
#
# Maps written before the origin field existed carry none, and every one of those came from a
# composed arm asset, so the absent case reads as "flange" as a statement about those files.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gripper_placement import PlacementError, place_tool0_spheres  # noqa: E402

try:
    _placed = place_tool0_spheres(
        _gripper_cfg["collision_spheres"]["tool0"], origin=_origin, coupling_mm=COUPLING_MM
    )
except PlacementError as exc:
    raise SystemExit(f"{_SPHERE_MAP.name}: {exc}") from None
if COUPLING_MM:
    print(f"coupling: every tool0 sphere moved {COUPLING_MM:.1f} mm along the approach, "
          f"because {_SPHERE_MAP.name} starts at the hand's own mounting face")

sphere_map["tool0"] = _placed
print(f"tool0 spheres: {len(sphere_map['tool0'])} from {_SPHERE_MAP.name} ({_gripper_name})")

# --- 2b) ARM-link surface augmentation: every model with a baked bundle ------------------------------------
# cuRobo surface spheres fitted to this arm own collision meshes, placed through this arm own DH
# chain. Worth having: measured false-CLEAR 8.4% to 5.3% on the ur5e, against the Lula spheres alone.
#
# THE GATE IS THE DIRECTORY, NOT A MODEL NAME. This branch read `if MODEL == "ur5e"` and said the
# bundle and the DH placement were ur5e-specific. That was true while ur5e was the only arm with a
# baked bundle; five arms have one as of 2026-09-09 and the sentence had no edge to the bake that
# turned it over. Both halves are per-model and always were, and what was missing was the geometry.
#
# A model with no bundle keeps its own Lula arm spheres, which are model-tuned and correct, just
# coarser. ur10 is permanently in that state: its asset collides the whole arm with primitives.
_ARM_NPZ = REPO / f"src/robot/safety/data/{MODEL}_collision_meshes.npz"
if _ARM_NPZ.is_file():
    try:
        # The arm bundle, which is a different thing from the gripper map read above: these are
        # this model own link meshes, baked from its own Isaac asset and gated against the ur5e.
        _MESH = np.load(_ARM_NPZ, allow_pickle=True)
        import trimesh  # type: ignore[import-not-found]
        from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh  # type: ignore[import-not-found]

        # DH (a, d, alpha), inlined rather than imported: this branch only runs in the cuRobo
        # sidecar environment, where the repository package cannot be imported. The values match
        # ``src/robot/safety/_ur_kinematics`` row for row, and ``tests/test_inlined_dh_tables.py``
        # compares all three copies, because three copies of a safety-critical table with nothing
        # between them is how one of them ends up quietly wrong.
        _DH_TABLES = {
            "ur3": [(0.0, 0.1519, 1.570796327), (-0.24365, 0.0, 0.0), (-0.21325, 0.0, 0.0),
                    (0.0, 0.11235, 1.570796327), (0.0, 0.08535, -1.570796327), (0.0, 0.0819, 0.0)],
            "ur3e": [(0.0, 0.15185, 1.570796327), (-0.24355, 0.0, 0.0), (-0.2132, 0.0, 0.0),
                     (0.0, 0.13105, 1.570796327), (0.0, 0.08535, -1.570796327), (0.0, 0.0921, 0.0)],
            "ur5": [(0.0, 0.089159, 1.570796327), (-0.425, 0.0, 0.0), (-0.39225, 0.0, 0.0),
                    (0.0, 0.10915, 1.570796327), (0.0, 0.09465, -1.570796327), (0.0, 0.0823, 0.0)],
            "ur5e": [(0.0, 0.1625, 1.570796327), (-0.425, 0.0, 0.0), (-0.3922, 0.0, 0.0),
                     (0.0, 0.1333, 1.570796327), (0.0, 0.0997, -1.570796327), (0.0, 0.0996, 0.0)],
            "ur10": [(0.0, 0.1273, 1.570796327), (-0.612, 0.0, 0.0), (-0.5723, 0.0, 0.0),
                     (0.0, 0.163941, 1.570796327), (0.0, 0.1157, -1.570796327), (0.0, 0.0922, 0.0)],
            "ur10e": [(0.0, 0.1807, 1.570796327), (-0.6127, 0.0, 0.0), (-0.57155, 0.0, 0.0),
                      (0.0, 0.17415, 1.570796327), (0.0, 0.11985, -1.570796327), (0.0, 0.11655, 0.0)],
        }
        _DH: list[tuple[float, float, float]] = _DH_TABLES[MODEL]

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

        def _declared_primitives(text, link_name):
            """Every <collision> PRIMITIVE a link declares, as (centre, axis, half_length, radius).

            In the LINK frame, which is the frame the sphere map is written in, so these need no
            transform at all. A sphere is fully described by a centre and a radius, so a box or a
            plane cannot be represented and is skipped rather than approximated.
            """
            import xml.etree.ElementTree as _ETp

            out = []
            for _ln in _ETp.fromstring(text).findall("link"):
                if _ln.attrib.get("name") != link_name:
                    continue
                for _col in _ln.findall("collision"):
                    _cyl = _col.find("geometry/cylinder")
                    _sph = _col.find("geometry/sphere")
                    _o = _col.find("origin")
                    _r = [float(v) for v in _o.attrib.get("rpy", "0 0 0").split()] if _o is not None else [0.0] * 3
                    _x = [float(v) for v in _o.attrib.get("xyz", "0 0 0").split()] if _o is not None else [0.0] * 3
                    _M = np.eye(4)
                    _M[:3, :3] = _aa((0, 0, 1), _r[2]) @ _aa((0, 1, 0), _r[1]) @ _aa((1, 0, 0), _r[0])
                    _M[:3, 3] = _x
                    if _cyl is not None:
                        out.append((_M[:3, 3], _M[:3, 2], float(_cyl.attrib["length"]) / 2.0,
                                    float(_cyl.attrib["radius"])))
                    elif _sph is not None:
                        out.append((_M[:3, 3], _M[:3, 2], 0.0, float(_sph.attrib["radius"])))
            return out

        _urdf_frame = _urdf_link_frame()
        _Tdh = _dh_frames_m()
        _Rz_pi = np.eye(4)
        _Rz_pi[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
        # bundle mesh name: (URDF link, DH frame index, sphere budget)
        _ARM_FIT = {"shoulder": ("shoulder_link", 1, 12), "upper_arm": ("upper_arm_link", 2, 22),
                    "forearm": ("forearm_link", 3, 22), "wrist_1": ("wrist_1_link", 4, 10),
                    "wrist_2": ("wrist_2_link", 5, 10), "wrist_3": ("wrist_3_link", 6, 6)}
        # ⛔ A VENDOR SPHERE THAT DOES NOT TOUCH ITS OWN LINK IS NOT THAT LINK'S SPHERE.
        # Measured 2026-09-10: Isaac's ur10 Lula map puts all three WRIST links' spheres about 61 mm
        # from where that link's geometry is (wrist_1 mesh y[60.8, 160.7] with spheres at y = 0, and
        # the same on the other two). The bundle is not the wrong half: through the DH chain the
        # baked arm is CONNECTED, every gap between consecutive links under 3.3 mm against 0.2 to
        # 0.8 mm on ur10e.
        #
        # Keeping both sets made each wrist a body twice its size spanning two positions, so cuRobo
        # found every configuration in collision and returned None from plan_pose AND plan_cspace,
        # including a plan from a pose to itself. The descriptor loaded perfectly and planned nothing.
        #
        # The threshold is not tuned: a sphere whose centre is further outside its link's own mesh
        # than its own RADIUS does not intersect that body at all. ur10's wrists fail it at 61 mm
        # with r 45-50; ur10e's worst link passes at 13 mm with r 60; every other arm is at 0.0 mm.
        _stray = {}
        for _mesh_key, (_L, _f, _n) in _ARM_FIT.items():
            if _L not in sphere_map or f"{_mesh_key}__v" not in _MESH:
                continue
            _mv = np.asarray(_MESH[f"{_mesh_key}__v"], float) / 1000.0
            _MX = np.linalg.inv(_urdf_frame(_L)) @ _Rz_pi @ _Tdh[_f]
            _mv = (_MX[:3, :3] @ _mv.T).T + _MX[:3, 3]
            _lo, _hi = _mv.min(0), _mv.max(0)
            _keep, _drop = [], 0
            for _sp in sphere_map[_L]:
                _c = np.asarray(_sp["center"], float)
                _out = float(np.linalg.norm(np.maximum(np.maximum(_lo - _c, _c - _hi), 0.0)))
                if _out > float(_sp["radius"]):
                    _drop += 1
                    continue
                _keep.append(_sp)
            if _drop:
                _stray[_L] = (_drop, len(sphere_map[_L]))
                sphere_map[_L] = _keep
                if not _keep:
                    del sphere_map[_L]          # nothing survived: the link is authored below
        for _L, (_d, _t) in _stray.items():
            print(f"vendor spheres: dropped {_d} of {_t} on {_L}; their centres sit further outside "
                  f"this arm's own baked mesh than their own radius, so they are not on this link")

        _added, _authored, _coverage = 0, [], {}
        for _mesh, (_L, _f, _n) in _ARM_FIT.items():
            _v = np.asarray(_MESH[f"{_mesh}__v"], float) / 1000.0
            _faces = np.asarray(_MESH[f"{_mesh}__f"], np.int64)
            _tm = trimesh.Trimesh(vertices=_v, faces=_faces, process=False)
            _X = np.linalg.inv(_urdf_frame(_L)) @ _Rz_pi @ _Tdh[_f]
            _is_authored = _L not in sphere_map
            if _is_authored:
                # ⛔ AUTHORED FROM THE DECLARED PRIMITIVES, NOT FROM A FIT. Measured 2026-09-10:
                # SURFACE places fixed 10 mm spheres, so 100 of them cover 25.6 % of this link;
                # VOXEL places large spheres that never reach the surface and cover 0.0 %. A real
                # vendor map is FEW spheres, LARGE, along the link axis -- ur10e's shoulder is two
                # spheres of r 83.5 mm at 31.4 % coverage, and its weakest link is 10.2 %.
                #
                # The description already declares exactly that shape. Spheres along the two
                # collision cylinders it gives this link score 48.7 %, better than ur10e's own
                # shoulder. And a <collision> origin is already in the link frame, which is the frame
                # the sphere map is written in, so nothing is transformed and nothing can be
                # transformed wrongly.
                _prims = _declared_primitives(urdf, _L)
                if not _prims:
                    raise SystemExit(
                        f"{MODEL}/{_L} has no vendor spheres and no declared collision primitive to "
                        f"author them from, so this link would be unguarded. Nothing written."
                    )
                sphere_map[_L] = []
                _authored.append(_L)
                for _centre, _axis, _half, _rad in _prims:
                    # Spaced at half a radius: the coverage curve is flat past that (48.7 % at 0.5
                    # against 49.9 % at 0.25), so more spheres buy nothing but planner cost.
                    _n_along = max(2, int(np.ceil(2.0 * _half / max(_rad * 0.5, 1e-6))) + 1)
                    for _t in np.linspace(-_half, _half, _n_along):
                        _pt = _centre + _t * _axis
                        sphere_map[_L].append({"center": [round(float(x), 4) for x in _pt],
                                               "radius": round(float(_rad), 4)})
                        _added += 1
            else:
                _res = fit_spheres_to_mesh(_tm, num_spheres=_n, surface_radius=0.010,
                                           fit_type=SphereFitType.SURFACE)
                _c = (_res.centers.detach().cpu().numpy() if hasattr(_res.centers, "detach")
                      else np.asarray(_res.centers))
                _r = (_res.radii.detach().cpu().numpy() if hasattr(_res.radii, "detach")
                      else np.asarray(_res.radii)).reshape(-1)
                for _ci, _ri in zip(_c, _r):
                    if _ri <= 0.004:
                        continue
                    _cl = _X[:3, :3] @ _ci + _X[:3, 3]
                    sphere_map[_L].append({"center": [round(float(x), 4) for x in _cl],
                                           "radius": round(float(_ri), 4)})
                    _added += 1

            # COVERAGE, in the bundle's own frame so the placement above cannot flatter it: what
            # fraction of this link's surface lies inside SOME sphere of its final map?
            _pts = _tm.sample(4000) if len(_tm.faces) else np.zeros((0, 3))
            if len(_pts):
                _all = [(np.asarray(sp["center"], float), float(sp["radius"])) for sp in sphere_map[_L]]
                _back = np.linalg.inv(_X)
                _inside = np.zeros(len(_pts), dtype=bool)
                for _sc, _sr in _all:
                    _local = (_back[:3, :3] @ np.asarray(_sc) + _back[:3, 3])
                    _inside |= np.linalg.norm(_pts - _local, axis=1) <= _sr
                _coverage[_L] = float(_inside.mean())

        print(f"arm-link spheres: +{_added} from {_ARM_NPZ.name} placed through the {MODEL} DH chain"
              + (f"; AUTHORED {_authored} outright, because the vendor map has none for them"
                 if _authored else ""))
        if _coverage:
            _vendor = {k: v for k, v in _coverage.items() if k not in _authored}
            for _L, _cov in sorted(_coverage.items()):
                _tag = "AUTHORED" if _L in _authored else "vendor+fit"
                print(f"  coverage {_L:16s} {_cov * 100:5.1f} %  ({_tag})")
            if _authored and _vendor:
                _worst_authored = min(_coverage[_L] for _L in _authored)
                _worst_vendor = min(_vendor.values())
                print(f"  the authored link(s) cover {_worst_authored * 100:.1f} % against "
                      f"{_worst_vendor * 100:.1f} % for the weakest vendor-mapped link on this arm")
                if _worst_authored < _worst_vendor - 0.05:
                    raise SystemExit(
                        f"an authored link covers {_worst_authored * 100:.1f} % of its own surface "
                        f"where the weakest vendor-mapped link on this same robot covers "
                        f"{_worst_vendor * 100:.1f} %. A thinner shell than the vendor ships is a "
                        f"pose the guard clears and should refuse, so nothing is written. Author "
                        f"them from a finer primitive, or supply a vendor map for this link."
                    )
    except Exception as exc:  # noqa: BLE001 (no cuRobo or trimesh here: Lula-only arm spheres, still valid)
        # This prints and keeps going, so the run still writes a yml. The banner is the whole
        # enforcement: nothing downstream can tell an augmented ur5e.yml from a Lula-only one.
        print("\n" + "!" * 100)
        print(f"!! {MODEL} arm-link surface augmentation SKIPPED ({type(exc).__name__}: {exc})")
        print("!! The emitted ur5e.yml is Lula-only. The augmentation adds surface spheres the Lula map")
        print("!! does not have, so a Lula-only ur5e.yml clears arm poses the augmented one refuses.")
        print("!! Re-run under the cuRobo env python (needs curobo.sphere_fit + trimesh) before shipping it.")
        print("!" * 100 + "\n")
else:
    print(f"arm-link surface augmentation SKIPPED for {MODEL}: no {_ARM_NPZ.name} to fit. "
          f"{MODEL} keeps its own model-tuned Lula arm spheres, which are correct for its link "
          f"lengths and coarser. Bake one with scripts/isaac/bake_ur_collision_meshes.py, unless "
          f"that refuses: a model whose asset collides with primitives has no exact geometry to "
          f"fit and never will.")

print(f"converted spheres for {len(sphere_map)} links: {list(sphere_map)}  default_q={default_q}")

# --- 3) cuRobo ur10e.yml as the template; swap urdf + spheres + default config ----------------------------
_tpl = CUROBO / "configs/robot/ur10e.yml"
if not _tpl.is_file():
    raise SystemExit(f"missing cuRobo template {_tpl} (needed for the shared UR joint/link/ee scaffolding)")
cfg = yaml.safe_load(_tpl.read_text(encoding="utf-8"))
kin = cfg["robot_cfg"]["kinematics"]
kin["urdf_path"] = f"robot/ur_description/{MODEL}.urdf"
kin["collision_spheres"] = sphere_map

# ⛔ EVERY LINK THE TEMPLATE GUARDS MUST HAVE SPHERES, CHECKED HERE RATHER THAN AT THE FIRST PLAN.
# MEASURED 2026-09-09: the template declares seven collision links and the ur10 Lula description
# covers six. Isaac ships no shoulder_link spheres for that model and for no other. cuRobo then
# raises KeyError: shoulder_link the first time anything loads the file, with every step before it
# reporting success -- a descriptor probe reads the path, finds a file, and says ok all the way to
# the first motion.
#
# The near miss is the worse half. Had the template not happened to name the link, this would have
# written a ur10 with NO collision spheres on its shoulder and planned against it in silence.
_guarded = set(kin.get("collision_link_names") or [])
# ⛔ COUNT THE SPHERES, DO NOT COUNT THE KEYS. This read `_guarded - set(sphere_map)`, which
# an EMPTY list passes: the key is present and the link is unguarded. Reproduced on
# 2026-09-10 by a fit that raised after its key was created -- the build wrote a ur10
# whose shoulder_link had zero spheres and said nothing. A mechanism that does not fire
# looks exactly like a mechanism with nothing to catch.
_unarmed = sorted(_l for _l in _guarded if not sphere_map.get(_l))
if _unarmed:
    raise SystemExit(
        f"{MODEL}: the template guards {sorted(_guarded)} and this build has spheres for "
        f"{sorted(k for k, v in sphere_map.items() if v)}. Missing or empty: {_unarmed}.\n"
        f"  cuRobo raises KeyError on the first load, so no file is written rather than one that\n"
        f"  fails later. A link with no spheres would not be checked for collision at all.\n"
        f"  Isaac ships no Lula spheres for {MODEL}/{_unarmed}; the other UR models all have them.\n"
        f"\n"
        f"  For ur10 specifically, generating the missing spheres would NOT be enough, and this is the\n"
        f"  deeper reason it has no descriptor. ISAAC SHIPS ur10 IN TWO INCOMPATIBLE LINK-FRAME FAMILIES.\n"
        f"  Its Lula sphere map belongs to the +z family; ur10_robot.urdf beside it and cuRobo own shipped\n"
        f"  ur_description/ur10.urdf are both -x family. Every other UR is -x on both sides, so ur10 is the\n"
        f"  one model where the OBVIOUS pairing is the wrong one.\n"
        f"\n"
        f"  MEASURED 2026-09-09 through the full FK chain, as the signed distance of each sphere centre to\n"
        f"  the arm body, with two working pairings as calibration:\n"
        f"      sphere map + curobo ur10.urdf     23 of 30 outside, worst +341.5 mm   <- the obvious one\n"
        f"      sphere map + importer ur10.urdf    6 of 30 outside, worst  +38.3 mm\n"
        f"      ur10e, which works today           5 of 33 outside, worst  +45.3 mm\n"
        f"      ur5e,  which works today           4 of 39 outside, worst  +51.0 mm\n"
        f"\n"
        f"  So the failure mode is FAIL-OPEN by 341.5 mm: a planner would model the arm where it is not\n"
        f"  and leave unguarded the space where it is, and nothing would raise.\n"
        f"\n"
        f"  The importer urdf at isaacsim.asset.importer.urdf/data/urdf/robots/ur10/urdf/ur10.urdf IS the\n"
        f"  right frame family, and it is still not a route to a descriptor today: cuRobo parser (yourdfpy)\n"
        f"  refuses to load it, it has no tool0, and neither Lula file has shoulder_link spheres. The\n"
        f"  frame-identical ur10_robot_suction.urdf does carry tool0, and all 14 of its meshes are missing\n"
        f"  from disk. See docs/runbooks/ur_family_bringup.md for what a real fix would have to author."
    )
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

# --- provenance: which arm, which hand, and how the hand was placed ------------------------
# Until this existed the descriptor carried the spheres and dropped every statement about
# where they came from, so nothing on the box and no probe in this repository could answer
# "which gripper is this modelling". A cell that swaps hands and does not re-run this script
# plans with the old one indefinitely, and the file looks correct either way. cuRobo ignores
# unknown top-level keys, so this costs the planner nothing.
cfg["_provenance"] = {
    "arm": MODEL,
    "gripper": _gripper_name,
    "gripper_key": GRIPPER,
    "gripper_spheres": _SPHERE_MAP.name,
    "gripper_origin": _origin,
    "coupling_mm": COUPLING_MM,
    "arm_spheres": f"lula + {MODEL} surface augmentation" if _ARM_NPZ.is_file() else "lula",
    "generated_by": "scripts/curobo/build_ur_config.py",
}

YML_OUT.parent.mkdir(parents=True, exist_ok=True)
YML_OUT.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
print(f"wrote {YML_OUT}")
print(f"  provenance: {MODEL} + {_gripper_name} from {_SPHERE_MAP.name}"
      + (f", coupling {COUPLING_MM:.1f} mm" if COUPLING_MM is not None else ""))
print(f"  joint order check: cspace.joint_names={cs.get('joint_names')}")
print(f"\nNext: point the planner at it with WILLY_CUROBO_ROBOT={MODEL}.yml, or set robot_model: {MODEL} "
      f"in the sim config, which derives the name automatically.")
print(f"Then confirm the box agrees: python -m src.robot.safety.planning --check --model {MODEL}")
