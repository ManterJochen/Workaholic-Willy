"""Assemble a cuRobo robot config for any bundled UR model from the ingredients already on the box.

This is the one script that builds a UR. Willy drives more than one of them, and a descriptor is named by the arm
alone, ``willy_{model}.yml``. That file is not shipped with the repository: it is assembled here, once per box and per
arm, and written into the gitignored cuRobo content directory. A fresh cuRobo clone always needs this step.

Ingredients, and where each comes from:
  * kinematics: Universal Robots' own description, rendered here from their vendored config (``--urdf-from ur``,
    the default; ``_urdf_source.py`` names the alternative and what it promises). Written into the cuRobo content
    with relative mesh paths. Read from an Isaac install instead, a customer box cannot build a descriptor at all,
    and for ur10 that asset is a different frame family from the vendor's, up to 65 mm out.
  * arm collision spheres: this repository's own cover fit for the arm, and where there is none, Isaac's Lula
    ``{model}_robot_description.yaml`` refined with surface spheres fitted to the committed collision bundle. That
    fallback is the last thing on this path that needs a simulator. A map that does not describe the arm the bundle
    holds is refused rather than filtered (``VENDOR_SPHERE_LIMIT_MM``), which is what ur16e and a ur10 built from
    UR's description both run into.
  * default_q: the retract the rule judged on the exact meshes with every registry hand
    (``src/robot/safety/planning/robot/ur_retract.yaml``). No fallback: an arm with no row stops the build rather
    than inheriting a pose nobody measured.
  * everything else: copied from cuRobo's ``ur10e.yml`` template, which carries the six UR joint and link names
    every model shares and names ``tool0`` as its end effector.

No hand. A descriptor used to carry the hand's sphere map on ``tool0``, one ``{model}_{gripper}.yml`` per pairing,
chosen by ``--gripper`` and ``--coupling-mm``. The hand is a body link the planner sidecar adds when it starts, from
the hand the cell names (``robot.gripper.model``, ``coupling_plates_mm``) and where its declared tool frame puts it
(``src/robot/safety/planning/_curobo_body_links.py``, ``body_link.py``). ``_arm_descriptor.arm_only`` takes tool0 out
of the template's collision links, and both flags are refused by name. The per hand files an earlier build wrote stay
in the content directory, and this script never writes one again.

Arm link surface augmentation runs for every model that has a baked bundle and no cover fit of its own. Each Lula arm
link is augmented with cuRobo surface spheres fitted to ``{model}_collision_meshes.npz`` and placed through that
model's own DH chain, and both halves are per model, so the only thing that ever limits it is which bundles exist.
Worth having: a measured false clear of 8.4% against 5.3% on the ur5e. A model with no bundle keeps its own
model-tuned Lula spheres, which are correct for its link lengths and coarser, and ``ur10`` is permanently in that
state because its Isaac asset collides the whole arm with primitives and has no mesh to fit.

Usage, on the box that owns the cell::

    python scripts/curobo/build_ur_config.py ur3e          # writes ur3e.urdf and willy_ur3e.yml into the content
    python scripts/curobo/build_ur_config.py ur5e          # the ur5e recipe, including arm augmentation

The descriptor is named by the arm alone, ``willy_{model}.yml``, and its ``_provenance`` says ``carries_hand: false``.
A planner refuses one built for another arm, or one that carries a hand, and adds the hand its own cell names.

Run it with the cuRobo environment's interpreter. cuRobo then answers where its own content directory is and nothing
has to be guessed::

    ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py ur5e

Paths are auto-detected and overridable, so no machine-specific path is baked in:
  * ``WILLY_ISAAC_ROOT``       the Isaac Sim install directory, the one thing this repo cannot relocate
  * ``WILLY_ISAAC_MP_CONFIGS`` the Isaac ``motion_policy_configs/universal_robots`` directory
  * ``WILLY_CUROBO_CONTENT``   the cuRobo ``content`` directory, holding ``configs/robot`` and ``assets/robot``

These three are build-time only and are read here and nowhere else. The variables the planner reads at runtime are the
``WILLY_CUROBO_*`` set declared in ``src/robot/safety/planning/environment.py``.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]  # scripts/curobo/this.py, then scripts, then the repository root

# --- 0) model + path resolution -------------------------------------------------------------------------
MODEL = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WILLY_UR_MODEL", "ur3e")).lower()
# One descriptor per arm, and no hand in it. The hand is a body link the planner sidecar adds when it starts, from the
# hand the cell names and where its declared tool frame puts it. A flag that names a hand is refused rather than
# ignored, because a build that asked for a hand and was silently given none reads as one that modelled it.
for _arg in sys.argv[2:]:
    # Both spellings, `--gripper hande` and `--gripper=hande`. A refusal that misses the second writes an arm
    # descriptor while the operator believes they asked for a hand.
    if _arg in ("--gripper", "--coupling-mm") or _arg.startswith(("--gripper=", "--coupling-mm=")):
        raise SystemExit(
            f"{_arg} is refused: this script writes one descriptor per arm, willy_{MODEL}.yml, with no hand in it. The "
            "planner adds the hand a cell names when it starts, from robot.gripper.model and "
            f"robot.gripper.coupling_plates_mm, so build the arm alone: build_ur_config.py {MODEL}"
        )

# How the arm spheres are fitted to the committed bundle, and how far a kept vendor sphere may reach past its link's
# hull. The default recipe is the surface fit and no bound is the bounding box rule, and `_provenance.arm_spheres`
# says which of them a descriptor was written with.
# `--yml-out` writes the descriptor somewhere else, so a candidate can be measured beside the one cells load.
# `--seed` seeds every random draw of the fit and is recorded in `_provenance`, so one recipe and one seed write one
# descriptor: unseeded, two builds of one recipe for ur3e with the Hand-E read 8 and 15 false clears.
ARM_FIT = "surface"
BOUND_MM: "float | None" = None
YML_OUT_OVERRIDE: "Path | None" = None
FIT_SEED = 0
#: Where the description comes from: `ur` renders Universal Robots' own, pinned and vendored, and `isaac` reads a file
#: from a simulator install. See _urdf_source.py for what each promises.
#:
#: `ur` is the default, because it is what a customer box can do: no simulator, and the description is the vendor's own
#: rather than somebody's copy of it. The ready gate over all 18 arm and hand pairs reads identically to every decimal
#: whether the five arms are built from UR's description or Isaac's. ur10 is the one exception, and `install.ps1`
#: passes `--urdf-from isaac` for it by name.
URDF_FROM = "ur"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--seed" and _i + 1 < len(sys.argv):
        FIT_SEED = int(sys.argv[_i + 1])
    elif _arg == "--urdf-from" and _i + 1 < len(sys.argv):
        URDF_FROM = sys.argv[_i + 1].lower()
    elif _arg == "--arm-fit" and _i + 1 < len(sys.argv):
        ARM_FIT = sys.argv[_i + 1]
    elif _arg == "--bound-mm" and _i + 1 < len(sys.argv):
        BOUND_MM = float(sys.argv[_i + 1])
    elif _arg == "--yml-out" and _i + 1 < len(sys.argv):
        YML_OUT_OVERRIDE = Path(sys.argv[_i + 1])

#: Relative to an Isaac install root: where the per-model Lula descriptions live. Stable across Isaac
#: versions; the install root is what differs from box to box, so only that is searched for.
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


def _isaac_model_dir():
    """Isaac's folder for this model, or None where this box has no simulator.

    Resolved lazily, because with --urdf-from ur the description comes from the vendor's own config and
    nothing here needs Isaac until the Lula sphere map is read. A box without a simulator gets a sentence
    naming what is missing rather than a failure at import.
    """
    global _ISAAC_DIR
    if _ISAAC_DIR is _UNRESOLVED:
        try:
            _ISAAC_DIR = _resolve_isaac_mp() / MODEL
        except SystemExit as exc:
            print(f"isaac: not on this box ({exc})")
            _ISAAC_DIR = None
    return _ISAAC_DIR


_UNRESOLVED = object()
_ISAAC_DIR: object = _UNRESOLVED
CUROBO = _resolve_curobo_content()
URDF_OUT = CUROBO / f"assets/robot/ur_description/{MODEL}.urdf"
YML_OUT = YML_OUT_OVERRIDE if YML_OUT_OVERRIDE is not None else CUROBO / f"configs/robot/willy_{MODEL}.yml"
print(f"[paths] model={MODEL}\n  urdf   = {URDF_FROM}\n  curobo = {CUROBO}")

# --- 1) URDF: a description that actually describes a body ------------------------------------------------
# Isaac names every UR description `{model}.urdf` except one: `ur10` ships `ur10_robot.urdf`. That is
# the small half. The large half is that Isaac ships ur10 in two incompatible link frame families, and
# the description sitting beside its Lula sphere map belongs to the wrong one. Pairing them puts 23 of
# 30 sphere centres off the arm, worst 341.5 mm, with nothing raising, so the planner models the arm
# where it is not and leaves unguarded the space where it is.
#
# Two general rules decide it, and neither names a model:
#   1. A description with no geometry describes no body. `ur10_robot.urdf` is 74 lines of pure
#      kinematics: 0 <collision>, 0 <visual>, 0 mesh references. Every description that plans today
#      carries 7 / 7 / 14. That one rule excludes the wrong-family file and changes nothing for the
#      five that already work.
#   2. Its mesh references have to resolve, beside it or under the cuRobo asset root where the descriptor
#      finds them. `ur10_robot_suction.urdf` has geometry and all 14 of its mesh files are missing from
#      disk, so it can supply frames but not a body.
#
# For ur10 those select the URDF-importer package copy, which is the +z-family description whose
# frames match Isaac's own sphere map. The edge that keeps this honest is elsewhere: the description
# written here and the sphere map have to agree about every link.


def _importer_root():
    """The URDF-importer package, which ships a second copy of some robots, or None."""
    for _root in _isaac_roots():
        for _hit in _root.glob("**/isaacsim.asset.importer.urdf/data/urdf/robots"):
            return _hit
    return None


# The check lives beside this script, in `_description_check.py`, so it can be loaded by path. A mesh reference
# resolves beside the description, one folder up, or under the cuRobo asset root, where the descriptor finds it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _arm_spheres import (ArmFitRecipe, CoverMapError, LinkHull,  # noqa: E402
                          arm_spheres_label, cover_spheres_label, filter_spheres, link_frames,
                          link_spheres, place_spheres)

try:
    RECIPE = ArmFitRecipe.parse(ARM_FIT)
except ValueError as _exc:
    raise SystemExit(f"--arm-fit: {_exc}") from _exc
if BOUND_MM is not None and BOUND_MM < 0.0:
    raise SystemExit(f"--bound-mm {BOUND_MM:g}: a reach bound is a distance of 0 or more millimetres")


from _urdf_source import choose as _choose_urdf  # noqa: E402 (beside this script)

_choice = _choose_urdf(
    MODEL,
    source=URDF_FROM,
    isaac_dir=_isaac_model_dir(),
    importer_root=_importer_root(),
    asset_root=CUROBO / "assets/robot/ur_description",
)
print(f"urdf: {_choice.render()}")
_urdf_src = _choice.path
urdf = _choice.text
#: What to call the description in a message, whether it was read from a file or rendered here.
_source_name = _urdf_src.name if _urdf_src is not None else f"{MODEL} as UR describes it"

# One malformed number is the whole "cannot be loaded" blocker. The importer ur10.urdf carries
# `xyz="0 0.0 0.0027000046)"`, with a stray closing parenthesis, and cuRobo's parser (yourdfpy) raises
# ValueError on it. Repaired in the copy written here, never in Isaac's own tree.
_repaired = re.sub(r'(xyz|rpy)="([^"]*)"',
                         lambda m: f'{m.group(1)}="{m.group(2).replace(chr(41), "")}"', urdf)
if _repaired != urdf:
    print(f"urdf: {_source_name} carries a malformed number (a stray parenthesis); repaired in the "
          f"copy written here, not in the source tree")
    urdf = _repaired

urdf = urdf.replace("package://ur_description/", "")  # -> meshes/<model>/... relative to asset_root_path

# UR plans the elbow within +-180 degrees while the arm turns +-360, and a description carries whatever it was shipped
# with: six of them carry UR's limit and ur10 carries the full turn. Inherited, the planner offers an elbow turn no UR
# planner would. The limit is declared in `_planner_limits.py` beside this script, which is loaded by path, and it is
# applied before either write below so the file on disk carries it too.
from _planner_limits import (  # noqa: E402
    ELBOW_LIMIT_RAD,
    POSITION_LIMIT_CLIP_RAD,
    clamp_elbow_limit,
)

urdf, _elbow_clamped = clamp_elbow_limit(urdf)
if _elbow_clamped:
    print(f"elbow: {_source_name} lets the elbow turn past UR's planning limit; clamped to "
          f"+-{ELBOW_LIMIT_RAD:.8f} rad in the copy written here, not in Isaac's tree")

# The meshes travel with the description. A urdf referencing files that are not under the cuRobo
# asset root produces a config that loads and then has no body to check against, and `ur3`, `ur3e`
# and `ur5` have shipped in exactly that state. Anything resolvable beside the source is copied and
# the reference is rewritten to where it landed.
_mesh_dir = URDF_OUT.parent / "meshes" / MODEL
for _ref in sorted(set(re.findall(r'filename="([^"]+)"', urdf))):
    _src_mesh = None
    # A rendered description names the pinned meshes where they already are, under the asset root, so there
    # is nothing beside it to copy. Only a file read from somewhere else brings its meshes with it.
    for _base in () if _urdf_src is None else (_urdf_src.parent, _urdf_src.parent.parent):
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

# --- 1b) tool0, for a description of the older ur_description generation ---------------------------------
# Of the six Isaac UR descriptions, five carry base_link_inertia, flange and tool0, and ``ur10`` carries
# world and ee_link and neither of the other two. The shared cuRobo scaffolding names ``tool0`` as its end
# effector, so a ur10 built from it loads and then dies at the first plan with "Link tool0 not found in
# parent map", late and nowhere near the cause.
#
# Transcribed from a sibling rather than written from memory. The two fixed joints are copied out of a
# description that has them, and a tool frame written from memory passes every extent check while pointing
# the hand somewhere else.
#
# And gated. The transplant is only meaningful if ``wrist_3_link`` is the same frame on both robots. That
# is asserted here rather than assumed: base to wrist_3 at q=0 is computed from each description and the
# rotations have to agree. They are identical on all six, which is what makes this a transcription.
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


# The donor supplies whatever is missing, not only tool0. A description can be the right robot
# in the right frame family and still be incomplete in a way that only shows at load.
def _inertias_are_complete(text: str) -> bool:
    """Every link with an <inertial> also carries an <inertia> tensor.

    The importer ur10.urdf gives every link a <mass> and no tensor. cuRobo reads
    inertia_matrix[0, 0], gets None, and dies with a TypeError naming no link at all, after
    the descriptor has been written and reported as a success.
    """
    for _m in re.finditer(r"<inertial>(.*?)</inertial>", text, re.S):
        if "<inertia " not in _m.group(1):
            return False
    return True


_needs_tool0 = '<link name="tool0"' not in urdf
_needs_inertia = not _inertias_are_complete(urdf)
if (_needs_tool0 or _needs_inertia) and _isaac_model_dir() is None:
    raise SystemExit(
        f"{MODEL}'s description is missing "
        f"{'tool0' if _needs_tool0 else ''}{' and ' if _needs_tool0 and _needs_inertia else ''}"
        f"{'complete inertia' if _needs_inertia else ''}, and there is no sibling description here to "
        f"transcribe it from. A frame written from somewhere else is how a hand ends up pointing "
        f"where it is not."
    )
if _needs_tool0 or _needs_inertia:
    # The donor may be a sibling description of the same robot rather than another model. For ur10
    # the frame-identical `ur10_robot_suction.urdf` carries tool0 92.2 mm out and rotated, where a
    # ur10e donor carries an identity chain, so a ur10e graft would put tool0 92.2 mm short and
    # 90 degrees off. The rotation gate below is what tells them apart; this list only has to offer
    # both, and offering more costs nothing because the gate refuses the wrong ones.
    _donor_dirs = [URDF_OUT.parent, _isaac_model_dir()]
    _donors = sorted(
        {p for d in _donor_dirs for p in d.glob("*.urdf")
         if p.name != URDF_OUT.name and '<link name="tool0"' in p.read_text(encoding="utf-8")},
        key=lambda p: (p.parent != _isaac_model_dir(), p.name),  # same-robot siblings first
    )
    if not _donors:
        raise SystemExit(
            f"{MODEL} has no tool0 and no sibling description in {URDF_OUT.parent} has one to copy from. "
            f"Build any other UR model first (its urdf lands in the same directory), then this one."
        )
    # The gate picks the donor. Same wrist frame, or the transplant authors a tool frame pointing
    # somewhere else, which is a mirrored hand in another costume. Every donor is measured and the
    # first that agrees is used; if none agrees, nothing is written.
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
            f"{MODEL} wrist_3_link is not the frame any available donor uses (closest differs by "
            f"{_worst:.6f}), so the fixed wrist_3 -> flange -> tool0 chain cannot be transcribed onto "
            f"it. Read the tool frame off this description instead of copying one; a tool frame "
            f"written from memory passes every extent check while pointing the hand elsewhere."
        )

    import re as _re
    # `flange` is an intermediate on one generation and absent on the other, so only `tool0` is
    # required. The modern chain is wrist_3 to flange to tool0, with tool0 landing exactly on
    # wrist_3_link; the older one goes wrist_3 to tool0 directly, 92.2 mm out and rotated -90 degrees
    # about x, with no flange at all. Requiring both refuses the one donor whose wrist frame agrees.
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
        # Whole blocks, mass included, because the importer file says the base weighs 200 kg where the
        # donor says 4, and half of one dynamics model plus half of another is worse than either.
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
              f"one to {_worst:.2e}: a copy rather than an invention.")


# --- 2) arm spheres: the committed cover fit, else Isaac's Lula map ----------------------------------------
# The sphere map is this repository's own. It is fitted to the same collision bundle the exact mesh guard
# judges against, by `scripts/curobo/fit_cover_spheres.py`, and what makes it worth having is that the safe
# half is a property of the construction: every surface sample of every link ends up inside a sphere, so
# there is no hole for the planner to walk the arm through, and no sphere reaches further past its own link
# than the reach the map was fitted at.
#
# Measured against Isaac's Lula family, which this replaces: 0 to 61 mm of reach past the links, gaps up to
# 45 mm once clamped, and on ur10 a map belonging to a different link-frame family altogether.
#
# Lula stays as the fallback for an arm with no committed fit, and says so out loud. With a fit present this
# build needs no simulator at all: `--urdf-from ur` renders the vendor's own description, and the spheres
# come from here.
_FITTED_MAP = REPO / f"src/robot/safety/planning/robot/{MODEL}_arm_spheres.yml"
_fitted_doc = None
if _FITTED_MAP.is_file():
    _fitted_doc = yaml.safe_load(_FITTED_MAP.read_text(encoding="utf-8")) or {}
else:
    if _isaac_model_dir() is None:
        raise SystemExit(
            f"{MODEL} has no committed arm sphere map at {_FITTED_MAP.name} and this box has no Isaac motion "
            f"policy configs to fall back to. Fit one: scripts/curobo/fit_cover_spheres.py --arm {MODEL} --write"
        )
    _lula_src = _isaac_model_dir() / f"rmpflow/{MODEL}_robot_description.yaml"
    if not _lula_src.is_file():
        raise SystemExit(f"missing Isaac Lula robot description: {_lula_src}")
    lula = yaml.safe_load(_lula_src.read_text(encoding="utf-8"))
# The retract comes from the committed table rather than from Isaac's Lula pose. cuRobo biases its graph search and
# its IK regularisation towards this pose and warms up from it, and Isaac's sits inside the guard's own refusal band
# on two arms, where ur5 keeps 6.9 mm to the EGU-50. The table is chosen by a rule that judges both the exact meshes
# and the planner's own sphere model: scripts/curobo/choose_ur_retract.py.
#
# What goes in here is a fallback. The retract belongs to the arm and the hand together, and a descriptor is per arm:
# ur3 keeps its anchor with the 2F-85 and needs a pose three steps away with the Hand-E, because the sphere model
# refuses the first one with that hand on. A cell's driver reads its own pair's row and hands it to the sidecar,
# which writes it into the config it loads. What a descriptor carries is what a sidecar with no hand would start
# from, and no real cell is in that state.
#
# The reader lives beside the table, because three interpreters read that file and none can import the other
# two. Loaded by path here for the same reason the DH table is inlined.
import importlib.util as _importlib_util  # noqa: E402

_table_module_path = REPO / "src/robot/safety/planning/robot/retract_table.py"
_table_spec = _importlib_util.spec_from_file_location("willy_retract_table", _table_module_path)
_retract_table = _importlib_util.module_from_spec(_table_spec)
sys.modules["willy_retract_table"] = _retract_table
_table_spec.loader.exec_module(_retract_table)

# The seed build, and why it cannot mislead a cell. The retract chooser asks a real planner whether a pose
# clears, which means spawning a sidecar, which means a descriptor, so an arm with no judged row cannot get
# one, cannot be judged, and stays out of the family for good. `--retract-from-anchor` breaks that circle by
# writing the arm's anchor, which is Isaac's default_q, as the fallback, and saying so in the provenance and
# on stdout.
#
# It is a flag and never a silent default, because a safety pose nobody chose is not a safety pose. What
# keeps it honest is the other side: a cell's driver reads its own pair's row from the table and refuses
# before a sidecar is spawned, so a seed descriptor's fallback is never what a cell holds.
_SEED_RETRACT = "--retract-from-anchor" in sys.argv
try:
    _RETRACT = _retract_table.read_arm_retract(_retract_table.TABLE_PATH, MODEL)
    default_q = list(_RETRACT.retract)
except _retract_table.RetractMissing as exc:
    if not _SEED_RETRACT:
        raise SystemExit(
            f"{MODEL}: {exc}\n"
            f"  If you are building this only so the chooser can ask its planner, say so: "
            f"--retract-from-anchor writes the anchor and marks the descriptor unjudged."
        ) from None
    _RETRACT = None
    # The anchors live with the rule that uses them, loaded by path for the same reason the table
    # reader is: stdlib plus PyYAML, and three interpreters that cannot import one another.
    _rule_spec = _importlib_util.spec_from_file_location(
        "willy_retract_rule", Path(__file__).resolve().parent / "_retract_rule.py")
    _rule = _importlib_util.module_from_spec(_rule_spec)
    sys.modules["willy_retract_rule"] = _rule  # dataclasses resolves annotations through sys.modules
    _rule_spec.loader.exec_module(_rule)
    _anchor = _rule.ANCHORS.get(MODEL)
    default_q = list(_anchor) if _anchor else None
    if default_q is None:
        raise SystemExit(
            f"{MODEL} has no judged retract AND no anchor to seed from, so there is no pose to write at all."
        ) from None
    print(f"!! retract: not judged. This is a seed build: default_q is {MODEL}'s anchor {default_q}, written "
          f"so scripts/curobo/choose_ur_retract.py {MODEL} can ask this arm's planner. A cell reads its "
          f"pair's row from the table and refuses before a sidecar starts, so this pose never reaches one.")
if _RETRACT is not None:
    print(f"retract (fallback for a sidecar with no hand): {default_q}, "
          f"{'the anchor' if not any(_RETRACT.steps) else 'moved by steps ' + str(_RETRACT.steps)}; "
          f"the pairs this table judged for {MODEL}: {', '.join(sorted(_RETRACT.hands))}")
sphere_map: dict = {}
if _fitted_doc is not None:
    try:
        # Still in the bundle's DH frame here. It is placed into each link's URDF frame below, where the DH
        # table has been checked against UR's own description; doing it here would use an unchecked table.
        sphere_map = link_spheres(_fitted_doc, source=_FITTED_MAP.name)
    except CoverMapError as exc:
        raise SystemExit(str(exc)) from None
    _rows = (_fitted_doc.get("_provenance") or {}).get("bodies") or []
    _worst = max((float(row.get("fresh_reach_max_mm", 0.0)) for row in _rows), default=0.0)
    _FIT_FRAMES = {f"{body}_link": int(block["frame"])
                   for body, block in (_fitted_doc.get("collision_spheres") or {}).items()}
    ARM_SPHERES_LABEL = cover_spheres_label(
        source=_FITTED_MAP.name, reach_mm=_worst, spheres=sum(len(v) for v in sphere_map.values()))
    print(f"arm spheres: the committed cover fit, {_FITTED_MAP.name}, reaching at most {_worst:.1f} mm past "
          f"the links and leaving no hole")
else:
    for entry in lula["collision_spheres"]:  # Lula = list of single-key {link: [ {center,radius}, ... ]}
        for link, spheres in entry.items():
            sphere_map[link] = [{"center": [float(c) for c in s["center"]], "radius": float(s["radius"])}
                                for s in spheres]
    ARM_SPHERES_LABEL = None  # section 2b decides; this arm is on the vendor's map
    print(f"!! arm spheres: no committed fit for {MODEL}, falling back to Isaac's Lula map. That map is "
          f"measured to reach up to 61 mm past its own links and to leave gaps; fit one with "
          f"scripts/curobo/fit_cover_spheres.py --arm {MODEL} --write")

# --- 2a) no hand ---------------------------------------------------------------------------------------------
# Nothing is put on tool0. The hand a cell names becomes a body link of the planner's robot when the sidecar starts,
# placed by the cell's declared tool frame and plates, so one arm descriptor serves every hand.

# --- 2b) arm link surface augmentation: the vendor's map only ---------------------------------------------
# Skipped entirely when the committed cover fit above supplied the map. That fit already covers every
# surface sample of every link by construction, so there is nothing for a surface augmentation to add but
# spheres, and with them reach and plan time. This whole section repairs a map somebody else's tool wrote,
# and it only runs when that map is what was read.
# cuRobo surface spheres fitted to this arm's own collision meshes, placed through this arm's own DH chain.
# Worth having: a measured false clear of 8.4% against 5.3% on the ur5e, against the Lula spheres alone.
#
# The gate is the directory and not a model name. Both halves are per model, so the only thing that ever
# limits this is which bundles exist; a gate on a model name says the bundle and the DH placement belong
# to one arm, and it has no edge to the next bake.
#
# A model with no bundle (ur10, whose asset collides with primitives) keeps its own Lula arm spheres,
# which are model-tuned and correct, just coarser.
_ARM_NPZ = REPO / f"src/robot/safety/data/{MODEL}_collision_meshes.npz"
#: Whether the arm spheres were fitted to the bundle, and cuRobo's own metrics per link for a MorphIt fit. Both
#: reach `_provenance`, so a descriptor says what its arm spheres are rather than what its model's name suggests.
# DH (a, d, alpha), inlined rather than imported from the package. This module runs in the cuRobo sidecar
# environment, which is Python 3.10, where `src.robot...` is un-importable because it uses enum.StrEnum, and
# the broad except below would swallow that ImportError and silently emit a Lula-only yml, giving up the
# measured false clear of 8.4% against 5.3% that the arm augmentation buys.
#
# It is the third copy of this table (safety/_ur_kinematics.py and scripts/isaac/bake_ur_meshes_from_urdf.py
# hold the others), each duplicated for the same reason: three interpreters that cannot import each other.
# All three are compared against one another, because three copies of a safety-critical table with nothing
# between them is how one of them ends up quietly wrong.
#
# It is looked up here, outside the try below. Inside it, a model with a bundle and no row raises a KeyError
# the except swallows, and the build then writes a Lula-only descriptor: an arm silently guarded by the
# vendor's spheres alone, which is the one outcome this table exists to prevent.
#: How far a vendor sphere this builder keeps may sit outside the mesh baked for its own link, in millimetres.
#: Measured across the family: 0.0 on ur3, ur3e, ur5 and ur5e, 11.6 on ur10, 13.0 on ur10e, and 60.8 on
#: ur16e, whose map describes the longer ur10e upper arm.
VENDOR_SPHERE_LIMIT_MM = 60.0

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
    "ur16e": [(0.0, 0.1807, 1.570796327), (-0.4784, 0.0, 0.0), (-0.36, 0.0, 0.0),
              (0.0, 0.17415, 1.570796327), (0.0, 0.11985, -1.570796327), (0.0, 0.11655, 0.0)],
}
_DH = _DH_TABLES[MODEL]

# And it is held against UR's own description rather than trusted: the rows above are a transcription, and a
# transcription nobody compares is a number that drifts. Outside the try for the same reason as the lookup.
try:
    import _ur_description as _urd  # noqa: E402 (beside this script, stdlib plus PyYAML)
except Exception as _exc:  # noqa: BLE001 (said out loud; a silent skip is how the copies drift apart)
    print(f"[dh] UR's description is unreadable here ({type(_exc).__name__}: {_exc}), so the inlined rows "
          f"are used unchecked", flush=True)
else:
    _rows = _urd.dh_rows(MODEL)
    assert np.allclose(np.asarray(_DH, dtype=float), np.asarray(_rows, dtype=float), rtol=0.0, atol=1e-12), (
        f"the DH table inlined in this builder disagrees with UR's own description for {MODEL}: "
        f"{_DH} against {_rows}"
    )

# The committed fit is in the bundle's DH frame and a descriptor is in the URDF link frame. They are not the
# same frame, they differ per link, and nothing downstream measures where a sphere landed. On the ur5e the
# upper arm's two frames sit 425 mm apart along the link, which is the link's own length, and the shoulder's
# differ by a quarter turn about x. A build that skips this placement writes a descriptor whose upper arm
# spheres are 425 mm from the upper arm, and it loads, plans, and reports ready for every arm and hand pair.
# The one thing that catches it is the planner's own boot table: a displaced arm dips under the slab the
# sidecar stands on, and the refusal says so.
if _fitted_doc is not None:
    _place = link_frames(urdf, _DH)
    sphere_map = {link: place_spheres(spheres, _place(link, _FIT_FRAMES[link]))
                  for link, spheres in sphere_map.items()}
    print(f"arm spheres placed from the bundle's DH frames into the URDF link frames of {MODEL}")

_augmented = False
_fit_metrics: dict = {}
if _fitted_doc is not None:
    print("arm-link surface augmentation skipped: the committed cover fit already covers every surface "
          "sample of every link, so there is nothing for it to add but spheres, reach and plan time")
elif _ARM_NPZ.is_file():
    try:
        import trimesh  # type: ignore[import-not-found]
        from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh  # type: ignore[import-not-found]

        # The fit samples through trimesh's shared generator and torch's CUDA generator, and cuRobo seeds
        # neither, so two builds of one recipe differ: ur3e with the Hand-E reads 8 against 15 false clears
        # at 4 mm. With no seed passed, trimesh draws from `util._RANDOM_DEFAULT`, which numpy.random.seed
        # does not reach, so that generator itself is replaced.
        import torch  # type: ignore[import-not-found]
        import trimesh.util as _trimesh_util  # type: ignore[import-not-found]

        _trimesh_util._RANDOM_DEFAULT = np.random.default_rng(FIT_SEED)
        torch.manual_seed(FIT_SEED)
        torch.cuda.manual_seed_all(FIT_SEED)

        # The arm bundle, which carries no hand this build reads: these are this model's own link
        # meshes, baked from its own Isaac asset by scripts/isaac/bake_ur_collision_meshes.py and
        # gated there against the committed ur5e.
        _MESH = np.load(_ARM_NPZ, allow_pickle=True)

        # DH (a, d, alpha), inlined rather than imported from the repository. This branch only runs in
        # the cuRobo sidecar environment, which is Python 3.10, where `src.robot...` is un-importable
        # because it uses enum.StrEnum, and the bare except below would swallow that ImportError and
        # silently emit a Lula-only yml, giving up the measured false clear of 8.4% against 5.3% that
        # the arm augmentation buys.
        #
        # It is the third copy of this table (safety/_ur_kinematics.py and
        # scripts/isaac/bake_ur_collision_meshes.py hold the others), each duplicated for the same
        # reason: three interpreters that cannot import each other. All three are compared against one
        # another, because three copies of a safety-critical table with nothing between them is how one
        # of them ends up quietly wrong.
        def _dh_T(row: tuple) -> np.ndarray:
            a, d, al = row
            ca, sa = np.cos(al), np.sin(al)
            return np.array([[1.0, 0.0, 0.0, a], [0.0, ca, -sa, 0.0], [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]])

        def _dh_frames_m() -> list:
            T = np.eye(4)
            out = [T.copy()]
            for row in _DH:
                T = T @ _dh_T(row)
                out.append(T.copy())
            return out

        def _aa(axis: tuple, th: float) -> np.ndarray:
            a = np.asarray(axis, float)
            a = a / (np.linalg.norm(a) or 1.0)
            c, s = np.cos(th), np.sin(th)
            x, y, z = a
            return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                             [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                             [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])

        def _urdf_link_frame():
            import xml.etree.ElementTree as ET
            root = ET.fromstring(urdf)
            kids = {}
            for jt in root.findall("joint"):
                o = jt.find("origin")
                rpy = [float(x) for x in o.attrib.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
                xyz = [float(x) for x in o.attrib.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
                M = np.eye(4)
                M[:3, :3] = _aa((0, 0, 1), rpy[2]) @ _aa((0, 1, 0), rpy[1]) @ _aa((1, 0, 0), rpy[0])
                M[:3, 3] = xyz
                kids[jt.find("child").attrib["link"]] = (jt.find("parent").attrib["link"], M)
            base_link = next(iter({v[0] for v in kids.values()} - set(kids)))

            def frame(link: str) -> np.ndarray:
                if link == base_link:
                    return np.eye(4)
                p, M = kids[link]
                return frame(p) @ M
            return frame

        def _declared_primitives(text, link_name):
            """Every <collision> primitive a link declares, as (centre, axis, half_length, radius).

            In the link frame, which is the frame the sphere map is written in, so these need no
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
        # The half turn is not a convention here, it is a fact about this URDF: base_link_inertia is mounted
        # yawed by pi, and that is the whole reason every placement below is inv(urdf) @ Rz(pi) @ Tdh. A URDF
        # without that yaw would place every link mesh half a turn out, and nothing downstream measures orientation.
        #
        # Checked where the link exists. Isaac's importer ur10.urdf declares none, because that asset is the
        # older ur_description generation, which is also why it has no tool0 and why its numbers carry a stray
        # parenthesis. Rendering ur10 from UR's own description (--urdf-from ur) brings the link, and the check.
        try:
            _base_inertia = _urdf_frame("base_link_inertia")
        except KeyError:
            print("base frame: this URDF declares no base_link_inertia, so the half turn is taken on trust here")
        else:
            assert np.allclose(_base_inertia[:3, :3], _Rz_pi[:3, :3], rtol=0.0, atol=1e-12), (
                "this URDF does not yaw base_link_inertia by pi, so the DH placement below would be half a turn out"
            )
        _ARM_FIT = {"shoulder": ("shoulder_link", 1, 12), "upper_arm": ("upper_arm_link", 2, 22),
                    "forearm": ("forearm_link", 3, 22), "wrist_1": ("wrist_1_link", 4, 10),
                    "wrist_2": ("wrist_2_link", 5, 10), "wrist_3": ("wrist_3_link", 6, 6)}
        # A link the vendor map skips is authored rather than augmented, and the two are different jobs.
        # Isaac's ur10 Lula description covers five links where every sibling covers six, and the missing
        # one is shoulder_link. For the five, these spheres refine a map that already covers the body.
        # For an absent link they would be the only spheres, so a count tuned to add surface detail
        # leaves a shell with gaps, and a gap in a collision proxy is a pose the guard clears and should
        # refuse. So an authored link gets a volume fit at a higher count, and its coverage is measured
        # against a link that does have vendor spheres on this same robot in this same run, because a
        # number with no comparator is not evidence.
        # A vendor sphere that does not touch its own link is not that link's sphere. Isaac's ur10 Lula
        # map puts all three wrist links' spheres about 61 mm from where that link's geometry is
        # (wrist_1 mesh y[60.8, 160.7] with spheres at y = 0, and the same on the other two). The bundle
        # is not the wrong half: through the DH chain the baked arm is connected, every gap between
        # consecutive links under 3.3 mm against 0.2 to 0.8 mm on ur10e.
        #
        # Keeping both sets makes each wrist a body twice its size spanning two positions, so cuRobo
        # finds every configuration in collision and returns None from plan_pose and plan_cspace alike,
        # including a plan from a pose to itself. Such a descriptor loads perfectly and plans nothing.
        #
        # The threshold is not tuned: a sphere whose centre is further outside its link's own mesh
        # than its own radius does not intersect that body at all. ur10's wrists fail it at 61 mm
        # with r 45 to 50, ur10e's worst link passes at 13 mm with r 60, and every other arm is at 0.0 mm.
        _stray = {}
        for _mesh_key, (_L, _f, _n) in _ARM_FIT.items():
            if _L not in sphere_map or f"{_mesh_key}__v" not in _MESH:
                continue
            _mv = np.asarray(_MESH[f"{_mesh_key}__v"], float) / 1000.0
            _MX = np.linalg.inv(_urdf_frame(_L)) @ _Rz_pi @ _Tdh[_f]
            _mv = (_MX[:3, :3] @ _mv.T).T + _MX[:3, 3]
            if BOUND_MM is not None:
                # A vendor sphere is judged by how far it reaches past the hull of its own link
                # (_arm_spheres.py), because a sphere centred inside the bounding box passes the rule below
                # however far it reaches, which on ur10e is up to 79.6 mm past the hull.
                _keep, _dropped = filter_spheres(sphere_map[_L], LinkHull.from_vertices(_mv),
                                                 bound_m=BOUND_MM / 1000.0)
                _drop = len(_dropped)
            else:
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
        # And a map that survives the filter can still be the wrong arm's. The filter above drops a sphere
        # that does not touch its own link at all, and it cannot see a map that is merely the wrong size:
        # all ten of Isaac's ur16e upper arm spheres are kept and the worst sits 60.8 mm outside that arm's
        # own baked mesh, because the map describes the longer ur10e upper arm. Such a descriptor loads,
        # plans, and guards a body where the body is not.
        #
        # 60 mm is the rule the whole family is measured against: 0.0 mm on ur3, ur3e, ur5 and ur5e, 11.6 mm
        # on ur10 and 13.0 mm on ur10e, where the vendor fits spheres to the visual hull and this bundle
        # holds the collision meshes. A build outside it is refused rather than written.
        _far = {}
        for _mesh_key, (_L, _f, _n) in _ARM_FIT.items():
            if _L not in sphere_map or f"{_mesh_key}__v" not in _MESH:
                continue
            _mv = np.asarray(_MESH[f"{_mesh_key}__v"], float) / 1000.0
            _MX = np.linalg.inv(_urdf_frame(_L)) @ _Rz_pi @ _Tdh[_f]
            _mv = (_MX[:3, :3] @ _mv.T).T + _MX[:3, 3]
            _lo, _hi = _mv.min(0), _mv.max(0)
            _c = np.asarray([_sp["center"] for _sp in sphere_map[_L]], float)
            _out_mm = float(np.max(np.maximum(np.maximum(_lo - _c, _c - _hi), 0.0).sum(axis=1))) * 1000.0
            if _out_mm > VENDOR_SPHERE_LIMIT_MM:
                _far[_L] = _out_mm
        if _far:
            _worst = ", ".join(f"{_k} {_v:.1f} mm" for _k, _v in sorted(_far.items()))
            raise SystemExit(
                f"the vendor sphere map for {MODEL} does not describe this arm: {_worst} outside the "
                f"mesh baked for the same link, against a limit of {VENDOR_SPHERE_LIMIT_MM:g} mm. "
                f"A descriptor built from it would guard bodies where the bodies are not, and would "
                f"load and plan perfectly while doing it. Refit this arm's spheres to its own baked "
                f"mesh rather than inheriting another arm's map."
            )
        for _L, (_d, _t) in _stray.items():
            if BOUND_MM is not None:
                print(f"vendor spheres: dropped {_d} of {_t} on {_L}; they reach more than {BOUND_MM:g} mm past "
                      f"the hull of this arm's own baked mesh")
            else:
                print(f"vendor spheres: dropped {_d} of {_t} on {_L}; their centres sit further outside "
                      f"this arm's own baked mesh than their own radius, so they are not on this link")

        _added, _authored, _coverage = 0, [], {}
        for _mesh, (_L, _f, _n) in _ARM_FIT.items():
            _v = np.asarray(_MESH[f"{_mesh}__v"], float) / 1000.0
            _faces = np.asarray(_MESH[f"{_mesh}__f"], np.int64)
            _tm = trimesh.Trimesh(vertices=_v, faces=_faces, process=False)
            _X = np.linalg.inv(_urdf_frame(_L)) @ _Rz_pi @ _Tdh[_f]
            # A link with no vendor spheres is authored from its declared primitives for the surface fit, whose
            # shell of 10 mm spheres leaves gaps. A MorphIt fit fills the volume itself, so under it such a link
            # (every vendor sphere past the bound, or ur10's shoulder that Lula skips) is fitted like any other.
            _is_authored = _L not in sphere_map and RECIPE.fit_type == "surface"
            sphere_map.setdefault(_L, [])
            if _is_authored:
                # Authored from the declared primitives rather than from a fit. A surface fit places
                # fixed 10 mm spheres, so 100 of them cover 25.6 % of this link, and a voxel fit places
                # large spheres that never reach the surface and cover 0.0 %. A real vendor map is few
                # spheres, large, along the link axis: ur10e's shoulder is two spheres of r 83.5 mm at
                # 31.4 % coverage, and its weakest link is 10.2 %.
                #
                # The description already declares exactly that shape. Spheres along the two collision
                # cylinders it gives this link score 48.7 %, better than ur10e's own shoulder. And a
                # <collision> origin is already in the link frame, which is the frame the sphere map is
                # written in, so nothing is transformed and nothing can be transformed wrongly.
                _prims = _declared_primitives(urdf, _L)
                if not _prims:
                    raise SystemExit(
                        f"{MODEL}/{_L} has no vendor spheres and no declared collision primitive to "
                        f"author them from, so this link would be unguarded. Nothing written."
                    )
                sphere_map[_L] = []
                _authored.append(_L)
                for _centre, _axis, _half, _rad in _prims:
                    # Spaced at half a radius, because the coverage curve is flat past that (48.7 % at
                    # 0.5 against 49.9 % at 0.25), so more spheres buy nothing but planner cost.
                    _n_along = max(2, int(np.ceil(2.0 * _half / max(_rad * 0.5, 1e-6))) + 1)
                    for _t in np.linspace(-_half, _half, _n_along):
                        _pt = _centre + _t * _axis
                        sphere_map[_L].append({"center": [round(float(x), 4) for x in _pt],
                                               "radius": round(float(_rad), 4)})
                        _added += 1
            else:
                if RECIPE.fit_type == "surface":
                    _res = fit_spheres_to_mesh(_tm, num_spheres=_n, surface_radius=0.010,
                                               fit_type=SphereFitType.SURFACE)
                else:
                    # cuRobo's optimised fit. Its count follows the density, its reach past the mesh is
                    # penalised by the weight, and its own metrics per link go into the provenance.
                    _res = fit_spheres_to_mesh(_tm, sphere_density=float(RECIPE.density),
                                               protrusion_weight=float(RECIPE.protrusion_weight),
                                               fit_type=SphereFitType.MORPHIT, compute_metrics=True)
                    if _res.metrics is not None:
                        _fit_metrics[_L] = {
                            "spheres": int(_res.metrics.num_spheres),
                            "coverage": round(float(_res.metrics.coverage), 3),
                            "protrusion_p95_mm": round(float(_res.metrics.protrusion_dist_p95) * 1000.0, 1),
                            "surface_gap_p95_mm": round(float(_res.metrics.surface_gap_p95) * 1000.0, 1),
                        }
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

            # Coverage, in the bundle's own frame so the placement above cannot flatter it: what
            # fraction of this link's surface lies inside some sphere of its final map.
            _pts = _tm.sample(4000) if len(_tm.faces) else np.zeros((0, 3))
            if len(_pts):
                _all = [(np.asarray(sp["center"], float), float(sp["radius"])) for sp in sphere_map[_L]]
                _back = np.linalg.inv(_X)
                _inside = np.zeros(len(_pts), dtype=bool)
                for _sc, _sr in _all:
                    _local = (_back[:3, :3] @ np.asarray(_sc) + _back[:3, 3])
                    _inside |= np.linalg.norm(_pts - _local, axis=1) <= _sr
                _coverage[_L] = float(_inside.mean())

        _augmented = True
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
        # enforcement: nothing downstream can tell an augmented descriptor from a Lula-only one.
        print("\n" + "!" * 100)
        print(f"!! {MODEL} arm-link surface augmentation skipped ({type(exc).__name__}: {exc})")
        print(f"!! The emitted {MODEL}.yml is Lula-only: it gives up the measured false clear of 8.4% against 5.3%.")
        print("!! Re-run under the cuRobo env python (needs curobo.sphere_fit + trimesh) before shipping it.")
        print("!" * 100 + "\n")
else:
    print(f"arm-link surface augmentation skipped for {MODEL}: no {_ARM_NPZ.name} to fit. {MODEL} keeps "
          f"its own model-tuned Lula arm spheres, which are correct for its link lengths and coarser. "
          f"Bake one with scripts/isaac/bake_ur_collision_meshes.py, unless that refuses: a model "
          f"whose asset collides with primitives has no exact geometry to fit and never will.")

print(f"converted spheres for {len(sphere_map)} links: {list(sphere_map)}  default_q={default_q}")

# --- 3) cuRobo ur10e.yml as the template; swap urdf + spheres + default config ----------------------------
# The shared UR scaffolding: joint names, link names, ee_link, the dynamics block. Read from
# cuRobo's own shipped ur10e config, which is a template here and not this repository's ur10e.
#
# A copy is taken on first use, because the template and the output can be the same path. A build that
# wrote its result over the pristine template would leave every later build of any model reading a file
# that is already a product, and the first such build looks perfect. The copy is what every build reads.
_pristine = CUROBO / "configs/robot/ur10e.yml"
_tpl = CUROBO / "configs/robot/_ur_template.yml"
if not _tpl.is_file() and _pristine.is_file():
    _tpl.write_text(_pristine.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"template: took a copy of {_pristine.name} as {_tpl.name}, so a ur10e build cannot "
          f"overwrite the scaffolding every model reads")
if not _tpl.is_file():
    _tpl = _pristine
if not _tpl.is_file():
    raise SystemExit(f"missing cuRobo template {_tpl} (needed for the shared UR joint/link/ee scaffolding)")
cfg = yaml.safe_load(_tpl.read_text(encoding="utf-8"))
cfg["robot_cfg"]["kinematics"]["urdf_path"] = f"robot/ur_description/{MODEL}.urdf"
cfg["robot_cfg"]["kinematics"]["collision_spheres"] = sphere_map
# One descriptor per arm. The template guards tool0, where a descriptor used to carry the hand, and cuRobo raises on a
# collision link with no spheres, so tool0 leaves the collision links and the buffer table before the check below
# counts what is guarded (`_arm_descriptor.py`, beside this script for the same reason as the DH table).
from _arm_descriptor import ArmDescriptorError, arm_only  # noqa: E402

try:
    cfg = arm_only(cfg)
except ArmDescriptorError as exc:
    raise SystemExit(f"{MODEL}: {exc}") from None
kin = cfg["robot_cfg"]["kinematics"]

# Every link the template guards must have spheres, checked here rather than at the first plan.
# The template declares seven collision links and the ur10 Lula description covers six: Isaac ships
# no shoulder_link spheres for that model and for no other. cuRobo then raises KeyError on
# shoulder_link the first time anything loads the file, with every step before it reporting success,
# so a descriptor probe reads the path, finds a file, and says ok all the way to the first motion.
#
# The near miss is the worse half. Where the template does not happen to name the link, this writes
# an arm with no collision spheres on that link and plans against it in silence.
_guarded = set(kin.get("collision_link_names") or [])
# Count the spheres, not the keys. `_guarded - set(sphere_map)` is passed by an empty list: the key
# is present and the link is unguarded, which is what a fit that raises after creating its key
# leaves behind. A mechanism that does not fire looks exactly like a mechanism with nothing to catch.
_unarmed = sorted(_l for _l in _guarded if not sphere_map.get(_l))
if _unarmed:
    raise SystemExit(
        f"{MODEL}: the template guards {sorted(_guarded)} and this build has spheres for "
        f"{sorted(k for k, v in sphere_map.items() if v)}. Missing or empty: {_unarmed}.\n"
        f"  cuRobo raises KeyError on the first load, so no file is written rather than one that\n"
        f"  fails later. A link with no spheres would not be checked for collision at all.\n"
        f"  Isaac ships no Lula spheres for {MODEL}/{_unarmed}; the other UR models all have them.\n"
        f"\n"
        f"  For ur10 specifically, generating the missing spheres would not be enough, and this is the\n"
        f"  deeper reason it has no descriptor. Isaac ships ur10 in two incompatible link-frame families.\n"
        f"  Its Lula sphere map belongs to the +z family; ur10_robot.urdf beside it and cuRobo own shipped\n"
        f"  ur_description/ur10.urdf are both -x family. Every other UR is -x on both sides, so ur10 is the\n"
        f"  one model where the obvious pairing is the wrong one.\n"
        f"\n"
        f"  Measured through the full FK chain, as the signed distance of each sphere centre to\n"
        f"  the arm body, with two working pairings as calibration:\n"
        f"      sphere map + curobo ur10.urdf     23 of 30 outside, worst +341.5 mm   <- the obvious one\n"
        f"      sphere map + importer ur10.urdf    6 of 30 outside, worst  +38.3 mm\n"
        f"      ur10e, which works today           5 of 33 outside, worst  +45.3 mm\n"
        f"      ur5e,  which works today           4 of 39 outside, worst  +51.0 mm\n"
        f"\n"
        f"  So the failure mode is fail-open by 341.5 mm: a planner would model the arm where it is not\n"
        f"  and leave unguarded the space where it is, and nothing would raise.\n"
        f"\n"
        f"  The importer urdf at isaacsim.asset.importer.urdf/data/urdf/robots/ur10/urdf/ur10.urdf is the\n"
        f"  right frame family, and it is still not a route to a descriptor: cuRobo parser (yourdfpy)\n"
        f"  refuses to load it, it has no tool0 and the tool0 graft gate above correctly rejects it (its\n"
        f"  wrist_3 rotation differs from the donor by 1.0 against a 1e-6 tolerance), and neither Lula file\n"
        f"  has shoulder_link spheres. The frame-identical ur10_robot_suction.urdf does carry tool0, and all\n"
        f"  14 of its meshes are missing from disk. See docs/runbooks/ur_family_bringup.md for what a real\n"
        f"  fix would have to author.\n"
        f"\n"
        f"  That comparison holds for the five models where both halves exist. All five agree, and ur10 is\n"
        f"  the only one that does not."
    )
kin.pop("usd_path", None)  # the ur10e usd is wrong for this model
if "dynamics" in cfg["robot_cfg"]:
    cfg["robot_cfg"]["dynamics"].pop("neural_inverse_dynamics_state_dict", None)
# Fix the upstream ur10e typo: self_collision_ignore has "forarm_link", which should be "forearm_link".
# With the dense Lula spheres the misspelled key never ignores the forearm to wrist_1 pair, so every
# configuration self-collides and plan_pose returns None.
sci = kin.get("self_collision_ignore", {})
if "forarm_link" in sci:
    sci["forearm_link"] = sorted(set(sci.get("forearm_link", []) + sci.pop("forarm_link")))
    print(f"  fixed the self_collision_ignore typo; forearm_link is now {sci['forearm_link']}")
cs = kin.get("cspace", {})
if "default_joint_position" in cs:
    print(f"  default_joint_position {cs['default_joint_position']} becomes {default_q}")
    cs["default_joint_position"] = default_q
# What keeps the planner inside the guard, written rather than inherited. cuRobo narrows every joint by this when it
# loads (0.1 rad is 5.73 degrees), and the guard refuses outside the factory envelope less its margin_deg, 5 degrees by
# default. The one number is wider than the other, and that is the whole reason the planner cannot offer a
# configuration the guard then refuses. Inheriting it from whichever template the content holds is not a guarantee.
cs["position_limit_clip"] = POSITION_LIMIT_CLIP_RAD

# --- provenance: which arm, and that no hand is in it -------------------------------------------
# Without this the descriptor carries the spheres and drops every statement about where they came
# from. `carries_hand: false`, set by arm_only, is what a planner start checks before it adds the
# hand a cell names, so a per hand file from an earlier build is never taken for an arm. cuRobo
# ignores unknown top-level keys, so this costs the planner nothing.
cfg["_provenance"] = {
    **cfg.get("_provenance", {}),
    "arm": MODEL,
    # What this descriptor lets the planner do, so a reader does not have to parse the URDF to find out.
    "planner_joint_limits": {"elbow_joint": [-ELBOW_LIMIT_RAD, ELBOW_LIMIT_RAD]},
    "position_limit_clip": POSITION_LIMIT_CLIP_RAD,
    "elbow_clamped": bool(_elbow_clamped),
    # Which pose this descriptor starts from, and what it was judged against: an arm descriptor carries no hand, so
    # the retract has to clear every hand a cell could name, and the row says which ones were measured.
    "retract": {
        "steps": list(_RETRACT.steps),
        "anchor": list(_RETRACT.anchor),
        "anchor_source": _RETRACT.anchor_source,
        "judged_hands": {hand: plates for hand, plates in sorted(_RETRACT.hands.items())},
        "table_sha256": _RETRACT.table_sha256,
    } if _RETRACT is not None else {
        # A seed build (--retract-from-anchor): the pose is this arm's anchor and no judge has seen it. Said
        # here so a reader of the descriptor learns it from the descriptor, not from the shell it was built in.
        "judged_hands": {},
        "anchor": list(default_q),
        "anchor_source": "seed build: the arm's anchor, unjudged, so the chooser can ask this planner",
    },
    # What the arm spheres are, from what this build ran rather than from the model's name: a label derived from the
    # name says "lula" for every arm but the one it was written for, while every arm with a bundle is fitted.
    "arm_spheres": ARM_SPHERES_LABEL or arm_spheres_label(
        bundle=_augmented, recipe=RECIPE.render(), bound_mm=BOUND_MM),
    "arm_fit": RECIPE.render(),
    "fit_seed": FIT_SEED,
    "arm_vendor_bound_mm": BOUND_MM,
    **({"arm_fit_metrics": _fit_metrics} if _fit_metrics else {}),
    "generated_by": "scripts/curobo/build_ur_config.py",
}

YML_OUT.parent.mkdir(parents=True, exist_ok=True)
YML_OUT.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
print(f"wrote {YML_OUT}")
print(f"  provenance: {MODEL}, no hand, arm spheres {cfg['_provenance'].get('arm_spheres')}")
print(f"  joint order check: cspace.joint_names={cs.get('joint_names')}")
print(f"\nNext: check it plans with a hand added as a body link: "
      f"scripts/curobo/check_ur_descriptors.py {YML_OUT.stem} --hand <robot.gripper.model>")
