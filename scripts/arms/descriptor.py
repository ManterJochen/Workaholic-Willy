"""Assemble a cuRobo robot config for any bundled UR model from the ingredients already on the box.

This is the one script that builds a UR. Willy drives more than one of them, and a descriptor is named by the arm
alone, ``willy_{model}.yml``. That file is not shipped with the repository: it is assembled here, once per box and per
arm, and written into the gitignored cuRobo content directory. A fresh cuRobo clone always needs this step.

Ingredients, and where each comes from:
  * kinematics: Universal Robots' own description, rendered here from their vendored config (``--urdf-from ur``,
    the only source since P0f F5). Written into the cuRobo content with relative mesh paths. Read from an Isaac
    install instead, a customer box could not build a descriptor at all, and for ur10 that asset was a different
    frame family from the vendor's, up to 65 mm out.
  * arm collision spheres: the committed cover fit ``{model}_arm_spheres.yml``, which covers every surface sample
    of every link by construction and needs no simulator. An arm with no fit is refused. The Lula fallback and its
    surface augmentation were retired in P0f F5, since every arm this repository builds has a fit.
  * default_q: the retract the rule judged on the exact meshes with every registry hand
    (``src/robot/safety/planning/robot/retract.yaml``). No fallback: an arm with no row stops the build rather
    than inheriting a pose nobody measured.
  * everything else: copied from cuRobo's ``ur10e.yml`` template, which carries the six UR joint and link names
    every model shares and names ``tool0`` as its end effector.

No hand. A descriptor used to carry the hand's sphere map on ``tool0``, one ``{model}_{gripper}.yml`` per pairing,
chosen by ``--gripper`` and ``--coupling-mm``. The hand is a body link the planner sidecar adds when it starts, from
the hand the cell names (``robot.gripper.model``, ``coupling_plates``) and where its declared tool frame puts it
(``src/robot/safety/planning/_curobo_body_links.py``, ``body_link.py``). ``_arm_descriptor.arm_only`` takes tool0 out
of the template's collision links, and both flags are refused by name. The per hand files an earlier build wrote stay
in the content directory, and this script never writes one again.

Usage, on the box that owns the cell::

    python scripts/arms/descriptor.py ur3e          # writes ur3e.urdf and willy_ur3e.yml into the content
    python scripts/arms/descriptor.py ur5e          # writes ur5e.urdf and willy_ur5e.yml

The descriptor is named by the arm alone, ``willy_{model}.yml``, and its ``_provenance`` says ``carries_hand: false``.
A planner refuses one built for another arm, or one that carries a hand, and adds the hand its own cell names.

Run it with the cuRobo environment's interpreter. cuRobo then answers where its own content directory is and nothing
has to be guessed::

    ext_deps/curobo_env/python.exe scripts/arms/descriptor.py ur5e

Paths are auto-detected and overridable, so no machine-specific path is baked in:
  * ``WILLY_CUROBO_CONTENT``   the cuRobo ``content`` directory, holding ``configs/robot`` and ``assets/robot``

It is build-time only and is read here and nowhere else. The variables the planner reads at runtime are the
``WILLY_CUROBO_*`` set declared in ``src/robot/safety/planning/environment.py``.
"""
from __future__ import annotations

import os
import re
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
            f"robot.gripper.coupling_plates, so build the arm alone: scripts/arms/descriptor.py {MODEL}"
        )

# What the retired Lula augmentation's fit recipe, reach bound and seed were. They tuned nothing once every arm had a
# cover fit, and the augmentation left in P0f F5, but `_provenance` records all three and `composed_sha256` covers
# `_provenance`, so they stay as the constants every committed evidence file was measured with, and the flags that set
# them are refused.
# `--yml-out` writes the descriptor somewhere else, so a candidate can be measured beside the one cells load.
ARM_FIT = "surface"
BOUND_MM: "float | None" = None
YML_OUT_OVERRIDE: "Path | None" = None
FIT_SEED = 0
#: Where the description comes from: Universal Robots' own, pinned and vendored (`_urdf_source.py`). The `isaac` source
#: read a file from a simulator install and was retired in P0f F5. `--urdf-from ur`, which `install.ps1` passes for
#: every arm, is still accepted, and anything else is refused by the source chooser.
URDF_FROM = "ur"
for _i, _arg in enumerate(sys.argv):
    if _arg in ("--seed", "--arm-fit", "--bound-mm") or _arg.startswith(("--seed=", "--arm-fit=", "--bound-mm=")):
        raise SystemExit(
            f"{_arg} is refused: it tuned the Lula fallback's surface fit, retired in P0f F5. The arm spheres are the "
            f"committed cover fit; refit them with scripts/arms/fit.py --arm {MODEL} --write"
        )
    if _arg == "--urdf-from" and _i + 1 < len(sys.argv):
        URDF_FROM = sys.argv[_i + 1].lower()
    elif _arg == "--yml-out" and _i + 1 < len(sys.argv):
        YML_OUT_OVERRIDE = Path(sys.argv[_i + 1])


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


CUROBO = _resolve_curobo_content()
URDF_OUT = CUROBO / f"assets/robot/ur_description/{MODEL}.urdf"
YML_OUT = YML_OUT_OVERRIDE if YML_OUT_OVERRIDE is not None else CUROBO / f"configs/robot/willy_{MODEL}.yml"
print(f"[paths] model={MODEL}\n  urdf   = {URDF_FROM}\n  curobo = {CUROBO}")

# --- 1) URDF: Universal Robots' own description, rendered ------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _arm_spheres import (ArmFitRecipe, CoverMapError, cover_spheres_label, link_frames,  # noqa: E402
                          link_spheres, place_spheres)

RECIPE = ArmFitRecipe.parse(ARM_FIT)


from _urdf_source import choose as _choose_urdf  # noqa: E402 (beside this script)

_choice = _choose_urdf(MODEL, source=URDF_FROM)
print(f"urdf: {_choice.render()}")
urdf = _choice.text
#: What to call the description in a message.
_source_name = f"{MODEL} as UR describes it"

urdf = urdf.replace("package://ur_description/", "")  # -> meshes/<model>/... relative to asset_root_path

# UR plans the elbow within +-180 degrees while the arm turns +-360, and a description carries whatever its source
# declares: Isaac's copies carried UR's limit for six arms and the full turn for ur10. Inherited, the planner offers an
# elbow turn no UR planner would. The limits are the arm description's (planner_limit_deg, read through _arm_file
# beside this script since P0f F7), and they are applied before either write below so the file on disk carries them too.
import _arm_file  # noqa: E402 (beside this script, stdlib plus PyYAML)
from _planner_limits import clamp_to_planner_limits, narrowed_rad  # noqa: E402

_PLANNER_NARROWED = narrowed_rad(_arm_file.joint_names(MODEL), _arm_file.planner_limits_deg(MODEL),
                                 _arm_file.limits_deg(MODEL))
urdf, _limits_clamped = clamp_to_planner_limits(urdf, _PLANNER_NARROWED)
if _limits_clamped:
    print(f"planner limits: {_source_name} lets {', '.join(_PLANNER_NARROWED)} turn past the planner's limit; "
          f"clamped to {_PLANNER_NARROWED} rad in the copy written here")

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
          f"{CUROBO / 'assets/robot/ur_description'} with scripts/arms/fetch.py before planning.")
else:
    print("  mesh references resolve OK")

# --- 1b) a description with no tool0 or with incomplete inertias is refused ---------------------------------
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
if _needs_tool0 or _needs_inertia:
    # UR's own description carries both for every arm this builds. The graft that transcribed them from a sibling
    # Isaac description was retired in P0f F5 with the Isaac source it served.
    raise SystemExit(
        f"{MODEL}'s description is missing "
        f"{'tool0' if _needs_tool0 else ''}{' and ' if _needs_tool0 and _needs_inertia else ''}"
        f"{'complete inertia' if _needs_inertia else ''}. A frame written from somewhere else is how a hand ends up "
        f"pointing where it is not, and cuRobo dies at load on a link with a mass and no tensor."
    )


# --- 2) arm spheres: the committed cover fit ---------------------------------------------------------------
# The sphere map is this repository's own. It is fitted to the same collision bundle the exact mesh guard
# judges against, by `scripts/arms/fit.py`, and what makes it worth having is that the safe
# half is a property of the construction: every surface sample of every link ends up inside a sphere, so
# there is no hole for the planner to walk the arm through, and no sphere reaches further past its own link
# than the reach the map was fitted at.
#
# Measured against Isaac's Lula family, which this replaces: 0 to 61 mm of reach past the links, gaps up to
# 45 mm once clamped, and on ur10 a map belonging to a different link-frame family altogether.
#
# An arm with no committed fit is refused. It used to fall back to Lula; that fallback and the surface augmentation
# that refined it were retired in P0f F5, and a model UR describes but nobody fitted (ur20, ur30) stops here.
_FITTED_MAP = REPO / f"src/robot/safety/planning/robot/{MODEL}_arm_spheres.yml"
if not _FITTED_MAP.is_file():
    raise SystemExit(
        f"{MODEL} has no committed arm sphere map at {_FITTED_MAP.name}. Fit one: "
        f"scripts/arms/fit.py --arm {MODEL} --write"
    )
_fitted_doc = yaml.safe_load(_FITTED_MAP.read_text(encoding="utf-8")) or {}
# The retract comes from the committed table rather than from Isaac's Lula pose. cuRobo biases its graph search and
# its IK regularisation towards this pose and warms up from it, and Isaac's sits inside the guard's own refusal band
# on two arms, where ur5 keeps 6.9 mm to the EGU-50. The table is chosen by a rule that judges both the exact meshes
# and the planner's own sphere model: scripts/arms/retract.py.
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
if _table_spec is None or _table_spec.loader is None:
    raise ImportError(f"no loader for the retract table reader at {_table_module_path}")
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
    # The anchor is the arm description's (retract_anchor_rad), read through _arm_file like every other number here.
    _anchor = _arm_file.retract_anchor(MODEL)
    if not _anchor:
        raise SystemExit(
            f"{MODEL} has no judged retract AND no anchor to seed from, so there is no pose to write at all."
        ) from None
    default_q = list(_anchor)
    print(f"!! retract: not judged. This is a seed build: default_q is {MODEL}'s anchor {default_q}, written "
          f"so scripts/arms/retract.py {MODEL} can ask this arm's planner. A cell reads its "
          f"pair's row from the table and refuses before a sidecar starts, so this pose never reaches one.")
if _RETRACT is not None:
    print(f"retract (fallback for a sidecar with no hand): {default_q}, "
          f"{'the anchor' if not any(_RETRACT.steps) else 'moved by steps ' + str(_RETRACT.steps)}; "
          f"the pairs this table judged for {MODEL}: {', '.join(sorted(_RETRACT.hands))}")
# The fit names a link by its bundle part (wrist_1) and a descriptor by its URDF link (wrist_1_link): the arm description
# says which is which, rather than a rule that only holds for UR's spelling.
_LINK_OF = {part: link for part, link, _frame in _arm_file.collision_links(MODEL)}
try:
    # Still in the bundle's DH frame here. It is placed into each link's URDF frame below, where the DH
    # table has been checked against UR's own description; doing it here would use an unchecked table.
    sphere_map = link_spheres(_fitted_doc, source=_FITTED_MAP.name, links=_LINK_OF)
except CoverMapError as exc:
    raise SystemExit(str(exc)) from None
_rows = (_fitted_doc.get("_provenance") or {}).get("bodies") or []
_worst = max((float(row.get("fresh_reach_max_mm", 0.0)) for row in _rows), default=0.0)
_FIT_FRAMES = {_LINK_OF[body]: int(block["frame"])
               for body, block in (_fitted_doc.get("collision_spheres") or {}).items()}
ARM_SPHERES_LABEL = cover_spheres_label(
    source=_FITTED_MAP.name, reach_mm=_worst, spheres=sum(len(v) for v in sphere_map.values()))
print(f"arm spheres: the committed cover fit, {_FITTED_MAP.name}, reaching at most {_worst:.1f} mm past "
      f"the links and leaving no hole")

# --- 2a) no hand ---------------------------------------------------------------------------------------------
# Nothing is put on tool0. The hand a cell names becomes a body link of the planner's robot when the sidecar starts,
# placed by the cell's declared tool frame and plates, so one arm descriptor serves every hand.

# The DH rows (a, d, alpha) are the arm description's, read through _arm_file. This module runs in the cuRobo sidecar
# environment, Python 3.10, where `src.config` cannot be imported, so the description is read by path with PyYAML.
# They place the committed fit from the bundle's DH frames into the URDF link frames below. Until P0f F7 they were
# inlined here.
_DH = _arm_file.dh_rows(MODEL)

# And they are held against UR's own description rather than trusted: the description's rows are a transcription, and
# a transcription nobody compares is a number that drifts. Outside the try for the same reason as the lookup.
try:
    import _ur_description as _urd  # noqa: E402 (beside this script, stdlib plus PyYAML)
except Exception as _exc:  # noqa: BLE001 (said out loud; a silent skip is how the copies drift apart)
    print(f"[dh] UR's description is unreadable here ({type(_exc).__name__}: {_exc}), so the described rows "
          f"are used unchecked", flush=True)
else:
    _rows = _urd.dh_rows(MODEL)
    assert np.allclose(np.asarray(_DH, dtype=float), np.asarray(_rows, dtype=float), rtol=0.0, atol=1e-12), (
        f"the arm description's DH rows disagree with UR's own description for {MODEL}: "
        f"{_DH} against {_rows}"
    )

# The committed fit is in the bundle's DH frame and a descriptor is in the URDF link frame. They are not the
# same frame, they differ per link, and nothing downstream measures where a sphere landed. On the ur5e the
# upper arm's two frames sit 425 mm apart along the link, which is the link's own length, and the shoulder's
# differ by a quarter turn about x. A build that skips this placement writes a descriptor whose upper arm
# spheres are 425 mm from the upper arm, and it loads, plans, and reports ready for every arm and hand pair.
# The one thing that catches it is the planner's own boot table: a displaced arm dips under the slab the
# sidecar stands on, and the refusal says so.
_place = link_frames(urdf, _DH, chain_yaw_deg=_arm_file.chain_yaw_deg(MODEL))
sphere_map = {link: place_spheres(spheres, _place(link, _FIT_FRAMES[link]))
              for link, spheres in sphere_map.items()}
print(f"arm spheres placed from the bundle's DH frames into the URDF link frames of {MODEL}")


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
        f"  Refit the arm so every guarded link has spheres: scripts/arms/fit.py --arm {MODEL} --write"
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
# loads (0.1 rad is 5.73 degrees), and the guard refuses outside the factory envelope less its margin, 5 degrees on
# every cell (joint_limits.JOINT_LIMIT_MARGIN_DEG). The one number is wider than the other, and that is the whole reason
# the planner cannot offer a configuration the guard then refuses. Inheriting it from whichever template the content
# holds is not a guarantee.
cs["position_limit_clip"] = _arm_file.position_limit_clip_rad(MODEL)

# --- provenance: which arm, and that no hand is in it -------------------------------------------
# Without this the descriptor carries the spheres and drops every statement about where they came
# from. `carries_hand: false`, set by arm_only, is what a planner start checks before it adds the
# hand a cell names, so a per hand file from an earlier build is never taken for an arm. cuRobo
# ignores unknown top-level keys, so this costs the planner nothing.
cfg["_provenance"] = {
    **cfg.get("_provenance", {}),
    "arm": MODEL,
    # What this descriptor lets the planner do, so a reader does not have to parse the URDF to find out.
    "planner_joint_limits": _PLANNER_NARROWED,
    "position_limit_clip": _arm_file.position_limit_clip_rad(MODEL),
    "limits_clamped": bool(_limits_clamped),
    # Which pose this descriptor starts from, and what it was judged against: an arm descriptor carries no hand, so
    # the retract has to clear every hand a cell could name, and the row says which ones were measured.
    "retract": {
        "steps": list(_RETRACT.steps),
        "anchor": list(_RETRACT.anchor),
        "anchor_source": _RETRACT.anchor_source,
        "judged_hands": {hand: plates for hand, plates in sorted(_RETRACT.hands.items())},
        "arm_rows_sha256": _RETRACT.rows_sha256,
    } if _RETRACT is not None else {
        # A seed build (--retract-from-anchor): the pose is this arm's anchor and no judge has seen it. Said
        # here so a reader of the descriptor learns it from the descriptor, not from the shell it was built in.
        "judged_hands": {},
        "anchor": list(default_q),
        "anchor_source": "seed build: the arm's anchor, unjudged, so the chooser can ask this planner",
    },
    # What the arm spheres are, from what this build ran. It was written from the model's name and said "lula" for
    # every arm but ur5e, while every arm with a bundle was fitted.
    "arm_spheres": ARM_SPHERES_LABEL,
    "arm_fit": RECIPE.render(),
    "fit_seed": FIT_SEED,
    "arm_vendor_bound_mm": BOUND_MM,
    "generated_by": "scripts/arms/descriptor.py",
}

YML_OUT.parent.mkdir(parents=True, exist_ok=True)
YML_OUT.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
print(f"wrote {YML_OUT}")
print(f"  provenance: {MODEL}, no hand, arm spheres {cfg['_provenance'].get('arm_spheres')}")
print(f"  joint order check: cspace.joint_names={cs.get('joint_names')}")
print(f"\nNext: check it plans with a hand added as a body link: "
      f"scripts/arms/check.py {YML_OUT.stem} --hand <robot.gripper.model>")
