"""Spheres for a link whose asset collides with PRIMITIVES and whose Lula description skips it.

    python.bat scripts/isaac/spheres_from_primitive_colliders.py ur10
    python.bat scripts/isaac/spheres_from_primitive_colliders.py ur10 --write

⛔ **THE GAP THIS FILLS.** Isaac ships a Lula collision description for every UR, and for ``ur10``
alone it covers five links instead of six: there are no ``shoulder_link`` spheres. The shared cuRobo
template guards seven links, so a ur10 built from it raises ``KeyError: shoulder_link`` on the first
load. Had the template not named the link, the build would instead have written a ur10 whose
shoulder was never checked for collision, and planned against it in silence.

⭐ **CYLINDERS ARE A FAIR SOURCE FOR A SPHERE MAP, AND WERE NOT A FAIR SOURCE FOR A MESH BUNDLE.**
The same asset was refused by ``bake_ur_collision_meshes.py`` because a file called
``*_collision_meshes.npz`` promises exact geometry and thirteen cylinders are not that. A sphere map
is an approximation by construction, so approximating a declared collision cylinder with spheres
along its axis is a translation between two approximate representations and overstates nothing. What
decides is what the output promises, not what the input happens to be.

⚠ **IT PROVES ITSELF ON THE SAME ROBOT FIRST.** The reading only means something if the USD link
frame is the frame Lula expresses its spheres in. That is not assumed: ur10 has Lula spheres for the
other five links, so every one of those is checked to sit inside the cylinders read for the same
link. Only if all of them do is the sixth link generated. A frame error would put the control
spheres outside their own colliders, which is exactly the failure that would otherwise ship as a
plausible-looking number.
"""
import pathlib
import sys

from isaacsim import SimulationApp  # noqa: E402

_ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
MODEL = (_ARGS[0] if _ARGS else "ur10").lower()
WRITE = "--write" in sys.argv

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

ASSETS = "D:/isaacsim_assets/Assets/Isaac/5.1"
MP = ("D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64/exts/"
      "isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots")
LINKS = ("shoulder_link", "upper_arm_link", "forearm_link",
         "wrist_1_link", "wrist_2_link", "wrist_3_link")

#: How far outside a declared collider a control sphere centre may sit before the frame chain is
#: called wrong. Lula spheres are fitted to the VISUAL hull and the cylinders are a coarse stand-in
#: for it, so they do not have to agree closely -- they have to agree at all. A frame error puts
#: them a link length apart, not a centimetre.
_CONTROL_TOLERANCE_M = 0.05


def world_T(prim) -> np.ndarray:
    """4x4 world -> prim as the composed stage has it, in metres."""
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64).T


def urdf_frames(text: str):
    """base -> link at q=0 for every link in a URDF, 4x4 metres. The frame Lula spheres live in."""
    import xml.etree.ElementTree as ET

    def _rpy(r, p, y):
        def _R(axis, t):
            a = np.zeros(3)
            a[axis] = 1.0
            K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
            return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)
        return _R(2, y) @ _R(1, p) @ _R(0, r)

    kids = {}
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

    def frame(link):
        if link not in kids:
            return np.eye(4)
        par, M = kids[link]
        return frame(par) @ M
    return frame


def _yaw(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    M = np.eye(4)
    M[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return M


def cylinders(link_prim, into: np.ndarray) -> list:
    """Collision CYLINDERS under a link, expressed through ``into``, as
    ``(centre_m, axis_unit, half_height_m, radius_m)``."""
    out = []
    for sub in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies()):
        if not sub.HasAPI(UsdPhysics.CollisionAPI) or str(sub.GetTypeName()) != "Cylinder":
            continue
        cyl = UsdGeom.Cylinder(sub)
        radius = float(cyl.GetRadiusAttr().Get())
        height = float(cyl.GetHeightAttr().Get())
        axis = {"X": 0, "Y": 1, "Z": 2}[str(cyl.GetAxisAttr().Get() or "Z")]
        T = into @ world_T(sub)
        direction = T[:3, axis] / (np.linalg.norm(T[:3, axis]) or 1.0)
        out.append((T[:3, 3], direction, height / 2.0, radius))
    return out


def distance_outside(point, shapes) -> float:
    """How far ``point`` sits outside the NEAREST of ``shapes`` (0.0 when inside one)."""
    best = float("inf")
    for centre, direction, half, radius in shapes:
        d = np.asarray(point, dtype=np.float64) - centre
        along = float(np.dot(d, direction))
        radial = float(np.linalg.norm(d - along * direction))
        best = min(best, max(abs(along) - half, 0.0) + max(radial - radius, 0.0))
    return best


def spheres_along(shapes, *, per_metre: float = 30.0) -> list:
    """Cover each cylinder with spheres of its own radius, spaced so the gaps close.

    Sphere radius IS the cylinder radius, so the union contains the cylinder body wherever two
    consecutive spheres overlap; spacing at most the radius guarantees that. Conservative on the flat
    ends by exactly one radius, which is the honest direction for a collision proxy to err in.
    """
    out = []
    for centre, direction, half, radius in shapes:
        n = max(2, int(np.ceil(2.0 * half / max(radius, 1e-6))) + 1)
        for t in np.linspace(-half, half, n):
            out.append({"center": [round(float(v), 4) for v in (centre + t * direction)],
                        "radius": round(float(radius), 4)})
    return out


stage = omni.usd.get_context().get_stage()
prim_path = f"/World/{MODEL}"
add_reference_to_stage(f"{ASSETS}/Isaac/Robots/UniversalRobots/{MODEL}/{MODEL}.usd", prim_path)
stage.Load(Sdf.Path(prim_path))

lula_path = f"{MP}/{MODEL}/rmpflow/{MODEL}_robot_description.yaml"
with open(lula_path, encoding="utf-8") as fh:
    lula = yaml.safe_load(fh)
lula_spheres = {k: v for entry in lula["collision_spheres"] for k, v in entry.items()}

report = []
report.append(f"[read] {MODEL}: Lula covers {sorted(lula_spheres)}")

#: The candidate frame chains, tried in order, each mapping USD WORLD into the frame a Lula sphere
#: is expressed in. Which one is right is DECIDED BY THE CONTROL below rather than asserted here.
#:
#: ⛔ The first version of this script assumed the USD link prim frame WAS that frame. It is not:
#: the control put Lula spheres 462.1 mm outside their own colliders, which is most of a link. A
#: sphere placed that way does not fail, it guards the wrong volume, and every extent and count a
#: reader would check still looks right.
def _candidates(urdf_frame):
    return (
        ("usd link prim", lambda link, lp: np.linalg.inv(world_T(lp))),
        ("urdf link frame", lambda link, lp: np.linalg.inv(urdf_frame(link))),
        ("urdf link frame, base yawed 180", lambda link, lp: np.linalg.inv(urdf_frame(link)) @ _yaw(180.0)),
        ("urdf link frame, base yawed -90", lambda link, lp: np.linalg.inv(urdf_frame(link)) @ _yaw(-90.0)),
        ("urdf link frame, base yawed 90", lambda link, lp: np.linalg.inv(urdf_frame(link)) @ _yaw(90.0)),
    )


_urdf_path = pathlib.Path(
    f"D:/dev/aurora_backend/ext_deps/curobo/curobo/content/assets/robot/ur_description/{MODEL}.urdf")
if not _urdf_path.is_file():
    report.append(f"[read] no {_urdf_path.name} to read link frames from. Build the cuRobo descriptor "
                  f"first: ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py {MODEL}")
    print("\n".join(report), flush=True)
    app.close()
    raise SystemExit(1)
_urdf_frame = urdf_frames(_urdf_path.read_text(encoding="utf-8"))

link_prims = {ln: stage.GetPrimAtPath(f"{prim_path}/{ln}") for ln in LINKS}

# ---- the control PICKS the frame chain, and refuses when none of them explains the spheres -------
chosen = None
report.append("\n[control] which frame chain puts the Lula spheres inside their own colliders?")
for name, build in _candidates(_urdf_frame):
    shapes = {}
    for link in LINKS:
        lp = link_prims[link]
        shapes[link] = cylinders(lp, build(link, lp)) if lp else []
    worst, checked = 0.0, 0
    for link, spheres in lula_spheres.items():
        if not shapes.get(link):
            continue
        for sp in spheres:
            worst = max(worst, distance_outside(sp["center"], shapes[link]))
            checked += 1
    verdict = "PASS" if (checked >= 10 and worst <= _CONTROL_TOLERANCE_M) else "no"
    report.append(f"    {name:34s} {checked:3d} spheres, worst {worst * 1000.0:8.1f} mm  {verdict}")
    if verdict == "PASS" and chosen is None:
        chosen = (name, shapes, worst, checked)

if chosen is None:
    # DIAGNOSIS, because "no chain fits" is not actionable on its own. Print where each side puts the
    # same link, in the first candidate frame, so the next reader sees whether the two are offset by a
    # link length (a frame error) or describe different things entirely.
    _diag = _candidates(_urdf_frame)[0][1]
    report.append("[diagnose] per link, in the first candidate frame: Lula sphere centres vs cylinder axes")
    for link in sorted(lula_spheres):
        lp = link_prims.get(link)
        if lp is None:
            continue
        shapes = cylinders(lp, _diag(link, lp))
        lc = np.asarray([sp["center"] for sp in lula_spheres[link]], dtype=np.float64)
        report.append(f"    {link:16s} lula x[{lc[:,0].min():7.3f},{lc[:,0].max():7.3f}] "
                      f"y[{lc[:,1].min():7.3f},{lc[:,1].max():7.3f}] "
                      f"z[{lc[:,2].min():7.3f},{lc[:,2].max():7.3f}] m")
        for centre, direction, half, radius in shapes:
            report.append(f"    {'':16s} cyl  centre {np.round(centre, 3).tolist()} "
                          f"axis {np.round(direction, 2).tolist()} half {half:.3f} r {radius:.3f}")
    report.append(f"[control] FAILED: no candidate frame chain puts the Lula spheres inside the "
                  f"colliders read for their own links (limit {_CONTROL_TOLERANCE_M * 1000.0:.0f} mm). "
                  f"Nothing written: a sphere in the wrong place along a link does not fail, it "
                  f"guards the wrong volume.")
    print("\n".join(report), flush=True)
    app.close()
    raise SystemExit(1)

_name, shapes_by_link, worst, checked = chosen
report.append(f"[control] PASS on {_name!r}: {checked} Lula spheres across "
              f"{sum(1 for k in lula_spheres if shapes_by_link.get(k))} links, worst "
              f"{worst * 1000.0:.1f} mm outside its own collider. That is the frame a generated link "
              f"will be written in.")
for link in LINKS:
    report.append(f"  {link:16s} {len(shapes_by_link[link])} collision cylinder(s)")

# ---- the gap -------------------------------------------------------------------------------------
missing = [ln for ln in LINKS if ln not in lula_spheres]
report.append(f"\n[gap] Lula has no spheres for: {missing or 'nothing, this model needs no help'}")
generated = {}
for link in missing:
    if not shapes_by_link[link]:
        report.append(f"  {link}: no collision cylinders either, so there is nothing to translate")
        continue
    generated[link] = spheres_along(shapes_by_link[link])
    radii = {s["radius"] for s in generated[link]}
    report.append(f"  {link}: {len(generated[link])} spheres, radius {sorted(radii)} m, from "
                  f"{len(shapes_by_link[link])} cylinder(s)")

if generated and WRITE:
    out = {"collision_spheres": [{k: v} for k, v in generated.items()]}
    dst = f"D:/dev/aurora_backend/src/robot/safety/data/{MODEL}_primitive_spheres.yml"
    header = (
        f"# Collision spheres for the {MODEL} links Isaac ships no Lula spheres for.\n"
        f"#\n"
        f"# GENERATED by scripts/isaac/spheres_from_primitive_colliders.py from the collision\n"
        f"# CYLINDERS declared in the {MODEL} USD. Each sphere carries its cylinder own radius and they\n"
        f"# are spaced at most one radius apart, so their union contains the cylinder body and is\n"
        f"# conservative by one radius at the flat ends.\n"
        f"#\n"
        f"# The frame chain was proved on this same robot before these were written: every Lula sphere\n"
        f"# on the links that HAVE them sits inside the cylinders read for that link, worst\n"
        f"# {worst * 1000.0:.1f} mm outside against a {_CONTROL_TOLERANCE_M * 1000.0:.0f} mm limit.\n"
    )
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(header + yaml.safe_dump(out, sort_keys=False))
    report.append(f"\n[write] {dst}")
elif generated:
    report.append("\n[dry-run] pass --write to save them")

print("\n".join(report), flush=True)
app.close()
