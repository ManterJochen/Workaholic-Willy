"""Isaac's sphere map and this repository's baked meshes must describe the same link, in one frame.

⛔ **WHY THIS IS LOAD-BEARING.** A cuRobo UR descriptor carries BOTH: Isaac's Lula spheres as the
base map, and surface spheres fitted to ``{model}_collision_meshes.npz`` and placed through
``inv(urdf_link_frame) @ Rz(180) @ T_dh[frame]``. If those two frames disagree for a model, the
descriptor guards two sets of geometry sitting in different places, and nothing reports it: the file
loads, the planner runs, the spheres are simply somewhere they are not.

⚠ **AND IT IS NOT HYPOTHETICAL. THIS ASSERTION FOUND THE DEFECT IT WAS WRITTEN FOR.** Measured
2026-09-10: Isaac's ``ur10`` map places the spheres of all three WRIST links about 61 mm from where
that link's geometry actually is. The bundle is not the wrong half -- through the DH chain the baked
ur10 arm is CONNECTED, every gap between consecutive links under 3.3 mm against 0.2 to 0.8 mm on
ur10e -- so the vendor's wrist spheres are simply somewhere the wrist is not.

Keeping both sets made each wrist a body twice its size spanning two positions. cuRobo found every
configuration in collision and returned None from plan_pose AND plan_cspace, for every start pose
including a plan from a pose to itself. The descriptor loaded perfectly and planned nothing, which
is the most expensive shape of failure: everything upstream reports healthy.

``build_ur_config.py`` now drops a sphere whose centre lies further outside its own link mesh than
its own RADIUS, which is the point where a sphere and a body stop intersecting rather than a tuned
number. So this file checks the spheres the BUILDER KEEPS, because those are what plans.

MEASURED again on 2026-09-16, after every arm bundle was re-baked from Universal Robots' own collision STLs
(B5 S6 to S8), as the worst distance a KEPT sphere centre sits outside the mesh bounding box for its own
link. The numbers are in :data:`WORST_KEPT_MM` and are asserted against a recomputation, so a bundle or a
map that moves fails here rather than drifting::

    ur3 0.0   ur3e 0.0   ur5 0.0   ur5e 0.0   ur10 75.9   ur10e 13.0   ur16e 60.8 mm

⛔ **TWO ARMS ARE OVER THE RULE, AND BOTH SAY THE SAME THING: THAT MAP IS NOT THAT ARM.**

ur10 was 0.0 while its bundle was itself built from Isaac's importer asset, so the two halves agreed with
each other and with nothing else. Against UR's own geometry its worst KEPT sphere sits **75.9 mm** outside
its own upper arm, and 63.7 mm outside wrist_2. And ur16e keeps all ten upper arm spheres with the worst
60.8 mm out: that map describes the LONGER ur10e upper arm. Both are over the 60 mm rule, so
``build_ur_config.py`` refuses to build either arm from the vendor map.

⚠ **AND SINCE B6 NO ARM IS BUILT FROM THAT MAP AT ALL.** Every arm here has a committed cover fit
(`{arm}_arm_spheres.yml`) fitted to the same bundle this file measures against, and the builder reads Isaac's
family only where there is no fit, which is nowhere. So what this file now holds is the reason the vendor map
is a FALLBACK and not the source: it is the measurement that says the fallback would be wrong for two of the
seven arms, and by how much. ur10 and ur16e are exactly the two arms that had no descriptor at all until the
fit gave them one.

The 13 mm on ur10e is expected and small: Lula spheres are fitted to the VISUAL hull, the bundle
holds the COLLISION meshes, and the collision hull of that arm is slightly the tighter of the two.
A frame error is not 13 mm, it is a link length, which is what the tolerance below is set to catch.
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np

from src.robot.safety._ur_kinematics import UR_DH_TABLES_M
from src.robot.safety.planning.environment import collision_mesh_bundle

_REPO = pathlib.Path(__file__).resolve().parents[1]
_URDF_DIR = _REPO / "ext_deps" / "curobo" / "curobo" / "content" / "assets" / "robot" / "ur_description"
_MP = pathlib.Path(
    "D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64/exts/"
    "isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots"
)

#: bundle key -> (URDF link, DH frame index). The same map the bake and the augmentation both use.
_LINKS = {"shoulder": ("shoulder_link", 1), "upper_arm": ("upper_arm_link", 2),
          "forearm": ("forearm_link", 3), "wrist_1": ("wrist_1_link", 4),
          "wrist_2": ("wrist_2_link", 5), "wrist_3": ("wrist_3_link", 6)}

#: A frame error puts a sphere a LINK LENGTH away, so this is set an order of magnitude below the
#: shortest UR link and an order above the 13 mm hull difference that is real and expected.
_TOLERANCE_MM = 60.0

#: The worst distance a KEPT vendor sphere sits outside the mesh baked for its own link, in millimetres,
#: measured 2026-09-16. A record, recomputed here every run: this is the number, not a ceiling.
WORST_KEPT_MM = {
    "ur3": 0.0, "ur3e": 0.0, "ur5": 0.0, "ur5e": 0.0, "ur10": 75.9, "ur10e": 13.0, "ur16e": 60.8,
}

#: The arms whose vendor map is over the rule, so ``build_ur_config.py`` refuses to build them from it. Derived
#: from the record above rather than listed, so an arm cannot be quietly excused by being named twice.
_REFUSED = frozenset(model for model, mm in WORST_KEPT_MM.items() if mm > _TOLERANCE_MM)


def _urdf_frames(text: str):
    import xml.etree.ElementTree as ET

    def rpy(r: float, p: float, y: float) -> np.ndarray:
        def rot(axis: int, t: float) -> np.ndarray:
            a = np.zeros(3)
            a[axis] = 1.0
            K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
            return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)
        return rot(2, y) @ rot(1, p) @ rot(0, r)

    kids: dict = {}
    for jt in ET.fromstring(text).findall("joint"):
        child, parent = jt.find("child"), jt.find("parent")
        if child is None or parent is None:
            continue
        o = jt.find("origin")
        r = [float(v) for v in o.attrib.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
        x = [float(v) for v in o.attrib.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
        M = np.eye(4)
        M[:3, :3] = rpy(*r)
        M[:3, 3] = x
        kids[child.attrib["link"]] = (parent.attrib["link"], M)

    def frame(link: str) -> np.ndarray:
        if link not in kids:
            return np.eye(4)
        parent, M = kids[link]
        return frame(parent) @ M
    return frame


def _dh_frames(model: str) -> list:
    T = np.eye(4)
    out = [T.copy()]
    for row in UR_DH_TABLES_M[model]:
        ca, sa = np.cos(row.alpha_rad), np.sin(row.alpha_rad)
        T = T @ np.array([[1.0, 0.0, 0.0, row.a_m], [0.0, ca, -sa, 0.0],
                          [0.0, sa, ca, row.d_m], [0.0, 0.0, 0.0, 1.0]])
        out.append(T.copy())
    return out


def _comparable() -> list[str]:
    """Models with a committed bundle, a built URDF and an Isaac Lula description, all three."""
    out = []
    for model in sorted(UR_DH_TABLES_M):
        if (collision_mesh_bundle(model).is_file()
                and (_URDF_DIR / f"{model}.urdf").is_file()
                and (_MP / model / "rmpflow" / f"{model}_robot_description.yaml").is_file()):
            out.append(model)
    return out


class TheTwoGeometriesLandInOneFrameTests(unittest.TestCase):
    """Skipped off-box: the Lula descriptions ship with Isaac, not with this repository."""

    def setUp(self) -> None:
        if not _MP.is_dir():
            self.skipTest(f"no Isaac motion-policy configs at {_MP}")
        if not _URDF_DIR.is_dir():
            self.skipTest("no built cuRobo URDFs; run scripts/curobo/build_ur_config.py first")

    def test_there_are_models_to_compare(self) -> None:
        """The control: a discovery that found nothing would make the comparison pass in silence."""
        self.assertGreaterEqual(len(_comparable()), 2, f"found {_comparable()}")

    def test_every_lula_sphere_sits_inside_its_own_link_mesh(self) -> None:
        import yaml

        worst: dict = {}

        for model in _comparable():
            mesh = np.load(collision_mesh_bundle(model), allow_pickle=True)
            urdf = _urdf_frames((_URDF_DIR / f"{model}.urdf").read_text(encoding="utf-8"))
            dh = _dh_frames(model)
            lula_doc = yaml.safe_load(
                (_MP / model / "rmpflow" / f"{model}_robot_description.yaml").read_text(encoding="utf-8"))
            lula = {k: v for entry in lula_doc["collision_spheres"] for k, v in entry.items()}

            rz = np.eye(4)
            rz[:3, :3] = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

            for key, (link, frame) in _LINKS.items():
                if link not in lula or f"{key}__v" not in mesh:
                    continue
                with self.subTest(model=model, link=link):
                    v = np.asarray(mesh[f"{key}__v"], dtype=np.float64) / 1000.0
                    X = np.linalg.inv(urdf(link)) @ rz @ dh[frame]
                    v = (X[:3, :3] @ v.T).T + X[:3, 3]
                    lo, hi = v.min(0), v.max(0)
                    c = np.asarray([s["center"] for s in lula[link]], dtype=np.float64)
                    # THE SPHERES AS THE BUILDER LEAVES THEM, not as Isaac ships them.
                    # MEASURED 2026-09-10, by this very assertion: Isaac's ur10 map puts all three
                    # WRIST links' spheres about 61 mm from where that link's geometry is. Keeping
                    # both sets made each wrist a body twice its size spanning two positions, and
                    # cuRobo then returned None from every plan, including one from a pose to
                    # itself. The builder now drops a sphere whose centre lies further outside its
                    # own link mesh than its own radius, so what is checked here is what plans.
                    keep = [sp for sp, ctr in zip(lula[link], c)
                            if float(np.linalg.norm(
                                np.maximum(np.maximum(lo - ctr, ctr - hi), 0.0))) <= float(sp["radius"])]
                    self.assertTrue(
                        keep,
                        f"{model}/{link}: not one of the {len(lula[link])} vendor spheres touches "
                        f"the mesh this repository baked for the same link, so the two halves "
                        f"describe different robots and the builder has nothing to keep.",
                    )
                    kept = np.asarray([sp["center"] for sp in keep], dtype=np.float64)
                    gap = float(np.max(np.maximum(np.maximum(lo - kept, kept - hi), 0.0).sum(axis=1)))
                    worst[model] = max(worst.get(model, 0.0), gap * 1000.0)
                    self.assertLess(
                        gap * 1000.0, _TOLERANCE_MM if model not in _REFUSED else 200.0,
                        f"{model}/{link}: a sphere the builder KEEPS sits {gap * 1000.0:.1f} mm "
                        f"outside the mesh baked for the same link. The descriptor would guard two "
                        f"geometries in two places and report nothing.",
                    )

        for model, measured in sorted(worst.items()):
            with self.subTest(recorded=model):
                self.assertAlmostEqual(measured, WORST_KEPT_MM[model], delta=0.1,
                                       msg=f"{model} now measures {measured:.1f} mm, and the record says "
                                           f"{WORST_KEPT_MM[model]:.1f} mm: a bundle or a map moved.")

    def test_the_builder_refuses_exactly_the_arms_over_the_rule(self) -> None:
        """A number in a test is a note unless something acts on it. The builder is what acts.

        Read as source, because building a descriptor needs the cuRobo environment and a GPU. What it holds is
        that the limit the builder enforces IS this file's rule, and that the arms over it are refused rather
        than filtered down to whatever survived.
        """
        source = (_REPO / "scripts" / "curobo" / "build_ur_config.py").read_text(encoding="utf-8")
        self.assertIn(f"VENDOR_SPHERE_LIMIT_MM = {_TOLERANCE_MM:g}", source,
                      "the builder's limit and this file's rule are two numbers again")
        self.assertIn("does not describe this arm", source)
        self.assertTrue(_REFUSED, "no arm is over the rule, so this assertion has nothing to check")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
