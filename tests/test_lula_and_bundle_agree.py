"""Isaac's sphere map and this repository's baked meshes must describe the same link, in one frame.

⛔ **WHY THIS IS LOAD-BEARING.** A cuRobo UR descriptor carries BOTH: Isaac's Lula spheres as the
base map, and surface spheres fitted to ``{model}_collision_meshes.npz`` and placed through
``inv(urdf_link_frame) @ Rz(180) @ T_dh[frame]``. If those two frames disagree for a model, the
descriptor guards two sets of geometry sitting in different places, and nothing reports it: the file
loads, the planner runs, the spheres are simply somewhere they are not.

⚠ **AND IT IS NOT HYPOTHETICAL.** Measured 2026-09-09: ``ur10``'s Lula description disagrees with
``ur10``'s own URDF. Its forearm spheres run along +z from 0 to 0.570 m where the asset's collision
cylinders run along -x from 0 to -0.572 m. Same link, same length, rotated frame. That is why ur10
gets no cuRobo descriptor at all, and it is the reason this file exists: the augmentation was turned
on for three more arms the same day, and "it looked right on ur5e" is not evidence about ur3.

MEASURED for the five models that have a bundle, as the worst distance a Lula sphere CENTRE sits
outside the mesh bounding box for its own link::

    ur3   0.0 mm     ur3e  0.0 mm     ur5   0.0 mm     ur5e  0.0 mm     ur10e  13.0 mm

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
                    gap = float(np.max(np.maximum(np.maximum(lo - c, c - hi), 0.0).sum(axis=1)))
                    self.assertLess(
                        gap * 1000.0, _TOLERANCE_MM,
                        f"{model}/{link}: an Isaac sphere sits {gap * 1000.0:.1f} mm outside the "
                        f"mesh this repository baked for the same link. The descriptor would guard "
                        f"two geometries in two places and report nothing.",
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
