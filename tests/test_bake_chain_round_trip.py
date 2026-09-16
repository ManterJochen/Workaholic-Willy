"""The bake's frame chain and the builder's placement are inverses of one another (B5, S4).

A collision bundle is only a safety authority if its vertices sit where the arm's parts actually are. The bake takes a
mesh from its own file frame into the link's DH frame; the descriptor builder takes it back out again when it places
spheres. If those two disagree, nothing fails: the guard simply watches empty space, and every verdict it gives is a
pass for the wrong reason.

So this puts a synthetic tetrahedron at the rendered mesh origin, runs it through the bake's transform and then back
through the builder's, and requires the identity. No mesh file, no Isaac and no GPU: the chain is arithmetic over the
URDF the renderer writes.

The three controls each break one link of that chain and must fail: the base yaw that reconciles Isaac's cell with the
DH chain, a reflection, and a DH table that disagrees with the description it was derived from.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"


def _load(name: str, path: Path):
    """Load a script by path with its own folder importable, the way Python starts it."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(_SCRIPTS / "curobo"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS / "curobo"))
        sys.path.remove(str(path.parent))
    return module


def _bake():
    return _load("bake_ur_meshes_from_urdf_under_test", _SCRIPTS / "isaac" / "bake_ur_meshes_from_urdf.py")


def _description():
    return _load("_ur_description_under_test", _SCRIPTS / "curobo" / "_ur_description.py")


#: Four points that span three dimensions, so a reflection cannot be undone by a rotation.
_TETRAHEDRON = np.array([[0.0, 0.0, 0.0], [0.031, 0.0, 0.0], [0.0, 0.017, 0.0], [0.0, 0.0, 0.043]])
_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)


class TheBakeAndThePlacementAreInversesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bake = _bake()
        self.description = _description()

    def _placement(self, urdf_text: str, model: str, link: str, frame: int) -> np.ndarray:
        """The builder's own placement: out of the DH frame and back into the URDF link frame."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(urdf_text)
        joints = {}
        for joint in root.findall("joint"):
            joints[joint.attrib["name"]] = {
                "parent": joint.find("parent").attrib["link"],
                "child": joint.find("child").attrib["link"],
                "M": self.bake._origin(joint.find("origin")),  # noqa: SLF001
            }
        urdf_frame = self.bake.link_frame(joints, link)
        rz = np.eye(4)
        rz[:3, :3] = self.bake._rpy(0.0, 0.0, np.pi)  # noqa: SLF001
        return np.linalg.inv(urdf_frame) @ rz @ self.bake.dh_frames(model)[frame]

    def _mesh_origin(self, urdf_text: str, link: str) -> np.ndarray:
        import xml.etree.ElementTree as ET

        for element in ET.fromstring(urdf_text).findall("link"):
            if element.attrib["name"] != link:
                continue
            for candidate in (element.find("collision"), element.find("visual")):
                if candidate is not None and candidate.find("geometry/mesh") is not None:
                    return self.bake._origin(candidate.find("origin"))  # noqa: SLF001
        raise AssertionError(f"{link} has no mesh in the rendered URDF")

    def test_a_point_survives_the_round_trip_for_every_link_of_every_model(self) -> None:
        for model in sorted(self.bake.UR_DH):
            urdf = self.description.render_urdf(model)
            for key, (link, frame) in self.bake.LINKS.items():
                with self.subTest(model=model, link=key):
                    v_mm, faces, reported = self.bake.to_dh_frame(_TETRAHEDRON, _FACES, model, key, urdf)
                    self.assertEqual(reported, frame)
                    self.assertEqual(faces.tolist(), _FACES.tolist())

                    back = self._placement(urdf, model, link, frame)
                    in_metres = np.asarray(v_mm, dtype=np.float64) / 1000.0
                    returned = (back[:3, :3] @ in_metres.T).T + back[:3, 3]

                    origin = self._mesh_origin(urdf, link)
                    expected = (origin[:3, :3] @ _TETRAHEDRON.T).T + origin[:3, 3]
                    np.testing.assert_allclose(returned, expected, atol=1e-9)

    def test_the_base_yaw_is_load_bearing(self) -> None:
        """⭐ THE CONTROL. 180 degrees is what reconciles the Isaac cell with the DH chain. At 0 the whole arm is
        rotated half a turn about its own base, and the round trip above stops closing."""
        urdf = self.description.render_urdf("ur5e")
        original = self.bake.BASE_YAW_DEG
        try:
            self.bake.BASE_YAW_DEG = 0.0
            v_mm, _, _ = self.bake.to_dh_frame(_TETRAHEDRON, _FACES, "ur5e", "upper_arm", urdf)
        finally:
            self.bake.BASE_YAW_DEG = original
        turned, _, _ = self.bake.to_dh_frame(_TETRAHEDRON, _FACES, "ur5e", "upper_arm", urdf)
        self.assertGreater(float(np.max(np.abs(np.asarray(v_mm) - np.asarray(turned)))), 1.0)

    def test_a_reflected_chain_is_refused_rather_than_baked(self) -> None:
        """⭐ THE CONTROL. A mirror keeps every distance and every bounding box, so nothing downstream can see it: the
        bundle would guard a left handed arm and agree with every extent check there is. The determinant of the
        composed transform is the one place it shows, so the bake looks there before it writes a vertex."""
        urdf = self.description.render_urdf("ur5e")
        original = self.bake.dh_frames

        def reflected(model: str) -> list:
            # The DH chain rather than the URDF walk, which is recursive: reflecting that would flip the sign once
            # per level and cancel itself out on any link at an even depth.
            frames = [frame.copy() for frame in original(model)]
            for frame in frames:
                frame[:3, :3] = frame[:3, :3] @ np.diag([1.0, 1.0, -1.0])
            return frames

        try:
            self.bake.dh_frames = reflected
            with self.assertRaises(ValueError):
                self.bake.to_dh_frame(_TETRAHEDRON, _FACES, "ur5e", "upper_arm", urdf)
        finally:
            self.bake.dh_frames = original

    def test_a_dh_table_that_disagrees_with_the_description_is_caught_before_a_mesh_is_read(self) -> None:
        """⭐ THE CONTROL. The bake keeps its own inlined DH copy. A copy that drifted from UR's own numbers places
        every vertex of every link, so it is checked against the description before anything is transformed."""
        urdf = self.description.render_urdf("ur5e")
        original = self.bake.UR_DH["ur5e"]
        drifted = list(original)
        a, d, alpha = drifted[3]
        drifted[3] = (a, d + 1e-6, alpha)
        try:
            self.bake.UR_DH["ur5e"] = tuple(drifted)
            with self.assertRaises(AssertionError):
                self.bake.to_dh_frame(_TETRAHEDRON, _FACES, "ur5e", "upper_arm", urdf)
        finally:
            self.bake.UR_DH["ur5e"] = original


if __name__ == "__main__":
    unittest.main()
