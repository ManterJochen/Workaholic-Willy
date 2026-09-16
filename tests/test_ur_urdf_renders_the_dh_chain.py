"""The URDF written from Universal Robots' description moves every link exactly as the guard's DH chain does.

``build_ur_config.py`` places each link's exact mesh with ``inv(URDF link frame at q=0) @ Rz(pi) @ DH frame`` and the
guard places the same mesh by the DH frame alone. Both are right only if that product is the same at EVERY joint vector,
not just at zero. So for every model with a DH row, at q = 0 and at 64 seeded joint vectors, this proves

    C(link, q) = inv(F_urdf(link, q)) @ Rz(pi) @ F_dh(frame, q)  is constant,

that ``base_link_inertia`` is the DH base turned half a turn, and that ``wrist_3_link`` and ``tool0`` are both the DH
flange itself, the frame every hand model hangs from (``build_ur_config.py`` measured the same on 2026-09-10). F_dh is
the guard's own ``ur_link_transforms_mm``; F_urdf chains the rendered joint origins (URDF rpy is Rz(yaw) Ry(pitch)
Rx(roll)) with a rotation about each joint's z axis.

Three controls must fail, or the constancy above is a property of the arithmetic rather than of the robot: a d4 moved by
a micrometre, the flange joint's rotation sign flipped, and wrist_1 moved by a millimetre before rendering. A fourth
holds the test's own rotation helper to the textbook matrices: a sign slip in Ry turned tool0 into a half turn about x
on the first run of this file, identically for UR's macro and for Isaac's copy, which is exactly how a wrong helper
looks like a fact about the robot.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys
import unittest
import xml.etree.ElementTree as ET
from types import MappingProxyType

import numpy as np

from src.robot.safety._ur_kinematics import UR_DH_TABLES_M, ur_link_transforms_mm

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODULE = _ROOT / "scripts" / "curobo" / "_ur_description.py"
_ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
               "wrist_3_joint")
_LINK_FRAMES = (("base_link_inertia", 0), ("shoulder_link", 1), ("upper_arm_link", 2), ("forearm_link", 3),
                ("wrist_1_link", 4), ("wrist_2_link", 5), ("wrist_3_link", 6), ("tool0", 6))
_TOL = 1e-9


def _load():
    name = "_ur_description_for_fk_tests"
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rot(axis: int, angle: float) -> np.ndarray:
    """The right handed rotation about x (0), y (1) or z (2), as a 4x4."""
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4)
    if axis == 0:
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == 1:
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def _origin(joint: ET.Element) -> np.ndarray:
    element = joint.find("origin")
    xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
    roll, pitch, yaw = (float(v) for v in element.get("rpy", "0 0 0").split())
    t = _rot(2, yaw) @ _rot(1, pitch) @ _rot(0, roll)
    t[:3, 3] = xyz
    return t


def _urdf_frames(robot: ET.Element, q: np.ndarray) -> dict[str, np.ndarray]:
    joints = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
    frames: dict[str, np.ndarray] = {"base_link": np.eye(4)}

    def frame(link: str) -> np.ndarray:
        if link not in frames:
            joint = joints[link]
            t = frame(joint.find("parent").get("link")) @ _origin(joint)
            name = joint.get("name")
            if name in _ARM_JOINTS:
                t = t @ _rot(2, float(q[_ARM_JOINTS.index(name)]))
            frames[link] = t
        return frames[link]

    for link in joints:
        frame(link)
    return frames


def _guard_frames(model: str, q: np.ndarray) -> list[np.ndarray]:
    frames = ur_link_transforms_mm(model, np.asarray(q, dtype=np.float64))
    assert frames is not None
    out = []
    for t in frames:
        m = np.array(t, dtype=np.float64)
        m[:3, 3] /= 1000.0
        out.append(m)
    return out


def _dh_frames(rows: list[tuple[float, float, float]], q: np.ndarray) -> list[np.ndarray]:
    """The standard DH chain from plain rows, so a control can perturb one number."""
    out = [np.eye(4)]
    for theta, (a, d, alpha) in zip(q, rows):
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        out.append(out[-1] @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st],
                                       [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]]))
    return out


def _offsets(robot: ET.Element, dh: list[np.ndarray], q: np.ndarray) -> dict[str, np.ndarray]:
    urdf = _urdf_frames(robot, q)
    return {link: np.linalg.inv(urdf[link]) @ _rot(2, np.pi) @ dh[frame] for link, frame in _LINK_FRAMES}


def _samples() -> list[np.ndarray]:
    rng = np.random.default_rng(20260915)
    return [np.zeros(6)] + [rng.uniform(-np.pi, np.pi, 6) for _ in range(64)]


def _worst_drift(robot: ET.Element, dh_of_q) -> dict[str, float]:
    samples = _samples()
    base = _offsets(robot, dh_of_q(samples[0]), samples[0])
    worst = {link: 0.0 for link, _ in _LINK_FRAMES}
    for q in samples[1:]:
        for link, c in _offsets(robot, dh_of_q(q), q).items():
            worst[link] = max(worst[link], float(np.abs(c - base[link]).max()))
    return worst


class TheRotationHelperIsTheTextbookTests(unittest.TestCase):
    def test_a_quarter_turn_moves_each_axis_where_the_right_hand_rule_says(self) -> None:
        x, y, z = np.eye(4)[:, 0], np.eye(4)[:, 1], np.eye(4)[:, 2]
        np.testing.assert_allclose(_rot(0, np.pi / 2) @ y, z, atol=1e-15)
        np.testing.assert_allclose(_rot(1, np.pi / 2) @ z, x, atol=1e-15)
        np.testing.assert_allclose(_rot(2, np.pi / 2) @ x, y, atol=1e-15)


class TheRenderedChainIsTheDhChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ur = _load()
        cls.robots = {model: ET.fromstring(cls.ur.render_urdf(model)) for model in UR_DH_TABLES_M}

    def test_the_local_dh_chain_is_the_guard_s(self) -> None:
        """The controls below perturb a local chain; it has to be the guard's chain before it is perturbed."""
        for model, table in UR_DH_TABLES_M.items():
            rows = [(r.a_m, r.d_m, r.alpha_rad) for r in table]
            for q in _samples()[:8]:
                for mine, guard in zip(_dh_frames(rows, q), _guard_frames(model, q)):
                    with self.subTest(model=model):
                        np.testing.assert_allclose(mine, guard, atol=1e-12)

    def test_every_link_keeps_one_offset_to_its_dh_frame(self) -> None:
        for model, robot in self.robots.items():
            drift = _worst_drift(robot, lambda q, m=model: _guard_frames(m, q))
            for link, value in drift.items():
                with self.subTest(model=model, link=link):
                    self.assertLess(value, _TOL)

    def test_base_link_inertia_is_the_dh_base_turned_half_a_turn(self) -> None:
        for model, robot in self.robots.items():
            with self.subTest(model=model):
                np.testing.assert_allclose(_urdf_frames(robot, np.zeros(6))["base_link_inertia"], _rot(2, np.pi),
                                           atol=1e-12)

    def test_wrist_3_and_tool0_are_the_dh_flange(self) -> None:
        for model, robot in self.robots.items():
            offsets = _offsets(robot, _guard_frames(model, np.zeros(6)), np.zeros(6))
            for link in ("wrist_3_link", "tool0"):
                with self.subTest(model=model, link=link):
                    np.testing.assert_allclose(offsets[link], np.eye(4), atol=_TOL)

    def test_a_micrometre_in_d4_breaks_it(self) -> None:
        rows = [(r.a_m, r.d_m, r.alpha_rad) for r in UR_DH_TABLES_M["ur5e"]]
        rows[3] = (rows[3][0], rows[3][1] + 1e-6, rows[3][2])
        drift = _worst_drift(self.robots["ur5e"], lambda q: _dh_frames(rows, q))
        self.assertGreater(max(drift["wrist_1_link"], drift["wrist_2_link"], drift["tool0"]), _TOL)

    def test_a_flipped_flange_breaks_tool0(self) -> None:
        robot = ET.fromstring(self.ur.render_urdf("ur5e"))
        flange = next(j for j in robot.findall("joint") if j.find("child").get("link") == "flange")
        rpy = [-float(v) for v in flange.find("origin").get("rpy").split()]
        flange.find("origin").set("rpy", " ".join(repr(v) for v in rpy))
        offsets = _offsets(robot, _guard_frames("ur5e", np.zeros(6)), np.zeros(6))
        self.assertGreater(float(np.abs(offsets["tool0"] - np.eye(4)).max()), _TOL)

    def test_a_millimetre_on_wrist_1_breaks_it(self) -> None:
        description = self.ur.load("ur5e")
        kinematics = dict(description.kinematics)
        kinematics["wrist_1"] = dataclasses.replace(kinematics["wrist_1"], x=kinematics["wrist_1"].x + 1e-3)
        moved = dataclasses.replace(description, kinematics=MappingProxyType(kinematics))
        robot = ET.fromstring(self.ur.render_urdf_from(moved))
        drift = _worst_drift(robot, lambda q: _guard_frames("ur5e", q))
        self.assertGreater(max(drift.values()), _TOL)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
