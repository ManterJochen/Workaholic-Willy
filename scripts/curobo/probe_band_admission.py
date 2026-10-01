"""Probe, on the GPU box, that the exact guard decides the planner's self pairs it judges, and nothing else (F1).

Run from the repository root with the repository's own interpreter; the sidecar is spawned with the cuRobo
environment's, as a cell spawns it::

    python scripts/curobo/probe_band_admission.py
    python scripts/curobo/probe_band_admission.py --curobo-python <cuRobo env python> --coal-prefix <coal env>

Why it exists. The owner's LOOK[0] on the UR10 with the Hand-E, (-42.5, -67.9, -78.6, -136.2, 94.6, -34.1) deg: the
exact meshes keep 19.04 mm on forearm|wrist_2 against the guard's 10 mm, and the planner's spheres, padded by the 4 mm
margin, overlap by 1.2 mm. Every motion out of it was refused. The owner decided on 2026-09-30 that the exact guard
decides the planner's self pairs it judges, where it accepted the configuration, and that the carried part, a pair the
guard does not check, the planner's world and its joint bounds stay the planner's. The CPU tests hold the driver with a
planner double; this holds the real sidecar, the real composed robot and the real driver path on this box.

What it does, in order, on an owner-like cell (UR10, Hand-E on a 20 mm plate, approach +Z and closing +X on the flange,
planner_margin_mm 4.0, the committed retract row and evidence file):

  1. starts the planner through the driver's own start, which refuses a sidecar whose composed_sha256 is not the one the
     committed evidence names, so a start here says the composed config did not change;
  2. for each case: the planner's plain check (check_js, as today), its report of every refused sample (the new
     report_refused), the exact guard's distance on each pair it names, and the driver's own decision
     (``URRobotArm._exact_guard_decides``), which is what every call site asks:
       LOOK[0] and LOOK[1] must come out VALID (the band); the folded wrist (wrist_1 80, wrist_2 -140), LOOK[0] with a
       box through the forearm, a joint outside the planner's bounds that the exact guard accepts (wrist_3 354.6 deg)
       and the Hand-E touching the UR10's own base, which the exact guard holds no part of, must stay INVALID;
  3. the wrist_1 edge at LOOK[0]'s wrist_2: -140.0 deg refused, -140.5 deg clear by the planner alone (the CPU replica
     read +0.11 and -0.06 mm);
  4. the kernel's self term against the pair naming over the wrist_1 x wrist_2 torus at LOOK[0], 5 degree steps: every
     configuration the self term refuses must name at least one pair, or a pair the kernel counts would be one the
     driver never saw (none may disagree);
  5. a move to LOOK[0] standing at LOOK[0], through ``move_to_joints`` with a recording controller: it must run;
  6. the escape leg a planned move out of LOOK[0] would take, and out of LOOK[1] (``URRobotArm._band_leg``), each at
     most ``band.BAND_LEG_MAX_DEG`` (20) per joint: LOOK[1]'s edge is wrist_1 +19.5 deg, which the owner's first cap of
     10 left to straight lines alone, and the grid, turning wrist_1 and wrist_2 in 1.33 deg steps, takes its leg at
     +20.0, the cap itself; and the screen of each pose.

It prints one line per case and writes ``logs/curobo/band_admission_probe.json``. Exit code 0 when every expectation
held, 1 otherwise. The planner is stopped on every way out. On the GPU it spawns a sidecar of its own, so stop the
console, the API and every other planner first: one sidecar on the GPU at a time.

The cell is built here, in code, and reads nothing of the cell's tree: it carries no wrist camera, so the robobrain.eye
housing's pairs are not in it. A pass says the kernel, the sidecar and the driver on this box decide as they should;
the cell's own looks are screened by ``real_cell --start-planner``, one line per look.

``--cpu-replica <cuRobo content root>`` runs the same cases on a CPU replica of the sidecar instead (:class:`
CpuReplicaClient`: the same composed robot, cuRobo's spheres by numpy forward kinematics of the descriptor's URDF, the
kernel's rules), for a box where an application-control policy refuses cuRobo's kernels, as Smart App Control refused
``cuda/bindings/cyruntime`` on the development box on 2026-09-30. Every line it prints then starts with ``[replica]``,
every log record it makes carries the same tag, its ``planner under test`` line says CPU REPLICA and its JSON names the
replica, so no line of it passes for a GPU line: it proves the driver's decision on the real geometry, not the GPU
kernel's parity. Its self term is the kernel's rule sphere pair by sphere pair, apart from the pair naming, and its rows
are built by the sidecar's own ``refused_rows`` and read by the client's own reader, so step 4 holds the naming's
shortcut and tolerance against the rule; the kernel itself only the GPU run holds.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LOOK0_DEG = (-42.5, -67.9, -78.6, -136.2, 94.6, -34.1)
LOOK1_DEG = (-89.08, -42.64, -105.24, -121.77, 88.65, -102.29)
FOLDED_DEG = (-42.5, -67.9, -78.6, 80.0, -140.0, -34.1)
#: LOOK[0] with wrist_3 past the planner's window, +-(2 pi - 0.1) rad, which is +-354.27 deg, and inside the joint-limit
#: guard's +-355: the exact guard accepts it (it is -5.4 deg to the meshes), so what refuses it is the planner's bound
#: alone. At 358 deg the joint-limit guard refused it too and the case proved nothing (review of F1, 2026-09-30).
OUT_OF_BOUNDS_DEG = (-42.5, -67.9, -78.6, -136.2, 94.6, 354.6)
#: The Hand-E's finger touching the UR10's own base mesh, 10 mm above the plate (review of F1, 2026-09-30): the exact
#: guard holds no part for the base and accepts it, and the planner's shoulder_link cushion is the only model of it.
BASE_TOUCH_DEG = (70.85, -113.42, 163.88, -98.73, -135.96, -157.92)
#: The torus step of the kernel-against-naming count, degrees.
TORUS_STEP_DEG = 5
EDGE_REFUSED_DEG = (-42.5, -67.9, -78.6, -140.0, 94.6, -34.1)
EDGE_CLEAR_DEG = (-42.5, -67.9, -78.6, -140.5, 94.6, -34.1)
#: A box far below the cell, which replaces the one through the forearm when that case is done.
FAR_BOX = {"name": "probe_far", "dims_m": [0.05, 0.05, 0.05], "pose": [0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0]}
#: What every line of a ``--cpu-replica`` run starts with, and every log record it makes carries, so a line pasted out
#: of it can never pass for a GPU line. A GPU run prints its lines untagged.
REPLICA_TAG = "[replica] "

#: The tag this run prints with: :data:`REPLICA_TAG` once :func:`tag_replica_output` ran, else nothing.
_tag = ""


def say(text: str = "") -> None:
    """Print ``text``, every line of it starting with this run's tag: every line either probe prints goes here."""
    print("\n".join(f"{_tag}{line}" for line in (str(text).splitlines() or [""])), flush=True)


def tag_replica_output() -> None:
    """Tag every line this run prints (:func:`say`) and every log record it makes with :data:`REPLICA_TAG`.

    Called before anything is printed or logged, so the run's first line is tagged too. The records are tagged where
    they are made, once each, whatever handler writes them.
    """
    global _tag
    _tag = REPLICA_TAG
    make = logging.getLogRecordFactory()

    def tagged(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = make(*args, **kwargs)
        record.msg = f"{REPLICA_TAG}{record.msg}"
        return record

    logging.setLogRecordFactory(tagged)


def _rad(values: "tuple[float, ...]") -> list[float]:
    return [math.radians(v) for v in values]


def _within_cap(leg: "list[float] | None", pose_deg: "tuple[float, ...]") -> bool:
    """Whether an escape leg was found and turns no joint more than the owner's cap from ``pose_deg``."""
    from src.robot.safety.planning.band import BAND_LEG_MAX_DEG

    return leg is not None and max(abs(math.degrees(a) - b) for a, b in zip(leg, pose_deg)) <= BAND_LEG_MAX_DEG + 1e-9


def _leg_said(leg: "list[float] | None", pose_deg: "tuple[float, ...]") -> str:
    """An escape leg as a person reads it: where it ends and which joints it turns, in degrees."""
    from src.robot.safety.planning.band import BAND_LEG_MAX_DEG

    if leg is None:
        return (f"none within {BAND_LEG_MAX_DEG:g} deg per joint: a planned move out of it is refused with the band's "
                "sentence, straight lines run")
    return ("(" + ", ".join(f"{math.degrees(v):.2f}" for v in leg) + ") deg, turning "
            + ", ".join(f"joint {i + 1} {math.degrees(a) - b:+.2f}" for i, (a, b) in enumerate(zip(leg, pose_deg))
                        if abs(math.degrees(a) - b) > 1e-6))


def _owner_like_config() -> Any:
    from src.config.schema.robot import RobotConfig

    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10", "motion_planner": "curobo"},
        "workspace_limits": {"x_min": -1500.0, "x_max": 1500.0, "y_min": -1500.0, "y_max": 1500.0,
                             "z_min": -500.0, "z_max": 1500.0},
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": {"payload": {"enforce": False},
                   "self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "kinematics_model": "ur10",
                                      "planner_margin_mm": 4.0}},
    })


def _forearm_box(joints: "list[float]") -> dict:
    """A 60 mm cube in the middle of the forearm at ``joints``, in the controller's base, metres."""
    import numpy as np

    from src.robot.safety._ur_kinematics import ur_link_origins_mm

    origins = ur_link_origins_mm("ur10", np.asarray(joints, dtype=np.float64))
    assert origins is not None
    middle = (np.asarray(origins[2], dtype=np.float64) + np.asarray(origins[3], dtype=np.float64)) / 2000.0
    return {"name": "probe_forearm_box", "dims_m": [0.06, 0.06, 0.06],
            "pose": [float(middle[0]), float(middle[1]), float(middle[2]), 1.0, 0.0, 0.0, 0.0]}


class CpuReplicaClient:
    """The sidecar's judgement on the CPU, for a box where cuRobo's kernels cannot load: labelled so in every line.

    It is NOT the GPU sidecar. It holds the robot the sidecar composes (``compose_for_cell`` with the cell's hand link,
    plates, retract row, margin and payload slots), places cuRobo's own spheres by numpy forward kinematics of the URDF
    the descriptor names (the research's replica, 2026-09-30, which reproduces LOOK[0]'s 1.21 mm), and applies the
    kernel's rules: the self term sphere pair by sphere pair (:meth:`_kernel_self_hit`, apart from the pair naming), the
    bounds as the planner's window, and the world as sphere against box at the clearance asked. Its report rows are
    built by the sidecar's own ``refused_rows`` and read by the client's own reader, and it sets the camera's boxes
    aside (``ignore_perceived``) with the sidecar's own ``_curobo_perceived.requested_aside`` and ``judged_world``, one
    flag per box it holds. It says who it is as the sidecar does, so the driver's own descriptor and evidence checks
    judge it. It plans nothing.
    """

    def __init__(self, arm: Any, content_root: Path) -> None:
        self._arm = arm
        self._root = content_root
        self._attach = 0
        self.joint_names: list[str] = []
        self.identity: Any = None
        self.last_refusal: Any = None
        self._boxes: list[dict] = []
        #: One flag per box it holds, by name: the storage ``SetAside`` switches, as the sidecar's cuRobo's.
        self._on: dict[str, bool] = {}
        self._joints: dict[str, dict] = {}

    # the client's lifecycle, as the planner glue drives it
    def reserve_world(self, reservation: Any) -> None:
        return None

    def reserve_attach_spheres(self, slots: int) -> None:
        self._attach = max(0, int(slots))

    def start(self) -> None:
        import hashlib
        import xml.etree.ElementTree as ET

        import yaml

        from src.robot.safety.planning._curobo_body_links import body_report, canonical_sha256, compose_for_cell
        from src.robot.safety.planning._curobo_pairs import SphereLayout
        from src.robot.safety.planning.body_link import HandLink, coupling_bodies
        from src.robot.safety.planning.curobo_client import SidecarIdentity
        from src.robot.safety.planning.robot.retract_table import read_retract

        descriptor = self._root / "configs" / "robot" / "willy_ur10.yml"
        raw = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
        hand = self._arm.safety_preflight.planner_hand(self._arm)
        link = HandLink.from_hand(hand)
        bodies = [link.to_dict(), *coupling_bodies(hand)]
        default_q = read_retract("ur10", hand.model, float(hand.coupling_mm), 4.0,
                                 placement=f"{link.placement.approach}{link.placement.closing}")
        loaded, evidence, _, _, _ = compose_for_cell(raw, bodies=bodies, margin_mm=4.0, attach_spheres=self._attach,
                                                     default_q=default_q)
        self._config = loaded
        layout = SphereLayout.from_robot_config(loaded)
        if layout is None:
            raise RuntimeError(f"{descriptor} names no sphere ownership, so the replica cannot name a pair")
        self._layout = layout
        import numpy as np

        # The kernel's pairs: two slots of different links the descriptor does not ignore, each once.
        owners = np.asarray(layout.owners)
        first, second = np.triu_indices(layout.slots, k=1)
        counted = (owners[first] != owners[second]) & ~np.asarray(
            [(min(a, b), max(a, b)) in layout.ignored for a, b in zip(owners[first], owners[second])], dtype=bool)
        self._kernel_pairs = (first[counted], second[counted])
        self._pads = np.asarray(layout.pads_m, dtype=np.float64)
        kinematics = loaded["robot_cfg"]["kinematics"]
        urdf = self._root / "assets" / str(kinematics["urdf_path"])
        root: Any = ET.parse(urdf).getroot()
        for joint in root.findall("joint"):
            origin: Any = joint.find("origin")
            axis: Any = joint.find("axis")
            child: Any = joint.find("child")
            parent: Any = joint.find("parent")
            self._joints[str(child.get("link"))] = {
                "name": joint.get("name"), "type": joint.get("type"), "parent": parent.get("link"),
                "xyz": [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()],
                "rpy": [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()],
                "axis": [float(v) for v in (axis.get("xyz") if axis is not None else "0 0 1").split()],
            }
        self._extra = dict(kinematics.get("extra_links") or {})
        self.joint_names = list(kinematics["cspace"]["joint_names"])
        self.identity = SidecarIdentity(
            provenance=raw.get("_provenance"),
            arm_descriptor_sha256=canonical_sha256(raw.get("robot_cfg", raw)),
            urdf_sha256=hashlib.sha256(urdf.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            composed_sha256=canonical_sha256(evidence),
            bodies=tuple(body_report(loaded, [str(body["link"]) for body in bodies])),
        )

    def close(self) -> None:
        return None

    def set_world(self, cuboids: Any, *args: Any, **kwargs: Any) -> int:
        self._boxes = [dict(box) for box in cuboids]
        self._on = {str(box["name"]): True for box in self._boxes}
        return len(self._boxes)

    # the box storage the sidecar's SetAside switches, here a flag per box it holds
    def names(self) -> "list[str]":
        return [str(box["name"]) for box in self._boxes]

    def enabled(self, name: str) -> bool:
        return self._on[name]

    def set_enabled(self, name: str, enabled: bool) -> None:
        self._on[name] = bool(enabled)

    def plan_joint(self, start: Any, goal: Any) -> None:
        self.last_refusal = None
        return None

    # the arithmetic
    def _pose(self, link: str, q: "list[float]") -> Any:
        import numpy as np

        if link in self._extra:
            extra = self._extra[link]
            fixed = [float(v) for v in extra["fixed_transform"]]
            transform = np.eye(4)
            transform[:3, :3] = _quat_wxyz(fixed[3:])
            transform[:3, 3] = fixed[:3]
            return self._pose(str(extra["parent_link_name"]), q) @ transform
        if link not in self._joints:
            return np.eye(4)
        joint = self._joints[link]
        transform = np.eye(4)
        transform[:3, :3] = _rpy(*joint["rpy"])
        transform[:3, 3] = joint["xyz"]
        if joint["type"] in ("revolute", "continuous"):
            turn = np.eye(4)
            turn[:3, :3] = _axis_angle(joint["axis"], q[self.joint_names.index(joint["name"])])
            transform = transform @ turn
        return self._pose(joint["parent"], q) @ transform

    def _spheres(self, q: "list[float]") -> Any:
        import numpy as np

        kinematics = self._config["robot_cfg"]["kinematics"]
        extra = kinematics.get("extra_collision_spheres") or {}
        rows: list[list[float]] = []
        for link in kinematics["collision_link_names"]:
            if link in extra:
                rows.extend([[0.0, 0.0, 0.0, -100.0]] * int(extra[link]))
                continue
            pose = self._pose(str(link), q)
            for sphere in kinematics["collision_spheres"][link]:
                centre = pose[:3, :3] @ np.asarray(sphere["center"], dtype=np.float64) + pose[:3, 3]
                rows.append([*centre.tolist(), float(sphere["radius"])])
        return np.asarray(rows, dtype=np.float64)[None, ...]

    def _kernel_self_hit(self, spheres: Any) -> bool:
        """cuRobo's self term as its kernel computes it, apart from the pair naming: every counted pair of spheres on
        its own, a sphere counted where its padded radius is not negative, a pair where the padded radii reach past the
        distance of the centres. No bounding-sphere shortcut and no tolerance: what the naming is held against."""
        import numpy as np

        pose = np.asarray(spheres, dtype=np.float64)[0]
        reach = pose[:, 3] + self._pads
        first, second = self._kernel_pairs
        gap = np.linalg.norm(pose[first, :3] - pose[second, :3], axis=1)
        return bool(np.any((reach[first] >= 0.0) & (reach[second] >= 0.0) & (reach[first] + reach[second] - gap > 0.0)))

    def _terms(self, q: "list[float]", clearance_m: float) -> "tuple[bool, bool, bool, Any]":
        import numpy as np

        from src.robot.drivers.ur.curobo_motion import PLANNER_JOINT_ENVELOPE_RAD

        spheres = self._spheres(q)
        bound_ok = all(low <= v <= high for v, low, high in zip(q, *PLANNER_JOINT_ENVELOPE_RAD))
        self_ok = not self._kernel_self_hit(spheres)
        world_ok = True
        for box in self._boxes:
            if not self._on.get(str(box["name"]), True):
                continue  # set aside for this judgement (SetAside)
            x, y, z, qw, qx, qy, qz = (float(v) for v in box["pose"])
            rotation = _quat_wxyz([qw, qx, qy, qz])
            local = (spheres[0, :, :3] - np.asarray([x, y, z])) @ rotation
            half = np.asarray(box["dims_m"], dtype=np.float64) / 2.0
            outside = np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)
            inside = np.minimum(np.max(np.abs(local) - half, axis=1), 0.0)
            gap = outside + inside - spheres[0, :, 3]
            real = spheres[0, :, 3] > 0.0
            if np.any(real & (gap < clearance_m if clearance_m > 0.0 else gap < 0.0)):
                world_ok = False
        return bound_ok, self_ok, world_ok, spheres

    def check_joints(self, configs: Any, *, clearance_mm: float = 0.0) -> Any:
        from src.robot.safety.planning import JointCheckVerdict

        judged = self.judge_joints(configs, clearance_mm=clearance_mm)
        if not judged.refused:
            return JointCheckVerdict(valid=True, first_invalid=None, checked=judged.checked, reason="replica: clear")
        first = judged.refused[0]
        return JointCheckVerdict(valid=False, first_invalid=first.index, checked=judged.checked,
                                 reason=f"the CPU replica refuses sample {first.index}", refusal=self._refusal(first,
                                                                                                          configs))

    def _refusal(self, row: Any, configs: Any) -> Any:
        from src.robot.safety.planning import StateRefusal, StateRefusalKind, StateWhere

        joints = tuple(float(v) for v in configs[row.index])
        if not row.self_ok:
            pair = row.pairs[0] if row.pairs else None
            return StateRefusal(where=StateWhere.PATH, kind=StateRefusalKind.SELF_COLLISION, joints=joints,
                                **({"link_a": pair.link_a, "link_b": pair.link_b, "depth_mm": pair.depth_mm}
                                   if pair is not None else {}))
        if not row.bound_ok:
            return StateRefusal(where=StateWhere.PATH, kind=StateRefusalKind.JOINT_LIMIT, joints=joints)
        return StateRefusal(where=StateWhere.PATH, kind=StateRefusalKind.WORLD, joints=joints)

    def judge_joints(self, configs: Any, *, clearance_mm: float = 0.0, name_pairs: bool = True,
                     ignore_perceived: Any = None) -> Any:
        """The sidecar's check_js with ``report_refused``: its rows by the sidecar's ``refused_rows``, read back by the
        client's ``_judgement_from_reply``, which refuses anything but a whole report as it does for the GPU. With
        ``ignore_perceived`` the camera's boxes are set aside by the sidecar's own ``requested_aside`` and
        ``judged_world``, and a request either refuses is a failed call, as the sidecar answers it. The replica is handed
        no carried part."""
        import numpy as np

        from src.robot.safety.planning._curobo_pairs import refused_rows
        from src.robot.safety.planning._curobo_perceived import PerceivedAsideError, judged_world, requested_aside
        from src.robot.safety.planning._curobo_plan_policy import CLEARANCE_KEY
        from src.robot.safety.planning._curobo_protocol import (
            IGNORE_PERCEIVED_KEY,
            PERCEIVED_IGNORED_KEY,
            PERCEIVED_PREFIX,
        )
        from src.robot.safety.planning.curobo_client import CuroboUnavailableError, _judgement_from_reply

        rows = [[float(v) for v in config] for config in configs]
        clearance_m = float(clearance_mm) / 1000.0
        request: dict[str, Any] = {} if ignore_perceived is None else {IGNORE_PERCEIVED_KEY: list(ignore_perceived)}
        try:
            aside = requested_aside(request, key=IGNORE_PERCEIVED_KEY, prefix=PERCEIVED_PREFIX, carrying=False)
            with judged_world(aside, lambda: self, prefix=PERCEIVED_PREFIX):
                terms = [self._terms(row, clearance_m)[:3] for row in rows]
        except PerceivedAsideError as exc:
            raise CuroboUnavailableError(f"the cuRobo planning CALL failed (not a planning verdict): {exc}") from exc
        bound = [0.0 if ok else 1.0 for ok, _, _ in terms]
        self_hit = [0.0 if ok else 1.0 for _, ok, _ in terms]
        world_hit = [0.0 if ok else 1.0 for _, _, ok in terms]
        passes = [all(judged) for judged in terms]
        report, named = refused_rows(
            passes, bound, self_hit, world_hit,
            lambda hits: np.concatenate([self._spheres(rows[index]) for index in hits], axis=0), self._layout,
            name_pairs=name_pairs)
        first = next((index for index, ok in enumerate(passes) if not ok), None)
        reply: dict[str, Any] = {"success": True, "valid": first is None, "first_invalid": first, "checked": len(rows),
                                 CLEARANCE_KEY: clearance_m, "refused": report, "pairs_named": named}
        if aside is not None:
            reply[PERCEIVED_IGNORED_KEY] = list(aside)
        return _judgement_from_reply(reply, sent=len(rows), clearance_m=clearance_m, named=bool(name_pairs),
                                     ignored=aside)


def _rpy(r: float, p: float, y: float) -> Any:
    import numpy as np

    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _axis_angle(axis: "list[float]", angle: float) -> Any:
    import numpy as np

    x, y, z = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([[c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                     [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                     [z * x * t - y * s, z * y * t + x * s, c + z * z * t]])


def _quat_wxyz(q: "list[float]") -> Any:
    import numpy as np

    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Probe:
    def __init__(self, arm: Any) -> None:
        self.arm = arm
        self.planner: Any = None
        self.cases: list[dict[str, Any]] = []
        self.failed: list[str] = []

    def expect(self, name: str, ok: bool, said: str) -> None:
        say(f"  [{'ok' if ok else 'FAIL'}] {name}: {said}")
        if not ok:
            self.failed.append(name)

    def case(self, name: str, degrees: "tuple[float, ...]", *, admitted: bool,
             guard_accepts: "bool | None" = None) -> dict:
        """The plain check, the report, the exact distances and the driver's decision for one configuration; where
        ``guard_accepts`` is given, the exact guard's own verdict on it is expected too (a case whose point is that the
        planner's refusal stands where the guard accepts)."""
        from src.robot.core import JointPositions, MotionCommand

        config = _rad(degrees)
        started = time.perf_counter()
        verdict = self.planner.check_joint_path([config], refresh=False)
        plain_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        report = self.planner.judge_joint_path([config], clearance_mm=0.0)
        report_ms = (time.perf_counter() - started) * 1000.0
        exact = self.arm.safety_preflight.exact_pairs(self.arm)
        rows = []
        for row in report.refused:
            pairs = [dict(pair.__dict__, exact_mm=exact.distance_mm(config, pair.link_a, pair.link_b))
                     for pair in (row.pairs or ())]
            rows.append({"index": row.index, "bound_ok": row.bound_ok, "self_ok": row.self_ok,
                         "world_ok": row.world_ok, "pairs": pairs})
        standing = (self.arm._exact_guard_decides(self.planner, [config], verdict, clearance_mm=0.0,
                                                  command=MotionCommand.MOVE_JOINTS)
                    if not verdict.valid else None)
        why = standing.reason if standing is not None else None
        valid = verdict.valid or standing is None
        guard = self.arm.safety_preflight.gate_joint_target(JointPositions(tuple(config)), arm=self.arm)
        result = {
            "case": name, "joints_deg": list(degrees), "plain_valid": verdict.valid,
            "plain_refusal": verdict.refusal.render() if getattr(verdict, "refusal", None) is not None else None,
            "report": rows, "exact_guard": "accepts" if guard is None else guard.message,
            "admitted": valid, "stands_because": why, "plain_ms": round(plain_ms, 1), "report_ms": round(report_ms, 1),
        }
        self.cases.append(result)
        named = "; ".join(f"{p['link_a']}|{p['link_b']} {p['depth_mm']:.2f} mm padded, {p['unpadded_depth_mm']:.2f} "
                          + ("unpadded, not a pair the exact guard holds" if p["exact_mm"] is None
                             else f"unpadded, exact {round(p['exact_mm'], 2)} mm")
                          for r in rows for p in r["pairs"])
        terms = "; ".join(f"bound_ok={r['bound_ok']} self_ok={r['self_ok']} world_ok={r['world_ok']}" for r in rows)
        self.expect(name, valid is admitted,
                    f"{'VALID' if valid else 'INVALID'} (plain check {'valid' if verdict.valid else 'invalid'}; "
                    f"{terms or 'no refused sample'}; {named or 'no pair'}; exact guard: {result['exact_guard']}"
                    + (f"; stands: {why}" if why else "") + ")")
        if guard_accepts is not None:
            self.expect(f"{name}: the exact guard {'accepts' if guard_accepts else 'refuses'} it",
                        (guard is None) is guard_accepts, str(result["exact_guard"]))
        return result


def run(args: argparse.Namespace) -> int:
    if args.cpu_replica:
        tag_replica_output()
    if args.curobo_python:
        os.environ["WILLY_CUROBO_PYTHON"] = args.curobo_python
    if args.coal_prefix:
        os.environ["WILLY_COAL_PREFIX"] = args.coal_prefix
    if args.stderr_log:
        os.environ["WILLY_CUROBO_STDERR"] = args.stderr_log
    from src.robot.core import JointPositions, MotionCommand
    from src.robot.drivers.ur.arm import URRobotArm
    from src.robot.safety.planning.environment import resolve_collision_engine

    replica: "CpuReplicaClient | None" = None
    arm = URRobotArm(_owner_like_config())
    if args.cpu_replica:
        replica = CpuReplicaClient(arm, Path(args.cpu_replica))
        setattr(arm, "_curobo_client_factory", lambda: replica)  # noqa: B010 (the probe's own stand-in, by name)
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = _rad(LOOK0_DEG)
    arm._conn.moveJ.return_value = True
    probe = Probe(arm)
    out: dict[str, Any] = {"engine": resolve_collision_engine().backend,
                           "planner": "CPU REPLICA, not the GPU sidecar" if replica is not None else "GPU sidecar"}
    say(f"planner under test: {out['planner']}")
    started = time.perf_counter()
    try:
        identity = arm.start_planner()
        out["start_s"] = round(time.perf_counter() - started, 1)
        out["composed_sha256"] = identity.composed_sha256
        say(f"planner started in {out['start_s']} s, composed_sha256 {identity.composed_sha256} (the start refuses "
            f"any other than the committed evidence's), exact engine {out['engine']}")
        probe.planner = arm._curobo_ur
        # A world far from every case, so no case reads the world the sidecar booted with.
        probe.planner.set_world([FAR_BOX])
        say("cases:")
        probe.case("LOOK[0]", LOOK0_DEG, admitted=True)
        probe.case("LOOK[1]", LOOK1_DEG, admitted=True)
        probe.case("folded wrist (wrist_1 80, wrist_2 -140)", FOLDED_DEG, admitted=False)
        probe.case("a joint outside the planner's bounds the exact guard accepts (wrist_3 354.6 deg)",
                   OUT_OF_BOUNDS_DEG, admitted=False, guard_accepts=True)
        probe.case("the Hand-E touching the UR10's own base, which the exact guard holds no part of", BASE_TOUCH_DEG,
                   admitted=False, guard_accepts=True)
        probe.planner.set_world([_forearm_box(_rad(LOOK0_DEG))])
        try:
            probe.case("LOOK[0] with a box through the forearm", LOOK0_DEG, admitted=False)
        finally:
            probe.planner.set_world([FAR_BOX])
        probe.case("LOOK[0] again, the box gone", LOOK0_DEG, admitted=True)

        say("the wrist_1 edge at LOOK[0]'s wrist_2, the planner alone:")
        edge = {}
        for name, degrees, want in (("-140.0 deg", EDGE_REFUSED_DEG, False), ("-140.5 deg", EDGE_CLEAR_DEG, True)):
            verdict = probe.planner.check_joint_path([_rad(degrees)], refresh=False)
            depth = getattr(getattr(verdict, "refusal", None), "depth_mm", None)
            edge[name] = {"valid": verdict.valid, "depth_mm": depth}
            probe.expect(f"wrist_1 {name}", verdict.valid is want,
                         f"planner {'clears' if verdict.valid else f'refuses ({verdict.refusal.render()})'}")
        out["edge"] = edge

        say(f"the self term against the pair naming, wrist_1 x wrist_2 at LOOK[0]'s other joints, {TORUS_STEP_DEG} "
            "deg steps:")
        torus = [_rad((*LOOK0_DEG[:3], float(w1), float(w2), LOOK0_DEG[5]))
                 for w1 in range(-180, 180, TORUS_STEP_DEG) for w2 in range(-180, 180, TORUS_STEP_DEG)]
        started = time.perf_counter()
        judged = probe.planner.judge_joint_path(torus, clearance_mm=0.0)
        by_self = [row for row in judged.refused if not row.self_ok]
        unnamed = [row.index for row in by_self if not row.pairs]
        out["torus"] = {"configurations": len(torus), "refused_by_self": len(by_self), "unnamed": unnamed[:20],
                        "unnamed_count": len(unnamed), "seconds": round(time.perf_counter() - started, 1)}
        probe.expect("every configuration the self term refuses names a pair", not unnamed,
                     f"{len(by_self)} of {len(torus)} refused by the self term, {len(unnamed)} with no pair named"
                     + (" (the replica's self term is the kernel's rule sphere pair by sphere pair; the GPU run holds "
                        "the kernel itself)" if replica is not None else ""))

        say("the driver's own moves:")
        with arm.without_camera_world("probe: no camera on this bench"):
            result = arm.move_to_joints(JointPositions(tuple(_rad(LOOK0_DEG))))
        probe.expect("move_to_joints(LOOK[0]) standing at LOOK[0]", result.ok,
                     f"{result.status.value}: {result.message}; moveJ sent {arm._conn.moveJ.call_count} time(s)")
        out["move_to_look0"] = {"status": result.status.value, "message": result.message}

        assert arm.safety_preflight is not None
        exact = arm.safety_preflight.exact_pairs(arm)
        report = probe.planner.judge_joint_path([_rad(LOOK0_DEG)], clearance_mm=0.0)
        escape = arm._band_leg(probe.planner, exact, _rad(LOOK0_DEG), report.refused[0], escape=True,
                               command=MotionCommand.MOVE_JOINTS) if report.refused else None
        probe.expect("escape leg out of LOOK[0]", _within_cap(escape, LOOK0_DEG), _leg_said(escape, LOOK0_DEG))
        out["escape_deg"] = None if escape is None else [math.degrees(v) for v in escape]
        # LOOK[1] sits deeper in the band: its edge is wrist_1 +19.5 deg, and the grid, turning wrist_1 and wrist_2 in
        # 1.33 deg steps, takes its leg at +20.0, the owner's cap itself.
        report = probe.planner.judge_joint_path([_rad(LOOK1_DEG)], clearance_mm=0.0)
        escape1 = arm._band_leg(probe.planner, exact, _rad(LOOK1_DEG), report.refused[0], escape=True,
                                command=MotionCommand.MOVE_JOINTS) if report.refused else None
        probe.expect("escape leg out of LOOK[1]", _within_cap(escape1, LOOK1_DEG), _leg_said(escape1, LOOK1_DEG))
        out["escape_look1_deg"] = None if escape1 is None else [math.degrees(v) for v in escape1]

        say("screens:")
        screens = {}
        for name, degrees in (("LOOK[0]", LOOK0_DEG), ("LOOK[1]", LOOK1_DEG), ("folded", FOLDED_DEG)):
            screen = arm.screen_configuration(JointPositions(tuple(_rad(degrees))))
            screens[name] = {"verdict": screen.verdict.value, "line": screen.line(name)}
            say(f"  {screen.line(name)}")
        out["screens"] = screens
    except Exception as exc:  # noqa: BLE001 (reported with its trace, and the planner still stops)
        probe.failed.append(f"{type(exc).__name__}: {exc}")
        out["error"] = traceback.format_exc(limit=12)
        say(out["error"])
    finally:
        arm.stop_planner()
    out["cases"] = probe.cases
    out["failed"] = probe.failed
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    say(f"{'ALL EXPECTATIONS HELD' if not probe.failed else 'FAILED: ' + '; '.join(probe.failed)}; wrote {target}")
    return 0 if not probe.failed else 1


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curobo-python", default="", help="the cuRobo environment's python (WILLY_CUROBO_PYTHON)")
    parser.add_argument("--coal-prefix", default="", help="the Coal environment (WILLY_COAL_PREFIX)")
    parser.add_argument("--stderr-log", default="", help="where the sidecar's stderr goes (WILLY_CUROBO_STDERR)")
    parser.add_argument("--cpu-replica", default="", metavar="CONTENT_ROOT",
                        help="judge with a CPU replica of the sidecar instead of the GPU one, on the descriptor and URDF "
                             "under this cuRobo content root (ext_deps/curobo/curobo/content): for a box whose cuRobo "
                             "kernels an application-control policy blocks. Every line it prints then starts with "
                             "[replica]")
    parser.add_argument("--out", default=str(REPO / "logs" / "curobo" / "band_admission_probe.json"))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
