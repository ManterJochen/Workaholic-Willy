"""Outside rulers for the committed planner evidence: each hash restated from its recipe, not called (P0f F0).

A test that recomputes a committed hash with the code that wrote it proves the code still agrees with itself. P0f
rewrites that code (the guard's part composition and the pose rule take their joint count from the arm description),
so each hash is restated here once more from what its docstrings and the measured files say, with numpy and the
standard library only. Where the restatement and the committed file agree, the rewrite has a ruler that did not move
with it.

Measured 2026-09-23, Windows, numpy 2.4.4: 47 of 47 evidence files agree for every ruler below. Two conventions were
learnt from the files, not from the docstrings, and a rewrite that drops either moves committed hashes with no
geometric change at all:

- The guard turns a hand part into float64 and adds ``+ 0.0``. A turn by index leaves ``-0.0`` where the matrix
  product the guard uses gives ``+0.0``: without the ``+ 0.0`` 12 of the 12 Hand-E ``+Z`` files differ, and a turn
  that keeps the 2F-85's float32 moves its 6 ``+Z`` files instead.
- The pose rule is two roundings, ``low + (high - low) * u``. numpy's ``uniform`` gives exactly that on this
  machine for 6,000 of 6,000 draws; a fused multiply-add would round 3,043 of them differently. The ruler draws
  ``random()`` and does the arithmetic in Python floats, so it holds on any platform, and
  ``test_the_live_pose_rule_is_the_frozen_one`` says when numpy stops agreeing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "robot" / "safety" / "data"
PLANNING = ROOT / "src" / "robot" / "safety" / "planning" / "robot"
EVIDENCE = PLANNING / "evidence"
RETRACT_TABLE = PLANNING / "retract.yaml"

#: The hand parts every bundle names, and the one hand whose arrays ride in every arm bundle (hand.py:66).
HAND_PARTS = ("gripper", "lfinger", "rfinger")
ARM_BUNDLE_HAND = "robotiq_2f85"

#: The pose rule at P0e's end (scripts/arms/_matrix_gate.py:19-41): seed, draws, the two swept joints and the fixed
#: base of the sweep. ``-1.57`` is the literal the gate writes, not pi/2.
POSE_SEED = 20260915
RANDOM_POSES = 1000
SWEEP_5_DEG = (-180, 181, 10)
SWEEP_6_DEG = (-180, 181, 30)
SWEEP_BASE = (0.0, -1.57, 0.0, 0.0)
POSE_DOF = 6

_AXIS = {"+X": (0, 1.0), "-X": (0, -1.0), "+Y": (1, 1.0), "-Y": (1, -1.0), "+Z": (2, 1.0), "-Z": (2, -1.0)}


def evidence_files() -> list[Path]:
    return sorted(EVIDENCE.glob("*.json"))


def held(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _turn(approach: str, closing: str) -> np.ndarray:
    """The signed permutation that takes the hand model's ``+X`` closing and ``+Y`` approach to the declared ones."""
    turn = np.zeros((3, 3))
    row, sign = _AXIS[closing]
    turn[row, 0] = sign
    row, sign = _AXIS[approach]
    turn[row, 1] = sign
    turn[:, 2] = np.cross(turn[:, 0], turn[:, 1])
    return turn


def guard_sha256(record: dict[str, Any]) -> str:
    """The arrays the exact mesh guard composes for one evidence file, hashed as ``evidence.guard_sha256`` states it.

    Arm links from ``{arm}_collision_meshes.npz``; hand parts from the arm bundle for the 2F-85 and from
    ``{hand}_hand_meshes.npz`` otherwise, shifted along the model's ``+Y`` by the plate where the hand says its origin
    is the mounting face, then turned. No coupling boxes: no committed file was measured with any.
    """
    with np.load(DATA / f"{record['arm']}_collision_meshes.npz", allow_pickle=False) as bundle:
        arm = {key: np.array(bundle[key]) for key in bundle.files}
    if record["hand"] == ARM_BUNDLE_HAND:
        source, origin = arm, ""
    else:
        with np.load(DATA / f"{record['hand']}_hand_meshes.npz", allow_pickle=False) as bundle:
            source = {key: np.array(bundle[key]) for key in bundle.files}
        origin = str(source["gripper__origin"].reshape(-1)[0]) if "gripper__origin" in source else ""
    turn = _turn(str(record["approach"]), str(record["closing"]))
    turned = not np.array_equal(turn, np.eye(3))
    shift = float(record["coupling_mm"]) if origin == "mounting_face" else 0.0

    parts: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for key in arm:
        name, _, kind = key.partition("__")
        if kind == "v" and name not in HAND_PARTS:
            parts[name] = (arm[key], arm[f"{name}__f"], int(arm[f"{name}__frame"][0]))
    for name in HAND_PARTS:
        vertices = source[f"{name}__v"]
        if shift or turned:
            vertices = vertices.copy()
            if shift:
                vertices[:, 1] += shift
            if turned:
                out = np.empty(vertices.shape, dtype=np.result_type(vertices.dtype, np.float64))
                for row in range(3):
                    column = int(np.flatnonzero(turn[row])[0])
                    out[:, row] = vertices[:, column] * turn[row, column]
                vertices = out + 0.0
        parts[name] = (vertices, source[f"{name}__f"], int(source[f"{name}__frame"][0]))

    digest = hashlib.sha256()
    for name in sorted(parts):
        vertices, faces, frame = parts[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(frame).encode("utf-8"))
        for array in (vertices, faces):
            array = np.ascontiguousarray(array)
            digest.update(f"{array.dtype}{array.shape}".encode("utf-8"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def hand_map_sha256(hand: str) -> str:
    """The sphere list a hand's map hangs from ``tool0``, as canonical JSON (body_link.py:111-130)."""
    document = yaml.safe_load((PLANNING / f"{hand}_gripper_spheres.yml").read_text(encoding="utf-8")) or {}
    entries = (document.get("collision_spheres") or {}).get("tool0") or []
    spheres = [{"center": [float(v) for v in entry["center"]], "radius": float(entry["radius"])} for entry in entries]
    return hashlib.sha256(json.dumps(spheres, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def retract(record: dict[str, Any]) -> list[float]:
    """The one retract row an evidence file was measured from, matched on everything the table keys a row by."""
    table = yaml.safe_load(RETRACT_TABLE.read_bytes().decode("utf-8")) or {}
    default_placement = str((table.get("rule") or {}).get("placement") or "")
    placement = str(record["approach"]) + str(record["closing"])
    rows = [row for row in table.get("retracts") or [] if isinstance(row, dict)
            and row.get("arm") == record["arm"] and row.get("hand") == record["hand"]
            and abs(float(row.get("plate_mm", 0.0)) - float(record["coupling_mm"])) <= 1e-6
            and str(row.get("placement") or default_placement) == placement
            and abs(float(row.get("planner_margin_mm", 0.0)) - float(record["planner_margin_mm"])) <= 1e-6]
    if len(rows) != 1:
        raise AssertionError(f"{record['arm']} with {record['hand']} at {placement}: {len(rows)} retract rows, not 1")
    return [float(v) for v in rows[0]["retract"]]


def pose_set(start: list[float], *, seed: int = POSE_SEED) -> list[list[float]]:
    """The judged configurations: the retract, the seeded draws, then the sweep, in Python floats throughout."""
    low, high = -float(np.pi), float(np.pi)
    poses = [[float(v) for v in start]]
    for row in np.random.default_rng(seed).random(size=(RANDOM_POSES, POSE_DOF)):
        poses.append([low + (high - low) * float(u) for u in row])
    for q5 in np.radians(np.arange(*SWEEP_5_DEG)):
        for q6 in np.radians(np.arange(*SWEEP_6_DEG)):
            poses.append([*SWEEP_BASE, float(q5), float(q6)])
    return poses


def poses_sha256(poses: list[list[float]]) -> str:
    """The shape as text, then the float64 bytes in the order sent (_matrix_gate.py:44-55)."""
    asked = np.asarray(poses, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(f"{asked.shape}".encode("utf-8"))
    digest.update(np.ascontiguousarray(asked).tobytes())
    return digest.hexdigest()
