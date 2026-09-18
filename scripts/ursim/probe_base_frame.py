"""Which base frame the controller reports in: the DH chain's, or the URDF ``base_link`` the planner is rooted at.

The cuRobo planner is rooted at UR's URDF ``base_link``, and that frame is the DH base turned half a
turn about Z (``scripts/curobo/probe_payload_attach.py`` ``frame`` measures 0.0001 mm against the DH
chain turned 180 degrees, 1019 mm without the turn). The UR driver turns every pose it hands the
planner by that half turn, which is right only where the controller reports poses in the DH base.
This asks the controller which frame that is.

It reads, and commands nothing: the joints and the TCP pose over RTDE receive, then, where RTDE
control connects (the controller in Remote Control), the active TCP offset, so the TCP the
controller reports can be compared with the DH flange carrying that offset. Without control the
positions are compared alone, and the answer is still decisive wherever the flange stands well off
the base axis: the two frames put it ``2 * |x, y|`` apart.

Run (URSim up, in Remote Control for the exact comparison)::

    python scripts/ursim/probe_base_frame.py --json logs/step8/probe_base_frame.json
    python scripts/ursim/probe_base_frame.py --model ur3e

Exit codes: ``0`` the controller reports in the DH frame, ``1`` in the frame turned half a turn,
``3`` neither, ``2`` the controller could not be read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402

#: How close the controller's TCP has to be to one frame's prediction, with the offset known.
_EXACT_MM = 1.0


def _rotvec_matrix(rv: Any) -> np.ndarray:
    v = np.asarray(rv, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)


def _pose_matrix_mm(pose: Any) -> np.ndarray:
    """A UR pose ``[x, y, z, rx, ry, rz]`` (metres, rotation vector) as a 4x4 in millimetres."""
    out = np.eye(4)
    out[:3, :3] = _rotvec_matrix(pose[3:6])
    out[:3, 3] = np.asarray(pose[:3], dtype=np.float64) * 1000.0
    return out


def _yaw(deg: float) -> np.ndarray:
    out = np.eye(4)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    out[:2, :2] = [[c, -s], [s, c]]
    return out


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_base_frame", description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default="ur5e", help="the DH table to compare with (URSim UR5 is the ur5e)")
    parser.add_argument("--json", default="logs/step8/probe_base_frame.json")
    args = parser.parse_args(argv)

    try:
        import rtde_receive  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"ur_rtde is not installed in this interpreter: {exc}", file=sys.stderr)
        return 2
    try:
        receive = rtde_receive.RTDEReceiveInterface(args.host)
        q = [float(v) for v in receive.getActualQ()]
        tcp = [float(v) for v in receive.getActualTCPPose()]
    except Exception as exc:  # noqa: BLE001 (any failure here means the same thing: nothing was read)
        print(f"the controller at {args.host} could not be read over RTDE: {exc}", file=sys.stderr)
        return 2
    report: dict[str, Any] = {"host": args.host, "model": args.model, "q_rad": q, "tcp_pose": tcp}

    offset: "list[float] | None" = None
    try:
        import rtde_control  # type: ignore[import-not-found]

        control = rtde_control.RTDEControlInterface(args.host)
        try:
            offset = [float(v) for v in control.getTCPOffset()]
        finally:
            control.stopScript()
            control.disconnect()
    except Exception as exc:  # noqa: BLE001 (control is optional; positions alone still decide)
        report["tcp_offset_unread"] = f"{type(exc).__name__}: {exc}"
    report["tcp_offset"] = offset

    frames = ur_link_transforms_mm(args.model, np.asarray(q, dtype=np.float64))
    if frames is None:
        print(f"no DH table for {args.model!r}", file=sys.stderr)
        return 2
    flange = np.asarray(frames[-1], dtype=np.float64)
    reported = _pose_matrix_mm(tcp)
    tool = _pose_matrix_mm(offset) if offset is not None else None
    compared: dict[str, Any] = {}
    for yaw in (0, 180):
        placed = _yaw(yaw) @ flange
        row: dict[str, Any] = {"flange_to_tcp_mm": round(float(np.linalg.norm(placed[:3, 3] - reported[:3, 3])), 3)}
        if tool is not None:
            predicted = placed @ tool
            row["tcp_error_mm"] = round(float(np.linalg.norm(predicted[:3, 3] - reported[:3, 3])), 4)
            row["tcp_rotation_deg"] = round(_angle_deg(predicted[:3, :3], reported[:3, :3]), 4)
        compared[f"yaw_{yaw}"] = row
    report["compared"] = compared
    report["flange_off_axis_mm"] = round(float(np.linalg.norm(flange[:2, 3])), 1)

    if tool is not None:
        exact = [yaw for yaw in (0, 180) if compared[f"yaw_{yaw}"]["tcp_error_mm"] < _EXACT_MM
                 and compared[f"yaw_{yaw}"]["tcp_rotation_deg"] < 0.1]
    else:
        # No offset read: a TCP stands at most this far from its flange on any tool a UR carries.
        reach = 350.0
        exact = [yaw for yaw in (0, 180) if compared[f"yaw_{yaw}"]["flange_to_tcp_mm"] < reach]
        report["bound_mm"] = reach
    report["verdict"] = (
        "the controller reports in the DH frame" if exact == [0]
        else "the controller reports in the DH frame turned half a turn" if exact == [180]
        else "neither frame explains the controller's TCP"
    )
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("compared", "tcp_offset", "flange_off_axis_mm", "verdict")}, indent=2))
    return 0 if exact == [0] else 1 if exact == [180] else 3


if __name__ == "__main__":
    raise SystemExit(main())
