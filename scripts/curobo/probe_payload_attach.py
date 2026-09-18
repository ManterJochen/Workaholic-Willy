"""A carried part on the real planner, measured against a table and a post the empty hand clears.

The carried part model is on by default and its length is the cell's to declare. This asks the installed planner what
that model does, on a bare client composed the way a cell composes one: the ur5e descriptor, the 2F-85 as a body link
on the real flange's +Z with its closing along +X, a planner margin of 4 mm, the committed retract for that
combination, and sixteen attach spheres reserved before the start. The hand points straight down, its fingertips 5 mm
above a table; the part is the box ``self_envelope.carried_part_box`` hangs past the fingertips for a 30 mm grip.

The world is placed from the planner's own forward kinematics. The planner is rooted at the URDF ``base_link``, half a
turn about Z from the DH chain's base, which is the controller's, so a table placed from the DH chain stands a metre
from the hand and every claim is judged against empty space. The frame is therefore measured first and reported
(``frame``), and a positive control has to refuse before any claim is judged: a table raised into the fingers must
make the empty hand invalid, or the probe is not looking where it thinks.

Claims, each against the same 100 mm lift along the approach, judged with ``check_joints``:

* (p) the control: with the table top 20 mm above the fingertips, the empty hand is invalid from its first sample;
* (a) with nothing attached the lift is valid;
* (b) with a 120 mm part attached it is invalid from its first sample: the part stands in the table;
* (c) every swept part length is valid in every one of ``--repeat`` attaches with ``_CARRIED_AIR_MM`` of air between
  the part's end and the table;
* (d) with the table gone and a post 30 mm past the fingertips, a 60 mm part is invalid and the empty hand is valid.

If (p) fails nothing else is judged. If (b) is valid or (c) invalid, the model is not what the real cell checklist
promises.

The sphere fit of a box reaches past it: a lift with 2 mm of air under a 3 mm part fails (c). The sweep (``sweep``)
raises the air under the part's end for each length and records where the lift passes; one sweep passed from 5 mm
under the 3 mm part, 10 mm under the 60 and 8 mm under the 120. That is a refusal near a surface and never a pass
through one, the price of carrying a part at all.

The fit is not the same twice. Every attach fits its spheres again and the fit differs: another sweep needed 15 mm
under the 60 mm part and 10 mm under the 120. So one sweep is one sample. The sweep attaches ``--repeat`` times at
every length and every air and counts the passes, the bound is read from one run, and (c) confirms it on another. With
ten attaches a cell, every attach passes from 5 mm of air under a 3 mm part, 20 under a 30 mm part, 15 under 60, 12
under 90 and 12 under 120, so the bound is 20 mm.

Run from the repository root with the project venv, alone on the box (it starts the sidecar on the GPU)::

    .venv/Scripts/python.exe scripts/curobo/probe_payload_attach.py --json logs/curobo/probe_payload_attach.json

Exit codes: ``0`` every claim holds, ``1`` one does not, ``2`` the planner could not be started.
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

from src.config.schema.robot import RobotConfig  # noqa: E402
from src.contracts import chosen  # noqa: E402
from src.robot.drivers.sim.robot_models import curobo_arm_descriptor  # noqa: E402
from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402
from src.robot.safety.planning import CuroboPlanClient, CuroboUnavailableError  # noqa: E402
from src.robot.safety.planning.body_link import HandLink, coupling_bodies  # noqa: E402
from src.robot.safety.planning.hand import planner_hand  # noqa: E402
from src.robot.safety.planning.robot.retract_table import read_retract  # noqa: E402
from src.robot.safety.planning.self_envelope import carried_part_box, hand_spheres  # noqa: E402
from src.robot.safety.planning.world import planner_cuboid  # noqa: E402

_ARM = "ur5e"
_MARGIN_MM = 4.0
_SLOTS = 16
_GRIP_MM = 30.0
_LATERAL_MM = 10.0
_TABLE_GAP_MM = 5.0
_CONTROL_DEPTH_MM = 20.0
_LIFT_MM = 100.0
_STEPS = 10
_SWEEP_LENGTHS_MM = (3.0, 30.0, 60.0, 90.0, 120.0)
_SWEEP_AIR_MM = (2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0)
#: The air under a carried part's end the planner needs in every attach, read from a repeated sweep (the 30 mm part
#: passed ten of ten from 20 mm and eight of ten at 15) and confirmed by (c) on another run.
_CARRIED_AIR_MM = 20.0
#: Tool pointing down: the classic UR home with the wrist turned to face the bench.
_DOWN_Q = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)


def _cell() -> RobotConfig:
    """The combination the committed ``ur5e_robotiq_2f85_c0mm_+Z+X_m4mm_a0`` file names, declared as matrix_gate does."""
    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": _ARM, "motion_planner": "curobo"},
        "safety": {"payload": {"enforce": False},
                   "self_collision": {"backend": "fcl", "kinematics_model": _ARM, "planner_margin_mm": _MARGIN_MM}},
        "gripper": {"model": "robotiq_2f85", "coupling_plates": [], "tool_frame": {
            "source": "willy", "offset_mm": [0.0, 0.0, 1.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
    })


def _flange(q: Any) -> np.ndarray:
    frames = ur_link_transforms_mm(_ARM, np.asarray(q, dtype=np.float64))
    assert frames is not None
    return np.asarray(frames[-1], dtype=np.float64)


def _solve(q0: np.ndarray, target: np.ndarray, *, iterations: int = 60) -> np.ndarray:
    """Damped least squares on the DH chain: the configuration near ``q0`` whose flange is ``target``."""
    q = np.asarray(q0, dtype=np.float64).copy()
    for _ in range(iterations):
        now = _flange(q)
        position = (target[:3, 3] - now[:3, 3]) / 1000.0
        rotation = target[:3, :3] @ now[:3, :3].T
        angle = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                          rotation[1, 0] - rotation[0, 1]]) / 2.0
        error = np.concatenate([position, angle])
        if float(np.linalg.norm(error)) < 1e-7:
            break
        jacobian = np.zeros((6, 6))
        for joint in range(6):
            nudged = q.copy()
            nudged[joint] += 1e-6
            moved = _flange(nudged)
            jacobian[:3, joint] = (moved[:3, 3] - now[:3, 3]) / 1000.0 / 1e-6
            turn = moved[:3, :3] @ now[:3, :3].T
            jacobian[3:, joint] = np.array([turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0],
                                            turn[1, 0] - turn[0, 1]]) / 2.0 / 1e-6
        q += jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(6), error)
    return q


def _quat_wxyz_matrix(q: Any) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _yaw(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _planner_tool0(client: CuroboPlanClient, q: Any) -> tuple[np.ndarray, np.ndarray]:
    fk = client.fk([float(v) for v in q])
    if fk is None:
        raise RuntimeError("the planner returned no forward kinematics")
    return np.asarray(fk[0], dtype=np.float64) * 1000.0, _quat_wxyz_matrix(fk[1])


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_payload_attach", description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=str, default="logs/curobo/probe_payload_attach.json")
    parser.add_argument("--repeat", type=int, default=10, help="attaches per swept cell and per (c) length")
    parser.add_argument("--carried-air-mm", type=float, default=_CARRIED_AIR_MM)
    args = parser.parse_args(argv)
    air_mm = float(args.carried_air_mm)
    repeat = max(1, int(args.repeat))

    cell = _cell()
    hand = planner_hand(cell)
    if not chosen(hand) or not chosen(hand.placement):
        print("the cell places no hand on the flange, so there is no carried part to hang from it", file=sys.stderr)
        return 2
    link = HandLink.from_hand(hand)
    spheres = hand_spheres(hand, _ARM)
    assert spheres is not None
    approach_tool0 = np.asarray(hand.placement.approach_in_tool0, dtype=np.float64)
    tip_mm = max(float(np.dot(np.asarray(s.start_mm), approach_tool0)) + float(s.radius_mm) for s in spheres)

    # The lift, solved on the DH chain: a joint path is the same motion in either base frame.
    flange0 = _flange(_DOWN_Q)
    approach_dh = flange0[:3, :3] @ approach_tool0
    configs = [np.asarray(_DOWN_Q, dtype=np.float64)]
    for step in range(1, _STEPS + 1):
        target = flange0.copy()
        target[:3, 3] = flange0[:3, 3] - approach_dh * (_LIFT_MM * step / _STEPS)
        configs.append(_solve(configs[-1], target))
    path = [[float(v) for v in q] for q in configs]
    report: dict[str, Any] = {
        "arm": _ARM, "hand": hand.model, "placement": link.placement.approach + link.placement.closing,
        "tip_mm": tip_mm, "repeat": repeat, "carried_air_mm": air_mm,
        "lift_dh_error_mm": round(float(np.linalg.norm(
            _flange(configs[-1])[:3, 3] - (flange0[:3, 3] - approach_dh * _LIFT_MM))), 4),
    }

    default_q = read_retract(_ARM, hand.model, float(hand.coupling_mm), _MARGIN_MM,
                             placement=f"{link.placement.approach}{link.placement.closing}")
    client = CuroboPlanClient(robot_config=curobo_arm_descriptor(_ARM), self_collision_margin_mm=_MARGIN_MM,
                              body_links=[link.to_dict(), *coupling_bodies(hand)], default_q=default_q)
    client.reserve_attach_spheres(_SLOTS)
    try:
        client.start()
    except CuroboUnavailableError as exc:
        print(f"the planner could not be started: {exc}", file=sys.stderr)
        return 2

    def check(world: list[dict], length_mm: "float | None") -> dict[str, Any]:
        client.detach_payload()
        client.set_world(world)
        attached = None
        if length_mm is not None:
            dims_mm, centre_mm = carried_part_box(hand, spheres, grip_width_mm=_GRIP_MM, length_mm=length_mm,
                                                  lateral_margin_mm=_LATERAL_MM)
            pose = [*(float(c) / 1000.0 for c in centre_mm), 1.0, 0.0, 0.0, 0.0]
            attached = bool(client.attach_payload(path[0], [float(d) / 1000.0 for d in dims_mm], pose))
        verdict = client.check_joints(path)
        return {"attached": attached, "valid": bool(verdict.valid), "first_invalid": verdict.first_invalid,
                "reason": str(verdict.reason)}

    try:
        # The frame, measured: the planner's tool0 against the DH flange, the base turned by 0 and by 180 degrees.
        tool0_mm, tool0_rot = _planner_tool0(client, path[0])
        report["frame"] = {
            f"yaw_{yaw}": {
                "position_mm": round(float(np.linalg.norm(_yaw(yaw) @ flange0[:3, 3] - tool0_mm)), 4),
                "rotation_deg": round(_angle_deg(_yaw(yaw) @ flange0[:3, :3], tool0_rot), 4),
            }
            for yaw in (0, 180)
        }
        approach = tool0_rot @ approach_tool0
        report["approach_in_planner_base"] = [round(float(v), 4) for v in approach]
        if float(approach[2]) > -0.99:
            report["refused"] = "the probe's joint vector does not point the hand down in the planner's frame"
            print(json.dumps(report, indent=2))
            return 1
        end_mm, _ = _planner_tool0(client, path[-1])
        report["lift_planner_error_mm"] = round(float(np.linalg.norm(end_mm - (tool0_mm - approach * _LIFT_MM))), 4)
        tip = tool0_mm + approach * tip_mm
        report["tip_in_planner_base_mm"] = [round(float(v), 2) for v in tip]

        def table(top_mm: float) -> dict:
            return planner_cuboid("table", (float(tip[0]), float(tip[1]), top_mm - 25.0), (600.0, 600.0, 50.0))

        post = planner_cuboid("post", tuple(float(v) for v in tip + approach * 30.0), (20.0, 20.0, 20.0))
        claims = {"p_empty_in_a_raised_table": check([table(float(tip[2]) + _CONTROL_DEPTH_MM)], None)}
        if claims["p_empty_in_a_raised_table"]["valid"]:
            report["claims"] = claims
            report["refused"] = "the control passed: a table in the fingers did not refuse the empty hand"
        else:
            below = float(tip[2]) - _TABLE_GAP_MM
            claims.update({
                "a_empty_over_the_table": check([table(below)], None),
                "b_120_mm_over_the_table": check([table(below)], 120.0),
                "d_60_mm_beside_the_post": check([post], 60.0),
                "d_empty_beside_the_post": check([post], None),
            })
            for length in _SWEEP_LENGTHS_MM:
                runs = [check([table(float(tip[2]) - length - air_mm)], length) for _ in range(repeat)]
                claims[f"c_{length:g}_mm_with_{air_mm:g}_mm_air"] = {
                    "attached": all(run["attached"] is True for run in runs),
                    "valid": all(run["valid"] for run in runs),
                    "passes": sum(1 for run in runs if run["valid"]), "of": repeat,
                }
            sweep: dict[str, Any] = {}
            for length in _SWEEP_LENGTHS_MM:
                rows = {}
                for air in _SWEEP_AIR_MM:
                    rows[f"{air:g}"] = sum(
                        1 for _ in range(repeat) if check([table(float(tip[2]) - length - air)], length)["valid"])
                always = [air for air in _SWEEP_AIR_MM if rows[f"{air:g}"] == repeat]
                sweep[f"{length:g}_mm"] = {"passes_by_air_mm": rows, "of": repeat,
                                           "first_air_passing_every_attach_mm": always[0] if always else None}
            report["sweep"] = sweep
    finally:
        client.close()
    report["claims"] = claims
    control = claims["p_empty_in_a_raised_table"]
    passed = {"p": not control["valid"] and control["first_invalid"] == 0}
    if passed["p"]:
        passed.update({
            "a": claims["a_empty_over_the_table"]["valid"],
            "b": claims["b_120_mm_over_the_table"]["attached"] is True
            and not claims["b_120_mm_over_the_table"]["valid"]
            and claims["b_120_mm_over_the_table"]["first_invalid"] == 0,
            "c": all(claims[f"c_{length:g}_mm_with_{air_mm:g}_mm_air"]["attached"] is True
                     and claims[f"c_{length:g}_mm_with_{air_mm:g}_mm_air"]["valid"]
                     for length in _SWEEP_LENGTHS_MM),
            "d": claims["d_60_mm_beside_the_post"]["attached"] is True
            and not claims["d_60_mm_beside_the_post"]["valid"]
            and claims["d_empty_beside_the_post"]["valid"],
        })
    report["passed"] = passed
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"frame: {report['frame']}")
    for key, value in report.get("sweep", {}).items():
        print(f"sweep {key}: every attach passes from {value['first_air_passing_every_attach_mm']} mm of air, "
              f"passes of {value['of']} by air: {value['passes_by_air_mm']}")
    for key, value in claims.items():
        print(f"{key}: {value}")
    ok = all(passed.values()) and len(passed) == 5
    print(f"carried part claims: {passed} -> {'PASS' if ok else 'FAIL'}")
    print(f"numbers written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
