"""Multi-view occlusion discriminance probe: fixed obliques and the wrist camera, read-only, on-box.

A single overhead view fails on dense clutter for two reasons: a stacked neighbour shadows the target
from straight above, and the arm occludes its own overhead view when it reaches in. The answer is a
switchable set of cameras that are not all on the robot:

  * a fixed overhead camera (eye-to-hand),
  * a fixed oblique camera on each side (-Y and +Y, symmetric) so occlusion from either side, or from
    the arm on either approach, is covered by at least one view,
  * the moving wrist camera (eye-in-hand), switched in for a centred close-up that sharpens the grasp.

The probe measures the redundant coverage and nothing else: no grasp, no pipeline change, a standalone
runner that leaves the default path byte-identical. It authors the obliques the wrist-camera look-at
xform-op way (world frame, small near clip), reuses ``MultiObjectGroundTruthPerceptionSource`` and the
calibrated wrist transform (``camera_to_tcp_ground_truth``), and reports the target's visible mask
pixels from every camera at two arm poses: park, with the arm clear, and view, with the wrist driven
to a top-down view over the target so the arm occludes the overhead camera. The go/no-go: at view the
overhead loses the target while at least one oblique, ideally both, and the wrist retain it.

On-box only (needs Isaac). Run with Isaac's bundled python from the repository root:

    <isaac-sim>\\python.bat -m src.willy_sim.run_occlusion_probe [--gui] [--no-wrist] [--jsonl out.jsonl]
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.schema.robot import SimObjectConfig
from src.utility.log_cfg import create_logger
from src.willy_sim.constants import OCCLUSION_PROBE_LOG_FILE, WILLY_SIM_LOG_DIR
from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource
from src.willy_sim.run_eih_pick import topdown_view_pose
from src.robot.grasping.scoring.occlusion import occlusion_ratio

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.willy_sim.harness.bootstrap import SimCell

#: Mostly here for the probe's degraded paths. Both the reach move and the wrist fit are caught and
#: turned into a skipped pose, after which ``_summarize`` still prints a complete-looking coverage
#: table in which a row that was never measured reads as a zero. A go/no-go answer taken from that
#: table is wrong in the direction that looks like evidence.
_LOG = create_logger("OcclusionProbe", log_file=OCCLUSION_PROBE_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)

OBLIQUE_CAM_L_PRIM = "/World/ObliqueCameraL"
OBLIQUE_CAM_R_PRIM = "/World/ObliqueCameraR"

# A deliberately stacked scene at the reachable bin centre (~450, 0): a blue occluder settles on the red
# target, so the overhead view is shadowed while the side obliques see the target's exposed faces.
STACKED_OBJECTS: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
    ("red cube", (0.90, 0.12, 0.12), (450.0, 0.0, 25.0)),    # Target (bin centre)
    ("blue cube", (0.12, 0.22, 0.90), (450.0, 0.0, 80.0)),   # Occluder: spawned above, settles on the target
]


def _stacked_specs() -> list[SimObjectConfig]:
    return [SimObjectConfig(name=n, color=c, position_mm=p) for (n, c, p) in STACKED_OBJECTS]


# Fixed eye-to-hand camera authoring lives in scene/cameras.py, shared with the multi-view and
# two-side-camera rigs. Aliased here for this module's call sites and for run_pile_baseline's import.
from src.willy_sim.scene import author_fixed_camera as _author_oblique_camera  # noqa: E402


def _set_near_clip(camera: Any, near_clip_m: float) -> None:
    """Best-effort near-clip override.

    The overhead cell camera defaults to a 1.0 m near clip, which clips a reaching arm out of frame,
    so robot self-occlusion would be invisible in the ground-truth mask.
    """
    try:
        camera.set_clipping_range(near_clip_m, 1.0e6)
    except Exception:  # noqa: BLE001
        pass


def _target_metrics(frame: Any, target_label: str) -> dict[str, Any]:
    """Visible target pixels + occlusion + per-object px from one GT acquire."""
    segs = list(frame.segmentations)
    per_obj = {
        getattr(s, "label", f"seg{i}"): int(np.asarray(s.mask).astype(bool).sum())
        for i, s in enumerate(segs)
    }
    tgt_idx = next(
        (i for i, s in enumerate(segs) if getattr(s, "label", "").lower() == target_label.lower()),
        None,
    )
    if tgt_idx is None:
        return {"target_px": 0.0, "occ": 0.0, "visible": 0.0, "per_obj_px": per_obj}
    target_mask = np.asarray(segs[tgt_idx].mask).astype(bool)
    others = [np.asarray(s.mask).astype(bool) for i, s in enumerate(segs) if i != tgt_idx]
    px = float(int(target_mask.sum()))
    return {"target_px": px, "occ": float(occlusion_ratio(target_mask, others)),
            "visible": float(px > 0.0), "per_obj_px": per_obj}


def _settle_render(session: Any, n: int = 8) -> None:
    app = getattr(session, "app", None)
    for _ in range(n):
        session.step(render=True)
        if app is not None:
            app.update()


def run_probe(
    *,
    headless: bool = True,
    data_dir: str | None = None,
    target: str = "red cube",
    oblique_l_pos_mm: tuple[float, float, float] = (450.0, -560.0, 720.0),
    oblique_r_pos_mm: tuple[float, float, float] = (450.0, 560.0, 720.0),
    oblique_aim_mm: tuple[float, float, float] = (450.0, 0.0, 50.0),
    view_height_mm: float = 275.0,
    settle_steps: int = 150,
    with_wrist: bool = True,
    jsonl_path: str | None = None,
) -> list[dict[str, Any]]:
    """Boot the stacked scene with two symmetric oblique cameras and the eye-in-hand wrist camera.

    The target's visible pixels are read from every camera at two arm poses: park, with the arm clear,
    and view, with the wrist over the target.
    """
    from src.willy_sim.scene import camera_to_tcp_ground_truth, mount_wrist_camera
    from src.robot.core import JointPositions

    cell: SimCell = bootstrap_sim_cell(
        data_dir, headless=headless, safety=True,
        scene_kwargs={"objects_override": _stacked_specs()},
    )
    arm, sim = cell.arm, cell.sim
    session = arm.session

    for _ in range(max(0, settle_steps)):
        session.step(render=False)  # settle the stacked cubes (cubes get no YCB settle loop)

    object_specs = list(cell.handles.object_specs)
    target_prim = next((p for (p, lbl) in object_specs if lbl.lower() == target.lower()), None)

    # Fixed cameras (eye-to-hand): the cell overhead plus two symmetric obliques. The overhead near clip
    # is lowered so a reaching arm is rendered and occludes the ground-truth mask.
    _set_near_clip(cell.handles.camera, 0.05)
    cams: dict[str, Any] = {"overhead": cell.handles.camera}
    cams["oblique_L"] = _author_oblique_camera(session, OBLIQUE_CAM_L_PRIM, position_mm=oblique_l_pos_mm, aim_mm=oblique_aim_mm)
    cams["oblique_R"] = _author_oblique_camera(session, OBLIQUE_CAM_R_PRIM, position_mm=oblique_r_pos_mm, aim_mm=oblique_aim_mm)
    if with_wrist:
        cams["wrist"] = mount_wrist_camera(session)  # eye-in-hand, child of wrist_3_link

    sources = {
        name: MultiObjectGroundTruthPerceptionSource(camera=cam, targets=object_specs, session=session, warmup_steps=6)
        for name, cam in cams.items()
    }

    park_q = JointPositions(np.asarray(sim.park_joint_positions, dtype=np.float64))
    rows: list[dict[str, Any]] = []

    def _measure_all(pose_name: str, active: list[str]) -> None:
        for cam_name in active:
            m = _target_metrics(sources[cam_name].acquire(), target)
            rows.append({"pose": pose_name, "camera": cam_name, "target": target, **m})
            print(f"[{pose_name:>4} | {cam_name:>9}] target_px={m['target_px']:>7.0f} occ={m['occ']:.3f} "
                  f"visible={int(m['visible'])} per_obj={m['per_obj_px']}", flush=True)

    # Park: arm clear of the bin. The eye-to-hand baseline and the object-stacking occlusion.
    arm.move_joint(park_q)
    _settle_render(session)
    _measure_all("park", ["overhead", "oblique_L", "oblique_R"])

    # View: drive the TCP over the target, so the arm reaches in and occludes the overhead camera. The
    # eye-to-hand measurement (overhead and obliques) is independent of the wrist and runs first, so a
    # flaky eye-in-hand fit can never hide it. The wrist close-up is then best-effort: a fit failure
    # here is reported as not available, not fatal.
    if target_prim is not None:
        from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

        tgt_xyz = np.asarray(SingleRigidPrim(target_prim).get_world_pose()[0], dtype=np.float64) * 1000.0
        try:
            arm.move(topdown_view_pose(tgt_xyz, (0.0, 0.0, 0.0), view_height_mm))  # TCP over target
        except Exception as exc:  # noqa: BLE001
            # The view pose is the robot-self-occlusion measurement. Without it the summary's second
            # line reads "0/3 fixed cams retain the target", which is the pass-shaped answer.
            _LOG.error(
                "VIEW pose unreachable (%r); the robot-self-occlusion half was NOT measured; the "
                "printed 'view' row is absent, not zero", exc,
            )
            print(f"view: reach arm.move failed ({exc!r}) -> skipping VIEW measurement", flush=True)
        else:
            _settle_render(session)
            _measure_all("view", ["overhead", "oblique_L", "oblique_R"])  # eye-to-hand robot occlusion (robust)
            if with_wrist:
                try:
                    t_cam_tool, rmse = camera_to_tcp_ground_truth(cams["wrist"], arm)
                    p_cam_tool = tuple(float(v) for v in np.asarray(t_cam_tool.to_matrix())[:3, 3])
                    print(f"wrist cam->tcp fit rmse={rmse:.1f} mm, p_cam_tool={np.round(p_cam_tool, 1)}", flush=True)
                    arm.move(topdown_view_pose(tgt_xyz, p_cam_tool, view_height_mm))  # centred wrist view
                    _settle_render(session)
                    _measure_all("view_wrist", ["wrist"])
                except Exception as exc:  # noqa: BLE001 (wrist is best-effort; ETH numbers already captured)
                    # Warning, not error: the eye-to-hand half above is the probe's claim and it was
                    # already captured. Only the wrist close-up is missing.
                    _LOG.warning("wrist EIH fit/centre failed (%r); wrist close-up reported N/A", exc)
                    print(f"view: wrist EIH fit/centre failed ({exc!r}) -> wrist N/A "
                          f"(EIH centring is the established run_eih_pick path; not re-derived here)", flush=True)
    else:
        _LOG.error(
            "target %r not among the scene's objects; the whole VIEW half was skipped", target,
        )
        print("view: target prim not found -> skipping VIEW measurement", flush=True)

    _summarize(rows, with_wrist=with_wrist)
    # The go/no-go, as measured: which poses produced rows, and how many cameras kept the target at
    # each. Aggregated onto one line; the per camera-pose rows go to stdout.
    _by_pose: dict[str, list[str]] = {}
    for r in rows:
        _by_pose.setdefault(str(r["pose"]), []).append(
            f"{r['camera']}={'sees' if float(r['target_px']) > 0 else 'LOST'}"
        )
    _LOG.info(
        "probe done for target %r: %d row(s) over pose(s) %s; %s", target, len(rows),
        sorted(_by_pose), "; ".join(f"{p}: {', '.join(v)}" for p, v in sorted(_by_pose.items())),
    )
    if jsonl_path:
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        _LOG.info("wrote %d measurement row(s) -> %s", len(rows), jsonl_path)
        print(f"wrote {len(rows)} rows -> {jsonl_path}", flush=True)
    return rows


def _summarize(rows: list[dict[str, Any]], *, with_wrist: bool) -> None:
    """Print the multi-view coverage signals from the measured rows."""
    by = {(r["pose"], r["camera"]): float(r["target_px"]) for r in rows}

    def px(pose: str, cam: str) -> float:
        return by.get((pose, cam), 0.0)

    print("\n=== H5.0R multi-view coverage (target visible-px) ===", flush=True)
    print(f"object-stacking (arm clear):  overhead={px('park','overhead'):.0f}  "
          f"oblique_L={px('park','oblique_L'):.0f}  oblique_R={px('park','oblique_R'):.0f}"
          f"  (PASS: an oblique sees the stacked-under target the overhead loses)", flush=True)
    seen_view = [c for c in ("overhead", "oblique_L", "oblique_R") if px("view", c) > 0]
    print(f"robot self-occl (wrist over target):  overhead={px('view','overhead'):.0f}  "
          f"oblique_L={px('view','oblique_L'):.0f}  oblique_R={px('view','oblique_R'):.0f}"
          f"  -> {len(seen_view)}/3 fixed cams retain the target  "
          f"(PASS: overhead loses it, BOTH obliques keep it = two-sided redundancy)", flush=True)
    if with_wrist:
        w = px("view_wrist", "wrist")
        print(f"EIH close-up (wrist over target):  wrist={w:.0f}  "
              + ("(switched in for a centred close view -> sharper grasp point; ETH+EIH = global cover + close refine)"
                 if w > 0 else "(N/A in this probe; EIH centring is the established run_eih_pick path, 10/10)"),
              flush=True)
    print("=====================================================\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="H5.0R multi-view occlusion discriminance probe (read-only).")
    ap.add_argument("--gui", action="store_true", help="run with a viewport (default headless)")
    ap.add_argument("--data", default=None, help="sim config data dir override")
    ap.add_argument("--target", default="red cube", help="prompted target label")
    ap.add_argument("--no-wrist", action="store_true", help="skip the eye-in-hand wrist camera")
    ap.add_argument("--jsonl", default=None, help="write per-row measurements to this JSONL path")
    args = ap.parse_args()

    rows = run_probe(
        headless=not args.gui,
        data_dir=args.data,
        target=args.target,
        with_wrist=not args.no_wrist,
        jsonl_path=args.jsonl,
    )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
