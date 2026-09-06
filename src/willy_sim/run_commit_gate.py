"""On-box proof that the commit gate decides on real multi-view evidence (Isaac only).

Drives K genuinely different wrist viewpoints of one static object into one SceneFusion and shows that
the corridor evidence grows with diverse views (hit_voxels strictly increases) but stays flat when the
same view is re-ingested K times, the duplicate control. The gate therefore allows on real multi-view
evidence and refuses (NO_COMMIT_INSUFFICIENT_FUSION) on duplicate-view inflation, which is what
separates real evidence from an inflated views_accepted counter.

The proof uses rendered wrist depth, because ground-truth depth flattens the object mask to a single
plane and so gives degenerate diversity exactly where the corridor measures. SceneFusion runs
standalone, which isolates the claim from candidate generation, IK and grasp. The corridor radius and
length match the committed RobotGraspingCommitPolicyConfig defaults (25 mm and 120 mm); tau does not,
because this sparse single-cube scene needs 0.05 where the committed default is 0.30.

    <isaac-sim>\\python.bat -m src.willy_sim.run_commit_gate [--k 4 --tau 0.05 --tilt-deg 18 --ring-mm 45] [--gui]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.willy_sim.run_eih_pick import build_service, topdown_view_pose
from src.geometry import Frame, Pose
from src.geometry.conversions import axis_angle_to_quaternion_xyzw
from src.geometry.quaternion import multiply
from src.robot.grasping.multiview.fusion import FusionConfig, SceneFusion
from src.robot.grasping.loop.pick_loop import CommitPolicy

_RADIUS_MM = 25.0
_LENGTH_MM = 120.0
_APPROACH = np.array([0.0, 0.0, -1.0])


def _tilted_poses(obj_pos, p_cam_tool, view_height_mm, k, tilt_deg, ring_mm):
    """Build the K view poses: view 0 straight down, views 1..k-1 on a tilted ring around the object.

    Each ring view is tilted about base-X so the wrist cam sees the object obliquely. That gives
    genuine parallax and therefore different BASE voxels, rather than a parallel line-of-sight that
    fills the same corridor voxels.
    """
    base = topdown_view_pose(obj_pos, p_cam_tool, view_height_mm)
    q_down = np.asarray(base.quaternion_xyzw, dtype=np.float64)
    q_tilt = axis_angle_to_quaternion_xyzw(np.array([np.deg2rad(tilt_deg), 0.0, 0.0]))
    poses = [base]
    for i in range(1, k):
        ang = 2.0 * np.pi * (i - 1) / max(1, k - 1)
        centre = np.asarray(obj_pos, dtype=np.float64) + np.array(
            [ring_mm * np.cos(ang), ring_mm * np.sin(ang), 0.0]
        )
        nominal = topdown_view_pose(centre, p_cam_tool, view_height_mm)
        poses.append(
            Pose(
                position_mm=np.asarray(nominal.position_mm, dtype=np.float64),
                quaternion_xyzw=multiply(q_tilt, q_down),
                frame=Frame.BASE,
                label=f"cg-view-{i:02d}",
            )
        )
    return poses


def _adaptive_fusion(obj_pos, voxel_mm=6.0):
    """Size the fusion ROI from the measured object position so the corridor falls inside the grid.

    The SceneFusion grid is origin-centered (grid_origin_mm = -0.5*extent), while the eye-in-hand
    object sits at about x=450, outside the default 600 mm ROI, so a default grid would query 0
    corridor voxels.
    """
    pad = 170.0  # corridor (120) + ring (45) + margin
    raw = (
        max(2.0 * (abs(float(obj_pos[0])) + pad), 400.0),
        max(2.0 * (abs(float(obj_pos[1])) + pad), 400.0),
        max(2.0 * (abs(float(obj_pos[2])) + pad), 360.0),
    )
    # FusionConfig requires each extent to be an integer multiple of voxel_size_mm, so snap up.
    ext = tuple(float(int(np.ceil(e / voxel_mm)) * voxel_mm) for e in raw)
    return FusionConfig(enabled=True, voxel_size_mm=voxel_mm, roi_extent_mm=ext), ext


def _corridor(fusion, anchor):
    return fusion.corridor_evidence(
        position_mm=anchor, approach=_APPROACH, length_mm=_LENGTH_MM, radius_mm=_RADIUS_MM
    )


def run_proof(*, k=4, tau=0.05, tilt_deg=18.0, ring_mm=45.0, view_height_mm=275.0,
              headless=True, data_dir=None) -> bool:
    service, arm, gripper, handles, cfg, _view_pose, _cell = build_service(
        headless=headless, data_dir=data_dir, view_height_mm=view_height_mm
    )
    orch = service.runtime.orchestrator
    resolver = orch.frame_resolver
    perception = orch.perception
    perception._ground_truth_depth = False  # rendered wrist depth: GT depth flattens the mask

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    obj = SingleRigidPrim(handles.object_prim_path)
    obj_pos = np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0
    p_cam_tool = np.asarray(resolver.t_cam_to_tool.to_matrix())[:3, 3]
    fcfg, ext = _adaptive_fusion(obj_pos)
    print(f"obj_pos_mm={np.round(obj_pos, 1)} roi_extent_mm={tuple(round(e) for e in ext)} "
          f"(grid x in [{-ext[0] / 2:.0f},{ext[0] / 2:.0f}])", flush=True)
    anchor = obj_pos.copy()
    poses = _tilted_poses(obj_pos, p_cam_tool, view_height_mm, k, tilt_deg, ring_mm)

    # ---- diverse: k different poses into one fusion; save view 0 for the duplicate control ----
    div = SceneFusion(fcfg)
    hv_div = []
    dup_frame = None
    dup_t = None
    for i, p in enumerate(poses):
        res = arm.move(p)
        if getattr(res.status, "value", str(res.status)) != "executed":
            print(f"WARN diverse view {i} move={res.status}; skipping", flush=True)
            continue
        arm.session.step_n(30)
        frame = perception.acquire()
        t_cb = resolver.camera_to_base_for_frame(frame, arm=arm)
        if dup_frame is None:  # reuse view-0 for the duplicate control: no extra arm move, shorter and safer
            dup_frame, dup_t = frame, t_cb
        div.ingest(depth_map=frame.depth_map, intrinsics=frame.intrinsics, t_cam_to_base=t_cb, timestamp_ns=i * 1_000_000_000)
        ev = _corridor(div, anchor)
        dt = float(np.linalg.norm(np.asarray(p.position_mm) - np.asarray(poses[0].position_mm)))
        hv_div.append(int(ev.hit_voxels))
        print(f"DIVERSE v{i}: ||dt||={dt:.1f}mm queried={ev.queried_voxels} hit_voxels={ev.hit_voxels} "
              f"hit_frac={ev.hit_fraction:.3f} views={ev.views_accepted}", flush=True)
    div_final = _corridor(div, anchor)

    if dup_frame is None or dup_t is None:
        print("PROOF ABORTED: no diverse view succeeded (every arm move failed)", flush=True)
        return False

    # ---- duplicate control: re-ingest the saved view-0 (depth, t_cam_to_base) K times (no move/acquire) ----
    dup = SceneFusion(fcfg)
    hv_dup = []
    for i in range(k):
        dup.ingest(depth_map=dup_frame.depth_map, intrinsics=dup_frame.intrinsics, t_cam_to_base=dup_t, timestamp_ns=i * 1_000_000_000)
        ev = _corridor(dup, anchor)
        hv_dup.append(int(ev.hit_voxels))
        print(f"DUP v{i}: hit_voxels={ev.hit_voxels} hit_frac={ev.hit_fraction:.3f} views={ev.views_accepted}", flush=True)
    dup_final = _corridor(dup, anchor)

    # ---- verdict via the real CommitPolicy two-predicate gate ----
    pol = CommitPolicy(
        enabled=True, min_views_accepted=2, min_corridor_hit_fraction=tau,
        corridor_radius_mm=_RADIUS_MM, corridor_length_mm=_LENGTH_MM,
        max_reobserve_attempts=k, apply_modes=frozenset({"auto"}),
    )

    def _allow(ev) -> bool:
        return ev.views_accepted >= pol.min_views_accepted and ev.hit_fraction >= pol.min_corridor_hit_fraction

    grew = len(hv_div) >= 2 and all(b >= a for a, b in zip(hv_div, hv_div[1:])) and hv_div[-1] > hv_div[0]
    flat = len(hv_dup) >= 2 and len(set(hv_dup)) == 1
    allow_div = _allow(div_final)
    refuse_dup = not _allow(dup_final)
    ok = bool(grew and flat and allow_div and refuse_dup)
    print(
        f"COMMIT-GATE DIVERSE-VIEW PROOF: diverse_grew={grew} (hv={hv_div}) duplicate_flat={flat} (hv={hv_dup}) "
        f"diverse_allows={allow_div}(hf={div_final.hit_fraction:.3f}) "
        f"duplicate_refuses={refuse_dup}(hf={dup_final.hit_fraction:.3f}) -> PASS={ok}",
        flush=True,
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy U6 commit-gate diverse-vs-duplicate proof (Isaac).")
    ap.add_argument("--k", type=int, default=4)
    # tau is the corridor-coverage floor. A lone cube gives sparse coverage, so the proof runs
    # tau=0.05, low enough that diverse views clear it and duplicates cannot. The production default
    # 0.30 needs a denser scene; the diverse-versus-duplicate growth contrast is scene-independent.
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--tilt-deg", type=float, default=18.0)
    ap.add_argument("--ring-mm", type=float, default=45.0)
    ap.add_argument("--view-height-mm", type=float, default=275.0)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-headless", action="store_true")
    ap.add_argument("--data-dir", type=str, default=None)
    a = ap.parse_args()
    ok = run_proof(
        k=a.k, tau=a.tau, tilt_deg=a.tilt_deg, ring_mm=a.ring_mm, view_height_mm=a.view_height_mm,
        headless=not (a.gui or a.no_headless), data_dir=a.data_dir,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
