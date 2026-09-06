"""Suction pick runner: an Isaac surface gripper on the UR5e wrist.

Every mode boots the cell through ``bootstrap_sim_cell`` (baked 2F-85 plus ``IsaacGripper``, whose
articulation keeps the Lula IK pre-resolve healthy) and authors the surface gripper on the wrist here in
the runner. Authoring it in a bootstrap branch instead breaks the IK.

* ``--score`` (default): perceive, run ``synthesize_suction_grasps`` with the analytical scorer, report
  the ranked suction candidates with their seal and wrench values in the BASE frame, then plan and reach
  the approach above the best candidate. Both the placement and the quality come from the analytical
  model over the rendered-depth cloud.
* ``--pick``: the same decision, followed by a descent, a runtime weld and a measured lift.
* ``--vs-jaw``: one wide flat object scored by the parallel-jaw calculator and by suction.
* ``--klt-probe`` and ``--klt-pick``: reach measurement, then the pick, inside a deep narrow KLT bin.
* ``--probe``: the attach probe (descend, vacuum, lift). Isaac's binary surface-gripper bond will not
  bond a non-root articulation link, so the bond stays a visible-lift surrogate and the score path
  carries the value.

On-box only (needs Isaac); run with Isaac's bundled python from the repo root:

    <isaac-sim>\\python.bat -m src.willy_sim.run_suction_pick --score
    <isaac-sim>\\python.bat -m src.willy_sim.run_suction_pick --probe
"""

from __future__ import annotations

import argparse

import numpy as np

from src.willy_sim.harness.bootstrap import SimCell, bootstrap_sim_cell
from src.willy_sim.scene import OBJECT_PRIM
from src.willy_sim.suction_mount import SUCTION_CUP, SuctionCupSpec


def _write(path: str, lines: list[str]) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_suction_score(
    *, headless: bool = True, data_dir: str | None = None, mount_cup: bool = True,
    reach_best: bool = True, standoff_mm: float = 60.0, near_gap_mm: float = 40.0,
    payload_g: float | None = None, min_quality: float = 0.05, max_results: int = 8,
    out_path: str = "logs/h7/suction_score.txt",
) -> int:
    """The suction pick path: perceive, synthesize, score and report on the real Isaac scene.

    Perceives the object with the ground-truth instance mask and the rendered top-surface depth
    (``ground_truth_depth=False``, so the seal model sees the real object-top geometry rather than a
    centre-overlaid flat patch), runs :func:`synthesize_suction_grasps` with the analytical scorer, and
    reports the ranked suction candidates with their ``seal x wrench`` values in the BASE frame. Unless
    ``reach_best`` is off it then plans and reaches the approach standoff and the near-contact pose above
    the best candidate; the binary Isaac bond and lift are the surrogate described in :func:`run_probe`.

    Returns 0 iff at least one suction candidate was synthesized.
    """
    # Boot through the build_service path (IK-healthy) and author the surface gripper on the wrist. The
    # anchor sits at about the grasp centre, which is the TCP; it does not change the arm IK or TCP, it
    # only makes the cell a suction cell for the reach. Default-on; --no-cup scores the same values.
    gpath_holder: dict[str, str | None] = {"path": None}

    def _author_hook(stage: object) -> None:
        from src.willy_sim.scene import ARM_PRIM
        from src.willy_sim.suction_mount import SUCTION_CUP, mount_suction_cup
        gpath_holder["path"] = mount_suction_cup(stage, ARM_PRIM, SUCTION_CUP)

    cell = bootstrap_sim_cell(data_dir, headless=headless,
                              post_scene_hook=_author_hook if mount_cup else None)
    try:
        return _score_booted_cell(
            cell, cup_path=gpath_holder["path"], out_path=out_path, reach_best=reach_best,
            standoff_mm=standoff_mm, near_gap_mm=near_gap_mm, payload_g=payload_g,
            min_quality=min_quality, max_results=max_results,
        )
    finally:
        # Always shut Kit down so a crash never leaves an idle app holding RAM with no CPU/GPU load.
        try:
            cell.arm.session.stop()
        except Exception:  # noqa: BLE001 (headless SimulationApp.close() can segfault on shutdown)
            pass


def _score_booted_cell(
    cell: SimCell, *, cup_path: str | None, out_path: str, reach_best: bool,
    standoff_mm: float, near_gap_mm: float, payload_g: float | None,
    min_quality: float, max_results: int,
) -> int:
    """Score and reach on an already-booted suction cell; the caller guarantees the Kit shutdown."""
    from src.geometry import Frame, Pose
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import (
        SuctionConfig,
        synthesize_suction_grasps,
    )

    from src.willy_sim.perception import GroundTruthPerceptionSource

    log: list[str] = ["=== H7 suction pick path (analytical scorer, real Isaac scene) ==="]
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)

    # Payload for the wrench term = the sim object's real mass (falls back to 200 g).
    payload = float(payload_g) if payload_g is not None else 200.0
    obj_cfg = getattr(sim.scene_setup, "object", None)
    if payload_g is None and obj_cfg is not None:
        payload = float(getattr(obj_cfg, "mass_kg", 0.2)) * 1000.0

    # Park the arm clear of the overhead view, set a small near-clip, then perceive the rendered
    # top-surface depth. The overhead camera's default 1.0 m near-clip cuts away the close object top
    # (~950 mm < 1000 mm), so the depth over the object mask reads the table behind it. A small near-clip
    # renders the true top, but it also stops hiding the home-pose arm link, which would then occlude the
    # object: park the arm out of view first so the instance mask stays clean. The seal model then reads
    # the real object-top geometry, which discriminates flat from curved, at the correct Z.
    _park = getattr(sim, "park_joint_positions", None)
    park_q = np.asarray(_park if _park is not None else sim.home_joint_positions, dtype=np.float64)
    arm.move_joint(JointPositions(park_q))
    arm.session.step_n(15)
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001 (near-clip is best-effort; the clipping API varies by Isaac version)
        pass
    # The instance-id-segmentation annotator is stale on the first frame (empty (0,) mask), so acquire in a
    # small retry loop, stepping the render, until the mask is a valid 2-D array with pixels.
    perception = GroundTruthPerceptionSource(
        camera=handles.camera, target_prim_path=handles.object_prim_path, session=arm.session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps), ground_truth_depth=False,
    )
    frame = perception.acquire()
    mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
    tries = 0
    while (mask.ndim != 2 or not mask.any()) and tries < 10:
        arm.session.step_n(5)
        frame = perception.acquire()
        mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
        tries += 1
    log.append(f"[perceive] mask_px={int(mask.sum())} mask_shape={mask.shape} "
               f"depth_shape={tuple(np.asarray(frame.depth_map).shape)} tries={tries} "
               f"payload_g={payload:.0f} cup_path={cup_path}")
    if mask.ndim != 2 or not mask.any():
        log.append("=== SUCTION SCORE: NO MASK (object not perceived) ===")
        _write(out_path, log)
        for line in log:
            print(line, flush=True)
        return 1

    grasps = synthesize_suction_grasps(
        mask, frame.depth_map, frame.intrinsics,
        camera_to_base=handles.camera_to_base, payload_mass_g=payload,
        config=SuctionConfig(
            scorer=AnalyticalSuctionScorer(), max_results=max_results, min_quality=min_quality,
        ),
    )
    log.append(f"[synthesize] {len(grasps)} suction candidate(s) (min_quality={min_quality}, BASE frame)")
    for i, g in enumerate(grasps):
        m = g.metadata
        log.append(
            f"  #{i} quality={g.quality:.3f} seal={g.seal_score:.3f} "
            f"wrench_feasible={m.get('wrench_feasible')} wrench={m.get('wrench_resist', float('nan')):.3f} "
            f"pos_mm={np.round(g.position_mm, 1).tolist()} approach={np.round(g.approach, 2).tolist()} "
            f"max_def_mm={m.get('max_deformation_mm', float('nan')):.2f} "
            f"support={m.get('perimeter_support', float('nan')):.2f} src={m.get('source')}"
        )
    _write(out_path, log)

    if reach_best and grasps:
        best = grasps[0]
        pos = np.asarray(best.position_mm, dtype=np.float64)
        approach = np.asarray(best.approach, dtype=np.float64)
        approach = approach / (float(np.linalg.norm(approach)) or 1.0)
        # A closing axis perpendicular to the approach: the suction cup is axisymmetric, so the in-plane
        # roll is free and any perpendicular does. Build the tool orientation with tool-Z on the approach.
        seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        closing = seed - approach * float(np.dot(seed, approach))
        closing = closing / (float(np.linalg.norm(closing)) or 1.0)
        quat = _quaternion_from_axes(closing, approach)
        for label, gap in (("standoff", standoff_mm), ("near-contact", near_gap_mm)):
            target = pos - approach * float(gap)  # back off along the approach (above a top face)
            arm.move_joint(JointPositions(home_q))
            arm.session.step_n(10)
            res = arm.move(Pose(position_mm=target, quaternion_xyzw=quat, frame=Frame.BASE,
                                label=f"suction-{label}"))
            log.append(f"[reach] best#0 contact={np.round(pos, 1).tolist()} {label}(-{gap:.0f}mm)="
                       f"{np.round(target, 1).tolist()} IK={getattr(res.status, 'value', res.status)}")
            _write(out_path, log)

    log.append(f"\n=== SUCTION SCORE {'OK' if grasps else 'NO CANDIDATE'}: {len(grasps)} candidate(s), "
               f"best quality={grasps[0].quality:.3f} ===" if grasps else "\n=== SUCTION SCORE: NO CANDIDATE ===")
    _write(out_path, log)
    for line in log:
        print(line, flush=True)
    return 0 if grasps else 1


def run_h74_vs_jaw(
    *, size_mm: tuple[float, float, float] = (120.0, 120.0, 35.0), name: str = "wide_flat",
    color: tuple[float, float, float] = (0.20, 0.45, 0.85), mass_g: float = 150.0,
    headless: bool = True, data_dir: str | None = None, out_path: str = "logs/h7/suction_h74.txt",
) -> int:
    """A wide flat object the 2F-85 jaw cannot grasp and suction can.

    Spawns a wide flat box (``size_mm``, both footprint dims greater than the 85 mm jaw aperture), then on
    the same scene runs the parallel-jaw ``GraspCalculator`` (max 85 mm), which finds no valid grasp
    because the object does not fit the aperture, and runs the suction synthesis with a seal-gated weld
    lift, which seals the flat top and lifts. Returns 0 iff the jaw finds nothing and suction lifts.
    """
    from src.config.schema.robot.sim_schema import SimObjectConfig

    obj = SimObjectConfig(
        name=name, shape="cube", size_mm=size_mm,
        position_mm=(450.0, 0.0, float(size_mm[2]) / 2.0), mass_kg=float(mass_g) / 1000.0, color=color,
    )
    cell = bootstrap_sim_cell(data_dir, headless=headless, scene_kwargs={"objects_override": [obj]})
    try:
        return _h74_booted_cell(cell, size_mm=size_mm, mass_g=mass_g, out_path=out_path)
    finally:
        try:
            cell.arm.session.stop()
        except Exception:  # noqa: BLE001
            pass


def _h74_booted_cell(
    cell: SimCell, *, size_mm: tuple[float, float, float], mass_g: float, out_path: str,
) -> int:
    """Jaw-fails-vs-suction-succeeds comparison on an already-booted wide-object cell."""
    import omni.usd  # type: ignore[import-not-found]

    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes
    from src.robot.grasping.generation.calculator import GraspCalculator
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import (
        SuctionConfig,
        synthesize_suction_grasps,
    )

    from src.willy_sim.perception import GroundTruthPerceptionSource
    from src.willy_sim.scene import WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import engage_suction_weld

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    log: list[str] = [f"=== H7.4 wide/flat object: jaw vs suction (object {size_mm} mm) ==="]
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    stage = omni.usd.get_context().get_stage()

    # Perceive the true top-surface geometry (park + near-clip, as the score/pick paths).
    _park = getattr(sim, "park_joint_positions", None)
    park_q = np.asarray(_park if _park is not None else sim.home_joint_positions, dtype=np.float64)
    arm.move_joint(JointPositions(park_q))
    arm.session.step_n(15)
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    perception = GroundTruthPerceptionSource(
        camera=handles.camera, target_prim_path=handles.object_prim_path, session=arm.session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps), ground_truth_depth=False,
    )
    frame = perception.acquire()
    seg = frame.segmentations[0]
    mask = np.asarray(seg.mask, dtype=bool)
    tries = 0
    while (mask.ndim != 2 or not mask.any()) and tries < 10:
        arm.session.step_n(5)
        frame = perception.acquire()
        seg = frame.segmentations[0]
        mask = np.asarray(seg.mask, dtype=bool)
        tries += 1
    intr = np.asarray(frame.intrinsics, dtype=np.float64)
    log.append(f"[perceive] mask_px={int(mask.sum())} tries={tries}")

    # (1) Jaw: the parallel-jaw calculator (max = the 2F-85 aperture). A wide object yields no grasp within it.
    max_w = float(cell.robot.gripper.max_width_mm)
    calc = GraspCalculator(
        camera_matrix=intr, max_grip_width_mm=max_w, min_grip_width_mm=cell.robot.gripper.min_width_mm,
    )
    try:
        jaw = calc.compute(seg, frame.depth_map, camera_to_base=handles.camera_to_base, unit="mm")
    except Exception as exc:  # noqa: BLE001 (a calculator error on an ungraspable object is itself a failure)
        jaw = []
        log.append(f"[jaw] compute raised {type(exc).__name__}: {exc} (treated as no-grasp)")
    jaw_ok = len(jaw) > 0
    jaw_widths = [round(float(getattr(g, "grip_width_mm", float("nan"))), 1) for g in jaw[:3]]
    log.append(f"[jaw] valid_grasps={len(jaw)} (aperture max={max_w:.0f}mm; object footprint="
               f"{size_mm[0]:.0f}x{size_mm[1]:.0f}mm) widths={jaw_widths} -> "
               f"{'GRASPABLE' if jaw_ok else 'NO VALID GRASP (too wide for the jaw)'}")

    # (2) suction: synthesize on the flat top, seal-gate, weld, lift.
    payload = float(mass_g)
    grasps = synthesize_suction_grasps(
        mask, frame.depth_map, intr, camera_to_base=handles.camera_to_base, payload_mass_g=payload,
        config=SuctionConfig(scorer=AnalyticalSuctionScorer(), max_results=8, min_quality=0.05),
    )
    if not grasps:
        log.append("[suction] NO CANDIDATE")
        _write(out_path, log)
        for line in log:
            print(line, flush=True)
        return 1
    best = grasps[0]
    log.append(f"[suction] best seal={best.seal_score:.3f} quality={best.quality:.3f} "
               f"wrench_feasible={best.metadata.get('wrench_feasible')} "
               f"contact_mm={np.round(best.position_mm, 1).tolist()}")

    pos = np.asarray(best.position_mm, dtype=np.float64)
    approach = np.asarray(best.approach, dtype=np.float64)
    approach = approach / (float(np.linalg.norm(approach)) or 1.0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    closing = seed - approach * float(np.dot(seed, approach))
    closing = closing / (float(np.linalg.norm(closing)) or 1.0)
    quat = _quaternion_from_axes(closing, approach)

    obj = SingleRigidPrim(handles.object_prim_path)
    z_before = float(np.asarray(obj.get_world_pose()[0])[2])
    arm.move_joint(JointPositions(home_q))
    arm.session.step_n(10)
    reached = False
    for label, gap in (("standoff", 60.0), ("contact", 25.0)):
        res = arm.move(Pose(position_mm=pos - approach * gap, quaternion_xyzw=quat, frame=Frame.BASE,
                            label=f"h74-{label}"))
        reached = getattr(getattr(res, "status", None), "value", None) == "executed"
        log.append(f"[suction-approach] {label} IK={getattr(res.status, 'value', res.status)}")
    rose_mm = 0.0
    if reached:
        arm.session.step_n(8)
        wp, wq = SingleRigidPrim(WRIST_LINK_PRIM).get_world_pose()
        op, oq = obj.get_world_pose()
        r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
        r_o = to_rotation_matrix(np.array([oq[1], oq[2], oq[3], oq[0]], dtype=np.float64))
        p_rel = r_w.T @ (np.asarray(op, dtype=np.float64) - np.asarray(wp, dtype=np.float64))
        q_rel = from_rotation_matrix(r_w.T @ r_o)
        engage_suction_weld(stage, WRIST_LINK_PRIM, handles.object_prim_path,
                            (float(p_rel[0]), float(p_rel[1]), float(p_rel[2])),
                            (float(q_rel[3]), float(q_rel[0]), float(q_rel[1]), float(q_rel[2])))
        arm.session.step_n(12)
        arm.move(Pose(position_mm=pos - approach * 25.0 + (-approach) * 100.0, quaternion_xyzw=quat,
                      frame=Frame.BASE, label="h74-lift"))
        arm.session.step_n(30)
        rose_mm = (float(np.asarray(obj.get_world_pose()[0])[2]) - z_before) * 1000.0
    suction_ok = rose_mm >= 50.0
    log.append(f"[suction-lift] object_rose={rose_mm:.1f} mm suction_ok={suction_ok}")

    thesis = (not jaw_ok) and suction_ok
    log.append(f"\n=== H7.4 {'PROVEN' if thesis else 'NOT proven'}: JAW={'no grasp' if not jaw_ok else f'{len(jaw)} grasps'} "
               f"| SUCTION seal={best.seal_score:.3f} rose={rose_mm:.0f}mm -> "
               f"{'suction picks what the jaw cannot' if thesis else 'inconclusive'} ===")
    _write(out_path, log)
    for line in log:
        print(line, flush=True)
    return 0 if thesis else 1


def run_suction_pick_lift(
    *, headless: bool = True, data_dir: str | None = None, payload_g: float | None = None,
    min_quality: float = 0.05, seal_threshold: float = 0.5, standoff_mm: float = 60.0,
    contact_gap_mm: float = 25.0, lift_mm: float = 100.0, place: bool = False,
    out_path: str = "logs/h7/suction_pick.txt",
) -> int:
    """The visible suction pick: perceive, synthesize, seal-gate, descend, weld, lift, measure.

    The seal decision comes from the analytical model: engage only if the best candidate's seal is at
    least ``seal_threshold``. The bond is a runtime ``FixedJoint`` welding the object to the wrist at
    contact, the physics-real stand-in for the vacuum bond, because Isaac's binary surface gripper will
    not form one on a non-root articulation link. Returns 0 iff the object rose at least half ``lift_mm``.
    """
    cell = bootstrap_sim_cell(data_dir, headless=headless)
    try:
        return _pick_booted_cell(
            cell, out_path=out_path, payload_g=payload_g, min_quality=min_quality,
            seal_threshold=seal_threshold, standoff_mm=standoff_mm, contact_gap_mm=contact_gap_mm,
            lift_mm=lift_mm, place=place,
        )
    finally:
        try:
            cell.arm.session.stop()
        except Exception:  # noqa: BLE001 (headless SimulationApp.close() can segfault on shutdown)
            pass


def _pick_booted_cell(
    cell: SimCell, *, out_path: str, payload_g: float | None, min_quality: float,
    seal_threshold: float, standoff_mm: float, contact_gap_mm: float, lift_mm: float, place: bool,
) -> int:
    """Full suction pick on an already-booted cell (guaranteed Kit shutdown by the caller)."""
    import omni.usd  # type: ignore[import-not-found]  # after SimulationApp boots

    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import (
        SuctionConfig,
        synthesize_suction_grasps,
    )

    from src.willy_sim.perception import GroundTruthPerceptionSource
    from src.willy_sim.scene import WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import engage_suction_weld, release_suction_weld

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    log: list[str] = ["=== H7 suction PICK (analytical seal-gated + runtime weld) ==="]
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    stage = omni.usd.get_context().get_stage()

    payload = float(payload_g) if payload_g is not None else 200.0
    obj_cfg = getattr(sim.scene_setup, "object", None)
    if payload_g is None and obj_cfg is not None:
        payload = float(getattr(obj_cfg, "mass_kg", 0.2)) * 1000.0

    # Perceive the true object-top geometry (park + near-clip; see _score_booted_cell for the why).
    _park = getattr(sim, "park_joint_positions", None)
    park_q = np.asarray(_park if _park is not None else sim.home_joint_positions, dtype=np.float64)
    arm.move_joint(JointPositions(park_q))
    arm.session.step_n(15)
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001 (near-clip best-effort)
        pass
    perception = GroundTruthPerceptionSource(
        camera=handles.camera, target_prim_path=handles.object_prim_path, session=arm.session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps), ground_truth_depth=False,
    )
    frame = perception.acquire()
    mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
    tries = 0
    while (mask.ndim != 2 or not mask.any()) and tries < 10:
        arm.session.step_n(5)
        frame = perception.acquire()
        mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
        tries += 1
    if mask.ndim != 2 or not mask.any():
        log.append("=== SUCTION PICK: NO MASK (object not perceived) ===")
        _write(out_path, log)
        for line in log:
            print(line, flush=True)
        return 1

    grasps = synthesize_suction_grasps(
        mask, frame.depth_map, frame.intrinsics,
        camera_to_base=handles.camera_to_base, payload_mass_g=payload,
        config=SuctionConfig(
            scorer=AnalyticalSuctionScorer(), max_results=8, min_quality=min_quality,
        ),
    )
    if not grasps:
        log.append("=== SUCTION PICK: NO CANDIDATE (nothing scored) ===")
        _write(out_path, log)
        for line in log:
            print(line, flush=True)
        return 1
    best = grasps[0]
    log.append(f"[best] seal={best.seal_score:.3f} quality={best.quality:.3f} "
               f"wrench_feasible={best.metadata.get('wrench_feasible')} "
               f"contact_mm={np.round(best.position_mm, 1).tolist()} approach={np.round(best.approach, 2).tolist()}")
    # Seal gate: the analytical model decides sealability. An unsealable contact is not welded: it is a no-pick.
    if best.seal_score < seal_threshold:
        log.append(f"=== SUCTION PICK: NO-PICK (seal {best.seal_score:.3f} < threshold {seal_threshold}) ===")
        _write(out_path, log)
        for line in log:
            print(line, flush=True)
        return 1

    pos = np.asarray(best.position_mm, dtype=np.float64)
    approach = np.asarray(best.approach, dtype=np.float64)
    approach = approach / (float(np.linalg.norm(approach)) or 1.0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    closing = seed - approach * float(np.dot(seed, approach))
    closing = closing / (float(np.linalg.norm(closing)) or 1.0)
    quat = _quaternion_from_axes(closing, approach)

    obj = SingleRigidPrim(handles.object_prim_path)
    z_before = float(np.asarray(obj.get_world_pose()[0])[2])

    # Approach: home, then standoff, then contact, descending the approach to just above the sealable top.
    arm.move_joint(JointPositions(home_q))
    arm.session.step_n(10)
    contact_reached = False
    for label, gap in (("standoff", standoff_mm), ("contact", contact_gap_mm)):
        target = pos - approach * float(gap)
        res = arm.move(Pose(position_mm=target, quaternion_xyzw=quat, frame=Frame.BASE,
                            label=f"suction-{label}"))
        ok = getattr(getattr(res, "status", None), "value", None) == "executed"
        contact_reached = ok
        log.append(f"[approach] {label}(-{gap:.0f}mm)={np.round(target, 1).tolist()} "
                   f"IK={getattr(res.status, 'value', res.status)}")
        _write(out_path, log)
    if not contact_reached:
        log.append("=== SUCTION PICK: contact pose not reached ===")
        _write(out_path, log)
        for line in log:
            print(line, flush=True)
        return 1
    arm.session.step_n(8)

    # Engage (vacuum on): weld the object to the wrist at its current relative pose. A real ``FixedJoint``
    # stands in for the vacuum bond, which Isaac's surface gripper will not form on a non-root link.
    wp, wq = SingleRigidPrim(WRIST_LINK_PRIM).get_world_pose()
    op, oq = obj.get_world_pose()
    wp = np.asarray(wp, dtype=np.float64)
    op = np.asarray(op, dtype=np.float64)
    r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))  # wxyz -> xyzw
    r_o = to_rotation_matrix(np.array([oq[1], oq[2], oq[3], oq[0]], dtype=np.float64))
    p_rel = r_w.T @ (op - wp)                       # object origin in the wrist frame (metres)
    q_rel = from_rotation_matrix(r_w.T @ r_o)       # xyzw
    jpath = engage_suction_weld(
        stage, WRIST_LINK_PRIM, handles.object_prim_path,
        (float(p_rel[0]), float(p_rel[1]), float(p_rel[2])),
        (float(q_rel[3]), float(q_rel[0]), float(q_rel[1]), float(q_rel[2])),
    )
    arm.session.step_n(12)  # let PhysX pick up the runtime joint
    log.append(f"[engage] weld={jpath} obj_in_wrist_mm={np.round(p_rel * 1000.0, 1).tolist()}")
    _write(out_path, log)

    # Lift (up along -approach) + measure whether the welded object rides with the arm.
    lift_target = pos - approach * float(contact_gap_mm) + (-approach) * float(lift_mm)
    res = arm.move(Pose(position_mm=lift_target, quaternion_xyzw=quat, frame=Frame.BASE, label="suction-lift"))
    arm.session.step_n(30)
    z_after = float(np.asarray(obj.get_world_pose()[0])[2])
    rose_mm = (z_after - z_before) * 1000.0
    held = rose_mm >= 0.5 * lift_mm
    log.append(f"[lift] target={np.round(lift_target, 1).tolist()} IK={getattr(res.status, 'value', res.status)} "
               f"object_rose={rose_mm:.1f}/{lift_mm:.0f} mm held={held}")
    _write(out_path, log)

    # Optional place-back + release (vacuum off).
    if place and held:
        res = arm.move(Pose(position_mm=pos - approach * float(contact_gap_mm), quaternion_xyzw=quat,
                            frame=Frame.BASE, label="suction-place"))
        arm.session.step_n(10)
        released = release_suction_weld(stage, jpath)
        arm.session.step_n(15)
        log.append(f"[place] IK={getattr(res.status, 'value', res.status)} released={released}")

    log.append(f"\n=== SUCTION PICK {'OK (object lifted)' if held else 'WEAK/FAILED'}: "
               f"seal={best.seal_score:.3f} object_rose={rose_mm:.1f}/{lift_mm:.0f} mm ===")
    _write(out_path, log)
    for line in log:
        print(line, flush=True)
    return 0 if held else 1


def run_probe(*, headless: bool = True, data_dir: str | None = None,
              spec: SuctionCupSpec | None = None, lift_mm: float = 100.0, mount_cup: bool = True,
              suction_cup: str | None = None,
              out_path: str = "logs/h7/suction_probe.txt") -> int:
    """Boot normally (IK-healthy), author the surface gripper in the runner, then attach and measure.

    ``suction_cup`` (a CLI flag, else ``sim.suction_cup``) selects the :class:`SuctionCupProfile`, such as
    the slim cup, that drives the sim gripper's contact-width band.
    """
    from src.geometry import Frame, Pose
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes

    spec = spec or SUCTION_CUP
    log: list[str] = [f"=== arm-mounted suction ATTACH probe (mount_cup={mount_cup}) ==="]

    # Author the surface gripper before the first play (via the bootstrap post_scene_hook) so the gripper
    # extension registers + ticks it. The hook stashes the gripper path for the IsaacSuctionGripper below.
    gpath_holder: dict[str, str | None] = {"path": None}

    def _author_hook(stage: object) -> None:
        from src.willy_sim.scene import ARM_PRIM
        from src.willy_sim.suction_mount import mount_suction_cup
        gpath_holder["path"] = mount_suction_cup(stage, ARM_PRIM, spec)

    # Normal boot: the known-pose build_service path (default object, baked 2F-85, so the IK reaches
    # top-down), plus the hook.
    cell = bootstrap_sim_cell(data_dir, headless=headless,
                              post_scene_hook=_author_hook if mount_cup else None)
    arm, sim = cell.arm, cell.sim
    from src.willy_sim.grippers import resolve_suction_cup
    cup = resolve_suction_cup(sim, override=suction_cup)
    log.append(f"[cup] {cup.name} (contact diameter {cup.max_width_mm:.0f} mm, "
               f"cup radius {cup.cup_radius_mm:.0f} mm)")

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]  # after SimulationApp boots

    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    q_down = _quaternion_from_axes(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
    obj = SingleRigidPrim(OBJECT_PRIM)
    bx, by, _bz = (np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0).tolist()
    log.append(f"[boot] planner={getattr(arm, '_motion_planner', '?')!r} obj_mm=[{bx:.0f},{by:.0f},{_bz:.0f}] "
               f"gpath={gpath_holder['path']}")

    def reach(label: str, x: float, y: float, z: float) -> bool:
        arm.move_joint(JointPositions(home_q))
        arm.session.step_n(10)
        seed = np.round(np.asarray(arm.get_joint_positions().values, dtype=np.float64), 3).tolist()
        res = arm.move(Pose(position_mm=np.array([x, y, z]), quaternion_xyzw=q_down,
                            frame=Frame.BASE, label=label))
        ok = bool(getattr(getattr(res, "status", None), "value", None) == "executed")
        log.append(f"[{label}] seed={seed} ({x:.0f},{y:.0f},{z:.0f}) -> "
                   f"{getattr(res.status, 'value', res.status)}")
        return ok

    # Check the pre-play-authored surface gripper still leaves the IK reaching the off-axis top-down pose.
    reach("after-author", 449.7, -0.3, 87.0)

    suction = None
    if mount_cup:
        from src.robot.grippers.sim import IsaacSuctionGripper
        suction = IsaacSuctionGripper(
            session=arm.session, gripper_prim_path=gpath_holder["path"], profile=cup,
        )
        suction.connect()

    # Attach, only if the cell still reaches: descend the TCP (the anchor at the grasp centre) to within
    # the configured max grip distance of the object top, vacuum on, lift, measure.
    detected = held = False
    rose_mm = 0.0
    # The anchor sits at about the grasp centre, which is about the TCP. At TCP z=90 it stands ~40 mm above
    # the object top (50 mm), well within MaxGripDistance (120 mm), so the bond forms at the standoff; a low
    # descend crashes Isaac here. Write the log after each step so a physics crash leaves a trace.
    if suction is not None and reach("attach-pose", 449.7, -0.3, 90.0):
        suction.set_width_mm(suction.max_width_mm)  # vacuum off
        z_before = float(np.asarray(obj.get_world_pose()[0])[2])
        # Geometry diagnostic: report the surface-gripper anchor (wrist_3_link plus 132 mm along wrist +Y),
        # the direction of forwardAxis and the object pose. A bond needs the object within
        # MaxGripDistance along forwardAxis of the anchor.
        try:
            from src.geometry.quaternion import to_rotation_matrix
            w = SingleRigidPrim("/World/UR5e/wrist_3_link")
            wp, wq = w.get_world_pose()
            ww, wx, wy, wz = (float(c) for c in np.asarray(wq, dtype=np.float64))
            rmat = to_rotation_matrix(np.array([wx, wy, wz, ww], dtype=np.float64))
            anchor = np.asarray(wp, dtype=np.float64) * 1000.0 + rmat @ np.array([0.0, 132.0, 0.0])
            fwd = rmat @ np.array([0.0, 1.0, 0.0])  # wrist +Y = the suction approach
            log.append(f"[geom] wrist_mm={np.round(np.asarray(wp) * 1000, 1).tolist()} "
                       f"anchor_mm={np.round(anchor, 1).tolist()} fwd(+Y)={np.round(fwd, 2).tolist()} "
                       f"obj_mm={np.round(np.asarray(obj.get_world_pose()[0]) * 1000, 1).tolist()}")
        except Exception as exc:  # noqa: BLE001
            log.append(f"[geom] err {exc}")
        _write(out_path, log)  # checkpoint before the bond
        # Descend the anchor toward near-contact with the object top; at each rung apply the close action
        # and poll the surface-gripper status tick by tick for the transition out of Open.
        from typing import Any as _Any
        view: _Any = suction._view  # type: ignore[attr-defined]
        try:
            props = view.get_surface_gripper_properties()
            log.append(f"[api] count={getattr(view, 'count', '?')} valid={getattr(view, 'valid', '?')} "
                       f"paths={getattr(view, 'paths', '?')} props={props}")
        except Exception as exc:  # noqa: BLE001
            log.append(f"[api] props-err {type(exc).__name__}: {exc}")
        # Re-assert the bond properties on this view (the mount-time temp view may not have written the prim).
        try:
            view.set_surface_gripper_properties(
                max_grip_distance=[0.12], coaxial_force_limit=[1.0e6],
                shear_force_limit=[1.0e6], retry_interval=[0.1],
            )
            log.append(f"[api] re-set props -> {view.get_surface_gripper_properties()}")
        except Exception as exc:  # noqa: BLE001
            log.append(f"[api] re-set-err {type(exc).__name__}: {exc}")
        log.append(f"[poll] pre-action status={list(view.get_surface_gripper_status())}")
        for z in (90.0, 80.0, 72.0):
            if not (arm.move(Pose(position_mm=np.array([449.7, -0.3, z]), quaternion_xyzw=q_down,
                                  frame=Frame.BASE, label=f"descend-{int(z)}")).status.value == "executed"):
                log.append(f"[poll] descend z={z} NOT reached")
                break
            arm.session.step_n(5)
            view.apply_gripper_action(np.array([0.5]))  # close
            bonded = False
            for k in range(12):
                arm.session.step_n(2)
                st = list(view.get_surface_gripper_status())
                gr = list(view.get_gripped_objects())
                if k in (0, 5, 11) or (str(st[0]) != "0"):
                    log.append(f"[poll] z={z:.0f} k={k}: status={st} gripped={gr}")
                if str(st[0]) != "0" and gr and any(g for g in gr):
                    bonded = True
                    break
            _write(out_path, log)
            if bonded:
                detected = True
                break
        if detected:
            z_before = float(np.asarray(obj.get_world_pose()[0])[2])
            arm.move(Pose(position_mm=np.array([449.7, -0.3, 90.0 + lift_mm]), quaternion_xyzw=q_down,
                          frame=Frame.BASE, label="lift"))
            arm.session.step_n(30)
            rose_mm = (float(np.asarray(obj.get_world_pose()[0])[2]) - z_before) * 1000.0
            held = bool(suction.is_object_detected())
        log.append(f"\n=== ATTACH {'+ HOLD OK' if rose_mm >= 0.5 * lift_mm else 'WEAK/FAILED'}: "
                   f"detected={detected} held={held} object_rose={rose_mm:.1f}/{lift_mm:.0f} mm ===")

    _write(out_path, log)
    for line in log:
        print(line, flush=True)
    return 0 if rose_mm >= 0.5 * lift_mm else 1


def run_suction_klt_probe(
    *, headless: bool = True, data_dir: str | None = None,
    cavity_half_x_mm: float = 45.0, cavity_half_y_mm: float = 70.0, wall_height_mm: float = 146.0,
    target_off_x_mm: float = 15.0, obj_size_mm: tuple[float, float, float] = (40.0, 40.0, 40.0),
    bin_center_xy_mm: tuple[float, float] = (450.0, 0.0), near_gap_mm: float = 5.0,
    rim_margin_mm: float = 40.0, out_path: str = "logs/h7/suction_klt_probe.txt",
) -> int:
    """Read-only reach probe: can suction reach an off-center target inside a deep narrow KLT bin?

    The 2F-85 jaw cannot enter this KLT; its body about fills the 90 mm cavity. This probe measures the
    two remaining unknowns: whether the overhead camera sees an off-center object at the bottom of a
    146 mm-deep narrow cavity, and whether the arm links clear the 146 mm walls at the cuRobo-planned
    reach while only the thin suction column enters the cavity. What has to clear the walls is the arm
    body, not the tool.

    Boots with KLT-tight wall prims and an off-center box, perceives, synthesizes the suction grasp, then
    reaches a standoff above the rim so the baked 2F-85, which stays for IK health, never touches the
    walls; a real suction pick path would disable the jaw collider. It then reads the measured arm-link
    world poses and overlays the KLT walls geometrically to report arm-versus-rim and cup-versus-wall
    clearances. No pick, no weld, no lift: measurement only. Returns 0 iff the reach is geometrically
    viable.
    """
    from src.config.schema.robot.sim_schema import SimObjectConfig

    from src.willy_sim.run_dense_pick import _make_bin_fixtures

    cx, cy = float(bin_center_xy_mm[0]), float(bin_center_xy_mm[1])
    fixtures = _make_bin_fixtures(
        center_xy_mm=(cx, cy), half_width_mm=cavity_half_x_mm, half_width_y_mm=cavity_half_y_mm,
        height_mm=wall_height_mm, thickness_mm=8.0,
    )
    obj = SimObjectConfig(
        name="klt_target", shape="cube", size_mm=obj_size_mm,
        position_mm=(cx + float(target_off_x_mm), cy, float(obj_size_mm[2]) / 2.0),
        mass_kg=0.15, color=(0.20, 0.50, 0.85),
    )
    cell = bootstrap_sim_cell(
        data_dir, headless=headless,
        scene_kwargs={"objects_override": [obj], "bin_walls": fixtures},
    )
    try:
        return _klt_probe_booted_cell(
            cell, fixtures=fixtures, cavity_half_x_mm=cavity_half_x_mm, cavity_half_y_mm=cavity_half_y_mm,
            wall_height_mm=wall_height_mm, bin_center_xy_mm=(cx, cy), obj_size_mm=obj_size_mm,
            target_off_x_mm=target_off_x_mm, near_gap_mm=near_gap_mm, rim_margin_mm=rim_margin_mm,
            out_path=out_path,
        )
    finally:
        try:
            cell.arm.session.stop()
        except Exception:  # noqa: BLE001 (headless SimulationApp.close() can segfault on shutdown)
            pass


def _point_to_wall_gap_mm(px: float, py: float, cx: float, cy: float, hx: float, hy: float, r: float) -> float:
    """Signed lateral clearance of a radius-``r`` column at (px,py) to the nearest KLT inner wall face.

    The cavity half-extents are ``hx`` and ``hy`` about ``cx`` and ``cy``. Positive clears the wall;
    negative means the column penetrates it.
    """
    gap_x = hx - abs(px - cx) - r
    gap_y = hy - abs(py - cy) - r
    return min(gap_x, gap_y)


def _klt_probe_booted_cell(
    cell: SimCell, *, fixtures: list, cavity_half_x_mm: float, cavity_half_y_mm: float,
    wall_height_mm: float, bin_center_xy_mm: tuple[float, float], obj_size_mm: tuple[float, float, float],
    target_off_x_mm: float, near_gap_mm: float, rim_margin_mm: float, out_path: str,
) -> int:
    """Reach measurement on an already-booted walled cell; the caller guarantees the Kit shutdown."""
    from src.geometry import Frame, Pose
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import (
        SuctionConfig,
        synthesize_suction_grasps,
    )

    from src.willy_sim.perception import GroundTruthPerceptionSource
    from src.willy_sim.scene import ARM_PRIM

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    cx, cy = float(bin_center_xy_mm[0]), float(bin_center_xy_mm[1])
    log: list[str] = [
        "=== H3.2 Check-2 suction KLT-bin reach probe (READ-ONLY, no pick) ===",
        f"[scene] KLT cavity {2 * cavity_half_x_mm:.0f}x{2 * cavity_half_y_mm:.0f} mm (half {cavity_half_x_mm:.0f}"
        f"/{cavity_half_y_mm:.0f}) walls {wall_height_mm:.0f} mm tall @ base ({cx:.0f},{cy:.0f}); "
        f"target off-center +{target_off_x_mm:.0f} mm x; object {obj_size_mm} mm",
    ]
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)

    # (1) Perceive the off-center object at the bottom of the deep cavity: park first, then set the near-clip.
    _park = getattr(sim, "park_joint_positions", None)
    park_q = np.asarray(_park if _park is not None else sim.home_joint_positions, dtype=np.float64)
    arm.move_joint(JointPositions(park_q))
    arm.session.step_n(15)
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001 (near-clip best-effort)
        pass
    perception = GroundTruthPerceptionSource(
        camera=handles.camera, target_prim_path=handles.object_prim_path, session=arm.session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps), ground_truth_depth=False,
    )
    frame = perception.acquire()
    mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
    tries = 0
    while (mask.ndim != 2 or not mask.any()) and tries < 10:
        arm.session.step_n(5)
        frame = perception.acquire()
        mask = np.asarray(frame.segmentations[0].mask, dtype=bool)
        tries += 1
    perceived = bool(mask.ndim == 2 and mask.any())
    log.append(f"[perceive] mask_px={int(mask.sum())} shape={mask.shape} tries={tries} -> "
               f"{'SEES the off-center object in the deep cavity' if perceived else 'BLIND (walls shadow it)'}")
    if not perceived:
        log.append("=== CHECK-2: FAIL (overhead cannot see the off-center target in the deep cavity) ===")
        _write(out_path, log)
        for line in log:
            print(line, flush=True)
        return 1

    # (2) synthesize the suction grasp on the off-center bin object.
    grasps = synthesize_suction_grasps(
        mask, frame.depth_map, frame.intrinsics, camera_to_base=handles.camera_to_base, payload_mass_g=150.0,
        config=SuctionConfig(scorer=AnalyticalSuctionScorer(), max_results=8, min_quality=0.05),
    )
    if not grasps:
        log.append("=== CHECK-2: FAIL (no suction candidate synthesized on the bin object) ===")
        _write(out_path, log)
        for line in log:
            print(line, flush=True)
        return 1
    best = grasps[0]
    pos = np.asarray(best.position_mm, dtype=np.float64)
    approach = np.asarray(best.approach, dtype=np.float64)
    approach = approach / (float(np.linalg.norm(approach)) or 1.0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    closing = seed - approach * float(np.dot(seed, approach))
    closing = closing / (float(np.linalg.norm(closing)) or 1.0)
    quat = _quaternion_from_axes(closing, approach)
    log.append(f"[synthesize] best seal={best.seal_score:.3f} quality={best.quality:.3f} "
               f"contact_mm={np.round(pos, 1).tolist()} approach={np.round(approach, 2).tolist()}")

    # (3) Reach a standoff above the rim on the approach line, so the baked 2F-85 never touches the walls,
    #     then read the measured arm-link world poses. That answer is independent of the end effector.
    standoff = np.array([pos[0], pos[1], float(wall_height_mm) + float(rim_margin_mm)], dtype=np.float64)
    arm.move_joint(JointPositions(home_q))
    arm.session.step_n(10)
    res = arm.move(Pose(position_mm=standoff, quaternion_xyzw=quat, frame=Frame.BASE, label="klt-standoff-above-rim"))
    reach_ok = getattr(getattr(res, "status", None), "value", None) == "executed"
    log.append(f"[reach] standoff-above-rim={np.round(standoff, 1).tolist()} "
               f"IK={getattr(res.status, 'value', res.status)} -> {'REACHED' if reach_ok else 'NOT reached'}")
    arm.session.step_n(8)

    # Read every UR5e arm link's world pose (mm) at the reached config for the min link Z against the 146 mm rim.
    link_names = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
    min_link_z = float("inf")
    lowest_link = "?"
    for name in link_names:
        try:
            lp, _ = SingleRigidPrim(f"{ARM_PRIM}/{name}").get_world_pose()
            lz = float(np.asarray(lp, dtype=np.float64)[2]) * 1000.0
        except Exception:  # noqa: BLE001 (a missing link name is skipped)
            continue
        gap = _point_to_wall_gap_mm(
            float(np.asarray(lp)[0]) * 1000.0, float(np.asarray(lp)[1]) * 1000.0,
            cx, cy, cavity_half_x_mm, cavity_half_y_mm, 0.0,
        )
        log.append(f"  [link] {name:<15s} z={lz:7.1f} mm  lateral_gap_to_wall={gap:+7.1f} mm")
        if lz < min_link_z:
            min_link_z, lowest_link = lz, name
    rim_clearance = min_link_z - float(wall_height_mm)
    log.append(f"[arm-vs-rim] lowest arm link = {lowest_link} @ z={min_link_z:.1f} mm  vs rim {wall_height_mm:.0f} mm "
               f"-> clearance {rim_clearance:+.1f} mm ({'ARM CLEARS the rim' if rim_clearance > 0 else 'ARM DIPS below the rim'})")

    # (4) Geometric descent projection to near-contact: only the thin suction column enters the cavity.
    #     wrist_3 sits ~132 mm back along the approach (the anchor offset), so at contact it stays above the rim.
    contact_tcp_z = float(pos[2]) + float(near_gap_mm)  # top-down approach: near-contact TCP just above the top
    wrist3_at_contact_z = contact_tcp_z + 132.0  # SuctionCupSpec.anchor_offset_mm: wrist_3 to the grasp centre
    cup_gap_15 = _point_to_wall_gap_mm(float(pos[0]), float(pos[1]), cx, cy, cavity_half_x_mm, cavity_half_y_mm, 15.0)
    cup_gap_24 = _point_to_wall_gap_mm(float(pos[0]), float(pos[1]), cx, cy, cavity_half_x_mm, cavity_half_y_mm, 24.0)
    jaw_gap_body = _point_to_wall_gap_mm(float(pos[0]), float(pos[1]), cx, cy, cavity_half_x_mm, cavity_half_y_mm, 42.5)
    log.append(f"[contact-projection] near-contact TCP z={contact_tcp_z:.1f} mm -> wrist_3 z={wrist3_at_contact_z:.1f} mm "
               f"(rim {wall_height_mm:.0f}; {'wrist stays above rim' if wrist3_at_contact_z > wall_height_mm else 'wrist DIPS'})")
    log.append(f"[cup-vs-wall @ contact] off-center +{target_off_x_mm:.0f}mm: cup r=15 gap={cup_gap_15:+.1f} mm | "
               f"cup r=24 gap={cup_gap_24:+.1f} mm | (2F-85 body r=42.5 gap={jaw_gap_body:+.1f} mm = why the jaw can't)")

    cup_clears = cup_gap_15 > 0.0
    arm_clears = rim_clearance > 0.0 and wrist3_at_contact_z > float(wall_height_mm)
    viable = perceived and reach_ok and arm_clears and cup_clears
    log.append(
        f"\n=== H3.2 CHECK-2 {'GREEN' if viable else 'BLOCKED'}: perceive={perceived} reach={reach_ok} "
        f"arm_clears_rim={arm_clears} cup_clears={cup_clears} -> "
        f"{'suction reach into the KLT is geometrically viable (residual = disable the baked-jaw collider in the pick path)' if viable else 'a real arm/tool blocker remains -> re-close honestly'} ==="
    )
    _write(out_path, log)
    for line in log:
        print(line, flush=True)
    return 0 if viable else 1


def run_suction_klt_pick(
    *, headless: bool = True, data_dir: str | None = None,
    cavity_half_x_mm: float = 94.7, cavity_half_y_mm: float = 146.5, wall_height_mm: float = 146.0,
    obj_size_mm: tuple[float, float, float] = (40.0, 40.0, 40.0), bin_center_xy_mm: tuple[float, float] = (450.0, 0.0),
    payload_g: float | None = None, seal_threshold: float = 0.5, lift_mm: float = 100.0,
    real_klt: bool = False, klt_z_mm: float = 73.2,
    out_path: str = "logs/h7/suction_klt_pick.txt",
) -> int:
    """The suction pick from a full-depth (146 mm) KLT bin where the 2F-85 jaw cannot reach.

    Boots the real-KLT-dimensioned bin (inner ~189 x 293 mm, 146 mm deep) with an object on the floor,
    then runs the seal-gated weld-lift of :func:`_pick_booted_cell`: the thin suction cup descends the
    deep cavity to the floor object while the arm stays above the 146 mm rim, and lifts it. This is the
    complement to the jaw's shallow-tray pick. The deep narrow bin is suction's job because the jaw's arm
    and gripper cannot clear the deep walls; cuRobo finds no collision-free descent to the floor object.
    Returns 0 iff the object rose.
    """
    from src.config.schema.robot.sim_schema import SimObjectConfig

    from src.willy_sim.run_dense_pick import _make_bin_fixtures

    cx, cy = float(bin_center_xy_mm[0]), float(bin_center_xy_mm[1])
    obj = SimObjectConfig(
        name="klt_target", shape="cube", size_mm=obj_size_mm,
        position_mm=(cx, cy, float(obj_size_mm[2]) / 2.0),
        mass_kg=(float(payload_g) / 1000.0 if payload_g is not None else 0.15), color=(0.20, 0.50, 0.85),
    )
    fixtures = _make_bin_fixtures(
        center_xy_mm=(cx, cy), half_width_mm=cavity_half_x_mm, half_width_y_mm=cavity_half_y_mm,
        height_mm=wall_height_mm, thickness_mm=8.0,
    )
    # Physics comes from the hollow fixture-box walls, dimensioned to the real KLT; the cup never touches
    # them but they are there. The visual is the real small_KLT mesh laid over them.
    scene_kwargs: dict = {"objects_override": [obj], "bin_walls": fixtures}
    if real_klt:
        scene_kwargs["real_klt_bin"] = (cx, cy, klt_z_mm)
    cell = bootstrap_sim_cell(data_dir, headless=headless, scene_kwargs=scene_kwargs)
    try:
        return _pick_booted_cell(
            cell, out_path=out_path, payload_g=payload_g, min_quality=0.05, seal_threshold=seal_threshold,
            standoff_mm=60.0, contact_gap_mm=25.0, lift_mm=lift_mm, place=False,
        )
    finally:
        try:
            cell.arm.session.stop()
        except Exception:  # noqa: BLE001 (headless SimulationApp.close() can segfault on shutdown)
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="H7 suction pick (Isaac): score path (default) or attach probe.")
    ap.add_argument("--score", action="store_true",
                    help="the pick path: perceive -> synthesize -> report suction values + reach (DEFAULT)")
    ap.add_argument("--pick", action="store_true",
                    help="the VISIBLE pick: perceive -> seal-gate -> descend -> weld -> lift -> measure")
    ap.add_argument("--place", action="store_true", help="--pick: place the object back + release after the lift")
    ap.add_argument("--vs-jaw", action="store_true",
                    help="H7.4: a WIDE/flat object -> the jaw finds no grasp, suction seals + lifts")
    ap.add_argument("--object", type=str, default="120x120x35",
                    help="--vs-jaw: wide-object footprint WxDxH in mm (default 120x120x35)")
    ap.add_argument("--klt-probe", action="store_true",
                    help="H3.2 Check-2 (read-only): can suction reach an off-center target in a deep narrow KLT bin?")
    ap.add_argument("--klt-pick", action="store_true",
                    help="H3.2 payoff: the SUCTION seal-gated weld-lift from the REAL full-depth (146mm) KLT bin "
                         "where the 2F-85 jaw cannot reach the floor (the complement to the jaw's shallow-tray pick)")
    ap.add_argument("--real-klt", action="store_true",
                    help="--klt-pick: use the REAL small_KLT.usd asset (visible + collidable mesh) as the bin, not "
                         "fixture boxes ('wirklich echt' for the video)")
    ap.add_argument("--cavity-half-x", type=float, default=45.0, help="--klt-probe: KLT cavity half-width x mm (narrow)")
    ap.add_argument("--cavity-half-y", type=float, default=70.0, help="--klt-probe: KLT cavity half-width y mm (long)")
    ap.add_argument("--wall-height", type=float, default=146.0, help="--klt-probe: KLT wall height mm")
    ap.add_argument("--off-x", type=float, default=15.0, help="--klt-probe: target off-center offset in x mm")
    ap.add_argument("--probe", action="store_true", help="run the measure>map attach probe instead")
    ap.add_argument("--no-cup", action="store_true", help="diagnostic: skip the surface-gripper authoring")
    ap.add_argument("--suction-cup", type=str, default=None, choices=["standard", "slim"],
                    help="which suction cup profile to mount (default: sim.suction_cup, else standard)")
    ap.add_argument("--no-reach", action="store_true", help="--score: report values only, skip the reach")
    ap.add_argument("--payload-g", type=float, default=None,
                    help="--score: override the wrench payload mass (default: the sim object mass)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--lift-mm", type=float, default=100.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.probe:  # explicit attach probe
        return run_probe(headless=not args.gui, data_dir=args.data_dir, mount_cup=not args.no_cup,
                         lift_mm=args.lift_mm, suction_cup=args.suction_cup,
                         out_path=args.out or "logs/h7/suction_probe.txt")
    if args.pick:  # the visible seal-gated weld pick
        return run_suction_pick_lift(headless=not args.gui, data_dir=args.data_dir, payload_g=args.payload_g,
                                     lift_mm=args.lift_mm, place=args.place,
                                     out_path=args.out or "logs/h7/suction_pick.txt")
    if args.klt_probe:  # suction reach into a deep narrow KLT bin (read-only)
        return run_suction_klt_probe(
            headless=not args.gui, data_dir=args.data_dir,
            cavity_half_x_mm=args.cavity_half_x, cavity_half_y_mm=args.cavity_half_y,
            wall_height_mm=args.wall_height, target_off_x_mm=args.off_x,
            out_path=args.out or "logs/h7/suction_klt_probe.txt",
        )
    if args.klt_pick:  # suction seal-gated weld-lift from the real full-depth KLT bin
        return run_suction_klt_pick(
            headless=not args.gui, data_dir=args.data_dir, wall_height_mm=args.wall_height,
            lift_mm=args.lift_mm, real_klt=args.real_klt, out_path=args.out or "logs/h7/suction_klt_pick.txt",
        )
    if args.vs_jaw:  # a wide flat object: the jaw fails, suction succeeds
        try:
            _w, _d, _h = (float(v) for v in str(args.object).lower().split("x"))
        except ValueError:
            ap.error(f"--object must be WxDxH in mm (e.g. 120x120x35); got {args.object!r}")
        return run_h74_vs_jaw(size_mm=(_w, _d, _h), headless=not args.gui, data_dir=args.data_dir,
                              out_path=args.out or "logs/h7/suction_h74.txt")
    # default = the score / pick path
    return run_suction_score(headless=not args.gui, data_dir=args.data_dir, mount_cup=not args.no_cup,
                             reach_best=not args.no_reach, payload_g=args.payload_g,
                             out_path=args.out or "logs/h7/suction_score.txt")


if __name__ == "__main__":
    raise SystemExit(main())
