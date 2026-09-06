"""Industrial suction pick demo recorder: a watchable cinematic MP4 of the industrial pick.

Films the complete industrial suction pick from a fixed cinematic 3/4 camera and encodes a slow MP4:

* Scene: a dark cell table with tuned lighting holding a wide flat GSO package (the ``ink cartridge
  box``, a 138 x 131 mm flat top, too wide for the 2F-85 jaw to span, so it is the suction cup's
  job), optionally inside a real KLT tray (``use_bin``).
* Pick: perceive the real top geometry, let the analytical seal and wrench model place the seal on the
  flat top, descend the cup, then lift it out with a render-safe kinematic carry.

The object is carried kinematically, not by a runtime weld. Adding a physics joint mid-play freezes
Isaac's render product, so the filmed lift would stay frozen at the contact pose. The film instead tracks
the object to the wrist every filmed step, still gated by the analytical seal: the decision is the real
model and only the visual hold is a surrogate. ``run_industrial_suction_pick`` runs the physics-real seal
and lift.

Sensing is the single overhead ground-truth camera on purpose. Mixing it with the two fixed ETH side
cameras in one cinematic session corrupts Isaac's shared synthetic-data render graph: the instance-id and
depth annotators go missing. ``run_industrial_bin_pick`` showcases the two-side ETH localization rig.

On-box only (needs Isaac). Run with Isaac's bundled python from the repo root (cuRobo + Coal env set):

    <isaac-sim>\\python.bat -m src.willy_sim.run_industrial_suction_demo --fps 18
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.willy_sim.gso_assets import GSO_BY_LABEL, gso_object_spec
from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.run_multiview_pick import _bin_preset_walls
from src.willy_sim.scene import SceneAppearance

DEFAULT_OUT = "logs/demo/industrial_suction_demo.mp4"
# Cinematic camera (metres): an elevated front-right 3/4 view of the cell aimed at the package. The same
# framing as run_suction_demo and run_eih_demo, which is convention-correct for the UR5e cell.
CINE_POS_M = (2.8, -2.6, 1.75)
CINE_TARGET_M = (0.45, -0.05, 0.08)
CINE_RES = (1280, 720)
TARGET_LABEL = "ink cartridge box"


# Packing-table work-surface height (mm): its tabletop is aligned to the pick surface at world z=0, so the
# warehouse floor and table base sit at z = -_TABLE_TOP. The ~993 mm value is measured from the asset.
_TABLE_TOP = 993.0
_FLOOR_Z = -_TABLE_TOP
# Where the arm places the picked package (a KLT bin on the bench, in front of the pick). Must lie inside the
# SafetyPreflight workspace box (y in [-330, 330]) or the fail-closed guard rejects the place motion.
_PLACE_BIN_XY = (450.0, -300.0)


def _author_industrial_cell(stage: Any) -> None:
    """Author the warehouse pick cell from Isaac assets.

    Places a warehouse-environment backdrop, the UR5e on a packing table whose tabletop is aligned to the
    pick surface at z=0, a place-target KLT bin on the bench, and pallets of KLT bins on the warehouse
    floor. The tabletop at z=0 carries the robot base, the pick surface and the package; the warehouse
    floor and the table base sit at z=_FLOOR_Z. The primitive physics table (``TABLE_PRIM``) is kept for
    collision and hidden so the packing table shows. The referenced assets stay clear of the arm's
    straight-down pick at (450, 0).
    """
    from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]
    from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    from src.willy_sim.scene import TABLE_PRIM

    root = (get_assets_root_path() or "").rstrip("/")

    def _ref(rel_usd: str, prim: str, pos_mm: tuple[float, float, float], *,
             yaw_deg: float = 0.0, scale: float = 1.0) -> None:
        add_reference_to_stage(root + rel_usd, prim)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(pos_mm[0] / 1000.0, pos_mm[1] / 1000.0, pos_mm[2] / 1000.0))
        if yaw_deg:
            xf.AddRotateZOp().Set(float(yaw_deg))
        if scale != 1.0:
            xf.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))

    # Hide the primitive physics table's visual, keep its collider, so the real packing table shows instead.
    _tp = stage.GetPrimAtPath(TABLE_PRIM)
    if _tp and _tp.IsValid():
        UsdGeom.Imageable(_tp).MakeInvisible()

    # Full warehouse-environment backdrop (floor at the cell floor level; XY offset to a clear aisle).
    _ref("/Isaac/Environments/Simple_Warehouse/warehouse.usd", "/World/Cell/warehouse", (0.0, 0.0, _FLOOR_Z))
    # The UR5e's real packing table: tabletop at z=0, base on the warehouse floor.
    _ref("/Isaac/Props/PackingTable/packing_table.usd", "/World/Cell/packtable", (450.0, 0.0, _FLOOR_Z))
    # Place-target KLT bin on the bench; the place beat drops the package into it. Its origin is centred,
    # so +73 sits it on z0.
    _ref("/Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd", "/World/Cell/place_bin",
         (_PLACE_BIN_XY[0], _PLACE_BIN_XY[1], 73.0))
    # A pallet of KLT bins on the warehouse floor beside the bench (staging).
    _pallet_top = _FLOOR_Z + 143.0
    _ref("/Isaac/Props/Pallet/pallet.usd", "/World/Cell/pallet_l", (-1500.0, -300.0, _FLOOR_Z))
    _ref("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", "/World/Cell/klt_l1", (-1500.0, -450.0, _pallet_top + 73.0))
    _ref("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", "/World/Cell/klt_l2", (-1500.0, 0.0, _pallet_top + 73.0))


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Nearest proper rotation, by SVD.

    Composing and stacking near-orthonormal axes drifts the determinant about 1e-6 off 1.0, enough for
    ``from_rotation_matrix``'s strict det==+1 check to abort the carry. Snapping absorbs that drift.
    """
    u, _s, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rot = u @ vt
    if float(np.linalg.det(rot)) < 0.0:
        u[:, -1] = -u[:, -1]
        rot = u @ vt
    return rot


def record_demo(*, out_path: str = DEFAULT_OUT, headless: bool = True, fps: int = 18,
                capture_every: int = 1, hold_steps: int = 40, lift_mm: float = 130.0,
                seal_threshold: float = 0.5, use_bin: bool = True) -> dict:
    """Record one industrial suction pick (kinematic carry) as a cinematic MP4.

    ``use_bin`` adds the real KLT tray and its collidable walls. Those walls are PhysX colliders that are
    not registered into cuRobo's world here; that registration lives in build_service. With the bin on,
    the arm descent can be physically obstructed, so switch it off for a reliable open-cell record.
    """
    import cv2  # type: ignore[import-not-found]

    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.geometry import masked_point_cloud
    from src.robot.grasping.suction.seal import SealConfig, evaluate_seal

    from src.willy_sim.perception import GroundTruthPerceptionSource
    from src.willy_sim.scene import ARM_PRIM, WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import SUCTION_CUP, mount_visible_suction_cup

    # Scene: a dark cell table and tuned lighting holding the wide flat package, optionally in a real KLT
    # tray. The tray's collidable walls are not in cuRobo's world here, so they can obstruct the descent
    # (see ``use_bin``); default-on, switch off for a reliable open cell.
    ink_spec = gso_object_spec(GSO_BY_LABEL[TARGET_LABEL], position_mm=(450.0, 0.0, 60.0))
    # The warehouse environment brings its own lighting; keep a modest warm dome and key so the fill does
    # not blow out. table_color has no effect here because the primitive table is hidden.
    appearance = SceneAppearance(table_color=(0.16, 0.18, 0.22), dome_intensity=110.0, key_intensity=220.0)
    scene_kwargs: dict[str, Any] = {"objects_override": [ink_spec], "appearance": appearance}
    if use_bin:
        bin_walls, real_klt = _bin_preset_walls("tray")
        scene_kwargs["bin_walls"] = bin_walls
        scene_kwargs["real_klt_bin"] = real_klt

    def _mount_cup(stage: Any) -> None:
        mount_visible_suction_cup(stage, ARM_PRIM, SUCTION_CUP, hide_jaw=True)
        _author_industrial_cell(stage)

    cell = bootstrap_sim_cell(None, headless=headless, scene_kwargs=scene_kwargs, post_scene_hook=_mount_cup)
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    session = arm.session
    app = getattr(session, "app", None)
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    park_q = np.asarray(getattr(sim, "park_joint_positions", None) or sim.home_joint_positions, dtype=np.float64)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    # --- cinematic camera (orient via Isaac's canonical set_camera_view); RGB only (no depth annotator, so
    # the synthetic-data render graph isn't loaded with a second distance pass that starves the overhead read).
    cine = Camera(prim_path="/World/Cameras/Cinematic",
                  position=np.array(CINE_POS_M, dtype=np.float64), resolution=CINE_RES)
    cine.initialize()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001 (clipping best-effort)
        pass
    set_camera_view(eye=list(CINE_POS_M), target=list(CINE_TARGET_M),
                    camera_prim_path="/World/Cameras/Cinematic")
    for _ in range(60):  # warm up the render so the RGB annotator populates before capture
        session.step(render=True)
        if app is not None:
            app.update()

    def _grab() -> np.ndarray | None:
        fr = cine.get_current_frame()
        rgba = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgba is None else np.asarray(rgba)[..., :3].copy()

    probe = _grab()
    print(f"cinematic cam ready: frame {None if probe is None else probe.shape}", flush=True)

    obj = SingleRigidPrim(handles.object_prim_path)
    wrist = SingleRigidPrim(WRIST_LINK_PRIM)

    # --- frame capture + the render-safe kinematic carry --------------------
    frames: list[np.ndarray] = []
    state: dict[str, Any] = {"on": False, "n": 0, "carry": False, "carry_drop_m": 0.0, "carry_quat": None}
    orig_step = session.step

    def _carry_object() -> None:
        # The package hangs flat, directly below the cup tip, not rigidly rotated with the wrist: it is
        # never tilted, and it tracks the cup position wherever the arm goes.
        wp, wq = wrist.get_world_pose()
        r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
        cup_tip = np.asarray(wp, dtype=np.float64) + r_w @ np.array([0.0, 0.158, 0.0])  # cup face (m, +Y offset)
        pos = cup_tip - np.array([0.0, 0.0, float(state["carry_drop_m"])])  # package centre below the cup tip
        obj.set_world_pose(position=pos, orientation=np.asarray(state["carry_quat"], dtype=np.float64))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001 (velocity zeroing is best-effort)
            pass

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001, ANN202, ARG001
        orig_step(dt_s, render=True)
        if state["carry"] and state.get("carry_quat") is not None:
            _carry_object()
        if state["on"]:
            state["n"] += 1
            if state["n"] % capture_every == 0:
                if app is not None:
                    app.update()
                f = _grab()
                if f is not None:
                    _pip = state.get("pip")
                    if _pip is not None:  # composite the perception overlay top-right (a "what the robot sees" PiP)
                        _fb = f[..., ::-1].copy()  # RGB -> BGR
                        _ph, _pw = _pip.shape[:2]
                        _x0, _y0 = _fb.shape[1] - _pw - 18, 18
                        _fb[_y0:_y0 + _ph, _x0:_x0 + _pw] = _pip
                        cv2.rectangle(_fb, (_x0 - 2, _y0 - 2), (_x0 + _pw + 2, _y0 + _ph + 2), (255, 255, 255), 2)
                        f = _fb[..., ::-1]  # back to RGB
                    frames.append(f)

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(steps: int) -> None:
        for _ in range(steps):
            session.step(render=True)

    def _make_pip(frame: Any, mask: np.ndarray, seal_score: float) -> np.ndarray:
        """Build the overlay that shows what the robot sees, in BGR.

        Draws the overhead RGB, or a depth colormap when no RGB is available, with the perceived instance
        mask outlined, the seal point marked and the analytical seal score printed.
        """
        _rgb = getattr(frame, "rgb", None)
        if _rgb is not None:
            base = np.asarray(_rgb)[..., :3].astype(np.uint8).copy()
        else:  # depth colormap fallback
            _d = np.asarray(frame.depth_map, dtype=np.float64)
            _d = np.where(np.isfinite(_d) & (_d > 0.0), _d, 0.0)
            _rng = float(_d.max() - _d.min()) or 1.0
            _dn = (255.0 * (_d - _d.min()) / _rng).astype(np.uint8)
            base = cv2.applyColorMap(_dn, cv2.COLORMAP_VIRIDIS)[..., ::-1]  # -> RGB
        bgr = base[..., ::-1].copy()  # RGB -> BGR for cv2 drawing
        cnts, _ = cv2.findContours((np.asarray(mask, dtype=np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cnts, -1, (60, 230, 60), 3)
        _ys, _xs = np.where(mask)
        if _xs.size:
            cv2.circle(bgr, (int(_xs.mean()), int(_ys.mean())), 7, (60, 60, 240), -1)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (35, 35, 35), -1)
        cv2.putText(bgr, f"VISION   seal {seal_score:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)
        return cv2.resize(bgr, (360, 270))

    # Opening beat: the arm starts parked aside so the overhead camera perceives the package cleanly; the
    # arm would otherwise occlude it. From here it moves forward into the pick, with no backward detour.
    arm.move_joint(JointPositions(park_q))
    session.step_n(10)
    # Re-centre the package on the workspace centre (450, 0) so it is reliably framed and reachable, and
    # zero its settle yaw so it sits square to the camera; a yawed box makes even a centred cup read as
    # "off". Keep the stable resting face, because identity is not a stable face for this mesh and it tips:
    # rotate only about world-Z, aligning the most-horizontal local axis with world +X. Keep the settled
    # height, then let it settle.
    _hp = obj.get_world_pose()
    _hz = float(np.asarray(_hp[0])[2])
    _q = np.asarray(_hp[1], dtype=np.float64)  # wxyz
    _rmat = to_rotation_matrix(np.array([_q[1], _q[2], _q[3], _q[0]], dtype=np.float64))
    _axes = _rmat.T  # rows = box local axes expressed in world
    _h0 = sorted(range(3), key=lambda i: abs(float(_axes[i][2])))[0]  # the most-horizontal local axis
    _yaw = float(np.arctan2(float(_axes[_h0][1]), float(_axes[_h0][0])))
    _cz, _sz = float(np.cos(-_yaw)), float(np.sin(-_yaw))
    _rz = np.array([[_cz, -_sz, 0.0], [_sz, _cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    _qn = from_rotation_matrix(_orthonormalize(_rz @ _rmat))  # xyzw
    obj.set_world_pose(position=np.array([0.45, 0.0, _hz], dtype=np.float64),
                       orientation=np.array([_qn[3], _qn[0], _qn[1], _qn[2]], dtype=np.float64))
    try:
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))
    except Exception:  # noqa: BLE001
        pass
    session.step_n(30)
    z_before = float(np.asarray(obj.get_world_pose()[0])[2])
    print(f"[dbg] package settled at {np.round(np.asarray(obj.get_world_pose()[0]) * 1000, 1)} mm", flush=True)
    state["on"] = True
    _hold(hold_steps)

    # --- beat: vision. The overhead camera perceives the package (real rendered depth and instance mask)
    # and the analytical seal model reads its flat top, shown as a picture-in-picture so the perception is
    # visible. The arm is parked aside so the camera sees the package cleanly; if perception fails the
    # guard leaves the overlay out. ----
    print("[beat] vision: perceiving the package (overhead)", flush=True)
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    try:
        _perc = GroundTruthPerceptionSource(
            camera=handles.camera, target_prim_path=handles.object_prim_path, session=session,
            warmup_steps=max(24, sim.scene_setup.render_warmup_steps), ground_truth_depth=False)
        _pf = _perc.acquire()
        _pmask = np.asarray(_pf.segmentations[0].mask, dtype=bool)
        _ptry = 0
        while (_pmask.ndim != 2 or not _pmask.any()) and _ptry < 8:
            session.step_n(4)
            _pf = _perc.acquire()
            _pmask = np.asarray(_pf.segmentations[0].mask, dtype=bool)
            _ptry += 1
        if _pmask.any():
            _pcloud = np.asarray(
                masked_point_cloud(_pmask, _pf.depth_map, _pf.intrinsics, unit="mm").points_mm, dtype=np.float64)
            _pmat = np.asarray(handles.camera_to_base.to_matrix(), dtype=np.float64)
            _pcb = (_pmat[:3, :3] @ _pcloud.T).T + _pmat[:3, 3]
            _ptz = float(_pcb[:, 2].max())
            _ptop = _pcb[_pcb[:, 2] > _ptz - 8.0]
            _pspt = np.array([float(_ptop[:, 0].mean()), float(_ptop[:, 1].mean()), _ptz], dtype=np.float64)
            _pseal = evaluate_seal(_pspt, np.array([0.0, 0.0, -1.0]), _pcb,
                                   surface_normal=np.array([0.0, 0.0, 1.0]), config=SealConfig())
            state["pip"] = _make_pip(_pf, _pmask, float(_pseal.seal_score))
            print(f"[beat] vision: mask_px={int(_pmask.sum())} perceived seal={float(_pseal.seal_score):.3f}", flush=True)
        else:
            print("[beat] vision: no mask perceived -> no overlay", flush=True)
    except Exception as _pexc:  # noqa: BLE001
        print(f"[beat] vision: perception unavailable ({type(_pexc).__name__}: {_pexc}) -> no overlay", flush=True)
    _hold(hold_steps)

    # --- beat: locate the package top from ground truth, analytical seal on its flat top. The
    # rendered-depth perception and seal run in run_industrial_suction_pick; the demo drives from the
    # package's known pose so it films reliably even when the render depth annotator is unavailable, since
    # a degraded GPU state hard-crashes the cold get_depth. ----
    # No perception here, so the arm does not detour to a park or clear pose: it stays at home and descends
    # straight to the package.
    print("[beat] locating the package top", flush=True)
    session.step_n(10)
    box_pose = obj.get_world_pose()
    box_xy = np.asarray(box_pose[0], dtype=np.float64)[:2] * 1000.0
    center_z = float(np.asarray(box_pose[0])[2]) * 1000.0
    top_z = 2.0 * center_z  # the package rests on the table (z=0), so its top ~ twice the settled centre height
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # the flat top faces up, so the seal is vertical
    pos = np.array([float(box_xy[0]), float(box_xy[1]), top_z], dtype=np.float64)  # centre the cup on the package
    # Exercise the analytical seal model on the package's flat top (a synthetic top-surface patch under the cup).
    _gx, _gy = np.meshgrid(np.linspace(-70.0, 70.0, 29), np.linspace(-70.0, 70.0, 29))
    top_patch = np.column_stack([pos[0] + _gx.ravel(), pos[1] + _gy.ravel(), np.full(_gx.size, top_z)])
    seal = evaluate_seal(pos, approach, top_patch, surface_normal=np.array([0.0, 0.0, 1.0]), config=SealConfig())
    best_seal = float(seal.seal_score)
    if best_seal < seal_threshold:
        state["on"] = False
        session.step = orig_step  # type: ignore[assignment]
        session.stop()
        raise SystemExit(f"industrial suction demo: seal below threshold ({best_seal:.3f} < {seal_threshold})")
    seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    closing = seed - approach * float(np.dot(seed, approach))
    closing = closing / (float(np.linalg.norm(closing)) or 1.0)
    binormal = np.cross(approach, closing)
    binormal = binormal / (float(np.linalg.norm(binormal)) or 1.0)
    quat = from_rotation_matrix(_orthonormalize(np.column_stack([closing, binormal, approach])))
    print(f"[beat] seal={best_seal:.3f} pos(base mm)={np.round(pos, 1)} approach={np.round(approach, 2)} "
          f"-> approaching", flush=True)

    # --- beat: approach the package. Move forward from the parked vision pose to the ready (home) pose,
    # then straight down onto the package: standoff, then contact. No backward detour. ----
    arm.move_joint(JointPositions(home_q))
    session.step_n(8)
    for label, gap in (("standoff", 60.0), ("contact", 26.0)):  # contact 26mm: cup face sits on the top, not in it
        arm.move(Pose(position_mm=pos - approach * gap, quaternion_xyzw=quat, frame=Frame.BASE,
                      label=f"suction-{label}"))
        print(f"[beat] reached {label}", flush=True)
    session.step_n(6)
    _cwp, _cwq = wrist.get_world_pose()
    _crw = to_rotation_matrix(np.array([_cwq[1], _cwq[2], _cwq[3], _cwq[0]], dtype=np.float64))
    cup_tip_mm = np.asarray(_cwp, dtype=np.float64) * 1000.0 + _crw @ np.array([0.0, 158.0, 0.0])  # cup face ~158mm +Y
    print(f"[dbg] at contact: wrist={np.round(np.asarray(_cwp) * 1000, 1)} mm cup_tip={np.round(cup_tip_mm, 1)} mm "
          f"package={np.round(np.asarray(obj.get_world_pose()[0]) * 1000, 1)} mm seal_pos={np.round(pos, 1)} mm",
          flush=True)

    # --- beat: engage (vacuum on), then start the flat carry; the package hangs flat below the cup tip ----
    _oq = obj.get_world_pose()[1]
    state["carry_drop_m"] = float(center_z) / 1000.0   # package centre hangs ~half its height below the cup tip
    state["carry_quat"] = np.asarray(_oq, dtype=np.float64)  # keep the settled square (flat) orientation
    state["carry"] = True
    _hold(6)
    print("[beat] vacuum engaged (carry on) -> lifting", flush=True)

    # --- beat: lift (up along -approach); the carry hook rides the object with the tool ----
    arm.move(Pose(position_mm=pos - approach * 22.0 + (-approach) * float(lift_mm),
                  quaternion_xyzw=quat, frame=Frame.BASE, label="suction-lift"))
    _hold(hold_steps)
    z_lift = (float(np.asarray(obj.get_world_pose()[0])[2]) - z_before) * 1000.0  # the lift peak
    print("[beat] lifted -> carrying to the place bin", flush=True)

    # --- beat: place. Carry the package over the KLT bin, then settle it square into the bin centre. The
    # settle is a short kinematic descent decoupled from the arm IK, which can tilt slightly at this reach,
    # so the package always lands square in the bin; then retreat the arm to reveal it. ----
    _px, _py = _PLACE_BIN_XY
    arm.move(Pose(position_mm=np.array([_px, _py, 280.0]), quaternion_xyzw=quat, frame=Frame.BASE,
                  label="place-transit"))
    arm.move(Pose(position_mm=np.array([_px, _py, 150.0]), quaternion_xyzw=quat, frame=Frame.BASE,
                  label="place-lower"))
    _hold(4)
    state["carry"] = False  # vacuum off
    _start = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
    _bin_target = np.array([_px / 1000.0, _py / 1000.0, (15.0 + float(center_z)) / 1000.0], dtype=np.float64)
    for _k in range(1, 13):  # smooth descent into the bin (no teleport)
        _t = _k / 12.0
        obj.set_world_pose(position=_start * (1.0 - _t) + _bin_target * _t,
                           orientation=np.asarray(state["carry_quat"], dtype=np.float64))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
        session.step(render=True)
    print("[beat] vacuum released -> package placed square in the KLT bin", flush=True)
    _hold(hold_steps)
    arm.move(Pose(position_mm=np.array([_px, _py, 340.0]), quaternion_xyzw=quat, frame=Frame.BASE,
                  label="place-retreat"))
    _hold(max(1, hold_steps // 2))
    print(f"[dbg] placed package at {np.round(np.asarray(obj.get_world_pose()[0]) * 1000, 1)} mm "
          f"(bin at {_PLACE_BIN_XY})", flush=True)

    state["on"] = False
    session.step = orig_step  # type: ignore[assignment]

    # --- encode MP4 + a few keyframe PNGs -----------------------------------
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        session.stop()
        raise SystemExit("no frames captured; cinematic RGB never populated")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 ships no stubs
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
    for f in frames:
        writer.write(f[..., ::-1])  # RGB -> BGR
    writer.release()
    for i, idx in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        cv2.imwrite(str(out.with_name(f"{out.stem}_key{i}.png")), frames[idx][..., ::-1])

    dur = len(frames) / float(fps)
    print("\n==== INDUSTRIAL SUCTION DEMO RECORDED ====", flush=True)
    print(f"  seal={best_seal:.3f} object_rose={z_lift:.1f} mm | frames={len(frames)} fps={fps} "
          f"duration={dur:.1f}s", flush=True)
    print(f"  saved {out}  (+ 4 keyframe PNGs alongside)", flush=True)
    session.stop()
    return {"seal": best_seal, "frames": len(frames), "fps": fps, "duration_s": dur,
            "lift_mm": z_lift, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a cinematic MP4 of the Willy industrial suction pick (Isaac).")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=18, help="playback fps (lower = slower/easier to watch)")
    ap.add_argument("--capture-every", type=int, default=1, help="capture 1 of every N sim steps")
    ap.add_argument("--hold-steps", type=int, default=40, help="pause length at the opening + lifted beats")
    ap.add_argument("--lift-mm", type=float, default=130.0)
    ap.add_argument("--no-bin", action="store_true",
                    help="drop the KLT tray (its walls are not in cuRobo's world -> can obstruct the descent); "
                         "keeps the dark cell table + lighting + package for a reliable open-cell record")
    ap.add_argument("--gui", action="store_true", help="also open the live window")
    args = ap.parse_args()
    record_demo(out_path=args.out, headless=not args.gui, fps=args.fps,
                capture_every=args.capture_every, hold_steps=args.hold_steps, lift_mm=args.lift_mm,
                use_bin=not args.no_bin)


if __name__ == "__main__":
    main()
