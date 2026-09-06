"""Industrial sorting-line pick-and-place demo (Isaac): a watchable cinematic MP4.

One robot, two totes in a sorting line: a source tote holding a small clutter of real scanned
objects, and an empty dest tote. The cell perceives the clutter, scores every object on the
analytical suction seal and on jaw feasibility, and picks the matching tool per object. The wide flat
package the jaw cannot span is sealed and lifted by the vacuum gripper; a small compact part is
grasped by the parallel jaw; each is placed into the dest tote. An object neither tool can take, such
as a domed mug, is rejected. The perception and the decision are shown as a "what the robot sees"
picture-in-picture (green = chosen, blue = jaw, red = rejected).

What keeps the filmed motion clean:
  * cuRobo plans every arm move, as a smooth collision-free joint trajectory. This runner requires
    the cuRobo env and exits if it is absent: it will not film a blind-IK run.
  * The tote walls are registered into cuRobo's collision world (``arm.set_curobo_world``), so the
    planner threads around them instead of planning straight through them.
  * Both totes sit on the robot centreline, inside the well-conditioned workspace, so no far-corner
    IK reconfigure is needed.
  * The place is a controlled kinematic set-down into the empty dest-tote centre, where no wall or
    object can eject the package.

Filmed in segments, each its own Isaac boot because SimulationApp is a singleton, then stitched by
``--concat`` with title cards, the way ``run_klt_combined_demo`` does it.

On-box only: needs Isaac plus the cuRobo/Coal engines. ``scripts/ext_deps/install_ext_deps.ps1`` installs both
into ``ext_deps/``, which is where the code looks by default, so no environment variable is needed.
Run with Isaac's bundled python from the repo root:

    ...\\python.bat -m src.willy_sim.run_sorting_demo --segment suction --out logs/demo/sort_1_suction.mp4
    ...\\python.bat -m src.willy_sim.run_sorting_demo --segment jaw     --out logs/demo/sort_2_jaw.mp4
    python -m src.willy_sim.run_sorting_demo --concat logs/demo/sort_1_suction.mp4,logs/demo/sort_2_jaw.mp4 --out logs/demo/sorting_demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

# --- cinematic framing (metres): an elevated front-right 3/4 view that sees both totes + the robot ---
CINE_POS_M = (2.55, -2.35, 1.85)
CINE_TARGET_M = (0.46, 0.0, 0.05)
CINE_RES = (1280, 720)

# --- cell heights (mm): the packing-table tabletop is aligned to the pick surface z=0 ---
_TABLE_TOP = 993.0
_FLOOR_Z = -_TABLE_TOP

# --- sorting line (mm, robot BASE frame): two totes on the robot centreline (Y=0), well-conditioned IK ---
SOURCE_XY = (585.0, 0.0)     # far tote: holds the clutter (the robot picks from here)
DEST_XY = (330.0, 0.0)       # near tote: empty (the robot places into here)
SRC_HALF_X, SRC_HALF_Y = 130.0, 225.0   # source tote inner half-extents (roomy: fits the wide box + spread clutter)
DST_HALF_X, DST_HALF_Y = 110.0, 150.0
TOTE_H = 45.0                # shallow tote walls: the cup and jaw reach in without grazing, and cuRobo knows them

# clutter object placements in the source tote. Spread wide (far apart) so each drops + settles independently
# and perceives as a distinct mask; the hero box + the mug are spawned in explicit stable orientations (below).
INK_XY = (585.0, 0.0)        # the flat-top package: suction target (jaw can't span its 138x131 mm face)
MUG_XY = (605.0, 155.0)      # a mug laid on its side (curved surface up): non-sealable and too wide for the jaw,
#                              so it is rejected. Its roll axis is world X, well clear of the box at Y=0.
TOY_XY = (560.0, -128.0)     # a small compact part (jaw target); kept off the extreme -Y reach, where the jaw
#                              grasp converges slowly or times out
DROP_Z = 60.0                # spawn height above the tote floor (the flat box finds its flat rest on the drop)

INK_LABEL = "ink cartridge box"
MUG_LABEL = "coffee mug"
TOY_LABEL = "toy rhino"
# The ink box mesh is 99(X) x 55(Y) x 145(Z) mm, so its thin axis is local y. At identity it stands on end
# and tips. A +90 deg rotation about X lays its largest face (99x145) down and the thin 55 mm dimension up,
# giving a stable flat rest with a wide flat top for the cup. wxyz = (cos45, sin45, 0, 0).
FLAT_INK_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)
# The mug mesh is tall along local Z (135 mm); the same +90-about-X lays it on its side, so the camera sees
# only its curved wall and no flat patch. The seal model then reads a curved surface, the seal is low, and
# the reject is reliable. Upright, the mug tips unpredictably and a bottom-up rest reads as sealable.
SIDE_MUG_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)

SEAL_THRESHOLD = 0.5         # flat-top seal vs domed/irregular (the repo convention)
JAW_MAX_FOOTPRINT_MM = 62.0  # a 2F-85 (85 mm span) grasps a min-footprint up to ~60 mm with margin


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Nearest proper rotation, by SVD.

    Absorbs the ~1e-6 determinant drift left by stacking near-orthonormal axes, so the strict det==+1
    check in ``from_rotation_matrix`` does not abort a pose build.
    """
    u, _s, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rot = u @ vt
    if float(np.linalg.det(rot)) < 0.0:
        u[:, -1] = -u[:, -1]
        rot = u @ vt
    return rot


def _move(arm: Any, pose: Any, label: str) -> Any:
    """Run one ``arm.move`` and log its typed status.

    ``arm.move`` classifies a workspace, IK or controller refusal into a ``MotionResult`` rather
    than raising, so a silent IK or preflight failure would otherwise be visible only as a
    mis-placed object.
    """
    r = arm.move(pose)
    status = getattr(getattr(r, "status", None), "name", None) or getattr(getattr(r, "status", None), "value", str(r))
    print(f"[move] {label:16s} -> {status}", flush=True)
    return r


def _source_specs() -> list:
    """The source-tote clutter as SimObjectConfigs (real scanned GSO meshes)."""
    from src.willy_sim.gso_assets import GSO_BY_LABEL, gso_object_spec

    return [
        gso_object_spec(GSO_BY_LABEL[INK_LABEL], position_mm=(INK_XY[0], INK_XY[1], 50.0),
                        orientation_wxyz=FLAT_INK_QUAT),  # spawn already flat + low so it settles flat + centred
        gso_object_spec(GSO_BY_LABEL[MUG_LABEL], position_mm=(MUG_XY[0], MUG_XY[1], 55.0),
                        orientation_wxyz=SIDE_MUG_QUAT),  # on its side: curved top, reliable reject
        gso_object_spec(GSO_BY_LABEL[TOY_LABEL], position_mm=(TOY_XY[0], TOY_XY[1], DROP_Z)),
    ]


def _tote_fixtures() -> "tuple[list, list]":
    """Return (source_walls, dest_walls) as uniquely-named FixtureBoxConfig lists for the two shallow totes."""
    from src.willy_sim.run_dense_pick import _make_bin_fixtures

    src = _make_bin_fixtures(center_xy_mm=SOURCE_XY, half_width_mm=SRC_HALF_X, half_width_y_mm=SRC_HALF_Y,
                             height_mm=TOTE_H, thickness_mm=10.0)
    dst = _make_bin_fixtures(center_xy_mm=DEST_XY, half_width_mm=DST_HALF_X, half_width_y_mm=DST_HALF_Y,
                             height_mm=TOTE_H, thickness_mm=10.0)
    # _make_bin_fixtures gives both totes the same wall names; rename them so cuRobo world entries stay unique.
    src = [w.model_copy(update={"name": f"src_{w.name}"}) for w in src]
    dst = [w.model_copy(update={"name": f"dst_{w.name}"}) for w in dst]
    return src, dst


def _register_curobo_world(arm: Any, fixtures: list) -> int:
    """Register every tote wall and a floor into cuRobo's collision world so it plans around them.

    Returns the number of obstacles registered; 0 means cuRobo is not active.
    """
    cuboids = [
        {
            "name": fx.name,
            "dims_m": [2.0 * float(fx.half_extents_mm[0]) / 1000.0,
                       2.0 * float(fx.half_extents_mm[1]) / 1000.0,
                       2.0 * float(fx.half_extents_mm[2]) / 1000.0],
            "pose": [float(fx.center_mm[0]) / 1000.0, float(fx.center_mm[1]) / 1000.0,
                     float(fx.center_mm[2]) / 1000.0, 1.0, 0.0, 0.0, 0.0],
        }
        for fx in fixtures
    ]
    cuboids.append({"name": "floor", "dims_m": [2.0, 2.0, 0.05], "pose": [0.0, 0.0, -0.026, 1.0, 0.0, 0.0, 0.0]})
    return int(arm.set_curobo_world(cuboids))


def _author_sorting_cell(stage: Any) -> None:
    """Author the warehouse pick cell from Isaac assets.

    A warehouse backdrop, the UR5e on a packing table whose tabletop is aligned to z=0, and a staging
    pallet of KLT bins. The two sorting totes are the fixture-wall prims the scene spawns from
    ``bin_walls``, so this hook adds only the backdrop. The primitive physics table is hidden and its
    collider stays.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]
    from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    from src.willy_sim.scene import TABLE_PRIM

    root = (get_assets_root_path() or "").rstrip("/")

    def _ref(rel_usd: str, prim: str, pos_mm: tuple[float, float, float], *, yaw_deg: float = 0.0) -> None:
        add_reference_to_stage(root + rel_usd, prim)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(pos_mm[0] / 1000.0, pos_mm[1] / 1000.0, pos_mm[2] / 1000.0))
        if yaw_deg:
            xf.AddRotateZOp().Set(float(yaw_deg))

    _tp = stage.GetPrimAtPath(TABLE_PRIM)
    if _tp and _tp.IsValid():
        UsdGeom.Imageable(_tp).MakeInvisible()

    _ref("/Isaac/Environments/Simple_Warehouse/warehouse.usd", "/World/Cell/warehouse", (0.0, 0.0, _FLOOR_Z))
    _ref("/Isaac/Props/PackingTable/packing_table.usd", "/World/Cell/packtable", (450.0, 0.0, _FLOOR_Z))
    # a staging pallet of KLT bins on the warehouse floor beside the bench (depth, not on the pick path)
    _pallet_top = _FLOOR_Z + 143.0
    _ref("/Isaac/Props/Pallet/pallet.usd", "/World/Cell/pallet_l", (-1500.0, -300.0, _FLOOR_Z))
    _ref("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", "/World/Cell/klt_l1", (-1500.0, -450.0, _pallet_top + 73.0))
    _ref("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", "/World/Cell/klt_l2", (-1500.0, 0.0, _pallet_top + 73.0))


# --------------------------------------------------------------------------------------------------------
# cinematic capture helpers
# --------------------------------------------------------------------------------------------------------
def _setup_cine(session: Any) -> "tuple[Any, Any, Any]":
    """Author the cinematic camera + return (camera, grab_fn, app) after a render warm-up (RGB only)."""
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
    from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

    cine = Camera(prim_path="/World/Cameras/Cinematic",
                  position=np.array(CINE_POS_M, dtype=np.float64), resolution=CINE_RES)
    cine.initialize()
    try:
        cine.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001 (clipping best-effort)
        pass
    set_camera_view(eye=list(CINE_POS_M), target=list(CINE_TARGET_M), camera_prim_path="/World/Cameras/Cinematic")
    app = getattr(session, "app", None)
    for _ in range(60):  # warm the render so the RGB annotator populates before capture
        session.step(render=True)
        if app is not None:
            app.update()

    def _grab() -> np.ndarray | None:
        fr = cine.get_current_frame()
        rgba = fr.get("rgb") if isinstance(fr, dict) else None
        return None if rgba is None else np.asarray(rgba)[..., :3].copy()

    return cine, _grab, app


def _score_scene(handles: Any, session: Any, sim: Any) -> "list[dict]":
    """Perceive every object from the overhead camera and decide the tool for each.

    For each perceived object, compute the analytical suction seal (``synthesize_suction_grasps``) and
    the back-projected footprint. A seal of 0.5 or more (``SEAL_THRESHOLD``) decides suction (green);
    otherwise a footprint within ``JAW_MAX_FOOTPRINT_MM`` decides jaw (blue); otherwise reject (red).
    Returns one dict per object: {label, mask, seal, footprint_mm, decision}.
    """
    from src.robot.grasping.geometry import masked_point_cloud
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import SuctionConfig, synthesize_suction_grasps

    from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource

    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    targets = list(handles.object_specs)  # [(prim_path, label), ...] in spec order
    src = MultiObjectGroundTruthPerceptionSource(
        camera=handles.camera, targets=targets, session=session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps))
    frame = src.acquire()
    segs = list(frame.segmentations)
    tries = 0
    while tries < 8 and not all(np.asarray(s.mask, dtype=bool).any() for s in segs):
        session.step_n(5)
        frame = src.acquire()
        segs = list(frame.segmentations)
        tries += 1
    depth, K, c2b = frame.depth_map, frame.intrinsics, handles.camera_to_base
    cfg = SuctionConfig(scorer=AnalyticalSuctionScorer(), max_results=8, min_quality=0.0)
    out: list[dict] = []
    # segs are built one-per-target in target order, so pair each with its known label (avoids relying on the
    # SegmentationLike protocol carrying a .label attribute).
    for (_prim, label), seg in zip(targets, segs):
        mask = np.asarray(seg.mask, dtype=bool)
        seal = 0.0
        footprint = float("inf")
        if mask.any():
            grasps = synthesize_suction_grasps(seg, depth, K, camera_to_base=c2b, payload_mass_g=200.0, config=cfg)
            seal = float(grasps[0].seal_score) if grasps else 0.0
            pts = np.asarray(masked_point_cloud(mask, depth, K, unit="mm").points_mm, dtype=np.float64)
            if pts.size:
                m = np.asarray(c2b.to_matrix(), dtype=np.float64)
                pb = (m[:3, :3] @ pts.T).T + m[:3, 3]
                footprint = float(min(np.ptp(pb[:, 0]), np.ptp(pb[:, 1])))  # min horizontal extent (jaw span)
        if seal >= SEAL_THRESHOLD:
            decision = "suction"
        elif footprint <= JAW_MAX_FOOTPRINT_MM:
            decision = "jaw"
        else:
            decision = "reject"
        out.append({"label": label, "mask": mask, "seal": seal, "footprint_mm": footprint, "decision": decision})
        print(f"[score] {label:18s} seal={seal:.3f} footprint={footprint:6.1f}mm -> {decision.upper()}", flush=True)
    return out


_DECISION_BGR = {"suction": (60, 220, 60), "jaw": (235, 170, 40), "reject": (60, 60, 235)}
_DECISION_TXT = {"suction": "SAUGEN", "jaw": "GREIFEN", "reject": "ABLEHNEN"}
_SHORT_NAME = {INK_LABEL: "Karton", MUG_LABEL: "Becher", TOY_LABEL: "Kleinteil"}
_PIP_W, _PIP_IMG_H, _PIP_ROW_H = 470, 282, 38  # PiP width / camera-image height / legend row height


def _make_decision_pip(frame_rgb: np.ndarray, scored: "list[dict]", chosen_label: str) -> np.ndarray:
    """Build the large, legible "what the robot sees" overlay (BGR).

    The overhead camera image with each object's mask outlined in its decision colour
    (green = suction, blue = jaw, red = reject), the chosen object outlined bolder, plus a legend strip
    below: one row per object with a colour swatch, the action word, a short name and the seal.
    """
    import cv2  # type: ignore[import-not-found]

    base = np.asarray(frame_rgb)[..., :3].astype(np.uint8)
    bgr = base[..., ::-1].copy()
    for s in scored:
        mask = np.asarray(s["mask"], dtype=np.uint8) * 255
        if not mask.any():
            continue
        col = _DECISION_BGR[s["decision"]]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cnts, -1, col, 8 if s["label"] == chosen_label else 4)
    img = cv2.resize(bgr, (_PIP_W, _PIP_IMG_H))
    cv2.rectangle(img, (0, 0), (_PIP_W, 36), (28, 28, 28), -1)
    cv2.putText(img, "PERZEPTION + AUSWAHL", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    rows = sorted(scored, key=lambda s: s["seal"], reverse=True)
    canvas = np.full((_PIP_IMG_H + _PIP_ROW_H * len(rows) + 14, _PIP_W, 3), 24, dtype=np.uint8)
    canvas[:_PIP_IMG_H] = img
    y = _PIP_IMG_H + 28
    for s in rows:
        col = _DECISION_BGR[s["decision"]]
        cv2.rectangle(canvas, (14, y - 20), (46, y + 4), col, -1)
        name = _SHORT_NAME.get(s["label"], str(s["label"]))
        cv2.putText(canvas, f"{_DECISION_TXT[s['decision']]}  {name}  (seal {s['seal']:.2f})",
                    (58, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, col, 2, cv2.LINE_AA)
        y += _PIP_ROW_H
    return canvas


def _write_mp4(frames: "list[np.ndarray]", out: str, fps: int, keyframes: int = 6) -> float:
    import cv2  # type: ignore[import-not-found]

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise SystemExit(f"no frames captured for {out}")
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))  # type: ignore[attr-defined]
    for f in frames:
        wr.write(np.ascontiguousarray(f[..., ::-1]))
    wr.release()
    for i, idx in enumerate(np.linspace(0, len(frames) - 1, keyframes).astype(int)):
        cv2.imwrite(str(p.with_name(f"{p.stem}_key{i}.png")), np.ascontiguousarray(frames[int(idx)][..., ::-1]))
    dur = len(frames) / float(fps)
    print(f"=== {out}: {len(frames)} frames @{fps}fps ({dur:.1f}s) (+{keyframes} PNGs) ===", flush=True)
    return dur


def _require_curobo() -> None:
    """Exit if the cuRobo env is absent: this demo must not film a blind-IK (flickering) run."""
    from src.robot.safety.planning import curobo_env_available

    if not curobo_env_available():
        raise SystemExit(
            "cuRobo env NOT FOUND -> the arm would degrade to BLIND IK (the flicker bug). Install it with "
            "scripts/ext_deps/install.ps1, or set WILLY_CUROBO_PYTHON if it lives outside ext_deps/. "
            "`python -m src.robot.safety.planning --doctor` says which it is.")


# --------------------------------------------------------------------------------------------------------
# Suction segment: the vacuum gripper seals + lifts the flat box out of the source tote and into the dest tote
# --------------------------------------------------------------------------------------------------------
def record_suction_segment(out: str, *, headless: bool = True, fps: int = 24, capture_every: int = 4,
                           hold: int = 14, lift_mm: float = 130.0) -> dict:
    """Film the suction segment: perceive the clutter, decide, then pick with the vacuum gripper.

    The ur10 vacuum gripper seals the flat box, lifts it out of the source tote, carries it as a
    render-safe kinematic carry, and sets it down square into the dest tote.
    """
    _require_curobo()
    from src.geometry import Frame, Pose
    from src.geometry.quaternion import to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes

    from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
    from src.willy_sim.scene import ARM_PRIM, WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import REAL_GRIPPER_TIP_MM, mount_real_suction_gripper

    src_walls, dst_walls = _tote_fixtures()
    from src.willy_sim.scene import SceneAppearance
    appearance = SceneAppearance(table_color=(0.16, 0.18, 0.22), dome_intensity=110.0, key_intensity=220.0)

    def _hook(stage: Any) -> None:
        mount_real_suction_gripper(stage, ARM_PRIM, hide_jaw=True)
        _author_sorting_cell(stage)

    cell = bootstrap_sim_cell(None, headless=headless, scene_kwargs={
        "objects_override": _source_specs(), "bin_walls": src_walls + dst_walls, "appearance": appearance,
    }, post_scene_hook=_hook)
    arm, handles, sim = cell.arm, cell.handles, cell.sim
    session = arm.session
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    park_q = np.asarray(getattr(sim, "park_joint_positions", None) or sim.home_joint_positions, dtype=np.float64)

    # Register both totes' walls and the floor into cuRobo's world so it plans around them.
    n_world = _register_curobo_world(arm, src_walls + dst_walls)
    print(f"[curobo-world] registered {n_world} obstacles (tote walls + floor)", flush=True)
    if n_world <= 0:
        session.stop()
        raise SystemExit("cuRobo world registration returned 0 -> the planner is not active; aborting (would flicker).")

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    obj = SingleRigidPrim(handles.object_specs[0][0])  # spec 0 == the ink box
    wrist = SingleRigidPrim(WRIST_LINK_PRIM)
    _cine, grab, app = _setup_cine(session)

    frames: list[np.ndarray] = []
    wrist_track: list[np.ndarray] = []
    state: dict[str, Any] = {"on": False, "n": 0, "carry": False, "drop_m": 0.0, "quat": None, "pip": None}
    orig_step = session.step

    def _carry() -> None:
        wp, wq = wrist.get_world_pose()
        r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
        cup_tip = np.asarray(wp, dtype=np.float64) + r_w @ np.array([0.0, REAL_GRIPPER_TIP_MM / 1000.0, 0.0])
        pos = cup_tip - np.array([0.0, 0.0, float(state["drop_m"])])
        obj.set_world_pose(position=pos, orientation=np.asarray(state["quat"], dtype=np.float64))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001, ANN202, ARG001
        orig_step(dt_s, render=True)
        if state["carry"] and state["quat"] is not None:
            _carry()
        if state["on"]:
            state["n"] += 1
            wrist_track.append(np.asarray(wrist.get_world_pose()[0], dtype=np.float64) * 1000.0)
            if state["n"] % capture_every == 0:
                if app is not None:
                    app.update()
                f = grab()
                if f is not None:
                    if state["pip"] is not None:
                        f = _composite_pip(f, state["pip"])
                    frames.append(f)

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(k: int) -> None:
        for _ in range(k):
            session.step(render=True)

    # Opening: arm parked aside so the overhead perceives the clutter cleanly; then perceive + decide.
    arm.move_to_joints(JointPositions(park_q))
    session.step_n(60)  # let all three (widely-spaced) objects drop + settle in the tote
    _bp = obj.get_world_pose()
    _bpos = np.round(np.asarray(_bp[0], dtype=np.float64) * 1000.0, 1)
    _bq = np.round(np.asarray(_bp[1], dtype=np.float64), 3)
    print(f"[dbg] ink box settled at {_bpos} mm quat(wxyz)={_bq}", flush=True)
    state["on"] = True
    _hold(hold)
    scored = _score_scene(handles, session, sim)
    ink = next((s for s in scored if s["label"] == INK_LABEL), None)
    if ink is None or ink["decision"] != "suction":
        state["on"] = False
        session.step = orig_step  # type: ignore[assignment]
        session.stop()
        raise SystemExit(f"suction: ink box not chosen as suction target (scored={[(s['label'], s['decision']) for s in scored]})")
    # freeze the perception overlay into the film for the rest of the segment
    _ov = grab()
    if _ov is not None:
        state["pip"] = _make_decision_pip(_ov, scored, INK_LABEL)
    _hold(hold)

    # Locate the box top from ground truth; the seal decision above is what chooses the tool.
    box_pose = obj.get_world_pose()
    box_xy = np.asarray(box_pose[0], dtype=np.float64)[:2] * 1000.0
    center_z = float(np.asarray(box_pose[0])[2]) * 1000.0
    top_z = 2.0 * center_z
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    pos = np.array([float(box_xy[0]), float(box_xy[1]), top_z], dtype=np.float64)
    seed = np.array([1.0, 0.0, 0.0]) if abs(approach[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    closing = seed - approach * float(np.dot(seed, approach))
    closing = closing / (float(np.linalg.norm(closing)) or 1.0)
    quat = _quaternion_from_axes(closing, approach)
    print(f"[suction] box top pos(base mm)={np.round(pos, 1)} center_z={center_z:.1f}", flush=True)

    # Approach: park, then home (ready), then standoff, then contact. Every arm.move is a smooth cuRobo
    # trajectory. The 29 mm contact gap cancels the cup's tip-below-TCP offset (cup tip ~161 mm against
    # the 2F-85 TCP ~132 mm), so the cup face rests on the box top.
    arm.move_to_joints(JointPositions(home_q))
    session.step_n(8)
    for label, gap in (("standoff", 95.0), ("contact", 29.0)):
        _move(arm, Pose(position_mm=pos - approach * gap, quaternion_xyzw=quat, frame=Frame.BASE, label=f"suction-{label}"), label)
    session.step_n(6)
    _wp, _wq = wrist.get_world_pose()
    _rw = to_rotation_matrix(np.array([_wq[1], _wq[2], _wq[3], _wq[0]], dtype=np.float64))
    _cup = np.asarray(_wp, dtype=np.float64) * 1000.0 + _rw @ np.array([0.0, REAL_GRIPPER_TIP_MM, 0.0])
    print(f"[dbg] contact: wrist={np.round(np.asarray(_wp) * 1000, 1)} cup_tip={np.round(_cup, 1)} box_top_z={top_z:.1f}", flush=True)

    # Engage: start the flat kinematic carry (the package hangs flat below the cup tip).
    state["drop_m"] = float(center_z) / 1000.0
    state["quat"] = np.asarray(obj.get_world_pose()[1], dtype=np.float64)
    state["carry"] = True
    _hold(6)
    print("[suction] vacuum engaged -> lifting", flush=True)

    # Lift straight up.
    _move(arm, Pose(position_mm=pos + (-approach) * float(lift_mm), quaternion_xyzw=quat, frame=Frame.BASE, label="suction-lift"), "lift")
    _hold(hold)

    # Transit high over the dest tote, then lower (cuRobo threads around the tote walls).
    dx, dy = DEST_XY
    _move(arm, Pose(position_mm=np.array([dx, dy, top_z + 240.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="place-transit"), "place-transit")
    _move(arm, Pose(position_mm=np.array([dx, dy, top_z + 150.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="place-lower"), "place-lower")
    _hold(4)

    # Place: carry off first, so the carry hook stops tracking the cup, then a controlled kinematic
    # set-down into the empty dest-tote centre, where no wall or object can eject it and nothing bounces.
    state["carry"] = False
    rest_z_m = float(center_z) / 1000.0  # box centre rests at ~half-height above the tote floor (z=0)
    start = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
    target = np.array([dx / 1000.0, dy / 1000.0, rest_z_m], dtype=np.float64)
    for k in range(1, 15):
        t = k / 14.0
        obj.set_world_pose(position=start * (1.0 - t) + target * t, orientation=np.asarray(state["quat"], dtype=np.float64))
        try:
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
        session.step(render=True)
    print(f"[suction] placed at {np.round(np.asarray(obj.get_world_pose()[0]) * 1000, 1)} mm (dest {DEST_XY})", flush=True)
    _hold(hold)

    # Retreat to reveal the placed package.
    _move(arm, Pose(position_mm=np.array([dx, dy, top_z + 260.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="place-retreat"), "place-retreat")
    _hold(hold)

    state["on"] = False
    session.step = orig_step  # type: ignore[assignment]
    _report_motion(wrist_track)
    dur = _write_mp4(frames, out, fps)
    session.stop()
    return {"frames": len(frames), "duration_s": dur, "out": out}


def _composite_pip(frame_rgb: np.ndarray, pip_bgr: np.ndarray) -> np.ndarray:
    """Composite the BGR perception PiP into the top-right of an RGB cinematic frame; return RGB."""
    import cv2  # type: ignore[import-not-found]

    fb = frame_rgb[..., ::-1].copy()  # RGB to BGR
    ph, pw = pip_bgr.shape[:2]
    x0, y0 = fb.shape[1] - pw - 20, 20
    fb[y0:y0 + ph, x0:x0 + pw] = pip_bgr
    cv2.rectangle(fb, (x0 - 2, y0 - 2), (x0 + pw + 2, y0 + ph + 2), (255, 255, 255), 2)
    return fb[..., ::-1]


def _report_motion(track: "list[np.ndarray]") -> None:
    """Report the per-step wrist translation as a flicker check.

    A flicker, or an IK branch flip, shows up as a huge single-step jump; a smooth cuRobo trajectory
    stays small and bounded.
    """
    if len(track) < 3:
        print("[motion] too few samples", flush=True)
        return
    arr = np.asarray(track, dtype=np.float64)
    d = np.diff(arr, axis=0)
    steps = np.linalg.norm(d, axis=1)
    dirs = d / (steps[:, None] + 1e-9)
    dots = np.sum(dirs[:-1] * dirs[1:], axis=1)  # cos angle between consecutive step directions
    # a flicker is the wrist reversing direction sharply between steps, at a non-trivial magnitude
    reversals = int(np.sum((dots < -0.3) & (steps[:-1] > 5.0) & (steps[1:] > 5.0)))
    print(f"[motion] wrist per-step mm: max={steps.max():.1f} mean={steps.mean():.2f} p95={np.percentile(steps, 95):.1f} "
          f"| sharp direction-reversals(>5mm)={reversals}  (0 == no flicker/oscillation)", flush=True)


# --------------------------------------------------------------------------------------------------------
# Jaw segment: the parallel jaw grasps the small part out of the source tote and into the dest tote
# --------------------------------------------------------------------------------------------------------
def record_jaw_segment(out: str, *, headless: bool = True, fps: int = 24, capture_every: int = 4,
                       hold: int = 10) -> dict:
    """Film the jaw segment: the 2F-85 takes the small compact part into the dest tote.

    Driven with explicit cuRobo moves plus gripper open and close, so every pose is well-conditioned
    and the motion does not flicker.
    """
    _require_curobo()
    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes

    from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
    from src.willy_sim.scene import SceneAppearance, WRIST_LINK_PRIM

    src_walls, dst_walls = _tote_fixtures()
    appearance = SceneAppearance(table_color=(0.16, 0.18, 0.22), dome_intensity=110.0, key_intensity=220.0)

    def _hook(stage: Any) -> None:
        _author_sorting_cell(stage)

    cell = bootstrap_sim_cell(None, headless=headless, scene_kwargs={
        "objects_override": _source_specs(), "bin_walls": src_walls + dst_walls, "appearance": appearance,
    }, post_scene_hook=_hook)
    arm, gripper, handles, sim = cell.arm, cell.gripper, cell.handles, cell.sim
    session = arm.session
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    park_q = np.asarray(getattr(sim, "park_joint_positions", None) or sim.home_joint_positions, dtype=np.float64)

    n_world = _register_curobo_world(arm, src_walls + dst_walls)
    print(f"[curobo-world] registered {n_world} obstacles (tote walls + floor)", flush=True)
    if n_world <= 0:
        session.stop()
        raise SystemExit("cuRobo world registration returned 0 -> the planner is not active; aborting.")

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    toy = SingleRigidPrim(handles.object_specs[2][0])  # spec 2 == the toy
    wrist = SingleRigidPrim(WRIST_LINK_PRIM)
    _cine, grab, app = _setup_cine(session)

    frames: list[np.ndarray] = []
    wrist_track: list[np.ndarray] = []
    state: dict[str, Any] = {"on": False, "n": 0, "pip": None, "carry": False, "p_rel": None, "r_rel": None}
    orig_step = session.step

    def _carry() -> None:
        # rigidly ride the part on the wrist at the pose captured on grasp (render-safe surrogate: the small
        # irregular part is a fragile physical grip; the jaw still actuates + the decision is real).
        wp, wq = wrist.get_world_pose()
        r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
        pos = np.asarray(wp, dtype=np.float64) + r_w @ state["p_rel"]
        q = from_rotation_matrix(_orthonormalize(r_w @ state["r_rel"]))
        toy.set_world_pose(position=pos, orientation=np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        try:
            toy.set_linear_velocity(np.zeros(3))
            toy.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001, ANN202, ARG001
        orig_step(dt_s, render=True)
        if state["carry"] and state["p_rel"] is not None:
            _carry()
        if state["on"]:
            state["n"] += 1
            wrist_track.append(np.asarray(wrist.get_world_pose()[0], dtype=np.float64) * 1000.0)
            if state["n"] % capture_every == 0:
                if app is not None:
                    app.update()
                f = grab()
                if f is not None:
                    if state["pip"] is not None:
                        f = _composite_pip(f, state["pip"])
                    frames.append(f)

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(k: int) -> None:
        for _ in range(k):
            session.step(render=True)

    gripper.open()
    arm.move_to_joints(JointPositions(park_q))
    session.step_n(30)
    state["on"] = True
    _hold(hold)
    scored = _score_scene(handles, session, sim)
    _ov = grab()
    if _ov is not None:
        state["pip"] = _make_decision_pip(_ov, scored, TOY_LABEL)
    _hold(hold)

    # Locate the toy (GT geometry); grasp it top-down.
    tp = toy.get_world_pose()
    txy = np.asarray(tp[0], dtype=np.float64)[:2] * 1000.0
    tcz = float(np.asarray(tp[0])[2]) * 1000.0
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    grasp = np.array([float(txy[0]), float(txy[1]), tcz], dtype=np.float64)  # grasp at the part centre
    seed = np.array([1.0, 0.0, 0.0])
    closing = seed - approach * float(np.dot(seed, approach))
    closing = closing / (float(np.linalg.norm(closing)) or 1.0)
    quat = _quaternion_from_axes(closing, approach)
    print(f"[jaw] toy at {np.round(grasp, 1)} mm", flush=True)

    arm.move_to_joints(JointPositions(home_q))
    session.step_n(8)
    for label, gap in (("standoff", 120.0), ("grasp", 12.0)):
        _move(arm, Pose(position_mm=grasp - approach * gap, quaternion_xyzw=quat, frame=Frame.BASE, label=f"jaw-{label}"), f"jaw-{label}")
    session.step_n(6)
    gripper.close()  # the jaw actuates + closes on the part (visible); the carry below rides it reliably
    session.step_n(10)

    # Engage the render-safe carry: capture the part's pose relative to the wrist so it rides rigidly
    # with the tool. The small irregular part is a fragile physical grip, so a kinematic carry
    # guarantees a clean place.
    wp, wq = wrist.get_world_pose()
    op, oq = toy.get_world_pose()
    r_w = to_rotation_matrix(np.array([wq[1], wq[2], wq[3], wq[0]], dtype=np.float64))
    r_o = to_rotation_matrix(np.array([oq[1], oq[2], oq[3], oq[0]], dtype=np.float64))
    state["p_rel"] = r_w.T @ (np.asarray(op, dtype=np.float64) - np.asarray(wp, dtype=np.float64))
    state["r_rel"] = r_w.T @ r_o
    state["carry"] = True
    _hold(6)

    # Lift + transit high over the dest tote + lower (the carry rides the part with the tool).
    _move(arm, Pose(position_mm=grasp + (-approach) * 130.0, quaternion_xyzw=quat, frame=Frame.BASE, label="jaw-lift"), "jaw-lift")
    _hold(hold)
    dx, dy = DEST_XY
    _move(arm, Pose(position_mm=np.array([dx, dy, tcz + 240.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="place-transit"), "place-transit")
    _move(arm, Pose(position_mm=np.array([dx, dy, tcz + 130.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="place-lower"), "place-lower")
    _hold(4)

    # Place: carry off, open the jaw, then a controlled kinematic set-down into the empty dest-tote centre.
    state["carry"] = False
    gripper.open()
    rest_q = np.asarray(toy.get_world_pose()[1], dtype=np.float64)
    start = np.asarray(toy.get_world_pose()[0], dtype=np.float64)
    target = np.array([dx / 1000.0, dy / 1000.0, float(tcz) / 1000.0], dtype=np.float64)
    for k in range(1, 13):
        t = k / 12.0
        toy.set_world_pose(position=start * (1.0 - t) + target * t, orientation=rest_q)
        try:
            toy.set_linear_velocity(np.zeros(3))
            toy.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
        session.step(render=True)
    _hold(hold)
    print(f"[jaw] placed toy at {np.round(np.asarray(toy.get_world_pose()[0]) * 1000, 1)} mm (dest {DEST_XY})", flush=True)
    _move(arm, Pose(position_mm=np.array([dx, dy, tcz + 260.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="place-retreat"), "place-retreat")
    _hold(hold)

    state["on"] = False
    session.step = orig_step  # type: ignore[assignment]
    _report_motion(wrist_track)
    dur = _write_mp4(frames, out, fps)
    session.stop()
    return {"frames": len(frames), "duration_s": dur, "out": out}


def concat(clips: "list[str]", out: str, fps: int = 24) -> int:
    """Stitch the segment clips into one MP4 with a title card before each (no Isaac needed)."""
    import cv2  # type: ignore[import-not-found]

    titles = ["Sortier-Zelle: Wahrnehmung + Auswahl", "1. Saug-Greifer: die breite flache Box",
              "2. Parallel-Greifer: das kleine Teil"]
    caps = [cv2.VideoCapture(c) for c in clips]
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))  # type: ignore[attr-defined]
    total = 0
    for ci, cap in enumerate(caps):
        card = np.zeros((h, w, 3), dtype=np.uint8)
        title = titles[ci + 1] if ci + 1 < len(titles) else f"Segment {ci + 1}"
        cv2.putText(card, title, (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        for _ in range(int(fps * 1.2)):
            wr.write(card)
            total += 1
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            wr.write(fr)
            total += 1
        cap.release()
    wr.release()
    print(f"=== COMBINED: {total} frames @{fps}fps ({total / fps:.1f}s) -> {out} ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Industrial sorting-line pick-and-place demo (suction + jaw).")
    ap.add_argument("--segment", choices=["suction", "jaw"], default=None)
    ap.add_argument("--concat", type=str, default=None, help="comma-separated clip paths to stitch")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    if args.concat:
        return concat([c.strip() for c in args.concat.split(",")], args.out, fps=args.fps)
    if args.segment == "suction":
        record_suction_segment(args.out, headless=not args.gui, fps=args.fps)
        return 0
    if args.segment == "jaw":
        record_jaw_segment(args.out, headless=not args.gui, fps=args.fps)
        return 0
    ap.error("pass --segment {suction,jaw} or --concat <clips>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
