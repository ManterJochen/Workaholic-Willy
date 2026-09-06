"""Autonomous bin-clearing demo (Isaac): one continuous cinematic MP4.

The robot empties a source KLT bin into a destination KLT bin, object by object, choosing a tool for
each and rejecting what it cannot grasp. It runs in one Isaac boot:

  * Scene, from Isaac assets: a warehouse backdrop, a conveyor feeder, the UR5e on a packing table,
    and two small-KLT bins on the robot centreline, the source holding a clutter of varied scanned
    objects and the dest empty. Real materials and warehouse light.
  * Perceive and decide, live, per object: the overhead camera segments every object, and the
    analytical suction seal plus the jaw-footprint model assign each a tool. Suction takes flat or
    wide objects the cup can seal, jaw takes small compact ones the 2F-85 can grip, and round, hollow
    or irregular ones are rejected. Each decision is shown as an AR tag floating on the object in the
    cinematic view (green = suction, blue = jaw, red = reject, with the score).
  * Clear loop: pick the best pickable object with its tool, swapping cup and jaw by visibility with
    no re-mount, carry it to the dest bin, set it down, re-perceive, and repeat until only rejects
    remain in the source bin.

The motion guarantees are those of ``run_sorting_demo``: cuRobo plans every arm move, all poses are
well-conditioned on the centreline, and the carry and set-down are render-safe kinematic surrogates.
The decision is what the demo is showing.

On-box only: needs Isaac plus the cuRobo/Coal engines. ``scripts/ext_deps/install_ext_deps.ps1`` installs both
into ``ext_deps/``, which is where the code looks by default, so no environment variable is needed.
Run with Isaac's bundled python from the repo root:

    ...\\python.bat -m src.willy_sim.run_bin_clearing_demo --out logs/demo/bin_clearing_demo.mp4
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from src.willy_sim.run_sorting_demo import (
    _DECISION_BGR,
    _DECISION_TXT,
    _move,
    _require_curobo,
    _setup_cine,
    _write_mp4,
)

# --- cinematic framing (metres): elevated front-right 3/4 view seeing both bins + the robot ---
CINE_POS_M = (2.55, -2.4, 1.9)
CINE_TARGET_M = (0.39, 0.0, 0.05)   # aimed between the source (0.515 m) and dest (0.235 m) bins, 15 mm toward the source
CINE_RES = (1280, 720)

_TABLE_TOP = 993.0
_FLOOR_Z = -_TABLE_TOP

# --- two real KLT bins on the robot centreline (mm, BASE frame) ---
# The source (pick) sits at x=515, inside the fixed overhead camera's field of view. A closer bin pushes
# the near objects out of frame, so they are not segmented and are mis-classified as rejects. The jaw
# reach at this distance is fine for a tall object such as the rhino.
# The dest (place) sits near and empty at x=235: only high transit and a kinematic set-down go there,
# never a low reach.
SOURCE_XY = (515.0, 0.0)     # source klt (pick): in the overhead camera FOV
DEST_XY = (235.0, 0.0)       # dest klt (place): near/empty
KLT_SCALE_XY = 1.4           # scale the real KLT footprint 1.4x: ~258 x 400 mm inner, which fits the roster plus spacing
# The walls are ~58 mm, so the bins read like real totes and contain a nudged object, in particular the
# rolling mug, instead of letting it slide toward the other bin. That is safe because the KLT walls are
# not registered in cuRobo, only a low floor is: the tool descends freely, and the rhino grasp at z~66
# stays above this rim.
KLT_ZSCALE = 0.40            # Z-squash the 146 mm KLT to ~58 mm walls
KLT_WALL_MM = 58.0           # fixture-wall height (matches the squashed KLT visual)
_KLT_HALF_X = 92.0 * KLT_SCALE_XY   # ~129 mm inner half-extent in X
_KLT_HALF_Y = 143.0 * KLT_SCALE_XY  # ~200 mm inner half-extent in Y

# Where picked objects land in the dest KLT (mm relative to DEST_XY). Suction objects, the big boxes, go
# in the centre column, because a big box in an edge slot overlaps a wall and is ejected upward; small
# jaw parts go to the edges.
_CENTER_SLOTS = [(0.0, -105.0), (0.0, 105.0), (0.0, 0.0)]
_EDGE_SLOTS = [(-90.0, -150.0), (90.0, -150.0), (-90.0, 150.0), (90.0, 150.0), (-90.0, 0.0), (90.0, 0.0)]

SEAL_THRESHOLD = 0.5
JAW_MAX_FOOTPRINT_MM = 62.0

# The clutter set. Each entry: (gso label, source-KLT slot (x,y) mm relative to SOURCE_XY, optional
# spawn orientation wxyz for a stable rest).
_FLAT_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)   # +90 about X: lay a box's largest face down / a mug on its side
# The bin-clearing clutter, spread in the source KLT. The runtime seal/footprint model decides the tool
# per object; positions are relative to SOURCE_XY, and each orientation is one the object settles in.
# The roster rules:
#  * A jaw object must settle tall, grasp-centre z >~ 60 mm, or the low top-down UR5e config
#    self-collides and cuRobo/IK refuse the pose. It must also sit off the centreline, |y| >~ 90 mm, to
#    dodge the shoulder singularity. A flat-settling toy has no low top-down jaw grasp here. The jaw
#    fingers spread +/-42 mm in X, so keep |x| <~ 80 mm.
#  * A suction object, a flat-top box, is approached high: the cup reaches down and the wrist stays up,
#    so it neither self-collides nor is sensitive to the singularity, and it can sit anywhere reachable.
# The reliable roster is therefore one jaw object, the rhino, kept off the centreline and off the walls;
# flat-top boxes for suction; and the round mug as the reject. Wide boxes stay at |x| <~ 60 mm so their
# ~138 mm face does not overhang the +/-129 mm inner wall, which PhysX answers by ejecting them. The jaw
# object is isolated in the +x/-y corner, >=150 mm from every other object, and is picked last: its low
# top-down grasp folds the arm into the bin, so firing it once the boxes are gone keeps it from knocking
# a neighbour into the dest bin. Suction boxes cluster in +y, where the high approach knocks nothing, and
# the mug reject sits in the opposite -x corner, clear of the rhino.
OBJECTS: list[dict] = [
    {"label": "ink cartridge box", "slot": (-60.0, 150.0), "quat": _FLAT_QUAT, "z": 55.0},  # suction (big flat box)
    {"label": "skincare box cube", "slot": (60.0, 145.0), "quat": _FLAT_QUAT, "z": 55.0},   # suction (compact box)
    {"label": "sponge pack", "slot": (0.0, 30.0), "quat": _FLAT_QUAT, "z": 55.0},           # suction (thin flat pack)
    {"label": "white mug", "slot": (-70.0, -100.0), "quat": _FLAT_QUAT, "z": 55.0},         # reject (round; -x side)
    {"label": "toy rhino", "slot": (70.0, -160.0), "quat": None, "z": 55.0},                # jaw (isolated -y corner)
]


def _objects() -> list:
    """SimObjectConfigs for the source-KLT clutter, at positions relative to SOURCE_XY."""
    from src.willy_sim.gso_assets import GSO_BY_LABEL, gso_object_spec

    specs = []
    for o in OBJECTS:
        x = SOURCE_XY[0] + o["slot"][0]
        y = SOURCE_XY[1] + o["slot"][1]
        specs.append(gso_object_spec(GSO_BY_LABEL[o["label"]], position_mm=(x, y, o["z"]),
                                     orientation_wxyz=o["quat"]))
    return specs


def _klt_fixtures() -> "tuple[list, list]":
    """Source and dest KLT-matching fixture walls, uniquely named.

    Physics only: both callers register a low floor into cuRobo and deliberately leave these walls
    out of its world.
    """
    from src.willy_sim.run_dense_pick import _make_bin_fixtures

    src = _make_bin_fixtures(center_xy_mm=SOURCE_XY, half_width_mm=_KLT_HALF_X, half_width_y_mm=_KLT_HALF_Y,
                             height_mm=KLT_WALL_MM, thickness_mm=8.0)
    dst = _make_bin_fixtures(center_xy_mm=DEST_XY, half_width_mm=_KLT_HALF_X, half_width_y_mm=_KLT_HALF_Y,
                             height_mm=KLT_WALL_MM, thickness_mm=8.0)
    src = [w.model_copy(update={"name": f"src_{w.name}"}) for w in src]
    dst = [w.model_copy(update={"name": f"dst_{w.name}"}) for w in dst]
    return src, dst


def _author_industrial_cell(stage: Any, n_walls: int) -> None:
    """Author the warehouse cell from Isaac assets.

    A warehouse backdrop, a conveyor feeder, a packing table, two real klt bins for source and dest,
    Z-squashed so both tools reach into them, and a staging pallet. Hides the grey fixture walls and
    the primitive table.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]
    from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

    from src.willy_sim.scene import BIN_WALL_PRIM, TABLE_PRIM

    root = (get_assets_root_path() or "").rstrip("/")

    def _ref(rel: str, prim: str, pos_mm: tuple[float, float, float], *, yaw: float = 0.0,
             scale: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        add_reference_to_stage(root + rel, prim)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(pos_mm[0] / 1000.0, pos_mm[1] / 1000.0, pos_mm[2] / 1000.0))
        if yaw:
            xf.AddRotateZOp().Set(float(yaw))
        if scale != (1.0, 1.0, 1.0):
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))

    # hide the primitive physics table (keep its collider)
    _tp = stage.GetPrimAtPath(TABLE_PRIM)
    if _tp and _tp.IsValid():
        UsdGeom.Imageable(_tp).MakeInvisible()
    # warehouse + packing table
    _ref("/Isaac/Environments/Simple_Warehouse/warehouse.usd", "/World/Cell/warehouse", (0.0, 0.0, _FLOOR_Z))
    _ref("/Isaac/Props/PackingTable/packing_table.usd", "/World/Cell/packtable", (450.0, 0.0, _FLOOR_Z))
    # conveyor feeder behind the cell (industrial backdrop; ~2 m belt, not interacted with)
    _ref("/Isaac/Props/Conveyors/ConveyorBelt_A05.usd", "/World/Cell/conveyor", (900.0, 1250.0, _FLOOR_Z), yaw=90.0)
    # two real klt bins, Z-squashed with the floor at z=0: klt_z = 73 * KLT_ZSCALE places the squashed floor there
    _klt_z = 73.0 * KLT_ZSCALE
    _ref("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", "/World/Cell/klt_src",
         (SOURCE_XY[0], SOURCE_XY[1], _klt_z), scale=(KLT_SCALE_XY, KLT_SCALE_XY, KLT_ZSCALE))
    _ref("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", "/World/Cell/klt_dst",
         (DEST_XY[0], DEST_XY[1], _klt_z), scale=(KLT_SCALE_XY, KLT_SCALE_XY, KLT_ZSCALE))
    # a staging pallet of bins on the warehouse floor (depth)
    _ref("/Isaac/Props/Pallet/pallet.usd", "/World/Cell/pallet_l", (-1500.0, -300.0, _FLOOR_Z))
    _ref("/Isaac/Props/KLT_Bin/small_KLT_visual.usd", "/World/Cell/klt_l1", (-1500.0, -450.0, _FLOOR_Z + 143.0 + 73.0))
    # hide the grey fixture wall prims; their colliders stay, so physics is unchanged and the real KLTs are the look
    for i in range(n_walls):
        wp = stage.GetPrimAtPath(f"{BIN_WALL_PRIM}_{i}")
        if wp and wp.IsValid():
            UsdGeom.Imageable(wp).MakeInvisible()


# ---------------------------------------------------------------------------------------------------------
# Tool swap: visibility only. Physics, IK and cuRobo are unaffected and the 2F-85 stays in the articulation.
# ---------------------------------------------------------------------------------------------------------
_JAW_KEYS = ("knuckle", "finger", "fingertip", "tip", "pad")
ARM_PRIM = "/World/UR5e"
CUP_ROOT = f"{ARM_PRIM}/wrist_3_link/suction_gripper"


def _jaw_imageables(stage: Any):  # noqa: ANN201
    """Yield only the 2F-85 moving-jaw meshes.

    The cup's own mesh is named ``gripper_tip``, which contains "tip", so the cup subtree has to be
    excluded. Otherwise the cup's tip is toggled with the jaw: the cup shows in jaw mode and its vacuum
    tip is hidden in suction mode.
    """
    from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

    for prim in Usd.PrimRange(stage.GetPrimAtPath(ARM_PRIM)):
        if "suction_gripper" in str(prim.GetPath()):  # the mounted cup subtree is never a jaw mesh
            continue
        name = prim.GetName().lower()
        if "base" in name:
            continue
        if any(k in name for k in _JAW_KEYS):
            img = UsdGeom.Imageable(prim)
            if img:
                yield img


def show_suction(stage: Any) -> None:
    """Cup visible, 2F-85 jaw hidden."""
    from pxr import UsdGeom  # type: ignore[import-not-found]

    UsdGeom.Imageable(stage.GetPrimAtPath(CUP_ROOT)).MakeVisible()
    for img in _jaw_imageables(stage):
        img.MakeInvisible()


def show_jaw(stage: Any) -> None:
    """2F-85 jaw visible, cup hidden."""
    from pxr import UsdGeom  # type: ignore[import-not-found]

    UsdGeom.Imageable(stage.GetPrimAtPath(CUP_ROOT)).MakeInvisible()
    for img in _jaw_imageables(stage):
        img.MakeVisible()


def _move_log(arm: Any, pose: Any, label: str) -> Any:
    """cuRobo move + log the typed status and message (so a non-EXECUTED grasp is diagnosable per object)."""
    r = arm.move(pose)
    st = getattr(getattr(r, "status", None), "name", None) or str(r)
    print(f"[move] {label:16s} -> {st}  ({getattr(r, 'message', '')})", flush=True)
    return r


# ---------------------------------------------------------------------------------------------------------
# Perceive + decide every object (returns per-object dict + the frame, so the caller can back-project centroids)
# ---------------------------------------------------------------------------------------------------------
def _score_all(perception: Any, handles: Any, session: Any) -> "tuple[list[dict], Any]":
    from src.robot.grasping.geometry import masked_point_cloud
    from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer
    from src.robot.grasping.suction.synthesis import SuctionConfig, synthesize_suction_grasps

    targets = list(handles.object_specs)
    frame = perception.acquire()
    segs = list(frame.segmentations)
    tries = 0
    while tries < 6 and not any(np.asarray(s.mask, dtype=bool).any() for s in segs):
        session.step_n(5)
        frame = perception.acquire()
        segs = list(frame.segmentations)
        tries += 1
    depth, K, c2b = frame.depth_map, frame.intrinsics, handles.camera_to_base
    cfg = SuctionConfig(scorer=AnalyticalSuctionScorer(), max_results=8, min_quality=0.0)
    m = np.asarray(c2b.to_matrix(), dtype=np.float64)
    out: list[dict] = []
    for (_prim, label), seg in zip(targets, segs):
        mask = np.asarray(seg.mask, dtype=bool)
        seal, footprint, centroid = 0.0, float("inf"), None
        mask_px, depth_px = int(mask.sum()), 0
        if mask.any():
            grasps = synthesize_suction_grasps(seg, depth, K, camera_to_base=c2b, payload_mass_g=200.0, config=cfg)
            seal = float(grasps[0].seal_score) if grasps else 0.0
            pts = np.asarray(masked_point_cloud(mask, depth, K, unit="mm").points_mm, dtype=np.float64)
            depth_px = int(pts.shape[0]) if pts.size else 0
            if pts.size:
                pb = (m[:3, :3] @ pts.T).T + m[:3, 3]
                footprint = float(min(np.ptp(pb[:, 0]), np.ptp(pb[:, 1])))
                centroid = pb.mean(axis=0)
        decision = "suction" if seal >= SEAL_THRESHOLD else ("jaw" if footprint <= JAW_MAX_FOOTPRINT_MM else "reject")
        # Carried so a reject can say why. `seal=0.000 foot=inf` is both fields at their initial values,
        # and two different faults produce it: an empty mask, or a mask with no valid depth under it.
        # `mask_px` and `depth_px` tell those two apart.
        out.append({"label": label, "seal": seal, "footprint_mm": footprint, "decision": decision,
                    "centroid_mm": centroid, "mask_px": mask_px, "depth_px": depth_px})
    return out, frame


def _select_by_prompt(detector: Any, frame: Any, prompt: str) -> "tuple[set[int], list]":
    """Select objects by language, running GroundingDINO on the overhead RGB.

    Returns the set of ground-truth object indices whose segmentation-mask centroid falls inside a
    detected box. Mask and box share the one 640x480 overhead pixel space, and the indices are aligned
    to ``handles.object_specs``. Fails open, selecting all, when the RGB is missing; fails closed,
    selecting none this frame, on a model error, and the loop re-perceives on the next tick.
    """
    rgb = getattr(frame, "rgb", None)
    if rgb is None:
        return set(range(len(frame.segmentations))), []
    bgr = np.ascontiguousarray(np.asarray(rgb)[..., ::-1])  # RGB to BGR for the detector
    try:
        dets = list(detector.detect_all(bgr, prompt))
    except Exception:  # noqa: BLE001
        return set(), []
    boxes = [tuple(float(c) for c in d.box) for d in dets]
    selected: set[int] = set()
    for i, seg in enumerate(frame.segmentations):
        m = np.asarray(seg.mask, dtype=bool)
        if not m.any():
            continue
        ys, xs = np.where(m)
        cx, cy = float(xs.mean()), float(ys.mean())
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for (x0, y0, x1, y1) in boxes):
            selected.add(i)
    return selected, dets


def _obj_in_source(objs: list, i: int) -> bool:
    """Is object ``i`` still in the source bin?

    Reads the object's true world pose. Perception drives the tool decision; which bin an object sits
    in is ground-truth bookkeeping.
    """
    p = np.asarray(objs[i].get_world_pose()[0], dtype=np.float64) * 1000.0
    return abs(p[0] - SOURCE_XY[0]) <= _KLT_HALF_X + 25.0 and abs(p[1] - SOURCE_XY[1]) <= _KLT_HALF_Y + 25.0


# ---------------------------------------------------------------------------------------------------------
# AR overlay: floating decision tags projected ON each object in the cinematic view
# ---------------------------------------------------------------------------------------------------------
_DIM_BGR = (135, 135, 135)   # a grey tag for an object the prompt did not request (robot leaves it)


def _draw_prompt_banner(bgr: np.ndarray, prompt: str) -> None:
    """Top-centre banner showing the natural-language request driving the pick."""
    import cv2  # type: ignore[import-not-found]

    txt = f'ANFRAGE:  "{prompt}"'
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
    x0 = max(10, (CINE_RES[0] - tw) // 2 - 22)
    cv2.rectangle(bgr, (x0, 16), (x0 + tw + 44, 16 + th + 26), (28, 28, 28), -1)
    cv2.rectangle(bgr, (x0, 16), (x0 + tw + 44, 16 + th + 26), (70, 210, 70), 2)
    cv2.putText(bgr, txt, (x0 + 22, 16 + th + 9), font, scale, (245, 245, 245), thick, cv2.LINE_AA)


def _draw_ar_tags(frame_rgb: np.ndarray, tags: "list[dict]", objs: list, cine_camera: Any,
                  prompt: "str | None" = None) -> np.ndarray:
    """Draw a floating decision tag (colour, action, score) above each live object.

    Projects each object's current world position to the cinematic pixel, so the tag tracks the object.
    With a prompt, a top banner shows the request and objects the prompt did not match are dimmed grey.
    """
    import cv2  # type: ignore[import-not-found]

    bgr = frame_rgb[..., ::-1].copy()
    if prompt:
        _draw_prompt_banner(bgr, prompt)
    if not tags:
        return bgr[..., ::-1]
    pts_m, meta = [], []
    for t in tags:
        p = np.asarray(objs[t["idx"]].get_world_pose()[0], dtype=np.float64)  # metres, world==base
        pts_m.append([p[0], p[1], p[2] + 0.11])  # tag floats ~110 mm above the object
        meta.append(t)
    try:
        uv = np.asarray(cine_camera.get_image_coords_from_world_points(np.asarray(pts_m, dtype=np.float64)),
                        dtype=np.float64)
    except Exception:  # noqa: BLE001 (projection unavailable -> keep the banner, no tags this frame)
        return bgr[..., ::-1]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2
    # in-frame tags, drawn top-to-bottom; overlapping boxes cascade downward so all stay legible.
    items = [(float(u), float(v), t) for (u, v), t in zip(uv, meta)
             if 0.0 <= u < CINE_RES[0] and 0.0 <= v < CINE_RES[1]]
    items.sort(key=lambda a: a[1])
    placed: list[tuple[int, int, int, int]] = []
    for u, v, t in items:
        decision = t["decision"]
        # With a prompt, a pickable object the prompt did not match is dimmed grey and tagged "(n.a.)",
        # since the robot leaves it. A matched object, or any object when there is no prompt, keeps its
        # bright decision colour. Rejects always stay red.
        matched = decision == "reject" or t.get("selected", True)
        col = _DECISION_BGR[decision] if matched else _DIM_BGR
        txt = f"{_DECISION_TXT[decision]} {t['seal']:.2f}" if matched else f"{_DECISION_TXT[decision]} (n.a.)"
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
        bx, by = int(u - tw / 2), int(v)
        for _ in range(len(placed)):  # push down until it clears every already-placed box
            hit = next((b for b in placed if bx - 9 < b[2] and bx + tw + 9 > b[0]
                        and by - th - 12 < b[3] and by + 9 > b[1]), None)
            if hit is None:
                break
            by = hit[3] + th + 14
        box = (bx - 9, by - th - 12, bx + tw + 9, by + 9)
        placed.append(box)
        cur = t.get("current", False)
        cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]), (25, 25, 25), -1)
        cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]), col, 3 if cur else 2)
        cv2.line(bgr, (int(u), box[3]), (int(u), int(v) + 22), col, 2)  # leader line down to the object
        cv2.putText(bgr, txt, (bx, by), font, scale, col, thick, cv2.LINE_AA)
    return bgr[..., ::-1]


# ---------------------------------------------------------------------------------------------------------
# the autonomous clearing loop
# ---------------------------------------------------------------------------------------------------------
def record_clearing(out: str, *, headless: bool = True, fps: int = 24, capture_every: int = 5,
                    hold: int = 10, max_picks: int = 12, prompt: str | None = None) -> dict:
    _require_curobo()
    if prompt:
        # A text prompt steers which objects are cleared, via GroundingDINO on the overhead RGB. Force
        # HF offline before Isaac boots, or Kit's httpx closes and the model load crashes; the
        # grounding-dino-tiny weights must already sit in the local HF cache.
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Run the detector on CPU. Loading GroundingDINO onto CUDA next to Kit's live renderer starves
        # the rendered-depth annotator: camera.get_depth() returns garbage, every object then scores
        # seal=0 and footprint=inf, everything is rejected and nothing is picked. Only the detector
        # consults get_device(); the suction seal is numpy-analytical and Isaac, cuRobo and Coal keep
        # the GPU, so CPU inference on one overhead frame per pick keeps the two decoupled. setdefault
        # leaves an explicit WILLY_DEVICE winning.
        os.environ.setdefault("WILLY_DEVICE", "cpu")
    from src.geometry import Frame, Pose
    from src.geometry.quaternion import to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes

    from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
    from src.willy_sim.scene import SceneAppearance, WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import mount_real_suction_gripper

    src_walls, dst_walls = _klt_fixtures()
    n_walls = len(src_walls) + len(dst_walls)
    appearance = SceneAppearance(table_color=(0.16, 0.18, 0.22), dome_intensity=120.0, key_intensity=240.0)

    def _hook(stage: Any) -> None:
        mount_real_suction_gripper(stage, ARM_PRIM, hide_jaw=True)  # cup exists (physics-neutralized); jaw hidden
        _author_industrial_cell(stage, n_walls)

    cell = bootstrap_sim_cell(None, headless=headless, scene_kwargs={
        "objects_override": _objects(), "bin_walls": src_walls + dst_walls, "appearance": appearance,
    }, post_scene_hook=_hook)
    arm, gripper, handles, sim, cfg = cell.arm, cell.gripper, cell.handles, cell.sim, cell.cfg
    session = arm.session
    import omni.usd  # type: ignore[import-not-found]  # available now that bootstrap booted Isaac
    stage = omni.usd.get_context().get_stage()
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    park_q = np.asarray(getattr(sim, "park_joint_positions", None) or sim.home_joint_positions, dtype=np.float64)

    # cuRobo's world is a single low floor with its top at z=-50 mm: no KLT walls, no table-surface
    # floor. cuRobo stays on and still owns every move, but the floor sits well below the bin, so it
    # never inflates into the jaw's near-floor finger descent. A floor at the table surface (z~=0), or
    # the bin walls, makes cuRobo refuse the low top-down reach: the plan times out, the arm never
    # descends, and the carried object reads as grasped in mid-air. Every arm move is either high
    # (standoff, transit, lift, retreat) or a vertical descent directly above a wall-spaced object, so
    # the walls buy nothing, while the low floor still keeps the arm from driving under the table. This
    # is the arrangement ``run_m2_pick`` uses to reach low table objects: no obstacle intrudes into the
    # grasp column.
    _floor = {"name": "table_floor", "dims_m": [3.0, 3.0, 1.0],
              "pose": [0.0, 0.0, -0.55, 1.0, 0.0, 0.0, 0.0]}  # 1 m-thick slab, top at z=-0.05 m
    n_world = int(arm.set_curobo_world([_floor]))
    print(f"[curobo-world] registered {n_world} obstacle(s) (low floor top=-50mm; walls omitted so the jaw "
          f"descends clean)", flush=True)
    if n_world <= 0:
        session.stop()
        raise SystemExit("cuRobo world registration returned 0 -> aborting (would flicker).")

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    objs = [SingleRigidPrim(p) for (p, _l) in handles.object_specs]
    wrist = SingleRigidPrim(WRIST_LINK_PRIM)
    _cine, grab, app = _setup_cine(session)

    # Per-object local bbox corners (metres, body frame), read once from the authored mesh extent.
    # Combined with the live physics pose they give each object's true world top. get_world_pose()
    # returns the mesh origin, not the centre: the GSO converter drops each mesh's bbox-min-Z to the
    # origin and centres xy. So "top = 2 * origin_z" is wrong for an origin-at-bottom mesh, putting the
    # jaw grasp below the object body and below the 20 mm workspace floor, which comes back as
    # WORKSPACE_REJECTED or as a grasp in mid-air. The real top drives every grasp, standoff and lift
    # height instead.
    from pxr import UsdGeom  # type: ignore[import-not-found]
    _corners: list[np.ndarray] = []
    for (p, _l) in handles.object_specs:
        ext = UsdGeom.Mesh(stage.GetPrimAtPath(p + "/mesh")).GetExtentAttr().Get()
        if ext is not None:
            lo = np.asarray(ext[0], dtype=np.float64)
            hi = np.asarray(ext[1], dtype=np.float64)
        else:  # authored on every converted GSO mesh; fall back to a ~60x60x50 mm box just in case
            lo, hi = np.array([-0.03, -0.03, 0.0]), np.array([0.03, 0.03, 0.05])
        _corners.append(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                   for z in (lo[2], hi[2])], dtype=np.float64))

    def _world_top_mm(idx: int) -> float:
        """The object's world-space top (mm), from its live pose and its static geometry (fabric-safe)."""
        pos_m, q = objs[idx].get_world_pose()
        r = to_rotation_matrix(np.array([q[1], q[2], q[3], q[0]], dtype=np.float64))
        cw = (r @ _corners[idx].T).T + np.asarray(pos_m, dtype=np.float64)
        return float(cw[:, 2].max()) * 1000.0

    # build the overhead perception once (re-acquired each loop iteration; re-constructing per iteration would
    # re-register the segmentation/depth annotators).
    from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource
    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    perception = MultiObjectGroundTruthPerceptionSource(
        camera=handles.camera, targets=list(handles.object_specs), session=session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps))

    # Language-driven selection: build GroundingDINO, the grounding-dino-tiny the sim profile configures
    # with fp32 weights, so a text prompt can pick which objects get cleared. With prompt=None the demo
    # clears everything pickable.
    detector = None
    if prompt:
        from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector
        detector = GroundingDinoObjectDetector(cfg.models.objectdetector)
        print(f"[lang] GroundingDINO ready; prompt={prompt!r} -> only matching objects are cleared", flush=True)

    frames: list[np.ndarray] = []
    wrist_track: list[np.ndarray] = []
    # state: the rigid carry, where the object holds the exact pose relative to the wrist captured at the
    # grasp and so follows the wrist without snapping, plus the live AR tags. "rel" is object-origin minus
    # wrist-origin in world at the grasp; a world offset holds only because the wrist orientation stays
    # ~constant top-down across lift and transit. "oquat" is the object's held orientation (wxyz).
    state: dict[str, Any] = {"on": False, "n": 0, "carry": False, "idx": -1, "rel": None,
                             "oquat": None, "tags": [], "prompt": prompt, "dets": []}
    orig_step = session.step

    def _carry() -> None:
        w_pos = np.asarray(wrist.get_world_pose()[0], dtype=np.float64)
        o = objs[state["idx"]]
        o.set_world_pose(position=w_pos + state["rel"], orientation=state["oquat"])
        try:
            o.set_linear_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001, ANN202, ARG001
        orig_step(dt_s, render=True)
        if state["carry"] and state["rel"] is not None:
            _carry()
        if state["on"]:
            state["n"] += 1
            wrist_track.append(np.asarray(wrist.get_world_pose()[0], dtype=np.float64) * 1000.0)
            if state["n"] % capture_every == 0:
                if app is not None:
                    app.update()
                f = grab()
                if f is not None:
                    frames.append(_draw_ar_tags(f, state["tags"], objs, _cine, prompt=state["prompt"]))

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(k: int) -> None:
        for _ in range(k):
            session.step(render=True)

    def _topdown_quat() -> Any:
        # Close the jaw along +X, the top-down branch that works for this UR5e at the centreline. A
        # +Y-closing orientation self-collides here and cuRobo's Coal check rejects the folded wrist, so
        # +X stays and the jaw objects are instead kept off the +/-X walls (see OBJECTS), which holds the
        # open fingers inside the bin footprint.
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        closing = np.array([1.0, 0.0, 0.0])
        closing = closing / (float(np.linalg.norm(closing)) or 1.0)
        return _quaternion_from_axes(closing, approach), approach

    # settle + record each object's stable home pose (reused for the dest set-down)
    arm.move_to_joints(JointPositions(park_q))
    session.step_n(60)
    homes = [o.get_world_pose() for o in objs]
    for i, (_p, _l) in enumerate(handles.object_specs):
        print(f"[dbg] obj {i} {_l:18s} settled at {np.round(np.asarray(homes[i][0]) * 1000, 1)} mm", flush=True)
    state["on"] = True
    _hold(hold)

    quat, approach = _topdown_quat()
    placed = n_center = n_edge = 0
    _skip: set[int] = set()   # jaw objects whose grasp was unreachable: parked, not re-picked, not air-grabbed
    # Language selection is locked on the first, full-scene frame and then reused. The request is a fixed
    # intent, "clear exactly these", and must not drift as objects leave: re-running GroundingDINO each
    # iteration on the emptying bin lets "box" ground to the KLT tote itself, so a left-behind object
    # whose centroid falls in that box would flip to selected and be wrongly picked. Tool scoring still
    # re-runs per frame.
    locked_selected: set[int] | None = None
    for _pick in range(max_picks):
        arm.move_to_joints(JointPositions(park_q))
        session.step_n(20)
        scored, _frame = _score_all(perception, handles, session)
        # With no prompt, everything pickable is a candidate: the plain autonomous-clearing behaviour.
        if detector is not None and prompt:
            if locked_selected is None:
                locked_selected, state["dets"] = _select_by_prompt(detector, _frame, prompt)
            selected = locked_selected
        else:
            selected = set(range(len(scored)))
        for i, s in enumerate(scored):
            _pp = np.asarray(objs[i].get_world_pose()[0], dtype=np.float64) * 1000.0
            print(f"[score] {s['label']:16s} seal={s['seal']:.3f} foot={s['footprint_mm']:6.0f} "
                  f"mask={s.get('mask_px', -1):6d}px depth={s.get('depth_px', -1):6d}px "
                  f"dec={s['decision']:7s} sel={i in selected} in_src={_obj_in_source(objs, i)} "
                  f"pose={np.round(_pp, 0)}", flush=True)
        # live AR tags for every object still in the source bin: selected, meaning the prompt matched it,
        # is bright, and not selected is dimmed "not requested".
        state["tags"] = [
            {"idx": i, "decision": s["decision"], "seal": s["seal"], "selected": i in selected}
            for i, s in enumerate(scored) if _obj_in_source(objs, i)
        ]
        _hold(hold)
        # candidates are the pickable, prompt-selected objects still in the source bin. A jaw object whose
        # grasp failed is parked in _skip, so the loop moves on instead of re-picking and air-grabbing.
        cands = [(i, s) for i, s in enumerate(scored)
                 if s["decision"] != "reject" and i not in _skip and _obj_in_source(objs, i) and i in selected]
        if not cands:
            print("[loop] only rejects/unselected/empty remain -> requested objects cleared", flush=True)
            break
        # Pick order: suction boxes first, since a high cup approach disturbs no neighbour, and the jaw
        # object last. The jaw's low top-down grasp folds the arm into the bin and can shove a neighbour
        # toward the dest bin, so firing it once only it and the reject remain keeps the bin clean.
        i, s = min(cands, key=lambda t: (t[1]["decision"] == "jaw", t[1]["footprint_mm"]))
        for tg in state["tags"]:
            tg["current"] = (tg["idx"] == i)
        print(f"[loop] pick {_pick}: {s['label']} -> {s['decision'].upper()} (seal {s['seal']:.2f})", flush=True)

        op = objs[i].get_world_pose()
        oxy = np.asarray(op[0], dtype=np.float64)[:2] * 1000.0
        ocz = float(np.asarray(op[0])[2]) * 1000.0     # mesh-origin z (the grasp reference run_sorting_demo uses)
        otop = _world_top_mm(i)                        # world top (mm), from the live pose and geometry
        pos = np.array([float(oxy[0]), float(oxy[1]), otop], dtype=np.float64)
        if s["decision"] == "suction":  # big boxes go to the centre column, small jaw parts to the edges
            _dslot = _CENTER_SLOTS[n_center % len(_CENTER_SLOTS)]
            n_center += 1
        else:
            _dslot = _EDGE_SLOTS[n_edge % len(_EDGE_SLOTS)]
            n_edge += 1
        dx = DEST_XY[0] + _dslot[0]
        dy = DEST_XY[1] + _dslot[1]

        arm.move_to_joints(JointPositions(home_q))
        session.step_n(6)
        if s["decision"] == "suction":
            show_suction(stage)
            gap = 29.0
            _move(arm, Pose(position_mm=pos - approach * 90.0, quaternion_xyzw=quat, frame=Frame.BASE, label="standoff"), "standoff")
            _move(arm, Pose(position_mm=pos - approach * gap, quaternion_xyzw=quat, frame=Frame.BASE, label="contact"), "contact")
        else:
            show_jaw(stage)
            gripper.open()
            # Grasp at the object origin + 12 mm, the jaw reference run_sorting_demo uses: cuRobo reaches
            # it for a tall object whose origin sits up in the body. The 40 mm floor keeps the
            # grasp-centre TCP clear of the 20 mm workspace floor and inside cuRobo's comfort zone. A flat
            # object, whose origin sits near the bottom, has no low top-down jaw grasp here and is kept
            # out of the roster.
            grasp_z = max(ocz + 12.0, 40.0)
            _move(arm, Pose(position_mm=pos - approach * 110.0, quaternion_xyzw=quat, frame=Frame.BASE, label="standoff"), "standoff")
            _move_log(arm, Pose(position_mm=np.array([float(oxy[0]), float(oxy[1]), grasp_z]), quaternion_xyzw=quat, frame=Frame.BASE, label="grasp"), "grasp")
            # Gate on the actual reach, not the strict EXECUTED status: cuRobo often ends a hair over its
            # 5 mm tolerance and reports a timeout for a pose that visually reached the object. Accept
            # anything the arm got close to, because the rigid carry then holds the real grasp offset.
            # Skip only a far miss, where a flat object's self-collision leaves the arm well short: that
            # is the air-grab, so park it and move on.
            _tcp = arm.get_tcp_pose()
            _miss = float(np.linalg.norm(np.asarray(_tcp.position_mm, dtype=np.float64)
                                         - np.array([float(oxy[0]), float(oxy[1]), grasp_z])))
            if _miss > 45.0:
                print(f"[loop] grasp UNREACHED ({_miss:.0f} mm) for {s['label']} -> skip (no air-grab)", flush=True)
                _skip.add(i)
                continue
            print(f"[loop] grasp reached ({_miss:.0f} mm)", flush=True)
        session.step_n(6)
        if s["decision"] == "jaw":
            gripper.close()
            session.step_n(8)

        # Engage the rigid render-safe carry: capture the object's pose relative to the wrist at the
        # grasp, where the arm has descended onto the object so the offset is small and real, and hold
        # it. The object stays where it was gripped and follows the wrist rigidly, so it reads as firmly
        # held rather than snapped to the tool or grasped in mid-air.
        o_pos, o_quat = objs[i].get_world_pose()
        state["rel"] = np.asarray(o_pos, dtype=np.float64) - np.asarray(wrist.get_world_pose()[0], dtype=np.float64)
        state["oquat"] = np.asarray(o_quat, dtype=np.float64)
        state["idx"] = i
        state["carry"] = True
        _hold(6)

        _move(arm, Pose(position_mm=pos + (-approach) * 210.0, quaternion_xyzw=quat, frame=Frame.BASE, label="lift"), "lift")
        _hold(hold)
        _move(arm, Pose(position_mm=np.array([dx, dy, otop + 340.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="transit"), "transit")
        _move(arm, Pose(position_mm=np.array([dx, dy, otop + 120.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="lower"), "lower")
        _hold(3)

        # place: carry off + kinematic set-down to the dest slot at the object's stable home orientation/height
        state["carry"] = False
        if s["decision"] == "jaw":
            gripper.open()
        hz = float(np.asarray(homes[i][0])[2])
        hq = np.asarray(homes[i][1], dtype=np.float64)
        start = np.asarray(objs[i].get_world_pose()[0], dtype=np.float64)
        target = np.array([dx / 1000.0, dy / 1000.0, hz], dtype=np.float64)
        for k in range(1, 11):
            t = k / 10.0
            objs[i].set_world_pose(position=start * (1.0 - t) + target * t, orientation=hq)
            try:
                objs[i].set_linear_velocity(np.zeros(3))
                objs[i].set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
            session.step(render=True)
        placed += 1
        print(f"[loop] placed {s['label']} at dest slot {placed - 1} "
              f"({np.round(np.asarray(objs[i].get_world_pose()[0]) * 1000, 1)} mm)", flush=True)
        _move(arm, Pose(position_mm=np.array([dx, dy, otop + 320.0]), quaternion_xyzw=quat, frame=Frame.BASE, label="retreat"), "retreat")

    arm.move_to_joints(JointPositions(park_q))
    _hold(hold)
    state["on"] = False
    session.step = orig_step  # type: ignore[assignment]
    for i, (_p, _l) in enumerate(handles.object_specs):
        fp = np.round(np.asarray(objs[i].get_world_pose()[0], dtype=np.float64) * 1000.0, 0)
        where = "SOURCE" if _obj_in_source(objs, i) else ("dest" if abs(fp[0] - DEST_XY[0]) <= _KLT_HALF_X + 40 else "OFF")
        print(f"[final] {_l:18s} at {fp} -> {where}", flush=True)
    _report_motion(wrist_track)
    dur = _write_mp4(frames, out, fps)
    session.stop()
    return {"placed": placed, "frames": len(frames), "duration_s": dur, "out": out}


def _report_motion(track: "list[np.ndarray]") -> None:
    if len(track) < 3:
        return
    arr = np.asarray(track, dtype=np.float64)
    d = np.diff(arr, axis=0)
    steps = np.linalg.norm(d, axis=1)
    dirs = d / (steps[:, None] + 1e-9)
    dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
    reversals = int(np.sum((dots < -0.3) & (steps[:-1] > 5.0) & (steps[1:] > 5.0)))
    print(f"[motion] wrist per-step mm: max={steps.max():.1f} mean={steps.mean():.2f} p95={np.percentile(steps, 95):.1f} "
          f"| sharp direction-reversals(>5mm)={reversals}  (0 == no flicker)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Autonomous bin-clearing demo (perceive -> decide -> pick w/ right tool -> place).")
    ap.add_argument("--out", type=str, default="logs/demo/bin_clearing_demo.mp4")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--max-picks", type=int, default=12)
    ap.add_argument("--prompt", type=str, default=None,
                    help="LANGUAGE-DRIVEN mode: a text prompt (e.g. 'box. package.' or 'a toy animal') -> REAL "
                         "GroundingDINO clears ONLY the matching objects, dimming the rest. Omit = clear all.")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    r = record_clearing(args.out, headless=not args.gui, fps=args.fps, max_picks=args.max_picks, prompt=args.prompt)
    print(f"placed {r['placed']} objects", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
