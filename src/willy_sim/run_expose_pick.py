"""Expose then pick demo (Isaac): the robot uncovers an occluded target and picks it amodally.

The scene is the industrial cell (UR5e and real KLT bins). The requested target is a compact toy
figure, by default the toy rhino; the blocker is a Perricone skincare box; a porcelain mug and a
metal clamp stand in as visible distractors. One Isaac boot runs five steps.

  * See: the robot perceives the scene with the target visible, runs GroundingDINO for the prompt,
    and remembers the centroid pixel of the target's mask in its own overhead camera. That belief is
    in image space and comes from the robot's own sensor, not from a ground-truth pose; a
    back-projected world centroid skews for an off-nadir object, image space does not.
  * Cover: the blocker is moved over the target so the overhead camera can no longer see it, and the
    target's rendered instance-segmentation mask drops to ~0 px.
  * Reason: the requested object is no longer visible. Ground truth is not consulted. The blocker is
    the visible object whose mask now sits at the pixel where the target was last seen. Belief
    persistence through occlusion is the buildable form of amodal reasoning here; there is no
    single-view amodal-segmentation net in this repository.
  * Redistribute: the blocker is released to a dynamic body and the arm runs the contact-redistribute
    corridor sweep of ``recovery.policy.execute_recovery_motion``. It descends beside the blocker's
    face into empty space, because descending onto it collides and stalls, sweeps +X into it near its
    centre of mass, and retracts. A swept jaw contact shoves the blocker off the target. No lift.
  * Pick: the target's mask reappears, GroundingDINO detects it, and the jaw, aligned to the figure's
    narrow body axis, picks it and carries it aside.

What is real and what is scripted, precisely:
  * Real: the "is the target visible" signal is Isaac's rendered instance-id segmentation, so a
    covered target reads ~0 px and reappears only once the blocker is physically pushed off. The
    amodal localisation uses the robot's own prior camera reading, not ``get_world_pose``; no
    ground-truth pose identifies the blocker. The GroundingDINO language match, the cuRobo grasp and
    the redistribute all execute: the arm drives descend, sweep and retract through ``arm.move``, and
    the swept contact shoves the now dynamic 1.5 kg blocker by physics.
  * Scripted: the occlusion arrangement is authored. The blocker is kinematically moved over the
    target, and the target is frozen underneath while covered so the overlapping box cannot eject it
    before the push. The grasp carry and the set-down are kinematic surrogates, as in every willy_sim
    pick demo.
  * Grounding limitation: the GSO meshes render untextured (the flat .obj dump carries geometry and
    UVs but no texture images), and GroundingDINO runs on the top-down overhead view. From straight
    above, only an object with a distinctive top-down silhouette grounds reliably, such as the rhino's
    bulk and horn or a screwdriver. Compact toy figures (lion, dino, unicorn) do not ground from
    overhead, coloured or not, so the reliable target is the rhino. ``--target`` exposes the others
    for when textured meshes land.

This is a demo runner, not the production service, and it is not validated on real hardware.

On-box only: needs Isaac plus the cuRobo and Coal engines. ``scripts/ext_deps/install.ps1``
installs both into ``ext_deps/``, which is where the code looks by default; no environment variable
is needed. Run with Isaac's bundled python from the repository root:

    <isaac-sim>\\python.bat -m src.willy_sim.run_expose_pick --target rhino --out logs/demo/expose_pick.mp4
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

# The bin-clearing cell and its scoring helpers, reused. These imports stay module-level and
# import-safe because Isaac is imported lazily inside record_expose.
from src.willy_sim.run_bin_clearing_demo import (
    ARM_PRIM,
    SOURCE_XY,
    _author_industrial_cell,
    _klt_fixtures,
    _score_all,
    _select_by_prompt,
    show_jaw,
)
from src.willy_sim.run_sorting_demo import (
    _move,
    _require_curobo,
    _setup_cine,
    _write_mp4,
)

CINE_POS_M = (2.55, -2.4, 1.9)
CINE_RES = (1280, 720)

# --- the occlusion roster (mm, relative to SOURCE_XY) ----------------------------------------------------
# Target: a compact toy figure, by default the rhino, the one figure that grounds top-down; see the
#   module docstring's grounding limitation. --target exposes the others for when textured meshes land.
# Blocker: the Perricone skincare box (84x60x149), held flat, with the orientation computed at runtime
#   from its real extent. Its narrow 84 mm side lies along the +X push axis, narrow enough to slide off
#   a mid-bin target inside the ~258 mm bin and to leave the push start reachable; its 149 mm side lies
#   along Y so it covers the figure's footprint. No suction seal is needed: it is redistributed by a
#   swept side-contact push, not lifted. A wide box, such as the 123 mm wooden-blocks box, cannot be
#   slid off a mid-bin target here at all.
# Distractors: a porcelain mug and a metal clamp, left in place, so the demo shows the robot clearing
#   only what blocks the requested object and then taking it.
_TARGETS = {
    # (GSO label, default prompt). Tools are distinct man-made shapes, so GroundingDINO grounds them
    # more readily than a grey untextured animal mesh, and their short axis fits a jaw grip.
    "screwdriver": ("screwdriver", "screwdriver"),
    "can_opener":  ("can opener", "can opener"),
    "spatula":     ("spatula", "spatula"),
    "clamp":       ("metal clamp", "clamp"),
    "rhino":       ("toy rhino", "toy rhino"),     # grounds top-down and settles tall enough to grip
    "lion":        ("toy lion", "lion"),
    "dino":        ("toy dinosaur", "toy animal"),
    "unicorn":     ("toy unicorn", "toy unicorn"),
    "bull":        ("toy bull", "toy bull"),
}
BLOCKER_LABEL = "skincare box tall"

DISTRACTOR_LABELS = ("porcelain mug", "metal clamp")  # a round mug and a metal clamp: visible variety
                                                      # without a second grey figure, which would
                                                      # confuse the top-down "toy rhino" grounding

TARGET_SLOT = (-25.0, 0.0)       # target a touch robot-side of centre: leaves +X room to slide the (narrow)
                                 # blocker off before the far wall, while keeping the push descend reachable
BLOCKER_START_SLOT = (-30.0, 150.0)  # blocker settles flat + visible beside the target, then slides over it
DISTRACTOR_SLOTS = ((55.0, -165.0), (60.0, 165.0))  # mug + clamp off the y=0 push line, clear of target+blocker
TARGET_Z = 60.0                  # target spawn height; it settles standing on the bin floor
BLOCKER_Z = 40.0                 # blocker spawned low; it settles flat, large face down, on the floor
OCC_REST_MM = 3.0                # blocker bottom rests this far above the target top; >=0 avoids
                                 # penetration, so releasing it to dynamic drops it onto the frozen
                                 # target instead of ejecting it
MIN_VISIBLE_PX = 250             # a target mask below this many pixels counts as occluded (not pickable)
BLOCKER_NEAR_PX = 45.0           # amodal ID tolerance (image space): a blocker counts as "over" the remembered
                                 # spot if its mask is within this many pixels of the target's last-seen centroid
MAX_PUSH_ROUNDS = 2              # try the redistribute push at most this many times before an honest abort


def _specs(target_key: str) -> list:
    """SimObjectConfigs: [target, blocker, mug, clamp] (index 0 is always the requested target)."""
    from src.willy_sim.gso_assets import GSO_BY_LABEL, gso_object_spec

    def at(label: str, slot: tuple[float, float], z: float, quat: Any, mass: float = 0.15) -> Any:
        x = SOURCE_XY[0] + slot[0]
        y = SOURCE_XY[1] + slot[1]
        return gso_object_spec(GSO_BY_LABEL[label], position_mm=(x, y, z), orientation_wxyz=quat, mass_kg=mass)

    tgt_label = _TARGETS[target_key][0]
    specs = [
        at(tgt_label, TARGET_SLOT, TARGET_Z, None),
        # heavy blocker (1.5 kg): a swept contact launches a 0.15 kg box off the table, while a firm mass
        # makes the push a controlled slide. Spawn flat (+90 deg about X lays its small face down) so it
        # settles in place; cover re-poses it exactly flat from its real extent anyway.
        at(BLOCKER_LABEL, BLOCKER_START_SLOT, BLOCKER_Z, (0.70710678, 0.70710678, 0.0, 0.0), mass=1.5),
    ]
    for lbl, slot in zip(DISTRACTOR_LABELS, DISTRACTOR_SLOTS):
        specs.append(at(lbl, slot, 55.0, None))
    return specs


def _mask_area(frame: Any, idx: int) -> int:
    """Visible-pixel count of object ``idx`` in the overhead frame (0 == fully occluded / out of view)."""
    segs = list(frame.segmentations)
    if idx >= len(segs):
        return 0
    return int(np.asarray(segs[idx].mask, dtype=bool).sum())


# --- cinematic AR narration (banner = the request; caption = the current phase) ---------------------------
_PHASE_TXT: dict[str, "tuple[str, tuple[int, int, int]]"] = {
    "recognized":   ("ZIEL ERKANNT  -  Position gemerkt", (235, 180, 60)),   # BGR cyan-blue
    "occluded":     ("ZIEL VERDECKT  -  Szene veraendert", (40, 170, 235)),  # BGR orange
    "redistribute": ("SCHIEBE Blocker weg  (redistribute)", (40, 200, 245)), # BGR amber
    "revealed":     ("ZIEL FREIGELEGT  -  greife", (60, 200, 60)),           # BGR green
    "done":         ("ZIEL GEGRIFFEN", (60, 200, 60)),
}


def _narrate(rgb: np.ndarray, prompt: str, phase: str, overhead: "np.ndarray | None" = None) -> np.ndarray:
    """Overlay the narration onto one cinematic frame.

    Draws the request as a top banner, the current phase as a bottom caption, and a picture-in-picture
    of the robot's overhead camera at the top right. The inset shows the occlusion the robot itself
    sees, so the oblique camera and the caption cannot contradict each other.
    """
    import cv2  # type: ignore[import-not-found]

    bgr = rgb[..., ::-1].copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = f'ANFRAGE:  "{prompt}"'
    (tw, th), _ = cv2.getTextSize(txt, font, 0.9, 2)
    x0 = max(10, (CINE_RES[0] - tw) // 2 - 22)
    cv2.rectangle(bgr, (x0, 16), (x0 + tw + 44, 16 + th + 26), (28, 28, 28), -1)
    cv2.rectangle(bgr, (x0, 16), (x0 + tw + 44, 16 + th + 26), (70, 210, 70), 2)
    cv2.putText(bgr, txt, (x0 + 22, 16 + th + 9), font, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
    cap, col = _PHASE_TXT.get(phase, _PHASE_TXT["occluded"])
    (cw, ch), _ = cv2.getTextSize(cap, font, 0.82, 2)
    cx = max(10, (CINE_RES[0] - cw) // 2 - 18)
    cy = CINE_RES[1] - 42
    cv2.rectangle(bgr, (cx, cy - ch - 16), (cx + cw + 36, cy + 12), (24, 24, 24), -1)
    cv2.rectangle(bgr, (cx, cy - ch - 16), (cx + cw + 36, cy + 12), col, 2)
    cv2.putText(bgr, cap, (cx + 18, cy), font, 0.82, col, 2, cv2.LINE_AA)
    if overhead is not None:
        pw, ph = 336, 252
        oh = cv2.resize(np.ascontiguousarray(overhead[..., ::-1]), (pw, ph))  # RGB to BGR, then downscale
        px, py = CINE_RES[0] - pw - 22, 74
        bgr[py:py + ph, px:px + pw] = oh
        cv2.rectangle(bgr, (px - 2, py - 2), (px + pw + 2, py + ph + 2), (70, 210, 70), 2)
        lbl = "ROBOTER-KAMERA (overhead)"
        (lw, lh), _ = cv2.getTextSize(lbl, font, 0.5, 1)
        cv2.rectangle(bgr, (px - 2, py - lh - 12), (px + lw + 12, py - 2), (24, 24, 24), -1)
        cv2.putText(bgr, lbl, (px + 4, py - 7), font, 0.5, (70, 210, 70), 1, cv2.LINE_AA)
    return bgr[..., ::-1]


def record_expose(out: str, *, headless: bool = True, prompt: str | None = None, target: str = "rhino",
                  capture_every: int = 5, hold: int = 10) -> dict:
    _require_curobo()
    # The detector runs on CPU: a GPU model load starves Isaac's rendered depth and every grasp is
    # rejected. run_bin_clearing_demo sets the same variable.
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WILLY_DEVICE", "cpu")
    if target not in _TARGETS:
        raise SystemExit(f"--target must be one of {sorted(_TARGETS)} (got {target!r})")
    prompt = prompt or _TARGETS[target][1]

    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix
    from src.robot.core import JointPositions
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes

    from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
    from src.willy_sim.scene import SceneAppearance, WRIST_LINK_PRIM
    from src.willy_sim.suction_mount import mount_real_suction_gripper

    src_walls, dst_walls = _klt_fixtures()
    n_walls = len(src_walls) + len(dst_walls)
    appearance = SceneAppearance(table_color=(0.16, 0.18, 0.22), dome_intensity=120.0, key_intensity=240.0)

    def _hook(stage: Any) -> None:
        mount_real_suction_gripper(stage, ARM_PRIM, hide_jaw=True)
        _author_industrial_cell(stage, n_walls)

    cell = bootstrap_sim_cell(None, headless=headless, scene_kwargs={
        "objects_override": _specs(target), "bin_walls": src_walls + dst_walls, "appearance": appearance,
    }, post_scene_hook=_hook)
    arm, gripper, handles, sim, cfg = cell.arm, cell.gripper, cell.handles, cell.sim, cell.cfg
    session = arm.session
    import omni.usd  # type: ignore[import-not-found]
    stage = omni.usd.get_context().get_stage()
    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)
    park_q = np.asarray(getattr(sim, "park_joint_positions", None) or sim.home_joint_positions, dtype=np.float64)

    # The cuRobo world is one low floor with its top at z=-50 mm and no KLT walls, so the jaw descends
    # clear. run_bin_clearing_demo registers the same floor.
    _floor = {"name": "table_floor", "dims_m": [3.0, 3.0, 1.0], "pose": [0.0, 0.0, -0.55, 1.0, 0.0, 0.0, 0.0]}
    n_world = int(arm.set_curobo_world([_floor]))
    print(f"[curobo-world] registered {n_world} obstacle(s)", flush=True)
    if n_world <= 0:
        session.stop()
        raise SystemExit("cuRobo world registration returned 0 -> aborting.")

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
    from pxr import UsdGeom  # type: ignore[import-not-found]

    objs = [SingleRigidPrim(p) for (p, _l) in handles.object_specs]
    labels = [lb for (_p, lb) in handles.object_specs]
    wrist = SingleRigidPrim(WRIST_LINK_PRIM)
    _cine, grab, app = _setup_cine(session)
    T_IDX, B_IDX = 0, 1  # target is always object 0, blocker object 1 (by _specs construction)

    # Per-object local bbox corners, in metres and in the body frame, used to derive the world top:
    # get_world_pose returns the mesh origin, not the top.
    _corners: list[np.ndarray] = []
    for (p, _l) in handles.object_specs:
        ext = UsdGeom.Mesh(stage.GetPrimAtPath(p + "/mesh")).GetExtentAttr().Get()
        lo = np.asarray(ext[0], dtype=np.float64) if ext is not None else np.array([-0.03, -0.03, 0.0])
        hi = np.asarray(ext[1], dtype=np.float64) if ext is not None else np.array([0.03, 0.03, 0.05])
        _corners.append(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                   for z in (lo[2], hi[2])], dtype=np.float64))

    def _world_corners_mm(idx: int) -> np.ndarray:
        pos_m, q = objs[idx].get_world_pose()
        r = to_rotation_matrix(np.array([q[1], q[2], q[3], q[0]], dtype=np.float64))
        return ((r @ _corners[idx].T).T + np.asarray(pos_m, dtype=np.float64)) * 1000.0

    def _world_top_mm(idx: int) -> float:
        return float(_world_corners_mm(idx)[:, 2].max())

    def _xy_aabb_mm(idx: int) -> tuple[float, float, float, float]:
        cw = _world_corners_mm(idx)
        return float(cw[:, 0].min()), float(cw[:, 0].max()), float(cw[:, 1].min()), float(cw[:, 1].max())

    def _flat_quat_wxyz(idx: int) -> np.ndarray:
        """A body orientation (wxyz) that lays object ``idx`` flat, from its local mesh extent.

        The smallest local axis maps to world Z (flat and low), the middle axis to world X (the +X push
        and slide axis, so the object fits the short bin), and the largest axis to world Y, covering a
        standing figure's long footprint. Computed at runtime because a raw scan's axis order is not
        known in advance.
        """
        loc = _corners[idx].max(axis=0) - _corners[idx].min(axis=0)
        small, mid, large = (int(a) for a in np.argsort(loc))
        R = np.zeros((3, 3), dtype=np.float64)
        R[:, small] = (0.0, 0.0, 1.0)
        R[:, mid] = (1.0, 0.0, 0.0)
        R[:, large] = (0.0, 1.0, 0.0)
        if np.linalg.det(R) < 0.0:          # keep it a proper (right-handed) rotation
            R[:, large] = (0.0, -1.0, 0.0)
        q = np.asarray(from_rotation_matrix(R), dtype=np.float64)   # xyzw
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # wxyz for Isaac set_world_pose

    try:
        handles.camera.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource
    perception = MultiObjectGroundTruthPerceptionSource(
        camera=handles.camera, targets=list(handles.object_specs), session=session,
        warmup_steps=max(20, sim.scene_setup.render_warmup_steps))

    from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector
    detector = GroundingDinoObjectDetector(cfg.models.objectdetector)
    print(f"[lang] GroundingDINO ready (CPU); target={target!r} prompt={prompt!r}", flush=True)

    frames: list[np.ndarray] = []
    # occ_idx/occ_pose: the blocker is held kinematically over the target while it covers it, re-asserted
    # each step. tgt_frozen: the target is frozen underneath so the overlapping box cannot eject it. The
    # redistribute push releases the blocker to a dynamic body; the target stays frozen so the push does
    # not drag it, and _pick_target releases it at the carry hand-off.
    state: dict[str, Any] = {"on": False, "carry": False, "idx": -1, "rel": None, "oquat": None,
                             "occ_idx": -1, "occ_pose": None, "phase": "recognized", "n": 0,
                             "tgt_frozen": False, "tgt_pose": None}

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

    def _hold_occluder() -> None:
        oi = state["occ_idx"]
        objs[oi].set_world_pose(position=state["occ_pose"][0], orientation=state["occ_pose"][1])
        try:
            objs[oi].set_linear_velocity(np.zeros(3))
            objs[oi].set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass

    def _freeze_target() -> None:
        objs[T_IDX].set_world_pose(position=state["tgt_pose"][0], orientation=state["tgt_pose"][1])
        try:
            objs[T_IDX].set_linear_velocity(np.zeros(3))
            objs[T_IDX].set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass

    def capturing_step(dt_s=None, *, render=False):  # noqa: ANN001, ANN202, ARG001
        orig_step(dt_s, render=True)
        if state["carry"] and state["rel"] is not None:
            _carry()
        if state["occ_idx"] >= 0 and state["occ_pose"] is not None:
            _hold_occluder()
        if state["tgt_frozen"] and state["tgt_pose"] is not None:
            _freeze_target()
        if state["on"]:
            state["n"] += 1
            if state["n"] % capture_every == 0:  # one frame per capture_every sim steps, for a watchable clip
                if app is not None:
                    app.update()
                f = grab()
                if f is not None:
                    oh = None  # the robot's live overhead view; the inset shows the real occlusion
                    try:
                        _ohf = handles.camera.get_current_frame()
                        _ohr = _ohf.get("rgb") if isinstance(_ohf, dict) else None
                        if _ohr is not None:
                            oh = np.asarray(_ohr)[..., :3]
                    except Exception:  # noqa: BLE001
                        oh = None
                    frames.append(_narrate(f, prompt, state["phase"], oh))

    session.step = capturing_step  # type: ignore[assignment]

    def _hold(k: int) -> None:
        for _ in range(k):
            session.step(render=True)

    def _topdown_quat() -> Any:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        closing = np.array([1.0, 0.0, 0.0])
        return _quaternion_from_axes(closing, approach), approach

    quat, approach = _topdown_quat()

    # --- perception + amodal helpers ----------------------------------------------------------------------
    def _perceive() -> "tuple[Any, list[dict], set[int]]":
        frame = perception.acquire()
        scored, _f = _score_all(perception, handles, session)
        selected, _dets = _select_by_prompt(detector, frame, prompt)
        return frame, scored, selected

    def _target_visible(frame: Any, scored: list, selected: set[int]) -> int:
        """T_IDX when the requested target is language-matched, visible enough and jaw-graspable, else -1.

        The check is on object T_IDX specifically, not on any jaw object the prompt grounds to. A small
        visible distractor such as the clamp can weakly ground to the prompt while the real target is
        hidden, and a looser check would let it masquerade as an exposed target and skip the
        redistribute.
        """
        if (T_IDX in selected and _mask_area(frame, T_IDX) >= MIN_VISIBLE_PX
                and T_IDX < len(scored) and scored[T_IDX]["decision"] == "jaw"):
            return T_IDX
        return -1

    def _mask_centroid_px(frame: Any, idx: int) -> "np.ndarray | None":
        """(row, col) centroid of object ``idx``'s mask in the overhead image, or None if not visible."""
        segs = list(frame.segmentations)
        if idx >= len(segs):
            return None
        m = np.asarray(segs[idx].mask, dtype=bool)
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        return np.array([ys.mean(), xs.mean()], dtype=np.float64)

    def _blocker_covering(frame: Any, belief_px: np.ndarray, exclude: set[int]) -> int:
        """The blocker, identified in image space: the visible object whose mask sits where the target was.

        The remembered pixel is the target's mask centroid from the robot's own camera, not a
        ground-truth pose. An image-space belief avoids the perspective bias that skews a
        back-projected world centroid for an off-nadir object. Prefers an object whose mask contains the
        remembered pixel, else the nearest mask within ``BLOCKER_NEAR_PX``. Returns -1 when nothing is
        near enough, so a bystander is never shoved.
        """
        br, bc = int(round(belief_px[0])), int(round(belief_px[1]))
        best, best_key = -1, (2, 1e9, 0.0)  # (not-contained, pixel-distance, -area); smallest wins
        for i in range(len(objs)):
            if i in exclude or i == T_IDX:
                continue
            segs = list(frame.segmentations)
            m = np.asarray(segs[i].mask, dtype=bool) if i < len(segs) else np.zeros((1, 1), bool)
            area = int(m.sum())
            if area < MIN_VISIBLE_PX:
                continue
            contains = bool(0 <= br < m.shape[0] and 0 <= bc < m.shape[1] and m[br, bc])
            if contains:
                dist = 0.0
            else:
                ys, xs = np.nonzero(m)
                dist = float(np.min(np.hypot(ys - br, xs - bc)))
            if dist > BLOCKER_NEAR_PX:
                continue
            key = (0 if contains else 1, dist, -float(area))
            if key < best_key:
                best, best_key = i, key
        return best

    # --- the redistribute push (contact-redistribute via execute_recovery_motion) -------------------------
    def _redistribute_push(bi: int) -> bool:
        """Shove blocker ``bi`` off the target with the corridor sweep of ``execute_recovery_motion``.

        Descend beside the blocker's -X face into empty space, not onto its top: descending onto it
        collides and either stalls or launches it. Descend to the box mid-height, because a side push
        near the centre of mass slides a flat box without toppling it. Then sweep +X so the closed jaw
        drives the box off the target, then retract. The blocker stays kinematically held through the
        hover approach so it cannot drift off the target on its own, and is released to a dynamic body
        just before the sweep; the target stays frozen so the push does not drag it. Returns True only
        when the recovery executor reports that it ran.
        """
        from src.robot.grasping.recovery.policy import (
            FixtureEnvelope,
            SceneRecoveryAction,
            SceneRecoveryPlan,
            SceneRecoveryPolicy,
            execute_recovery_motion,
        )
        show_jaw(stage)
        gripper.close()                 # a closed jaw = a defined push tool
        bx0, bx1, by0, by1 = _xy_aabb_mm(bi)
        b_top = _world_top_mm(bi)
        b_bot = float(_world_corners_mm(bi)[:, 2].min())
        start_x = bx0 - 22.0            # descend 22 mm to the -X of the box face, into empty space
        start_y = float((by0 + by1) * 0.5)
        mid_z = 0.5 * (b_top + b_bot)   # side contact near the CoM: the box slides, it does not tip
        hover_z = b_top + 120.0
        depth = hover_z - mid_z
        # Sweep just far enough that the box's -X edge passes the target centre: about half its width
        # plus a margin. The far end is capped short of the +X bin wall so the jaw never jams the box
        # into the wall and stalls; the box only has to slide clear of the target.
        amplitude = float(min((bx1 - bx0) * 0.5 + 72.0, 600.0 - start_x))
        _move(arm, Pose(position_mm=np.array([start_x, start_y, hover_z]), quaternion_xyzw=quat,
                        frame=Frame.BASE, label="push_hover"), "push_hover")   # approach while still held
        tcp = np.asarray(arm.get_tcp_pose().position_mm, dtype=np.float64)
        hover_pose = Pose(position_mm=tcp.copy(), quaternion_xyzw=quat, frame=Frame.BASE, label="push_start")
        fixture = FixtureEnvelope(
            center_mm=(float(tcp[0]) + amplitude * 0.5, float(tcp[1]), float(tcp[2]) - depth * 0.5),
            half_extents_mm=(amplitude + 90.0, 140.0, depth + 90.0),
            max_nudge_mm=5.0, max_agitate_amplitude_mm=amplitude + 10.0,
            agitate_contact_depth_mm=depth, agitate_sweep_offset_mm=0.0,
        )
        policy = SceneRecoveryPolicy(
            enabled=True, allowed_actions=(SceneRecoveryAction.CONTAINER_AGITATE,),
            max_recovery_actions=1, fixture=fixture, apply_modes=("dense_clutter",))
        plan = SceneRecoveryPlan(action=SceneRecoveryAction.CONTAINER_AGITATE,
                                 reason="expose_redistribute", agitate_amplitude_mm=amplitude)
        state["occ_idx"] = -1           # release the blocker to dynamic now, so the sweep shoves it by contact
        rep = execute_recovery_motion(arm=arm, plan=plan, policy=policy, current_tcp=hover_pose)
        print(f"[redistribute] outcome={rep.outcome} executed={rep.executed} amplitude={amplitude:.0f} "
              f"depth={depth:.0f} start=({start_x:.0f},{start_y:.0f}) mid_z={mid_z:.0f} tel={dict(rep.telemetry)}",
              flush=True)
        _move(arm, Pose(position_mm=np.array([start_x, start_y, hover_z + 120.0]), quaternion_xyzw=quat,
                        frame=Frame.BASE, label="push_retreat"), "push_retreat")
        arm.move_to_joints(JointPositions(park_q))
        _hold(8)
        return bool(rep.executed)

    # --- the target jaw pick ------------------------------------------------------------------------------
    def _pick_target(i: int, dest_xy: tuple[float, float]) -> bool:
        op = objs[i].get_world_pose()
        oxy = np.asarray(op[0], dtype=np.float64)[:2] * 1000.0
        otop = _world_top_mm(i)
        pos = np.array([float(oxy[0]), float(oxy[1]), otop], dtype=np.float64)
        # Align the jaw to the figure's narrow horizontal axis from its body orientation, not from an
        # axis-aligned world box: a figure settled at a yaw angle has a near-square world box that hides
        # its true narrow face. The 85 mm jaw cannot straddle a standing figure's ~140 mm long axis, so
        # the fingers close along its ~45 mm short axis or they drive into the body and the descent stalls.
        _q = np.asarray(objs[i].get_world_pose()[1], dtype=np.float64)
        _Rb = to_rotation_matrix(np.array([_q[1], _q[2], _q[3], _q[0]], dtype=np.float64))
        _loc = _corners[i].max(axis=0) - _corners[i].min(axis=0)          # local x,y,z extents
        _world_axes = _Rb @ np.eye(3)                                     # columns = body axes in world
        _horiz = [a for a in range(3) if abs(_world_axes[2, a]) < 0.7]    # axes that lie ~horizontal in world
        _narrow = min(_horiz, key=lambda a: _loc[a]) if _horiz else int(np.argmin(_loc[:2]))
        closing = _world_axes[:, _narrow].copy()
        closing[2] = 0.0                                                  # project the closing axis to horizontal
        n = float(np.linalg.norm(closing))
        closing = closing / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        g_quat = _quaternion_from_axes(closing, approach)
        arm.move_to_joints(JointPositions(home_q))
        session.step_n(6)
        show_jaw(stage)
        gripper.open()
        # Grip about 8 mm below the figure's top: the collider is a uniform box so a near-top grip holds,
        # and descending only that far keeps the straddling fingers clear of the frozen body all the way
        # down, so the descent does not stall. Floored at 40 mm so a short figure still gets a grip.
        grasp_z = max(40.0, otop - 8.0)
        _move(arm, Pose(position_mm=pos - approach * 110.0, quaternion_xyzw=g_quat, frame=Frame.BASE, label="standoff"), "standoff")
        _move(arm, Pose(position_mm=np.array([float(oxy[0]), float(oxy[1]), grasp_z]), quaternion_xyzw=g_quat, frame=Frame.BASE, label="grasp"), "grasp")
        _tcp = arm.get_tcp_pose()
        miss = float(np.linalg.norm(np.asarray(_tcp.position_mm, dtype=np.float64)
                                    - np.array([float(oxy[0]), float(oxy[1]), grasp_z])))
        if miss > 45.0:
            print(f"[pick] {labels[i]} grasp UNREACHED ({miss:.0f} mm) -> abort", flush=True)
            return False
        print(f"[pick] {labels[i]} grasp reached ({miss:.0f} mm)", flush=True)
        session.step_n(6)
        gripper.close()
        session.step_n(8)
        o_pos, o_quat = objs[i].get_world_pose()
        state["rel"] = np.asarray(o_pos, dtype=np.float64) - np.asarray(wrist.get_world_pose()[0], dtype=np.float64)
        state["oquat"] = np.asarray(o_quat, dtype=np.float64)
        state["idx"] = i
        if i == T_IDX:
            state["tgt_frozen"] = False  # release the freeze exactly at carry hand-off (freeze would else re-pin it)
        state["carry"] = True
        _hold(6)
        dx, dy = dest_xy
        _move(arm, Pose(position_mm=pos + (-approach) * 210.0, quaternion_xyzw=g_quat, frame=Frame.BASE, label="lift"), "lift")
        _hold(hold)
        _move(arm, Pose(position_mm=np.array([dx, dy, otop + 340.0]), quaternion_xyzw=g_quat, frame=Frame.BASE, label="transit"), "transit")
        _move(arm, Pose(position_mm=np.array([dx, dy, otop + 120.0]), quaternion_xyzw=g_quat, frame=Frame.BASE, label="lower"), "lower")
        _hold(3)
        state["carry"] = False
        gripper.open()
        start = np.asarray(objs[i].get_world_pose()[0], dtype=np.float64)
        target_p = np.array([dx / 1000.0, dy / 1000.0, otop / 1000.0], dtype=np.float64)
        for k in range(1, 11):
            t = k / 10.0
            objs[i].set_world_pose(position=start * (1.0 - t) + target_p * t, orientation=np.asarray(o_quat))
            try:
                objs[i].set_linear_velocity(np.zeros(3))
                objs[i].set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
            session.step(render=True)
        _move(arm, Pose(position_mm=np.array([dx, dy, otop + 320.0]), quaternion_xyzw=g_quat, frame=Frame.BASE, label="retreat"), "retreat")
        arm.move_to_joints(JointPositions(park_q))
        session.step_n(10)
        return True

    # ---- the see, cover, reason, redistribute, pick flow --------------------------------------------------
    result = {"seen": False, "occluded": False, "exposed": False, "picked": False, "pushes": 0}

    # settle
    arm.move_to_joints(JointPositions(park_q))
    session.step_n(60)
    for i, lbl in enumerate(labels):
        pmm = np.round(np.asarray(objs[i].get_world_pose()[0]) * 1000, 1)
        print(f"[dbg] obj {i} {lbl:18s} settled at {pmm} mm  top={_world_top_mm(i):.0f}", flush=True)

    # 1) see (scene memory): perceive with the target visible + remember where (its own sensor centroid).
    state["on"] = True
    state["phase"] = "recognized"
    arm.move_to_joints(JointPositions(park_q))
    session.step_n(20)
    frame, scored, selected = _perceive()
    ti = _target_visible(frame, scored, selected)
    if ti < 0:
        print("[flow] target NOT visible at start -> cannot remember it (scene setup issue) -> abort", flush=True)
        state["on"] = False
        session.step = orig_step  # type: ignore[assignment]
        session.stop()
        return {**result, "frames": len(frames), "duration_s": 0.0, "out": out}
    belief_px = _mask_centroid_px(frame, T_IDX)   # remember where in the camera the target appeared (image space)
    if belief_px is None:
        belief_px = np.array([0.0, 0.0], dtype=np.float64)
    result["seen"] = True
    print(f"[flow] SEE: target '{labels[ti]}' visible ({_mask_area(frame, ti)} px) -> "
          f"remember pixel=({belief_px[1]:.0f},{belief_px[0]:.0f}) (own camera, not GT)", flush=True)
    _hold(hold * 12)   # a clear beat on the visible target so the "see it + remember where" reads on camera

    # 2) cover (authored scene change): slide the blocker over the target (kinematic), freeze the target.
    _tp, _tq = objs[T_IDX].get_world_pose()
    state["tgt_pose"] = (np.asarray(_tp, dtype=np.float64), np.asarray(_tq, dtype=np.float64))
    state["tgt_frozen"] = True
    t_xy = np.asarray(_tp, dtype=np.float64)[:2]
    t_top_mm = _world_top_mm(T_IDX)
    # Hold the blocker flat, from its real extent, over the target: footprint centred on the target xy,
    # box bottom OCC_REST_MM above the target top, so releasing it to dynamic drops it onto the frozen
    # target instead of ejecting it. A flat box slides when pushed; a tall one topples and launches.
    _bq = _flat_quat_wxyz(B_IDX)
    _Rf = to_rotation_matrix(np.array([_bq[1], _bq[2], _bq[3], _bq[0]], dtype=np.float64))
    _rc_mm = (_Rf @ _corners[B_IDX].T).T * 1000.0            # flat-oriented corners, origin-relative (mm)
    _ctr_off_xy = _rc_mm[:, :2].mean(axis=0)                 # offset from origin to the footprint centre (xy, mm)
    _bot_off = float(_rc_mm[:, 2].min())                    # offset from origin to the box bottom (z, mm)
    over_origin_mm = np.array([t_xy[0] * 1000.0 - _ctr_off_xy[0], t_xy[1] * 1000.0 - _ctr_off_xy[1],
                               t_top_mm + OCC_REST_MM - _bot_off], dtype=np.float64)
    b_start = np.asarray(objs[B_IDX].get_world_pose()[0], dtype=np.float64)
    b_over = over_origin_mm / 1000.0
    state["phase"] = "occluded"
    for k in range(1, 19):  # animate the slide over the target (kinematic) so the "scene change" reads on cam
        t = k / 18.0
        pm = b_start * (1.0 - t) + b_over * t
        objs[B_IDX].set_world_pose(position=pm, orientation=_bq)
        try:
            objs[B_IDX].set_linear_velocity(np.zeros(3))
            objs[B_IDX].set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
        session.step(render=True)
    state["occ_idx"] = B_IDX
    state["occ_pose"] = (b_over, _bq)
    _hold(hold)
    frame = perception.acquire()
    areas = {labels[i]: _mask_area(frame, i) for i in range(len(objs))}
    bx0, bx1, by0, by1 = _xy_aabb_mm(B_IDX)
    print(f"[flow] COVER: visible-px {areas}", flush=True)
    print(f"[dbg-aabb] blocker footprint x[{bx0:.0f},{bx1:.0f}] y[{by0:.0f},{by1:.0f}] "
          f"belief_px=({belief_px[1]:.0f},{belief_px[0]:.0f}) target_top={t_top_mm:.0f} "
          f"blocker_top={_world_top_mm(B_IDX):.0f}", flush=True)
    result["occluded"] = _mask_area(frame, T_IDX) < MIN_VISIBLE_PX

    # 3-4) reason amodally, redistribute with a push, re-perceive; up to MAX_PUSH_ROUNDS.
    removed: set[int] = set()
    for _round in range(MAX_PUSH_ROUNDS):
        frame, scored, selected = _perceive()
        ti = _target_visible(frame, scored, selected)
        if ti >= 0:
            break
        bi = _blocker_covering(frame, belief_px, removed)
        if bi < 0:
            print("[flow] target occluded but no covering blocker found -> abort", flush=True)
            break
        print(f"[flow] REASON: target hidden -> amodal blocker = '{labels[bi]}' "
              f"(at the target's last-seen pixel) -> redistribute", flush=True)
        state["phase"] = "redistribute"
        _redistribute_push(bi)
        result["pushes"] += 1
        removed.add(bi)  # do not re-target the same blocker if a re-perceive still shows it near the belief
        _hold(hold)

    # 5) pick the revealed target.
    frame, scored, selected = _perceive()
    ti = _target_visible(frame, scored, selected)
    if ti >= 0:
        print(f"[flow] PICK: target '{labels[ti]}' exposed ({_mask_area(frame, ti)} px) -> grasp", flush=True)
        result["exposed"] = True
        state["phase"] = "revealed"
        # Keep the target frozen through the grasp descent: a re-dynamicised figure gets nudged or tips
        # and the grasp misses. The body-aligned open jaw straddles the narrow axis and descends clear to
        # a near-top grip. _pick_target releases the freeze at the carry hand-off.
        _hold(hold)
        # Set the target down within comfortable reach: the same x band as the bin, just off its -Y edge.
        # A destination too far or too close fails IK or times out on the carry moves, and the arm
        # stalls while the object rides along.
        result["picked"] = _pick_target(ti, (SOURCE_XY[0] - 30.0, SOURCE_XY[1] - 170.0))
        if result["picked"]:
            state["phase"] = "done"
            _hold(hold)
    else:
        print("[flow] target still occluded after redistribute -> honest no-pick", flush=True)

    arm.move_to_joints(JointPositions(park_q))
    _hold(hold)
    state["on"] = False
    session.step = orig_step  # type: ignore[assignment]
    for i, lbl in enumerate(labels):
        fp = np.round(np.asarray(objs[i].get_world_pose()[0], dtype=np.float64) * 1000.0, 0)
        print(f"[final] {lbl:18s} at {fp}", flush=True)
    dur = _write_mp4(frames, out, 24) if frames else 0.0
    session.stop()
    print(f"[result] {result}", flush=True)
    return {**result, "frames": len(frames), "duration_s": dur, "out": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="Expose-then-pick: amodally uncover an occluded target and pick it.")
    ap.add_argument("--out", type=str, default="logs/demo/expose_pick.mp4")
    ap.add_argument("--prompt", type=str, default=None, help="language request (default: the target's name)")
    ap.add_argument("--target", type=str, default="rhino", choices=sorted(_TARGETS),
                    help="which toy is the requested target (fallbacks if one grips poorly)")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    r = record_expose(args.out, headless=not args.gui, prompt=args.prompt, target=args.target)
    print(f"seen={r['seen']} occluded={r['occluded']} exposed={r['exposed']} "
          f"picked={r['picked']} pushes={r['pushes']}", flush=True)
    return 0 if r["picked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
