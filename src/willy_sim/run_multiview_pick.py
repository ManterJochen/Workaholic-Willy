"""Multi-view pick in Isaac: fixed-camera localize, wrist eye-in-hand refine, grasp.

Builds on the eye-in-hand pick service ``run_eih_pick.build_service``, which wires
``bootstrap_sim_cell(safety=True)`` plus ``wire_safety_guards`` and the cuRobo+Coal motion planner, so
the fail-closed safety chain is inherited unchanged. What this runner adds is the coarse localize step:
instead of one overhead camera, several fixed eye-to-hand cameras, an overhead plus symmetric obliques,
each localize the prompted target, and their BASE-frame centroids are fused with visibility weights, so
a target the overhead loses to occlusion or to the reaching arm is still localized from whichever
oblique sees it. The fused centroid drives the wrist view pose that ``service.pick()`` works from.
``run_occlusion_probe`` measures the same camera roles.

The switchable modes name the camera set that localizes; the wrist refine and grasp are identical in
all of them:
  * ``eth1``: overhead only, the single-camera baseline.
  * ``eth2``: overhead plus one oblique.
  * ``eth3``: overhead plus both symmetric obliques, for two-sided occlusion redundancy.

Scope: with ground-truth instance masks the overhead localizes the target's own mask correctly unless
the target is physically occluded, so ``eth1`` and ``eth3`` agree on separated cubes. Showing a
difference against the single-camera baseline needs a degradation source, such as real detection in
clutter or self-occlusion by the arm. The wrist shares the top-down blind spot: the obliques resolve
occlusion, the wrist refines the grasp up close.

On-box only, so Isaac is required. Run with Isaac's bundled python from the repository root:

    <isaac-sim>\\python.bat -m src.willy_sim.run_multiview_pick --mode eth3 --runs 5
"""

from __future__ import annotations

import argparse
import hashlib
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.schema.robot import SimObjectConfig
from src.willy_sim.harness.cli import add_cell_arguments, cell_profile_kwargs
from src.willy_sim.harness.gate import GateResult, gate_passed
from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource
from src.willy_sim.run_eih_pick import _load_calibrated_eih, build_service, coarse_centroid_base, topdown_view_pose
from src.willy_sim.scene import author_fixed_camera
from src.willy_sim.scene import OBJECT_PRIM
from src.robot.grasping.multiview.localize import (
    ViewLocalization,
    fuse_scene_points_base,
    fuse_view_localizations,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry import Transform
    from src.robot.grasping.types.grasp_point import GraspPoint

# A reachable clutter scene: 3 distinct-coloured cubes, well separated, graspable, around (450, 0).
CLUTTER_OBJECTS = [
    ("red cube", (0.90, 0.12, 0.12), (450.0, -95.0, 25.0)),
    ("green cube", (0.10, 0.65, 0.18), (450.0, 5.0, 25.0)),
    ("blue cube", (0.12, 0.22, 0.90), (450.0, 105.0, 25.0)),
]

# Close-neighbour scene: a neighbour within gripper reach of the target, so a grasp of the target would
# intrude on it. The wrist sees only the target from its single view and grasps blindly. Under
# --scene-collision the fused fixed-camera neighbour cloud lands in the gripper boxes and the colliding
# candidate is rejected.
CLOSE_NEIGHBOUR_OBJECTS = [
    ("green cube", (0.10, 0.65, 0.18), (450.0, 0.0, 25.0)),    # target (bin centre)
    ("red cube", (0.90, 0.12, 0.12), (450.0, -31.0, 25.0)),    # Touching neighbour (30 mm cubes, 1 mm gap)
]

# Hard-occlusion scene for the perception budget: a tall occluder on oblique_L's side. The obliques sit at
# y=-560 and y=+560 looking at the target at y=0, and oblique_L is the -Y camera the stop action commits
# on. The wall at y=-70 stands between oblique_L and the target and blocks its slanted downward view, so
# oblique_L's target_px collapses: stop localizes poorly and misses, while oblique_R stays clear and lets
# continue rescue the episode. The per-episode target jitter along Y varies the occlusion, a target toward
# +Y peeking past the wall and a target toward the wall hiding behind it, which is the
# occlusion-contingent outcome the easy scenes cannot produce. The wall stays 50 mm clear of the target's
# top-down pick column at y=0, so the wrist pick and the overhead are unaffected. The 4th tuple element is
# size_mm (w, d, h); without it the object is the default cube.
OCCLUDE_OBJECTS = [
    ("green cube", (0.10, 0.65, 0.18), (450.0, 0.0, 25.0)),                                  # target 30x30x50
    ("gray wall", (0.45, 0.45, 0.45), (450.0, -70.0, 60.0), (60.0, 40.0, 120.0)),            # Tall -Y occluder
]

# Tall-target scene: ``clutter`` with one variable changed, the target 90 mm tall instead of 50. It
# isolates the gate's table-clearance refusal: a 30x30x50 cube is too short for this jaw top-down, so
# the grasps reach under the table plane and the clearance check rejects them.
#
# 90 mm puts the top face well above the height at which the jaw envelope clears the table: rank-0
# envelope clearance grows with the object's height through the SFE stage and the
# ParallelJawGripperModel envelope. 90 mm also keeps a 3:1 aspect, because a 120 mm bar topples when a
# jaw touches it and a physics failure would be indistinguishable from a clearance failure. The 30x30
# footprint is unchanged, so height is the only thing that differs from ``clutter``.
#
# ``occlude`` is a different scene and does not exercise this: its 4th tuple element belongs to the
# occluder, its target stays the default cube, and it therefore returns the same numbers as
# ``clutter``. The rule that separates the two: in this scene the object the default prompt selects is
# the one carrying ``size_mm``.
TALL_TARGET_OBJECTS = [
    ("red cube", (0.90, 0.12, 0.12), (450.0, -95.0, 25.0)),                                # 30x30x50
    ("green cube", (0.10, 0.65, 0.18), (450.0, 5.0, 45.0), (30.0, 30.0, 90.0)),            # Target, tall
    ("blue cube", (0.12, 0.22, 0.90), (450.0, 105.0, 25.0)),                               # 30x30x50
]

# Blocker scene: the only scene that gives `grasping.ordering` something to decide.
#
# `_blocks(a, b)` in `loop/target_selector.py` is true when a is at least `depth_tolerance_mm` (10)
# closer to the camera than b and a's mask, dilated by `adjacency_radius_px` (5), touches b's. So the
# scene has to supply three things at once, and the other scenes each miss one:
#   * both objects graspable: `close` fails here, because 30x30x50 cubes are too short for a top-down
#     jaw (all_table_conflict), so `successes` never reaches two and the selector is not called.
#   * different heights: a top-down camera reads height as depth, and equal heights give no "in front
#     of" relation at all. `tall`'s cubes differ, 90 mm against 50, which is why they are reused here.
#   * adjacent masks: `tall` fails here, its cubes being 100 mm apart, which nothing bridges after a
#     5 px dilate.
#
# 50 mm centre to centre along +Y, a 20 mm face gap, with the tall cube as the blocker. The neighbour
# sits on the Y axis because a parallel jaw closes along one axis, and `close` puts its neighbour
# exactly where the fingers go. Whether a 20 mm gap falls inside the 5 px dilation depends on the
# camera and not on arithmetic here.
BLOCKER_OBJECTS = [
    ("green cube", (0.10, 0.65, 0.18), (450.0, 0.0, 45.0), (30.0, 30.0, 90.0)),   # Tall = the blocker
    ("red cube", (0.90, 0.12, 0.12), (450.0, 50.0, 25.0)),                        # shorter, 20 mm away
]

# Blocker-conflict scene: the one where ordering can change the choice. In the blocker scene above the
# blocker also carries the better local grasp, so preferring it says nothing; the baseline picks it
# anyway.
#
# To overturn a choice the unlock term has to beat the local-score gap, and `max_local_score_drop`
# (0.1) bounds how much quality ordering may trade away. So the blocker has to be worse to grasp and
# still worth taking first. Taller is forced, because a top-down camera reads height as depth and only
# the nearer object can block, which leaves width as the lever: a 60 mm wide jaw grasp scores below a
# 30 mm one while the height still makes the object the blocker.
BLOCKER_CONFLICT_OBJECTS = [
    ("green cube", (0.10, 0.65, 0.18), (450.0, 0.0, 45.0), (60.0, 30.0, 90.0)),   # Tall + wide: blocks, grasps worse
    ("red cube", (0.90, 0.12, 0.12), (450.0, 65.0, 25.0)),                        # narrow + short: 40 mm
    #                                                                               depth difference, above
    #                                                                               the 10 mm the blocker
    #                                                                               relation requires
]

# Corridor scene: the positive control `occlusion.hard_reject_enabled` has never had. The analyzer has
# only ever been observed lowering false blockage, never raising it on a neighbour that genuinely sits
# in the descent corridor. `occlude` cannot supply one: its wall blocks the fixed obliques' view, 70 mm
# to the side of a corridor whose radius is 20 mm.
#
# What the analyzer samples, per `scoring/corridor.py`: nine rays, the centre plus eight at
# `radius_mm` (20), marched up to `max_distance_mm` (200) along the approach axis over the depth map,
# with `confidence = 0.5*depth + 0.5*mask`. That second half is why the obstacle has to be a perceived
# object and not a fixture: a wall only the depth map sees caps the confidence at 0.5, below the 0.7
# reject threshold, however solidly it blocks. The same holds on a real cell, where hard_reject can
# only refuse what perception reports.
#
# So the scene uses a tall neighbour whose body sits inside the 20 mm corridor radius and towers
# through the march. It is placed on +Y, the axis a parallel jaw does not close on, so a 5 mm face gap
# does not make the target ungraspable.
CORRIDOR_OBJECTS = [
    # 90 mm rather than the 50 mm default: a 50 mm target passes the silhouette stage and then loses
    # every candidate at the support-footprint stage, the short-object top-down jaw limit `tall`
    # exists for. In a corridor scene that failure would read as the obstacle's doing and is nothing
    # of the kind.
    ("green cube", (0.10, 0.65, 0.18), (450.0, 0.0, 45.0), (30.0, 30.0, 90.0)),  # target, graspable
    ("gray wall", (0.45, 0.45, 0.45), (450.0, 40.0, 75.0), (40.0, 40.0, 150.0)), # near face at y=+20
    # This scene deliberately yields no grasp, and that is its result. The corridor radius is 20 mm
    # and the 2F-85's body needs more lateral clearance than that, so an obstacle inside the corridor
    # is an obstacle inside the gripper: the collision and footprint filter kills the candidate before
    # the corridor ever scores it. Move the wall far enough out for the target to be graspable and the
    # wall leaves the corridor too, so the blockage falls back to the clean-scene level. That is why a
    # corridor measurement can only ever see blockage fall.
]

# Each spec is (name, color, position_mm[, size_mm]); the optional 4th element makes the tuple heterogeneous.
SCENES: "dict[str, list[tuple[Any, ...]]]" = {
    "clutter": CLUTTER_OBJECTS, "close": CLOSE_NEIGHBOUR_OBJECTS, "occlude": OCCLUDE_OBJECTS,
    "tall": TALL_TARGET_OBJECTS, "blocker": BLOCKER_OBJECTS, "corridor": CORRIDOR_OBJECTS, "blocker_conflict": BLOCKER_CONFLICT_OBJECTS,
}

# --------------------------------------------------------------------------------------------------------
# Perception budget collect: the deterministic stop/continue data-collection policy.
# --------------------------------------------------------------------------------------------------------
# A LinUCB perception-budget policy trained on a corpus that carries fusion_latency on every record sees
# only the continue action and comes out degenerate: one cell, one action. Training a non-degenerate one
# needs a collect corpus with both actions, varied occlusion states and varied reward. Stop commits on the
# first oblique's view; continue spends an extra view and genuinely fuses the obliques, the strongest
# occlusion-resolving signal available here. So the collect policy explores: it continues with a
# probability that rises with the measured occlusion, while keeping enough spread that both actions appear
# across occlusion buckets and the reward can show which one pays. The draw is deterministic per (seed,
# episode) through a sha256 uniform, with no random-number state and no wall clock.
#
# The action strings equal PERCEPTION_ACTION_STOP and PERCEPTION_ACTION_CONTINUE in
# `perception_budget_policy`; they are the collect policy's own choice. The offline trainer re-derives the
# label independently from the presence of extra.fusion_latency_ms, so a continue episode must stamp
# fusion_latency or it is read back as a stop.
#
# A base of 0.45 keeps the stop and continue counts balanced even at low measured occlusion, where a
# well-seen target gives an occlusion near 0, so the action distribution is non-degenerate whatever the
# scene. The occlusion gain then tilts the draw toward continue as the single view loses the target,
# reaching about 0.85 at full occlusion.
PERCEPTION_COLLECT_BASE_CONTINUE_PROB: float = 0.45
PERCEPTION_COLLECT_OCC_GAIN: float = 0.40


def _trace_calc() -> bool:
    """Report whether ``WILLY_TRACE_CALC`` opts this run into per-pick calculator telemetry.

    The variable is read at call time rather than import time, so a value exported mid-session is
    honoured. The same variable drives ``run_dense_pick``'s trace, so one switch covers both runners.
    """
    import os

    return bool(os.environ.get("WILLY_TRACE_CALC", "").strip())


def _perception_seeded_uniform(seed: int, episode: int, *, salt: str = "") -> float:
    """A deterministic uniform in ``[0, 1)`` keyed on ``(seed, episode, salt)`` (no wall-clock / no rng state)."""

    h = hashlib.sha256(f"perception-collect:{salt}:{int(seed)}:{int(episode)}".encode()).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def perception_collect_decision(
    *,
    occlusion_ratio: float,
    seed: int,
    episode: int,
    base_continue_prob: float = PERCEPTION_COLLECT_BASE_CONTINUE_PROB,
    occ_gain: float = PERCEPTION_COLLECT_OCC_GAIN,
) -> "tuple[str, float]":
    """Return ``(action, continue_prob)`` for one collect episode; ``action`` in ``{"stop", "continue"}``.

    ``continue_prob = clamp(base + occ_gain * occlusion, 0, 1)``, and the action is continue when the
    seeded uniform falls below it. Higher occlusion means more continue, with per-episode exploration,
    so both actions get support and the reward can show which one pays, which is what makes the
    training corpus non-degenerate. Pure and deterministic, with no dependency on Isaac.
    """

    occ = 0.0 if occlusion_ratio < 0.0 else (1.0 if occlusion_ratio > 1.0 else float(occlusion_ratio))
    p = base_continue_prob + occ_gain * occ
    p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    u = _perception_seeded_uniform(seed, episode, salt="decision")
    return ("continue" if u < p else "stop"), float(p)

# Deterministic per-run target disturbances (dx, dy in mm) applied after the first localize, standing in
# for the target shifting during the approach through a neighbour nudge or settling. Sized to defeat an
# open-loop grasp: about 25 mm exceeds the parallel jaw's position tolerance for a 30 mm cube, so a stale
# grasp misses while the always-watching obliques re-perceive the new pose. Cycled by run index, so the
# Isaac run draws no random numbers.
DISTURBANCES = [(25.0, 0.0), (0.0, 25.0), (-25.0, 0.0), (0.0, -25.0), (18.0, 18.0)]

# Active fixed cameras per mode (the wrist eye-in-hand refine + grasp is identical across modes).
# eth1/2/3 = eye-to-hand camera count (overhead + N side cameras). "sides" is the two-side-camera rig.
MODE_CAMERAS = {
    "eth1": ["overhead"],                              # single overhead camera (the baseline)
    "eth2": ["overhead", "oblique_L"],                 # overhead + one side camera
    "eth3": ["overhead", "oblique_L", "oblique_R"],    # overhead + both side cameras
    # Two side cameras only, left and right, with no overhead: the arm is the dominant top-down
    # occluder, so two symmetric side views see the target past it and fuse into one localization,
    # through the same visibility-weighted fusion the eth* modes use. This is the industrial-cell rig.
    "sides": ["oblique_L", "oblique_R"],
}

# Industrial bin presets for the two-side rig. The real small_KLT is a rectangle (inner ~189 x 293 mm),
# oriented long-axis along base X so the two side cameras (at y = +/-560) look across the short (y) span and
# see over the near wall into the bin. "tray" is a Z-squashed shallow KLT the jaw reaches easily; "klt" is
# the real full-depth (146 mm) bin. Dimensions + KLT-visual placement measured from the shipped small_KLT.usd.
_KLT_HALF_X_MM = 146.5   # inner half-extent along base X (full ~293 mm, the long side)
_KLT_HALF_Y_MM = 94.7    # inner half-extent along base Y (full ~189 mm, the short side the cameras look across)
BIN_PRESETS: dict[str, dict[str, Any]] = {
    "tray": {"height_mm": 70.0, "real_klt": (450.0, 0.0, 35.1, 0.48)},   # shallow: the jaw reaches the floor
    "klt": {"height_mm": 146.0, "real_klt": (450.0, 0.0, 73.2)},         # real full-depth deep bin
}


def _bin_preset_walls(preset: str) -> "tuple[list, tuple[float, ...]]":
    """Build the (bin_walls, real_klt_bin) pair for a named industrial preset via the shared fixture factory."""
    from src.willy_sim.run_dense_pick import _make_bin_fixtures

    cfg = BIN_PRESETS[preset]
    walls = _make_bin_fixtures(
        center_xy_mm=(450.0, 0.0), half_width_mm=_KLT_HALF_X_MM, half_width_y_mm=_KLT_HALF_Y_MM,
        height_mm=float(cfg["height_mm"]), thickness_mm=8.0,
    )
    return walls, tuple(cfg["real_klt"])


def _scene_specs(scene: str) -> list[SimObjectConfig]:
    # The 4th tuple element (size_mm) is optional and gives a tall occluder or other non-cube object;
    # absent means the default cube.
    specs: list[SimObjectConfig] = []
    for spec in SCENES[scene]:
        n, c, p = spec[0], spec[1], spec[2]
        if len(spec) > 3:
            specs.append(SimObjectConfig(name=n, color=c, position_mm=p, size_mm=spec[3]))
        else:
            specs.append(SimObjectConfig(name=n, color=c, position_mm=p))
    return specs


def _select_by_prompt(labels: list[str], prompt: str) -> int:
    p = prompt.lower()
    scores = [sum(1 for w in lbl.lower().split() if w in p) for lbl in labels]
    return int(np.argmax(scores))


def multi_view_localize(
    cameras: "dict[str, tuple[MultiObjectGroundTruthPerceptionSource, Transform]]",
    target_idx: int,
) -> "tuple[np.ndarray | None, dict[str, Any]]":
    """Acquire from each active fixed camera, back-project its target centroid to BASE, and fuse.

    The per-camera back-projection (``coarse_centroid_base`` through that camera's CAMERA->BASE) and the
    visible pixel count are built here, which is the part that touches Isaac; the visibility-weighted
    fusion is the pure core :func:`fuse_view_localizations`. Returns the fused BASE XYZ in mm, or None
    when no camera saw the target, plus a per-camera diagnostic.
    """
    locs: list[ViewLocalization] = []
    per: dict[str, Any] = {}
    for name, (src, c2b) in cameras.items():
        frame = src.acquire()
        segs = list(frame.segmentations)
        px = int(np.asarray(segs[target_idx].mask).astype(bool).sum()) if target_idx < len(segs) else 0
        c = coarse_centroid_base(frame, c2b, seg_index=target_idx) if px > 0 else None
        per[name] = {"target_px": px, "centroid_mm": None if c is None else np.round(c, 1).tolist()}
        locs.append(ViewLocalization(name=name, centroid_base_mm=c, visible_px=px))
    return fuse_view_localizations(locs), per


def build_fused_scene_cloud(
    cameras: "dict[str, tuple[MultiObjectGroundTruthPerceptionSource, Transform]]",
    target_idx: int,
) -> "np.ndarray | None":
    """Fuse the neighbour geometry from every fixed camera into one BASE-frame scene cloud in mm.

    The neighbours are every object except the grasp target. Each camera back-projects its neighbour
    masks with ``masked_point_cloud`` and transforms them to BASE through its own CAMERA->BASE; the
    per-camera clouds are concatenated by the pure core :func:`fuse_scene_points_base`. The fixed
    cameras see neighbours the single grasp-synthesis view cannot, so feeding this cloud to the
    collision filter rejects grasps that would hit them. Returns the fused cloud, or None when no
    neighbour was seen, in which case nothing is injected and the run is byte-identical.
    """
    from src.robot.grasping.geometry import masked_point_cloud

    per_cam: list[np.ndarray] = []
    for _name, (src, c2b) in cameras.items():
        frame = src.acquire()
        m = np.asarray(c2b.to_matrix(), dtype=np.float64)  # CAMERA -> BASE
        cam_pts: list[np.ndarray] = []
        for j, seg in enumerate(frame.segmentations):
            if j == target_idx:  # the scene = neighbours; never the target (would self-reject every grasp)
                continue
            mask = np.asarray(getattr(seg, "mask", None), dtype=bool)
            if mask.size == 0 or not mask.any():
                continue
            pts = np.asarray(
                masked_point_cloud(mask, frame.depth_map, frame.intrinsics, unit="mm").points_mm,
                dtype=np.float64,
            )
            if pts.size:
                cam_pts.append((m[:3, :3] @ pts.T).T + m[:3, 3])  # camera -> BASE
        if cam_pts:
            per_cam.append(np.vstack(cam_pts))
    return fuse_scene_points_base(per_cam)


def build_fused_target_cloud(
    cameras: "dict[str, tuple[MultiObjectGroundTruthPerceptionSource, Transform]]",
    target_idx: int,
) -> "dict[str, np.ndarray]":
    """Back-project each camera's target mask to a BASE point cloud in mm.

    Mirror of :func:`build_fused_scene_cloud` but for the grasp target (``j == target_idx``), and keyed
    per camera so the caller can both fuse the clouds for 3D grasp synthesis over all views and compare
    each single view's biased footprint against the fused one. Returns ``{camera_name: (N,3) BASE
    points}`` for every camera that saw the target; a camera whose mask is empty is omitted.
    """
    from src.robot.grasping.geometry import masked_point_cloud

    per_cam: dict[str, np.ndarray] = {}
    for name, (src, c2b) in cameras.items():
        frame = src.acquire()
        segs = list(frame.segmentations)
        if target_idx >= len(segs):
            continue
        mask = np.asarray(getattr(segs[target_idx], "mask", None), dtype=bool)
        if mask.size == 0 or not mask.any():
            continue
        pts = np.asarray(
            masked_point_cloud(mask, frame.depth_map, frame.intrinsics, unit="mm").points_mm,
            dtype=np.float64,
        )
        if pts.size:
            m = np.asarray(c2b.to_matrix(), dtype=np.float64)  # CAMERA -> BASE
            per_cam[name] = (m[:3, :3] @ pts.T).T + m[:3, 3]
    return per_cam


def compare_grasp_views(
    cameras: "dict[str, tuple[MultiObjectGroundTruthPerceptionSource, Transform]]",
    target_idx: int,
    cfg: Any,
) -> "tuple[dict[str, Any], dict[str, GraspPoint | None]]":
    """Synthesize a grasp from each camera's view of the target and report which view yields the best one.

    When a view is occluded, by the arm or by clutter, its silhouette is empty or partial and it
    synthesizes no grasp or a poor one, while an oblique that still sees the target synthesizes a good
    one. The best-view selection then falls back to that oblique and produces a grasp the occluded view
    alone could not.

    Returns ``(summary, grasps)``: ``summary[name]`` is a JSON-friendly ``{px, success, score}``, and
    ``grasps[name]`` is the best BASE-frame :class:`GraspPoint` for that view, or None. Synthesis only:
    nothing is executed here, and an executor takes ``grasps[best_view]``.
    """
    # Through the factory, not by name, so `robot.grasping.calculator: deep` reaches the per-view
    # synthesis instead of silently grading the analytic generator on every camera.
    from src.robot.grasping.calculator_factory import build_calculator

    g = cfg.robot.gripper
    out: dict[str, Any] = {}
    grasps: dict[str, GraspPoint | None] = {}
    # One calculator per camera, cached, which under `deep` is the difference between a run and a
    # stall. The build sits in a per-camera loop that is itself inside the per-run loop, so it is
    # reached once per camera per run. For the analytic generator that is a cheap object; for the
    # learned one each build is a `torch.load` call and a CUDA context.
    #
    # The key is the intrinsics, not the camera name. `camera_matrix` is the one keyword argument that
    # genuinely differs per view, and a cache keyed on the name would hand view B the matrix of view A
    # the moment a rig reused a name across runs. So the key is the numbers themselves.
    cache: dict[bytes, Any] = {}
    for name, (src, c2b) in cameras.items():
        frame = src.acquire()
        segs = list(frame.segmentations)
        px = int(np.asarray(segs[target_idx].mask).astype(bool).sum()) if target_idx < len(segs) else 0
        if px == 0:
            # Before the build, deliberately. A blind camera contributes nothing, and building here
            # would make a fail-closed refusal fire for a view that was about to be skipped.
            out[name] = {"px": 0, "success": False, "score": None}
            grasps[name] = None
            continue
        intrinsics = np.asarray(frame.intrinsics, dtype=np.float64)
        key = intrinsics.tobytes()
        if key not in cache:
            cache[key] = build_calculator(
                cfg.robot,
                camera_matrix=intrinsics,
                max_grip_width_mm=g.max_width_mm, min_grip_width_mm=g.min_width_mm,
            )
        calc = cache[key]
        result = calc.compute_result(segs[target_idx], frame.depth_map, camera_to_base=c2b)
        out[name] = {"px": px, "success": bool(result.is_success),
                     "score": round(float(result.top_score), 3) if result.is_success else None}
        grasps[name] = result.best if result.is_success else None
    return out, grasps


def _best_view(summary: dict[str, Any]) -> str | None:
    """The highest-scoring view that synthesized a grasp (None if every view failed)."""
    ranked = [(n, r) for n, r in summary.items() if r["success"]]
    ranked.sort(key=lambda kv: kv[1]["score"], reverse=True)
    return ranked[0][0] if ranked else None


def _reorient_topdown(grasp: "GraspPoint") -> "GraspPoint":
    """Re-express an image-space silhouette grasp as a top-down grasp.

    The GraspCalculator defaults a grasp's approach to the camera's optical axis: ``[0,0,1]`` in CAMERA
    frame, which is ``camera_to_base @ [0,0,1]`` in BASE. For the overhead that is roughly base -Z, a
    real top-down grasp; for an oblique camera it is the camera's tilted viewing direction, about
    40 deg off vertical, which is an artifact of where the camera sits rather than a synthesized
    approach. What an oblique view does carry that is executable is the position, the grip width and
    the horizontal closing axis. This keeps those three and sets the approach to base -Z, the axis the
    sim executor grasps cubes along. A genuinely tilted, planned approach needs full 3D multi-view
    synthesis instead.
    """
    from src.robot.grasping.types.grasp_point import GraspPoint

    closing = np.asarray(grasp.axis, dtype=np.float64)
    horiz = np.array([closing[0], closing[1], 0.0], dtype=np.float64)
    n = float(np.linalg.norm(horiz))
    horiz = horiz / n if n > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float64)  # degenerate: base +X
    return GraspPoint(
        position=np.asarray(grasp.position, dtype=np.float64),
        approach=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        axis=horiz,
        grip_width_mm=float(grasp.grip_width_mm),
        score=float(grasp.score),
        frame=grasp.frame,
        label=f"{grasp.label}+topdown",
        metadata=dict(grasp.metadata),
    )


def _print_block_carriers(report: Any) -> None:
    """Print one line per pick naming every carrier the thirteen `grasping.*` blocks act through.

    Unconditional, because a null result that cannot be told apart from "the block never ran"
    is not a measurement. The calculator log line is the wrong place to look: only three of
    the thirteen blocks touch the calculator at all, and the rest act on the service and the
    orchestrator.

    The line also carries `telemetry['reason']`, which is how the closed-loop runs explain
    themselves: `mode_not_available` names a stage, while the reason names the cause, for
    instance `mode_requires_verification_but_no_verifier_wired`. `execution_failed` swallowing
    cuRobo's own message has the same shape.

    Every field name below is read off the dataclass. A guessed name prints None for every
    block in every run, which is indistinguishable from "no block fired": a broken instrument
    reporting exactly what it is meant to detect. The names that catch a guess out are
    `predicted_success_probability`, `reason_code` and `fused`. Read the dataclass before
    adding a field here.

    The two halves answer different questions. `cfg_on` is what the service believes is
    enabled, which proves the YAML arrived; the carriers are what actually ran. A block that
    shows enabled in the first half and nothing in the second is wired but inert.
    """
    tel = dict(getattr(report, "telemetry", None) or {})
    dec = getattr(report, "decision", None)
    unc = getattr(report, "uncertainty", None)
    shadow = getattr(report, "shadow_success_telemetry", None)
    eff = getattr(report, "effective_config", None)
    on: list[str] = []
    if eff is not None:
        try:
            on = sorted(k for k, v in eff.to_dict().items() if k.endswith("_enabled") and v is True)
        except Exception:  # noqa: BLE001 (a trace must never break the pick)
            on = ["<to_dict failed>"]
    # The per-stage latency spans and anything else the report folded in. Without them the
    # `performance` block cannot be judged: no SLO_BREACH firing makes "no breach because fast" and
    # "no breach because nothing was measured" look the same, and the spans that separate the two live
    # in report.telemetry. Filtered rather than dumped whole, because the dict also carries attempt_id
    # and sampling_mode, which say nothing about latency.
    lat = {k: v for k, v in tel.items() if "ms" in k or "latency" in k or "p95" in k}
    # Why the motion failed, per attempt, from the fields PickAttempt carries. `execution_failed`
    # names a stage; these name the cause, which on the occlude scene is the whole question.
    _atts = getattr(getattr(report, "pick_report", None), "attempts", None) or ()
    motion = [
        f"{getattr(a, 'motion_status', None)}/{getattr(a, 'motion_message', None)}"
        for a in _atts
        if getattr(a, "motion_status", None) or getattr(a, "motion_message", None)
    ]
    # Which object the loop took, and what the selector said about it. `ordering` changes exactly this
    # and nothing else, so a run that does not print it cannot measure the block however many picks it
    # does. Only meaningful without a target label: with one there is a single executable
    # segmentation.
    _last = _atts[-1] if _atts else None
    _ord = getattr(_last, "ordering_decision", None)
    parts = [
        f"cfg_on={on}",
        f"chosen_target={getattr(_last, 'target_index', None)}",
        f"ordering={_ord.to_dict() if _ord is not None else None}",
        f"latency={lat}",
        f"motion={motion}",
        f"reason={tel.get('reason')}",
        f"phase_gate={tel.get('phase_gate')}",
        f"decision={getattr(getattr(dec, 'action', None), 'value', None)}"
        f"/{getattr(dec, 'reason_code', None)}",
        f"dec_uncertainty={getattr(dec, 'uncertainty_score', None)}",
        f"unc_fused={getattr(unc, 'fused', None)}",
        f"unc_available={getattr(unc, 'fused_available', None)}",
        f"unc_fail_closed={getattr(unc, 'fail_closed', None)}",
        f"shadow_p={getattr(shadow, 'predicted_success_probability', None)}",
        f"shadow_phase={getattr(shadow, 'model_lifecycle_phase', None)}",
        # Count and names. "1 action" is not an answer when the safety rule is about which action: a
        # `rescan` after a planner refusal re-perceives and moves nothing, while a `next_viewpoint`
        # drives the arm somewhere the operator did not name. The trail carries the names, so print
        # them; the count alone cannot tell the two apart, and that distinction is the decision.
        f"recovery_actions={len(getattr(report, 'recovery_actions', ()) or ())}",
        f"recovery_trail={tel.get('recovery_trail_actions')}"
        f"/{tel.get('recovery_trail_terminal_reason')}",
        f"blend={getattr(report, 'ranking_blend_telemetry', None) is not None}",
        f"rerank={getattr(report, 'uncertainty_rerank_telemetry', None) is not None}",
        f"fusion={getattr(report, 'fusion_telemetry', None) is not None}",
        f"commit={getattr(getattr(report, 'commit_decision', None), 'committed', None)}",
    ]
    print("[report] " + " ".join(parts), flush=True)


def run_multiview_gate(
    runs: int = 5,
    *,
    mode: str = "eth3",
    headless: bool = True,
    data_dir: str | None = None,
    # Which cell to boot, from --robot-model and --profile. The multi-camera runner needs this more
    # than most: MODE_CAMERAS names overhead, oblique_L and oblique_R, and those are exactly the three
    # the `tiltcam` profile re-places, so a profile chain here is what lets the closed-loop path be
    # measured under the optics the real cell has.
    cell_kwargs: "dict | None" = None,
    marker: str = "ground_truth",
    prompt: str = "a green cube",
    view_height_mm: float = 275.0,
    continuous_guard: bool = False,
    scene_collision: bool = False,
    approach_validate: bool = False,
    fuse_geometry: bool = False,
    fuse_neighbours: bool = False,
    # Which boot path builds the service. "config", the default, goes through from_robot_config, the
    # path a real cell takes, so a number measured here is a statement about that cell rather than
    # about this runner's hand-wiring. "components" is the runner's own wiring and stays as the
    # comparison: the two must agree before the default is worth trusting.
    boot: str = "config",
    # Which GraspMode the service is built in, which is not the camera set: that is ``mode`` above.
    #
    # The mode decides which config blocks can fire, and "easy" reaches almost none of them: of the
    # `grasping.*` blocks that carry an `apply_modes` gate, easy is in exactly one, `success_model`.
    # Measuring the others without this knob gives a green "no effect, harmless" result for blocks
    # that are unreachable by construction, which would then run for the first time on a real cell.
    grasp_mode: str = "easy",
    # See ``run_eih_pick.build_service``: hand the sub-policy ownership to the config blocks so that
    # toggling `decision`/`closed_loop`/`verification` is what changes the run, not this runner.
    config_subpolicies: bool = False,
    # Opt-in reachability filter; see run_eih_pick.build_service. Two effects at once: it discards
    # unreachable candidates, and it is the precondition for `grasping.feasibility` to re-rank at all.
    # Measure them apart, IK alone first and then IK with the feasibility layer.
    ik_service: bool = False,
    # The paired baseline for the multi-object perception default. True restores the single-mask wrist
    # source, so the multi-object default and its predecessor can be measured as a pair on the same
    # scene and the same seed. A default changed without its own baseline gives a number nobody can
    # attribute afterwards.
    single_mask_perception: bool = False,
    # The only configuration in which `grasping.ordering` can be measured. With a target label exactly
    # one segmentation is executable, so the selector is handed a single candidate and the block is
    # structurally inert. Bin-clearing semantics, with no label and every object a legitimate target,
    # is the situation ordering exists for. See `run_eih_pick.build_service(clear_any_object=...)` for
    # why an unlabelled frame is correct here and a bug under a prompt.
    clear_any: bool = False,
    scene: str = "clutter",
    # Where the ground-truth oracle puts the perceived surface. The sources in
    # `perception/ground_truth.py` report the object's centre depth biased up by this, not the
    # object's top face, so the default 12 mm makes a 30x30x50 cube reconstruct as a 37 mm prism and a
    # 90 mm one as 57 mm. That is invisible to a pick that succeeds anyway and fatal to any check that
    # reasons about real geometry: the table check kills the short cube, and cuRobo refuses the tall
    # one because a top-down approach to the planned anchor would pass through the object's unseen
    # upper third. Set it to half the object's height and the perceived surface lands on the real top
    # face.
    grasp_lift_mm: float = 12.0,
    #: False lets the camera's own rendered depth reach the pick, instead of a flat ground-truth sheet
    #: at the object's centre. That is the real-hardware path: it is what a RealSense delivers.
    ground_truth_depth: bool = True,
    compare_views: bool = False,
    execute_best_view: bool = False,
    reorient_topdown: bool = False,
    synthesize_3d: bool = False,
    probe_refine: bool = False,
    closed_loop: bool = False,
    stale_baseline: bool = False,
    bin_preset: str | None = None,
    calibrated: bool = False,
    collect_perception: bool = False,
    perception_continue_mode: str = "fusion",
    perception_jitter_mm: float = 30.0,
    record_log: str | None = None,
    seed: int = 0,
) -> GateResult:
    """Localize from the ``mode`` camera set, refine at the wrist, grasp, and score on the lift gate.

    Safety is inherited from ``build_service``: ``bootstrap_sim_cell(safety=True)`` wires the static
    Coal mesh self-collision preflight at a min_distance of 10 mm together with the workspace and
    joint-limit guards, which are fail-closed on every commanded pose, and the default cuRobo+Coal
    motion planner plans collision-free trajectories. ``continuous_guard``, the extra per-waypoint
    avoidance monitor, is opt-in and off by default: its 8 mm avoidance margin aborts the
    retreat-to-home reset path here, which is physically safe with 7 mm of clearance. The static
    preflight is the baseline the other pick runners share.
    """
    if mode not in MODE_CAMERAS:
        raise SystemExit(f"unknown mode {mode!r}; expected one of {sorted(MODE_CAMERAS)}")
    print(f"=== MULTI-VIEW pick (mode={mode}: {MODE_CAMERAS[mode]} localize -> wrist refine -> grasp) ===", flush=True)

    # Select the target before build_service, so the wrist ground-truth perception targets the same prim
    # that is localized and aimed at; object prims are assigned in spec order as f"{OBJECT_PRIM}_{i}".
    # Otherwise the wrist perceives the default Object_0 while the arm stands over a different target,
    # its mask comes back empty and no grasp is synthesized.
    scene_labels = [spec[0] for spec in SCENES[scene]]  # (name, color, pos[, size]), so the len varies
    target_idx = _select_by_prompt(scene_labels, prompt)
    target_prim = f"{OBJECT_PRIM}_{target_idx}"
    target_label = scene_labels[target_idx]
    print(f"scene={scene!r} prompt={prompt!r} -> target[{target_idx}]={target_label!r} ({target_prim})", flush=True)

    # Opt-in industrial bin: a preset builds the KLT walls + real-KLT visual and enables cuRobo-world planning
    # around them (byte-identical when bin_preset is None). The two-side rig localizes the target inside the bin.
    _bin_walls, _real_klt = _bin_preset_walls(bin_preset) if bin_preset else (None, None)
    if bin_preset:
        print(f"industrial bin preset={bin_preset!r}: {len(_bin_walls or [])} walls + real small_KLT visual", flush=True)
    service, arm, gripper, handles, cfg, _view_pose, _cell = build_service(
        headless=headless, data_dir=data_dir, marker=marker,
        objects_override=_scene_specs(scene), wrist_target_prim=target_prim,
        # Static Coal preflight is always on (bootstrap safety=True); the per-waypoint continuous guard is opt-in.
        continuous_guard=continuous_guard,
        bin_walls=_bin_walls, real_klt_bin=_real_klt, curobo_bin_world=bin_preset is not None,
        cell_kwargs=cell_kwargs, service_from_config=(boot == "config"),
        grasp_lift_mm=grasp_lift_mm, ground_truth_depth=ground_truth_depth,
        mode=grasp_mode, config_subpolicies=config_subpolicies, ik_service=ik_service,
        multi_object_perception=not single_mask_perception, clear_any_object=clear_any,
    )
    print(f"[grasp-mode] service built in mode={grasp_mode!r} "
          f"(config_subpolicies={config_subpolicies})", flush=True)
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.willy_sim.scene import camera_to_base_ground_truth
    from src.robot.core import JointPositions

    sim = cfg.robot.sim
    gate = sim.scene_setup.gate
    session = arm.session

    cal = _load_calibrated_eih(marker)
    if cal is None:
        raise SystemExit(f"multi-view pick requires a calibrated EIH artifact (marker={marker}, quality good/excellent)")
    p_cam_tool = np.asarray(cal.to_matrix(), dtype=np.float64)[:3, 3]

    object_specs = list(handles.object_specs)  # [(prim_path, label), ...] in spec order

    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)

    # Build the active fixed-camera localizers. The overhead uses a small near clip so a reaching arm
    # is rendered; the obliques are authored world-fixed and their CAMERA->BASE is fit against the sim
    # ground truth. Under --calibrated the active fixed cameras and their CAMERA->BASE come from the
    # central config, from grasping.fusion.cameras through build_config_frame_resolvers, and each
    # camera uses its own persisted eye-to-hand calibration, written by
    # `run_eth_calibrate --camera <id>`, rather than the sim ground-truth oracle. With the flag off
    # the path is the byte-identical ground-truth path keyed by --mode.
    resolvers: dict[str, Any] = {}
    if calibrated:
        from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolvers
        from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver

        # --calibrated is an explicit opt-in, so fusion.enabled is forced for the resolver build: the
        # config only needs to declare grasping.fusion.cameras, and fusion stays off by default for
        # every other runner.
        gr = cfg.robot.grasping
        if gr is not None:
            gr = gr.model_copy(update={"fusion": gr.fusion.model_copy(update={"enabled": True})})
        resolvers = build_config_frame_resolvers(gr)
        # --calibrated keeps --mode's camera-set semantics and only swaps the extrinsics source, using
        # each camera's persisted calibration instead of the sim ground-truth oracle. So `active` still
        # comes from --mode, the header and summary labels stay accurate, and a calibrated subset such
        # as eth1, eth2 or sides is runnable. Fail closed when a camera the mode needs has no
        # calibrated eye_to_hand resolver: never silently run a smaller rig than was requested.
        wanted = MODE_CAMERAS[mode]
        missing = [n for n in wanted if not isinstance(resolvers.get(n), StaticCameraToBaseResolver)]
        if missing:
            raise SystemExit(
                f"--calibrated --mode {mode} needs a calibrated eye_to_hand camera for {missing} in "
                "grasping.fusion.cameras. Calibrate each (run_eth_calibrate --camera <id>) and point the "
                "config at the eth_<id>.json artifacts, or pick a --mode whose cameras are all calibrated.")
        active = list(wanted)
        unused = [n for n in resolvers if n not in wanted]
        if unused:  # no silent surprises: a calibrated camera this --mode doesn't use is called out.
            print(f"[calibrated] note: config declares calibrated cameras not used by --mode {mode}: "
                  f"{unused} (ignored)", flush=True)
        print(f"[calibrated] active cameras from config fusion.cameras: {active}", flush=True)
    else:
        active = MODE_CAMERAS[mode]

    cameras: dict[str, tuple[MultiObjectGroundTruthPerceptionSource, Any]] = {}
    try:
        cell_cam = handles.camera
        cell_cam.set_clipping_range(0.05, 1.0e6)
    except Exception:  # noqa: BLE001
        pass
    if "overhead" in active:
        overhead_c2b = resolvers["overhead"].transform if calibrated else handles.camera_to_base
        cameras["overhead"] = (
            MultiObjectGroundTruthPerceptionSource(camera=handles.camera, targets=object_specs, session=session, warmup_steps=6),
            overhead_c2b,
        )
        if calibrated:
            print(f"overhead -> CALIBRATED cam->base mm={np.round(np.asarray(overhead_c2b.to_matrix())[:3, 3], 1)}", flush=True)
    for name in ("oblique_L", "oblique_R"):
        if name not in active:
            continue
        # Config-driven: the oblique camera pose/aim/near-clip/resolution come from sim.cameras[name]
        # (robot.sim.yaml, the sim profile). mount_aim_mm is the WORLD look-at target for a fixed eye-to-hand cam.
        ccfg = sim.cameras.get(name)
        if ccfg is None or ccfg.position_mm is None or ccfg.mount_aim_mm is None:
            raise SystemExit(f"config sim.cameras[{name!r}] must define position_mm + mount_aim_mm for {mode}")
        cam = author_fixed_camera(
            session, ccfg.prim_path or f"/World/Cameras/{name}",
            position_mm=ccfg.position_mm, aim_mm=ccfg.mount_aim_mm,
            near_clip_m=ccfg.near_clip_m or 0.05,
            resolution=ccfg.resolution if ccfg.resolution else (640, 480),
        )
        if calibrated:
            c2b = resolvers[name].transform
            print(f"{name} @ {tuple(ccfg.position_mm)} -> CALIBRATED cam->base mm={np.round(np.asarray(c2b.to_matrix())[:3, 3], 1)}", flush=True)
        else:
            # Warm the annotator before reading it. ``author_fixed_camera`` returns before the depth
            # annotator has produced a frame, so an immediate read can come back with an empty buffer
            # and abort the gate. The process does not die on that exception either: it hangs at
            # idle, so a watcher waiting on the summary line waits forever. Retry with escalating
            # warmup rather than a fixed step count, as run_eih_pick does for its oracle seed, because
            # a fixed count fails intermittently. The print reports how many warmups were needed, so
            # a rising trend is visible instead of silently absorbed.
            c2b = rmse = None
            for attempt in range(5):
                session.step_n(6)
                try:
                    c2b, rmse = camera_to_base_ground_truth(cam)
                    break
                except RuntimeError as exc:
                    if attempt == 4:
                        raise
                    print(f"{name}: depth not ready, warming up again "
                          f"({attempt + 1}/5): {exc}", flush=True)
            assert rmse is not None  # narrowed by the raise above
            print(f"{name} @ {tuple(ccfg.position_mm)} -> aim {tuple(ccfg.mount_aim_mm)} "
                  f"(cam->base GT fit rmse={rmse:.1f} mm, warmup {(attempt + 1) * 6} steps)", flush=True)
        cameras[name] = (
            MultiObjectGroundTruthPerceptionSource(camera=cam, targets=object_specs, session=session, warmup_steps=6),
            c2b,
        )

    # The orchestrator seam that accepts the fused multi-view scene cloud in BASE for the collision
    # filter. Off by default and set only under --scene-collision. getattr keeps this safe if the
    # runtime shape changes.
    orch = getattr(getattr(service, "runtime", None), "orchestrator", None)
    # The swept-volume approach-path validator consumes the same cloud: it rejects or avoids a grasp
    # whose open-gripper descent would sweep through a neighbour the single synthesis view cannot see.
    # It reads result.metadata["scene_points_mm"], so the fused cloud has to be injected and
    # approach_validate implies scene_collision. Wired for the orchestrator's current mode_label, so
    # the mode does not change and there are no side effects.
    if approach_validate:
        scene_collision = True
        if orch is not None:
            from src.robot.grasping.motion.trajectory_safety import ApproachPathPolicy

            orch.approach_path_policy = ApproachPathPolicy(
                standoff_mm=50.0, retreat_mm=100.0, num_approach_samples=6, num_retreat_samples=4,
                collision_margin_mm=20.0,
            )
            orch._approach_path_modes = frozenset({orch.mode_label})
            print(f"Lever B / G12 approach validator wired (mode_label={orch.mode_label!r}, margin=20mm)", flush=True)
    if scene_collision and orch is None:
        print("WARN: --scene-collision set but no orchestrator on the service runtime -> Lever B inert", flush=True)

    # Multi-camera geometry fusion through the supported protocol, not another ad-hoc camera loop.
    # The scene-collision path above fuses the neighbour clouds for the collision filter; this fuses
    # each candidate object's own surface for the grasp generator. That is where the ranking gain
    # sits: a candidate is accepted on its closing axis, the closing axis comes from the silhouette,
    # and one view sees one side while an antipodal grasp needs two.
    #
    # This runner already holds a {name: (source, CAMERA->BASE)} map, which is the shape
    # ``MappedCameraRig`` and ``StaticCameraToBaseResolver`` adapt to the protocol that
    # ``from_robot_config`` wires from config through ``apply_orchestrator_overlays``. Going through
    # the same seam is what makes a measurement here a statement about the feature rather than about
    # this runner. The cameras are fixed and the sim scene is settled, so the adapter's
    # sequential-trigger caveat does not bite here.
    if fuse_geometry:
        if orch is None:
            print("WARN: --fuse-geometry set but no orchestrator on the service runtime -> inert", flush=True)
        else:
            from src.config.schema.robot.grasping_schema import FusionGeometryConfig
            from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
            from src.robot.grasping.types.perception import MappedCameraRig

            orch.multi_camera_perception = MappedCameraRig(
                {name: source for name, (source, _c2b) in cameras.items()}
            )
            orch.camera_frame_resolvers = {
                name: StaticCameraToBaseResolver(c2b) for name, (_src, c2b) in cameras.items()
            }
            orch.fusion_geometry_config = FusionGeometryConfig(
                enabled=True, neighbour_scene_enabled=fuse_neighbours
            )
            print(f"geometry fusion wired over {sorted(cameras)} "
                  f"(metric=overlap, min_score=0.30, "
                  f"neighbour_scene={'on' if fuse_neighbours else 'off'})", flush=True)
    elif fuse_neighbours:
        # The neighbour cloud is built from the same cross-camera association as the target cloud,
        # so it cannot be produced without it. Say so rather than accepting a flag that does nothing.
        print("WARN: --fuse-neighbours needs --fuse-geometry -> inert", flush=True)

    # Measure the best grasp-synthesis view. Move the arm to home, where it occludes the overhead, which
    # is the realistic in-workspace case, then synthesize a grasp from each camera and report which view
    # wins.
    if compare_views:
        arm.move_joint(JointPositions(home_q))
        session.step_n(20)
        cmp, _grasps = compare_grasp_views(cameras, target_idx, cfg)
        print("\n=== H5.1 Lever A: best grasp-synthesis view (arm at home) ===", flush=True)
        for name, r in cmp.items():
            print(f"  {name:>9}: visible_px={r['px']:>7d}  grasp_success={int(r['success'])}  score={r['score']}", flush=True)
        best = _best_view(cmp)
        print(f"  -> BEST grasp-synthesis view = {best!r} "
              f"({'overhead occluded by the arm -> an oblique carries the synthesis' if best and best != 'overhead' else 'top-down view wins'})", flush=True)
        print("============================================================\n", flush=True)
        return GateResult(runs=0, passed=0, results=[{"view_comparison": cmp, "best_view": best}], gate_passed=best is not None)

    obj = SingleRigidPrim(target_prim)
    home_pos, home_quat = obj.get_world_pose()
    # With no target label the gate must not watch one prim. `--clear-any` lets the loop take whichever
    # object it judges best, so measuring only `target_prim` would read about 0 mm on a perfectly good
    # pick of a neighbour and report a failure that did not happen. Watch every object, take the
    # largest lift and name which one moved; that name is the ordering measurement. Empty when a label
    # is set, leaving every other path unchanged.
    _all_objs = (
        [(lbl, SingleRigidPrim(prim)) for prim, lbl in handles.object_specs]
        if clear_any else []
    )

    # Perception-budget collect: emit a non-degenerate stop/continue corpus for the LinUCB policy. Per
    # episode, measure occlusion as the stop view's visible pixels against the best-viewing camera's,
    # decide stop or continue, localize accordingly, drive the same wrist-refine pick, and log the
    # record with the four perception fields the trainer needs (views_seen, candidate_count,
    # expected_occlusion_ratio, fusion_latency_ms) plus cycle_time_s and the sim ground-truth outcome.
    # Continue means spending an extra view: 'fusion' fuses every active camera, 'reperceive' fuses the
    # two obliques. Gated on --collect-perception, so with the flag off this path is inert and the run
    # is byte-identical.
    if collect_perception:
        from src.willy_sim.harness.instrumentation import write_pick_artifacts
        from src.willy_sim.run_dense_pick import _failure_taxonomy_from_sim, _scene_family_id

        obliques = [n for n in ("oblique_L", "oblique_R") if n in cameras]
        if "overhead" not in cameras or not obliques:
            raise SystemExit("--collect-perception needs overhead + >=1 oblique camera (use --mode eth3)")
        calc = getattr(getattr(getattr(service, "runtime", None), "orchestrator", None), "calculator", None)
        # The arm at home is the dominant top-down occluder and the overhead sees almost nothing, so the
        # obliques are the occlusion resolvers and the perception-budget tradeoff maps onto them. Stop
        # commits on one oblique's localization, spending a single view; it still picks, because the
        # wrist refine corrects a rough centroid, but one biased side view misses more often when the
        # target is partly hidden from it. Continue spends the extra view: 'fusion' fuses every active
        # camera and 'reperceive' fuses both obliques. Continue always stamps fusion_latency, and that
        # is what carries the continue label. Stopping on the overhead instead would localize nothing
        # under the arm and give no budget contrast at all.
        stop_cams = {obliques[0]: cameras[obliques[0]]}
        if perception_continue_mode == "reperceive":
            continue_cams = {n: cameras[n] for n in obliques}
        else:
            continue_cams = dict(cameras)
        object_labels = [lbl for (_p, lbl) in object_specs]
        run_id = f"mvpercep-{int(time.time())}"
        print(f"=== PERCEPTION-BUDGET COLLECT (continue_mode={perception_continue_mode!r}: STOP={list(stop_cams)} "
              f"vs CONTINUE={list(continue_cams)}) seed={seed} runs={runs} ===", flush=True)
        pb_results = []
        for i in range(runs):
            try:
                gripper.open()
                arm.move_joint(JointPositions(home_q))
                # Seeded per-episode target jitter so occlusion (hence the state cell) varies across episodes.
                jx = (_perception_seeded_uniform(seed, i, salt="jx") - 0.5) * 2.0 * perception_jitter_mm
                jy = (_perception_seeded_uniform(seed, i, salt="jy") - 0.5) * 2.0 * perception_jitter_mm
                new_pos = np.asarray(home_pos, dtype=np.float64).copy()
                new_pos[0] += jx / 1000.0
                new_pos[1] += jy / 1000.0
                obj.set_world_pose(position=new_pos, orientation=np.asarray(home_quat))
                try:
                    obj.set_linear_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                except Exception:  # noqa: BLE001
                    pass
                session.step_n(30)

                # 1) measure occlusion: how much the stop single-oblique view loses vs the best-viewing camera
                #    (a neighbour / the target's pose hides it from that one side). A genuine, measured [0,1]
                #    signal that varies with the per-episode jitter (which oblique sees the target best shifts).
                _cocc, per_occ = multi_view_localize(cameras, target_idx)
                stop_px = float(per_occ.get(obliques[0], {}).get("target_px", 0.0))
                best_px = max((float(v.get("target_px", 0.0)) for v in per_occ.values()), default=0.0)
                occ = 1.0 - (stop_px / best_px) if best_px > 0.0 else 1.0
                occ = 0.0 if occ < 0.0 else (1.0 if occ > 1.0 else occ)

                # 2) the collect policy's stop/continue decision (occlusion-correlated + seeded exploration).
                action, cont_p = perception_collect_decision(occlusion_ratio=occ, seed=seed, episode=i)

                # 3) localize per the action, timing the continue fusion (the extra-view cost).
                t_local0 = time.perf_counter()
                if action == "continue":
                    fused_c, per = multi_view_localize(continue_cams, target_idx)
                    fusion_latency_ms: "float | None" = (time.perf_counter() - t_local0) * 1000.0
                    views_seen = max(sum(1 for v in per.values() if float(v.get("target_px", 0.0)) > 0.0), 2)
                else:
                    fused_c, per = multi_view_localize(stop_cams, target_idx)
                    fusion_latency_ms = None
                    views_seen = 1

                if fused_c is None:
                    # No selected view saw the target: a failed episode, not an error. Typically the
                    # stop view was occluded.
                    outcome_str, pick_outcome, sim_lifted, lift_mm, candidate_count = "no_perception", "no_perception", False, 0.0, 0
                    report = None
                else:
                    arm.move(topdown_view_pose(fused_c, p_cam_tool, view_height_mm))
                    z0 = float(np.asarray(obj.get_world_pose()[0])[2])
                    report = service.pick()
                    z1 = float(np.asarray(obj.get_world_pose()[0])[2])
                    outcome_str = str(getattr(getattr(report, "outcome", None), "value", None))
                    _pr = getattr(report, "pick_report", None)
                    pick_outcome = str(getattr(getattr(_pr, "outcome", None), "value", None))
                    lift_mm = (z1 - z0) * 1000.0
                    sim_lifted = bool(lift_mm >= gate.lift_threshold_mm)
                    candidate_count = int((getattr(calc, "last_telemetry", None) or {}).get("final", 0))
                cycle_time_s = time.perf_counter() - t_local0

                extra: dict[str, Any] = {
                    "sim_runner": "multiview_perception",
                    "views_seen": int(views_seen),
                    "candidate_count": int(candidate_count),
                    "expected_occlusion_ratio": round(float(occ), 4),
                    "perception_cost": float(views_seen),  # OPE reward: more views cost more (the budget)
                    "cycle_time_s": float(cycle_time_s),
                    "sim_lift_mm": round(float(lift_mm), 2),
                    "sim_lifted": bool(sim_lifted),
                    "object_set": object_labels,
                    "scene_family_id": _scene_family_id(kind="multiview", target_label=str(target_label)),
                    "scene_id": f"{run_id}-ep{i:04d}",
                    "attempt_index": i,
                    "timestamp": float(i),
                    "perception_action_collect": action,  # the collect policy's own choice (audit)
                    "perception_continue_prob": round(float(cont_p), 4),
                    "camera_pose_hash": hashlib.sha256(f"{target_label}|{action}|{run_id}".encode()).hexdigest()[:16],
                }
                if fusion_latency_ms is not None:
                    extra["fusion_latency_ms"] = round(float(fusion_latency_ms), 3)
                # Demote a pipeline 'succeeded' that did not lift to a real failure token, so that
                # derive_outcome_class, and the OPE reward built on it, reflects the measured lift
                # rather than the pipeline's self-report.
                _ftc = _failure_taxonomy_from_sim(outcome_str, str(pick_outcome), bool(sim_lifted))
                if _ftc is not None:
                    extra["failure_taxonomy_class"] = _ftc

                passed = sim_lifted
                pb_results.append({"run": i, "action": action, "occ": round(occ, 3), "cont_p": round(cont_p, 3),
                                "views": views_seen, "cands": candidate_count, "lift_mm": round(lift_mm, 1),
                                "succeeded": sim_lifted, "passed": passed, "outcome": outcome_str})
                print(f"RUN {i}: occ={occ:.3f} p_cont={cont_p:.3f} -> {action.upper():8s} views={views_seen} "
                      f"cands={candidate_count} lift={lift_mm:.1f} lifted={sim_lifted} outcome={outcome_str}", flush=True)
                if report is not None and record_log:
                    write_pick_artifacts(report, attempt_id=f"{run_id}-{i:04d}", record_log=record_log, extra=extra)
            except Exception as exc:  # noqa: BLE001 (a safety abort / pick failure = a failed episode, not a crash)
                print(f"RUN {i}: ABORTED ({type(exc).__name__}: {exc})", flush=True)
                pb_results.append({"run": i, "succeeded": False, "passed": False, "aborted": str(exc)})

        n_pass = sum(1 for r in pb_results if r.get("passed"))
        n_stop = sum(1 for r in pb_results if r.get("action") == "stop")
        n_cont = sum(1 for r in pb_results if r.get("action") == "continue")
        gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
        print(f"\n=== PERCEPTION-BUDGET COLLECT: {n_pass}/{runs} lifted | STOP={n_stop} CONTINUE={n_cont} "
              f"-> record_log={record_log!r} ===", flush=True)
        return GateResult(runs=runs, passed=n_pass, results=pb_results, gate_passed=gate_ok, target=target_label)

    # Read-only discriminance check: whether a re-perceive from the always-watching obliques tracks a
    # target that shifted after the first localize, and whether an open-loop grasp carrying the stale
    # first centroid would miss. Per run: localize C0 with the arm at home, occluding the overhead as it
    # does in the workspace, disturb the target by a known delta, settle, and re-localize C1 from the
    # same obliques. Report the stale error ||C0 - true_new||, which is what open-loop carries, against
    # the closed-loop residual ||C1 - true_new||. A run passes when the re-perceive recovers, with a
    # residual below 5 mm, and the disturbance defeats open-loop, with a stale error above 12 mm, the
    # parallel jaw's half tolerance. Nothing is grasped: only the target pose is mutated.
    if probe_refine:
        arm.move_joint(JointPositions(home_q))
        session.step_n(20)
        home_xy_mm = np.asarray(home_pos, dtype=np.float64)[:2] * 1000.0
        results: list[dict[str, Any]] = []
        for i in range(runs):
            dx, dy = DISTURBANCES[i % len(DISTURBANCES)]
            # reset to home, settle, initial localize C0
            obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
            try:
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
            session.step_n(25)
            c0, _ = multi_view_localize(cameras, target_idx)
            # disturb: shift the target by (dx, dy) mm in BASE, settle
            new_pos = np.asarray(home_pos, dtype=np.float64).copy()
            new_pos[0] += dx / 1000.0
            new_pos[1] += dy / 1000.0
            obj.set_world_pose(position=new_pos, orientation=np.asarray(home_quat))
            try:
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
            session.step_n(25)
            true_new = np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0
            c1, _ = multi_view_localize(cameras, target_idx)
            if c0 is None or c1 is None:
                print(f"RUN {i}: localize failed (c0={c0 is not None}, c1={c1 is not None}) -> fail", flush=True)
                results.append({"run": i, "passed": False})
                continue
            detected = float(np.linalg.norm((c1 - c0)[:2]))
            true_shift = float(np.linalg.norm(true_new[:2] - home_xy_mm))
            stale_err = float(np.linalg.norm((c0 - true_new)[:2]))
            cl_resid = float(np.linalg.norm((c1 - true_new)[:2]))
            recovers = cl_resid < 5.0
            defeats_open = stale_err > 12.0
            passed = recovers and defeats_open
            results.append({"run": i, "passed": passed, "detected_shift_mm": round(detected, 1),
                            "true_shift_mm": round(true_shift, 1), "stale_err_mm": round(stale_err, 1),
                            "cl_resid_mm": round(cl_resid, 1)})
            print(f"RUN {i}: disturb=({dx:+.0f},{dy:+.0f})  detected_shift={detected:.1f} (true {true_shift:.1f})  "
                  f"OPEN-LOOP stale_err={stale_err:.1f}mm  CLOSED-LOOP resid={cl_resid:.1f}mm  "
                  f"-> recovers={recovers} defeats_open={defeats_open}", flush=True)
        n_pass = sum(1 for r in results if r["passed"])
        gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
        print(f"\n=== H5.2 probe-refine ({mode}): {n_pass}/{runs} (re-perceive recovers a disturbance that "
              f"defeats open-loop) -> gate_passed={gate_ok} ===", flush=True)
        return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, target=target_label)

    # Execute the best grasp-synthesis view, rather than the wrist-refine path, and measure the lift.
    # Per run: reset, drive the arm to home, where it occludes the overhead as it does in the workspace,
    # synthesize a grasp from every camera, take the best-scoring view's BASE grasp, optionally
    # re-orient it top-down to the position, width and yaw an oblique view carries, then run it through
    # the orchestrator's GraspExecutionPolicy, which moves via cuRobo+Coal through arm.move with the
    # standoff, retreat and close settings build_service supplies. An oblique-synthesized grasp is
    # executable where the overhead-only path, occluded and yielding no candidates, produces nothing.
    if execute_best_view:
        policy = getattr(orch, "policy", None)
        if policy is None:
            raise SystemExit("execute-best-view needs the orchestrator's GraspExecutionPolicy (none found)")
        results = []
        for i in range(runs):
            try:
                gripper.open()
                arm.move_joint(JointPositions(home_q))
                obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
                try:
                    obj.set_linear_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                except Exception:  # noqa: BLE001
                    pass
                session.step_n(30)

                cmp, grasps = compare_grasp_views(cameras, target_idx, cfg)
                best = _best_view(cmp)
                if best is None or grasps.get(best) is None:
                    print(f"RUN {i}: NO view synthesized a grasp (overhead occluded, obliques empty?) -> fail  "
                          f"{ {k: v['px'] for k, v in cmp.items()} }", flush=True)
                    results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False, "best_view": best})
                    continue
                grasp = grasps[best]
                assert grasp is not None  # for mypy; guarded above
                approach_note = "raw oblique tilt"
                if reorient_topdown:
                    grasp = _reorient_topdown(grasp)
                    approach_note = "re-oriented top-down"
                pos = np.round(np.asarray(grasp.position), 1)
                appr = np.round(np.asarray(grasp.approach), 2)
                print(f"RUN {i}: best view={best!r} (score={cmp[best]['score']}) -> grasp pos={pos} "
                      f"approach={appr} width={grasp.grip_width_mm:.1f} [{approach_note}]", flush=True)

                z0 = float(np.asarray(obj.get_world_pose()[0])[2])
                report = policy.execute(grasp)
                z1 = float(np.asarray(obj.get_world_pose()[0])[2])
                outcome_str = str(getattr(getattr(report, "outcome", None), "value", "?"))
                lift_mm = (z1 - z0) * 1000.0
                passed = bool(lift_mm >= gate.lift_threshold_mm)
                results.append({"run": i, "succeeded": outcome_str == "executed", "lift_mm": round(lift_mm, 1),
                                "passed": passed, "best_view": best, "outcome": outcome_str})
                print(f"RUN {i}: policy outcome={outcome_str} lift_mm={lift_mm:.1f} passed(lifted)={passed}", flush=True)
            except Exception as exc:  # noqa: BLE001 (a safety abort / motion failure = a failed run)
                print(f"RUN {i}: ABORTED ({type(exc).__name__}: {exc})", flush=True)
                results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False, "aborted": str(exc)})

        n_pass = sum(1 for r in results if r["passed"])
        gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
        print(f"\n=== Lever A-2 EXECUTE best-view ({mode}, {'top-down' if reorient_topdown else 'raw-tilt'}): "
              f"{n_pass}/{runs} lifted (>= {gate.lift_threshold_mm:.0f} mm) -> gate_passed={gate_ok} ===", flush=True)
        return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, target=target_label)

    # 3D multi-view grasp synthesis: not localization and not collision, but the grasp itself
    # synthesized from the fused 3D geometry of every camera. ``build_fused_target_cloud`` feeds
    # ``synthesize_grasp_from_cloud``, which runs PCA on the BASE XY footprint, closes across the minor
    # axis and anchors at the fused centroid. Each run also reports every single view's footprint
    # centroid against the fused one, which is the geometry-fusion accuracy: symmetric obliques each
    # see one biased side and fusing cancels the bias. The fused grasp is then executed through the
    # policy and the lift is measured.
    if synthesize_3d:
        from src.robot.grasping.multiview.synthesis import (
            fused_grasp_to_point,
            synthesize_grasp_from_cloud,
        )

        policy = getattr(orch, "policy", None)
        if policy is None:
            raise SystemExit("synthesize-3d needs the orchestrator's GraspExecutionPolicy (none found)")
        # The fused footprint of a symmetric cube is near-square, so the PCA minor axis is an
        # essentially free yaw that can land near the UR5e's unreachable top-down orientation, where
        # cuRobo intermittently fails to plan the descent. For a symmetric object the closing yaw is a
        # free choice, so it is snapped to base X, which is reachable, the same thing the dense pick
        # does with align_closing_to_base_x. The fused centroid accuracy is untouched: only the free
        # yaw moves. This holds for symmetric objects only. A fixed-axis object such as an elongated
        # box must keep the synthesized axis, and the fused footprint already tells the two apart.
        policy.align_closing_to_base_x = True
        grip = cfg.robot.gripper

        def _synth_fused() -> "object | None":
            per = build_fused_target_cloud(cameras, target_idx)
            if not per:
                return None
            return synthesize_grasp_from_cloud(
                np.vstack(list(per.values())), grasp_depth_mm=3.0,
                min_width_mm=grip.min_width_mm, max_width_mm=grip.max_width_mm,
            )

        # cuRobo cold-start warmup: the first trajopt plan after a build intermittently fails to plan.
        # One throwaway pick and reset warms the planner before the measured loop, so run 0 is not
        # penalised by the cold start. The synthesis is deterministic, so the warmup grasp is the same
        # grasp the measured run uses.
        try:
            gripper.open()
            arm.move_joint(JointPositions(home_q))
            obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
            session.step_n(20)
            _wg = _synth_fused()
            if _wg is not None:
                policy.execute(fused_grasp_to_point(_wg))  # type: ignore[arg-type]
            gripper.open()
            arm.move_joint(JointPositions(home_q))
            obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
            session.step_n(20)
        except Exception as exc:  # noqa: BLE001 (warmup is best-effort)
            print(f"warmup pick skipped ({type(exc).__name__}: {exc})", flush=True)

        results = []
        for i in range(runs):
            try:
                gripper.open()
                arm.move_joint(JointPositions(home_q))
                obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
                try:
                    obj.set_linear_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                except Exception:  # noqa: BLE001
                    pass
                session.step_n(30)

                per_cam = build_fused_target_cloud(cameras, target_idx)
                if not per_cam:
                    print(f"RUN {i}: NO camera saw the target cloud -> fail", flush=True)
                    results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False})
                    continue
                true_c = np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0
                fused_cloud = np.vstack(list(per_cam.values()))
                # A grasp_depth of 3 mm puts the anchor just below the cloud's top surface, at the
                # near-top grasp z of about 37 mm where the eye-in-hand and M1 picks grasp. The
                # default of 8 mm drives the interpolated descent closer to the table, and cuRobo
                # then intermittently fails to plan the lowest approach waypoint.
                fg = synthesize_grasp_from_cloud(
                    fused_cloud, grasp_depth_mm=3.0,
                    min_width_mm=grip.min_width_mm, max_width_mm=grip.max_width_mm,
                )
                if fg is None:
                    print(f"RUN {i}: fused cloud too small to synthesize -> fail", flush=True)
                    results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False})
                    continue

                # Geometry-fusion accuracy: each single view's centroid XY error vs the fused one (vs truth).
                per_err: dict[str, float] = {}
                for name, view_cloud in per_cam.items():
                    sg = synthesize_grasp_from_cloud(view_cloud, min_width_mm=grip.min_width_mm, max_width_mm=grip.max_width_mm)
                    if sg is not None:
                        per_err[name] = float(np.linalg.norm((sg.position_mm - true_c)[:2]))
                fused_err = float(np.linalg.norm((fg.position_mm - true_c)[:2]))
                best_single = min(per_err.values()) if per_err else float("nan")
                print(f"RUN {i}: 3D synth from {list(per_cam.keys())}  footprint={np.round(fg.footprint_mm, 1)} "
                      f"width={fg.width_mm:.1f}", flush=True)
                print(f"RUN {i}: centroid XY err: per-view {({k: round(v, 1) for k, v in per_err.items()})} "
                      f"-> FUSED {fused_err:.1f} mm (best single {best_single:.1f} mm)", flush=True)

                grasp = fused_grasp_to_point(fg)
                z0 = float(np.asarray(obj.get_world_pose()[0])[2])
                report = policy.execute(grasp)
                z1 = float(np.asarray(obj.get_world_pose()[0])[2])
                outcome_str = str(getattr(getattr(report, "outcome", None), "value", "?"))
                lift_mm = (z1 - z0) * 1000.0
                passed = bool(lift_mm >= gate.lift_threshold_mm)
                results.append({"run": i, "succeeded": outcome_str == "executed", "lift_mm": round(lift_mm, 1),
                                "passed": passed, "fused_err_mm": round(fused_err, 1),
                                "best_single_err_mm": round(best_single, 1), "outcome": outcome_str})
                print(f"RUN {i}: policy outcome={outcome_str} lift_mm={lift_mm:.1f} passed(lifted)={passed}", flush=True)
            except Exception as exc:  # noqa: BLE001 (a safety abort / motion failure = a failed run)
                print(f"RUN {i}: ABORTED ({type(exc).__name__}: {exc})", flush=True)
                results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False, "aborted": str(exc)})

        n_pass = sum(1 for r in results if r["passed"])
        gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
        print(f"\n=== Lever C 3D-synthesis ({mode}): {n_pass}/{runs} lifted (>= {gate.lift_threshold_mm:.0f} mm) "
              f"-> gate_passed={gate_ok} ===", flush=True)
        return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, target=target_label)

    # Closed-loop pick: re-decide the grasp as the arm closes in, recovering from a target disturbance
    # the always-watching obliques catch. Per run: localize, move the wrist over the first guess, at
    # which point the arm occludes the overhead and only the obliques still see, disturb the target,
    # re-perceive from the obliques, re-synthesize the corrected 3D grasp, and execute it. Under
    # --stale-baseline the run grasps the stale first guess instead, which aims at where the object was
    # and misses. Fixed obliques are what make this work: a moving wrist loses the target at the
    # standoff, while the obliques keep watching while the arm moves.
    if closed_loop:
        from src.robot.grasping.multiview.synthesis import (
            fused_grasp_to_point,
            synthesize_grasp_from_cloud,
        )

        policy = getattr(orch, "policy", None)
        if policy is None:
            raise SystemExit("closed-loop needs the orchestrator's GraspExecutionPolicy (none found)")
        policy.align_closing_to_base_x = True  # symmetric cube: snap the free yaw to reachable base X
        grip = cfg.robot.gripper
        # Re-perceive from the obliques only: while the arm closes in it occludes the overhead, wholly
        # or partly, and including the overhead's partial mask drags the fused centroid off the target.
        # The obliques resolve occlusion and stay clean, so the closed-loop perceive uses just them and
        # keeps the accurate fuse whose symmetric bias cancels.
        reperc_cams = {k: v for k, v in cameras.items() if k != "overhead"}
        if not reperc_cams:
            raise SystemExit("closed-loop needs at least one oblique camera (use --mode eth2/eth3)")

        def _synth_at(cloud_by_cam: "dict[str, np.ndarray]") -> "object | None":
            if not cloud_by_cam:
                return None
            return synthesize_grasp_from_cloud(
                np.vstack(list(cloud_by_cam.values())), grasp_depth_mm=3.0,
                min_width_mm=grip.min_width_mm, max_width_mm=grip.max_width_mm,
            )

        mode_label = "STALE(open-loop)" if stale_baseline else "CORRECTED(closed-loop)"
        print(f"=== H5.2 closed-loop pick ({mode_label}) ===", flush=True)
        # cuRobo cold-start warmup, one throwaway pick and reset, as in the 3D synthesis path above.
        try:
            gripper.open()
            arm.move_joint(JointPositions(home_q))
            obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
            session.step_n(20)
            _wg = _synth_at(build_fused_target_cloud(reperc_cams, target_idx))
            if _wg is not None:
                policy.execute(fused_grasp_to_point(_wg))  # type: ignore[arg-type]
            gripper.open()
            arm.move_joint(JointPositions(home_q))
            obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
            session.step_n(20)
        except Exception as exc:  # noqa: BLE001
            print(f"warmup pick skipped ({type(exc).__name__}: {exc})", flush=True)

        results = []
        for i in range(runs):
            dx, dy = DISTURBANCES[i % len(DISTURBANCES)]
            try:
                gripper.open()
                arm.move_joint(JointPositions(home_q))
                obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
                try:
                    obj.set_linear_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                except Exception:  # noqa: BLE001
                    pass
                session.step_n(25)

                # 1) initial localize + first-guess grasp (the "guess from afar")
                c0, _ = multi_view_localize(reperc_cams, target_idx)
                fg0 = _synth_at(build_fused_target_cloud(reperc_cams, target_idx))
                if c0 is None or fg0 is None:
                    print(f"RUN {i}: initial localize/synth failed -> fail", flush=True)
                    results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False})
                    continue
                # 2) approach: drive the wrist over the first guess, so the arm now occludes the overhead
                arm.move(topdown_view_pose(c0, p_cam_tool, view_height_mm))
                session.step_n(10)
                # 3) disturb the target mid-approach (a neighbour nudge / settling)
                new_pos = np.asarray(home_pos, dtype=np.float64).copy()
                new_pos[0] += dx / 1000.0
                new_pos[1] += dy / 1000.0
                obj.set_world_pose(position=new_pos, orientation=np.asarray(home_quat))
                try:
                    obj.set_linear_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                except Exception:  # noqa: BLE001
                    pass
                session.step_n(25)
                true_new = np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0

                # 4) stale (open-loop) grasps the first guess; closed-loop re-perceives from the obliques
                #    (which still see past the arm) and re-synthesizes the corrected grasp.
                if stale_baseline:
                    grasp_src = fg0
                else:
                    grasp_src = _synth_at(build_fused_target_cloud(reperc_cams, target_idx))
                    if grasp_src is None:
                        print(f"RUN {i}: re-perceive synth failed -> fail", flush=True)
                        results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False})
                        continue
                grasp = fused_grasp_to_point(grasp_src)  # type: ignore[arg-type]
                target_err = float(np.linalg.norm((np.asarray(grasp.position) - true_new)[:2]))
                print(f"RUN {i}: disturb=({dx:+.0f},{dy:+.0f})  grasp targets XY={np.round(np.asarray(grasp.position)[:2], 1)} "
                      f"(true_new {np.round(true_new[:2], 1)}, target_err {target_err:.1f}mm) [{mode_label}]", flush=True)

                # 5) execute + measure the lift
                z0 = float(np.asarray(obj.get_world_pose()[0])[2])
                report = policy.execute(grasp)
                z1 = float(np.asarray(obj.get_world_pose()[0])[2])
                outcome_str = str(getattr(getattr(report, "outcome", None), "value", "?"))
                lift_mm = (z1 - z0) * 1000.0
                passed = bool(lift_mm >= gate.lift_threshold_mm)
                results.append({"run": i, "succeeded": outcome_str == "executed", "lift_mm": round(lift_mm, 1),
                                "passed": passed, "target_err_mm": round(target_err, 1), "outcome": outcome_str})
                print(f"RUN {i}: outcome={outcome_str} lift_mm={lift_mm:.1f} passed(lifted)={passed}", flush=True)
            except Exception as exc:  # noqa: BLE001 (a safety abort / motion failure = a failed run)
                print(f"RUN {i}: ABORTED ({type(exc).__name__}: {exc})", flush=True)
                results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False, "aborted": str(exc)})

        n_pass = sum(1 for r in results if r["passed"])
        gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
        print(f"\n=== H5.2 closed-loop ({mode_label}, {mode}): {n_pass}/{runs} lifted (>= "
              f"{gate.lift_threshold_mm:.0f} mm) -> gate_passed={gate_ok} ===", flush=True)
        return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, target=target_label)

    results = []
    for i in range(runs):
        # A motion safety abort from the continuous collision-avoidance guard, or any pick failure, is a
        # failed run and not a crashed gate: catch it, record it, continue. The reset goes via home, the
        # reachable pose `run_fused_pick` uses, because the extreme park pose's retreat-to-park path
        # self-collides, with forearm against lfinger below the margin, and the guard correctly aborts
        # it. The obliques localize from home, seeing past the arm.
        try:
            gripper.open()
            arm.move_joint(JointPositions(home_q))  # reachable reset and clean view, as run_fused_pick
            obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
            try:
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
            session.step_n(30)

            fused_c, per = multi_view_localize(cameras, target_idx)
            true_c = np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0
            if fused_c is None:
                print(f"RUN {i}: NO active camera saw the target -> skip  per={per}", flush=True)
                results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False, "localize_err_mm": None})
                continue
            err = float(np.linalg.norm((fused_c - true_c)[:2]))
            # Z is printed next to XY on purpose and is not an error term: the fused centroid is the
            # visible surface while ``true_c`` is the prim's centre, so a correct pair differs by about
            # half the object's height. It is the cheap discriminator for a wrist cloud that sits too
            # low. The obliques resolve CAMERA->BASE from the sim ground truth and the wrist resolves
            # it from the calibrated eye-in-hand artifact, so a fused Z that is low too puts the offset
            # in the frame, while a low wrist cloud alone puts it in the hand-eye calibration.
            print(f"RUN {i}: multi-view localized XY={np.round(fused_c[:2], 1)} (true {np.round(true_c[:2], 1)}, "
                  f"err {err:.1f} mm) | fused surface Z={fused_c[2]:.1f} vs object centre Z={true_c[2]:.1f} "
                  f"from {[k for k, v in per.items() if v['target_px'] > 0]}", flush=True)

            # Fuse the fixed cameras' neighbour geometry in BASE and feed it to the grasp's collision
            # filter, so a grasp colliding with a neighbour the wrist cannot see is rejected.
            if scene_collision and orch is not None:
                cloud = build_fused_scene_cloud(cameras, target_idx)
                orch.external_scene_points_base_mm = cloud
                print(f"RUN {i}: Lever B fused scene cloud = "
                      f"{0 if cloud is None else int(cloud.shape[0])} BASE neighbour pts", flush=True)

            arm.move(topdown_view_pose(fused_c, p_cam_tool, view_height_mm))  # wrist over the fused centroid
            z0 = float(np.asarray(obj.get_world_pose()[0])[2])
            _z0_all = [(lbl, float(np.asarray(o.get_world_pose()[0])[2])) for lbl, o in _all_objs]
            report = service.pick()
            # WILLY_TRACE_CALC=1 turns on the per-pick calculator telemetry, the same environment
            # convention run_dense_pick uses. The true object pose is printed beside it deliberately:
            # the SFE cloud's BASE height span is only meaningful next to where the object actually is.
            #
            # A failed pick always explains itself, whether the variable is set or not; a successful
            # one stays quiet. Behind the variable alone, the reasons that identify a failure are
            # computed and then printed only when somebody set the variable before the run that
            # failed, and a failure is exactly the case where nobody knew to ask.
            _failed = str(getattr(getattr(report, "outcome", None), "value", None)) != "succeeded"
            if _trace_calc() or _failed:
                _calc = getattr(orch, "calculator", None)
                _true = np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0
                if _trace_calc():
                    print(f"[calc] target_true_mm={np.round(_true, 1).tolist()} "
                          f"telemetry={dict(getattr(_calc, 'last_telemetry', None) or {})}", flush=True)
                # The calculator telemetry stops at the surviving-candidate count. What comes after,
                # which candidate was taken and whether the motion or the gripper failed, lives on the
                # pick report. Without it a run that gets past the table check reports only
                # ``execution_failed``, which names a stage and not a cause.
                _pr = getattr(report, "pick_report", None)
                _attempts = getattr(_pr, "attempts", None) or ()
                _reasons = [str(getattr(r, "value", r)) for r in (_attempts[-1].reasons if _attempts else ())]
                _cands = getattr(getattr(_pr, "executed_grasp", None), "candidates", None) or ()
                _grasp = (f"pos={np.round(np.asarray(_cands[0].position), 1).tolist()} "
                          f"approach={np.round(np.asarray(_cands[0].approach), 2).tolist()} "
                          f"width={_cands[0].grip_width_mm:.1f}") if _cands else "none"
                print(f"[pick] outcome={getattr(getattr(_pr, 'outcome', None), 'value', None)} "
                      f"attempts={len(_attempts)} reasons={_reasons} grasp[0]: {_grasp}", flush=True)
                # Why the motion failed, in the detail only the policy report holds. ``PolicyReport``
                # carries a typed ``motion_status``, the driver's own ``motion_message``, the waypoints
                # and the exception, while ``_map_policy_outcome`` collapses the outcome into the
                # single string "execution_failed", so a cell that reports a failed motion otherwise
                # discards the explanation of it. The orchestrator keeps the last policy report, so
                # reading it here touches no contract.
                _pol = getattr(orch, "_last_policy_report", None)
                if _pol is not None:
                    _err = getattr(_pol, "error", None)
                    print(f"[motion] policy_outcome={getattr(getattr(_pol, 'outcome', None), 'value', None)} "
                          f"status={getattr(getattr(_pol, 'motion_status', None), 'value', None)} "
                          f"message={getattr(_pol, 'motion_message', None)!r} "
                          f"waypoints={len(getattr(_pol, 'waypoints', ()) or ())} "
                          f"error={type(_err).__name__ if _err else None}: {_err}", flush=True)
            _print_block_carriers(report)
            z1 = float(np.asarray(obj.get_world_pose()[0])[2])
            outcome = getattr(report, "outcome", None)
            outcome_str = str(getattr(outcome, "value", outcome))
            succeeded = outcome_str == "succeeded"
            lift_mm = (z1 - z0) * 1000.0
            lifted_label = None
            if _all_objs:
                _lifts = [
                    (lbl, (float(np.asarray(o.get_world_pose()[0])[2]) - z_before) * 1000.0)
                    for (lbl, o), (_l0, z_before) in zip(_all_objs, _z0_all)
                ]
                lifted_label, lift_mm = max(_lifts, key=lambda kv: kv[1])
            passed = bool(lift_mm >= gate.lift_threshold_mm)
            row = {"run": i, "succeeded": succeeded, "lift_mm": round(lift_mm, 1),
                   "passed": passed, "localize_err_mm": round(err, 1), "outcome": outcome_str}
            if lifted_label is not None:
                row["lifted"] = lifted_label
            results.append(row)
            print(f"RUN {i}: outcome={outcome_str} lift_mm={lift_mm:.1f} passed(lifted)={passed}"
                  f"{'' if lifted_label is None else f' lifted={lifted_label!r}'}", flush=True)
        except Exception as exc:  # noqa: BLE001 (a safety abort / pick failure = a failed run, not a crash)
            print(f"RUN {i}: ABORTED ({type(exc).__name__}: {exc})", flush=True)
            results.append({"run": i, "succeeded": False, "lift_mm": 0.0, "passed": False, "aborted": str(exc)})

    n_pass = sum(1 for r in results if r["passed"])
    gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
    print(f"\n=== multi-view {mode}: {n_pass}/{runs} lifted (>= {gate.lift_threshold_mm:.0f} mm) -> "
          f"gate_passed={gate_ok} ===", flush=True)
    return GateResult(runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, target=target_label)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-view fixed-camera localize -> EIH refine -> grasp (H5.1 Stage 2).")
    ap.add_argument("--mode", default="eth3", choices=sorted(MODE_CAMERAS), help="active fixed-camera set")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--marker", default="ground_truth")
    ap.add_argument("--prompt", default="a green cube")
    ap.add_argument("--continuous-guard", action="store_true",
                    help="enable the opt-in per-waypoint collision-avoidance guard (static Coal preflight is always on)")
    ap.add_argument("--scene-collision", action="store_true",
                    help="Lever B: feed the fused fixed-camera neighbour geometry to the grasp collision filter")
    ap.add_argument("--fuse-geometry", action="store_true",
                    help="fuse EVERY candidate object's surface across the active fixed cameras and feed it "
                         "to the grasp generator (the protocol the prod path wires from config)")
    ap.add_argument("--boot", default="config", choices=["config", "components"],
                    help="which path builds the service: 'config' = from_robot_config (what a real "
                         "cell runs), 'components' = the historical runner hand-wiring. Keep both: "
                         "they must agree before the default is worth trusting")
    ap.add_argument("--fuse-neighbours", action="store_true",
                    help="with --fuse-geometry: also give the COLLISION filter each object's view of "
                         "everything that is NOT it, fused across the same cameras (finger_collision "
                         "is 60 %% of the remaining mis-ranks once the target cloud is fused)")
    ap.add_argument("--approach-validate", action="store_true",
                    help="Lever B / G12: wire the swept-volume approach validator (implies --scene-collision)")
    ap.add_argument("--grasp-mode", default="easy",
                    choices=["easy", "auto", "closed_loop", "dense_clutter", "dense_autonomous"],
                    help="which GraspMode builds the service (NOT the camera set; that is --mode). "
                         "The mode decides which grasping.* config blocks can fire at all: EASY is in "
                         "the apply_modes of exactly one of them")
    ap.add_argument("--view-height-mm", type=float, default=275.0,
                    help="wrist-camera height above the localized centroid. The validated pick band "
                         "is 250-300; raise it only to fit MORE objects in one frame (--clear-any "
                         "needs at least two graspable at once before ordering has anything to decide)")
    ap.add_argument("--clear-any", action="store_true",
                    help="bin-clearing semantics: no target label, so EVERY object is an executable "
                         "target and the loop chooses which goes first. The only configuration in "
                         "which grasping.ordering can act; with a label it is structurally inert")
    ap.add_argument("--single-mask-perception", action="store_true",
                    help="restore the pre-2026-08-17 single-mask wrist perception (one segmentation = "
                         "the target, other_object_masks empty). The paired baseline for the "
                         "multi-object default; use it to attribute a change to the switch")
    ap.add_argument("--ik-service", action="store_true",
                    help="wire a RobotArmIKService on the live sim arm: candidates that fail IK are "
                         "DISCARDED (read rejected_ik), and grasping.feasibility can finally re-rank. "
                         "two effects in one flag; measure IK alone before adding feasibility")
    ap.add_argument("--config-subpolicies", action="store_true",
                    help="suppress the runner's hand-built decision engine / refiner / verifier so the "
                         "grasping.decision / .closed_loop / .verification config blocks are their only "
                         "source (needs --boot config). Without it the constructor argument wins and "
                         "toggling those blocks measures nothing")
    ap.add_argument("--scene", default="clutter", choices=sorted(SCENES),
                    help="object scene: clutter = 3 separated cubes | close = touching-neighbour value "
                         "| occlude = a tall OCCLUDER beside a default target | tall = the same three "
                         "cubes with a 90 mm TARGET (the table-clearance question)")
    ap.add_argument("--rendered-depth", action="store_true",
                    help="let the camera's OWN rendered depth reach the pick instead of the flat "
                         "ground-truth sheet at the object's centre; the September-shaped path "
                         "(it is what a RealSense delivers). --grasp-lift-mm has no effect with it.")
    ap.add_argument("--grasp-lift-mm", type=float, default=12.0,
                    help="how far ABOVE the object's centre the GT depth oracle reports its surface "
                         "(it reports the centre, not the top face). The default 12 makes a 90 mm "
                         "object look 57 mm tall; set it to half the object height to put the "
                         "perceived surface on the real top face")
    ap.add_argument("--bin", default=None, choices=sorted(BIN_PRESETS),
                    help="industrial bin preset: 'tray' (shallow KLT the jaw reaches) | 'klt' (real full-depth deep bin)")
    ap.add_argument("--compare-views", action="store_true",
                    help="Lever A-1: measure which camera view yields the best grasp synthesis (no execution)")
    ap.add_argument("--execute-best-view", action="store_true",
                    help="Lever A-2: EXECUTE the best grasp-synthesis view + measure the lift")
    ap.add_argument("--reorient-topdown", action="store_true",
                    help="Lever A-2: re-orient the best-view grasp top-down (position+width+yaw, base -Z approach)")
    ap.add_argument("--synthesize-3d", action="store_true",
                    help="Lever C: synthesize the grasp from the FUSED 3D target cloud of all cameras + execute")
    ap.add_argument("--probe-refine", action="store_true",
                    help="H5.2 Phase-0 (read-only): measure whether an oblique re-perceive recovers a target "
                         "disturbance that defeats open-loop")
    ap.add_argument("--closed-loop", action="store_true",
                    help="H5.2: closed-loop pick; re-perceive from the obliques after a disturbance + execute the corrected grasp")
    ap.add_argument("--stale-baseline", action="store_true",
                    help="H5.2: open-loop baseline; grasp the STALE first-guess (ignores the disturbance -> misses)")
    ap.add_argument("--calibrated", action="store_true",
                    help="use the per-camera CALIBRATED extrinsics from grasping.fusion.cameras (each camera's "
                         "eth_<id>.json via build_config_frame_resolvers) instead of --mode + the sim GT oracle")
    ap.add_argument("--collect-perception", action="store_true",
                    help="C: emit a NON-DEGENERATE perception-budget STOP/CONTINUE corpus (needs --mode eth3 + "
                         "--record-log). STOP=overhead-only, CONTINUE=genuine oblique fusion; occlusion-driven "
                         "seeded exploration -> both actions + varied reward for the v5 LinUCB trainer.")
    ap.add_argument("--perception-continue-mode", default="fusion", choices=("fusion", "reperceive"),
                    help="CONTINUE = 'fusion' (both obliques, the strongest signal) | 'reperceive' (a lighter "
                         "overhead + one-oblique 2nd view)")
    ap.add_argument("--perception-jitter-mm", type=float, default=30.0,
                    help="per-episode seeded target jitter (mm) so occlusion (hence the state) varies across episodes")
    ap.add_argument("--record-log", default=None, help="JSONL path to append the collected GraspAttemptRecords to")
    ap.add_argument("--seed", type=int, default=0, help="deterministic collect seed (STOP/CONTINUE + jitter)")
    add_cell_arguments(ap)   # --robot-model / --profile
    args = ap.parse_args()
    result = run_multiview_gate(
        runs=args.runs, mode=args.mode, headless=not args.gui, data_dir=args.data,
        marker=args.marker, prompt=args.prompt, continuous_guard=args.continuous_guard,
        scene_collision=args.scene_collision, approach_validate=args.approach_validate,
        fuse_geometry=args.fuse_geometry, fuse_neighbours=args.fuse_neighbours,
        boot=args.boot, grasp_mode=args.grasp_mode, config_subpolicies=args.config_subpolicies,
        ik_service=args.ik_service, single_mask_perception=args.single_mask_perception,
        clear_any=args.clear_any, view_height_mm=args.view_height_mm,
        scene=args.scene, grasp_lift_mm=args.grasp_lift_mm,
        ground_truth_depth=not args.rendered_depth,
        compare_views=args.compare_views, execute_best_view=args.execute_best_view,
        reorient_topdown=args.reorient_topdown, synthesize_3d=args.synthesize_3d,
        probe_refine=args.probe_refine, closed_loop=args.closed_loop, stale_baseline=args.stale_baseline,
        bin_preset=args.bin, calibrated=args.calibrated, cell_kwargs=cell_profile_kwargs(args),
        collect_perception=args.collect_perception, perception_continue_mode=args.perception_continue_mode,
        perception_jitter_mm=args.perception_jitter_mm, record_log=args.record_log, seed=args.seed,
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
