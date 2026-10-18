"""Multi-view pick in Isaac: fixed-camera localize, wrist eye-in-hand refine, grasp.

Builds on the eye-in-hand pick service ``run_eih_pick.build_service``, which wires
``bootstrap_sim_cell(safety=True)`` plus ``wire_safety_guards`` and the cuRobo+Coal motion planner, so
the fail-closed safety chain is inherited unchanged. What this runner adds is the coarse localize step:
instead of one overhead camera, several fixed eye-to-hand cameras, an overhead plus symmetric obliques,
each localize the prompted target, and their BASE-frame centroids are fused with visibility weights, so
a target the overhead loses to occlusion or to the reaching arm is still localized from whichever
oblique sees it. The fused centroid drives the wrist view pose that ``service.pick()`` works from.

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
from src.robot.core.camera_world import CameraWorldDecline

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Iterable
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

# Each spec is (name, color, position_mm[, size_mm]); the optional 4th element makes the tuple heterogeneous.
SCENES: "dict[str, list[tuple[Any, ...]]]" = {
    "clutter": CLUTTER_OBJECTS, "close": CLOSE_NEIGHBOUR_OBJECTS, "occlude": OCCLUDE_OBJECTS,
    "tall": TALL_TARGET_OBJECTS,
}


def _trace_calc() -> bool:
    """Report whether ``WILLY_TRACE_CALC`` opts this run into per-pick calculator telemetry.

    The variable is read at call time rather than import time, so a value exported mid-session is
    honoured. The same variable drives ``run_dense_pick``'s trace, so one switch covers both runners.
    """
    import os

    return bool(os.environ.get("WILLY_TRACE_CALC", "").strip())


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


def calibrated_camera_resolvers(cameras: "Iterable[str]", *, robot_model: str | None = None,
                                artifact_dir: str | None = None) -> dict[str, Any]:
    """Each fixed sim camera's CAMERA->BASE resolver, from its own eye-to-hand artifact, through the one rig loader.

    Under the sim profile the Isaac cameras are not rigs of the camera section, so no rig declares their
    calibration and ``--calibrated`` names each artifact in code: ``eth_<id>.json`` in ``artifact_dir``,
    which defaults to ``calibration_dir(robot_model)``, the directory ``run_eth_calibrate`` writes for that
    robot model. A camera whose artifact does not load raises ``RigCalibrationError`` naming the file.
    """
    from pathlib import Path

    from src.config.schema.camera.shared_schema import RigExtrinsicsConfig
    from src.calibration.rig_calibration import RigCalibration
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
    from src.willy_sim.calibration.paths import calibration_dir

    root = Path(artifact_dir) if artifact_dir is not None else calibration_dir(robot_model)
    resolvers: dict[str, Any] = {}
    for name in cameras:
        declared = RigExtrinsicsConfig(mounting_mode="eye_to_hand", artifact_path=str(root / f"eth_{name}.json"))
        resolvers[name] = StaticCameraToBaseResolver(
            transform=RigCalibration.from_config(name, declared).camera_to_base())
    return resolvers


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
    *,
    data_dir: str | None = None,
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
                data_dir=data_dir,
                # A simulated cell measures the generator; only a real one asks for its proof (2026-10-09).
                purpose="evaluate",
                camera_matrix=intrinsics,
                max_grip_width_mm=g.max_width_mm, min_grip_width_mm=g.min_width_mm,
            )
        calc = cache[key]
        # Over the bench the cell declares: the calculator plans every grasp in BASE over its support.
        from src.geometry import Frame  # noqa: PLC0415
        from src.robot.grasping.collision import SupportPlane  # noqa: PLC0415
        from src.robot.grasping.collision.support_resolver import bench_height_mm  # noqa: PLC0415

        bench = SupportPlane(offset_mm=bench_height_mm(cfg.robot), frame=Frame.BASE)
        result = calc.compute_result(segs[target_idx], frame.depth_map, camera_to_base=c2b, support_plane=bench)
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
    """Print one line per pick naming what the `grasping.*` blocks the service runs did on it.

    Unconditional, because a null result that cannot be told apart from "the block never ran"
    is not a measurement.

    The line also carries `telemetry['reason']`, which is how a refused run explains itself:
    `mode_not_available` names a stage, while the reason names the cause, for instance
    `per_call_mode_requires_different_sampling_mode`. `execution_failed` swallowing cuRobo's own
    message has the same shape.

    Every field name below is read off the dataclass. A guessed name prints None in every run,
    which is indistinguishable from "nothing fired". Read the dataclass before adding a field here.

    The two halves answer different questions. `cfg_on` is what the service believes is
    enabled, which proves the YAML arrived; the rest is what actually ran.
    """
    tel = dict(getattr(report, "telemetry", None) or {})
    eff = getattr(report, "effective_config", None)
    on: list[str] = []
    if eff is not None:
        try:
            on = sorted(k for k, v in eff.to_dict().items() if k.endswith("_enabled") and v is True)
        except Exception:  # noqa: BLE001 (a trace must never break the pick)
            on = ["<to_dict failed>"]
    # Why the motion failed, per attempt, from the fields PickAttempt carries. `execution_failed`
    # names a stage; these name the cause, which on the occlude scene is the whole question.
    _atts = getattr(getattr(report, "pick_report", None), "attempts", None) or ()
    motion = [
        f"{getattr(a, 'motion_status', None)}/{getattr(a, 'motion_message', None)}"
        for a in _atts
        if getattr(a, "motion_status", None) or getattr(a, "motion_message", None)
    ]
    # Which object the loop took. Only meaningful without a target label: with one there is a single
    # executable segmentation.
    _last = _atts[-1] if _atts else None
    parts = [
        f"cfg_on={on}",
        f"chosen_target={getattr(_last, 'target_index', None)}",
        f"motion={motion}",
        f"reason={tel.get('reason')}",
        # Count and names. "1 action" is not an answer when the safety rule is about which action: a
        # `rescan` after a planner refusal re-perceives and moves nothing. The trail carries the loop's
        # action names, so print them; the count alone cannot tell them apart. A push of the part
        # (`nudge_target`) runs inside the pick attempt and is never one of the trail's actions: it is
        # a row of `recovery_actions`, counted first, wherever its arm left the look, and every push the
        # pick considered is in the telemetry's `pushes`.
        f"recovery_actions={len(getattr(report, 'recovery_actions', ()) or ())}",
        f"recovery_trail={tel.get('recovery_trail_actions')}"
        f"/{tel.get('recovery_trail_terminal_reason')}",
        # The cameras fused into the picked object's cloud, empty when it came from one camera.
        f"fused={list(getattr(report, 'fused_views', ()) or ())}",
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
    # the `tiltcam` profile re-places, so a profile chain here is what lets the wrist-refine pick be
    # measured under the optics the real cell has.
    cell_kwargs: "dict | None" = None,
    marker: str = "ground_truth",
    prompt: str = "a green cube",
    view_height_mm: float = 275.0,
    continuous_guard: bool = False,
    scene_collision: bool = False,
    fuse_geometry: bool = False,
    # Which boot path builds the service. "config", the default, goes through from_robot_config, the
    # path a real cell takes, so a number measured here is a statement about that cell rather than
    # about this runner's hand-wiring. "components" is the runner's own wiring and stays as the
    # comparison: the two must agree before the default is worth trusting.
    boot: str = "config",
    # Which GraspMode the service is built in, which is not the camera set: that is ``mode`` above. The mode
    # decides the sampler and which recovery actions are allowed.
    grasp_mode: str = "easy",
    # Opt-in reachability filter; see run_eih_pick.build_service: it discards unreachable candidates.
    ik_service: bool = False,
    # The paired baseline for the multi-object perception default. True restores the single-mask wrist
    # source, so the multi-object default and its predecessor can be measured as a pair on the same
    # scene and the same seed. A default changed without its own baseline gives a number nobody can
    # attribute afterwards.
    single_mask_perception: bool = False,
    # Bin-clearing semantics, with no label and every object a legitimate target. See
    # `run_eih_pick.build_service(clear_any_object=...)` for why an unlabelled frame is correct here and a bug
    # under a prompt.
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
    probe_refine: bool = False,
    bin_preset: str | None = None,
    calibrated: bool = False,
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
        mode=grasp_mode, ik_service=ik_service,
        multi_object_perception=not single_mask_perception, clear_any_object=clear_any,
        camera_world=CameraWorldDecline("run_multiview_pick: this runner plans without a live camera world"),
    )
    print(f"[grasp-mode] service built in mode={grasp_mode!r}", flush=True)
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.willy_sim.scene import camera_to_base_ground_truth
    from src.robot.core import JointPositions

    sim = cfg.robot.sim
    gate = sim.scene_setup.gate
    session = arm.session

    from src.willy_sim.calibration.paths import calibration_dir

    # run_eih_calibrate writes per robot (logs/calibration/<model>/); the flat directory is the fallback.
    cal = _load_calibrated_eih(marker, cal_dir=calibration_dir(sim.robot_model)) or _load_calibrated_eih(marker)
    if cal is None:
        raise SystemExit(f"multi-view pick requires a calibrated EIH artifact (marker={marker}, quality good/excellent)")
    p_cam_tool = np.asarray(cal.to_matrix(), dtype=np.float64)[:3, 3]

    object_specs = list(handles.object_specs)  # [(prim_path, label), ...] in spec order

    home_q = np.asarray(sim.home_joint_positions, dtype=np.float64)

    # Build the active fixed-camera localizers. The overhead uses a small near clip so a reaching arm
    # is rendered; the obliques are authored world-fixed and their CAMERA->BASE is fit against the sim
    # ground truth. Under --calibrated each active fixed camera's CAMERA->BASE comes from its own
    # persisted eye-to-hand calibration, written by `run_eth_calibrate --camera <id>` and read through
    # the one rig loader, rather than from the sim ground-truth oracle. With the flag off the path is
    # the byte-identical ground-truth path keyed by --mode.
    resolvers: dict[str, Any] = {}
    if calibrated:
        from src.calibration.rig_calibration import RigArtifactMissing, RigCalibrationError

        # --calibrated keeps --mode's camera-set semantics and only swaps the extrinsics source, so
        # `active` still comes from --mode and a calibrated subset such as eth1, eth2 or sides is
        # runnable. Fail closed when a camera the mode needs has no artifact that loads: never silently
        # run a smaller rig than was requested.
        wanted = MODE_CAMERAS[mode]
        try:
            resolvers = calibrated_camera_resolvers(wanted, robot_model=sim.robot_model)
        except RigCalibrationError as exc:
            # The observation alone for a missing file: the real cell's remedy names a rig block and a
            # calibrate command, and these cameras are declared in code and calibrated by run_eth_calibrate.
            said = exc.observation if isinstance(exc, RigArtifactMissing) else str(exc)
            raise SystemExit(
                f"--calibrated --mode {mode} needs each of {list(wanted)} calibrated eye_to_hand: {said.rstrip('.')}. "
                "Calibrate it (run_eth_calibrate --camera <id>), or pick a --mode whose cameras are all "
                "calibrated.") from exc
        active = list(wanted)
        print(f"[calibrated] active cameras, each from its own eth_<id>.json: {active}", flush=True)
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
            from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
            from src.robot.grasping.types.perception import MappedCameraRig

            orch.multi_camera_perception = MappedCameraRig(
                {name: source for name, (source, _c2b) in cameras.items()}
            )
            orch.camera_frame_resolvers = {
                name: StaticCameraToBaseResolver(c2b) for name, (_src, c2b) in cameras.items()
            }
            print(f"geometry fusion wired over {sorted(cameras)} (overlap, min_score 0.30)", flush=True)

    # Measure the best grasp-synthesis view. Move the arm to home, where it occludes the overhead, which
    # is the realistic in-workspace case, then synthesize a grasp from each camera and report which view
    # wins.
    if compare_views:
        arm.move_to_joints(JointPositions(home_q)).raise_unless_executed()
        session.step_n(20)
        cmp, _grasps = compare_grasp_views(cameras, target_idx, cfg, data_dir=data_dir)
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
    # largest lift and name which one moved. Empty when a label is set, leaving every other path unchanged.
    _all_objs = (
        [(lbl, SingleRigidPrim(prim)) for prim, lbl in handles.object_specs]
        if clear_any else []
    )


    # Read-only discriminance check: whether a re-perceive from the always-watching obliques tracks a
    # target that shifted after the first localize, and whether an open-loop grasp carrying the stale
    # first centroid would miss. Per run: localize C0 with the arm at home, occluding the overhead as it
    # does in the workspace, disturb the target by a known delta, settle, and re-localize C1 from the
    # same obliques. Report the stale error ||C0 - true_new||, which is what open-loop carries, against
    # the closed-loop residual ||C1 - true_new||. A run passes when the re-perceive recovers, with a
    # residual below 5 mm, and the disturbance defeats open-loop, with a stale error above 12 mm, the
    # parallel jaw's half tolerance. Nothing is grasped: only the target pose is mutated.
    if probe_refine:
        arm.move_to_joints(JointPositions(home_q)).raise_unless_executed()
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
                arm.move_to_joints(JointPositions(home_q)).raise_unless_executed()
                obj.set_world_pose(position=np.asarray(home_pos), orientation=np.asarray(home_quat))
                try:
                    obj.set_linear_velocity(np.zeros(3))
                    obj.set_angular_velocity(np.zeros(3))
                except Exception:  # noqa: BLE001
                    pass
                session.step_n(30)

                cmp, grasps = compare_grasp_views(cameras, target_idx, cfg, data_dir=data_dir)
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

    # Two levers left here on 2026-09-29 with `multiview/synthesis.py`, the top-down grasp from the fused
    # footprint they both executed: `--synthesize-3d` (Lever C, that grasp from every camera's cloud) and
    # `--closed-loop` with its `--stale-baseline` (H5.2, the same grasp re-synthesized from the obliques
    # after a disturbance). A fused cell's grasp comes from the generator fed the fused cloud
    # (`--fuse-geometry`), and `--probe-refine` above still measures the re-perceive without a grasp.

    results = []
    for i in range(runs):
        # A motion safety abort from the continuous collision-avoidance guard, or any pick failure, is a
        # failed run and not a crashed gate: catch it, record it, continue. The reset goes via home, the
        # reachable pose `run_fused_pick` uses, because the extreme park pose's retreat-to-park path
        # self-collides, with forearm against lfinger below the margin, and the guard correctly aborts
        # it. The obliques localize from home, seeing past the arm.
        try:
            gripper.open()
            arm.move_to_joints(JointPositions(home_q)).raise_unless_executed()  # reachable reset and clean view, as run_fused_pick
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
    ap.add_argument("--grasp-mode", default="easy",
                    choices=["easy", "auto", "dense_clutter"],
                    help="which GraspMode builds the service (NOT the camera set; that is --mode). "
                         "The mode decides the sampler and the recovery actions allowed")
    ap.add_argument("--view-height-mm", type=float, default=275.0,
                    help="wrist-camera height above the localized centroid. The validated pick band "
                         "is 250-300; raise it only to fit MORE objects in one frame")
    ap.add_argument("--clear-any", action="store_true",
                    help="bin-clearing semantics: no target label, so EVERY object is an executable "
                         "target and the loop chooses which goes first")
    ap.add_argument("--single-mask-perception", action="store_true",
                    help="restore the pre-2026-08-17 single-mask wrist perception (one segmentation = "
                         "the target, other_object_masks empty). The paired baseline for the "
                         "multi-object default; use it to attribute a change to the switch")
    ap.add_argument("--ik-service", action="store_true",
                    help="wire a RobotArmIKService on the live sim arm: candidates that fail IK are "
                         "DISCARDED (read rejected_ik)")
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
    ap.add_argument("--probe-refine", action="store_true",
                    help="H5.2 Phase-0 (read-only): measure whether an oblique re-perceive recovers a target "
                         "disturbance that defeats open-loop")
    ap.add_argument("--calibrated", action="store_true",
                    help="use each camera's own CALIBRATED extrinsics, the eth_<id>.json run_eth_calibrate wrote "
                         "for this robot model, read through the rig loader, instead of --mode + the sim GT oracle")
    add_cell_arguments(ap)   # --robot-model / --profile
    args = ap.parse_args()
    result = run_multiview_gate(
        runs=args.runs, mode=args.mode, headless=not args.gui, data_dir=args.data,
        marker=args.marker, prompt=args.prompt, continuous_guard=args.continuous_guard,
        scene_collision=args.scene_collision,
        fuse_geometry=args.fuse_geometry,
        boot=args.boot, grasp_mode=args.grasp_mode,
        ik_service=args.ik_service, single_mask_perception=args.single_mask_perception,
        clear_any=args.clear_any, view_height_mm=args.view_height_mm,
        scene=args.scene, grasp_lift_mm=args.grasp_lift_mm,
        ground_truth_depth=not args.rendered_depth,
        compare_views=args.compare_views, execute_best_view=args.execute_best_view,
        reorient_topdown=args.reorient_topdown, probe_refine=args.probe_refine,
        bin_preset=args.bin, calibrated=args.calibrated, cell_kwargs=cell_profile_kwargs(args),
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
