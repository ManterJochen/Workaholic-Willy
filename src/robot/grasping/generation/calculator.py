"""The analytic grasp calculator: a segmentation mask and a depth image in, ranked :class:`GraspPoint` candidates out.

The stack runs in four stages:

1. the policy gates and the masked depth to a metric point cloud,
2. what the camera saw beside the part, as obstacles (``scene_obstacles``),
3. the support-footprint stage: the part's own body over the support it stands on (``body_grasps``), planned in BASE
   so the hand clears the support instead of being filtered for it,
4. the collision, table, workspace and IK filters, and the base-frame candidates.

The stage needs a CAMERA->BASE transform and a BASE support plane; without both a call offers nothing and says why.
The silhouette stage, which anchored grasps on the visible surface and let the filters refuse what did not fit, left
on 2026-10-16 with its dense and geometry-first samplers (the owner's cleanup): at a cell the support-footprint stage
replaced its candidates on every call.

The robot pipelines depend on the public :class:`GraspCalculator` name and its small ``compute(...)`` contract, which
returns :class:`GraspPoint` objects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final, TYPE_CHECKING

import numpy as np

from src.geometry import Frame, Transform
from src.robot.constants import ROBOT_LOG_DIR, ROBOT_LOG_FILE
from src.robot.grasping.collision import (
    REJECTION_KEYS,
    ParallelJawGripperModel,
    GripperGeometryStrategy,
    SupportPlane,
    filter_candidates,
)
from src.robot.grasping.generation._support_footprint_stage import (
    support_footprint_breakdowns,
)
from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps
from src.robot.grasping.generation.footprint_hull import hull_points, mask_hole_xy, mask_outline_xy, unread_pixels
from src.robot.grasping.generation.footprint_rim import footprint_rim
from src.robot.core.gripper import FINGER_OPENING_SLACK_MM
from src.robot.grasping.generation.scene_obstacles import (
    NO_GRASP_SAID,
    SceneObstacleRules,
    SceneObstacles,
    SceneView,
    SeenEnvelope,
    SeenInAnyFrame,
    corridor_seen_in,
    envelope_verdicts,
    named_surface,
    least_part_height_mm,
    scene_obstacle_points,
    why_no_grasp,
)
from src.robot.grasping.generation.hand import HandFloor, SupportFootprintJaw
from src.robot.grasping.generation.support_prism import DEFAULT_FLOOR_MARGIN_MM, reconstruct_support_prism
from src.robot.grasping.contacts import (
    scene_collision_cloud,
)
from src.robot.grasping.geometry import (
    CameraIntrinsics,
    depth_unit_to_mm,
    masked_point_cloud,
    validate_transform,
)
from src.robot.grasping.planning import (
    IKResult,
    IKService,
)
from src.robot.grasping.scoring import (
    GraspScoreBreakdown,
    WorkspaceBox,
)
from src.robot.grasping.visualization import DebugDrawConfig
from src.robot.grasping.types.perception import SegmentationLike
from src.utility.log_cfg import create_logger

from src.robot.grasping.generation._debug_renderer import _DebugRenderer
from src.robot.grasping.generation._camera_geometry import _SharedGeometryUtil
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint, fingers_held_at, outer_first
from src.robot.grasping.generation.mask_analyzer import MaskAnalyzer
from src.robot.grasping.scoring.topology_risk import topology_risk_from_mask
from src.robot.grasping.scoring.occlusion import approach_clearance_mm, occlusion_ratio
from src.robot.grasping.generation.deformables import DeformableClass, DeformableHandlingStrategy
from src.robot.grasping.generation._calculator_validation import validate_calculator_args

if TYPE_CHECKING:
    from src.robot.grasping.planning.multifinger import (
        FingerKinematicSpec,
        MultiContactGrasp,
        MultiContactGraspPlanner,
    )
from src.robot.grasping.scoring.semantic_policy import SemanticPolicy



# ``SegmentationLike`` (the ``.mask`` protocol these accept) is imported from
# ``grasping.types.perception`` above.

# Optional ``SegmentationResult`` fields propagated into
# ``GraspPoint.metadata["segmentation"]``. Every entry is read with
# ``getattr(seg, name, None)``, so a missing attribute is skipped and an
# object carrying only a subset of these fields is still accepted.
_SEGMENTATION_METADATA_FIELDS: tuple[str, ...] = (
    "label",
    "score",
    "bbox_xyxy",
    "derived_bbox_xyxy",
    "mask_area_px",
    "centroid_xy",
    "frame_id",
    "timestamp_utc",
    "inference_time_s",
)


def _segmentation_metadata(segmentation: object) -> dict:
    """Extract a JSON-friendly snapshot of segmentation metadata (empty dict when none present)."""
    snapshot: dict = {}
    for name in _SEGMENTATION_METADATA_FIELDS:
        value = getattr(segmentation, name, None)
        if value is None:
            continue
        snapshot[name] = value
    extra = getattr(segmentation, "metadata", None)
    if isinstance(extra, dict) and extra:
        # Copy so downstream mutation cannot leak back into the
        # SegmentationResult instance held by the segmenter.
        snapshot["sam2_metadata"] = dict(extra)
    return snapshot




#: The voxel the part's own cloud and the neighbours' are thinned to, millimetres: what reads them whole, the support
#: under the part, its heights in the telemetry and the post-hoc filter's scene. The stage reads the part at full
#: resolution (it thins at 3 mm itself, ``reconstruct_support_prism``). The number every cell ran with.
CLOUD_VOXEL_MM: Final[float] = 8.0


def _validate_camera_to_base_transform(
    value: Transform | np.ndarray,
    *,
    name: str = "T_cam_to_base",
) -> np.ndarray:
    """Validate a camera-to-base rigid transform and return a 4x4 matrix."""
    if isinstance(value, Transform):
        if value.from_frame is not Frame.CAMERA or value.to_frame is not Frame.BASE:
            raise ValueError(
                f"{name} must be Transform(CAMERA -> BASE); got "
                f"{value.from_frame.name} -> {value.to_frame.name}"
            )
        return validate_transform(value.to_matrix(), name=name)
    return validate_transform(np.asarray(value, dtype=np.float64), name=name)



class GraspCalculator:
    """Compute ranked parallel-jaw grasp candidates (generated as :class:`GraspPose`, returned as :class:`GraspPoint`)."""

    def __init__(
        self,
        mask_analyzer: MaskAnalyzer | None = None,
        approach_vector_cam: np.ndarray | None = None,
        max_grip_width_mm: float = 150.0,
        min_grip_width_mm: float = 5.0,
        camera_matrix: np.ndarray | None = None,
        max_candidates: int = 12,
        topology_risk_threshold: float | None = None,
        heavy_occlusion_threshold: float | None = None,
        deformable_strategy: DeformableHandlingStrategy | None = None,
        ik_service: "IKService | None" = None,
        gripper_model: GripperGeometryStrategy | None = None,
        # The rim the stage's footprint input loses, millimetres at the part: a look's mask eroded by ceil(rim * fx / z)
        # px for the cloud the stage reads and for nothing else, never more than 30 % of the mask (footprint_rim.py),
        # every mask pixel the stage does not read placed in the part's visual hull (footprint_hull.py,
        # ``_with_the_hull``). A cell's calculator cuts ``footprint_rim.FOOTPRINT_RIM_MM`` (``build_calculator``);
        # 0.0, the default here, reads the mask whole, without a hull.
        support_footprint_rim_mm: float = 0.0,
        # How far above the support a target point still counts as the support, for the stage. None keeps the
        # stage's own ``DEFAULT_FLOOR_MARGIN_MM`` (2.0). A real cell reads its table as high as its hand-eye and depth
        # error, which every cell's camera world holds in its bench band (``perceived.CAMERA_BENCH_BAND_MM``);
        # ``Scene.from_robot_config`` passes the larger of the two.
        support_footprint_floor_margin_mm: float | None = None,
        # What the camera saw beside the part, as obstacles (every cell's, fix plan Track A). A
        # prompt names one part, and on the owner's cell the calculator saw that part alone: it offered grasps whose open
        # fingers stood inside the parts beside the Zollstock, and the guard refused all of them. With rules, every
        # compute reads the depth around the part by the planner world's own rules (see scene_obstacles.py), hands it
        # to the stage and to the post-hoc filter, keeps the open hand off the support solids and the declared boxes the
        # guard holds, and says why a part got no grasp. A pre-filter: the guard judges every motion either way. None,
        # the default, reads the part alone.
        scene_obstacles: "SceneObstacleRules | None" = None,
        # Side approaches (every cell's, the owner's "gleichwertig nach Geometrie"): a lean is
        # ranked by the room it keeps. False, the library default, ranks every grasp as the stage does.
        side_approaches: bool = False,
    ) -> None:
        validate_calculator_args(
            min_grip_width_mm=min_grip_width_mm,
            max_grip_width_mm=max_grip_width_mm,
            max_candidates=max_candidates,
            topology_risk_threshold=topology_risk_threshold,
            heavy_occlusion_threshold=heavy_occlusion_threshold,
            deformable_strategy=deformable_strategy,
        )

        self.analyzer = mask_analyzer or MaskAnalyzer()
        self.logger = create_logger(
            "GraspCalculator", ROBOT_LOG_FILE, log_dir=ROBOT_LOG_DIR
        )
        # The debug-overlay renderer is an internal collaborator (shares the facade's logger so the
        # exception-path message is identical). The facade keeps last_debug_image_png + the rgb guard.
        self._debug_renderer = _DebugRenderer(logger=self.logger)

        approach = np.asarray(
            approach_vector_cam if approach_vector_cam is not None else [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        if approach.shape != (3,):
            raise ValueError(f"approach_vector_cam must be shape (3,), got {approach.shape}")
        norm = float(np.linalg.norm(approach))
        if norm < 1e-12:
            raise ValueError("approach_vector_cam cannot be the zero vector")
        #: The approach :meth:`plan_multifinger` plans along where it is handed none, CAMERA frame.
        self.approach_cam = approach / norm

        self.topology_risk_threshold: float | None = (
            float(topology_risk_threshold)
            if topology_risk_threshold is not None
            else None
        )
        # Optional occlusion-rejection threshold. ``None`` disables the gate
        # (occlusion ratio is still reported on telemetry when relevant); a
        # finite value enables short-circuit rejection with HEAVY_OCCLUSION
        # + ACTIVE_PERCEPTION_RECOMMENDED.
        self.heavy_occlusion_threshold: float | None = (
            float(heavy_occlusion_threshold)
            if heavy_occlusion_threshold is not None
            else None
        )
        # Optional deformable-handling strategy. ``None`` keeps the rigid
        # pipeline unchanged regardless of any ``deformable_class`` argument;
        # setting a strategy engages the gate so non-rigid targets short-circuit
        # with ``DEFORMABLE_ROUTING_REQUIRED``.
        self.deformable_strategy: DeformableHandlingStrategy | None = (
            deformable_strategy
        )
        # Optional default IK reachability oracle (vendor-injected). When set, every compute() that is
        # not handed a per-call ``ik_service`` uses this one, so the live pick path (orchestrator) gets
        # the IK filter (e.g. drop near-singular grasps) without threading a service through every
        # call. ``None`` (default) applies no IK filter.
        self._default_ik_service: "IKService | None" = ik_service
        self.max_grip_mm = float(max_grip_width_mm)
        self.min_grip_mm = float(min_grip_width_mm)
        self.max_candidates = int(max_candidates)
        #: The named parts' surfaces of the frame asked last (:meth:`_named_surface_of`), by the mask's identity.
        self._named_surfaces: dict[int, tuple[Any, Any, tuple[float, bytes], np.ndarray]] = {}
        #: The frame asked last as a scene view (:meth:`_frame_view`), with the depth it was made of and its placing.
        self._held_frame: "tuple[Any, tuple[float, bytes, bytes], SceneView] | None" = None
        # Default gripper collision envelope for this calculator (parallel-jaw or suction). A per-call
        # ``compute(gripper_model=...)`` wins; when both are None the checker falls back to a
        # ParallelJawGripperModel() default.
        self._gripper_model = gripper_model
        if not (np.isfinite(support_footprint_rim_mm) and support_footprint_rim_mm >= 0.0):
            raise ValueError(f"support_footprint_rim_mm must be finite and >= 0, got {support_footprint_rim_mm!r}")
        self._support_footprint_rim_mm = float(support_footprint_rim_mm)
        if support_footprint_floor_margin_mm is not None and not (
                np.isfinite(support_footprint_floor_margin_mm) and support_footprint_floor_margin_mm >= 0.0):
            raise ValueError("support_footprint_floor_margin_mm must be finite and >= 0, got "
                             f"{support_footprint_floor_margin_mm!r}")
        self._support_footprint_floor_margin_mm: float | None = (
            None if support_footprint_floor_margin_mm is None else float(support_footprint_floor_margin_mm))
        if scene_obstacles is not None and not isinstance(scene_obstacles, SceneObstacleRules):
            raise TypeError(f"scene_obstacles takes SceneObstacleRules or None, got {type(scene_obstacles).__name__}")
        self._scene_obstacles: SceneObstacleRules | None = scene_obstacles
        self._side_approaches = bool(side_approaches)
        _camera_matrix = (
            np.asarray(camera_matrix, dtype=np.float64)
            if camera_matrix is not None
            else None
        )
        if _camera_matrix is not None and _camera_matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix must be (3, 3), got {_camera_matrix.shape}")
        # The shared camera-geometry helper owns the calibrated K + the loud-once synthetic-intrinsics
        # latch (the fallback synthesizes focal=width/2, which back-projects to metrically wrong 3D:
        # never trust a BASE-frame metric grasp on it; pass a real camera_matrix on hardware).
        self._geom = _SharedGeometryUtil(camera_matrix=_camera_matrix, logger=self.logger)

        # Most recent compute() telemetry, exposed for ops/debugging.
        # Populated on every compute() call (including early returns).
        self.last_telemetry: dict = {}
        # Failure / recommendation codes from the most recent compute()
        # call. Empty tuple on success. Surfaced through compute_result().
        self.last_failure_reasons: tuple[GraspFailureReason, ...] = ()
        # Structured wrapper of the most recent compute() call. See
        # :meth:`compute_result`.
        self.last_result: GraspResult = GraspResult()
        # Most recent debug PNG bytes (only when ``rgb_image`` was passed
        # to compute()). Returned in-process to callers; the operator
        # console serves it over its overlay socket (api/routers/media.py,
        # api/viewfinder.py).
        self.last_debug_image_png: bytes | None = None
        # What that overlay was drawn from, all but its candidates: what redraw_debug_image draws again over the
        # candidates a caller kept. None where the last compute() drew no overlay.
        self._last_draw: dict[str, Any] | None = None
        # Opt-in switch so the live pick path (``pick_loop``) forwards the
        # perception frame's rgb into ``compute`` and the grasp-point overlay above is rendered
        # per pick. Default False: pick_loop forwards ``rgb_image=None``, which costs no render; a
        # runner flips it on for ``--debug-frames``.
        self.render_debug_images: bool = False
        #: Whether the cell's hand opens only as wide as a grasp needs before it closes (lever G, 2026-10-13: a Robotiq
        #: hand told a width): the part's body is then searched for jaws that open to each span and the room beside
        #: each jaw, and each such grasp carries that opening (``score_components["support_footprint"]["open_width_mm"]``),
        #: which the execution opens to and the guard holds the fingers at. Off, a hand that opens fully.
        self.opens_as_needed: bool = False
        #: Whether the part's bores are searched for grasps from inside (set by ``build_calculator`` for a Robotiq
        #: hand): the fingers come down narrow into a bore and open onto its
        #: wall (``generation.inner_grasps``). Each such grasp ranks after every outer grasp, here as in the stage
        #: (``types.grasp_point.grips_from_inside``). Off, a calculator built without the config's word, searches no
        #: bore, and every result is as before.
        self.inner_grasps: bool = False

    @property
    def camera_matrix(self) -> "np.ndarray | None":
        """The calibrated intrinsics this calculator was built with, or ``None``.

        Read by the orchestrator (``loop/pick_loop.py``) to back-project this view's target cloud for
        the support-plane refinement (``support_resolver.resolve_support_plane``).

        It returns the calibrated K and never the synthesized fallback. ``self._geom.intrinsics()``
        invents ``focal = width / 2`` when no calibration was supplied, which is good enough to rank
        silhouettes and metrically wrong in 3-D; the support plane is a BASE-frame metric quantity, so
        this branch stays down in a cell built without real intrinsics. That is why this reads ``_K``
        directly rather than calling ``intrinsics()``.

        Read-only: a settable attribute could disagree with the K the generator is already
        back-projecting with, and nothing would notice.
        """
        return self._geom._K

    @property
    def support_footprint_rim_mm(self) -> float:
        """The rim the stage's footprint input loses, millimetres at the part (a cell's:
        ``footprint_rim.FOOTPRINT_RIM_MM``), 0.0 for none.

        Read by the pick loop (``loop/pick_loop.py``), which cuts the same rim off the looks' surfaces it fuses and hands
        the result beside the fused cloud (``compute(footprint_points_base_mm=...)``): one number, so a look's own input
        and a fused one lose the same rim. Read-only for the same reason as :attr:`camera_matrix`.
        """
        return self._support_footprint_rim_mm

    def _support_footprint_jaw(
        self, gripper_model: GripperGeometryStrategy | None, min_table_clearance_mm: float,
    ) -> SupportFootprintJaw:
        """The jaw SFE plans with on this path: the per-call envelope, else this calculator's.

        A method rather than an inline call so ``Scene.from_robot_config``, which builds its jaw from
        the tree through ``SupportFootprintJaw.from_robot_config``, can be held to the same jaw on a
        profile; ``tests/robot/grasping/test_a_scene_plans_the_trees_hand.py`` does that on ``hande``.
        """
        return SupportFootprintJaw.from_model(
            gripper_model if isinstance(gripper_model, ParallelJawGripperModel)
            else (self._gripper_model
                  if isinstance(self._gripper_model, ParallelJawGripperModel) else None),
            # ⛔ THE STROKE, WHICH THIS CALL OMITTED UNTIL 2026-09-10. Without these two the jaw took
            # `from_model`'s keyword defaults, 85.0 and 5.0, which are a 2F-85's numbers. MEASURED on
            # the `hande` profile: the pick path planned a 49.99 mm hand as though it opened 85.0,
            # proposed grasps up to 83 mm wide, and the driver then clamped the close to max_width_mm,
            # which maps to count 0 = FULLY OPEN. The close command opened the hand at the grasp point.
            # `scene.py` passed the configured aperture; this path did not, and every OTHER dimension
            # of the jaw came out correct, which is what hid it. (`scene.py` had the mirror image: the
            # stroke right and every finger dimension a 2F-85's, until 2026-09-23.)
            aperture_mm=float(self.max_grip_mm),
            min_width_mm=float(self.min_grip_mm),
            table_clearance_mm=float(min_table_clearance_mm),
        )

    def _without_the_rim(
        self, mask: np.ndarray, sfe_mask: np.ndarray, depth_mm: np.ndarray, intrinsics: CameraIntrinsics,
        telemetry: dict,
    ) -> "tuple[np.ndarray, bool]":
        """``sfe_mask`` less the rim of ``mask`` (:attr:`support_footprint_rim_mm`, ``footprint_rim``), and whether that
        took a point off SFE's input; the rim in pixels and the points it took are stamped
        (``support_footprint_rim_px``, ``support_footprint_rim_points``) and said in the log, every compute.

        A point is a pixel with a depth the cloud keeps (1 mm and over). Where the rim would leave SFE no point the mask
        had, nothing is cut: a footprint is never built from nothing.
        """
        rim = footprint_rim(mask, depth_mm, intrinsics.fx, self._support_footprint_rim_mm)

        def points(pixels: np.ndarray) -> int:
            depth = depth_mm[pixels]
            with np.errstate(invalid="ignore"):
                kept = np.isfinite(depth) & (depth >= 1.0)
            return int(np.count_nonzero(kept))

        cut = sfe_mask & rim.mask
        off = points(sfe_mask & ~rim.mask)
        if off and not points(cut):
            telemetry["support_footprint_rim_px"] = 0
            telemetry["support_footprint_rim_points"] = 0
            self.logger.info("footprint rim: not cut, it would leave SFE no point (%s)", rim.said())
            return sfe_mask, False
        telemetry["support_footprint_rim_px"] = int(rim.px)
        telemetry["support_footprint_rim_points"] = off
        self.logger.info("footprint rim: %s; %d point(s) off SFE's input", rim.said(), off)
        return cut, off > 0

    def _read_for_the_footprint(
        self, mask: np.ndarray, sfe_mask: np.ndarray, depth_mm: np.ndarray, intrinsics: CameraIntrinsics, *,
        fused: bool,
    ) -> np.ndarray:
        """The pixels of ``mask`` whose depth SFE reads for its footprint: ``sfe_mask`` (the mask less its depth steps,
        and here less its rim), less its rim too where the footprint comes fused from the pick loop's looks, which cut
        it there (``fused``), where the depth is one the cloud keeps (1 mm and over)."""
        read = sfe_mask
        if fused:
            read = read & footprint_rim(mask, depth_mm, intrinsics.fx, self._support_footprint_rim_mm).mask
        with np.errstate(invalid="ignore"):
            kept = np.isfinite(depth_mm) & (depth_mm >= 1.0)
        return read & kept

    def _with_the_hull(
        self, target_base: np.ndarray, mask: np.ndarray, read: np.ndarray, intrinsics: CameraIntrinsics,
        camera_to_base: np.ndarray, support_mm: float, telemetry: dict,
    ) -> np.ndarray:
        """``target_base`` and a point for every pixel of ``mask`` this look's footprint does not read, placed in the
        part's visual hull (``footprint_hull``): the rim, the depth steps and the holes, from the mask, the support and
        the top of the prism ``target_base`` gives. How many were placed is stamped (``support_footprint_hull_points``)
        and said in the log, every compute; where ``target_base`` gives no prism, none is placed."""
        floor = (DEFAULT_FLOOR_MARGIN_MM if self._support_footprint_floor_margin_mm is None
                 else float(self._support_footprint_floor_margin_mm))
        prism = reconstruct_support_prism(target_base, support_mm, floor_margin_mm=floor)
        if prism is None:
            telemetry["support_footprint_hull_points"] = 0
            self.logger.info("footprint hull: none placed, the points read give the part no prism")
            return target_base
        matrix = np.array([[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]])
        unread = unread_pixels(mask, read)
        placed = hull_points(unread, mask, matrix, camera_to_base, support_mm, prism.z1, floor_margin_mm=floor)
        telemetry["support_footprint_hull_points"] = int(placed.shape[0])
        self.logger.info("footprint hull: %d of %d mask pixel(s) the footprint does not read placed in the part's "
                         "visual hull (support %.1f mm, top %.1f mm)", int(placed.shape[0]),
                         int(np.count_nonzero(unread)), support_mm, prism.z1)
        return np.vstack([target_base, placed]) if placed.shape[0] else target_base

    def _mask_at_its_top(self, mask: np.ndarray, intrinsics: CameraIntrinsics, camera_to_base: np.ndarray,
                         measured: np.ndarray, support_mm: float) -> "tuple[np.ndarray | None, np.ndarray | None]":
        """The part's mask outline and its holes in BASE at its top (``footprint_hull.mask_outline_xy`` and
        ``mask_hole_xy``), the 98th percentile of the points ``measured`` over the floor the stage cuts; ``None`` for
        both where the mask is empty or no point stands over it."""
        if not np.asarray(mask).any():
            return None, None
        floor = support_mm + (DEFAULT_FLOOR_MARGIN_MM if self._support_footprint_floor_margin_mm is None
                              else float(self._support_footprint_floor_margin_mm))
        points = np.asarray(measured, dtype=np.float64).reshape(-1, 3)
        over = points[np.isfinite(points).all(axis=1) & (points[:, 2] > floor), 2]
        if not over.size:
            return None, None
        matrix = np.array([[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]])
        top = float(np.percentile(over, 98.0))
        return (mask_outline_xy(mask, matrix, camera_to_base, top), mask_hole_xy(mask, matrix, camera_to_base, top))

    def _fused_footprint(self, fused: np.ndarray, footprint: Any, telemetry: dict) -> np.ndarray:
        """What SFE builds its footprint from where a fused cloud comes with its rim cut (``footprint_points_base_mm``,
        the pick loop's looks less :attr:`support_footprint_rim_mm`): that cloud, the points it holds fewer than the
        fused one stamped (``support_footprint_rim_points``) and said in the log. The fused cloud whole where the
        footprint holds no point and the fused cloud does: a footprint is never built from nothing."""
        points = np.asarray(footprint, dtype=np.float64).reshape(-1, 3)
        if not points.shape[0] and fused.shape[0]:
            telemetry["support_footprint_rim_points"] = 0
            self.logger.info("footprint rim: the fused cloud less its rim holds no point, so SFE reads it whole")
            return fused
        off = int(fused.shape[0] - points.shape[0])
        telemetry["support_footprint_rim_points"] = off
        self.logger.info("footprint rim: SFE's footprint reads the looks' surfaces less their rim, %d of %d point(s) off",
                         off, int(fused.shape[0]))
        return points

    def compute(
        self,
        segmentation: SegmentationLike,
        depth_map: np.ndarray,
        T_cam_to_base: Transform | np.ndarray | None = None,
        unit: str = "mm",
        *,
        camera_to_base: Transform | np.ndarray | None = None,
        scene_points_mm: np.ndarray | None = None,
        rigid_obstacle_points_mm: np.ndarray | None = None,
        geometry_points_base_mm: np.ndarray | None = None,
        # The fused cloud less its rim (:attr:`support_footprint_rim_mm`): the same views' surfaces, each mask eroded as
        # the rim says, BASE. Read beside ``geometry_points_base_mm`` alone, by the stage alone, for the footprint it
        # builds; everything else reads the fused cloud whole. None reads it whole.
        footprint_points_base_mm: np.ndarray | None = None,
        # The pick's other looks (``scene_obstacles.SceneView``): what they show beside the part joins this frame's
        # obstacles, as the guard holds the boxes of every look, and a point any look saw past counts as seen for a
        # tilted approach and an open finger over the part's outline. () this frame alone.
        scene_views: "Sequence[SceneView]" = (),
        support_plane: SupportPlane | None = None,
        workspace: WorkspaceBox | None = None,
        gripper_model: GripperGeometryStrategy | None = None,
        min_table_clearance_mm: float = 5.0,
        collision_margin_mm: float = 0.0,
        rgb_image: np.ndarray | None = None,
        debug_label: str | None = None,
        debug_config: DebugDrawConfig | None = None,
        ik_service: IKService | None = None,
        other_object_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
        semantic_policy: SemanticPolicy | None = None,
        deformable_class: DeformableClass | None = None,
        # The camera world's support model of this frame (``support_surfaces.SupportModel``, the fix plan's contract
        # 7), handed per call beside ``support_plane``. Read only with ``scene_obstacles``: a surface it holds decides
        # what of its pixels is support, the stage's support is the higher of the plane and the model's reading under
        # the part, and the open hand keeps the guard's distance from its solids. None reads the plane alone.
        support_model: Any = None,
    ) -> list[GraspPoint]:
        """Generate ranked grasp candidates.

        ``T_cam_to_base`` returns base-frame candidates only, which is the
        contract the physical robot pipelines use. The same value may be
        passed as ``camera_to_base`` to make the frame direction explicit.
        A typed ``Transform`` must be ``CAMERA -> BASE``.

        The grasps are planned in BASE over ``support_plane``, a BASE plane: without it and the transform the call
        offers nothing, says so in ``last_telemetry`` (``support_footprint_ready``) and in the log, and names
        ``NO_CANDIDATES_GENERATED``. ``scene_points_mm`` and ``workspace`` feed the post-hoc filters, each stage
        exposed in ``last_telemetry`` for ops debugging.
        """
        # Fall back to the calculator's default IK oracle when no per-call service was passed, so the
        # live pick path (the orchestrator calls ``compute_result``, which calls ``compute``) gets the
        # IK filter. None applies no filter.
        if ik_service is None:
            ik_service = self._default_ik_service
        # Same pattern for the gripper collision envelope: a per-call model wins, else the calculator's
        # configured default, else (both None) the checker's ParallelJawGripperModel() fallback.
        if gripper_model is None:
            gripper_model = self._gripper_model
        scale = depth_unit_to_mm(unit)
        if T_cam_to_base is not None and camera_to_base is not None:
            raise ValueError("Pass only one of T_cam_to_base or camera_to_base.")
        transform_input = camera_to_base if camera_to_base is not None else T_cam_to_base
        transform = (
            _validate_camera_to_base_transform(transform_input)
            if transform_input is not None
            else None
        )
        # The plane, in the frame the filter actually runs in. ``_apply_collision_filters`` sees
        # CAMERA-frame poses (it runs before ``_to_base_frame_only``), so a BASE-frame plane has to be
        # rotated in first. ``resolve_support_plane`` produces BASE; wiring it straight through
        # compares a table height against a camera depth, which over 730 candidates computed a median
        # 577 mm "clearance" where the true clearance above the table was -7.01 mm, and rejected
        # 0.4 % of candidates instead of 65.5 %. Fail closed on a BASE plane with no transform:
        # without one there is no way to place the table, and a safety check is never skipped
        # silently.
        plane_cam: SupportPlane | None = support_plane
        if support_plane is not None and support_plane.frame is not Frame.CAMERA:
            if transform is None:
                raise ValueError(
                    "support_plane is in BASE frame but no camera_to_base transform was given; "
                    "the table-clearance check cannot be placed and will not be skipped silently")
            plane_cam = support_plane.to_camera_frame(transform)
        # Reset per-call debug surface so a stale image cannot leak into
        # the next read of ``last_debug_image_png`` if this call returns early.
        self.last_debug_image_png = None
        self._last_draw = None
        # Snapshot SAM2/segmentation metadata so it can be attached to
        # every resulting GraspPoint for traceability.
        seg_meta = _segmentation_metadata(segmentation)
        seg_score = seg_meta.get("score") if seg_meta else None
        mask_confidence = float(seg_score) if isinstance(seg_score, (int, float)) else None
        telemetry: dict = {
            "rejected_collision": 0,
            "rejected_table": 0,
            "rejected_workspace": 0,
            "final": 0,
            "best_score": 0.0,
        }

        mask_array = np.asarray(segmentation.mask)
        mask_has_any = bool(np.any(mask_array)) if mask_array.size > 0 else False
        # What the mask was, stamped before anything can reject it.
        #
        # The mask rejection below is the first early return in compute(), and without these two keys
        # the telemetry names no cause. An empty mask and a mask the analyzer found too small have nothing in
        # common except the word "rejected", and they need opposite fixes.
        telemetry["mask_pixels"] = int(np.count_nonzero(mask_array)) if mask_array.size else 0
        telemetry["mask_size"] = int(mask_array.size)
        analysis = self.analyzer.analyze(segmentation.mask)
        if analysis is None:
            self.logger.debug("MaskAnalyzer rejected the mask; no candidates.")
            reason = (
                GraspFailureReason.MASK_TOO_SMALL
                if mask_has_any
                else GraspFailureReason.EMPTY_MASK
            )
            return self._finalize_empty(
                telemetry,
                reasons=(reason, GraspFailureReason.RESCAN_RECOMMENDED),
                depth_confidence=None,
                mask_confidence=mask_confidence,
            )

        # The optional policy reject-gates (deformable / topology-risk / occlusion / semantic).
        # ``None`` means continue.
        gated = self._run_policy_gates(
            segmentation,
            depth_map,
            mask_array,
            mask_confidence,
            semantic_policy=semantic_policy,
            deformable_class=deformable_class,
            other_object_masks=other_object_masks,
            telemetry=telemetry,
        )
        if gated is not None:
            return gated

        depth_arr = np.asarray(depth_map, dtype=np.float64)
        if depth_arr.ndim != 2:
            raise ValueError(f"depth_map must be 2-D, got {depth_arr.shape}")
        if depth_arr.shape != segmentation.mask.shape:
            raise ValueError(
                "segmentation.mask and depth_map must have the same shape, got "
                f"{segmentation.mask.shape} and {depth_arr.shape}"
            )

        mask_bool = np.asarray(segmentation.mask).astype(bool)
        depth_under = depth_arr[mask_bool]
        valid_depth = depth_under[np.isfinite(depth_under) & (depth_under > 0.0)]
        # The raw depth under the mask, and where the CAMERA thinks it is: stamped here, before any
        # early return can swallow it. A telemetry key that is absent precisely when the run fails is
        # not a diagnostic.
        #
        # The two readings it separates, which need opposite fixes:
        #   masked depth ~= camera height                 -> the object is absent from the depth
        #                                                    render; what lies under its own mask is
        #                                                    the table behind it.
        #   masked depth ~= camera height - object height -> the object is there, and the transform
        #                                                    or the frame convention is what is wrong.
        # Cheap: two reductions over the already-materialised `depth_under`, no new pass over the image.
        if transform is not None:
            _finite_under = depth_under[np.isfinite(depth_under)]
            if _finite_under.size:
                telemetry["target_depth_min_mm"] = round(float(_finite_under.min()) * scale, 2)
                telemetry["target_depth_max_mm"] = round(float(_finite_under.max()) * scale, 2)
            telemetry["camera_base_z_mm"] = round(float(transform[2, 3]), 2)
        if valid_depth.size == 0:
            self.logger.warning(
                "No valid depth pixels under mask (label=%r). Skipping.",
                getattr(segmentation, "label", "?"),
            )
            return self._finalize_empty(
                telemetry,
                reasons=(
                    GraspFailureReason.NO_VALID_DEPTH,
                    GraspFailureReason.RESCAN_RECOMMENDED,
                ),
                depth_confidence=0.0,
                mask_confidence=mask_confidence,
            )

        depth_confidence = float(valid_depth.size) / float(depth_under.size)
        intrinsics = self._geom.intrinsics(depth_arr.shape)
        # The stage plans in BASE over the support the part stands on (``support_footprint_breakdowns``): its
        # ``offset_mm`` is a height above the workspace, and a CAMERA-frame plane's offset is a depth, which taken for a
        # height makes the table check inert. Without a CAMERA->BASE transform and a BASE plane it cannot place the
        # table, and nothing is offered rather than a grasp nothing held against it.
        telemetry["geometry_stage"] = "support_footprint"
        ready = transform is not None and support_plane is not None and support_plane.frame is Frame.BASE
        telemetry["support_footprint_ready"] = bool(ready)
        if not ready:
            said = ("the support-footprint stage plans in BASE over the support the part stands on, and this call gave "
                    + ("no CAMERA->BASE transform" if transform is None else "no BASE support plane"))
            telemetry[NO_GRASP_SAID] = said
            self.logger.info("no grasp: %s", said)
            return self._finalize_empty(
                telemetry,
                reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,),
                depth_confidence=float(depth_confidence),
                mask_confidence=mask_confidence,
            )
        assert transform is not None and support_plane is not None  # narrowed by ready

        cloud = masked_point_cloud(
            segmentation.mask,
            depth_arr,
            intrinsics,
            unit=unit,
            min_depth_mm=1.0,
            voxel_size_mm=CLOUD_VOXEL_MM,
        )
        # Where the target was seen, in BASE, whenever this view's own masked cloud is non-empty. A masked depth image
        # can pick up support-plane pixels at the mask edge, which drags a grasp to the object's base; the span between
        # these two keys tells that apart from a short object, and the two need opposite fixes. Skipped on the fused
        # path when this view's cloud is empty and the injected fused cloud carries the geometry instead; the stage's
        # keys cover that case.
        if not cloud.is_empty:
            _cloud_base_z = (
                np.asarray(cloud.points_mm, dtype=np.float64).reshape(-1, 3) @ transform[:3, :3].T
                + transform[:3, 3]
            )[:, 2]
            telemetry["target_cloud_z_min_mm"] = round(float(_cloud_base_z.min()), 2)
            telemetry["target_cloud_z_max_mm"] = round(float(_cloud_base_z.max()), 2)
        if geometry_points_base_mm is not None and np.asarray(geometry_points_base_mm).size > 0:
            telemetry["fused_geometry_points"] = int(np.asarray(geometry_points_base_mm).reshape(-1, 3).shape[0])
            telemetry["geometry_source"] = "fused_multiview"
        if other_object_masks:
            # The other parts the detector named in this frame, as a cloud: the collision veto is one point
            # (`collision_checker.py`, `if colliding_indices:`), so the post-hoc filter and the stage keep the hand off
            # every one of them.
            neighbours = scene_collision_cloud(
                list(other_object_masks),
                depth_arr,
                intrinsics,
                target_index=None,
                unit=unit,
                min_depth_mm=1.0,
                voxel_size_mm=CLOUD_VOXEL_MM,
            )
            if neighbours.size > 0:
                telemetry["scene_neighbour_points"] = int(neighbours.shape[0])
                if scene_points_mm is None:
                    scene_points_mm = neighbours
                else:
                    scene_points_mm = np.vstack(
                        [np.asarray(scene_points_mm, dtype=np.float64), neighbours]
                    )
        # What the camera saw beside the part, by the planner world's rules (fix plan Track A). It joins the scene
        # points the way the neighbours' masks do above, so the stage's grid and the post-hoc filter both see it.
        scene: SceneObstacles | None = None
        seen_envelope: SeenEnvelope | None = None
        hand_floor: HandFloor | None = None
        scene_support_mm: float | None = None
        if self._scene_obstacles is not None:
            telemetry["candidates_final_source"] = "prompt_only"
            scene_support_mm = self._scene_support_mm(
                support_plane, support_model, cloud, geometry_points_base_mm, transform, telemetry)
            # The other parts the detector named in this frame: a finger may come to their measured surface
            # (the owner, 2026-10-05: "Finger dürfen streifen"), where the cell lets the fingers touch parts. Their
            # surfaces as the pick offers them to the camera world, each part its own owner, in every look.
            named: list[np.ndarray] = []
            if other_object_masks and float(self._scene_obstacles.part_soft_mm) > 0.0:
                for other in other_object_masks:
                    if np.shape(other) == mask_bool.shape:
                        named.append(self._named_surface_of(other, depth_map, scale, intrinsics, transform))
            # What the pick's other looks show beside the part joins this frame's before a box is built: the
            # guard holds the boxes of every look, and a grasp offered against this look's alone met a wall only
            # another look saw (2026-10-13, T4).
            scene = scene_obstacle_points(
                depth_arr * scale, mask_bool, intrinsics, transform, rules=self._scene_obstacles,
                support_height_mm=scene_support_mm, support_model=support_model, named=named, looks=scene_views,
                part_base_mm=geometry_points_base_mm, view=self._frame_view(depth_map, scale, intrinsics, transform))
            if scene_views:
                telemetry["scene_views"] = len(scene_views)
            telemetry["scene_obstacle_points"] = int(scene.points_base_mm.shape[0])
            telemetry["scene_obstacle_clusters"] = int(scene.clusters)
            telemetry["scene_obstacle_world_rule_points"] = int(scene.world_rule_base_mm.shape[0])
            telemetry["candidates_final_source"] = "scene"
            self.logger.info("scene obstacles: %s", scene.render())
            # The boxes the camera world will hold for these neighbours: the open hand keeps the guard's distance
            # from them, or the guard refuses the grasp once the arm stood over it (the owner, 2026-10-03).
            telemetry["scene_seen_boxes"] = len(scene.boxes)
            seen_envelope = SeenEnvelope.of(
                scene.boxes, gripper_model=gripper_model if gripper_model is not None else ParallelJawGripperModel(),
                open_width_mm=self.max_grip_mm, distance_mm=self._scene_obstacles.support_distance_mm)
            # The solids the guard holds under the part stand over the reading by its band and allowance, and the
            # guard keeps its distance from their tops: the stage plans the hand's height on them, not on the
            # reading, or the filter below refuses every grasp it lowered (``hand.HandFloor``).
            held = tuple(getattr(support_model, "solids", ()) or ()) if support_model is not None else ()
            if held:
                hand_floor = HandFloor(solids=held, distance_mm=float(self._scene_obstacles.support_distance_mm),
                                       fingers_to_the_reading=bool(self._scene_obstacles.fingers_to_the_reading),
                                       finger_floor_drop_mm=float(self._scene_obstacles.finger_floor_drop_mm))
            if scene.points_base_mm.shape[0]:
                _to_cam = np.linalg.inv(np.asarray(transform, dtype=np.float64))
                _scene_cam = scene.points_base_mm @ _to_cam[:3, :3].T + _to_cam[:3, 3]
                scene_points_mm = (_scene_cam if scene_points_mm is None
                                   else np.vstack([np.asarray(scene_points_mm, dtype=np.float64), _scene_cam]))

        # The support-footprint stage: the part's own body over its support (``body_grasps``), planned so the hand
        # clears the support instead of being filtered for it. Its own score is kept: the geometric ranker that left on
        # 2026-10-18 was anti-correlated with validity on this data (AUC 0.419).
        #
        # The stage's input at full resolution, without the mask pixels that measure the background behind the part's
        # edge: a band of table seen past the far edge through depth-to-colour misregistration pulls a face out to it;
        # at a 45-degree view that is 40 mm behind a 40 mm part. See depth_steps.py. Only the stage's input is trimmed,
        # and a mask with no step inside it gives the same cloud.
        sfe_cloud = cloud
        # The pixels of this look whose depth the stage reads for the footprint, where the cell cuts a rim: every other
        # pixel of the mask is placed in the part's visual hull below (``footprint_hull``).
        hull_read: np.ndarray | None = None
        if not cloud.is_empty:
            depth_mm = depth_arr * scale
            behind = pixels_behind_depth_steps(mask_bool, depth_mm)
            telemetry["support_footprint_depth_step_pixels"] = int(np.count_nonzero(behind))
            sfe_mask = mask_bool & ~behind
            # And without the mask's rim, where the cell cuts one (``support_footprint_rim_mm``): the ramp of depths
            # the D415 smears a far edge into lies on it, and the hull follows the ramp. A fused cloud comes with its
            # rim cut by the pick loop (``footprint_points_base_mm``), so it is cut here only where the stage reads
            # this look alone.
            if self._support_footprint_rim_mm > 0.0 and geometry_points_base_mm is None:
                sfe_mask, _rimmed = self._without_the_rim(mask_bool, sfe_mask, depth_mm, intrinsics, telemetry)
            if self._support_footprint_rim_mm > 0.0:
                hull_read = self._read_for_the_footprint(
                    mask_bool, sfe_mask, depth_mm, intrinsics, fused=geometry_points_base_mm is not None)
            sfe_cloud = masked_point_cloud(sfe_mask, depth_arr, intrinsics, unit=unit, min_depth_mm=1.0,
                                           voxel_size_mm=None)
        if geometry_points_base_mm is not None:
            target_base = np.asarray(geometry_points_base_mm, dtype=np.float64).reshape(-1, 3)
            whole_cloud = target_base
            if footprint_points_base_mm is not None:
                target_base = self._fused_footprint(target_base, footprint_points_base_mm, telemetry)
            if target_base is whole_cloud:
                # The fused cloud is read whole, its rim with it: no pixel of this look went unread.
                hull_read = None
        else:
            target_base = (transform[:3, :3] @ np.asarray(
                sfe_cloud.points_mm, dtype=np.float64).reshape(-1, 3).T).T + transform[:3, 3]
        obstacles_base = None
        if scene is not None and scene.part_points is not None and bool(scene.part_points.any()):
            # A named neighbour's points leave the grid: the boxes the camera world builds of them hold the
            # fingers to their measured surface (``SeenEnvelope.into_parts_mm``), the rest of the hand to the
            # guard's distance. What nobody named stays in the grid.
            obstacles_base = scene.points_base_mm[~scene.part_points]
        elif scene_points_mm is not None and np.asarray(scene_points_mm).size > 0:
            _sp = np.asarray(scene_points_mm, dtype=np.float64).reshape(-1, 3)
            obstacles_base = (transform[:3, :3] @ _sp.T).T + transform[:3, 3]
        # Declared geometry stays out of the sparse cloud above and gets its own,
        # undilated grid inside the generator; see hand._ObstacleSet.
        rigid_base = None
        if (rigid_obstacle_points_mm is not None
                and np.asarray(rigid_obstacle_points_mm).size > 0):
            _rp = np.asarray(rigid_obstacle_points_mm, dtype=np.float64).reshape(-1, 3)
            rigid_base = (transform[:3, :3] @ _rp.T).T + transform[:3, 3]
        sfe_jaw = self._support_footprint_jaw(gripper_model, min_table_clearance_mm)
        sfe_support_mm = (float(support_plane.offset_mm) if scene_support_mm is None
                          else max(float(support_plane.offset_mm), scene_support_mm))
        # The points a camera measured, before the visual hull's join them: where the part is widest is read
        # from these (``body_grasps._under_the_widest``); the mask's outline at the part's top, which how square
        # a pad meets it is read on (``body_grasps._squared``), and its holes, where a bore is (L, 2026-10-15).
        measured_base = target_base
        outline_base, hole_base = self._mask_at_its_top(mask_bool, intrinsics, transform, measured_base,
                                                        sfe_support_mm)
        if hull_read is not None:
            target_base = self._with_the_hull(
                target_base, mask_bool, hull_read, intrinsics, transform, sfe_support_mm, telemetry)
        # Only space this frame's depth saw counts as clear for a tilted approach (the fix plan's contract 5), and
        # for an open finger over the part's own outline, in a bore or a gap (``body_grasps``).
        seen_here = corridor_seen_in(depth_arr * scale, intrinsics, transform)
        sfe_ranked, sfe_telemetry = support_footprint_breakdowns(
            target_base,
            camera_to_base=transform,
            support_height_mm=sfe_support_mm,
            jaw=sfe_jaw,
            obstacle_points_base_mm=obstacles_base,
            rigid_obstacle_points_base_mm=rigid_base,
            max_candidates=self.max_candidates,
            floor_margin_mm=self._support_footprint_floor_margin_mm,
            side_approaches=self._side_approaches,
            corridor_seen=(SeenInAnyFrame((seen_here, *(view.seen() for view in scene_views)))
                           if scene_views else seen_here),
            seen_envelope=seen_envelope,
            hand_floor=hand_floor,
            opens_as_needed=self.opens_as_needed,
            inner_grasps=self.inner_grasps,
            measured_base_mm=measured_base,
            outline_base_mm=outline_base,
            hole_base_mm=hole_base,
        )
        telemetry.update(sfe_telemetry)
        if self._scene_obstacles is not None:
            telemetry["scene_table_clearance_mm"] = float(sfe_jaw.table_clearance_mm)
        if self._scene_obstacles is not None and not sfe_ranked:
            # What a "too short" reads against: how high the part stands over the support the stage planned on, and
            # how high it has to stand for this hand.
            _floor = sfe_support_mm + (DEFAULT_FLOOR_MARGIN_MM if self._support_footprint_floor_margin_mm is None
                                       else self._support_footprint_floor_margin_mm)
            _over = target_base[np.isfinite(target_base).all(axis=1) & (target_base[:, 2] > _floor), 2]
            if _over.size:
                telemetry["scene_part_height_mm"] = round(float(np.percentile(_over, 98.0)) - sfe_support_mm, 1)
            # The solid the guard holds under the part lifts the hand by as much as its top and distance stand
            # over the clearance the reading alone asks (``hand_floor``).
            _rise = 0.0
            if hand_floor is not None and _over.size:
                _under = hand_floor.at(target_base[np.isfinite(target_base).all(axis=1), :2], fingers=True)
                _under = _under[np.isfinite(_under)]
                if _under.size:
                    _rise = max(0.0, float(_under.max()) - sfe_support_mm - float(sfe_jaw.table_clearance_mm))
            telemetry["scene_least_part_height_mm"] = round(least_part_height_mm(sfe_jaw) + _rise, 1)
        # An empty list is the stage's answer: it looked and found no legal grasp, or it could not look at all
        # (``reconstruct_support_prism`` gives up below 20 surviving points). Refusing beats proposing a grasp nothing
        # held against the support.
        ranked_all: Sequence[GraspScoreBreakdown] = tuple(sfe_ranked)

        # Declared geometry rejoins the observed cloud for the post-hoc filter, which tests exact box
        # containment at margin 0 and therefore needs no separate treatment; the split above exists
        # only because the generator's grid dilates, and a wall must not be dilated.
        if rigid_obstacle_points_mm is not None and np.asarray(rigid_obstacle_points_mm).size > 0:
            _rigid = np.asarray(rigid_obstacle_points_mm, dtype=np.float64).reshape(-1, 3)
            telemetry["rigid_obstacle_points"] = int(_rigid.shape[0])
            scene_points_mm = (
                _rigid if scene_points_mm is None
                else np.vstack([np.asarray(scene_points_mm, dtype=np.float64), _rigid])
            )
        # Collision/IK filter + candidate build + base-frame + sort + clearance stamps.
        candidates = self._filter_and_build_candidates(
            ranked_all,
            seg_meta=seg_meta,
            scene_points_mm=scene_points_mm,
            support_plane=plane_cam,
            workspace=workspace,
            gripper_model=gripper_model,
            min_table_clearance_mm=min_table_clearance_mm,
            collision_margin_mm=collision_margin_mm,
            ik_service=ik_service,
            transform=transform,
            intrinsics=intrinsics,
            depth_map=depth_map,
            scale=scale,
            other_object_masks=other_object_masks,
            telemetry=telemetry,
            held_solids=(tuple(getattr(support_model, "solids", ()) or ())
                         if self._scene_obstacles is not None and support_model is not None else ()),
            seen_boxes=scene.boxes if scene is not None else (),
        )
        self.last_telemetry = telemetry
        if rgb_image is not None:
            self._last_draw = {
                "rgb_image": rgb_image,
                "segmentation": segmentation,
                "intrinsics": intrinsics,
                "gripper_model": gripper_model,
                "config": debug_config,
                "label": debug_label,
                "telemetry": telemetry,
                "transform": transform,
            }
            self.last_debug_image_png = self._debug_renderer.draw(candidates=candidates, **self._last_draw)
        self.logger.info(
            "Generated %d grasp candidates (best score=%.3f); telemetry=%s",
            len(candidates),
            candidates[0].score if candidates else 0.0,
            telemetry,
        )
        reasons = self._derive_failure_reasons(
            candidates=candidates,
            telemetry=telemetry,
            had_filters=True,
            scene_obstacles=self._scene_obstacles is not None,
        )
        if self._scene_obstacles is not None and not candidates:
            said = why_no_grasp(telemetry)[1]
            if said:
                # Said once, in the robot log and on the result, so a pick that got no grasp names why (RC6).
                telemetry[NO_GRASP_SAID] = said
                self.logger.info("no grasp: %s", said)
        self.last_failure_reasons = reasons
        self.last_result = GraspResult(
            candidates=tuple(candidates),
            reasons=reasons,
            telemetry=dict(telemetry),
            depth_confidence=float(depth_confidence),
            mask_confidence=mask_confidence,
            top_score=float(candidates[0].score) if candidates else 0.0,
            # Surface the CAMERA-frame neighbour collision cloud on the result so the
            # orchestrator can validate the approach/retreat sweep against it (the calculator only checks
            # the final grasp pose). The orchestrator transforms it to the candidate frame. Empty when no
            # neighbours (single-object), so the validator sees NO_OBSTACLES.
            metadata={
                **({
                    "scene_points_mm": np.asarray(scene_points_mm, dtype=np.float64).copy(),
                    "scene_points_frame": "camera",
                } if scene_points_mm is not None and np.asarray(scene_points_mm).size > 0 else {}),
                # The fix plan's contract 2, for the pick loop's push: what A kept, and every point the camera world's
                # rule keeps within the reach, both BASE. Only where the scene was read.
                **({
                    "scene_obstacle_points_base_mm": scene.points_base_mm.copy(),
                    "scene_points_world_rule_base_mm": scene.world_rule_base_mm.copy(),
                    # Which of each are a named neighbour part's: a push's finger comes down beside those as a grasp's
                    # finger does (the owner, 2026-10-06).
                    "scene_obstacle_part_points": None if scene.part_points is None else scene.part_points.copy(),
                    "scene_world_rule_part_points": (None if scene.world_rule_part is None
                                                     else scene.world_rule_part.copy()),
                    # The boxes the camera world builds of them and the guard's distance from those: the pick loop
                    # turns each grasp the way round its wrist camera keeps it (``wrist_turn``).
                    "scene_seen_boxes": tuple(scene.boxes),
                    "scene_seen_distance_mm": float(self._scene_obstacles.support_distance_mm)
                    if self._scene_obstacles is not None else None,
                } if scene is not None else {}),
            },
        )
        return candidates

    def _run_policy_gates(
        self,
        segmentation: SegmentationLike,
        depth_map: np.ndarray,
        mask_array: np.ndarray,
        mask_confidence: float | None,
        *,
        semantic_policy: SemanticPolicy | None,
        deformable_class: DeformableClass | None,
        other_object_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
        telemetry: dict,
    ) -> list[GraspPoint] | None:
        """The optional policy reject-gates of :meth:`compute`: deformable, topology-risk, occlusion,
        semantic. Each gate stamps its telemetry and, on a breach, returns the empty
        :meth:`_finalize_empty` list; returns ``None`` to continue. ``telemetry`` is mutated in place."""
        # Optional deformable-handling gate. Strictly additive: when no strategy is configured the gate
        # is skipped entirely and the ``deformable_class`` argument is
        # ignored, so behaviour is byte-identical. A configured strategy
        # always runs (even for ``RIGID`` / ``None``) so it can stamp its
        # own telemetry; the gate only short-circuits when the strategy
        # returns ``proceed=False``.
        if self.deformable_strategy is not None:
            requested_class = (
                deformable_class
                if deformable_class is not None
                else DeformableClass.RIGID
            )
            decision = self.deformable_strategy.handle(
                deformable_class=requested_class,
                mask=mask_array,
                depth_map=np.asarray(depth_map),
            )
            if decision.telemetry:
                telemetry.update(decision.telemetry)
            if not decision.proceed:
                reasons: tuple[GraspFailureReason, ...] = (
                    decision.reasons
                    if decision.reasons
                    else (GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,)
                )
                return self._finalize_empty(
                    telemetry,
                    reasons=reasons,
                    depth_confidence=None,
                    mask_confidence=mask_confidence,
                )

        # Topology-risk signal + optional rejection gate. The risk score is always stamped when
        # computable, so non-convexity is visible without enabling the
        # gate. Rejection is opt-in via ``topology_risk_threshold``; a
        # threshold breach short-circuits with an explicit reason code so
        # the failure is never silent.
        topology_risk = topology_risk_from_mask(segmentation.mask)
        if topology_risk is not None:
            telemetry["topology_risk"] = float(topology_risk)
        if (
            self.topology_risk_threshold is not None
            and topology_risk is not None
            and topology_risk > self.topology_risk_threshold
        ):
            telemetry["topology_risk_threshold"] = float(self.topology_risk_threshold)
            telemetry["topology_risk_rejected"] = True
            return self._finalize_empty(
                telemetry,
                reasons=(
                    GraspFailureReason.TOPOLOGY_RISK_REJECTED,
                    GraspFailureReason.RESCAN_RECOMMENDED,
                ),
                depth_confidence=None,
                mask_confidence=mask_confidence,
            )

        # Occlusion ratio + optional rejection gate. Always compute the convex-hull occlusion ratio when other
        # masks are supplied, so operators can monitor clutter without
        # opting into rejection. The gate only fires when
        # ``heavy_occlusion_threshold`` is set and exceeded.
        if other_object_masks:
            try:
                target_occlusion = occlusion_ratio(
                    segmentation.mask, list(other_object_masks)
                )
            except ValueError:
                # Shape mismatch between target and neighbour masks is
                # not a calculator-internal failure; skip the metric
                # rather than aborting the pipeline.
                target_occlusion = None
            if target_occlusion is not None:
                telemetry["occlusion_ratio"] = float(target_occlusion)
                if (
                    self.heavy_occlusion_threshold is not None
                    and target_occlusion > self.heavy_occlusion_threshold
                ):
                    telemetry["heavy_occlusion_threshold"] = float(
                        self.heavy_occlusion_threshold
                    )
                    telemetry["heavy_occlusion_rejected"] = True
                    return self._finalize_empty(
                        telemetry,
                        reasons=(
                            GraspFailureReason.HEAVY_OCCLUSION,
                            GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED,
                        ),
                        depth_confidence=None,
                        mask_confidence=mask_confidence,
                    )

        # Optional semantic / task-aware policy gate. No policy, or a no-op policy, adds not even a
        # telemetry key, so a caller that passes none sees zero behavior
        # change. When a real policy rejects this candidate the failure
        # surfaces as ``SEMANTIC_REJECTED`` with the policy's stable
        # reason string in telemetry for routing.
        if semantic_policy is not None and not semantic_policy.is_noop:
            seg_label = getattr(segmentation, "label", None)
            semantic_decision = semantic_policy.evaluate(seg_label, mask_confidence)
            telemetry["semantic_decision"] = (
                "accepted"
                if semantic_decision.accept
                else (semantic_decision.reason or "rejected")
            )
            if not semantic_decision.accept:
                return self._finalize_empty(
                    telemetry,
                    reasons=(
                        GraspFailureReason.SEMANTIC_REJECTED,
                        GraspFailureReason.TRY_NEXT_CANDIDATE,
                    ),
                    depth_confidence=None,
                    mask_confidence=mask_confidence,
                )
        return None

    def _filter_and_build_candidates(
        self,
        ranked_all: Sequence[GraspScoreBreakdown],
        *,
        seg_meta: dict | None,
        scene_points_mm: np.ndarray | None,
        support_plane: SupportPlane | None,
        workspace: WorkspaceBox | None,
        gripper_model: GripperGeometryStrategy | None,
        min_table_clearance_mm: float,
        collision_margin_mm: float,
        ik_service: IKService | None,
        transform: np.ndarray | None,
        intrinsics: CameraIntrinsics,
        depth_map: np.ndarray,
        scale: float,
        other_object_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
        telemetry: dict,
        held_solids: Sequence[Any] = (),
        seen_boxes: Sequence[Any] = (),
    ) -> list[GraspPoint]:
        """The post-ranking stage of :meth:`compute`: collision/table/workspace + IK filtering, the
        :class:`GraspPoint` build, the optional base-frame transform, the final sort, and the approach-clearance /
        force-closure stamps. The ``telemetry`` dict is mutated
        in place. Returns the final sorted candidates."""
        # Collision / table / workspace filtering (camera-frame; matches
        # the frame the candidates were planned in).
        if support_plane is not None or scene_points_mm is not None or workspace is not None:
            # Where the calculator holds the guard's own boxes (``scene_obstacles``), they judge what the hand meets
            # beside the part (``_keep_off_what_the_guard_holds``), the points only the table and the box: the points
            # held a named neighbour the fingers may brush as whole, and on the deep crate of 2026-10-12 at 11:26
            # refused both of the body's grasps, which the guard admits.
            mirrored = self._scene_obstacles is not None and transform is not None
            ranked_all, stage_telemetry = self._apply_collision_filters(
                ranked_all,
                scene_points_mm=None if mirrored else scene_points_mm,
                support_plane=support_plane,
                workspace=workspace,
                gripper_model=gripper_model,
                min_table_clearance_mm=min_table_clearance_mm,
                collision_margin_mm=collision_margin_mm,
            )
            telemetry.update(stage_telemetry)
        if self._scene_obstacles is not None and transform is not None:
            ranked_all = self._keep_off_what_the_guard_holds(
                ranked_all, held_solids=held_solids, gripper_model=gripper_model, transform=transform,
                telemetry=telemetry, seen_boxes=seen_boxes)

        telemetry.setdefault("rejected_ik", 0)
        if ik_service is not None:
            ranked_all, ik_telemetry = self._apply_ik_filter(
                ranked_all,
                ik_service=ik_service,
                transform=transform,
            )
            telemetry["rejected_ik"] = ik_telemetry["rejected_ik"]

        candidates = [self._to_grasp_point(score, seg_meta) for score in ranked_all]
        if transform is not None:
            candidates = self._to_base_frame_only(candidates, transform)
        # Best first, and every grasp from inside a bore after every outer grasp: it is
        # taken where no outer grasp is. With none from inside, the order of the scores alone, as before: both sorts
        # keep grasps of equal keys in the order they came.
        candidates.sort(key=outer_first)
        telemetry["final"] = len(candidates)
        telemetry["best_score"] = candidates[0].score if candidates else 0.0
        # Approach-clearance for the best (camera-frame)
        # candidate. Computed from ``ranked_all[0].pose``, which the
        # optional base-frame transform never touches:
        # ``_to_base_frame_only`` rebuilds ``candidates`` only, so the
        # pose is still camera-frame here and re-projects through the
        # pinhole intrinsics. ``last_telemetry`` only carries the
        # scalar; the per-candidate stamp on metadata happens above.
        if ranked_all:
            best_pose = ranked_all[0].pose
            try:
                clearance_mm = approach_clearance_mm(
                    best_pose.position_mm,
                    best_pose.approach_axis,
                    np.asarray(depth_map, dtype=np.float64),
                    intrinsics,
                    max_distance_mm=200.0,
                    step_mm=5.0,
                    scale_to_mm=float(scale),
                )
            except ValueError:
                clearance_mm = None
            if clearance_mm is not None:
                telemetry["approach_clearance_mm"] = float(clearance_mm)
                # GraspPoint is frozen, but its metadata dict is mutable
                # by reference; the stamp lands on the best (sorted) candidate.
                candidates[0].metadata["approach_clearance_mm"] = float(
                    clearance_mm
                )
        return candidates

    def compute_result(self, *args: Any, **kwargs: Any) -> GraspResult:
        """Run :meth:`compute` and return the structured :class:`GraspResult`.

        Thin wrapper that surfaces failure / recommendation codes in
        addition to the ranked candidate list. The ``compute()`` return
        is unchanged.
        """
        self.compute(*args, **kwargs)
        return self.last_result

    def redraw_debug_image(self, candidates: Sequence[GraspPoint]) -> bytes | None:
        """Draw the last overlay again over ``candidates``, best first, and keep it as ``last_debug_image_png``.

        For a caller that kept only some of the last :meth:`compute`'s candidates, or turned some: the pick loop, where
        the closing axis its program named (``GraspMotion(closing_axis=...)``) left some out or turned some half a turn
        about their approach, so the overlay shows the grasps the pick chooses among, each the way round it closes, and
        its ``final`` and ``best_score`` count those. ``None``, and nothing drawn, where the last compute drew none.
        """
        if self._last_draw is None:
            return None
        kept = list(candidates)
        drawn = {**self._last_draw, "telemetry": {**self._last_draw["telemetry"], "final": len(kept),
                                                  "best_score": float(kept[0].score) if kept else 0.0}}
        self.last_debug_image_png = self._debug_renderer.draw(candidates=kept, **drawn)
        return self.last_debug_image_png

    def plan_multifinger(
        self,
        *,
        segmentation: SegmentationLike,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        kinematics: "FingerKinematicSpec",
        planner: "MultiContactGraspPlanner | None" = None,
        max_results: int = 5,
        scale_to_mm: float = 1.0,
        approach_axis_cam: np.ndarray | None = None,
    ) -> list["MultiContactGrasp"]:
        """Plan N-finger grasp candidates.

        This entry point is strictly additive: it runs alongside
        the existing parallel-jaw pipeline and has zero effect on
        :meth:`compute`. Both flows can be invoked independently on
        the same calculator instance.

        Planner selection
        -----------------

        * ``planner`` when provided must implement
          :class:`src.robot.grasping.planning.MultiContactGraspPlanner`.
          The caller-supplied planner is used verbatim.
        * ``planner is None`` infers the default planner from
          ``kinematics.finger_count``: 2 selects
          :class:`ParallelJawContactPlanner`, >=3 selects
          :class:`RadialMultiFingerPlanner`.

        Returns
        -------
        list[MultiContactGrasp]
            Candidates ordered by descending score (planner-defined).
            Empty when the mask is missing, the depth map is invalid,
            or no contact configuration satisfies the kinematic
            envelope.
        """
        from src.robot.grasping.planning.multifinger import (
            ParallelJawContactPlanner,
            RadialMultiFingerPlanner,
            MultiContactPlanRequest,
            MultiContactGraspPlanner as _PlannerProtocol,
        )

        mask = getattr(segmentation, "mask", None)
        if mask is None:
            return []
        approach = (
            np.asarray(approach_axis_cam, dtype=np.float64)
            if approach_axis_cam is not None
            else self.approach_cam
        )
        request = MultiContactPlanRequest(
            mask=np.asarray(mask),
            depth_map=np.asarray(depth_map),
            intrinsics=intrinsics,
            kinematics=kinematics,
            approach_axis_cam=approach,
            scale_to_mm=float(scale_to_mm),
            max_results=int(max_results),
        )
        if planner is None:
            planner = (
                ParallelJawContactPlanner()
                if int(kinematics.finger_count) == 2
                else RadialMultiFingerPlanner()
            )
        if not isinstance(planner, _PlannerProtocol):
            raise TypeError(
                "planner must implement MultiContactGraspPlanner; "
                f"got {type(planner).__name__}"
            )
        return planner.plan(request)

    def _finalize_empty(
        self,
        telemetry: dict,
        *,
        reasons: tuple[GraspFailureReason, ...],
        depth_confidence: float | None,
        mask_confidence: float | None,
    ) -> list[GraspPoint]:
        """Record an empty result and return ``[]``, the list :meth:`compute` returns."""
        self.last_telemetry = telemetry
        self.last_failure_reasons = reasons
        self.last_result = GraspResult(
            candidates=(),
            reasons=reasons,
            telemetry=dict(telemetry),
            depth_confidence=depth_confidence,
            mask_confidence=mask_confidence,
            top_score=0.0,
        )
        return []

    def _apply_ik_filter(
        self,
        ranked: Sequence[GraspScoreBreakdown],
        *,
        ik_service: IKService,
        transform: np.ndarray | None,
    ) -> tuple[list[GraspScoreBreakdown], dict]:
        """Drop candidates the IK service declares unreachable.

        Candidates are passed to the service in the final return frame:
        base frame when a ``camera_to_base`` / ``T_cam_to_base`` was
        supplied, otherwise camera frame. The IK adapter is expected to
        be configured for that frame.
        """
        from src.robot.grasping.planning import transform_grasp_pose

        kept: list[GraspScoreBreakdown] = []
        rejected = 0
        target_frame = Frame.BASE if transform is not None else Frame.CAMERA
        for score in ranked:
            pose = score.pose
            if transform is not None:
                try:
                    query_pose = transform_grasp_pose(
                        pose, transform, target_frame=target_frame
                    )
                except ValueError:
                    # Defence in depth: transform_grasp_pose re-orthonormalises the rotation so a
                    # slightly-slack calibration matrix does not trip GraspPose's strict gate. A
                    # candidate that still fails validity is dropped on its own (counted as an IK
                    # rejection) rather than letting the exception abort the whole pick: fail closed.
                    rejected += 1
                    continue
            else:
                query_pose = pose
            result = ik_service.query(query_pose)
            if not isinstance(result, IKResult):
                raise TypeError(
                    "IKService.query must return an IKResult, "
                    f"got {type(result).__name__}"
                )
            if result.reachable:
                kept.append(score)
            else:
                rejected += 1
        return kept, {"rejected_ik": rejected}

    @staticmethod
    def _derive_failure_reasons(
        *,
        candidates: list[GraspPoint],
        telemetry: dict,
        had_filters: bool,
        scene_obstacles: bool = False,
    ) -> tuple[GraspFailureReason, ...]:
        """Map telemetry counters to reason codes for the success path.

        If the pipeline reached the end and still has candidates the
        result is a success and no reasons are emitted. If the
        post-filter list is empty but filters were active, the reasons
        name which stage exhausted the candidates.

        With ``scene_obstacles`` the reasons are read off SFE's refusal counts and the filters' counters by
        ``scene_obstacles.why_no_grasp``: a neighbour the camera saw is ``ALL_COLLIDED``, the one reason the push
        answers; a declared body, the support, or a part too short for the hand is ``ALL_TABLE_CONFLICT``.
        """
        if candidates:
            return ()
        if scene_obstacles:
            return why_no_grasp(telemetry)[0]
        if not had_filters:
            return (
                GraspFailureReason.NO_VALID_GRASP,
                GraspFailureReason.RESCAN_RECOMMENDED,
            )
        reasons: list[GraspFailureReason] = []
        rejected_collision = int(telemetry.get("rejected_collision", 0))
        rejected_table = int(telemetry.get("rejected_table", 0))
        rejected_workspace = int(telemetry.get("rejected_workspace", 0))
        rejected_ik = int(telemetry.get("rejected_ik", 0))
        if rejected_workspace > 0:
            reasons.append(GraspFailureReason.ALL_OUT_OF_WORKSPACE)
        if rejected_table > 0:
            reasons.append(GraspFailureReason.ALL_TABLE_CONFLICT)
        if rejected_collision > 0:
            reasons.append(GraspFailureReason.ALL_COLLIDED)
        if rejected_ik > 0:
            reasons.append(GraspFailureReason.IK_FAILED)
        if not reasons:
            reasons.append(GraspFailureReason.NO_VALID_GRASP)
        reasons.append(GraspFailureReason.RESCAN_RECOMMENDED)
        return tuple(reasons)

    def _frame_view(self, depth_map: Any, scale: float, intrinsics: Any, transform: np.ndarray) -> SceneView:
        """The frame of ``depth_map`` as a ``scene_obstacles.SceneView``, the same one for each part of a frame the pick
        asks the calculator of, so what the frame reads the same for each part is read once (``SceneView.reading``)."""
        placed = (float(scale), np.asarray(transform, dtype=np.float64).tobytes(),
                  np.asarray(intrinsics.to_matrix(), dtype=np.float64).tobytes())
        held = self._held_frame
        if held is None or held[0] is not depth_map or held[1] != placed:
            held = (depth_map, placed, SceneView(depth_mm=np.asarray(depth_map, dtype=np.float64) * scale,
                                                 intrinsics=intrinsics, camera_to_base=transform))
            self._held_frame = held
        return held[2]

    def _named_surface_of(self, mask: Any, depth_map: Any, scale: float, intrinsics: Any,
                          transform: np.ndarray) -> np.ndarray:
        """The surface a named part ``mask`` covers in ``depth_map`` (``scene_obstacles.named_surface``), worked out once
        a frame: the pick asks the calculator of each part of a frame in turn, every other part's mask the same object
        each time, and on the deep crate of 2026-10-12 each of 52 parts read its 51 neighbours' surfaces again."""
        from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

        placed = (float(scale), np.asarray(transform, dtype=np.float64).tobytes())
        held = self._named_surfaces.get(id(mask))
        if held is not None and held[0] is mask and held[1] is depth_map and held[2] == placed:
            return held[3]
        if any(entry[1] is not depth_map for entry in self._named_surfaces.values()):
            self._named_surfaces.clear()
        surface = named_surface(to_base_mm(np.asarray(mask).astype(bool), np.asarray(depth_map, dtype=np.float64) * scale,
                                           intrinsics.to_matrix(), transform))
        self._named_surfaces[id(mask)] = (mask, depth_map, placed, surface)
        return surface

    def _scene_support_mm(
        self,
        support_plane: SupportPlane | None,
        support_model: Any,
        cloud: Any,
        geometry_points_base_mm: np.ndarray | None,
        transform: np.ndarray,
        telemetry: dict,
    ) -> float | None:
        """The support the part stands on, BASE z: the plane the pick resolved, raised to the support model's highest
        local reading under the part's foot where one lies higher (design 3.8). ``None`` where neither says."""
        plane = (float(support_plane.offset_mm)
                 if support_plane is not None and support_plane.frame is Frame.BASE else None)
        under: float | None = None
        if support_model is not None:
            if geometry_points_base_mm is not None and np.asarray(geometry_points_base_mm).size:
                foot = np.asarray(geometry_points_base_mm, dtype=np.float64).reshape(-1, 3)
            elif not cloud.is_empty:
                foot = (np.asarray(cloud.points_mm, dtype=np.float64).reshape(-1, 3) @ transform[:3, :3].T
                        + transform[:3, 3])
            else:
                foot = np.zeros((0, 3))
            if foot.shape[0]:
                value = support_model.height_under(foot)
                under = None if value is None else float(value)
        if under is not None and (plane is None or under > plane):
            telemetry["scene_support_mm"] = round(under, 2)
            telemetry["scene_support_from"] = "support_model"
            return under
        if plane is not None:
            telemetry["scene_support_mm"] = round(plane, 2)
            telemetry["scene_support_from"] = "plane"
        return plane

    def _keep_off_what_the_guard_holds(
        self,
        ranked: Sequence[GraspScoreBreakdown],
        *,
        held_solids: Sequence[Any],
        gripper_model: GripperGeometryStrategy | None,
        transform: np.ndarray,
        telemetry: dict,
        seen_boxes: Sequence[Any] = (),
    ) -> list[GraspScoreBreakdown]:
        """Drop the candidates whose open hand stands nearer a support's solid, a declared box or a neighbour's box than
        the guard keeps.

        The guard holds a support's solid as a camera box and keeps ``perceived_min_distance_mm`` from it, and keeps
        ``min_distance_mm`` from a declared box. A candidate whose envelope comes nearer would be refused there, and a
        try spent on it is a try not spent on one the guard admits. Counted as ``rejected_support`` and
        ``rejected_declared``; the nearest held solid and its distance are stamped for the sentence. A neighbour's box
        (``seen_boxes``, the boxes the camera world builds of what it saw beside the part) keeps
        ``perceived_min_distance_mm`` too, counted as ``rejected_collision``: a neighbour the camera saw, which a
        blocker cleared or a push may free. A candidate planned for jaws that open only as needed (lever G,
        ``score_components["support_footprint"]["open_width_mm"]``) is judged with its fingers that wide open and
        ``FINGER_OPENING_SLACK_MM``, where the guard holds them as it comes down, a grasp from inside a bore with them
        at its wall (``types.grasp_point.fingers_held_at``); every other fully open.
        """
        rules = self._scene_obstacles
        assert rules is not None  # noqa: S101 (the caller checks)
        telemetry["rejected_support"] = 0
        telemetry["rejected_declared"] = 0
        telemetry["scene_support_distance_mm"] = float(rules.support_distance_mm)
        telemetry["scene_declared_distance_mm"] = float(rules.declared_distance_mm)
        if not ranked or (not held_solids and not rules.declared_boxes and not seen_boxes):
            return list(ranked)
        rotation_cb, translation_cb = transform[:3, :3], transform[:3, 3]
        poses = [(rotation_cb @ np.asarray(score.pose.position_mm, dtype=np.float64) + translation_cb,
                  rotation_cb @ np.asarray(score.pose.rotation_matrix, dtype=np.float64)) for score in ranked]
        openings = [self.max_grip_mm if planned is None else min(planned + FINGER_OPENING_SLACK_MM, self.max_grip_mm)
                    for planned in (fingers_held_at(score) for score in ranked)]
        verdicts: list[Any] = [None] * len(ranked)
        for opening in sorted(set(openings)):
            at = [n for n, width in enumerate(openings) if width == opening]
            for n, verdict in zip(at, envelope_verdicts(
                    [poses[n] for n in at],
                    gripper_model=gripper_model if gripper_model is not None else ParallelJawGripperModel(),
                    open_width_mm=opening, solids=held_solids, support_distance_mm=rules.support_distance_mm,
                    declared_boxes=rules.declared_boxes, declared_distance_mm=rules.declared_distance_mm,
                    seen_boxes=seen_boxes, seen_distance_mm=rules.support_distance_mm,
                    fingers_to_the_reading=bool(rules.fingers_to_the_reading),
                    finger_floor_drop_mm=float(rules.finger_floor_drop_mm))):
                verdicts[n] = verdict
        kept: list[GraspScoreBreakdown] = []
        for score, verdict in zip(ranked, verdicts):
            if verdict.refused == "declared":
                telemetry["rejected_declared"] += 1
                telemetry["scene_declared_nearest"] = verdict.nearest
            elif verdict.refused == "support":
                telemetry["rejected_support"] += 1
                telemetry["scene_support_nearest"] = verdict.nearest
            elif verdict.refused == "seen":
                telemetry["rejected_collision"] = int(telemetry.get("rejected_collision", 0) or 0) + 1
                telemetry["scene_seen_nearest"] = verdict.nearest
            else:
                kept.append(score)
        finite = [v.distance_mm for v in verdicts if math.isfinite(v.distance_mm)]
        if finite:
            telemetry["scene_held_least_mm"] = round(min(finite), 2)
        return kept

    def _apply_collision_filters(
        self,
        ranked: Sequence[GraspScoreBreakdown],
        *,
        scene_points_mm: np.ndarray | None,
        support_plane: SupportPlane | None,
        workspace: WorkspaceBox | None,
        gripper_model: GripperGeometryStrategy | None,
        min_table_clearance_mm: float,
        collision_margin_mm: float,
    ) -> tuple[list[GraspScoreBreakdown], dict]:
        """Reject candidates that collide, hit the table, or exit the box.

        Operates in the candidates' frame (camera frame at this point in the pipeline). The returned
        telemetry counts the rejection reason for each dropped candidate so ops can see which stage is
        the bottleneck without re-running the calculator.

        Delegates to `collision.candidate_filter.filter_candidates`, which is the one rejection stage
        both grasp generators run. A second implementation would drift from it.

        The `support_plane_offset_camera_mm` key keeps its name because the plane reaching this method
        is always camera-frame by the time `compute()` is done with it; the shared stage derives the
        tag from the plane itself, which is the same string here.
        """
        outcome = filter_candidates(
            [score.pose for score in ranked],
            scene_points_mm=scene_points_mm,
            support_plane=support_plane,
            workspace=workspace,
            gripper_model=gripper_model,
            min_table_clearance_mm=min_table_clearance_mm,
            collision_margin_mm=collision_margin_mm,
        )
        kept = [ranked[index] for index in outcome.kept]
        # The rejection counters stay int-valued at runtime: the dict is float-typed only so the
        # clearance stamps can share it, and a caller reading `rejected_table` expects an int.
        stage: dict[str, float] = {}
        for key, value in outcome.telemetry.items():
            stage[key] = int(value) if key in REJECTION_KEYS else value
        return kept, stage

    def _to_grasp_point(
        self,
        score: GraspScoreBreakdown,
        segmentation_metadata: dict | None = None,
    ) -> GraspPoint:
        pose = score.pose
        metadata = {
            **pose.metadata,
            "total_score": round(score.total_score, 4),
            "geometric_score": round(score.geometric_score, 4),
            "stability_score": round(score.stability_score, 4),
            "reachability_score": round(score.reachability_score, 4),
            "pose_confidence": round(float(pose.confidence), 4),
            "score_components": score.components,
        }
        if segmentation_metadata:
            # Stored under a dedicated sub-dict so the GraspPoint can
            # always be traced back to the SAM2 boundary input without
            # colliding with grasp-internal metadata keys.
            metadata["segmentation"] = dict(segmentation_metadata)
        return GraspPoint(
            position=pose.position_mm,
            approach=pose.approach_axis,
            axis=pose.closing_axis,
            grip_width_mm=pose.grip_width_mm,
            score=float(np.clip(score.total_score, 0.0, 1.0)),
            frame=GraspFrame(pose.frame.value),
            label=str(pose.metadata.get("label", "grasp")),
            metadata=metadata,
        )

    @staticmethod
    def _to_base_frame_only(
        candidates: list[GraspPoint],
        T_cam_to_base: np.ndarray,
    ) -> list[GraspPoint]:
        rotation = T_cam_to_base[:3, :3]
        translation = T_cam_to_base[:3, 3]
        out: list[GraspPoint] = []
        for grasp in candidates:
            out.append(
                GraspPoint(
                    position=rotation @ grasp.position + translation,
                    approach=rotation @ grasp.approach,
                    axis=rotation @ grasp.axis,
                    grip_width_mm=grasp.grip_width_mm,
                    score=grasp.score,
                    frame=GraspFrame.BASE,
                    label=grasp.label,
                    metadata={**grasp.metadata, "source_frame": "camera"},
                )
            )
        return out
