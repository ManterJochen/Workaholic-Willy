"""Geometry-first grasp calculator compatibility adapter.

The stack runs in five stages:

1. masked depth to metric point cloud,
2. local surface normals,
3. antipodal contact-pair search,
4. 6D grasp-pose generation,
5. transparent geometric/stability/reachability scoring.

The robot pipelines depend on the public :class:`GraspCalculator` name
and its small ``compute(...)`` contract, which returns
:class:`GraspPoint` objects.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

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
from src.robot.grasping.generation.support_footprint import SupportFootprintJaw
from src.robot.grasping.contacts import (
    ContactPair,
    dense_surface_samples,
    scene_collision_cloud,
)
from src.robot.grasping.geometry import (
    CameraIntrinsics,
    CloudOutlierConfig,
    apply_cloud_outlier_filter,
    depth_unit_to_mm,
    masked_point_cloud,
    validate_transform,
)
from src.robot.grasping.planning import (
    GraspPose,
    IKResult,
    IKService,
)
from src.robot.grasping.planning.reachability import IKQualityMetrics
from src.robot.grasping.scoring import (
    FeasibilityInputs,
    FeasibilityScoreConfig,
    GeometricScoreConfig,
    GraspScoreBreakdown,
    GraspScoreWeights,
    ReachabilityScoreConfig,
    WorkspaceBox,
    feasibility_grasp_score,
    rank_grasp_poses,
    rerank_breakdowns_with_feasibility,
)
from src.robot.grasping.visualization import DebugDrawConfig
from src.robot.grasping.types.perception import SegmentationLike
from src.utility.log_cfg import create_logger

from src.robot.grasping.generation._candidate_generator import (
    _GraspCandidateGenerator,
    _mask_non_convexity as _mask_non_convexity,  # re-export: the name stays importable here
)
from src.robot.grasping.generation._debug_renderer import _DebugRenderer
from src.robot.grasping.generation._camera_geometry import _SharedGeometryUtil
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.generation.mask_analyzer import MaskAnalyzer
from src.robot.grasping.types.modes import (
    GraspSamplingMode,
    mode_to_dense_sampling,
    resolve_grasp_sampling_mode,
)
from src.robot.grasping.scoring.topology_risk import topology_risk_from_mask
from src.robot.grasping.scoring.occlusion import approach_clearance_mm, occlusion_ratio
from src.robot.grasping.scoring.corridor import (
    CorridorAnalysisConfig,
    CorridorAnalysisInputs,
    analyze_corridor,
)
from src.robot.grasping.generation.deformables import DeformableClass, DeformableHandlingStrategy
from src.robot.grasping.generation._calculator_validation import validate_calculator_args

if TYPE_CHECKING:
    from src.robot.grasping.planning.multifinger import (
        FingerKinematicSpec,
        MultiContactGrasp,
        MultiContactGraspPlanner,
    )
from src.robot.grasping.scoring.semantic_policy import SemanticPolicy


# RL shadow-only feasibility config: enable the two IK-quality sub-signals (Jacobian conditioning +
# joint-margin) so ``feasibility_grasp_score`` maps the per-candidate ``IKQualityMetrics`` onto a
# [0, 1] reachability score. Used only to stamp ``metadata["shadow"]["feasibility_score"]`` on the
# RL ranking rows: a pre-execution predictor of the descent-timeout and reachability failures the
# deterministic geometric score is blind to. It never feeds ``GraspPoint.score``, the ranking order,
# or the success-model feature vector.
_SHADOW_FEASIBILITY_CONFIG = FeasibilityScoreConfig(ik_quality_enabled=True, joint_margin_enabled=True)

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
        axis_end_fraction: float = 0.4,
        camera_matrix: np.ndarray | None = None,
        normal_radius_mm: float = 18.0,
        max_candidates: int = 12,
        geometry_voxel_size_mm: float = 8.0,
        max_geometry_points: int = 700,
        dense_runtime_budget_ms: float = 150.0,
        dense_non_convexity_threshold: float = 0.15,
        topology_risk_threshold: float | None = None,
        friction_coefficient: float | None = None,
        heavy_occlusion_threshold: float | None = None,
        deformable_strategy: DeformableHandlingStrategy | None = None,
        corridor_risk_per_candidate: bool = False,
        oblique_approach: bool = False,          # generate grasps at multiple approach angles (top-down
        oblique_tilt_deg: float = 30.0,          # + tilted variants) so an oblique gap-threading grasp is
        oblique_azimuths: int = 4,               # proposed where a top-down one rams. Default-off.
        ik_service: "IKService | None" = None,
        grasp_depth_reference: str = "centre",
        grasp_top_penetration_mm: float = 0.0,
        grasp_top_quantile: float = 10.0,
        depth_band_mm: float | None = None,
        depth_band_near_pct: float = 10.0,
        cloud_outlier_filter: "CloudOutlierConfig | None" = None,
        feasibility_weight: float = 0.0,
        feasibility_config: "FeasibilityScoreConfig | None" = None,
        gripper_model: GripperGeometryStrategy | None = None,
        # When the silhouette is near-isotropic its principal axis is numerically degenerate, so the
        # closing direction it implies is noise. This spends that free yaw on the direction the arm
        # can actually reach instead: radial, from base to grasp. Needs a BASE-frame transform to mean
        # anything. Default False keeps the path byte-identical. See generation/_isotropic_closing.py.
        isotropic_radial_closing: bool = False,
        # A parallel jaw closes across an object that stands on something, so its closing axis lies in
        # the support plane: 98.9 % of the datagen reference's 28 332 true jaw labels are exactly
        # horizontal. Zeroing z in the CAMERA frame lands the axis in the image plane instead; on this
        # rig's 36.87 deg obliques that is a median 8.78 deg off horizontal on 94.4 % of executed
        # candidates, and it is why ``not_antipodal`` is 42 % of wrong candidates on objects standing
        # completely free. Levelling is worth +10 pp of top-1 (22.0 -> 32.1 %) over all 270 reference
        # scenes, so the default is True; False reproduces the unlevelled axis byte-for-byte. Needs a
        # CAMERA->BASE transform; without one the direction of "up" is unknown and nothing is levelled.
        # See generation/_camera_geometry.py::level_axis_to_support_plane.
        level_closing_to_support_plane: bool = True,
        # Support-footprint enumeration replaces the silhouette geometry stage: it reconstructs the
        # target as its footprint extruded onto the support plane and solves for an anchor height that
        # clears the table, instead of anchoring on the visible surface and being filtered afterwards.
        # Measured top-1 on the D5 reference: 58.5 % against the silhouette path's 27.2 %, better on
        # all four scene families, precision 77.5 % against 8.9 %. Through the real calculator at the
        # D5 gate it moves top-1 24.29 -> 43.50 % and precision 22.08 -> 63.59 %, better on packed,
        # pile and sparse; ``bin`` falls 12.5 -> 8.3 % on that 24-object subset while measuring
        # 27.6 % against 14.5 % on the full 152-object set. That is why the default is True; False
        # restores the silhouette stage. It needs a BASE-frame support plane and a CAMERA->BASE
        # transform, and without both it silently cannot run, so the path is byte-identical when the
        # inputs are missing. See support_footprint.py.
        support_footprint_geometry: bool = True,
        support_footprint_inflate_mm: float = 0.0,
        # Keep the silhouette candidates when SFE produced nothing (``reconstruct_support_prism`` in
        # support_footprint.py gives up below 20 cloud points), instead of letting its empty list
        # replace them. Default False. Over the 354 jaw-graspable reference objects it moves top-1
        # 43.5 -> 43.8 %, which is one object: paired, 1 rescued and 0 destroyed. It hands candidates
        # to 108 more objects and 106 of those gain only invalid ones (no-valid-candidate 85 -> 191),
        # dragging precision 63.6 -> 59.3 % and coverage-per-candidate 70.0 -> 51.2 %. Where SFE
        # abstains the silhouette cannot grasp the object either, so refusing is the answer.
        support_footprint_fallback: bool = False,
        # Give SFE the target cloud at full resolution instead of the 8 mm-voxelised one built for
        # the contact search. See the call site: SFE refuses below 20 points, the point count is set
        # by an unrelated downsample knob, and SFE re-voxelises at 3 mm anyway, so the pre-downsample
        # is both arbitrary and destructive. Over the 354 jaw-graspable reference objects: top-1
        # 43.5 -> 46.3 %, candidate precision 63.6 -> 64.1 %, ranking gap 41 -> 38 objects, at
        # +5.2 ms median (79.4 -> 84.6); paired per object, 24 rescued against 10 lost, net +14.
        # Default True on that measurement; False restores the voxelised input.
        support_footprint_full_resolution: bool = True,
        # Let SFE reason about the palm, which the collision envelope carries and the planner does
        # not. Two models of one gripper: SFE plans the table clearance from finger points (27 mm
        # wide), ``ParallelJawGripperModel`` re-checks it with the palm (70 mm wide) included, so on
        # a tilted grasp the palm hangs below everything the planner looked at. The gap is 8.00 mm
        # offline over a 50-120 mm height sweep and 8.0 mm on-box (SFE planning 6.0 mm of clearance
        # where the envelope granted -2.0), which is exactly the 35.0 vs 27.0 mm the two height
        # requirements differ by at 90 degrees of tilt.
        #
        # The direction is fail-safe, so this buys no safety, and it buys no candidates either.
        # Measured on the D5 reference (n=1555 view-objects with a jaw label, all 270 scenes), paired
        # per object:
        #
        #   sfe       -> sfe_palm         top-1 41.9 -> 41.4 %   0 rescued,  9 lost
        #   sfe_fused -> sfe_fused_palm   top-1 54.7 -> 54.4 %   3 rescued,  8 lost
        #
        # Purely subtractive, and precision does not rise to pay for it (65.7 -> 64.8 % plain,
        # 69.8 -> 69.7 % fused). The relocation the height solve could buy does not materialise: the
        # heights it lands on are already taken by the finger requirement or are above the object.
        # Default False.
        #
        # The losses are the open question. The reference calls a grasp valid by matching an analytic
        # D5 label and not by any table check, so the 9 lost grasps match ground truth. Either those
        # labels do not model the palm (they are generated for a floating-base 2F-85, which has no
        # collision in Isaac at all) or this palm term is stricter than the real hardware. The flag
        # makes that answerable once the reference labels are settled, and the 8 mm it closes is real
        # (8.00 -> 0.00 mm of planner optimism over a 50-120 mm height sweep).
        support_footprint_palm_aware: bool = False,
        # An experiment knob, not a config key, and deliberately not in the schema. SFE blends five
        # measured margins into one rank, and the first, `cone_slack` (`1 - contact_angle/cone`), is
        # the only one that measurably separates: on 414 `sfe_fused` candidate lists the contact angle
        # alone orders valid before invalid at within-list AUC 0.6907 against the full blend's 0.5698,
        # and ranking by it alone measured top-1 86.23 % against 83.09 % (26 lists rescued, 13 lost,
        # +3.14 pp, McNemar chi2 3.69 on 39 discordant; p ~ 0.055, suggestive and not significant, on
        # a denominator that is not the ladder's). None keeps the shipped blend, byte-identical.
        support_footprint_score_weights: "tuple[float, float, float, float, float] | None" = None,
    ) -> None:
        validate_calculator_args(
            min_grip_width_mm=min_grip_width_mm,
            max_grip_width_mm=max_grip_width_mm,
            normal_radius_mm=normal_radius_mm,
            max_candidates=max_candidates,
            geometry_voxel_size_mm=geometry_voxel_size_mm,
            max_geometry_points=max_geometry_points,
            dense_runtime_budget_ms=dense_runtime_budget_ms,
            dense_non_convexity_threshold=dense_non_convexity_threshold,
            topology_risk_threshold=topology_risk_threshold,
            friction_coefficient=friction_coefficient,
            heavy_occlusion_threshold=heavy_occlusion_threshold,
            deformable_strategy=deformable_strategy,
            grasp_depth_reference=grasp_depth_reference,
            grasp_top_penetration_mm=grasp_top_penetration_mm,
            grasp_top_quantile=grasp_top_quantile,
            depth_band_mm=depth_band_mm,
            depth_band_near_pct=depth_band_near_pct,
            cloud_outlier_filter=cloud_outlier_filter,
            feasibility_weight=feasibility_weight,
            feasibility_config=feasibility_config,
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
        self.approach_cam = approach / norm

        self.topology_risk_threshold: float | None = (
            float(topology_risk_threshold)
            if topology_risk_threshold is not None
            else None
        )
        # Optional Coulomb friction coefficient. When set, every
        # parallel-jaw contact pair is certified analytically and the
        # certificate is stamped onto each candidate's metadata.
        self.friction_coefficient: float | None = (
            float(friction_coefficient)
            if friction_coefficient is not None
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
        # setting a strategy (e.g. ``RefuseDeformableStrategy()``) engages the
        # gate so non-rigid targets short-circuit with ``DEFORMABLE_ROUTING_REQUIRED``.
        self.deformable_strategy: DeformableHandlingStrategy | None = (
            deformable_strategy
        )
        # Optional per-candidate corridor-risk producer, off by default. It runs when this flag is
        # set or when the caller passes ``corridor_config`` to ``compute()``; with neither there are
        # no analyze_corridor calls, no metadata stamp, and ranking and telemetry are unchanged.
        # When it runs, analyze_corridor is called once per surviving candidate and stamps
        # ``corridor_blockage_confidence`` in [0,1] (higher = more blocked = more uncertain) on each
        # GraspPoint.metadata. It is the non-circular per-candidate uncertainty signal the re-ranker
        # consumes. Cost: N depth-marches.
        self.corridor_risk_per_candidate: bool = bool(corridor_risk_per_candidate)
        # Oblique/multi-angle approach generation (default-off; see _candidate_generator).
        self._oblique_approach: bool = bool(oblique_approach)
        self._oblique_tilt_deg: float = float(oblique_tilt_deg)
        self._oblique_azimuths: int = int(oblique_azimuths)
        # Optional default IK reachability oracle (vendor-injected). When set, every compute() that is
        # not handed a per-call ``ik_service`` uses this one, so the live pick path (orchestrator) gets
        # the IK filter (e.g. drop near-singular grasps) without threading a service through every
        # call. ``None`` (default) applies no IK filter.
        self._default_ik_service: "IKService | None" = ik_service
        self.max_grip_mm = float(max_grip_width_mm)
        self.min_grip_mm = float(min_grip_width_mm)
        self.axis_end_frac = float(np.clip(axis_end_fraction, 0.05, 0.5))
        self.normal_radius_mm = float(normal_radius_mm)
        self.max_candidates = int(max_candidates)
        self.geometry_voxel_size_mm = float(geometry_voxel_size_mm)
        self.max_geometry_points = int(max_geometry_points)
        self.dense_runtime_budget_ms = float(dense_runtime_budget_ms)
        self.dense_non_convexity_threshold = float(dense_non_convexity_threshold)
        # Silhouette grasp-depth reference + penetration. The defaults ("centre", 0.0) give a
        # median-referenced grasp depth with no penetration. "top" references a near-surface quantile
        # of the masked depth; the penetration descends every silhouette candidate below its
        # referenced surface so the fingers wrap the object instead of resting on it. The dense
        # path is unchanged by both: it builds its own cloud and finds its own contacts, so a
        # silhouette-level depth reference does not apply to it.
        #
        # That sentence used to say the dense path "uses real point-cloud contacts", which was the
        # word for something it could not do. While the producers replaced the depth under each mask
        # with one scalar, the dense sampler measured no curvature and found no opposing normals on
        # any object: 660 samples and 0 pairs, because every normal on a plane points the same way.
        # The contacts are real now because the input is.
        self._grasp_depth_reference = str(grasp_depth_reference)
        self._grasp_top_penetration_mm = float(grasp_top_penetration_mm)
        self._grasp_top_quantile = float(grasp_top_quantile)
        # Optional robust depth-band cleaning of the under-mask depth. On real (non-uniform) depth,
        # mask-edge table bleed leaves finite pixels deeper than the object top, pulling the median
        # (centre reference) toward the table; the band keeps only [near surface, near + band] so centre
        # re-references the top. Default None is off: valid_depth and the clouds are unchanged.
        self._depth_band_mm = None if depth_band_mm is None else float(depth_band_mm)
        self._depth_band_near_pct = float(depth_band_near_pct)
        # Optional point-cloud outlier filter (kNN-distance / radius). Applied to the geometry cloud +
        # the dense samples before normal estimation. A real-hardware robustness lever (stereo/RGB-D
        # speckle); a no-op in the clean Isaac sim. Default None never calls the filter, which keeps
        # the path byte-identical.
        self._cloud_outlier_filter = cloud_outlier_filter
        # The feasibility ranking axis. Off unless feasibility_weight > 0 and a config is given. When
        # on, the IK survivors are re-ranked with the feasibility term using the per-candidate IK
        # quality that only exists after the reachability filter; when off, no feasibility inputs are
        # fed, so the score layout and the success-model feas_* features stay byte-identical (NaN).
        # The weights reuse the default geo/stab/reach (matching the up-front rank) plus this weight.
        # Trap: enabling the ik_quality sub-signal moves the UR5e overhead sim pick 10/10 -> 0/10 at
        # every weight, because the good overhead top-down grasp is itself near-singular (high cond#),
        # so "demote high-cond#" demotes the good grasp. It is not a validated improvement for that
        # cell; default-off keeps it byte-identical.
        self._feasibility_config = feasibility_config
        self._feasibility_weights: GraspScoreWeights | None = (
            GraspScoreWeights(feasibility=float(feasibility_weight))
            if (feasibility_weight > 0.0 and feasibility_config is not None)
            else None
        )
        # Default gripper collision envelope for this calculator (parallel-jaw or suction). A per-call
        # ``compute(gripper_model=...)`` wins; when both are None the checker falls back to a
        # ParallelJawGripperModel() default, so leaving this unset is byte-identical.
        self._gripper_model = gripper_model
        self._isotropic_radial_closing = bool(isotropic_radial_closing)
        self._level_closing_to_support_plane = bool(level_closing_to_support_plane)
        self._support_footprint_geometry = bool(support_footprint_geometry)
        self._support_footprint_inflate_mm = float(support_footprint_inflate_mm)
        self._support_footprint_fallback = bool(support_footprint_fallback)
        self._support_footprint_full_resolution = bool(support_footprint_full_resolution)
        self._support_footprint_palm_aware = bool(support_footprint_palm_aware)
        # The float() sweep is what makes YAML's ints and numpy scalars behave; naming the five
        # elements keeps the shape the callee promises. Widening it to "some floats" would let a
        # four-weight config reach the scorer silently short.
        self._support_footprint_score_weights: "tuple[float, float, float, float, float] | None" = (
            None if support_footprint_score_weights is None
            else (float(support_footprint_score_weights[0]),
                  float(support_footprint_score_weights[1]),
                  float(support_footprint_score_weights[2]),
                  float(support_footprint_score_weights[3]),
                  float(support_footprint_score_weights[4])))
        # Memory of whether the previous dense call exceeded the
        # runtime budget. When True and the current call's
        # ``dense_sampling`` is left to "auto", the heuristic suppresses
        # dense regardless of the convexity / clutter signals, so the
        # budget overrun is not paid twice in a row.
        self._dense_last_overran: bool = False
        _camera_matrix = (
            np.asarray(camera_matrix, dtype=np.float64)
            if camera_matrix is not None
            else None
        )
        if _camera_matrix is not None and _camera_matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix must be (3, 3), got {_camera_matrix.shape}")
        # The shared camera-geometry helper owns the calibrated K + the loud-once synthetic-intrinsics
        # latch (the fallback synthesizes focal=width/2, which back-projects to metrically wrong 3D:
        # fine for silhouette ranking, but never trust a BASE-frame metric grasp on it; pass a real
        # camera_matrix on hardware). Constructed once here and shared with the generator so the
        # warning fires once ever.
        self._geom = _SharedGeometryUtil(camera_matrix=_camera_matrix, logger=self.logger)
        # The candidate generator owns the pose-generation cluster. Loose injection of the (already
        # validated / normalised) config; the cross-call dense-overrun latch stays facade-owned and is read
        # via a getter closure (single source of truth, no setter).
        self._generator = _GraspCandidateGenerator(
            geom=self._geom,
            approach_cam=self.approach_cam,
            min_grip_mm=self.min_grip_mm,
            max_grip_mm=self.max_grip_mm,
            max_candidates=self.max_candidates,
            max_geometry_points=self.max_geometry_points,
            normal_radius_mm=self.normal_radius_mm,
            axis_end_frac=self.axis_end_frac,
            dense_non_convexity_threshold=self.dense_non_convexity_threshold,
            friction_coefficient=self.friction_coefficient,
            dense_overran_getter=lambda: self._dense_last_overran,
            oblique_approach=self._oblique_approach,
            oblique_tilt_deg=self._oblique_tilt_deg,
            oblique_azimuths=self._oblique_azimuths,
        )

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
        # Opt-in switch so the live pick path (``pick_loop``) forwards the
        # perception frame's rgb into ``compute`` and the grasp-point overlay above is rendered
        # per pick. Default False: pick_loop forwards ``rgb_image=None``, which is byte-identical
        # and costs no render; a runner flips it on for ``--debug-frames``.
        self.render_debug_images: bool = False


    @property
    def camera_matrix(self) -> "np.ndarray | None":
        """The calibrated intrinsics this calculator was built with, or ``None``.

        Read by the orchestrator (``loop/pick_loop.py``) to back-project this view's target cloud for
        the support-plane refinement under ``support.refine_from_target``.

        It returns the calibrated K and never the synthesized fallback. ``self._geom.intrinsics()``
        invents ``focal = width / 2`` when no calibration was supplied, which is good enough to rank
        silhouettes and metrically wrong in 3-D; the support plane is a BASE-frame metric quantity, so
        this branch stays down in a cell built without real intrinsics. That is why this reads ``_K``
        directly rather than calling ``intrinsics()``.

        Read-only: a settable attribute could disagree with the K the generator is already
        back-projecting with, and nothing would notice.
        """
        return self._geom._K

    def _resolve_grasp_depth(
        self, valid_depth: np.ndarray, median_depth_mm: float, scale: float
    ) -> float:
        """The silhouette grasp-depth reference (camera-frame mm, before the penetration descend).

        ``"centre"`` (default) returns ``median_depth_mm`` unchanged. ``"top"`` returns a near-surface
        quantile of the masked depth (a smaller depth is nearer the object top); a low percentile references
        the top robustly, ignoring the few nearest speckle pixels a pure ``min`` would latch onto.

        "top" is worse, and monotonically so; the default stays "centre". The 2F-85 fingertip reaches
        28.72 mm past the grasp centre, so a higher anchor buys table clearance. On the gate subset
        (n=354, ``neighbours`` rung):

        ==============  ==========  ===========  ========
        reference       precision   top-1        coverage
        ==============  ==========  ===========  ========
        centre (q50)    22.08 %     24.29 %      28.53 %
        top, q=25       19.44 %     22.03 %      27.68 %
        top, q=10       18.24 %     21.47 %      27.40 %
        ==============  ==========  ===========  ========

        It does exactly what it promises and that is not enough: rank-0 "no candidate" falls 65.4 % ->
        60.2 %, so 5.2 pp more objects do clear the table, but every failure reason rises with it
        (``anchor_outside`` 0.1 -> 1.0 %, ``below_table`` itself 5.6 -> 7.2 %) and valid falls 8.0 ->
        7.1 %. The contact patch runs 14.4 mm behind the grasp centre and 23.6 mm in front, so lifting
        the anchor onto the top face puts most of the patch in the air above the object: it buys table
        clearance by giving up grip, one-for-one and then some. The estimate that 47.3 % of
        below_table candidates are savable by a higher anchor is an upper bound over clearance alone
        and does not account for the grip cost."""
        if self._grasp_depth_reference == "top":
            near_depth = float(np.percentile(valid_depth, self._grasp_top_quantile))
            return near_depth * scale
        return median_depth_mm

    def _apply_depth_band(
        self, valid_depth: np.ndarray, scale: float
    ) -> "tuple[np.ndarray, float | None]":
        """Optionally clean the under-mask depth band (for real non-uniform depth).

        Returns ``(valid_depth_for_reference, cloud_max_depth_mm)``. Off (``depth_band_mm is None``) returns
        ``(valid_depth, None)`` unchanged. On: ``near = percentile(valid_depth, near_pct)`` is the robust
        object-top surface; keep only depths ``<= near + depth_band_mm`` (rejecting the deeper mask-edge table
        bleed), and cap the geometry/dense clouds at the same ``max_depth_mm`` (non-empty keep set guaranteed)."""
        if self._depth_band_mm is None:
            return valid_depth, None
        near_mm = float(np.percentile(valid_depth, self._depth_band_near_pct)) * scale
        max_keep_mm = near_mm + self._depth_band_mm
        banded = valid_depth[valid_depth * scale <= max_keep_mm]
        if banded.size == 0:  # defensive: a degenerate band leaves the reference unfiltered, no cap
            return valid_depth, None
        return banded, max_keep_mm

    def compute(
        self,
        segmentation: SegmentationLike,
        depth_map: np.ndarray,
        T_cam_to_base: Transform | np.ndarray | None = None,
        pixel_to_mm: float | None = None,
        point_3d_cam: np.ndarray | None = None,
        unit: str = "mm",
        *,
        camera_to_base: Transform | np.ndarray | None = None,
        scene_points_mm: np.ndarray | None = None,
        rigid_obstacle_points_mm: np.ndarray | None = None,
        geometry_points_base_mm: np.ndarray | None = None,
        support_plane: SupportPlane | None = None,
        workspace: WorkspaceBox | None = None,
        gripper_model: GripperGeometryStrategy | None = None,
        # The operator's corridor geometry, from ``grasping.occlusion``. None (default) leaves the
        # per-candidate corridor producer as it is: it runs only when the
        # ``corridor_risk_per_candidate`` constructor flag is set, and with hardcoded defaults.
        #
        # It arrives as a parameter rather than a constructor argument on purpose. A constructor
        # argument that no caller passes, on any path, sim or real, leaves the config block dead;
        # the feasibility block is that case. A per-call parameter is supplied by the orchestrator
        # (like ``gripper_model`` and ``support_plane``), which is built from config in one place,
        # so wiring it once wires it for every caller.
        corridor_config: "CorridorAnalysisConfig | None" = None,
        # The operator's feasibility re-rank, from ``grasping.feasibility``. None (default) uses the
        # constructor's values. Supplied per call for the same reason as ``corridor_config``: the
        # constructor arguments this block also carries are passed by no caller, on any path, sim or
        # real.
        feasibility_config: "FeasibilityScoreConfig | None" = None,
        feasibility_weight: float | None = None,
        min_table_clearance_mm: float = 5.0,
        collision_margin_mm: float = 0.0,
        rgb_image: np.ndarray | None = None,
        debug_label: str | None = None,
        debug_config: DebugDrawConfig | None = None,
        seed: int | None = None,
        ik_service: IKService | None = None,
        dense_sampling: bool | None = None,
        grasp_sampling_mode: GraspSamplingMode | bool | str | None = None,
        other_object_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
        semantic_policy: SemanticPolicy | None = None,
        deformable_class: DeformableClass | None = None,
    ) -> list[GraspPoint]:
        """Generate ranked grasp candidates.

        ``T_cam_to_base`` returns base-frame candidates only, which is the
        contract the physical robot pipelines use. The same value may be
        passed as ``camera_to_base`` to make the frame direction explicit.
        A typed ``Transform`` must be ``CAMERA -> BASE``.

        Optional ``support_plane`` / ``scene_points_mm`` arguments enable
        collision-aware filtering before ranking, and ``workspace`` feeds
        the reachability scorer with a real workspace box. When any of
        these is provided, the corresponding stage is exposed in the
        ``last_telemetry`` dict for ops debugging.
        """
        # Fall back to the calculator's default IK oracle when no per-call service was passed, so the
        # live pick path (the orchestrator calls ``compute_result``, which calls ``compute``) gets the
        # IK filter. None applies no filter, which is byte-identical.
        if ik_service is None:
            ik_service = self._default_ik_service
        # Same pattern for the gripper collision envelope: a per-call model wins, else the calculator's
        # configured default, else (both None) the checker's ParallelJawGripperModel() fallback.
        if gripper_model is None:
            gripper_model = self._gripper_model
        scale = depth_unit_to_mm(unit)
        if seed is not None:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError(f"seed must be a non-negative int, got {seed!r}")
            # Reserved for future stochastic samplers (dense surface
            # sampling, contact-point jitter). The current geometric and
            # antipodal stages are deterministic by construction; the
            # rng is built here so callers can rely on a single seeding
            # boundary today even though no sampler consumes it yet.
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
        if T_cam_to_base is not None and camera_to_base is not None:
            raise ValueError("Pass only one of T_cam_to_base or camera_to_base.")
        transform_input = camera_to_base if camera_to_base is not None else T_cam_to_base
        transform = (
            _validate_camera_to_base_transform(transform_input)
            if transform_input is not None
            else None
        )
        # The support normal, in the CAMERA frame. This is what the closing axis has to lie
        # perpendicular to; see _camera_geometry.level_axis_to_support_plane for why and for what the
        # error costs. It prefers the caller's actual ``support_plane`` normal, so a tilted tray or a
        # bin floor is handled by whoever knows about it, and falls back to BASE +Z, which is the
        # right answer for every cell whose table is level. Without a CAMERA->BASE transform there is
        # no way to know which direction is up, so nothing is levelled and the path is byte-identical.
        #
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
        up_cam: np.ndarray | None = None
        if self._level_closing_to_support_plane and transform is not None:
            # Already camera-frame by construction above. When no plane is given, BASE +Z is the right
            # answer for any level cell and gets rotated in the same way.
            up_cam = (np.asarray(plane_cam.normal, dtype=np.float64) if plane_cam is not None
                      else transform[:3, :3].T @ np.array([0.0, 0.0, 1.0]))
        # Reset per-call debug surface so a stale image cannot leak into
        # the next read of ``last_debug_image_png`` if this call returns early.
        self.last_debug_image_png = None
        # Snapshot SAM2/segmentation metadata so it can be attached to
        # every resulting GraspPoint for traceability.
        seg_meta = _segmentation_metadata(segmentation)
        seg_score = seg_meta.get("score") if seg_meta else None
        mask_confidence = float(seg_score) if isinstance(seg_score, (int, float)) else None
        telemetry: dict = {
            "candidates_silhouette": 0,
            "candidates_geometry": 0,
            "rejected_collision": 0,
            "rejected_table": 0,
            "rejected_workspace": 0,
            "final": 0,
            "best_score": 0.0,
        }

        # ``grasp_sampling_mode`` (typed enum / bool / str alias / None)
        # is the production-facing knob; ``dense_sampling`` is the legacy
        # bool surface and keeps working. A mode supplied by the operator
        # always wins; otherwise the legacy bool maps back to a mode so
        # telemetry is always meaningful (a literal ``True`` here means
        # ``DENSE_CLUTTER``, ``False`` means ``SINGLE_OBJECT``, ``None``
        # means ``AUTO``).
        if grasp_sampling_mode is not None:
            resolved_mode = resolve_grasp_sampling_mode(grasp_sampling_mode)
            dense_sampling = mode_to_dense_sampling(resolved_mode)
        else:
            resolved_mode = resolve_grasp_sampling_mode(dense_sampling)
        telemetry["grasp_sampling_mode"] = resolved_mode.value

        mask_array = np.asarray(segmentation.mask)
        mask_has_any = bool(np.any(mask_array)) if mask_array.size > 0 else False
        # What the mask was, stamped before anything can reject it.
        #
        # The mask rejection below is the first early return in compute(), and without these two keys
        # the telemetry names a stage and no cause: `candidates_silhouette: 0` is an initial value,
        # not a measurement. An empty mask and a mask the analyzer found too small have nothing in
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
        # not a diagnostic, and a run whose silhouette generator produces no poses returns at
        # `candidates_silhouette: 0`, before the cloud exists, so a probe placed next to the point
        # cloud never runs on that failure.
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
        # Optionally clean the under-mask depth band (default-off byte-identical). depth_confidence
        # above stays the original valid fraction; the band only re-references the depth + caps the clouds.
        valid_depth, cloud_max_depth_mm = self._apply_depth_band(valid_depth, scale)
        median_depth_mm = float(np.median(valid_depth)) * scale
        intrinsics = self._geom.intrinsics(depth_arr.shape)
        effective_pixel_to_mm = (
            float(pixel_to_mm)
            if pixel_to_mm is not None
            else self._geom.estimate_pixel_to_mm(median_depth_mm, depth_arr)
        )
        # Re-reference the silhouette centre depth (the pixel_to_mm scale above still uses the median).
        # Default "centre" makes grasp_base_depth_mm the median_depth_mm (byte-identical); the
        # penetration default 0.0 means no descend in the generator.
        grasp_base_depth_mm = self._resolve_grasp_depth(valid_depth, median_depth_mm, scale)

        silhouette_poses = self._generator.silhouette_contact_poses(
            analysis=analysis,
            depth_map=depth_arr,
            median_depth_mm=grasp_base_depth_mm,
            pixel_to_mm=effective_pixel_to_mm,
            point_3d_cam=point_3d_cam,
            scale_to_mm=scale,
            depth_confidence=depth_confidence,
            penetration_mm=self._grasp_top_penetration_mm,
            up_cam=up_cam,
        )
        telemetry["candidates_silhouette"] = len(silhouette_poses)
        telemetry["closing_levelled"] = up_cam is not None
        if not silhouette_poses:
            return self._finalize_empty(
                telemetry,
                reasons=(
                    GraspFailureReason.NO_CANDIDATES_GENERATED,
                    GraspFailureReason.RESCAN_RECOMMENDED,
                ),
                depth_confidence=float(depth_confidence),
                mask_confidence=mask_confidence,
            )

        cloud = masked_point_cloud(
            segmentation.mask,
            depth_arr,
            intrinsics,
            unit=unit,
            min_depth_mm=1.0,
            max_depth_mm=cloud_max_depth_mm,  # None when the band is off; byte-identical
            voxel_size_mm=self.geometry_voxel_size_mm,
        )
        # Where the target was seen, in BASE, whenever this view's own masked cloud is non-empty.
        # The SFE stage stamps its own ``support_footprint_cloud_z_*`` keys, but SFE is not the
        # only path: the silhouette generator runs without it, and a gate that went 10/10 -> 0/10
        # with cuRobo refusing a TCP target at z=16.7 mm raised no rejection counter at all: 3
        # candidates, all filters clean, and no telemetry saying what the object looked like. A
        # masked depth image can pick up support-plane pixels at the mask edge, which drags a grasp
        # to the object's base; the span between these two keys tells that apart from a short
        # object, and the two need opposite fixes. Stamped only when a transform is available,
        # because without one there is no BASE to report in, and skipped on the fused path when this
        # view's cloud is empty and the injected fused cloud carries the geometry instead; SFE's
        # keys cover that case.
        if transform is not None and not cloud.is_empty:
            _cloud_base_z = (
                np.asarray(cloud.points_mm, dtype=np.float64).reshape(-1, 3) @ transform[:3, :3].T
                + transform[:3, 3]
            )[:, 2]
            telemetry["target_cloud_z_min_mm"] = round(float(_cloud_base_z.min()), 2)
            telemetry["target_cloud_z_max_mm"] = round(float(_cloud_base_z.max()), 2)
        # (`target_depth_*` + `camera_base_z_mm` are stamped above the first early return, so a run
        # that dies before the cloud exists still reports what it saw.)
        # Optional outlier filter on the geometry cloud before contact search. None lets the same
        # array flow through with no filter call, which is byte-identical. Defensive: if the filter
        # would empty a non-empty cloud, keep the unfiltered points; a real speckle filter never
        # removes a whole object surface.
        geometry_points = cloud.points_mm
        if self._cloud_outlier_filter is not None and not cloud.is_empty:
            _keep = apply_cloud_outlier_filter(geometry_points, self._cloud_outlier_filter)
            if _keep.size > 0:
                geometry_points = geometry_points[_keep]
        # (default-off) a fused multi-view target cloud (BASE frame) overrides the single-view
        # geometry source so the antipodal generator searches real side-face contacts a top-down view
        # cannot see. That was only true of the cloud's provenance and not of its content until the
        # producers stopped flattening the depth under each mask: every camera contributed a sheet at
        # its own view of the top face, so fusing two views fused two sheets and no side face existed
        # to find. Transform BASE->CAMERA (the generator's frame) and force the geometry-first
        # path: the mask-based dense sampler rebuilds the cloud from depth+mask, so it cannot consume
        # a pre-built cloud. Default None leaves geometry_points and dense_decision unchanged, which
        # is byte-identical.
        _fused_geometry = False
        if geometry_points_base_mm is not None and transform is not None:
            _gb = np.asarray(geometry_points_base_mm, dtype=np.float64).reshape(-1, 3)
            if _gb.size > 0:
                _inv = np.linalg.inv(np.asarray(transform, dtype=np.float64))  # CAMERA->BASE inverted = BASE->CAMERA
                _fc = (_inv[:3, :3] @ _gb.T).T + _inv[:3, 3]
                if self._cloud_outlier_filter is not None and _fc.shape[0] > 0:
                    _fkeep = apply_cloud_outlier_filter(_fc, self._cloud_outlier_filter)
                    if _fkeep.size > 0:
                        _fc = _fc[_fkeep]
                geometry_points = _fc
                _fused_geometry = True
                telemetry["fused_geometry_points"] = int(_fc.shape[0])
                telemetry["geometry_source"] = "fused_multiview"
        # ``dense_sampling=None`` triggers automatic selection based on
        # clutter (other_object_masks present) and mask non-convexity
        # (1 - solidity > threshold). Explicit True/False is respected.
        dense_decision = False if _fused_geometry else self._decide_dense_sampling(
            dense_sampling=dense_sampling,
            mask=segmentation.mask,
            other_object_masks=other_object_masks,
            telemetry=telemetry,
        )
        if dense_decision:
            # Dense runtime budget with silhouette fallback.
            import time as _time

            t0 = _time.monotonic()
            samples = dense_surface_samples(
                segmentation.mask,
                depth_arr,
                intrinsics,
                unit=unit,
                min_depth_mm=1.0,
                max_depth_mm=cloud_max_depth_mm,  # None when the band is off; byte-identical
                voxel_size_mm=self.geometry_voxel_size_mm,
                max_points=self.max_geometry_points,
                cloud_outlier_filter=self._cloud_outlier_filter,  # None keeps it byte-identical
                # `normal_radius_mm` is deliberately not forwarded, and that is a discrepancy
                # rather than a decision. This sampler defaults to 12.0 mm while the
                # geometry-first path uses `self.normal_radius_mm`, 18.0 by default, so one
                # pipeline estimates surface normals at two radii depending on which branch it
                # took. Not corrected here because forwarding it moves the dense rung's measured
                # numbers, and that rung measured 28.4 % coverage against 28.5 % for the branch
                # beside it: a number that small is not worth moving without re-running the
                # ladder that produced it.
            )
            dense_geometry_poses = self._generator.dense_geometry_poses(
                samples,
                analysis=analysis,
                depth_confidence=depth_confidence,
            )
            dense_elapsed_ms = (_time.monotonic() - t0) * 1000.0
            telemetry["dense_runtime_ms"] = float(dense_elapsed_ms)
            telemetry["dense_sampling_points"] = int(samples.size)
            if dense_elapsed_ms > self.dense_runtime_budget_ms:
                telemetry["dense_timeout"] = True
                # ``dense_timeout`` already signals the overrun; the
                # extra ``dense_fallback`` key tells routing logic which
                # path actually produced the candidates so downstream
                # tools never have to infer it from runtime numbers.
                telemetry["dense_fallback"] = "silhouette_after_overrun"
                self._dense_last_overran = True
                self.logger.warning(
                    "Dense sampling overran budget: %.1f ms > %.1f ms; "
                    "falling back to silhouette geometry for this call.",
                    dense_elapsed_ms,
                    self.dense_runtime_budget_ms,
                )
                # `geometry_points`, the same array every other call site uses, and not
                # `cloud.points_mm`. This branch reached past the outlier filter applied above, so a
                # call that overran its dense budget searched contacts on unfiltered points while an
                # identical call that did not overrun searched filtered ones. Invisible while the
                # perception producer flattened the depth under each mask, because a plane has no
                # outliers to remove; on a measured surface the filter is the thing that keeps a
                # speckle pixel in front of the part from becoming a contact.
                geometry_poses = [] if cloud.is_empty else self._generator.geometry_first_poses(
                    self._generator.limit_geometry_points(geometry_points),
                    analysis=analysis,
                    depth_confidence=depth_confidence,
                )
            else:
                telemetry["dense_timeout"] = False
                telemetry["dense_fallback"] = "none"
                self._dense_last_overran = False
                geometry_poses = dense_geometry_poses
        else:
            # When a fused cloud was injected (dense_decision forced False above), the single-view
            # ``cloud`` may be empty even though the fused cloud is not, so the guard also reads
            # _fused_geometry.
            geometry_poses = [] if (cloud.is_empty and not _fused_geometry) else self._generator.geometry_first_poses(
                self._generator.limit_geometry_points(geometry_points),
                analysis=analysis,
                depth_confidence=depth_confidence,
            )
        if other_object_masks:
            # The same outlier filter the target cloud gets. Every other cloud in this method is
            # filtered and this one was not, which cost nothing while the perception producer
            # flattened the depth under a mask: a plane has no outliers. On a measured surface it is
            # the asymmetry that matters most, because the collision veto is one point
            # (`collision_checker.py`, `if colliding_indices:`), so a single speckle pixel on a
            # neighbour can reject every candidate on the target. `None`, the default, means no
            # filter call and the same array as before.
            neighbours = scene_collision_cloud(
                list(other_object_masks),
                depth_arr,
                intrinsics,
                target_index=None,
                unit=unit,
                min_depth_mm=1.0,
                voxel_size_mm=self.geometry_voxel_size_mm,
            )
            if self._cloud_outlier_filter is not None and neighbours.size > 0:
                _nkeep = apply_cloud_outlier_filter(neighbours, self._cloud_outlier_filter)
                if _nkeep.size > 0:  # a filter that empties a real surface is a filter, not a scene
                    neighbours = neighbours[_nkeep]
            if neighbours.size > 0:
                telemetry["scene_neighbour_points"] = int(neighbours.shape[0])
                if scene_points_mm is None:
                    scene_points_mm = neighbours
                else:
                    scene_points_mm = np.vstack(
                        [np.asarray(scene_points_mm, dtype=np.float64), neighbours]
                    )
        telemetry["candidates_geometry"] = len(geometry_poses)
        ranking_config = GeometricScoreConfig(
            min_width_mm=self.min_grip_mm,
            max_width_mm=self.max_grip_mm,
        )
        reachability_config = (
            ReachabilityScoreConfig(
                workspace=workspace,
                preferred_approach_axis=self.approach_cam,
            )
            if workspace is not None
            else None
        )
        ranked_silhouette = rank_grasp_poses(
            silhouette_poses,
            geometric_config=ranking_config,
            reachability_config=reachability_config,
            max_results=self.max_candidates,
        )
        remaining = max(self.max_candidates - len(ranked_silhouette), 0)
        ranked_geometry = (
            rank_grasp_poses(
                geometry_poses,
                geometric_config=ranking_config,
                reachability_config=reachability_config,
                max_results=remaining,
            )
            if remaining > 0 and geometry_poses
            else []
        )
        ranked_all: Sequence[GraspScoreBreakdown] = (
            *ranked_silhouette,
            *ranked_geometry,
        )

        # Support-footprint enumeration. Default on: ``support_footprint_geometry`` is True and
        # ``grasping.geometry.stage`` defaults to ``support_footprint``. When on it replaces
        # everything above rather than adding to it: SFE plans the support clearance instead of being
        # filtered for it, and mixing its candidates with silhouette ones would let a candidate that
        # was never checked against the table outrank one that was. It needs a BASE support plane and
        # a CAMERA->BASE transform; with either missing it cannot place the table, so the silhouette
        # path stands and the telemetry says which one ran. Its own score is kept:
        # ``rank_grasp_poses`` is not applied, because on this data the geometric ranker is
        # anti-correlated with validity (AUC 0.419).
        # The key is added only when the stage is enabled. Stamping "silhouette" on the default path
        # would change the telemetry contract for every caller that has not opted in, and default-off
        # has to mean byte-identical including what gets logged.
        if self._support_footprint_geometry:
            telemetry["geometry_stage"] = "silhouette"
            # The BASE frame is not optional here: SFE reads ``offset_mm`` as a height above the
            # workspace. A CAMERA-frame plane's offset is a depth, and taking it for a height
            # makes the table check inert.
            sfe_ready = (transform is not None and support_plane is not None
                         and support_plane.frame is Frame.BASE)
            telemetry["support_footprint_ready"] = bool(sfe_ready)
            if sfe_ready:
                assert transform is not None and support_plane is not None  # narrowed by sfe_ready
                # SFE's input at full resolution. ``cloud`` was voxelised at
                # ``geometry_voxel_size_mm`` (8 mm by default) for the contact search, and handing
                # that to SFE couples two numbers that have nothing to do with each other:
                # ``reconstruct_support_prism`` refuses below 20 points, and how many points an
                # object has is decided by a downsample knob. A ~32 mm cube is 4x4 = 16 cells at
                # 8 mm and falls off that cliff; measured on the multiview sim gate, that loses
                # every candidate with all four rejection counters at zero and scores 0/5.
                # The pre-downsample is also redundant: ``reconstruct_support_prism`` voxelises at
                # its own ``voxel_mm`` default of 3 mm, so this discards resolution SFE would thin
                # more finely.
                sfe_cloud = cloud
                if self._support_footprint_full_resolution and not cloud.is_empty:
                    sfe_cloud = masked_point_cloud(
                        segmentation.mask, depth_arr, intrinsics, unit=unit, min_depth_mm=1.0,
                        max_depth_mm=cloud_max_depth_mm, voxel_size_mm=None,
                    )
                target_base = (
                    np.asarray(geometry_points_base_mm, dtype=np.float64).reshape(-1, 3)
                    if geometry_points_base_mm is not None
                    else (transform[:3, :3] @ np.asarray(
                        sfe_cloud.points_mm, dtype=np.float64).reshape(-1, 3).T).T + transform[:3, 3]
                )
                obstacles_base = None
                if scene_points_mm is not None and np.asarray(scene_points_mm).size > 0:
                    _sp = np.asarray(scene_points_mm, dtype=np.float64).reshape(-1, 3)
                    obstacles_base = (transform[:3, :3] @ _sp.T).T + transform[:3, 3]
                # Declared geometry stays out of the sparse cloud above and gets its own,
                # undilated grid inside the generator; see support_footprint._ObstacleSet.
                rigid_base = None
                if (rigid_obstacle_points_mm is not None
                        and np.asarray(rigid_obstacle_points_mm).size > 0):
                    _rp = np.asarray(rigid_obstacle_points_mm, dtype=np.float64).reshape(-1, 3)
                    rigid_base = (transform[:3, :3] @ _rp.T).T + transform[:3, 3]
                sfe_ranked, sfe_telemetry = support_footprint_breakdowns(
                    target_base,
                    camera_to_base=transform,
                    support_height_mm=float(support_plane.offset_mm),
                    jaw=SupportFootprintJaw.from_model(
                        gripper_model if isinstance(gripper_model, ParallelJawGripperModel)
                        else (self._gripper_model
                              if isinstance(self._gripper_model, ParallelJawGripperModel) else None),
                        # ⛔ THE STROKE, WHICH THIS CALL OMITTED UNTIL 2026-09-10. Without these two
                        # the jaw took `from_model`'s keyword defaults, 85.0 and 5.0, which are a
                        # 2F-85's numbers. MEASURED on the `hande` profile: the pick path planned a
                        # 49.99 mm hand as though it opened 85.0, proposed grasps up to 83 mm wide,
                        # and the driver then clamped the close to max_width_mm, which maps to count
                        # 0 = FULLY OPEN. The close command opened the hand at the grasp point.
                        # `scene.py` passed the configured aperture; this path did not, and every
                        # OTHER dimension of the jaw came out correct, which is what hid it.
                        aperture_mm=float(self.max_grip_mm),
                        min_width_mm=float(self.min_grip_mm),
                        table_clearance_mm=float(min_table_clearance_mm),
                    ),
                    obstacle_points_base_mm=obstacles_base,
                    rigid_obstacle_points_base_mm=rigid_base,
                    max_candidates=self.max_candidates,
                    inflate_mm=self._support_footprint_inflate_mm,
                    palm_aware=self._support_footprint_palm_aware,
                    score_weights=self._support_footprint_score_weights,
                )
                telemetry.update(sfe_telemetry)
                telemetry["geometry_stage"] = "support_footprint"
                # An abstention is not a rejection. SFE returns an empty list for two very different
                # reasons: it looked and found no legal grasp, or it could not look at all;
                # ``reconstruct_support_prism`` gives up below 20 surviving points. Replacing
                # ``ranked_all`` unconditionally treats the second case as a veto, so a small or
                # grazing-angle object loses the silhouette candidates that would have worked and
                # the attempt reports no candidate with every rejection counter at zero; measured on
                # the multiview sim gate: 3 silhouette candidates in, 16 cloud points, 0 out,
                # 0/5 lifted.
                if sfe_ranked or not self._support_footprint_fallback:
                    ranked_all = tuple(sfe_ranked)
                else:
                    telemetry["geometry_stage"] = "silhouette_fallback"

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
        # Collision/IK filter + candidate build + per-candidate corridor + base-frame + sort + clearance/
        # force-closure stamps.
        candidates = self._filter_and_build_candidates(
            ranked_all,
            seg_meta=seg_meta,
            scene_points_mm=scene_points_mm,
            support_plane=plane_cam,
            workspace=workspace,
            gripper_model=gripper_model,
            corridor_config=corridor_config,
            feasibility_config=feasibility_config,
            feasibility_weight=feasibility_weight,
            min_table_clearance_mm=min_table_clearance_mm,
            collision_margin_mm=collision_margin_mm,
            ik_service=ik_service,
            transform=transform,
            intrinsics=intrinsics,
            depth_map=depth_map,
            scale=scale,
            other_object_masks=other_object_masks,
            telemetry=telemetry,
        )
        self.last_telemetry = telemetry
        if rgb_image is not None:
            self.last_debug_image_png = self._debug_renderer.draw(
                rgb_image=rgb_image,
                segmentation=segmentation,
                candidates=candidates,
                intrinsics=intrinsics,
                gripper_model=gripper_model,
                config=debug_config,
                label=debug_label,
                telemetry=telemetry,
                transform=transform,
            )
        self.logger.info(
            "Generated %d grasp candidates (best score=%.3f); telemetry=%s",
            len(candidates),
            candidates[0].score if candidates else 0.0,
            telemetry,
        )
        reasons = self._derive_failure_reasons(
            candidates=candidates,
            telemetry=telemetry,
            had_filters=(
                plane_cam is not None
                or scene_points_mm is not None
                or workspace is not None
                or ik_service is not None
            ),
        )
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
            # neighbours (single-object), so the validator sees NO_OBSTACLES (byte-identical).
            metadata=(
                {
                    "scene_points_mm": np.asarray(scene_points_mm, dtype=np.float64).copy(),
                    "scene_points_frame": "camera",
                }
                if scene_points_mm is not None and np.asarray(scene_points_mm).size > 0
                else {}
            ),
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
        corridor_config: "CorridorAnalysisConfig | None",
        feasibility_config: "FeasibilityScoreConfig | None",
        feasibility_weight: float | None,
        min_table_clearance_mm: float,
        collision_margin_mm: float,
        ik_service: IKService | None,
        transform: np.ndarray | None,
        intrinsics: CameraIntrinsics,
        depth_map: np.ndarray,
        scale: float,
        other_object_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
        telemetry: dict,
    ) -> list[GraspPoint]:
        """The post-ranking stage of :meth:`compute`: collision/table/workspace + IK filtering, the
        :class:`GraspPoint` build, the per-candidate corridor-risk stamp, the optional base-frame transform,
        the final sort, and the approach-clearance / force-closure stamps. The ``telemetry`` dict is mutated
        in place. Returns the final sorted candidates."""
        # Collision / table / workspace filtering (camera-frame; matches
        # the frame the candidates were planned in).
        if support_plane is not None or scene_points_mm is not None or workspace is not None:
            ranked_all, stage_telemetry = self._apply_collision_filters(
                ranked_all,
                scene_points_mm=scene_points_mm,
                support_plane=support_plane,
                workspace=workspace,
                gripper_model=gripper_model,
                min_table_clearance_mm=min_table_clearance_mm,
                collision_margin_mm=collision_margin_mm,
            )
            telemetry.update(stage_telemetry)

        telemetry.setdefault("rejected_ik", 0)
        kept_quality: list[IKQualityMetrics | None] = []
        feasibility_reranked = False
        if ik_service is not None:
            ranked_all, kept_quality, ik_telemetry = self._apply_ik_filter(
                ranked_all,
                ik_service=ik_service,
                transform=transform,
            )
            telemetry["rejected_ik"] = ik_telemetry["rejected_ik"]
            # Feasibility re-rank of the IK survivors (default-off). The per-candidate IK quality only
            # exists post-filter, so this demotion-only re-rank runs here, not in the up-front ranking.
            # Off (weights None) feeds no feasibility inputs, so the order and the success-model feas_*
            # stay byte-identical. Per-call (config-driven) values win over the constructor's; when
            # neither is given the default path is untouched.
            _feas_cfg = (
                feasibility_config if feasibility_config is not None else self._feasibility_config
            )
            _feas_w = (
                GraspScoreWeights(feasibility=float(feasibility_weight))
                if (feasibility_weight is not None and feasibility_weight > 0.0
                    and feasibility_config is not None)
                else self._feasibility_weights
            )
            if (
                _feas_w is not None
                and _feas_cfg is not None
                and ranked_all
            ):
                feas_inputs: list[FeasibilityInputs | None] = [
                    FeasibilityInputs(ik_quality=q) for q in kept_quality
                ]
                ranked_all = rerank_breakdowns_with_feasibility(
                    ranked_all,
                    feas_inputs,
                    weights=_feas_w,
                    feasibility_config=_feas_cfg,
                )
                feasibility_reranked = True

        candidates = [self._to_grasp_point(score, seg_meta) for score in ranked_all]
        # RL shadow-only feasibility_score: the IK filter above computed a per-candidate
        # ``IKQualityMetrics``, Jacobian conditioning and joint margin, and the default path discards
        # it. Routing it into ``metadata["shadow"]["feasibility_score"]`` gives the RL ranking rows a
        # pre-execution reachability predictor; the descent-timeout failures of the sim are
        # reachability failures the geometric score is blind to.
        #
        # Stamped only when the feasibility re-rank did not run, which is what
        # ``feasibility_reranked`` records. The re-rank re-sorts ``ranked_all`` while ``kept_quality``
        # keeps the order the IK filter left it in, so a stamp written afterwards would pair each
        # candidate with another candidate's metrics. The flag is the condition the re-rank itself
        # branches on, not the constructor field: the effective weights fall back to the per-call
        # ``feasibility_weight`` and ``feasibility_config``, which is the path the pick loop takes.
        # When the re-rank does run, the top-level ``feasibility_score`` is live and carries the same
        # signal.
        #
        # Shadow precedence outranks the dead top-level ``0.0`` (``_shadow_aggregator``), while
        # ``GraspPoint.score``, the deterministic order and the success-model vector are untouched.
        if not feasibility_reranked and kept_quality:
            for _cand, _q in zip(candidates, kept_quality):
                if _q is not None:
                    _cand.metadata.setdefault("shadow", {})["feasibility_score"] = round(
                        feasibility_grasp_score(
                            ik_quality=_q, approach_path=None, config=_SHADOW_FEASIBILITY_CONFIG
                        ),
                        4,
                    )
        # Per-candidate corridor-risk producer (default off).
        # Runs in CAMERA frame (before the base-frame transform below, so depth+intrinsics agree with the
        # pose) and index-aligned with ranked_all (candidates not yet sorted). The analyzer is pure +
        # never raises on missing depth/intrinsics (returns a skipped report with neutral 0.5). The stamp
        # rides through _to_base_frame_only (which copies metadata) and the sort below. Cost: N
        # depth-marches (vs the single occlusion march). When off the ranking is byte-identical.
        if (self.corridor_risk_per_candidate or corridor_config is not None) and ranked_all:
            # The operator's geometry when `grasping.occlusion.directional_enabled` supplied one,
            # otherwise the hardcoded defaults, which is byte-identical for every existing caller.
            # This is the scoring half only: `hard_reject_enabled` is a separate switch and a
            # separate question, because a change to how candidates are ranked and a change to
            # which candidates are refused cannot be read apart from one run that moves both.
            corridor_cfg = corridor_config if corridor_config is not None else CorridorAnalysisConfig()
            masks_tuple = (
                tuple(np.asarray(m) for m in other_object_masks)
                if other_object_masks
                else None
            )
            depth_for_corridor = np.asarray(depth_map, dtype=np.float64)
            blocked_count = 0
            max_blockage = 0.0
            _corridor_trace = os.getenv("WILLY_CORRIDOR_TRACE")  # per-candidate corridor diagnostic (default off)
            for _i, (cand, score) in enumerate(zip(candidates, ranked_all)):
                pose = score.pose
                report = analyze_corridor(
                    CorridorAnalysisInputs(
                        target_point_cam_mm=np.asarray(pose.position_mm, dtype=np.float64),
                        approach_axis_cam=np.asarray(pose.approach_axis, dtype=np.float64),
                        depth_map=depth_for_corridor,
                        intrinsics=intrinsics,
                        other_object_masks=masks_tuple,
                        depth_scale_to_mm=float(scale),
                    ),
                    corridor_cfg,
                )
                conf = float(report.blockage_confidence)
                # GraspPoint is frozen, but .metadata is a mutable dict (stamped in place).
                cand.metadata["corridor_blockage_confidence"] = conf
                cand.metadata["corridor_mode"] = report.mode.value
                # RL shadow-only: alias the live per-candidate corridor risk into ``uncertainty_score`` so that
                # RL ranking row (idx 0, otherwise a dead 0.0) carries a real approach-uncertainty signal,
                # physically independent of the geometric score. Shadow-precedence, zero behavioural impact.
                cand.metadata.setdefault("shadow", {})["uncertainty_score"] = conf
                blocked_count += int(report.blocked)
                max_blockage = max(max_blockage, conf)
                if _corridor_trace:
                    _p = np.round(np.asarray(pose.position_mm, dtype=np.float64), 1)
                    print(
                        f"[corridor-trace] cand{_i} pos={_p} mode={report.mode.value} conf={conf:.3f} "
                        f"appr_clear={report.approach_clearance_mm:.0f} retr_clear={report.retreat_clearance_mm:.0f} "
                        f"masks={'Y' if masks_tuple else 'N'}",
                        flush=True,
                    )
            telemetry["corridor_candidates_analyzed"] = len(candidates)
            telemetry["corridor_blocked_count"] = int(blocked_count)
            telemetry["corridor_max_blockage_confidence"] = float(max_blockage)
        if transform is not None:
            candidates = self._to_base_frame_only(candidates, transform)
            if self._isotropic_radial_closing:
                # Applied here and not in the generator: "radial" is defined about the robot base, so it
                # only exists once the candidates have left camera frame. Elongated silhouettes and
                # tilted approaches are skipped inside, and the score is untouched (only the spin about
                # the approach changes), so the ranking below is unperturbed.
                from ._isotropic_closing import snap_isotropic_closing_to_radial

                candidates = snap_isotropic_closing_to_radial(candidates)
        candidates.sort(key=lambda grasp: grasp.score, reverse=True)
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
        # Surface analytical force-closure stats. Only stamped
        # when the operator opted in via ``friction_coefficient``; the
        # heuristic-only telemetry shape is unchanged when the knob is
        # disabled.
        if self.friction_coefficient is not None:
            telemetry["friction_coefficient"] = float(self.friction_coefficient)
            telemetry["force_closure_certified_count"] = sum(
                1
                for grasp in candidates
                if grasp.metadata.get("force_closure_certified") is True
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

    def _decide_dense_sampling(
        self,
        *,
        dense_sampling: bool | None,
        mask: np.ndarray,
        other_object_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
        telemetry: dict,
    ) -> bool:
        """Facade shim to the candidate generator. The method stays on the facade because it is called
        directly and because the ``_dense_last_overran`` latch is facade-owned; the generator reads that
        latch through its injected getter."""
        return self._generator.decide_dense_sampling(
            dense_sampling=dense_sampling,
            mask=mask,
            other_object_masks=other_object_masks,
            telemetry=telemetry,
        )

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
    ) -> tuple[list[GraspScoreBreakdown], list[IKQualityMetrics | None], dict]:
        """Drop candidates the IK service declares unreachable.

        Candidates are passed to the service in the final return frame:
        base frame when a ``camera_to_base`` / ``T_cam_to_base`` was
        supplied, otherwise camera frame. The IK adapter is expected to
        be configured for that frame.

        Also returns the per-survivor ``IKQualityMetrics`` (index-aligned with the kept list, ``None`` when
        the adapter publishes none); the feasibility re-rank consumes it.
        """
        from src.robot.grasping.planning import transform_grasp_pose

        kept: list[GraspScoreBreakdown] = []
        kept_quality: list[IKQualityMetrics | None] = []
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
                kept_quality.append(result.quality)
            else:
                rejected += 1
        return kept, kept_quality, {"rejected_ik": rejected}

    @staticmethod
    def _derive_failure_reasons(
        *,
        candidates: list[GraspPoint],
        telemetry: dict,
        had_filters: bool,
    ) -> tuple[GraspFailureReason, ...]:
        """Map telemetry counters to reason codes for the success path.

        If the pipeline reached the end and still has candidates the
        result is a success and no reasons are emitted. If the
        post-filter list is empty but filters were active, the reasons
        name which stage exhausted the candidates.
        """
        if candidates:
            return ()
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

    def _apply_force_closure_certificates(
        self,
        pairs: list[ContactPair],
        poses: list[GraspPose],
    ) -> int:
        """Facade shim to the candidate generator's single force-closure certification site, kept on the
        facade because it is called directly. Returns the count of certified candidates."""
        return self._generator.apply_force_closure_certificates(pairs, poses)

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
            "feasibility_score": round(score.feasibility_score, 4),
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
