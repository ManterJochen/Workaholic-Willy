"""Vision-language grasp planning: the public surface.

The package is a downward-dependency stack of tiers. The re-exports below are
grouped in that stack order:

    types
    geometry, contacts, collision
    generation
    scoring, planning
    motion
    closed_loop
    recovery
    telemetry
    uncertainty

Importing ``grasping`` gives the stable public names regardless of which tier
module they physically live in.
"""

# --- types/ : shared value objects ---------------------------------------
from .types.feedback import GraspFailureReason, GraspResult
from .types.grasp_point import GraspFrame, GraspPoint
from .types.modes import (
    GraspSamplingMode,
    mode_to_dense_sampling,
    resolve_grasp_sampling_mode,
)
from .types.perception import SegmentationLike

# --- geometry / contacts / collision -------------------------------------
from .collision import (
    CollisionBox,
    GraspCollisionResult,
    GripperGeometryStrategy,
    ParallelJawGripperModel,
    SuctionCupGripperModel,
    SupportPlane,
    colliding_point_indices,
    gripper_table_clearance_mm,
    points_to_grasp_frame,
    validate_grasp_collision,
    validate_grasp_collisions,
)
from .contacts import ContactPair, ContactPoint, find_antipodal_pairs
from .geometry import (
    CameraIntrinsics,
    MaskedPointCloud,
    NormalEstimationConfig,
    SurfaceNormals,
    depth_unit_to_mm,
    estimate_surface_normals,
    masked_point_cloud,
    masked_points,
)

# --- generation/ : mask + depth to ranked candidates ---------------------
from .generation.calculator import GraspCalculator
from .generation.deformables import (
    CablePCAStrategy,
    DeformableClass,
    DeformableHandlingDecision,
    DeformableHandlingStrategy,
    RefuseDeformableStrategy,
)
from .generation.mask_analyzer import MaskAnalysis, MaskAnalyzer
# `Scene` is the geometry-stage noun, not a scene graph and not a simulator scene. It sits beside
# six `SceneRecovery*` types here, which are about what a cell does when a pick fails; this one is a
# cloud on a support surface. Measured: exporting it costs 1 module and 0.7 ms on top of what this
# package already imports, and pulls nothing from `deep/`.
from .scene import Scene, SceneGrasps

# --- scoring / planning --------------------------------------------------
from .scoring.semantic_policy import SemanticDecision, SemanticPolicy
from .scoring.topology_risk import (
    depth_continuity_risk,
    topology_risk_from_mask,
)
from .planning import GraspPose, generate_grasp_poses, grasp_pose_from_contact_pair
from .planning.multifinger import (
    FingerKinematicSpec,
    GripperKind,
    MultiContactGrasp,
    MultiContactGraspPlanner,
    MultiContactPlanRequest,
    ParallelJawContactPlanner,
    RadialMultiFingerPlanner,
)
from .scoring import (
    ForceClosureCertificate,
    GeometricScoreConfig,
    GraspScoreBreakdown,
    GraspScoreWeights,
    ReachabilityScoreConfig,
    StabilityScoreConfig,
    WorkspaceBox,
    approach_alignment_score,
    approach_clearance_mm,
    certify_contact_pair,
    geometric_grasp_score,
    geometric_score_components,
    occlusion_ratio,
    rank_grasp_poses,
    reachability_grasp_score,
    reachability_score_components,
    score_grasp_pose,
    stability_grasp_score,
    stability_score_components,
    table_clearance_score,
    width_fit_score,
)

# --- motion/ : approach, grasp, retreat + path validation ----------------
from .motion.execution_policy import GraspExecutionPolicy, PolicyOutcome, PolicyReport
from .motion.frame_resolver import (
    EyeInHandFrameResolver,
    FrameResolver,
    IdentityFrameResolver,
    StaticCameraToBaseResolver,
)
from .motion.trajectory_safety import (
    AcceptAllTrajectorySafetyCheck,
    ApproachPathOutcome,
    ApproachPathPolicy,
    ApproachPathReport,
    TrajectorySafetyCheck,
    TrajectoryStepReport,
    validate_approach_and_retreat,
    validate_approach_path,
    validate_retreat_path,
)

# --- closed_loop/ : refine / verify / next-best-view ---------------------
from .closed_loop.refinement import (
    DefaultPreGraspRefiner,
    IoUCentroidTargetTracker,
    PreGraspRefiner,
    RefinementOutcome,
    RefinementPolicy,
    RefinementReport,
    TargetIdentity,
    TargetTracker,
    WorldSpacePoseTracker,
    target_identity_from_segmentation,
)
from .closed_loop.verification import (
    CompositeGraspVerifier,
    GraspVerificationContext,
    GraspVerificationPolicy,
    GraspVerificationReport,
    GraspVerifier,
    NoOpVerifier,
    ObjectDetectingGripperVerifier,
    VerificationOutcome,
    VisionTargetDisplacementVerifier,
    WidthDeltaGripperVerifier,
)
from .closed_loop.active_perception import (
    AcceptAllViewpointSafetyCheck,
    ScoringViewpointPlanner,
    ViewScoringPolicy,
    ViewpointCandidate,
    ViewpointHistory,
    ViewpointObservation,
    ViewpointSafetyCheck,
    ViewpointSignals,
    WorkspaceBoxSafetyCheck,
)

# --- recovery/ -----------------------------------------------------------
from .recovery.policy import (
    ActivePerceptionRecoveryStrategy,
    ContainerAgitateStrategy,
    FixtureEnvelope,
    NextTargetRecoveryStrategy,
    NoRecoveryStrategy,
    SceneRecoveryAction,
    SceneRecoveryContext,
    SceneRecoveryPlan,
    SceneRecoveryPolicy,
    SceneRecoveryReport,
    SceneRecoveryStrategy,
    SmallNudgeStrategy,
    execute_recovery_motion,
)
from .recovery.orchestrator import (
    RecoveryDispatcher,
    RecoveryHistoryEntry,
    RecoveryOrchestrator,
    RecoveryTrail,
    RecoveryTrailEntry,
    VALID_TERMINAL_REASONS,
    run_recovery_loop,
)

# --- telemetry/ ----------------------------------------------------------
from .telemetry.outcome_logging import (
    GraspAttemptRecord,
    append_jsonl,
    frame_metadata_from,
    grasp_metadata_from,
    iter_jsonl,
    json_safe,
    profile_metadata_from,
    recovery_metadata_from,
    refinement_metadata_from,
    target_metadata_from,
    verification_metadata_from,
)

# --- visualization / uncertainty -----------------------------------------
from .visualization import (
    DebugDrawConfig,
    GraspViewerConfig,
    build_grasp_scene,
    draw_grasp_debug_image,
    save_grasp_debug_image,
    show_grasp_scene,
)
from .uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannel,
    UncertaintyChannelValues,
    UncertaintyMonotoneMap,
    UncertaintySnapshot,
    UncertaintyWeights,
    apply_uncertainty_ranking_penalty,
    fuse_uncertainty,
    should_bias_recovery_for_uncertainty,
)

__all__ = [
    "AcceptAllTrajectorySafetyCheck",
    "AcceptAllViewpointSafetyCheck",
    "ActivePerceptionRecoveryStrategy",
    "append_jsonl",
    "apply_uncertainty_ranking_penalty",
    "approach_alignment_score",
    "approach_clearance_mm",
    "ApproachPathOutcome",
    "ApproachPathPolicy",
    "ApproachPathReport",
    "build_grasp_scene",
    "CablePCAStrategy",
    "CameraIntrinsics",
    "certify_contact_pair",
    "colliding_point_indices",
    "CollisionBox",
    "CompositeGraspVerifier",
    "ContactPair",
    "ContactPoint",
    "ContainerAgitateStrategy",
    "DebugDrawConfig",
    "DefaultPreGraspRefiner",
    "DeformableClass",
    "DeformableHandlingDecision",
    "DeformableHandlingStrategy",
    "depth_continuity_risk",
    "depth_unit_to_mm",
    "draw_grasp_debug_image",
    "estimate_surface_normals",
    "execute_recovery_motion",
    "EyeInHandFrameResolver",
    "find_antipodal_pairs",
    "FingerKinematicSpec",
    "ForceClosureCertificate",
    "frame_metadata_from",
    "FrameResolver",
    "fuse_uncertainty",
    "generate_grasp_poses",
    "geometric_grasp_score",
    "geometric_score_components",
    "GeometricScoreConfig",
    "grasp_metadata_from",
    "grasp_pose_from_contact_pair",
    "GraspAttemptRecord",
    "GraspCalculator",
    "GraspCollisionResult",
    "GraspExecutionPolicy",
    "GraspFailureReason",
    "GraspFrame",
    "GraspPoint",
    "GraspPose",
    "GraspResult",
    "GraspSamplingMode",
    "GraspScoreBreakdown",
    "GraspScoreWeights",
    "GraspVerificationContext",
    "GraspVerificationPolicy",
    "GraspVerificationReport",
    "GraspVerifier",
    "GraspViewerConfig",
    "gripper_table_clearance_mm",
    "GripperGeometryStrategy",
    "GripperKind",
    "IdentityFrameResolver",
    "IoUCentroidTargetTracker",
    "iter_jsonl",
    "json_safe",
    "MaskAnalysis",
    "MaskAnalyzer",
    "masked_point_cloud",
    "masked_points",
    "MaskedPointCloud",
    "mode_to_dense_sampling",
    "MultiContactGrasp",
    "MultiContactGraspPlanner",
    "MultiContactPlanRequest",
    "NextTargetRecoveryStrategy",
    "NoOpVerifier",
    "NoRecoveryStrategy",
    "NormalEstimationConfig",
    "ObjectDetectingGripperVerifier",
    "occlusion_ratio",
    "ParallelJawContactPlanner",
    "ParallelJawGripperModel",
    "points_to_grasp_frame",
    "PolicyOutcome",
    "PolicyReport",
    "PreGraspRefiner",
    "profile_metadata_from",
    "RadialMultiFingerPlanner",
    "rank_grasp_poses",
    "reachability_grasp_score",
    "reachability_score_components",
    "ReachabilityScoreConfig",
    "recovery_metadata_from",
    "RecoveryDispatcher",
    "RecoveryHistoryEntry",
    "RecoveryOrchestrator",
    "RecoveryTrail",
    "RecoveryTrailEntry",
    "refinement_metadata_from",
    "RefinementOutcome",
    "RefinementPolicy",
    "RefinementReport",
    "RefuseDeformableStrategy",
    "resolve_grasp_sampling_mode",
    "run_recovery_loop",
    "save_grasp_debug_image",
    "Scene",
    "SceneGrasps",
    "SceneRecoveryAction",
    "SceneRecoveryContext",
    "SceneRecoveryPlan",
    "SceneRecoveryPolicy",
    "SceneRecoveryReport",
    "SceneRecoveryStrategy",
    "score_grasp_pose",
    "ScoringViewpointPlanner",
    "SegmentationLike",
    "SemanticDecision",
    "SemanticPolicy",
    "should_bias_recovery_for_uncertainty",
    "show_grasp_scene",
    "SmallNudgeStrategy",
    "stability_grasp_score",
    "stability_score_components",
    "StabilityScoreConfig",
    "StaticCameraToBaseResolver",
    "SuctionCupGripperModel",
    "SupportPlane",
    "SurfaceNormals",
    "table_clearance_score",
    "target_identity_from_segmentation",
    "target_metadata_from",
    "TargetIdentity",
    "TargetTracker",
    "topology_risk_from_mask",
    "TrajectorySafetyCheck",
    "TrajectoryStepReport",
    "UncertaintyCalibration",
    "UncertaintyChannel",
    "UncertaintyChannelValues",
    "UncertaintyMonotoneMap",
    "UncertaintySnapshot",
    "UncertaintyWeights",
    "VALID_TERMINAL_REASONS",
    "validate_approach_and_retreat",
    "validate_approach_path",
    "validate_grasp_collision",
    "validate_grasp_collisions",
    "validate_retreat_path",
    "verification_metadata_from",
    "VerificationOutcome",
    "ViewpointCandidate",
    "ViewpointHistory",
    "ViewpointObservation",
    "ViewpointSafetyCheck",
    "ViewpointSignals",
    "ViewScoringPolicy",
    "VisionTargetDisplacementVerifier",
    "width_fit_score",
    "WidthDeltaGripperVerifier",
    "WorkspaceBox",
    "WorkspaceBoxSafetyCheck",
    "WorldSpacePoseTracker",
]
