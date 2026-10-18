"""Vision-language grasp planning: the public surface.

The package is a downward-dependency stack of tiers. The re-exports below are
grouped in that stack order:

    types
    geometry, contacts, collision
    generation
    scoring, planning
    motion
    recovery
    telemetry

Importing ``grasping`` gives the stable public names regardless of which tier
module they physically live in. ``closed_loop``, the tier between motion and
recovery, left on 2026-09-29 with the last thing it held, the post-grasp
verifiers: whether a close holds a part is the gripper's own hold evidence,
read by :class:`~src.robot.grasping.motion.execution_policy.GraspExecutionPolicy`
after every close.
"""

# --- types/ : shared value objects ---------------------------------------
from .types.feedback import GraspFailureReason, GraspResult
from .types.grasp_point import GraspFrame, GraspPoint
from .types.modes import GraspSamplingMode
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
    pose_from_grasp_axes,
)

# --- generation/ : mask + depth to ranked candidates ---------------------
from .generation.calculator import GraspCalculator
from .generation.deformables import DeformableClass, DeformableHandlingDecision, DeformableHandlingStrategy
from .generation.mask_analyzer import MaskAnalysis, MaskAnalyzer
# `Scene` is the geometry-stage noun, not a scene graph and not a simulator scene. It sits beside
# six `SceneRecovery*` types here, which are about what a cell does when a pick fails; this one is a
# cloud on a support surface. Measured: exporting it costs 1 module and 0.7 ms on top of what this
# package already imports, and pulls nothing from `deep/`.
from .scene import Scene, SceneGrasps

# --- scoring / planning --------------------------------------------------
from .scoring.semantic_policy import SemanticDecision, SemanticPolicy
from .scoring.topology_risk import topology_risk_from_mask
from .planning import GraspPose
from .planning.multifinger import (
    FingerKinematicSpec,
    GripperKind,
    MultiContactGrasp,
    MultiContactGraspPlanner,
    MultiContactPlanRequest,
    ParallelJawContactPlanner,
    RadialMultiFingerPlanner,
)
from .scoring import GraspScoreBreakdown, WorkspaceBox, approach_clearance_mm, occlusion_ratio

# --- motion/ : approach, grasp, retreat ----------------------------------
from .motion.execution_policy import GraspExecutionPolicy, PolicyOutcome, PolicyReport
from .motion.frame_resolver import (
    EyeInHandFrameResolver,
    FrameResolver,
    IdentityFrameResolver,
    StaticCameraToBaseResolver,
)

# --- recovery/ -----------------------------------------------------------
from .recovery.policy import (
    SceneRecoveryAction,
    SceneRecoveryContext,
    SceneRecoveryPlan,
    SceneRecoveryPolicy,
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
    grasp_metadata_from,
    iter_jsonl,
    json_safe,
    profile_metadata_from,
)

# --- visualization -------------------------------------------------------
from .visualization import DebugDrawConfig, draw_grasp_debug_image

__all__ = [
    "approach_clearance_mm",
    "CameraIntrinsics",
    "colliding_point_indices",
    "CollisionBox",
    "ContactPair",
    "ContactPoint",
    "DebugDrawConfig",
    "DeformableClass",
    "DeformableHandlingDecision",
    "DeformableHandlingStrategy",
    "depth_unit_to_mm",
    "draw_grasp_debug_image",
    "estimate_surface_normals",
    "EyeInHandFrameResolver",
    "find_antipodal_pairs",
    "FingerKinematicSpec",
    "FrameResolver",
    "grasp_metadata_from",
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
    "gripper_table_clearance_mm",
    "GripperGeometryStrategy",
    "GripperKind",
    "IdentityFrameResolver",
    "iter_jsonl",
    "json_safe",
    "MaskAnalysis",
    "MaskAnalyzer",
    "masked_point_cloud",
    "masked_points",
    "MaskedPointCloud",
    "MultiContactGrasp",
    "MultiContactGraspPlanner",
    "MultiContactPlanRequest",
    "NormalEstimationConfig",
    "occlusion_ratio",
    "ParallelJawContactPlanner",
    "ParallelJawGripperModel",
    "points_to_grasp_frame",
    "pose_from_grasp_axes",
    "PolicyOutcome",
    "PolicyReport",
    "profile_metadata_from",
    "RadialMultiFingerPlanner",
    "RecoveryDispatcher",
    "RecoveryHistoryEntry",
    "RecoveryOrchestrator",
    "RecoveryTrail",
    "RecoveryTrailEntry",
    "run_recovery_loop",
    "Scene",
    "SceneGrasps",
    "SceneRecoveryAction",
    "SceneRecoveryContext",
    "SceneRecoveryPlan",
    "SceneRecoveryPolicy",
    "SegmentationLike",
    "SemanticDecision",
    "SemanticPolicy",
    "StaticCameraToBaseResolver",
    "SuctionCupGripperModel",
    "SupportPlane",
    "SurfaceNormals",
    "topology_risk_from_mask",
    "VALID_TERMINAL_REASONS",
    "validate_grasp_collision",
    "WorkspaceBox",
]
