"""Safety checks and motion guards for robot execution."""

from __future__ import annotations

from .attestation import SafetyAttestation, SafetyGated, SafetyPosture
from .continuity import MotionContinuityGuard
from .decision import SafetyDecision, SafetyReason, safety_reason_to_motion_status
from .guard import SafetyContext, SafetyGuard
from .ik_quality import IKQualityGuard
from .joint_limits import (
    UR_JOINT_LIMITS_DEG,
    JointLimitGuard,
    resolve_joint_limits_deg,
)
from .payload import PayloadGuard
from .path_samples import (
    MAX_PATH_SAMPLES,
    LineSamples,
    PathSamples,
    joint_path_samples,
    line_samples,
)
from .preflight import SafetyPreflight, WorkspaceSafetyGuard
from .self_collision import SelfCollisionGuard
from .singularity import (
    SingularityGuard,
    SingularityReport,
    SingularityThresholds,
    analyze_joint_singularity,
    analyze_pose_singularity,
    assert_pose_not_singular,
)
from .workspace import WorkspaceGuard

__all__ = [
    "IKQualityGuard",
    "JointLimitGuard",
    "LineSamples",
    "MAX_PATH_SAMPLES",
    "MotionContinuityGuard",
    "PathSamples",
    "PayloadGuard",
    "SafetyAttestation",
    "SafetyContext",
    "SafetyDecision",
    "SafetyGated",
    "SafetyGuard",
    "SafetyPosture",
    "SafetyPreflight",
    "SafetyReason",
    "SelfCollisionGuard",
    "SingularityGuard",
    "SingularityReport",
    "SingularityThresholds",
    "UR_JOINT_LIMITS_DEG",
    "WorkspaceGuard",
    "WorkspaceSafetyGuard",
    "analyze_joint_singularity",
    "analyze_pose_singularity",
    "assert_pose_not_singular",
    "joint_path_samples",
    "line_samples",
    "resolve_joint_limits_deg",
    "safety_reason_to_motion_status",
]
