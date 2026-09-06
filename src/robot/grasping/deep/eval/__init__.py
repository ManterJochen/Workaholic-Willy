"""Diagnostics and measurement for the deep grasp generator.

Separate from the training loop on purpose: everything here answers a question about a model or a
dataset without changing either, so it can be run at any point, including before a single training
step.

The three modules this package re-exports, `asset_consistency`, `geometric_floor` and
`label_coverage`, ask about the corpus and about a purely geometric baseline and are
architecture-independent by construction: the binned generator's own machinery has no measurement
here, and its approach bins, its PointNet++ sampling levels and its seed proposer do not exist in
the family that ships. The other modules in the directory, `charts`, `gripper_differential`,
`probes`, `proposal_scorer_training`, `propose` and `run_report`, read a trained model or a run.

`scene_family` is a filename split, not a measurement, and lives in `deep.corpus`.
"""

from src.robot.grasping.deep.eval.asset_consistency import (
    AssetConsistency,
    format_consistency,
    measure_asset_consistency,
)
from src.robot.grasping.deep.eval.geometric_floor import (
    GeometricRule,
    ObjectDirections,
    format_directions,
    format_geometric,
    object_direction_report,
    run_geometric_rules,
)
from src.robot.grasping.deep.eval.label_coverage import (
    SamplingReport,
    format_sampling_report,
    sampling_report,
    write_visual,
)

__all__ = [
    "AssetConsistency",
    "GeometricRule",
    "ObjectDirections",
    "SamplingReport",
    "format_consistency",
    "format_directions",
    "format_geometric",
    "format_sampling_report",
    "measure_asset_consistency",
    "object_direction_report",
    "run_geometric_rules",
    "sampling_report",
    "write_visual",
]
