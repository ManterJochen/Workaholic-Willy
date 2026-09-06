"""Suction, the vacuum grasp modality.

A second end-effector modality rather than a flag on the parallel-jaw path. A suction grasp is one
sealable and roughly flat surface approached along its normal, so it has no aperture, no antipodal pair
and no closing direction. It covers what the 2F-85 is weakest at: wide flat objects, top-face bin picks
and the mustard-aperture class.

The quality model is analytical physics, written from the published equations rather than adapted from a
dataset-bound implementation. :mod:`.seal` scores seal formation over concentric deformable cup rings on
the rendered-depth cloud, :mod:`.wrench` scores wrench resistance, asking whether vacuum, friction and the
elastic moment hold the part, and the two are multiplied behind the :class:`.scorer.SuctionScorer` seam.
That Protocol admits a learned scorer, but no learned implementation exists here: a learned scorer needs
licence-clean training data rather than another vendor, which :mod:`.scorer` sets out. :mod:`.synthesis`
turns a perceived object into ranked candidates.

Pure numpy. Isaac performs the attach and the lift, and real seal physics stays a real-hardware concern.
"""

from __future__ import annotations

from src.robot.grasping.suction.scorer import (
    AnalyticalSuctionScorer,
    SuctionQuality,
    SuctionScorer,
)
from src.robot.grasping.suction.seal import (
    SealConfig,
    SealResult,
    evaluate_seal,
)
from src.robot.grasping.suction.synthesis import (
    SuctionConfig,
    SuctionGrasp,
    synthesize_suction_grasps,
)
from src.robot.grasping.suction.wrench import (
    WrenchConfig,
    WrenchResult,
    evaluate_wrench_resistance,
)

__all__ = [
    # seal
    "SealConfig",
    "SealResult",
    "evaluate_seal",
    # wrench
    "WrenchConfig",
    "WrenchResult",
    "evaluate_wrench_resistance",
    # scorer seam
    "SuctionQuality",
    "SuctionScorer",
    "AnalyticalSuctionScorer",
    # synthesis
    "SuctionConfig",
    "SuctionGrasp",
    "synthesize_suction_grasps",
]
