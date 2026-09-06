"""Reading a public grasp dataset into the shape the training loop expects.

Two producers feed the same loop. The generator in this repository renders and labels scenes; this
package converts a dataset somebody else published. The loop cannot tell them apart: a user with no
simulator can still train, and a user with their own cell can still generate.

    contract    the nine arrays a sample must carry, and what each one means
    convention  which axis is the approach, which is the closing axis, what the translation refers to

Start at `convention`. A published pose matrix does not say how to read itself, and reading it wrong
produces a corpus that trains confidently to a wrong answer. Nothing in this package assumes a
convention; it measures one, and refuses when the geometry does not settle it.
"""

from src.robot.grasping.deep.foreign.contract import (
    SAMPLE_KEYS, SampleProblem, describe_sample, validate_sample)
from src.robot.grasping.deep.foreign.convention import (
    Convention, ConventionProof, ConventionScore, agree, prove_convention)

__all__ = ["SAMPLE_KEYS", "SampleProblem", "validate_sample", "describe_sample",
           "Convention", "ConventionScore", "ConventionProof", "prove_convention", "agree"]
