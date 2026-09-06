"""The learned 6-DoF grasp generator, selected by `robot.grasping.calculator: deep`.

The package lands seam-first, and the seam is what it re-exports: `protocol.py` names the contract
the analytic calculator has always satisfied implicitly, so that a second implementation can be
typed and swapped by config rather than by editing every construction site. Behind that seam sit
the model in `net`, the corpus reader in `corpus`, the training loop in `train`, the instruments in
`eval`, the learned ranker in `ranker` and the importers in `foreign`.

No weights ship. A cell trains its own on its own scenes.
"""

from src.robot.grasping.deep.protocol import GraspCandidateGenerator

__all__ = ["GraspCandidateGenerator"]
