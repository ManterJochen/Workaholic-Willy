"""Producing grasp labels, and screening them with physics. One job.

    labels          the geometry that proposes a grasp, and writes `grasps.jsonl`
    verdict         the gripper envelopes and the per-candidate verdict
    shapes          the solid a label is computed against
    physics         the referee's contract, engine-independent
    physics_isaac   the referee on Isaac
    physics_mujoco  the referee on MuJoCo
    masks           the learned mask source, as an alternative to the renderer's ground truth
    identity        when two rows are the same grasp, so everything that must agree, agrees
    service         the physics sampling noun, its two verbs and the output-file rule

Grading a calculator lives in `datagen/eval/`, and the direction is one-way: `eval/` imports this
package, this package imports nothing from `eval/`. The one real crossing there ever was is now
`corpus.clouds.fuse_instance_clouds`, which is where it belonged: it unprojects depth into base
millimetres and fuses per instance, and there is nothing evaluative in it.

Nothing is re-exported here. Import the submodule you mean.
"""


from __future__ import annotations

__all__ = ["__doc__"]
