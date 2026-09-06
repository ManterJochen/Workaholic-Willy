"""Instruments that grade something, as opposed to producing it.

The direction is one-way, and that is what keeps the split safe: this package imports `grasps/`, and
`grasps/` imports nothing from here.

    ladder                  grades `GraspCalculator` against the geometry, rung by rung
    floors                  what luck and the dumbest sensible heuristic score on the same scenes
    approach_tilt           coverage by approach tilt, which makes the side-grasp population visible
    support_estimators      four ways to answer "where is the surface this object stands on"
    cross_view_association  does cross-view association pick the right object, on real masks
    camera_replay           drives the real camera adapter over rendered views and stored masks
    service                 the code API: a labelled dataset and the verbs that grade a calculator

    label_set_shape         how multimodal a supervised point is on the current corpus
    lateral_spread          is the lateral target multi-modal per supervised point
    sweep_closing_axes      does a denser closing-axis sample find grasps the current one misses
    sweep_label_density     how fine should the corpus be labelled, and what does each step cost
    sweep_anchor_and_poses  the anchor pitch and the remaining rest poses, side by side

The five sweeps are package modules, not scripts, and each carries the three rules that difference
imposes: `main(argv=None)` parses its own arguments, every module-level constant is a default rather
than a value parsed at import, and there is no import-time side effect at all. A module that reads
`sys.argv` at import takes the caller's arguments instead of its own; one that calls
`logging.disable` at import silences the importer's whole process; and an absolute path with a drive
letter runs on one machine.

"No importer, no CLI route, no `main()`" is the wrong test for whether a module here is dead.
Several of these have only ever been invoked by hand, from code, because there was no other way in:
in datagen that is the symptom of a missing code API, not evidence of death.

Nothing is re-exported here. Import the submodule you mean.
"""
