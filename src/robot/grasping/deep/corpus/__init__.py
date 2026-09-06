"""What a scene is, what a training unit is, and how a grasp is written down.

No network, no training loop, no torch at module level except where a batch is assembled. This is
the layer both sides of the train/serve seam read, and that is the point of grouping it: the runtime
calculator imports `corpus.sample.build_sample`, the very function the trainer uses, so a change to
the input convention cannot land on one side only.

The pieces. `corpus.sample` turns one rendered `.npz` into one training sample, `corpus.index`
holds the asset grouping, the asset-disjoint folds and the unit index, `corpus.grasp_encoding` is
the grasp/label encoding and the per-point graspability field.

Nothing is re-exported here; import the intended submodule directly.
"""
