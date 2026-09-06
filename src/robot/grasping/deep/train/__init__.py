"""Everything that fits a set generator, and nothing that serves one.

The split against the package root is the one a reader needs: `deep/calculator.py` is what a cell
touches, and everything here runs offline, on a corpus, on a machine with a GPU. A cell never
imports this package, which is also why the root can stay free of torch at module level.

The pieces. `train.trainer` runs the folds (`train_set_generator`), `train.step` is one step of
one batch, `train.recipes` holds the named bundles and the two compute tiers, `train.progress` is
the terminal bar, `train.synthetic_control` replaces the corpus target with a synthetic one so a run
can prove the pipeline learns anything at all.

A control run is not a result. `train.synthetic_control` fits a target that has nothing to do with
grasping; its numbers show only that the machinery can fit something. The run's report stamps that,
so a diagnostic is not quoted later as a result.

Nothing is re-exported here; import the submodule you mean.
"""
