"""The learned grasp ranker, which is a different product from the generator around it.

The sibling `net/`, `train/` and `eval/` groups are a learned grasp generator. These five modules
are not it. They score candidates the analytic stack already produced, they ship a gradient-boosted
tree rather than a network, their artifact kind is `grasp_gbt_ranker`, and they reach the runtime
through `DeepRankerContext` and the pick loop rather than through `DeepGraspCalculator`.

Import a submodule. `ranker.runtime` is the scorer a cell loads, `ranker.training` fits one,
`ranker.features` is the tabular row it eats, `ranker.context` loads one per cell, `ranker.shadow`
is the compare-without-acting seam.

Nothing is re-exported here, deliberately, and `net/__init__.py` holds the same rule. A package
that re-exports gives every symbol two addresses, and its `__init__` runs on every import beneath
it, pulling modules into memory the caller never asked for.
"""
