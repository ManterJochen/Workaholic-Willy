"""The shared grasping value objects, the vocabulary every other tier speaks.

They are frozen, self-validating carriers with no behaviour of their own: the grasp
candidate in :mod:`grasp_point`, the pipeline result and its failure reasons in
:mod:`feedback`, the sampling-mode contract in :mod:`modes`, and the perception
snapshot in :mod:`perception`. Nothing here imports another grasping tier, so
everything else depends on these names freely.
"""
