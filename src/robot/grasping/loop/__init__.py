"""The bin-picking orchestrator that drives one pick attempt end to end.

:mod:`pick_loop` closes the loop that perceives, scores, decides, commits,
executes and recovers (:class:`BinPickingOrchestrator`).
This is the top of the grasping stack: it imports downward into every other
tier and nothing imports it back.
"""
