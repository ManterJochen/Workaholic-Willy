"""Second-look refinement, post-grasp verification, and next-best-view planning.

:mod:`refinement` re-perceives the target in a second frame and accepts a
bounded pose correction. :mod:`verification` decides whether a completed
pick actually holds the object. :mod:`active_perception` scores candidate
camera viewpoints when the scene needs another look.
"""
