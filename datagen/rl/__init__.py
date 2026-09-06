"""Driving the real grasping stack over datagen scenes, to feed the offline RL trainers.

`perception` replays a scene as a camera rig, `occupancy` checks that the feature keys are
populated and vary, `collect` grades candidates in physics, and `proof` trains on the result.
`service` joins those into one object with one verb each.
"""
