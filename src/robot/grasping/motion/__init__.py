"""Approach, grasp and retreat choreography, with swept-path validation.

:mod:`execution_policy` drives the arm and the gripper through the three legs
of a pick, :mod:`trajectory_safety` checks the swept gripper volume against the
scene cloud before anything is committed, and :mod:`frame_resolver` supplies
the camera-to-base transform each pick needs, failing closed when it cannot.
"""
