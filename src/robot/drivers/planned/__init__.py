"""The planned arm: the judging and planning half every planned arm shares, and the cuRobo planner it drives (P0e E8).

``arm.PlannedArm`` is the base class of every arm whose motions are judged and planned; ``curobo_motion.CuroboPlanner``
plans and sends a cuRobo path over the arm's transport; ``planner_frame`` turns poses between the controller's base and
the planner's. Nothing here imports a vendor package.
"""
