"""The self-collision guard on the exact link meshes or on a capsule proxy, and which one runs here.

`robot.safety.self_collision.backend` is `fcl`, the exact meshes, or `capsule`, a proxy that cannot see the gripper on a
joint move. An `fcl` guard with no mesh engine, or no mesh bundle for its arm, runs the proxy instead and says so only in
a log line, so the reading to trust is the probe.
"""

from willy import JointPositions, MotionStack, SafetyPreflight, create_arm, load_tree

# The base tree with a hand named, in memory: an exact mesh guard reads the hand, and the base tree names none on
# purpose. Put your own hand's registry name here.
cell = load_tree(None).with_values({"robot.gripper.model": "robotiq_2f85"})

# Which arm the mesh bundle is keyed on, which key named it, and what this machine holds for it.
print(MotionStack.from_robot_config(cell.robot).probe())

# One folded configuration in joint radians, a gripper finger driven into the forearm, judged as a joint move is
# judged at its target. The arm is built for its joint limits; nothing connects.
arm = create_arm(cell.robot.vendor, config=cell.robot)
folded = JointPositions([1.95, 0.38, -1.33, -0.55, 2.00, 0.79])
for backend in ("capsule", "fcl"):
    gate = SafetyPreflight.from_tree(cell.with_values({"robot.safety.self_collision.backend": backend}))
    refusal = gate.gate_joint_target(folded, arm=arm)
    print(f"{backend}: {'accepted' if refusal is None else refusal}")
