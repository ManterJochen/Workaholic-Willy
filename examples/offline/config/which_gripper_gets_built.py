"""Which gripper a config tree builds, and what stands in when the arm cannot drive the one named.

A tree can name a gripper its arm has no way to drive. The build does not raise: it puts a stand-in
on the flange that holds nothing and carries the reason, and connecting that robot is refused.
"""

from willy import NoRealGripper, Robot, load_tree

# The desk profile: its arm drives no controller, so every build below runs on a machine with no
# robot and no vendor SDK. Its own gripper is a desk gripper, and that one builds.
tree = load_tree("console_dummy")
print(Robot.from_tree(tree))

# A suction cup switches the arm's digital outputs, and the desk arm has none. The vendor is given
# in memory and validated as a file would be; nothing is written.
cup = Robot.from_tree(tree.with_values({"robot.gripper.vendor": "vacuum"}))
print(cup)

# The stand-in never reaches a pick: connecting refuses it before the arm is commanded, and the
# refusal says what was asked for, why it could not be built and what to change.
try:
    with cup.connected():
        print(cup.home())  # never reached
except NoRealGripper as refused:
    print(f"connect refused: {refused}")

# The arm alone, with no gripper built at all: how a hand-eye calibration connects.
print(Robot.from_tree(tree, gripper=None))
