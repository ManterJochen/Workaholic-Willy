"""One key chooses the grasp generator: the analytic one that ships, or one trained on your parts.

`robot.grasping.calculator` is `geometric` or `deep`. No trained weights ship with the library, so
on a machine with none trained the learned half stops at its refusal, which says what is missing.
"""

from willy import build_calculator, load_tree, preflight_calculator

# The base tree: the grasping block every profile starts from.
tree = load_tree(None)

# Check the selector without building anything, then build what it chose. A cell builds through
# the factory and never names a calculator class, so a tree asking for one generator cannot run
# the other.
print("selector:", preflight_calculator(tree.robot, data_dir=tree.root))
print("built:", type(build_calculator(tree.robot, data_dir=tree.root)).__name__)

# The same tree asking for the learned generator, given in memory and validated as a file would be.
deep = tree.with_values({"robot.grasping.calculator": "deep"})
try:
    print("selector:", preflight_calculator(deep.robot, data_dir=deep.root))
except FileNotFoundError as no_weights:
    # It never falls back to the analytic generator: grasps filed under the wrong generator's
    # name cannot be told apart afterwards.
    print(f"{no_weights} Train one on your own parts (examples/offline/training/) and name it in "
          f"robot.grasping.deep_generator.artifact_path.")
    raise SystemExit
print("built:", type(build_calculator(deep.robot, data_dir=deep.root)).__name__)
