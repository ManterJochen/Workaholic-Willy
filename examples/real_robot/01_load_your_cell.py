"""Load your cell's config tree, ask it about one key, and try a change in memory before making it.

Nothing here opens a device: a tree is read from files and validated. Run it under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/01_load_your_cell.py
"""

from willy import load_tree

# The layers WILLY_PROFILE names, validated as one tree. A tree that does not load is a verdict,
# not an exception: printing it gives the refusal with the file and the line to fix.
tree = load_tree()
print(tree)
if not tree.ok:
    raise SystemExit(tree.exit_code)

# A key the desk checklist can block a real cell on: with robot.safety.payload.enforce true, a
# mass of 0.0 tells the controller the tool weighs nothing. The answer gives the value, its
# bounds, and the file and line that set it.
print(tree.explain("robot.safety.payload.mass_kg"))

# The weighed tool, given in memory: the tree loads again the way a load does, validated as a
# whole, and nothing is written. Every noun built from `weighed`, such as Cell.from_tree(weighed),
# reads these values; to keep them, write them in your cell's profile. The numbers are yours,
# from a bench scale and the tool's drawing.
weighed = tree.with_values({
    "robot.safety.payload.mass_kg": 1.2,
    "robot.safety.payload.cog_mm": [0.0, 0.0, 60.0],
})
print(weighed.explain("robot.safety.payload.mass_kg"))  # the file's value, then the one that wins

# A value the schema refuses comes back as a tree that did not load, with the sentence the same
# line in YAML would get.
print(tree.with_values({"robot.safety.payload.mass_kg": 80.0}))
