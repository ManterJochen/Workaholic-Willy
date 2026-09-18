"""Lay out a dataset's scenes before anything renders: the family mix, one scene in full, and what the seed fixes.

A layout takes milliseconds and needs no engine, so a wrong scene costs seconds here instead of a night of rendering.
"""

import collections

from willy import DatasetBuild, layout_scene

# Forty scenes described in code; layout_scene reads the build's config. Nothing renders and nothing is
# written, so no engine is needed.
config = DatasetBuild.from_file(name="planned", scenes=40, seed=0).config

# Four families, four physics regimes: sparse, packed, pile and bin, interleaved by their weights.
print(collections.Counter(layout_scene(config, index).family.value for index in range(40)))

# One scene in full. Millimetres and XYZW quaternions; these are spawn poses, which the engine then settles.
scene = layout_scene(config, 0)
print(scene.scene_id, scene.family.value, len(scene.objects), "objects,", len(scene.bin_walls), "bin walls")
for camera in scene.cameras:
    print(f"  camera {camera.name} ({camera.mount.value}) at {[round(v) for v in camera.position_mm]} mm")
for placed in scene.objects[:3]:
    print(f"  {placed.asset_id} at {[round(v) for v in placed.position_mm]} mm, "
          f"turned {[round(q, 3) for q in placed.orientation_xyzw]}")

# The dice are rolled here and never in the renderer, so a seed and an index reproduce a scene exactly.
print("scene 0 again is the same:", layout_scene(config, 0) == scene)
print("scene 1 is another:", layout_scene(config, 1) != scene)

# Whether objects on a flat surface stand on their base or lie where they tip decides which grasps exist.
tipped = DatasetBuild.from_file(name="tipped", scenes=40, seed=0,
                                overrides={"families": {"flat_orientation": "random"}}).config
for name, chosen in (("upright", config), ("random", tipped)):
    scenes = [layout_scene(chosen, index) for index in range(12)]
    # A pile drops its objects, so they have no rest pose yet and draw a full orientation either way.
    flat = [placed for spec in scenes if spec.family.value != "pile" for placed in spec.objects]
    standing = sum(placed.orientation_xyzw[:2] == (0.0, 0.0) for placed in flat)
    print(f"{name:8s} {standing} of {len(flat)} flat placements stand on their base")
