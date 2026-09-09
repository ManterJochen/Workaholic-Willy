"""What a scene is, decided in full before anything renders.

`layout_scene` takes a config and a scene index and returns a `SceneSpec`: which assets, where each
one spawns, which cameras look at it. No engine, no GPU, milliseconds, which is why a layout mistake
costs seconds instead of a night of path tracing. These are spawn poses; the settled ones come back
from whichever engine renders them.
"""

import collections
import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 06_datagen, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from datagen.config import DatagenConfig, FamilyMixConfig  # noqa: E402
from datagen.scenes import ObjectPlacement, SceneFamily, layout_scene, plan_families  # noqa: E402

# 1. Describe the dataset. The four family weights are four physics regimes, not four densities.
config = DatagenConfig(scenes=40, seed=0, families=FamilyMixConfig(flat_orientation="upright"))

# 2. The mix is planned up front and interleaved, so 40 scenes at equal weights is exactly 10 each.
print(collections.Counter(family.value for family in plan_families(config)))

# 3. One scene in full. Millimetres and XYZW quaternions: this repository's units, not a renderer's.
spec = layout_scene(config, 0)
print(spec.scene_id, spec.family.value, spec.domain.value, len(spec.objects),
      spec.drop_height_mm, len(spec.bin_walls))
for camera in spec.cameras:
    print(camera.name, camera.mount.value, camera.position_mm, camera.horizontal_fov_deg)
for placed in spec.objects[:4]:
    print(placed.asset_id, placed.position_mm, placed.orientation_xyzw, placed.mass_kg)

# 4. The dice are rolled here and never in the renderer, so a seed reproduces a scene exactly.
print(layout_scene(config, 0) == spec, layout_scene(config, 1) != spec)

# 5. The quieter decision. `random` tips objects over, and a tipped object admits a jaw grasp about
#    three times as often as one standing on its base. The pile family is skipped here: an object
#    about to be dropped has no rest pose yet, so it draws a full orientation either way.
tipped = DatagenConfig(scenes=40, seed=0, families=FamilyMixConfig(flat_orientation="random"))
for label, chosen in (("upright", config), ("random", tipped)):
    placements: list[ObjectPlacement] = []
    for index in range(12):
        scene = layout_scene(chosen, index)
        if scene.family is not SceneFamily.PILE:
            placements.extend(scene.objects)
    standing = sum(p.orientation_xyzw[0] == 0.0 and p.orientation_xyzw[1] == 0.0 for p in placements)
    print(label, standing, "of", len(placements), "flat placements stand on their base")
