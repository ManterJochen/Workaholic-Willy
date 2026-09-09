"""Import a public 6-DoF grasp corpus as the same .npz scene files the local generator writes.

Nothing is vendored, and this one needs the network: the four archives are addressed by HTTP range
request, so 229 GB can be sized without fetching a byte and only the members of the scenes you ask
for are read. Two producers, one consumer, and the corpus loader cannot tell them apart.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 07_training, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.robot.grasping.deep.foreign import grasp_anything  # noqa: E402
from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY  # noqa: E402

OUT = "logs/examples/training/foreign"

# 1. What the source declares about itself. The licence is the part that cannot be fixed later.
source = grasp_anything.SOURCE
print(f"{source.title}\n  {source.url}\n  {source.scenes:,} scenes, licence {source.licence}")

# 2. Size the four archives without downloading them: a zip keeps its directory at the end, so the
#    last few kilobytes of each answer what is inside.
try:
    archives = grasp_anything.archives()
except OSError as unreachable:
    print(f"the host did not answer ({unreachable}), and everything below needs it")
else:
    print(f"{sum(z.total for z in archives) / 1e9:.0f} GB across four archives, none of it fetched")

    # 3. Which hand the labels are cut for. The aperture is this repository's number for that jaw,
    #    and a published opening wider than it is refused rather than believed.
    gripper = grasp_anything.DEFAULT_GRIPPER
    print(f"labels for the {gripper} jaw at {JAW_GEOMETRY[gripper]['aperture_mm']:.0f} mm")

    # 4. Import a few scenes. The full index is read once and cached beside them, about 460 MB: a
    #    sliced index drops the masks, and a scene without masks collapses to ONE training unit.
    written = grasp_anything.import_scenes(OUT, limit=4, gripper=gripper)
    print(f"{written['written']} written and {written['skipped_present']} already present in {OUT},"
          f" {written['training_units']} training unit(s), {written['refused_too_wide']} label(s) "
          f"refused as wider than that jaw")

    # 5. How to read the published poses is MEASURED against the cloud being imported, not assumed,
    #    and lands in provenance.json beside the scenes. A resumed import fetches nothing to fit.
    print(f"centre offset {written.get('centre_offset_m', '(nothing fetched to fit)')}")
