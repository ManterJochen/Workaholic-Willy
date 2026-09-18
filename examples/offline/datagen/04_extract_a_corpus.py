"""Extract the point-cloud corpus a grasp generator trains on from a labelled dataset, with jaw grasps alone
or with suction grasps beside them.

Each scene becomes one .npz: every rendered view fused into one cloud in millimetres, the bin and the table kept
in it so a generator learns not to grasp into them, and the grasp labels beside the geometry.
"""

import tempfile
from pathlib import Path

import numpy as np

from willy import DatasetBuild

with tempfile.TemporaryDirectory() as work:
    build = DatasetBuild.from_file(name="extracted", scenes=8, seed=0, engine="none", out_root=work)
    build.render()
    labelled = build.label(density="default")

    # An object a suction cup can take and a jaw cannot carries nothing in a jaw-only corpus.
    for family, bucket in sorted(labelled.summary["by_family"].items()):
        print(f"{family:7s} {bucket['objects']:3d} objects, {bucket['objects_with_jaw']:3d} with a jaw "
              f"grasp, {bucket['objects_with_suction']:3d} with a suction grasp")

    # `kinds` is the one place a corpus decides which grasps exist; the labeller wrote both.
    for kinds in (("jaw",), ("jaw", "suction")):
        corpus = Path(work) / "_".join(kinds)
        extracted = build.clouds(corpus, kinds=kinds)
        print(f"{' and '.join(kinds)}: {extracted.summary['scenes_written']} scenes, "
              f"{extracted.summary['points']} points")
        # The first scene that holds a jaw grasp. The suction arrays exist in both corpora and are empty in
        # the jaw-only one: kinds changes the rows, never the keys.
        for path in sorted(corpus.glob("*.npz")):
            with np.load(path) as scene:
                if len(scene["grasp_position_mm"]):
                    print(f"  {path.stem}: points {scene['points_mm'].shape}, jaw grasps "
                          f"{scene['grasp_position_mm'].shape}, suction grasps {scene['suction_position_mm'].shape}")
                    break
