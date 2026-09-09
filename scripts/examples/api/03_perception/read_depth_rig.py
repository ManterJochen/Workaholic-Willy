"""Which camera a cell opens, and whether it can deliver the depth a grasp is planned from.

`source` is a discriminated union: an `rgbd` rig hands back colour plus uint16 millimetre depth and
a camera matrix the device reports, while a stereo rig hands back two views, no depth channel and
no pinhole K, so nothing can unproject its frames. `camera.cameras.primary_rig_id` names the cell's
camera, and the cell builder refuses one that is switched off or that carries no depth.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 03_perception, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402
from src.config import load_config  # noqa: E402
from src.camera.orchestration.frame_provider import FrameProvider  # noqa: E402
from src.camera.setup.image_taking.frames import RGBDFrame  # noqa: E402

# 1. The camera tree, and the one rig a cell would open. Named, not inferred: three predicates over
#    one rig list used to disagree, and on the shipped profile they disagreed about this very rig.
cameras = load_config().camera.cameras
rigs = list(cameras.rigs)
primary = next(r for r in rigs if r.rig_id == cameras.primary_rig_id)

# 2. What each rig would hand a consumer, read off `source` and `rgbd_backend`. The `opencv`
#    default returns colour with an empty depth channel and no intrinsics on any device whose
#    depth lives behind a vendor SDK, and the calculator then refuses far away from the cause.
for rig in rigs:
    print(f"{rig.rig_id:16} {rig.source:14} enabled={rig.enabled} "
          f"rgbd_backend={getattr(rig, 'rgbd_backend', None)}")

# 3. The two things a grasp cell demands of its primary, in the order `build_real_components`
#    asks them. Both are CellBuildRefused there, and that refusal names this tree's RGB-D rigs.
usable = bool(primary.enabled) and primary.source == "rgbd"
print(f"primary {primary.rig_id!r} is {primary.source}, enabled={primary.enabled}: "
      + ("a cell opens this one" if usable else "a cell refuses to build on this one"))

# 4. Open exactly one rig. Constructing the provider touches no device, so knowing every rig costs
#    nothing; holding one is a deliberate act, and `provider.open()` would open all of them.
if usable:
    provider = FrameProvider(rigs)
    provider.open_rig(primary.rig_id)
    try:
        # 5. One frame, and the matrix this rig reports. A stereo rig answers None here.
        frame = provider.grab(primary.rig_id)
        matrix = provider.get_intrinsics(primary.rig_id)
        print(type(frame).__name__, "fx", None if matrix is None else float(matrix[0][0]))
        # 6. Depth as delivered: a zero is a hole, left honest rather than filled in.
        if isinstance(frame, RGBDFrame):
            depth = np.asarray(frame.depth)
            print(depth.shape, f"{int((depth == 0).sum())}/{depth.size} holes")
    finally:
        provider.release_rig(primary.rig_id)
