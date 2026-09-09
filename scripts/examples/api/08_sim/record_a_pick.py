"""Film one eye-in-hand pick as a cinematic MP4 and read back where the file landed.

A gate returns a rate; a recorder returns a picture and scores nothing, so it films a failed run as
willingly as a good one and the file carries no claim about whether the cell works. It needs Isaac's
bundled interpreter and OpenCV's mp4v writer, and underneath it the motion engines the sim cell is
configured for: the shared bootstrap raises ``DegradedMotionStackError`` rather than film a run
whose paths came from a planner the cell was not configured to use.
"""

import importlib.util
import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 08_sim, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.willy_sim.run_eih_demo import record_demo  # noqa: E402

# 1. The two things the recorder cannot film without, asked before either is used. cv2 is a plain
#    pip dependency and is checked anyway: a recorder that boots Isaac and films a whole run before
#    discovering it cannot encode has spent minutes to reach an ImportError.
missing = [name for name in ("isaacsim", "cv2") if importlib.util.find_spec(name) is None]

if missing:
    print(f"not importable in this interpreter: {', '.join(missing)} -- so nothing was filmed.")
    print("Isaac's own bundled Python runs this file unchanged:")
    print(r"    <isaac-sim>\python.bat scripts/examples/api/08_sim/record_a_pick.py")
else:
    # 2. One pick, filmed headless from a fixed cinematic three-quarter camera. `fps` is playback
    #    speed, not capture rate: `capture_every=1` films every simulator step, and raising it
    #    trades smoothness for file size and encode time. The path is passed rather than defaulted,
    #    because the module's own DEFAULT_OUT is an absolute path into a tree that no longer exists.
    summary = record_demo(out_path="logs/demo/eih_pick_demo.mp4", headless=True,
                          fps=18, capture_every=1)

    # 3. What was filmed. `lift_mm` is the object's ground-truth rise, so the clip and one honest
    #    number about the run it shows come out of the same call.
    print(f"{summary['frames']} frames, {summary['duration_s']:.1f} s, "
          f"lift {summary['lift_mm']:.1f} mm")

    # 4. Where it landed, read off the recorder's own return value rather than restated from the
    #    argument: that is what makes the path a fact instead of an echo.
    out = Path(summary["out"]).resolve()
    print(f"{out} ({out.stat().st_size / 1_000_000:.1f} MB)")

    # 5. Four keyframe PNGs sit beside the MP4, evenly spaced across the clip, for stills.
    print([p.name for p in sorted(out.parent.glob(f"{out.stem}_key*.png"))])
