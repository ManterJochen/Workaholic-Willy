"""Film one wrist-camera pick in Isaac Sim as an MP4, and print what was filmed and where the file landed.

Run it from the repository root with Isaac Sim's own interpreter and the root on its path, so it finds
willy (in PowerShell, `$env:PYTHONPATH = (Get-Location).Path`; in cmd, `set PYTHONPATH=%CD%`):
    <isaac-sim>/python.bat examples/simulation/04_isaac_record_a_pick.py
"""

import importlib.util
from pathlib import Path

from willy import record_demo

# OpenCV writes the MP4; it is asked for before Isaac boots rather than after the pick was filmed.
missing = [name for name in ("isaacsim", "cv2") if importlib.util.find_spec(name) is None]
if missing:
    print(f"Not importable here: {', '.join(missing)}; run this file with Isaac Sim's own python.bat.")
    raise SystemExit

# One pick, headless, filmed from a fixed camera beside the cell. A film scores nothing: a failed
# pick is filmed as willingly as a good one. fps is the playback speed; every simulator step is a frame.
summary = record_demo(out_path="logs/demo/pick.mp4", headless=True, fps=18)

# The part's rise is the simulator's own reading, so the clip comes with one fact about the pick.
print(f"{summary['frames']} frames, {summary['duration_s']:.1f} s, the part rose {summary['lift_mm']:.1f} mm")

# Where it landed, read off what the recorder returned; four stills sit beside it.
out = Path(summary["out"]).resolve()
print(f"{out} ({out.stat().st_size / 1_000_000:.1f} MB)")
print(sorted(still.name for still in out.parent.glob(f"{out.stem}_key*.png")))
