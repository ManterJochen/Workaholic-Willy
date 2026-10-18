# Grasp debug images (`src.robot.grasping.visualization`)

Pictures that answer one question: why were these grasps ranked this way. A 2D overlay of the camera
image with the mask, the gripper boxes, the contact arrows and the scores. It commands no motion and
touches no robot. The offline 3D viewer (Open3D) and `save_grasp_debug_image` left on 2026-10-18 with
the code nothing called.

The overlay is a switch on the pick service, off by default. This runs at a desk on the dummy arm of the
`console_dummy` profile:

```python
from pathlib import Path

from willy import Cell, load_tree

cell = Cell.rehearsal(load_tree("console_dummy").robot)   # a dummy arm and a synthetic scene
service = cell.build()
service.enable_debug_image_rendering()                    # off by default
with cell.connected() as live:
    print(live.service.pick().outcome)
png = service.last_debug_image_png                        # PNG bytes, or None when the frame had no RGB
if png is not None:
    Path("grasp_debug.png").write_bytes(png)
```

Call it directly to draw a frame of your own:

```python
import cv2

from src.robot.grasping.geometry import CameraIntrinsics
from src.robot.grasping.visualization import draw_grasp_debug_image

image = draw_grasp_debug_image(
    rgb_image,              # HxWx3, HxWx4 or HxW uint8
    candidates,             # camera-frame GraspPoint list, best first
    intrinsics=CameraIntrinsics(fx, fy, cx, cy),
    mask=segmentation_mask, # optional HxW bool or uint8
)
cv2.imwrite("one_frame.png", image)   # the overlay is BGR, as OpenCV writes it
```

A simulation runner writes one PNG per attempt with a flag, under Isaac's own interpreter:
`<isaac-sim>/python.bat -m src.willy_sim.run_m2_pick --runs 3 --debug-frames logs/grasp_debug/m2`.

## The calls

| Call | Also takes | Returns |
| --- | --- | --- |
| `draw_grasp_debug_image(rgb, grasps, intrinsics=...)` | a mask, a gripper model, a `DebugDrawConfig` | an annotated BGR `uint8` image |

The overlay needs only OpenCV. The package depends downward on [`collision/`](../collision/README.md),
[`geometry/`](../geometry/README.md) and `GraspPoint`; the pick run imports `scene_debug.py` for its
per-look pictures.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `DebugDrawConfig` | a count, alpha, thickness or scale out of range | fix the value |
| a skipped candidate | a BASE-frame grasp: no BASE to CAMERA transform is available here | draw camera-frame candidates |

## Traps

- The overlay needs RGB. `GraspCalculator.compute()` renders the PNG only when it received an
  `rgb_image`, and the pick loop forwards the frame's RGB only while rendering is on. A ground-truth
  perception source whose camera returns no colour image gives `None`.
- With a `closing_axis`, or the cell's `robot.motion.natural_closing_axis`, `last_debug_image_png` shows the
  grasps the pick chooses among, each the way round it closes, with `final` and `best_score` counted on
  them: the pick loop draws it again (`GraspCalculator.redraw_debug_image`) wherever it changed a result,
  and drops it where the calculator cannot redraw.
- Everything is one frame at a time: no video, no temporal overlay, no browser output of its own.
- The operator console serves the cached overlay at `GET /v1/camera`, which the browser polls, and
  shows the image's age rather than pretending it is live.

## Status

| Capability | Evidence |
| --- | --- |
| The 2D overlay in a pick | measured in simulation: `run_m2_pick --debug-frames` writes one per attempt |

## Files

| File | Holds |
| --- | --- |
| `debug_draw.py` | `DebugDrawConfig`, `draw_grasp_debug_image` |
| `scene_debug.py` | `draw_scene_debug`, `save_scene_debug`, `grasp_rectangle`, `GraspRectangle`: what one look saw, every detection's box, mask, label and score, and the grasp the pick chose; import it by module path |

## Details

- [`collision/`](../collision/README.md) for the gripper models and the support plane drawn here, and
  [`geometry/`](../geometry/README.md) for `CameraIntrinsics`.
- [The console](../../../../api/README.md) for the route that serves the overlay.
- Tests: `tests/robot/grasping/visualization/test_grasp_visualization_smoke.py`.
