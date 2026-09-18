# Grasp debug images (`src.robot.grasping.visualization`)

Pictures that answer one question: why were these grasps ranked this way. A 2D overlay of the camera
image with the mask, the gripper boxes, the contact arrows and the scores, and an offline 3D view of the
cloud with wireframe grippers. Neither commands motion or touches a robot.

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
from src.robot.grasping.geometry import CameraIntrinsics
from src.robot.grasping.visualization import save_grasp_debug_image

save_grasp_debug_image(
    "logs/grasp_debug/one_frame.png",
    rgb_image,              # HxWx3, HxWx4 or HxW uint8
    candidates,             # camera-frame GraspPoint list, best first
    intrinsics=CameraIntrinsics(fx, fy, cx, cy),
    mask=segmentation_mask, # optional HxW bool or uint8
)
```

A simulation runner writes one PNG per attempt with a flag, under Isaac's own interpreter:
`<isaac-sim>/python.bat -m src.willy_sim.run_m2_pick --runs 3 --debug-frames logs/grasp_debug/m2`.

## The calls

| Call | Also takes | Returns |
| --- | --- | --- |
| `draw_grasp_debug_image(rgb, grasps, intrinsics=...)` | a mask, a gripper model, a `DebugDrawConfig` | an annotated BGR `uint8` image |
| `save_grasp_debug_image(path, rgb, grasps, intrinsics=...)` | the same; it creates the parent folders | the path it wrote |
| `build_grasp_scene(grasps, points_mm=...)` | a support plane, a gripper model, a `GraspViewerConfig` | a list of Open3D geometries |
| `show_grasp_scene(...)` | the same | nothing; it opens a blocking Open3D window |

The 2D overlay needs only OpenCV. The 3D view imports Open3D inside its entry points, so a machine
without Open3D still runs everything else and raises `ImportError` only when a 3D call is made. The
package depends downward on [`collision/`](../collision/README.md), [`geometry/`](../geometry/README.md)
and `GraspPoint`, and nothing imports it back.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `OSError` from `save_grasp_debug_image` | OpenCV could not write the PNG | check the path and the disk |
| `ImportError` from a 3D call | Open3D is not installed | install it, or use the 2D overlay |
| `ValueError` from `DebugDrawConfig` | a count, alpha, thickness or scale out of range | fix the value |
| a skipped candidate | a BASE-frame grasp: no BASE to CAMERA transform is available here | draw camera-frame candidates |

## Traps

- The overlay needs RGB. `GraspCalculator.compute()` renders the PNG only when it received an
  `rgb_image`, and the pick loop forwards the frame's RGB only while rendering is on. A ground-truth
  perception source whose camera returns no colour image gives `None`.
- The 3D path has no caller in the pick path. `show_grasp_scene` blocks until the window closes, so
  it belongs in a notebook or an offline script, never in a request.
- Everything is one frame at a time: no video, no temporal overlay, no browser output of its own.
- The operator console serves the cached overlay at `GET /v1/camera`, which the browser polls, and
  shows the image's age rather than pretending it is live.

## Status

| Capability | Evidence |
| --- | --- |
| The 2D overlay in a pick | measured in simulation: `run_m2_pick --debug-frames` writes one per attempt |
| The 3D viewer | never touched hardware: offline only |

## Files

| File | Holds |
| --- | --- |
| `debug_draw.py` | `DebugDrawConfig`, `draw_grasp_debug_image`, `save_grasp_debug_image` |
| `open3d_viewer.py` | `GraspViewerConfig`, `build_grasp_scene`, `show_grasp_scene` |

## Details

- [`collision/`](../collision/README.md) for the gripper models and the support plane drawn here, and
  [`geometry/`](../geometry/README.md) for `CameraIntrinsics`.
- [The console](../../../../api/README.md) for the route that serves the overlay.
- Tests: `tests/test_grasp_visualization_smoke.py`.
