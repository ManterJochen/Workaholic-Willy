# Grasp visualization (`src.robot.grasping.visualization`)

Debug rendering that answers one question, why were these grasps ranked this way, and nothing else.
A leaf of the grasping stack: it depends downward on [`collision/`](../collision/README.md),
[`geometry/`](../geometry/README.md) and the `GraspPoint` value objects, and nothing imports it back.

## What it guarantees

Opt-in and off by default. With the switch off no RGB reaches the calculator, nothing is rendered,
and the pick path costs nothing extra. Neither surface commands motion or touches a robot.

| File | Surface | Dependency | Output |
| --- | --- | --- | --- |
| `debug_draw.py` | 2D overlay | OpenCV, always | an annotated BGR `uint8` image, and a PNG on disk or in process |
| `open3d_viewer.py` | 3D scene | Open3D, imported lazily | an interactive point cloud with wireframe grippers, offline |

The 3D viewer imports Open3D inside the entry points, so a host without Open3D still runs the 2D path
and the rest of the package; it raises `ImportError` only when a 3D entry point is actually called.

## The public surface

2D, from `debug_draw`:

- `DebugDrawConfig(...)`, a frozen config for candidate count, mask alpha, line and arrow thickness,
  font scale and the score-bar and metadata toggles. It validates its ranges on construction.
- `draw_grasp_debug_image(rgb_image, grasps, *, intrinsics, mask=None, gripper_model=None,
  config=None, label=None, telemetry=None)` returns the annotated BGR image. It paints the mask
  overlay, each projected gripper collision box under the pinhole model with points behind the camera
  dropped, a contact arrow along the closing axis, per-candidate rank and score labels, a score bar
  and a metadata strip. No matplotlib and no server.
- `save_grasp_debug_image(path, ...)` renders and writes the PNG, creating parent directories, and
  raises `OSError` if the write fails.

3D, from `open3d_viewer`:

- `GraspViewerConfig(...)`, the frozen config for the scene builder.
- `build_grasp_scene(grasps, *, points_mm=None, support_plane=None, gripper_model=None, config=None)`
  returns a list of Open3D geometries: a coordinate frame, an optional support-plane mesh, an optional
  point cloud, and one wireframe gripper per candidate coloured by rank.
- `show_grasp_scene(...)` opens a blocking Open3D window.

## Usage

```python
from src.robot.grasping.geometry import CameraIntrinsics
from src.robot.grasping.visualization import save_grasp_debug_image

save_grasp_debug_image(
    "logs/demo/grasp_debug.png",
    rgb_image,              # HxWx3, HxWx4 or HxW uint8
    candidates,             # camera-frame GraspPoint list
    intrinsics=CameraIntrinsics(fx, fy, cx, cy),
    mask=segmentation_mask, # optional HxW bool or uint8
)
```

In a live cell the calculator already holds the frame and the candidates, so the overlay is a switch
rather than a call:

```python
service.enable_debug_image_rendering()   # opt in; default off
report = service.pick()
png = service.last_debug_image_png       # bytes, or None if the frame carried no RGB
```

A simulation runner writes one PNG per attempt with a flag:

```bash
python -m src.willy_sim.run_m2_pick --runs 3 --prompt "a red cube" \
    --debug-frames logs/grasp_debug/m2 --record-log logs/grasp_debug/m2/records.jsonl
```

## Traps

Only camera-frame candidates are drawn. A base-frame candidate is skipped, because no BASE to CAMERA
transform is available here.

The overlay needs RGB. `GraspCalculator.compute()` renders and caches the PNG only when an
`rgb_image` was passed, and the pick loop forwards the perception frame RGB only while
`render_debug_images` is on. That means the overlay fires on a vision path and not on a
ground-truth perception path whose camera returns no colour image.

The 3D path has no first-party caller. It is for offline and notebook debugging, and
`show_grasp_scene` blocks until the window is closed, so it must never be reached from a request
path. The rendering is single-frame throughout: no video, no temporal overlay, no interactive
picking, no browser output.

The operator console reads the cached 2D PNG over `GET /v1/camera`, which the browser polls, and
shows the age of the image rather than pretending it is live.

## See also

- [`../README.md`](../README.md) for the pick pipeline this package annotates
- [`../collision/README.md`](../collision/README.md) for the gripper models and support plane drawn here
- [`../geometry/README.md`](../geometry/README.md) for `CameraIntrinsics` and the point-cloud helpers
- [`../../../../api/README.md`](../../../../api/README.md) for the console route that serves the overlay
