"""Debug visualisation primitives for the grasping pipeline.

Two surfaces:

* :mod:`debug_draw` is pure 2D and needs only OpenCV. It renders the RGB image with a mask
  overlay, the projected gripper boxes, the contact arrows and a score bar. It always
  imports, but the in-process overlay is opt-in: ``grasp_calculator`` renders it only when
  ``render_debug_images=True``, and the default of ``False`` forwards no rgb, so nothing is
  drawn. The PNG bytes go back to ``grasp_calculator`` in process, since no web server
  ships. It projects CAMERA-frame grasps only, and it does not fire on the dense-vision
  path, where the demo draws its own perception panel.
* :mod:`open3d_viewer` is a 3D Open3D viewer for offline debugging. It imports Open3D
  lazily, so an environment without Open3D still runs the rest of the package.

Both surfaces take the same ``GraspPoint`` and point-cloud inputs, so the scene reads the
same wherever it is inspected.
"""

from .debug_draw import (
    DebugDrawConfig,
    draw_grasp_debug_image,
    save_grasp_debug_image,
)
from .open3d_viewer import (
    GraspViewerConfig,
    build_grasp_scene,
    show_grasp_scene,
)

__all__ = [
    "DebugDrawConfig",
    "GraspViewerConfig",
    "build_grasp_scene",
    "draw_grasp_debug_image",
    "save_grasp_debug_image",
    "show_grasp_scene",
]
