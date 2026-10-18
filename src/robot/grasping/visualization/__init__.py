"""Debug visualisation primitives for the grasping pipeline.

* :mod:`debug_draw` is pure 2D and needs only OpenCV. It renders the RGB image with a mask
  overlay, the projected gripper boxes, the contact arrows and a score bar. It always
  imports, but the in-process overlay is opt-in: ``grasp_calculator`` renders it only when
  ``render_debug_images=True``, and the default of ``False`` forwards no rgb, so nothing is
  drawn. The PNG bytes go back to ``grasp_calculator`` in process, since no web server
  ships. It projects CAMERA-frame grasps only, and it does not fire on the dense-vision
  path, where the demo draws its own perception panel.

The 3D Open3D viewer left on 2026-10-18: nothing called it.
"""

from .debug_draw import DebugDrawConfig, draw_grasp_debug_image

__all__ = [
    "DebugDrawConfig",
    "draw_grasp_debug_image",
]
