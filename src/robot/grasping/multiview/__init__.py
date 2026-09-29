"""Fuse and localize objects across multiple camera views.

:mod:`association` decides which detection in another camera is the same
physical object and :mod:`scene_geometry` turns that decision into one cloud
per object, which the grasp generator receives. :mod:`localize` fuses
per-camera target centroids into one BASE-frame estimate. :mod:`unseen_side`
asks whether both faces a jaw closes on were seen, and names the generated
wrist view, an orbit about the target, that would turn an unseen one toward
the camera.

``synthesis``, which derived a top-down grasp from the fused cloud outside the
generator, was removed on 2026-09-29: only two sim levers called it, and a
fused cell's grasp comes from the generator fed the fused cloud.
"""
