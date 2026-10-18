"""Turn a segmentation mask + depth into ranked grasp candidates.

The :mod:`calculator` facade drives the whole generate step: it gates the mask
(:mod:`mask_analyzer`), reads the lens (:mod:`_camera_geometry`), plans the part's
own body over its support (:mod:`body_grasps`, through :mod:`_support_footprint_stage`,
with the hand of :mod:`hand`), routes non-rigid targets (:mod:`deformables`), and
renders debug overlays (:mod:`_debug_renderer`).
"""
