"""Turn a segmentation mask + depth into ranked grasp candidates.

The :mod:`calculator` facade drives the whole generate step: it reads mask
geometry (:mod:`mask_analyzer`), back-projects pixels to millimetres
(:mod:`_camera_geometry`), synthesizes antipodal poses
(:mod:`_candidate_generator`), routes non-rigid targets
(:mod:`deformables`), and renders debug overlays (:mod:`_debug_renderer`).
"""
