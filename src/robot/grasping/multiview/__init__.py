"""Fuse, localize, and synthesize grasps across multiple camera views.

:mod:`association` decides which detection in another camera is the same
physical object and :mod:`scene_geometry` turns that decision into one cloud
per object. :mod:`fusion` accumulates per-view depth into a bounded
voxel-occupancy grid that no pick path reads, :mod:`localize` fuses per-camera
target centroids into one BASE-frame estimate, and :mod:`synthesis` derives a
top-down grasp from the fused point cloud. :mod:`_fusion_geometry` and
:mod:`_fusion_queries` are the numpy hot loop and the read-side query leaves.
"""
