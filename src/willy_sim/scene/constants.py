"""Scene prim paths + geometry constants shared across the scene authoring modules."""
from __future__ import annotations


# Prim paths: the contract the runner and perception share.
ARM_PRIM = "/World/UR5e"
GRIPPER_VARIANT = "Robotiq_2f_85"
OBJECT_PRIM = "/World/Object"
CAMERA_PRIM = "/World/Cameras/Overhead"
TABLE_PRIM = "/World/Table"
BIN_WALL_PRIM = "/World/BinWall"  # static bin/tray walls (opt-in; spawned per `bin_walls` entry)
KLT_BIN_PRIM = "/World/RealKLT"  # the real small_KLT.usd asset as a visible static bin (opt-in)
MARKER_PRIM = "/World/CalibrationMarker"  # dedicated hand-eye calibration target (opt-in)

# Scene geometry in metres; the world frame and the robot base frame coincide. Of these, only
# CAMERA_POS_M and CAMERA_RESOLUTION are read: build_combined_scene falls back to them when the
# overhead camera config gives no position or resolution. The object, table and marker values are
# pre-config duplicates of the SimObjectConfig, SimTableConfig and SimMarkerConfig defaults the
# builder reads instead, so tuning one here reaches nothing; MARKER_POS_M has already drifted 1 mm
# from its schema default.
OBJECT_SIZE_M = (0.030, 0.030, 0.050)   # 30 mm footprint, 50 mm tall
OBJECT_POS_M = (0.45, 0.0, 0.025)       # centred under the camera (on the base X-axis)
CAMERA_POS_M = (0.45, 0.0, 1.00)        # straight above the object, 1 m over the tabletop
TABLE_SIZE_M = (0.80, 0.80, 0.40)
TABLE_POS_M = (0.45, 0.0, -0.20)        # top face at Z = 0 = base Z
CAMERA_RESOLUTION = (640, 480)
# Dedicated calibration marker: on the table, offset from the grasp object so the two never
# interfere. A distinct target the wrist camera views from many angles. Marker kind "flat" is a
# plain plate for the ground-truth calibration path; "aruco" builds a detectable ArUco board for
# the perception path, from solid-colour geometry rather than a texture. See markers.py.
MARKER_SIZE_M = (0.080, 0.080, 0.010)
MARKER_POS_M = (0.45, -0.15, 0.005)     # top face ~Z=10 mm; reachable, clear of the object
# ArUco board parameters for the perception-based calibration. The board is solid-colour geometry, a
# white base plus black cells, not a texture: Isaac RTX renders solid FixedCuboids reliably, while a
# single image on an analytic cube and a custom mesh with OmniPBR both map wrong, showing corners
# only or an untextured grey. The sizes follow the wrist camera's optics: it is narrow-FOV (fx about
# 1527, about 24 deg HFOV) and views the board from about 235 mm, so the frame is about 98 mm wide
# and a 48 mm marker on a 72 mm white base fits with margin and detects.
ARUCO_MARKER_ID = 0
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_LENGTH_MM = 48.0                 # the black 6x6-cell grid outer edge (solvePnP marker length)
ARUCO_PLATE_SIZE_M = 0.072             # white base plate (marker + ~12 mm quiet-zone each side)
ARUCO_MARKER_POS_M = (0.45, -0.15, 0.006)  # board on the table, facing up
