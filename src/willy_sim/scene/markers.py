"""ArUco calibration-board authoring (solid-colour geometry, RTX-reliable).

Builds the fixed board (perception-based hand-eye calibration) and the tool-mounted board (the
eye-to-hand overhead observing a moving marker). ``cv2`` / ``isaacsim`` / ``pxr`` imports are lazy.
"""
from __future__ import annotations

import numpy as np

from .constants import (
    ARUCO_DICT_NAME,
    ARUCO_LENGTH_MM,
    ARUCO_MARKER_ID,
    ARUCO_MARKER_POS_M,
    ARUCO_PLATE_SIZE_M,
    MARKER_PRIM,
)




def add_aruco_marker(
    stage: object,
    *,
    prim_path: str = MARKER_PRIM,
    pos_m: tuple[float, float, float] = ARUCO_MARKER_POS_M,
    plate_size_m: float = ARUCO_PLATE_SIZE_M,
    marker_id: int = ARUCO_MARKER_ID,
    dict_name: str = ARUCO_DICT_NAME,
    length_mm: float = ARUCO_LENGTH_MM,
) -> str:
    """Author an ArUco board from solid-colour geometry: a white base plus black cells.

    Reads the marker's NxN cell grid from ``cv2.aruco`` (a DICT_4X4 marker is a 6x6 grid: a 1-cell
    black border plus 4x4 data), then builds a white :class:`FixedCuboid` base plus one black
    :class:`FixedCuboid` per black cell. Solid colours render reliably in Isaac RTX, where a single
    texture on an analytic cube and a mesh with OmniPBR both map wrong. Black cells are slightly
    oversized so adjacent ones merge into the solid regions ArUco expects. The black grid's outer
    edge is :data:`ARUCO_LENGTH_MM`, the solvePnP marker length; the white base adds the quiet-zone.
    Returns the prim path. The ``cv2``, ``isaacsim`` and ``pxr`` imports are lazy.
    """
    import cv2  # type: ignore[import-not-found]
    from isaacsim.core.api.objects import FixedCuboid  # type: ignore[import-not-found]
    from pxr import UsdGeom  # type: ignore[import-not-found]

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    grid_n = int(dictionary.markerSize) + 2  # data cells + 1-cell black border each side
    cell_px = 12
    img = aruco.generateImageMarker(dictionary, int(marker_id), grid_n * cell_px)

    px, py, pz = (float(v) for v in pos_m)
    UsdGeom.Xform.Define(stage, prim_path)  # type: ignore[arg-type]  # parent holding base + cells
    base_th = 0.002
    FixedCuboid(
        prim_path=prim_path + "/base", name="aruco_base",
        position=np.array([px, py, pz], dtype=np.float64),
        scale=np.array([plate_size_m, plate_size_m, base_th], dtype=np.float64),
        color=np.array([1.0, 1.0, 1.0]),
    )
    cell_m = (float(length_mm) / 1000.0) / grid_n
    # Thin black cells sit just above the base top, so the relief is minimal and the detected outer
    # corners stay in the marker plane. Relief shifts those corners and biases the PnP pose at
    # oblique views.
    cell_z = pz + base_th / 2.0 + 0.0003
    n = 0
    for r in range(grid_n):
        for c in range(grid_n):
            if int(img[r * cell_px + cell_px // 2, c * cell_px + cell_px // 2]) < 128:  # black cell
                cx = px + (c - (grid_n - 1) / 2.0) * cell_m
                cy = py - (r - (grid_n - 1) / 2.0) * cell_m  # image row down -> -Y (orientation is irrelevant for AX=XB)
                FixedCuboid(
                    prim_path=f"{prim_path}/cell_{n}", name=f"aruco_cell_{n}",
                    position=np.array([cx, cy, cell_z], dtype=np.float64),
                    scale=np.array([cell_m * 1.02, cell_m * 1.02, 0.0006], dtype=np.float64),
                    color=np.array([0.0, 0.0, 0.0]),
                )
                n += 1
    return prim_path



def mount_tool_aruco_marker(
    session: object,  # noqa: ARG001 (kept for call symmetry; authoring uses the default stage)
    *,
    parent_prim: str = "/World/UR5e/wrist_3_link",
    length_mm: float = ARUCO_LENGTH_MM,
    dict_name: str = ARUCO_DICT_NAME,
    marker_id: int = ARUCO_MARKER_ID,
    y_offset_mm: float = 70.0,
) -> str:
    """Author an ArUco board rigidly on the tool, for eye-to-hand calibration.

    The board is a child of ``parent_prim``, so it moves with the arm, and it faces the wrist -Y
    axis, which points up toward the overhead camera at a downward tool pose. Its cells are
    solid-colour ``UsdGeom.Cube`` prims, like the fixed board. ``run_eth_calibrate`` uses it: the
    fixed overhead camera observes this moving marker. The ``cv2``, ``pxr`` and ``omni`` imports are
    lazy, so this module imports without Isaac Sim installed. Returns the board prim path.
    """
    import cv2  # type: ignore[import-not-found]
    import omni.usd  # type: ignore[import-not-found]
    from isaacsim.core.api.materials import OmniPBR  # type: ignore[import-not-found]
    from pxr import Gf, UsdGeom, UsdShade  # type: ignore[import-not-found]

    stage = omni.usd.get_context().get_stage()
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    grid_n = int(dictionary.markerSize) + 2
    cell_px = 12
    img = aruco.generateImageMarker(dictionary, int(marker_id), grid_n * cell_px)

    # Matte materials (roughness 1, no metallic) so the upward-facing board does not throw a specular
    # highlight straight into the overhead camera: displayColor cubes blow out and leave the ArUco
    # pattern unreadable. Two shared materials, one white for the base and one black for the cells.
    def _matte(path: str, name: str, color: tuple[float, float, float]) -> object:
        mat = OmniPBR(prim_path=path, name=name, color=np.array(color, dtype=np.float64))
        for setter, val in (("set_reflection_roughness", 1.0), ("set_metallic_amount", 0.0)):
            try:
                getattr(mat, setter)(val)
            except Exception:  # noqa: BLE001 (OmniPBR setter surface varies by version)
                pass
        return UsdShade.Material(stage.GetPrimAtPath(path))

    white_mat = _matte("/World/Looks/EthMarkerWhite", "eth_marker_white", (1.0, 1.0, 1.0))
    black_mat = _matte("/World/Looks/EthMarkerBlack", "eth_marker_black", (0.0, 0.0, 0.0))

    root = parent_prim.rstrip("/") + "/eth_aruco"
    UsdGeom.Xform.Define(stage, root)
    cell_m = (float(length_mm) / 1000.0) / grid_n
    yo = float(y_offset_mm) / 1000.0
    plate = float(length_mm) / 1000.0 + 2.0 * cell_m  # white base with a quiet-zone margin

    def _cube(path: str, center: tuple[float, float, float], half: tuple[float, float, float],
              material: object) -> None:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(2.0)  # spans [-1, 1]; scale below sets the half-extents in metres
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*center))
        xf.AddScaleOp().Set(Gf.Vec3f(*half))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)

    # Thin white base (half-Y 0.3 mm) with black cells flush on top of it, toward -Y, which is the
    # camera side: -Y is "up" toward the overhead camera at a downward tool pose. The cells poke
    # about 0.4 mm proud. A thicker base buries them and the pattern comes out garbled.
    h_base = 0.0003
    h_cell = 0.0002
    cell_y = -yo - h_base - h_cell  # cell centre: bottom face flush with the base top face
    _cube(root + "/base", (0.0, -yo, 0.0), (plate / 2.0, h_base, plate / 2.0), white_mat)
    n = 0
    for r in range(grid_n):
        for c in range(grid_n):
            if int(img[r * cell_px + cell_px // 2, c * cell_px + cell_px // 2]) < 128:  # black cell
                cx = (c - (grid_n - 1) / 2.0) * cell_m
                cz = -(r - (grid_n - 1) / 2.0) * cell_m  # flip row->-Z: un-mirror for the overhead view
                _cube(
                    root + f"/cell_{n}", (cx, cell_y, cz),
                    (cell_m * 0.52, h_cell, cell_m * 0.52), black_mat,
                )
                n += 1
    return root
