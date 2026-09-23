"""What a hand-eye sweep looks at, and what one frame of it showed.

A sweep poses one calibration target per frame: a single ArUco marker, or a ChArUco board. This module
holds the vocabulary both share, in plain values and with no config import, so the config schema, the
marker source and the two estimators speak about the same thing:

* :class:`Observation`: what one frame showed. The pose (``T_cam_to_target``, 4x4, millimetres) or the
  reason there is none, the markers seen, the image points the pose was solved from and their RMS
  reprojection in pixels.
* :class:`CharucoPoseEstimator`: a ChArUco board posed from its interpolated chessboard corners.
  ``ArucoPoseEstimator.observe`` (``stereo/sub_modules/aruco_esti.py``) is the single-marker counterpart.
* :func:`estimator_for`: the estimator a target describes, read by attribute or by key.
* :func:`parse_board_spec`: the ``--board`` spelling of a target, as the mapping the schema validates.
* :func:`canonical_aruco_dict_name` and :func:`aruco_dictionary_size`: dictionary names as OpenCV spells
  them, and how many ids a dictionary holds.

Only relative target motions enter AX=XB (``B_i = M_{i+1} M_i^-1``), so a constant offset of the target
frame cancels. A board's origin at its corner is as valid as a marker's at its centre.

A per-frame failure of an estimator (a frame of the wrong shape, an OpenCV error on degenerate points) is
raised as :class:`StereoCalibrationError`, a :class:`CalibrationError`. A target that is simply not in
view is not an error: ``observe`` returns an :class:`Observation` whose ``why_not`` says so.

Measured offline on a synthetic render (known camera matrix and board pose), never on a physical camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError

__all__ = [
    "CharucoPoseEstimator",
    "Observation",
    "aruco_dictionary_size",
    "canonical_aruco_dict_name",
    "checked_camera",
    "describe_target",
    "estimator_for",
    "ids_text",
    "markers_of_other_dictionaries",
    "other_dictionaries_clause",
    "parse_board_spec",
    "reprojection_rms_px",
    "rvec_tvec_to_matrix",
    "to_gray",
]

#: The dictionaries scanned when the configured one shows nothing, one per family, so ``why_not`` can say a
#: marker of another dictionary is in view. The largest of each NxN family holds the smaller ones' ids.
_OTHER_FAMILIES = (
    "DICT_4X4_1000", "DICT_5X5_1000", "DICT_6X6_1000", "DICT_7X7_1000", "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
    "DICT_ARUCO_MIP_36h12",
)

_BOARD_FORMS = ("aruco:ID:SIZE_MM[:DICT]", "charuco:XxY:SQUARE_MM:MARKER_MM[:DICT][:legacy]")


# ---------------------------------------------------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------------------------------------------------


def _dictionary_names() -> dict[str, str]:
    """``{UPPER-CASE name: the name as the config schema spells it}`` for every predefined dictionary.

    ``cv2.aruco`` carries the AprilTag and MIP names twice, ``DICT_APRILTAG_36h11`` and ``DICT_APRILTAG_36H11``;
    the mixed-case spelling is the one the schema accepts, so it wins.
    """
    names: dict[str, str] = {}
    for attr in dir(cv.aruco):
        if not attr.startswith("DICT_"):
            continue
        key = attr.upper()
        if key not in names or attr != key:
            names[key] = attr
    return names


def canonical_aruco_dict_name(name: str) -> str:
    """The dictionary ``name`` names, spelled as OpenCV and the config schema spell it.

    Case does not matter and the ``DICT_`` prefix is optional, so ``5x5_100`` is ``DICT_5X5_100`` and
    ``apriltag_36h11`` is ``DICT_APRILTAG_36h11``. An unknown name raises ``ValueError``.
    """
    key = str(name).strip().upper()
    if not key.startswith("DICT_"):
        key = "DICT_" + key
    found = _dictionary_names().get(key)
    if found is None:
        raise ValueError(f"unknown ArUco dictionary {name!r}")
    return found


def _dictionary(name: str) -> Any:
    try:
        canonical = canonical_aruco_dict_name(name)
    except ValueError as exc:
        raise StereoCalibrationError(f"Unknown ArUco dictionary: {name}") from exc
    return cv.aruco.getPredefinedDictionary(getattr(cv.aruco, canonical))


def aruco_dictionary_size(name: str) -> int:
    """How many marker ids the dictionary holds: ids run from 0 to this minus one."""
    return int(_dictionary(name).bytesList.shape[0])


_OTHER_DETECTORS: dict[str, Any] = {}


def markers_of_other_dictionaries(gray: np.ndarray, configured: str) -> dict[str, list[int]]:
    """``{dictionary: ids}`` for markers of other dictionary families in ``gray``; the configured one is skipped.

    A diagnostic for a frame in which the configured dictionary found nothing, measured at about 11 to 18 ms on a
    1280x720 frame for the ten families, so it runs once per missed pose and never on a pose that was found.
    """
    found: dict[str, list[int]] = {}
    for name in _OTHER_FAMILIES:
        if name == configured:
            continue
        detector = _OTHER_DETECTORS.get(name)
        if detector is None:
            detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(getattr(cv.aruco, name)))
            _OTHER_DETECTORS[name] = detector
        _, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids):
            found[name] = sorted({int(i) for i in np.asarray(ids).ravel()})
    return found


def ids_text(ids: Any, limit: int = 12) -> str:
    """A list of marker ids for a sentence, cut after ``limit`` with the count of the rest."""
    listed = sorted({int(i) for i in ids})
    if len(listed) <= limit:
        return str(listed)
    return f"[{', '.join(str(i) for i in listed[:limit])}, ... {len(listed) - limit} more]"


def other_dictionaries_clause(gray: np.ndarray, configured: str) -> str:
    """``""``, or a clause naming the dictionaries whose markers are in view, the most markers first.

    A family scan can also decode a stray pattern as a marker of an unrelated dictionary, so this names at most
    three families, and the one with the most markers is the likely print.
    """
    others = markers_of_other_dictionaries(gray, configured)
    if not others:
        return ""
    ranked = sorted(others.items(), key=lambda item: -len(item[1]))[:3]
    listed = ", ".join(f"{name} ids {ids_text(ids)}" for name, ids in ranked)
    return f"; markers of other dictionaries are in view ({listed}): check aruco_dict_name"


# ---------------------------------------------------------------------------------------------------------------------
# One frame
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Observation:
    """What one frame showed of a calibration target.

    ``T_cam_to_target`` is the 4x4 target pose in millimetres, or ``None``, and ``why_not`` then says why in one
    sentence. ``hint`` is a warning that did not stop the pose, such as several markers of the dictionary in view
    of a single-marker target. ``marker_ids`` and ``marker_corners`` are every marker of the configured dictionary
    detected, in the same order (each corner array ``(4, 2)`` pixels). ``points_px`` are the image points the pose
    was solved from, the marker's four corners or the board's chessboard corners, and ``point_ids`` the board's
    corner ids (empty for a marker). ``reprojection_px`` is their RMS reprojection error, and ``rvec``/``tvec`` the
    solved pose as OpenCV gives it, for drawing its axes. ``frame_shape`` is the shape of the frame judged. Nothing
    here holds the frame itself.
    """

    kind: str
    T_cam_to_target: np.ndarray | None = None
    why_not: str = ""
    hint: str = ""
    marker_ids: tuple[int, ...] = ()
    marker_corners: tuple[np.ndarray, ...] = ()
    points_px: np.ndarray | None = None
    point_ids: tuple[int, ...] = ()
    reprojection_px: float | None = None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    frame_shape: tuple[int, ...] = ()

    @property
    def found(self) -> bool:
        return self.T_cam_to_target is not None

    @property
    def n_points(self) -> int:
        return 0 if self.points_px is None else int(len(self.points_px))

    @property
    def distance_mm(self) -> float | None:
        """How far the target's origin is from the camera, in millimetres; ``None`` when it was not posed."""
        if self.T_cam_to_target is None:
            return None
        return float(np.linalg.norm(self.T_cam_to_target[:3, 3]))

    def summary(self) -> str:
        """What the pose rests on, as the sweep prints it: ``24 corners, 0.31 px, 520 mm away``."""
        if not self.found:
            return self.why_not
        parts = [f"{self.n_points} corners"]
        if self.reprojection_px is not None:
            parts.append(f"{self.reprojection_px:.2f} px")
        distance = self.distance_mm
        if distance is not None:
            parts.append(f"{distance:.0f} mm away")
        return ", ".join(parts)


def checked_camera(K: Any, dist: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(K, dist)`` as float arrays, refusing a camera matrix no pose can be solved with."""
    if K is None:
        raise StereoCalibrationError("K_rect is not available (set rectified intrinsics).")
    matrix = np.asarray(K, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise StereoCalibrationError("K_rect must be a finite 3x3 matrix")
    coefficients = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
    return matrix, coefficients


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """A grayscale view of a grayscale or BGR frame."""
    frame = np.asarray(bgr)
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise StereoCalibrationError(f"rect_left_bgr must be grayscale or BGR, got shape {frame.shape}")
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def reprojection_rms_px(obj: np.ndarray, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                        K: np.ndarray, dist: np.ndarray) -> float:
    projected, _ = cv.projectPoints(np.asarray(obj, dtype=np.float64).reshape(-1, 3), rvec, tvec, K, dist)
    residual = projected.reshape(-1, 2) - np.asarray(img, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


# ---------------------------------------------------------------------------------------------------------------------
# ChArUco
# ---------------------------------------------------------------------------------------------------------------------


class CharucoPoseEstimator:
    """A ChArUco board posed from its interpolated chessboard corners, ``T_cam_to_board`` in millimetres.

    The board frame is OpenCV's: its origin at the board's outer corner, x along ``squares_x``, z out of the
    printed face. ``marker_length_mm`` is the edge of the ArUco markers inside the white squares, not the square.
    ``legacy_pattern`` selects the layout OpenCV generated before 4.6, which differs for an even ``squares_y``:
    a board printed in one layout and read in the other shows its markers and interpolates no corners.
    A pose needs at least ``min_corners`` corners, not all on one line.
    """

    kind = "charuco"

    def __init__(
        self,
        *,
        squares_x: int,
        squares_y: int,
        square_length_mm: float,
        marker_length_mm: float,
        dict_name: str = "DICT_5X5_100",
        legacy_pattern: bool = False,
        min_corners: int = 6,
    ) -> None:
        if int(squares_x) < 2 or int(squares_y) < 2:
            raise StereoCalibrationError("squares_x and squares_y must be at least 2")
        if not 0.0 < float(marker_length_mm) < float(square_length_mm):
            raise StereoCalibrationError("require 0 < marker_length_mm < square_length_mm")
        if int(min_corners) < 4:
            raise StereoCalibrationError("min_corners must be at least 4, the fewest points a pose is solved from")
        dictionary = _dictionary(dict_name)
        self.squares = (int(squares_x), int(squares_y))
        self.square_length_mm = float(square_length_mm)
        self.marker_length_mm = float(marker_length_mm)
        self.dict_name = canonical_aruco_dict_name(dict_name)
        self.legacy_pattern = bool(legacy_pattern)
        self.min_corners = int(min_corners)
        markers = (self.squares[0] * self.squares[1]) // 2
        if markers > int(dictionary.bytesList.shape[0]):
            raise StereoCalibrationError(
                f"a {self.squares[0]}x{self.squares[1]} board carries {markers} markers and {self.dict_name} "
                f"holds {int(dictionary.bytesList.shape[0])}")
        # Typed Any: the cv2 stubs type matchImagePoints for a list of corners, and the detector returns an array.
        self._board: Any = cv.aruco.CharucoBoard(self.squares, self.square_length_mm, self.marker_length_mm,
                                                 dictionary)
        if self.legacy_pattern:
            self._board.setLegacyPattern(True)
        self._board_ids = frozenset(int(i) for i in np.asarray(self._board.getIds()).ravel())
        self._detector = cv.aruco.CharucoDetector(self._board)

    @property
    def board(self) -> Any:
        """The ``cv2.aruco.CharucoBoard``, for drawing."""
        return self._board

    def observe(self, bgr: np.ndarray, K: Any, dist: Any) -> Observation:
        """Detect the board in one frame and pose it, or say why not."""
        matrix, coefficients = checked_camera(K, dist)
        shape = tuple(np.asarray(bgr).shape)
        try:
            gray = to_gray(bgr)
            corners, corner_ids, marker_corners, marker_ids = self._detector.detectBoard(gray)
        except cv.error as exc:
            raise StereoCalibrationError(f"ChArUco detection failed: {exc}") from exc
        ids = () if marker_ids is None else tuple(int(i) for i in np.asarray(marker_ids).ravel())
        quads = () if marker_corners is None else tuple(np.asarray(c, dtype=np.float64).reshape(4, 2)
                                                        for c in marker_corners)
        seen: dict[str, Any] = dict(kind=self.kind, marker_ids=ids, marker_corners=quads, frame_shape=shape)
        foreign = sorted({i for i in ids if i not in self._board_ids})
        hint = (f"{self.dict_name} ids {ids_text(foreign)} are in view and are not on this "
                f"{self.squares[0]}x{self.squares[1]} board: check squares_x and squares_y" if foreign else "")
        if not ids:
            return Observation(**seen, why_not=f"no {self.dict_name} marker of the board in view"
                                              + other_dictionaries_clause(gray, self.dict_name))
        n_corners = 0 if corner_ids is None else int(len(corner_ids))
        if n_corners:
            seen["points_px"] = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            seen["point_ids"] = tuple(int(i) for i in np.asarray(corner_ids).ravel())
        if n_corners < self.min_corners:
            why = (f"{len(ids)} board markers seen, but {n_corners} chessboard corners interpolated "
                   f"(need {self.min_corners})")
            # MEASURED on a render: a 6x4 board read in the other layout showed all 12 markers and interpolated no
            # corner. A board partly out of view interpolates about as many corners as it shows markers.
            if len(ids) >= 2 and n_corners <= len(ids) // 2:
                why += ("; the target reads the board in the layout OpenCV generated before 4.6, and a board "
                        "generated by 4.6 or later needs legacy_pattern: false" if self.legacy_pattern else
                        "; a board generated before OpenCV 4.6 (or by a tool using its layout) needs "
                        "legacy_pattern: true")
            return Observation(**seen, why_not=why, hint=hint)
        obj, img = self._board.matchImagePoints(corners, corner_ids)
        obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
        img = np.asarray(img, dtype=np.float64).reshape(-1, 2)
        seen["points_px"] = img
        spread = np.linalg.svd(obj[:, :2] - obj[:, :2].mean(axis=0), compute_uv=False)
        if len(spread) < 2 or spread[1] < 0.5 * self.square_length_mm:
            return Observation(**seen, why_not=f"the {n_corners} corners seen lie along one line, and a pose "
                                               "needs corners in two directions", hint=hint)
        try:
            ok, rvec, tvec = cv.solvePnP(obj, img, matrix, coefficients)
        except cv.error as exc:
            raise StereoCalibrationError(f"solvePnP failed on {n_corners} ChArUco corners: {exc}") from exc
        if not ok:
            return Observation(**seen, why_not=f"solvePnP found no board pose from {n_corners} corners", hint=hint)
        return Observation(
            **seen, T_cam_to_target=rvec_tvec_to_matrix(rvec, tvec), hint=hint,
            reprojection_px=reprojection_rms_px(obj, img, rvec, tvec, matrix, coefficients),
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3), tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
        )


# ---------------------------------------------------------------------------------------------------------------------
# A target, read by attribute or by key
# ---------------------------------------------------------------------------------------------------------------------


def _field(target: Any, name: str, default: Any = None) -> Any:
    if isinstance(target, Mapping):
        return target.get(name, default)
    return getattr(target, name, default)


def estimator_for(target: Any) -> Any:
    """The estimator ``target`` describes: an ``ArucoPoseEstimator`` for one marker, a ``CharucoPoseEstimator``.

    ``target`` is a validated ``ArucoTargetConfig`` / ``CharucoTargetConfig`` or a mapping with the same keys; it is
    read by ``kind``. Build it from a validated config: the estimators refuse what OpenCV cannot use, not every value
    the schema refuses.
    """
    kind = _field(target, "kind")
    if kind == "aruco":
        from src.calibration.stereo.sub_modules.aruco_esti import ArucoPoseEstimator

        return ArucoPoseEstimator(marker_length_mm=float(_field(target, "marker_length_mm")),
                                  dict_name=str(_field(target, "aruco_dict_name", "DICT_5X5_100")),
                                  target_id=int(_field(target, "marker_id", 0)))
    if kind == "charuco":
        return CharucoPoseEstimator(
            squares_x=int(_field(target, "squares_x")), squares_y=int(_field(target, "squares_y")),
            square_length_mm=float(_field(target, "square_length_mm")),
            marker_length_mm=float(_field(target, "marker_length_mm")),
            dict_name=str(_field(target, "aruco_dict_name", "DICT_5X5_100")),
            legacy_pattern=bool(_field(target, "legacy_pattern", False)),
            min_corners=int(_field(target, "min_corners", 6)),
        )
    raise ValueError(f"unknown calibration target kind {kind!r}; the kinds are aruco and charuco")


def describe_target(target: Any) -> str:
    """The target in one line, as the config stage prints it.

    A marker: ``50.0 mm, id 0, DICT_5X5_100``. A board: ``charuco 7x5, square 30.0 mm, marker 22.0 mm,
    DICT_5X5_100``, with ``legacy pattern`` and the corner minimum after it.
    """
    kind = _field(target, "kind")
    dictionary = _field(target, "aruco_dict_name", "")
    if kind == "charuco":
        text = (f"charuco {_field(target, 'squares_x')}x{_field(target, 'squares_y')}, "
                f"square {float(_field(target, 'square_length_mm')):.1f} mm, "
                f"marker {float(_field(target, 'marker_length_mm')):.1f} mm, {dictionary}")
        if _field(target, "legacy_pattern", False):
            text += ", legacy pattern"
        return f"{text}, at least {_field(target, 'min_corners', 6)} corners"
    return f"{float(_field(target, 'marker_length_mm')):.1f} mm, id {_field(target, 'marker_id', 0)}, {dictionary}"


def parse_board_spec(text: str) -> dict[str, Any]:
    """The mapping a ``--board`` spec names, for the schema to validate.

    ``aruco:ID:SIZE_MM[:DICT]`` is one marker: its id and its printed edge. ``charuco:XxY:SQUARE_MM:MARKER_MM[:DICT]
    [:legacy]`` is a board of X by Y squares, its square and its marker edge, and ``legacy`` for the layout OpenCV
    generated before 4.6. A dictionary left out is not in the mapping, so the caller supplies its own default. The
    spelling is checked here; the values (a marker larger than its square, an id the dictionary does not hold) are
    the schema's to refuse. A spec that does not parse raises ``ValueError`` with one sentence.
    """
    spec = str(text).strip()
    parts = [part.strip() for part in spec.split(":")]
    kind = parts[0].lower()
    if kind == "aruco":
        if len(parts) not in (3, 4):
            raise ValueError(f"board {spec!r}: a marker is written {_BOARD_FORMS[0]}, such as aruco:0:50")
        out: dict[str, Any] = {"kind": "aruco", "marker_id": _integer(spec, "the marker id", parts[1]),
                               "marker_length_mm": _number(spec, "the marker edge", parts[2])}
        if len(parts) == 4 and parts[3]:
            out["aruco_dict_name"] = parts[3]
        return out
    if kind == "charuco":
        if not 4 <= len(parts) <= 6:
            raise ValueError(f"board {spec!r}: a board is written {_BOARD_FORMS[1]}, such as charuco:7x5:30:22")
        squares = parts[1].lower().split("x")
        if len(squares) != 2:
            raise ValueError(f"board {spec!r}: the squares are written XxY, such as 7x5; the form is "
                             f"{_BOARD_FORMS[1]}")
        out = {"kind": "charuco", "squares_x": _integer(spec, "squares_x", squares[0]),
               "squares_y": _integer(spec, "squares_y", squares[1]),
               "square_length_mm": _number(spec, "the square edge", parts[2]),
               "marker_length_mm": _number(spec, "the marker edge", parts[3])}
        tail = parts[4:]
        if tail and tail[-1].lower() == "legacy":
            out["legacy_pattern"] = True
            tail = tail[:-1]
        if len(tail) > 1 or (tail and tail[0].lower() == "legacy"):
            raise ValueError(f"board {spec!r}: after the marker edge come an optional dictionary and an optional "
                             f"'legacy'; the form is {_BOARD_FORMS[1]}")
        if tail and tail[0]:
            out["aruco_dict_name"] = tail[0]
        return out
    raise ValueError(f"board {spec!r}: the kind is aruco or charuco, written {_BOARD_FORMS[0]} or {_BOARD_FORMS[1]}")


def _integer(spec: str, what: str, text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"board {spec!r}: {what} is a whole number, not {text!r}") from None


def _number(spec: str, what: str, text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"board {spec!r}: {what} is a number of millimetres, not {text!r}") from None
