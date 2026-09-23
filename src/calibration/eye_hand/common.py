"""Implementation shared by the two eye-hand calibrators.

Sample acceptance, the readiness checks a solve needs and the AX=XB call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.geometry.matrix import invert_homogeneous

from src.calibration.constants import CALIBRATION_LOG_DIR, EYE_HAND_CALIBRATOR_LOG_FILE
from src.calibration.exceptions import CalibrationDataError, CalibrationSolveError
from src.calibration.quality import DEFAULT_BANDS_MM, QualityBandsMm
from src.calibration.solver import HandEyeAXXB
from src.utility.log_cfg import create_logger

from .dataset import EyeHandDataset, EyeHandSample
from .types import EyeHandCalibrationSettings

__all__ = ["BaseEyeHandCalibrator", "SampleRejection"]

logger = create_logger(
    "EyeHandCalibrator", EYE_HAND_CALIBRATOR_LOG_FILE, log_dir=CALIBRATION_LOG_DIR
)


def _rotation_angle_deg(T_first: np.ndarray, T_second: np.ndarray) -> float:
    rotation_delta = T_first[:3, :3].T @ T_second[:3, :3]
    cosine = float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _poses_are_diverse(
    T_first: np.ndarray,
    T_second: np.ndarray,
    *,
    min_distance_mm: float,
    min_angle_deg: float,
) -> bool:
    distance = float(np.linalg.norm(T_first[:3, 3] - T_second[:3, 3]))
    angle = _rotation_angle_deg(T_first, T_second)
    return distance > min_distance_mm or angle > min_angle_deg


@dataclass(frozen=True, slots=True)
class SampleRejection:
    """Why :meth:`BaseEyeHandCalibrator.add_sample` returned ``False``, with the numbers it decided on.

    ``reason`` is ``"no_marker"`` or ``"not_diverse"``. For ``"not_diverse"`` the nearest stored sample
    among those the new pose is too close to is named by its 1-based position in the dataset, with its
    distance in millimetres and its rotation in degrees, beside the two thresholds a new pose must beat
    one of. Those fields are ``None`` for ``"no_marker"``.
    """

    reason: str
    stored: int = 0
    nearest_sample: int | None = None
    nearest_mm: float | None = None
    nearest_deg: float | None = None
    min_distance_mm: float | None = None
    min_angle_deg: float | None = None

    def render(self) -> str:
        """One sentence, ASCII."""
        if self.reason != "not_diverse":
            return "no marker pose was detected"
        return (
            f"pose not diverse: stored sample {self.nearest_sample} is {self.nearest_mm:.1f} mm and "
            f"{self.nearest_deg:.1f} deg away, and a new pose needs > {self.min_distance_mm:.1f} mm or "
            f"> {self.min_angle_deg:.1f} deg from every one of the {self.stored} stored"
        )

    def to_dict(self) -> dict[str, object]:
        """Plain data, ``json.dumps`` safe."""
        return {
            "reason": self.reason,
            "stored": self.stored,
            "nearest_sample": self.nearest_sample,
            "nearest_mm": self.nearest_mm,
            "nearest_deg": self.nearest_deg,
            "min_distance_mm": self.min_distance_mm,
            "min_angle_deg": self.min_angle_deg,
        }


def _rotation_axis_from_relative(relative_transform: np.ndarray) -> np.ndarray | None:
    rotation = relative_transform[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < math.radians(1.0):
        return None
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        return None
    return axis / norm


class BaseEyeHandCalibrator:
    """Sample collection and solving shared by the two hand-eye workflows.

    Subclasses build the A and B motion pairs their mounting requires and tag
    the solved transform with its frames.
    """

    #: Why the last :meth:`add_sample` returned ``False``; ``None`` after one that stored its sample. A class
    #: default, so a calibrator built without ``__init__`` reads ``None`` too.
    last_rejection: SampleRejection | None = None

    def __init__(
        self,
        *,
        settings: EyeHandCalibrationSettings | object | None = None,
        dataset: EyeHandDataset | None = None,
        solver: HandEyeAXXB | None = None,
        bands: QualityBandsMm = DEFAULT_BANDS_MM,
    ) -> None:
        if settings is None:
            self.settings = EyeHandCalibrationSettings()
        elif isinstance(settings, EyeHandCalibrationSettings):
            self.settings = settings
        else:
            self.settings = EyeHandCalibrationSettings.from_config(settings)
        self.dataset = dataset if dataset is not None else EyeHandDataset()
        self.solver = solver if solver is not None else HandEyeAXXB()
        self.bands = bands

    def add_sample(
        self,
        T_base_to_tool: np.ndarray,
        T_cam_to_marker: np.ndarray | None,
        marker_id: int = 0,
    ) -> bool:
        """Append a sample, returning False when it cannot be used.

        A sample is refused when no marker pose was detected, or when the robot
        pose is not more than ``min_distance_mm`` or ``min_angle_deg`` away from
        every pose already stored. Both refusals are logged, and
        :attr:`last_rejection` says which, with the nearest stored sample's
        distance and angle for a pose that was not diverse.
        """
        if T_cam_to_marker is None:
            # The caller only gets `False` back, so this line and `last_rejection` are the
            # record of why the sample was dropped.
            logger.info("Sample skipped: no marker pose (marker_id=%d).", marker_id)
            self.last_rejection = SampleRejection(reason="no_marker", stored=len(self.dataset))
            return False
        sample = EyeHandSample(
            T_base_to_tool=T_base_to_tool,
            T_cam_to_marker=T_cam_to_marker,
            marker_id=marker_id,
        )
        rejection = self._diversity_rejection(sample.T_base_to_tool)
        if rejection is not None:
            logger.info("Sample rejected: %s.", rejection.render())
            self.last_rejection = rejection
            return False
        self.last_rejection = None
        self.dataset.add_sample(sample)
        logger.debug(
            "Sample accepted (marker_id=%d, dataset now %d samples).",
            marker_id,
            len(self.dataset),
        )
        return True

    def _diversity_rejection(self, T_base_to_tool: np.ndarray) -> SampleRejection | None:
        """``None`` when the pose beats a threshold against every stored sample, else the nearest blocking one.

        A stored sample blocks the new pose when it is within ``min_distance_mm`` and within
        ``min_angle_deg`` of it. Of the blocking samples, the one nearest in translation is reported.
        """
        min_mm = float(self.settings.min_distance_mm)
        min_deg = float(self.settings.min_angle_deg)
        nearest: tuple[float, float, int] | None = None
        for index, existing in enumerate(self.dataset.iter_samples(), start=1):
            if _poses_are_diverse(existing.T_base_to_tool, T_base_to_tool,
                                  min_distance_mm=min_mm, min_angle_deg=min_deg):
                continue
            distance = float(np.linalg.norm(existing.T_base_to_tool[:3, 3] - T_base_to_tool[:3, 3]))
            angle = _rotation_angle_deg(existing.T_base_to_tool, T_base_to_tool)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, angle, index)
        if nearest is None:
            return None
        return SampleRejection(
            reason="not_diverse", stored=len(self.dataset), nearest_sample=nearest[2],
            nearest_mm=nearest[0], nearest_deg=nearest[1], min_distance_mm=min_mm, min_angle_deg=min_deg,
        )

    def save_dataset(self, path: str | Path) -> Path:
        return self.dataset.save(path)

    def load_dataset(self, path: str | Path) -> int:
        self.dataset = EyeHandDataset.load(path)
        return len(self.dataset)

    def _sample_matrices(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self.dataset.base_to_tool_matrices(), self.dataset.cam_to_marker_matrices()

    def _assert_ready(self) -> None:
        if len(self.dataset) < self.settings.min_samples:
            raise CalibrationDataError(
                f"too few samples: got {len(self.dataset)}, need >= {self.settings.min_samples}"
            )
        T_base_to_tool_list = self.dataset.base_to_tool_matrices()
        relative_robot_motions = HandEyeAXXB.relative_motions(T_base_to_tool_list)
        axes = [
            axis
            for axis in (
                _rotation_axis_from_relative(relative)
                for relative in relative_robot_motions
            )
            if axis is not None
        ]
        if len(axes) < 2:
            raise CalibrationDataError(
                "sample set needs at least two meaningful robot rotation motions"
            )
        axis_rank = int(np.linalg.matrix_rank(np.vstack(axes), tol=0.1))
        if axis_rank < 2:
            raise CalibrationDataError(
                "sample set needs robot rotations around at least two independent axes"
            )

    def _solve_axxb(
        self,
        A_mats: list[np.ndarray],
        B_mats: list[np.ndarray],
    ) -> tuple[np.ndarray, float, float]:
        try:
            transform_matrix, rmse = self.solver.solve(A_mats, B_mats)
        except CalibrationDataError:
            raise
        except Exception as exc:
            raise CalibrationSolveError("AX=XB solve failed") from exc
        residuals = [
            float(np.linalg.norm(A @ transform_matrix - transform_matrix @ B, "fro"))
            for A, B in zip(A_mats, B_mats)
        ]
        max_error = max(residuals) if residuals else float(rmse)
        logger.info(
            "AX=XB solved over %d motion pairs from %d samples: rmse=%.3f mm, max=%.3f mm.",
            len(A_mats),
            len(self.dataset),
            float(rmse),
            float(max_error),
        )
        return transform_matrix, float(rmse), float(max_error)

    @staticmethod
    def _inverse(matrix: np.ndarray) -> np.ndarray:
        return invert_homogeneous(matrix)