"""The camera registry schema: one validated body per camera model.

The files live in ``config/cameras/`` and :mod:`src.config.cameras` loads them.
"""

from .camera_schema import CameraBodySpec, OpticalBox

__all__ = ["CameraBodySpec", "OpticalBox"]
