"""One camera body, described once.

A camera the arm carries can hit the cell unnoticed unless every collision model carries its housing. One YAML file
per camera model describes that housing, ``config/cameras/<model>.yaml``, and ``camera.cameras.rigs[<id>].body.model``
names it. The repository's registry is the one authority, as it is for grippers.

The vocabulary is the frame the eye in hand calibration solves in: the colour camera's optical frame, x to the right
in the image, y down, z out of the lens, in millimetres. The calibration gives CAMERA to TOOL in that frame, so a box
written in it is placed on the flange by the calibration and nothing else.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from .._base import StrictModel
from ..grippers import MODEL_NAME_PATTERN

__all__ = ["CameraBodySpec", "OpticalBox"]


class OpticalBox(StrictModel):
    """A box with its faces along the axes of the colour camera's optical frame, in millimetres.

    Every number is finite: an infinite size passes every bound a consumer checks and describes no box.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    #: The full extent along x, y and z of the optical frame, each above 0 and at most one metre.
    size_mm: tuple[float, float, float]
    #: Where the box's centre sits in the optical frame, each coordinate within one metre of the lens.
    centre_mm: tuple[float, float, float]

    @field_validator("size_mm")
    @classmethod
    def _sizes_are_a_box(cls, size: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(not 0.0 < value <= 1000.0 for value in size):
            raise ValueError(f"size_mm {list(size)} must hold three extents, each above 0 and at most 1000 mm")
        return size

    @field_validator("centre_mm")
    @classmethod
    def _centre_is_near_the_lens(cls, centre: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(abs(value) > 1000.0 for value in centre):
            raise ValueError(f"centre_mm {list(centre)} must lie within 1000 mm of the lens on every axis")
        return centre


class CameraBodySpec(StrictModel):
    """One registry file: which camera, where its numbers came from, and its housing."""

    model: str = Field(pattern=MODEL_NAME_PATTERN)
    #: Where the numbers came from: the vendor drawing and its revision, in words a person can check.
    source: str = Field(min_length=1)
    #: The optical frame the housing is written in. The colour camera's, because the eye in hand calibration solves
    #: the marker on the colour image, so CAMERA to TOOL is the colour camera's pose.
    optical_frame: Literal["color"]
    #: The housing, as the smallest box around the drawing's maximum extents, in the colour optical frame.
    housing: OpticalBox
