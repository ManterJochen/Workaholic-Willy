"""Iteration-2 regression tests: geometry frame-mismatch and serialization
schema errors derive from ``GeometryError`` (not a bare ``ValueError``), so a
single ``except GeometryError`` catches the whole subsystem as documented.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry.exceptions import (
    FrameMismatchError,
    GeometryError,
    InvalidPoseError,
    InvalidTransformError,
)
from src.geometry.frame import Frame
from src.geometry.pose import Pose
from src.geometry.quaternion import IDENTITY_QUAT_XYZW
from src.geometry.serialization import (
    pose_from_dict,
    pose_to_dict,
    transform_from_dict,
    transform_to_dict,
)
from src.geometry.transform import Transform


def _pose(frame: Frame) -> Pose:
    return Pose(
        position_mm=np.zeros(3, dtype=np.float64),
        quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
        frame=frame,
    )


class PoseFrameErrorTypeTests(unittest.TestCase):
    def test_distance_to_frame_mismatch_raises_frame_mismatch_error(self) -> None:
        p1, p2 = _pose(Frame.BASE), _pose(Frame.CAMERA)
        with self.assertRaises(FrameMismatchError):
            p1.distance_to(p2)
        # ...and it is catchable via the package-wide base (the documented contract).
        with self.assertRaises(GeometryError):
            p1.distance_to(p2)

    def test_angle_to_frame_mismatch_raises_frame_mismatch_error(self) -> None:
        p1, p2 = _pose(Frame.BASE), _pose(Frame.CAMERA)
        with self.assertRaises(FrameMismatchError):
            p1.angle_to(p2)
        with self.assertRaises(GeometryError):
            p1.angle_to(p2)

    def test_same_frame_still_computes(self) -> None:
        p1, p2 = _pose(Frame.BASE), _pose(Frame.BASE)
        self.assertEqual(p1.distance_to(p2), 0.0)
        self.assertEqual(p1.angle_to(p2), 0.0)


class SerializationSchemaErrorTypeTests(unittest.TestCase):
    def _transform(self) -> Transform:
        return Transform(
            translation_mm=np.zeros(3, dtype=np.float64),
            quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
            from_frame=Frame.CAMERA,
            to_frame=Frame.BASE,
        )

    def test_pose_from_dict_bad_schema_raises_invalid_pose_error(self) -> None:
        data = pose_to_dict(_pose(Frame.BASE))
        data["schema"] = "wrong/schema"
        with self.assertRaises(InvalidPoseError):
            pose_from_dict(data)
        with self.assertRaises(GeometryError):
            pose_from_dict(data)

    def test_transform_from_dict_bad_schema_raises_invalid_transform_error(self) -> None:
        data = transform_to_dict(self._transform())
        data["schema"] = "wrong/schema"
        with self.assertRaises(InvalidTransformError):
            transform_from_dict(data)
        with self.assertRaises(GeometryError):
            transform_from_dict(data)

    def test_round_trips_still_work(self) -> None:
        self.assertEqual(pose_from_dict(pose_to_dict(_pose(Frame.BASE))).frame, Frame.BASE)
        self.assertEqual(
            transform_from_dict(transform_to_dict(self._transform())).from_frame,
            Frame.CAMERA,
        )

    # --- L0.1: missing keys / bad frame values raise GeometryError, not bare KeyError/ValueError ---
    def test_pose_from_dict_missing_key_raises_invalid_pose_error(self) -> None:
        data = pose_to_dict(_pose(Frame.BASE))
        del data["position_mm"]
        with self.assertRaises(InvalidPoseError):
            pose_from_dict(data)

    def test_pose_from_dict_bad_frame_value_raises_invalid_pose_error(self) -> None:
        data = pose_to_dict(_pose(Frame.BASE))
        data["frame"] = "not_a_frame"
        with self.assertRaises(InvalidPoseError):
            pose_from_dict(data)

    def test_transform_from_dict_missing_key_raises_invalid_transform_error(self) -> None:
        data = transform_to_dict(self._transform())
        del data["translation_mm"]
        with self.assertRaises(InvalidTransformError):
            transform_from_dict(data)

    def test_transform_from_dict_bad_frame_value_raises_invalid_transform_error(self) -> None:
        data = transform_to_dict(self._transform())
        data["to_frame"] = "not_a_frame"
        with self.assertRaises(InvalidTransformError):
            transform_from_dict(data)


if __name__ == "__main__":
    unittest.main()
