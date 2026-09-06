"""R10.1 — behavioral conformance of every FrameResolver implementation to its typed contract.

The :class:`FrameResolver` Protocol (``backend/src/robot/grasping/frame_resolver.py``) closes the
Phase S2 frame-contract gap: it resolves ``T_cam_to_base`` for a captured frame so the orchestrator
never silently hands a camera-frame pose to the execution policy. The documented contract every
implementation must honour:

* :meth:`camera_to_base_for_frame` returns a :class:`Transform` (never ``None`` / a bare matrix);
* that transform is tagged ``from_frame is Frame.CAMERA`` and ``to_frame is Frame.BASE`` — the
  load-bearing invariant. ``resolve_or_none`` treats ANY other frame pair as
  ``RESOLVE_REASON_WRONG_FRAME``, so a resolver returning e.g. ``BASE -> CAMERA`` would route a
  camera-frame pose straight into a base-frame motion command (the exact safety hole this module
  exists to close);
* the resolver MUST NOT mutate the arm (no motion / state side effects);
* it MUST be callable repeatedly per attempt (initial capture, S3 recapture, active perception)
  and stay consistent across calls.

The suite parametrizes over all three production implementers built with the *exact* construction
idioms from ``tests/test_frame_resolver.py`` (same ``_StaticTcpArm`` double, same valid
``CAMERA -> BASE`` / ``CAMERA -> TOOL`` calibration matrices). The eye-in-hand resolver supplies the
two-path anchor: it is the only one that consults the live TCP, so moving the arm MUST change its
output, while the eye-to-hand resolvers (Static / Identity) MUST stay invariant — proving the suite
exercises real divergent output, not an all-accept shape.

``resolve_or_none``'s frame-pair check (the production consumer of this contract) is also asserted
to AGREE with every resolver, anchoring the wire contract end-to-end.
"""

from __future__ import annotations

import unittest
from typing import cast

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.core import (
    MotionCommand,
    MotionResult,
    RobotArm,
    RobotCapabilities,
)
from src.robot.grasping import (
    EyeInHandFrameResolver,
    FrameResolver,
    IdentityFrameResolver,
    StaticCameraToBaseResolver,
)
from src.robot.grasping.motion.frame_resolver import (
    FrameResolutionFailure,
    resolve_or_none,
)
from src.robot.grasping.types.perception import PerceptionFrame

# ---------------------------------------------------------------------------
# Test doubles — copied verbatim from tests/test_frame_resolver.py so the
# conformance suite exercises the same construction idioms the unit tests use.
# ---------------------------------------------------------------------------

_REAL_UR_CAPS = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=False,
)


class _StaticTcpArm:
    """Fake arm reporting a fixed TCP pose. Records any motion side effects."""

    def __init__(self, tcp: Pose) -> None:
        self._tcp = tcp
        self.move_calls: list[Pose] = []

    @property
    def capabilities(self) -> RobotCapabilities:
        return _REAL_UR_CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.move_calls.append(pose)
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


def _as_arm(arm: _StaticTcpArm) -> RobotArm:
    """Present the minimal TCP double at the RobotArm seam.

    The resolver contract only touches ``get_tcp_pose`` (and eye-to-hand resolvers ignore the arm
    entirely), so a full driver is needless. This single ``cast`` keeps the test bodies free of
    per-call ``# type: ignore`` while leaving ``.move_calls`` / ``._tcp`` reachable on the concrete
    variable for the no-motion / live-TCP assertions. (The same ``_StaticTcpArm`` double is wired into
    orchestrators with ``# type: ignore[arg-type]`` in ``tests/test_frame_resolver.py``.)
    """

    return cast(RobotArm, arm)


def _base_tcp(x_mm: float = 200.0, y_mm: float = 100.0, z_mm: float = 500.0) -> Pose:
    return Pose(
        position_mm=np.array([x_mm, y_mm, z_mm], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE,
        label="tcp",
    )


def _perception_frame() -> PerceptionFrame:
    depth = np.full((16, 16), 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, 8.0], [0.0, 400.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(depth_map=depth, intrinsics=intrinsics, segmentations=())


def _static_cam_to_base() -> Transform:
    return Transform.from_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, -50.0],
                [0.0, 0.0, 1.0, 800.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        from_frame=Frame.CAMERA,
        to_frame=Frame.BASE,
    )


def _cam_to_tool() -> Transform:
    # Camera 30 mm "above" the TCP, identity rotation.
    return Transform.from_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        from_frame=Frame.CAMERA,
        to_frame=Frame.TOOL,
    )


def _production_resolvers() -> tuple[FrameResolver, ...]:
    """Every shipped FrameResolver, built the way the unit tests build them."""
    return (
        StaticCameraToBaseResolver(transform=_static_cam_to_base()),
        EyeInHandFrameResolver(t_cam_to_tool=_cam_to_tool()),
        IdentityFrameResolver(),
    )


class FrameResolverConformanceTests(unittest.TestCase):
    """Every production FrameResolver honours the typed ``CAMERA -> BASE`` contract."""

    def setUp(self) -> None:
        self._resolvers = _production_resolvers()
        # All three shipped implementers are present (guards against one being dropped).
        self.assertEqual(len(self._resolvers), 3)

    def test_every_resolver_satisfies_runtime_protocol(self) -> None:
        for r in self._resolvers:
            with self.subTest(resolver=type(r).__name__):
                self.assertIsInstance(r, FrameResolver)

    def test_returns_camera_to_base_transform_without_touching_arm(self) -> None:
        # THE contract: returns a Transform tagged CAMERA -> BASE, and never raises / never moves the
        # arm on a well-formed call. A resolver returning a bare matrix, None, or a BASE -> CAMERA
        # transform — or one that commanded motion — would fail here.
        for r in self._resolvers:
            with self.subTest(resolver=type(r).__name__):
                arm = _StaticTcpArm(_base_tcp())
                try:
                    resolved = r.camera_to_base_for_frame(
                        _perception_frame(), arm=_as_arm(arm)
                    )
                except Exception as exc:  # noqa: BLE001 - the point is to prove no resolver raises
                    self.fail(
                        f"{type(r).__name__} raised {type(exc).__name__} on a well-formed call "
                        f"instead of returning a Transform: {exc!r}"
                    )
                self.assertIsInstance(resolved, Transform)
                self.assertIs(resolved.from_frame, Frame.CAMERA)
                self.assertIs(resolved.to_frame, Frame.BASE)
                # MUST NOT mutate the arm.
                self.assertEqual(arm.move_calls, [])

    def test_repeatable_per_attempt_and_consistent(self) -> None:
        # The Protocol promises repeated invocation per attempt. With a fixed arm pose, every resolver
        # must return the same translation on back-to-back calls (consistency), still CAMERA -> BASE.
        for r in self._resolvers:
            with self.subTest(resolver=type(r).__name__):
                arm = _StaticTcpArm(_base_tcp())
                first = r.camera_to_base_for_frame(_perception_frame(), arm=_as_arm(arm))
                second = r.camera_to_base_for_frame(
                    _perception_frame(), arm=_as_arm(arm)
                )
                self.assertIs(second.from_frame, Frame.CAMERA)
                self.assertIs(second.to_frame, Frame.BASE)
                np.testing.assert_allclose(first.translation_mm, second.translation_mm)
                self.assertEqual(arm.move_calls, [])

    def test_resolve_or_none_agrees_with_every_resolver(self) -> None:
        # The production non-raising consumer (resolve_or_none) enforces the same frame-pair contract.
        # It must hand back the SAME Transform each resolver produced — never a FrameResolutionFailure —
        # anchoring the wire contract end-to-end.
        for r in self._resolvers:
            with self.subTest(resolver=type(r).__name__):
                arm = _StaticTcpArm(_base_tcp())
                frame = _perception_frame()
                direct = r.camera_to_base_for_frame(frame, arm=_as_arm(arm))
                via_helper = resolve_or_none(r, frame, arm=_as_arm(arm))
                self.assertNotIsInstance(via_helper, FrameResolutionFailure)
                assert isinstance(via_helper, Transform)  # narrow for the type checker
                self.assertIs(via_helper.from_frame, Frame.CAMERA)
                self.assertIs(via_helper.to_frame, Frame.BASE)
                np.testing.assert_allclose(
                    via_helper.translation_mm, direct.translation_mm
                )

    def test_eye_in_hand_tracks_live_tcp_while_eye_to_hand_is_invariant(self) -> None:
        # Two-path non-vacuity anchor exercising the ACTUAL numeric output:
        #   * EyeInHandFrameResolver consults the live TCP, so moving the arm MUST shift its
        #     translation by exactly the TCP delta (and the value must be the documented composition).
        #   * The eye-to-hand resolvers (Static / Identity) ignore the arm, so the SAME move MUST
        #     leave their output unchanged.
        # An implementation that returned a constant, or that read/ignored the arm wrongly, breaks one
        # of the two branches — so this cannot be satisfied vacuously.
        frame = _perception_frame()

        # Eye-in-hand: identity rotation makes the composition pure addition — camera origin maps to
        # TCP position + (0, 0, -30). Verify the exact value, then verify it tracks a TCP move.
        eih = EyeInHandFrameResolver(t_cam_to_tool=_cam_to_tool())
        arm = _StaticTcpArm(_base_tcp(200.0, 100.0, 500.0))
        before = eih.camera_to_base_for_frame(frame, arm=_as_arm(arm))
        np.testing.assert_allclose(
            before.translation_mm, np.array([200.0, 100.0, 470.0], dtype=np.float64)
        )
        arm._tcp = _base_tcp(400.0, 100.0, 500.0)
        after = eih.camera_to_base_for_frame(frame, arm=_as_arm(arm))
        np.testing.assert_allclose(
            after.translation_mm, np.array([400.0, 100.0, 470.0], dtype=np.float64)
        )
        self.assertFalse(np.allclose(before.translation_mm, after.translation_mm))

        # Eye-to-hand: identical TCP move must NOT change the resolved transform.
        for r in (
            StaticCameraToBaseResolver(transform=_static_cam_to_base()),
            IdentityFrameResolver(),
        ):
            with self.subTest(resolver=type(r).__name__):
                e2h_arm = _StaticTcpArm(_base_tcp(200.0, 100.0, 500.0))
                t0 = r.camera_to_base_for_frame(frame, arm=_as_arm(e2h_arm))
                e2h_arm._tcp = _base_tcp(400.0, 100.0, 500.0)
                t1 = r.camera_to_base_for_frame(frame, arm=_as_arm(e2h_arm))
                np.testing.assert_allclose(t0.translation_mm, t1.translation_mm)

    def test_eye_in_hand_fails_closed_on_non_base_tcp(self) -> None:
        # Fail-closed anchor: the eye-in-hand resolver REFUSES a tool-frame TCP read (the common driver
        # footgun) with a ValueError rather than silently composing a wrong transform.
        eih = EyeInHandFrameResolver(t_cam_to_tool=_cam_to_tool())
        bad_tcp = Pose(
            position_mm=np.zeros(3, dtype=np.float64),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.TOOL,
        )
        arm = _StaticTcpArm(bad_tcp)
        with self.assertRaises(ValueError):
            eih.camera_to_base_for_frame(_perception_frame(), arm=_as_arm(arm))
        # Refusing is still no-motion: the arm was never commanded.
        self.assertEqual(arm.move_calls, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
