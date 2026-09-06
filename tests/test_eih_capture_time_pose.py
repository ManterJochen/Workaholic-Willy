"""An eye-in-hand frame resolves against where the tool was WHEN IT WAS CAPTURED.

⛔⛔ IT DID NOT. `EyeInHandFrameResolver.camera_to_base_for_frame` carried
`# noqa: ARG002 - frame not needed for the transform itself` and read `arm.get_tcp_pose()` live. For a
camera bolted to the wrist, CAMERA -> BASE depends on where the tool was when the SHUTTER opened, and
the closed-loop path moves the arm between capture and resolve BY DESIGN and then re-perceives. Every
millimetre travelled in between landed in the grasp, in a frame nothing downstream checks.

⚠ NEVER RUN ON HARDWARE. The stamp exists and the resolver reads it; no eye-in-hand cell has produced
a stamped frame yet. Bucket 3 until one does.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
from src.robot.grasping.types.perception import PerceptionFrame


class _Arm:
    """An arm that has MOVED since the frame was captured, which is the whole scenario."""

    def __init__(self, pose: Pose) -> None:
        self._pose = pose
        self.reads = 0

    def get_tcp_pose(self) -> Pose:
        self.reads += 1
        return self._pose


def _pose(x: float) -> Pose:
    return Pose(position_mm=np.asarray([x, 0.0, 300.0]),
                quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE)


def _frame(tool_pose: Pose | None) -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.zeros((4, 4)), intrinsics=np.eye(3), segmentations=(),
        tool_pose=tool_pose)


def _resolver() -> EyeInHandFrameResolver:
    return EyeInHandFrameResolver(
        t_cam_to_tool=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.TOOL))


class CaptureTimeTests(unittest.TestCase):

    def test_a_STAMPED_frame_resolves_against_its_own_pose(self) -> None:
        """⛔ THE DEFECT ITSELF. The arm has moved 120 mm since the shutter; the grasp must be
        computed where the picture was taken."""
        arm = _Arm(_pose(500.0))                      # where the tool is NOW
        transform = _resolver().camera_to_base_for_frame(_frame(_pose(380.0)), arm=arm)
        self.assertAlmostEqual(float(np.asarray(transform.translation_mm)[0]), 380.0, places=6)

    def test_a_stamped_frame_does_not_even_ASK_the_arm(self) -> None:
        """⚠ Not a performance point. An arm read that happens is an arm read that can disagree, and
        the whole repair is that this transform stops depending on the present."""
        arm = _Arm(_pose(500.0))
        _resolver().camera_to_base_for_frame(_frame(_pose(380.0)), arm=arm)
        self.assertEqual(arm.reads, 0)

    def test_an_UNSTAMPED_frame_behaves_exactly_as_before(self) -> None:
        """⚠ THE FALLBACK IS THE OLD BEHAVIOUR, so every producer that does not stamp is
        byte-identical. Stamping is what makes it correct; the fallback is what keeps it working."""
        arm = _Arm(_pose(500.0))
        transform = _resolver().camera_to_base_for_frame(_frame(None), arm=arm)
        self.assertAlmostEqual(float(np.asarray(transform.translation_mm)[0]), 500.0, places=6)
        self.assertEqual(arm.reads, 1)

    def test_the_two_paths_DISAGREE_when_the_arm_moved(self) -> None:
        """⭐ The control that makes the three tests above mean something: if a stamped and an
        unstamped frame resolved the same way, the stamp would be decoration."""
        stamped = _resolver().camera_to_base_for_frame(_frame(_pose(380.0)), arm=_Arm(_pose(500.0)))
        live = _resolver().camera_to_base_for_frame(_frame(None), arm=_Arm(_pose(500.0)))
        gap = abs(float(np.asarray(stamped.translation_mm)[0])
                  - float(np.asarray(live.translation_mm)[0]))
        self.assertAlmostEqual(gap, 120.0, places=6)

    def test_the_default_is_UNSTAMPED(self) -> None:
        """`None` means the producer did not stamp it, not "the tool was at the origin". Inventing a
        pose would turn a missing measurement into a wrong one."""
        frame = PerceptionFrame(depth_map=np.zeros((2, 2)), intrinsics=np.eye(3), segmentations=())
        self.assertIsNone(frame.tool_pose)

    def test_the_resolver_no_longer_declares_the_frame_UNUSED(self) -> None:
        """The `noqa: ARG002` was the defect written down: the argument that carried the answer was
        marked as not needed.

        ⚠ THE SUPPRESSION, NOT THE WORDS. The docstring that replaced it QUOTES the old comment, so a
        test looking for the phrase passes on the defect and fails on the repair. It has to read the
        signature. The two sibling resolvers keep their own `ARG002` and are right to: an eye-to-hand
        camera genuinely does not care which frame it is asked about.
        """
        import ast
        from pathlib import Path

        source = Path("src/robot/grasping/motion/frame_resolver.py").read_text(
            encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source)
        holder = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef) and node.name == "EyeInHandFrameResolver")
        method = next(node for node in holder.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "camera_to_base_for_frame")
        argument = next(a for a in method.args.args if a.arg == "frame")
        self.assertNotIn("ARG002", lines[argument.lineno - 1],
                         "the argument carrying the capture pose is still marked unused")


if __name__ == "__main__":
    unittest.main()
