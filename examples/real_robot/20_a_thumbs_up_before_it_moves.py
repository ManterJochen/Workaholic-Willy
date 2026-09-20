"""A person at the cell approves the grasp with a thumbs-up, and stops it with a thumbs-down.

15 asks at a terminal, which is the wrong place to stand when the arm is about to move. This asks
through the camera the cell already has: find the part, show what would be picked, then watch for a
gesture. Nothing moves until one arrives.

Four answers, not two, and the difference decides what to do. `THUMB_UP` is go. `THUMB_DOWN` is a
person saying no to THIS grasp. `OTHER` is a hand the classifier saw and did not read as either,
which is not consent. `NONE` is no hand at all. The last three all end here without motion, because
the only reading that may move an arm is the one that says so.

MediaPipe is standalone and needs its own `.task` file (`models.gesturedetect.model_path`), fetched
with `python scripts/model_weights/fetch.py --mediapipe`.

Run it at the cell, under the cell's profile, once its camera is calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/20_a_thumbs_up_before_it_moves.py
"""

import time

from willy import (Camera, HandGesture, Locator, RGBDFrame, Robot,
                   build_gesture_recognizer, load_tree)

tree = load_tree()
timeout_s = 20.0
# How sure the classifier has to be before the reading counts as an answer. Its own floor is in
# models.gesturedetect.min_gesture_confidence; this is the second one, for a motion decision.
confidence_floor = 0.7

with Camera.from_tree(tree) as camera:
    robot = Robot.from_tree(tree, cameras=[camera])
    locator = Locator.from_tree(tree, camera=camera, tool_pose=robot.arm.get_tcp_pose)

    with robot.connected(), build_gesture_recognizer(tree.app_config.models.gesturedetect) as reader:
        located = locator.locate("a red cube")
        print(located)
        best = located.scene(0, tree.robot).grasps().best if located.objects else None
        if best is None:
            raise SystemExit("nothing located for that prompt, or no grasp on what was")
        print(f"would pick at {[round(float(v)) for v in best.pose().position_mm]} mm, "
              f"{best.grip_width_mm:.0f} mm wide. Thumbs up to go, thumbs down to stop.")

        answer, deadline = HandGesture.NONE, time.monotonic() + timeout_s
        while answer in (HandGesture.NONE, HandGesture.OTHER) and time.monotonic() < deadline:
            frame = camera.grab()
            if not isinstance(frame, RGBDFrame):
                raise SystemExit(f"rig {camera.rig_id!r} is a stereo pair: no colour to read")
            # One hand, or none: two people answering at once is a reason to keep waiting rather
            # than to take the first reading. The gesture model returns the palm from the same
            # pass, so nothing is detected twice here.
            hands = reader.observe(frame.color)
            if len(hands) == 1 and hands[0].gesture.confidence >= confidence_floor:
                answer = hands[0].gesture.gesture
                print(f"  saw {answer.value} ({hands[0].gesture.confidence:.2f}), "
                      f"raw {hands[0].gesture.raw_label!r}")

        if answer is not HandGesture.THUMB_UP:
            # Every other reading, including the timeout, ends here. Silence is not consent.
            raise SystemExit(f"no thumbs-up ({answer.value}); nothing moved")

        print(robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0)))
