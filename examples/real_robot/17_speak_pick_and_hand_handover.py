"""Speak what to pick, locate and pick it by camera, then wait for your open hand before releasing.

Combines three pieces that already exist as their own examples: speech (15), camera-grounded pick
(11), and a MediaPipe hand detector used as the hand-over signal instead of the TCP wrench (16 uses
force; this uses vision). MediaPipe is optional and standalone (`src/models/handdetection`): nothing
in the grasp pipeline imports it, and it needs its own `.task` model file, fetched separately (see
that package's `model_files.py`).

Run it at the cell, under the cell's profile, once its camera is calibrated (07-10) and the
hand-landmark model is downloaded (models.handdetect.model_path):
    WILLY_PROFILE=<your cell> python examples/real_robot/17_speak_pick_and_hand_handover.py
"""

import threading
import time

from willy import (Camera, Confirmation, Locator, Pose, PushToTalkSource, Robot, TalkButton,
                   TerminalConfirmer, load_speech_section, load_tree, shared_speech)
from src.models.handdetection import build_palm_detector

tree = load_tree()
speech = load_speech_section()  # models.stt of the same profile; needs no camera or robot

# --- Speak what to pick, the person confirms -----------------------------------------------------
button = TalkButton.from_parts()


def release_on_enter() -> None:
    input()
    button.release()


with PushToTalkSource.from_config(config=speech, switch=button) as microphone:
    input("Press Enter, say what to pick, then press Enter again. ")
    button.press()
    threading.Thread(target=release_on_enter, daemon=True).start()
    turn = microphone.record(timeout_s=1.0)
print(turn)
if not turn.ok:
    raise SystemExit(1)

proposal = shared_speech().for_config(config=speech).propose(turn.samples, samplerate=turn.samplerate)
print(proposal)

confirmation = Confirmation.from_proposal(proposal=proposal, confirmer=TerminalConfirmer.from_parts())
print(confirmation)
if confirmation.confirmed is None:
    raise SystemExit(1)

# --- Locate what was confirmed, and pick it, camera-grounded -------------------------------------
handover = Pose.tool_down(300.0, 0.0, 400.0)

with Camera.from_tree(tree) as camera:
    robot = Robot.from_tree(tree, cameras=[camera])
    print(robot)
    locator = Locator.from_tree(tree, camera=camera, tool_pose=robot.arm.get_tcp_pose)

    # The same hand-landmark model the locator's camera feeds; no second device is opened.
    hand_detector = build_palm_detector(tree.app_config.models.handdetect)
    hand_present_threshold = 1  # at least one hand in frame counts as "reach for it"
    poll_interval_s = 0.1
    timeout_s = 30.0

    with robot.connected():
        located = locator.locate(confirmation.confirmed)
        print(located)
        if not located.objects:
            raise SystemExit("nothing located for that prompt")

        grasps = located.scene(0, tree.robot).grasps()
        print(grasps)
        best = grasps.best
        if best is None:
            raise SystemExit("no grasp found for the located object")

        picked = robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0))
        print(picked)
        if not picked.ok:
            raise SystemExit("pick failed, nothing to hand over")

        print(robot.move(handover))

        print("holding the part out; show your open hand to the camera")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = camera.grab()
            hands = hand_detector.detect(frame.color)
            if len(hands) >= hand_present_threshold:
                break
            time.sleep(poll_interval_s)
        else:
            print(f"no hand seen within {timeout_s} s; releasing anyway")

        print(robot.release())
