"""Speak what to pick, pick it by camera, then bring it to where the person's hand actually is.

Speech is 12's; the pick is the camera's, as in 11, taken one step at a time here: a Locator places what
the camera sees in BASE, the scene of the part gives its grasps, and robot.pick runs the best one. The
hand-over needs no written pose and no force reading: the camera locates the open hand, and seeing it IS the signal.

The hand comes back in BASE millimetres through the camera's own calibration, so it needs a FIXED camera:
a wrist rig's artifact is CAMERA to TOOL, and the builder refuses it rather than composing one of its own.

Run it at the cell, under its profile, once that camera is calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/13_speak_pick_and_hand_handover.py
"""

import threading
import time

from willy import (Camera, Confirmation, Locator, Pose, PushToTalkSource, Robot, TalkButton,
                   TerminalConfirmer, build_hand_finder_on_camera, load_speech_section, load_tree,
                   shared_speech)

tree, speech = load_tree(), load_speech_section()  # speech is models.stt of the same profile
standoff_mm, timeout_s = 120.0, 30.0  # the arm stops OVER the palm; set it above your tallest part
button = TalkButton.from_parts()


def release_on_enter() -> None:
    input()
    button.release()


with PushToTalkSource.from_config(config=speech, switch=button) as microphone:  # 12, verbatim
    input("Press Enter, say what to pick, then press Enter again. ")
    button.press()
    threading.Thread(target=release_on_enter, daemon=True).start()
    turn = microphone.record(timeout_s=1.0)
said = shared_speech().for_config(config=speech).propose(turn.samples, samplerate=turn.samplerate)
spoken = Confirmation.from_proposal(proposal=said, confirmer=TerminalConfirmer.from_parts())
print(spoken)
if spoken.confirmed is None:
    raise SystemExit("nothing confirmed; the arm never moves on an unconfirmed word")

with Camera.from_tree(tree) as camera:
    robot = Robot.from_tree(tree, cameras=[camera])
    locator = Locator.from_tree(tree, camera=camera, tool_pose=robot.arm.get_tcp_pose)
    # MediaPipe over the camera already open, and that camera's own calibration; it claims no device.
    hands = build_hand_finder_on_camera(tree.app_config.models.handdetect, camera)

    with robot.connected():
        located = locator.locate(spoken.confirmed)
        best = located.scene(0, tree.robot).grasps().best if located.objects else None
        if best is None:
            raise SystemExit("nothing located for that prompt, or no grasp on what was")
        picked = robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0))
        print(picked)
        if not picked.ok:
            raise SystemExit("pick refused, nothing to hand over")
        print("hold your open hand out where you want the part")
        deadline, seen = time.monotonic() + timeout_s, None
        while seen is None and time.monotonic() < deadline:
            # One hand or nothing: two is a reason to stop, because this decides where the arm goes.
            seen, _annotated = hands.find_hand()
        if seen is None:
            raise SystemExit(f"no single hand with usable depth in {timeout_s} s; part stays held")
        palm = seen.position.position_base
        handover = Pose.tool_down(float(palm[0]), float(palm[1]), float(palm[2]) + standoff_mm)
        # Planned and camera-world checked like every move; only a move that arrived opens the hand.
        print(moved := robot.move(handover))
        if not moved.ok:
            raise SystemExit("the arm did not reach the hand; the part stays held, nothing was released")
        print(robot.release())
