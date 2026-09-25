"""Speak what to pick, let a person confirm it, and pick what was confirmed.

Run it at the cell, under the cell's profile, at a terminal where a person can answer:
    WILLY_PROFILE=<your cell> python examples/real_robot/14_speak_a_command.py
"""

import threading

from willy import (Cell, Confirmation, PickRun, PushToTalkSource, Recording, TalkButton,
                   TerminalConfirmer, load_speech_section, load_tree, shared_speech)

cell = Cell.from_tree(load_tree())  # built when the pick runs
speech = load_speech_section()  # models.stt of the same profile; speech needs no camera or robot

# A turn is what the microphone catches while a talk switch is held. No reader for a USB hand or
# foot switch is built yet: the operator console's talk key (a switch that types a key can press
# it) and POST /v1/voice/talk press the console's button. Here the Enter key presses this one.
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

# A voice detector decides whether anything was said, then the transcriber what; both load here.
proposal = shared_speech().for_config(config=speech).propose(turn.samples, samplerate=turn.samplerate)
print(proposal)

# Speech never acts on its own: a person answers yes, or types the words they meant. Any other
# answer, or no terminal to ask, confirms nothing. Stopping stays with the emergency stop.
confirmation = Confirmation.from_proposal(proposal=proposal, confirmer=TerminalConfirmer.from_parts())
print(confirmation)
if confirmation.confirmed is None:
    raise SystemExit(1)

# The confirmed words are what the cameras ground; the arm plans against the world they build.
pick = PickRun.from_cell(cell, runs=1, prompt=confirmation.confirmed, recording=Recording.off())
report = pick.execute()
print(report)
raise SystemExit(report.exit_code)
