"""The three grasp modes, what each one switches on, and how a mode the service was not built in refuses.

A pick is not one behaviour. `easy` is the plain open-loop attempt on one object; `auto` decides per scene
whether to sample densely; `dense_clutter` samples a bin rather than one object, and is the one mode whose
profile lists a push (`nudge_target`, and only inside a declared fixture). The push runs inside the pick
attempt of a wrist camera, never from the recovery loop: a part every grasp of which collided, with a
neighbour seen within 25 mm, is pushed with the open jaws and picked from the look taken again.

The mode is chosen when the SERVICE IS BUILT, and `pick(mode=...)` only narrows the behaviour of one attempt
within it -- so a service built for one sampler refuses another by name instead of quietly sampling the other
way.

That refusal is the point: a downgraded pick reported under the name of the one asked for would misstate what
ran. `closed_loop` and `dense_autonomous` were removed on 2026-09-29 with the two-scan refinement they ran; a
cell that still names one is refused with the mode to name instead.

A dummy arm and a synthetic box, so the wiring is what runs. Grasp quality is measured elsewhere.
"""

from willy import Cell, GraspMode, load_tree

robot = load_tree("console_dummy").robot

# What each mode locks. The profile travels on every report, so an attempt can always say which
# toggles were in effect, whatever the config was edited to afterwards.
print(f"{'mode':15} {'sampling':16} recovery it may use")
for mode in GraspMode:
    cell = Cell.rehearsal(robot, mode=mode)
    service = cell.build()  # a separate step, so preflight() can run before anything is built
    with cell.connected():
        profile = service.pick().profile
    print(f"{profile.mode.value:15} {profile.sampling_mode.value:16} "
          f"{', '.join(profile.recovery_allowed_actions) or '(none)'}")

# Built in one mode, asked for another. Only `auto` runs here, and the rest come back refused by
# name: the service will not sample a bin because a caller passed `dense_clutter` to an attempt.
print()
print("built in auto, asked per attempt for:")
auto = Cell.rehearsal(robot)  # no mode=, so the service's own default, which is auto
service = auto.build()
with auto.connected():
    for mode in GraspMode:
        report = service.pick(mode=mode)
        print(f"  {mode.value:15} {report.outcome.value}")

# A mode that left is refused before anything is built, with the one to name instead.
try:
    Cell.rehearsal(robot, mode="closed_loop").build()
except ValueError as refused:
    print()
    print(refused)

# The whole report, which is what a campaign records. `layers` is read off the attempt rather than
# off the config: a block switched on in YAML but never reached does not appear there.
with auto.connected():
    print()
    print(service.pick())

# At a cell it is the same three lines with the cell's own arm, cameras and models:
#     cell = Cell.from_tree(load_tree(), prompt="a red cube", mode="dense_clutter")
#     with cell.connected():
#         print(cell.build().pick())
# `dense_clutter` is the bin mode; `docs/grasping-config-reference.md` lists every `robot.grasping`
# block a mode needs, and each one ships off.
