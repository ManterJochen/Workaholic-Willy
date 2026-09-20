"""The five grasp modes, what each one switches on, and how a mode that is not wired refuses.

A pick is not one behaviour. `easy` is the plain open-loop attempt; `dense_clutter` samples a bin
rather than one object; `closed_loop` re-scans at the standoff before it closes; `dense_autonomous`
does both and may push a blocker aside. The mode is chosen when the SERVICE IS BUILT, and
`pick(mode=...)` only narrows the behaviour of one attempt within it -- so a service built for one
sampler refuses another by name instead of quietly sampling the other way.

That refusal is the point. Every advanced block under `robot.grasping` ships `enabled: false`, so a
cell asking for a mode whose machinery nobody wired gets `MODE_NOT_AVAILABLE` rather than a
downgraded pick reported under the name of the one it asked for.

A dummy arm and a synthetic box, so the wiring is what runs. Grasp quality is measured elsewhere.
"""

from willy import Cell, GraspMode, load_tree

robot = load_tree("console_dummy").robot

# What each mode locks. The profile travels on every report, so an attempt can always say which
# toggles were in effect, whatever the config was edited to afterwards.
print(f"{'mode':18} {'sampling':16} {'refine':7} {'verify':7} recovery it may use")
for mode in GraspMode:
    cell = Cell.rehearsal(robot, mode=mode)
    service = cell.build()  # a separate step, so preflight() can run before anything is built
    with cell.connected():
        profile = service.pick().profile
    print(f"{profile.mode.value:18} {profile.sampling_mode.value:16} "
          f"{str(profile.refinement_enabled):7} {str(profile.verification_enabled):7} "
          f"{', '.join(profile.recovery_allowed_actions) or '(none)'}")

# Built in one mode, asked for another. Only `auto` runs here, and the rest come back refused by
# name: the service will not sample a bin because a caller passed `dense_clutter` to an attempt.
print("\nbuilt in auto, asked per attempt for:")
auto = Cell.rehearsal(robot)  # no mode=, so the service's own default, which is auto
service = auto.build()
with auto.connected():
    for mode in GraspMode:
        report = service.pick(mode=mode)
        print(f"  {mode.value:18} {report.outcome.value}")

# The whole report, which is what a campaign records. `layers` is read off the attempt rather than
# off the config: a block switched on in YAML but never reached does not appear there.
with auto.connected():
    print()
    print(service.pick())

# At a cell it is the same three lines with the cell's own arm, cameras and models:
#     cell = Cell.from_tree(load_tree(), prompt="a red cube", mode="dense_clutter")
#     with cell.connected():
#         print(cell.build().pick())
# `dense_clutter` and `dense_autonomous` are the bin modes; `docs/grasping-config-reference.md`
# lists every `robot.grasping` block a mode needs, and each one ships off.
