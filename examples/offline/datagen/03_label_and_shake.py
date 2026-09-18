"""Label a rendered dataset's grasps from its geometry, then shake a few of them in a physics engine to see
whether they hold.

Labelling is closed-form geometry on the CPU and re-runs on a dataset without rendering it again; the shake needs
MuJoCo in this interpreter and says so when it is missing.
"""

import tempfile

from willy import DatasetBuild, PhysicsSampling, engine_is_available

with tempfile.TemporaryDirectory() as work:
    build = DatasetBuild.from_file(name="labelled", scenes=8, seed=0, engine="none", out_root=work)
    build.render()

    # Every jaw and suction grasp each object's geometry admits, written into the dataset as grasps.jsonl.
    labelled = build.label(density="default")
    summary = labelled.summary
    # A label count means nothing without the jaw it was cut for and the density it was drawn at: the
    # same object earns many times the labels at density "grid".
    print(f"{summary['objects']} objects, {summary['jaw']} jaw and {summary['suction']} suction labels, "
          f"cut for the {summary['jaw_model']} jaw at density default")

    # Why candidates were refused; finger_collision is the one caused by the scene around the object.
    for reason, count in sorted(summary["rejected"].items(), key=lambda row: -row[1])[:5]:
        print(f"  {reason:24s} {count:6d}")

    # The labels know geometry, not dynamics and not reach. The shake closes a jaw on a label, takes the
    # table away and watches whether the object stays in the hand.
    available, why_not = engine_is_available("mujoco")
    if not available:
        print(why_not)
        raise SystemExit
    shaken = PhysicsSampling.from_dataset(build.root, engine="mujoco").sample(per_class=4)
    print(shaken)
