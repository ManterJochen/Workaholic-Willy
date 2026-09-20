"""What a dataset will cost, in hours and gigabytes, before anybody starts it.

Every coefficient is measured and says what it was measured over; where a number extrapolates past
what was measured, the estimate says so. No engine, no GPU: this is arithmetic over a config, so it
answers on the machine you are reading this on rather than on the one that would run the build.
"""

from willy import DatasetBuild

# Isaac path-traces RGB and needs its own GPU install; mujoco settles with a real solver, no RGB;
# none seats objects analytically and needs nothing. The three price the same 500 usable scenes very
# differently, and only isaac's aggregate figure came from more than one process.
for engine in ("isaac", "mujoco", "none"):
    priced = DatasetBuild.from_file(name="priced", scenes=500, engine=engine)
    print(f"--- {engine} ---")
    print(priced.cost(jobs=4, epochs=60, folds=5))  # the same report `datagen cost` prints
    print()

# The three traps this exists to keep you out of, made concrete.
mujoco = DatasetBuild.from_file(name="priced", scenes=500, engine="mujoco")
single, parallel = mujoco.cost(jobs=1), mujoco.cost(jobs=8)

# Per-shard time is not throughput: concurrent shards share one GPU.
print(f"500 usable scenes on mujoco: {single.hours:.1f} h on 1 job, {parallel.hours:.1f} h on 8, "
      f"not an eighth of it")

# Scenes requested are not scenes usable: an engine that refuses a family by name yields fewer than
# it is asked for, so the number to put in the config is the request, not the target.
print(f"ask for {single.requested_scenes} to end up with {single.usable_scenes}")

# Rendering is not the whole bill: labelling, corpus extraction and training are each priced beside
# it, and on the cheap engines they are most of it.
for stage in single.stages:
    print(f"  {stage.name:22} {stage.hours:6.2f} h  {stage.gigabytes:6.2f} GB")
