"""What a dataset will cost, in hours and gigabytes, before anybody starts it.

Every coefficient is measured and says what it was measured over; where a number extrapolates past
what was measured, the estimate says so. No engine, no GPU: this is arithmetic over a config.
"""

from datagen.cost import ENGINE_COSTS, estimate, format_estimate

# Isaac path-traces RGB and needs its own GPU install; mujoco settles with a real solver, no RGB;
# none seats objects analytically and needs nothing. The three price the same 500 usable scenes very
# differently, and only isaac's aggregate figure came from more than one process.
for engine in sorted(ENGINE_COSTS):
    result = estimate(500, engine=engine, jobs=4, epochs=60, folds=5, refit=True)
    print(f"--- {engine} ---")
    print(format_estimate(result))
    print()

# The three traps this module exists to keep you out of, made concrete: per-shard time is not
# throughput (jobs=1 vs jobs=8 on the same engine), scenes requested are not scenes usable (a
# refused family lowers the yield), and rendering is not the whole bill (label and corpus stages
# are priced beside it).
single = estimate(500, engine="mujoco", jobs=1)
parallel = estimate(500, engine="mujoco", jobs=8)
print(f"mujoco at 500 usable scenes: {single.hours:.1f} h on 1 job, {parallel.hours:.1f} h on 8 "
      f"(not 1/8th: concurrent jobs share one GPU)")
