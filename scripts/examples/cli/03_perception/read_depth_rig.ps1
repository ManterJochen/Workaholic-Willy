# Which camera a cell opens, and whether it can deliver depth, from the command line.
#
# The Python twin of this file is scripts/examples/api/03_perception/read_depth_rig.py, and it
# reads the same rig list before deciding whether opening anything is worth it. A stereo rig hands
# back two views and no depth channel, so no grasp can be synthesised from what it returns.

# 1. Which rig this cell opens, where that was set, and the rigs it could have named instead.
python -m src.config explain camera.cameras.primary_rig_id

# 2. The two keys that hold rig identity. Note what is NOT reachable from here: `source`,
#    `enabled` and `rgbd_backend` live inside the rig list, and `where` does not descend into it.
python -m src.config where rig

# 3. Prove the rig standalone: open it, grab, detect, segment, then report intrinsics and the hole
#    fraction of every mask counted on raw sensor depth. This one needs the camera plugged in.
#    python -m src.robot.perception --rig realsense_d435 --warmup 10

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
