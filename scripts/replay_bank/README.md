# The cell's recorded picks, judged again (`scripts/replay_bank/`)

**What today's code makes of the frames the cell already took, before the cell takes new ones.** A cell keeps
every look of every pick (`record_views`): the colour, the measured and the surface depth, the
camera's matrix, CAMERA and TOOL to BASE, the joints, the segmentations the detector gave and the target's cloud
fused over the looks. `run.py` builds the owner's cell from its config tree, offline, and asks of every recorded pick
what the cell would do with those frames now: the camera world of all its looks, the target held out as the cell
holds it, the target's grasps ranked by the pick's own orchestrator, and each grasp's line down and lift judged sample
by sample by the cell's own path gate, in the world of that grasp.

```bash
python scripts/replay_bank/run.py                                        # every pick of every day
python scripts/replay_bank/run.py --days 2026-10-12 --limit 10           # a day, its first ten picks
python scripts/replay_bank/run.py --picks 114555Z 115142Z                # picks by a part of their name
python scripts/replay_bank/run.py --out logs/replay_bank/after.jsonl --compare logs/replay_bank/before.jsonl
python scripts/replay_bank/run.py --config my_copy/config                # the same picks, another config tree
python scripts/replay_bank/run.py --off ground parts_beside              # levers set aside, to measure them
python scripts/replay_bank/run.py --push --views                         # where no grasp is admitted: push, view
```

A pick is judged on the last look whose detector named its target (`RecordedPick.judged`): the cell went on with that
look, and projecting the fused cloud into a later look that did not see the target made a sliver of a mask of it (21
of the 486 picks of 2026-10-12 to 10-15). With `--push` and `--views`, a pick with no admitted grasp asks the cell's own
pick loop what it would do next, nothing moving: the push of the part it would plan as every grasp collided, the push
of the scene a task taking every part would plan, and the view it would generate, screened by the arm's inverse
kinematics, the travel cap and the cable window on the instant controller. Its straight joint line is judged as on the
cell only with `--plan`, whose cuRobo needs the GPU; the instant controller then stands at the view.

Nothing moves and nothing is sent: the controller is the bench's instant stand-in (`scripts/bench/instant_ur.py`),
cuRobo is not asked (where only a camera's box refuses, the exact guard decides on the cell too), and a look the
recording lacks is never made up. The whole bank, 125 picks of 2026-10-08 and 2026-10-12, takes about six minutes on
the desk's CPU.

## The gate

Each pick is one JSON line in `--out`: the grasps the calculator offered (or why none), how many the guard admits, the
first admitted, and what refused the others (`rfinger at a box the camera saw`, `a finger or the hand at the support`,
`a joint limit`). `--compare` sets a run before beside it, pick by pick, and exits 1 where any pick has fewer grasps
admitted than before. That is the measure every lever and every port is held to: a change that admits fewer grasps on
the bank says so here, not on the cell.

| Run | Picks with a grasp the guard admits | Grasps admitted |
|---|---|---|
| 2026-10-13, commit 2 (parts beside, the guard on the controller's flange), 3 mm | 71 of 116 | 504 |
| the same, without the parts beside and the crate's floor | 69 of 116 | 473 |
| the same at 2 mm from a camera's box | 75 of 116 | 534 |
| the same at 1 mm | 75 of 116 | 546 |
| 2026-10-16, five days: the body search of 2026-10-15 (a lean only where straight down was refused, no roll) | 416 of 479 | 4577 |
| the same, the part's body grasped as SFE grasped (a lean for every grasp to 45 degrees, rolls, upright first) | 427 of 479 | 4637 |
| the same, all of SFE (leans to 90 degrees over the palm, the fine pass, rolled leans) | 428 of 479 | 4641 |
| the same, a finger beside a taller neighbour only where a camera saw (5 mm), as shipped | 425 of 479 | 4614 |

## Where the recordings lie

Outside the repository, in `D:/dev/willy_replay_bank` (or wherever `WILLY_REPLAY_BANK` points):
`<bank>/<day>/views/*.npz`, as the cell wrote them, and `<bank>/robominds_config`, the cell's config tree of that
day. A bank run takes the tree each pick ran with (`--config`): since 2026-10-14 10:03 the owner's cell runs a Robotiq
at 2 mm from a camera's box (`robominds_config_robotiq_ported`), before it the toggle Hand-E, at 2 mm on 10-14
(`robominds_config_toggle2mm_ported`) and at 3 mm before (`robominds_config_ported`). A day's recordings are copied there from the cell's logs folder (`robot/views`) as they are. Both recording
formats read: a pick recorded before 2026-10-12 carries no segmentation, so its target's mask is its fused cloud
projected into the look, and it has no neighbour the detector named.

`tests/scripts/replay_bank/test_the_replay_bank_reads_what_a_cell_recorded.py` pins what needs no recording: a pick read back as written,
a refusal named by what met what, and the gate.
