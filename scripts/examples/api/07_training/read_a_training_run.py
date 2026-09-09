"""Read what a training run produced: four numbers, and which one of them means anything.

The lift is the number, the held-out hit rate minus its own measured floor, and the verdict on top
of it -- was it still improving when it stopped -- is what decides the next run. Everything here is
computed from the `epochs.json` the trainer wrote after each epoch, never scraped from a log.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 07_training, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.robot.grasping.deep.eval import run_report  # noqa: E402

RUN = Path("logs/examples/training/my_parts_model")

if not (RUN / "epochs.json").is_file():
    print(f"no training run at {RUN}: a run is the trainer's own output DIRECTORY, and "
          f"train_on_your_own_meshes.py in this folder writes one in about a minute")
else:
    # 1. The epoch rows. A resumed run is one experiment split across wall-clock gaps, so the
    #    sittings are counted and the x axis stays epochs, never time.
    card = run_report.build_report(RUN.name, RUN)
    print(f"{len(card.epochs)} epoch(s) across {len(card.segments)} sitting(s), "
          f"{card.compute_hours:.2f} h of compute")

    # 2. The three numbers that are NOT the answer: a wider net reaches a lower loss without
    #    proposing one better grasp, a raw hit rate says nothing without its floor, and the best
    #    epoch is a maximum over noisy draws rather than a level the model ever held.
    print(f"loss {card.final.loss:.4f}, raw test hit {card.final.test_hit:.4f} against a floor of "
          f"{card.final.test_floor:.4f}, best at epoch {card.best.epoch}")

    # 3. The lift, and the whole report around it.
    print(f"test lift {card.final.test_lift:+.4f}")
    print(run_report.format_report(card))

    # 4. The verdict, which is what decides the next run. Below two full windows the report declines
    #    to answer rather than printing a placeholder that reads like a converged run.
    print("no verdict: fewer than two windows" if not card.comparable
          else f"still improving over the last {card.window} epoch(s)" if card.still_improving
          else f"flattened, judged over the last {card.window} epoch(s)")

    # 5. The curve, with the floors drawn on it: without them the test line reads as rising from
    #    zero, and every run looks like it learned something.
    print(run_report.write_curve(card, RUN / "curve.png"))
