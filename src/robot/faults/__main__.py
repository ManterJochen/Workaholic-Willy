"""Acknowledge the fault of a cell, as the person standing at it.

    python -m src.robot.faults acknowledge --profile ursim          # asks at the prompt
    python -m src.robot.faults acknowledge --profile ursim --yes    # the person at the cell says yes here

Connects the arm alone under the cell lock, reads the controller's state and names the fault it describes. A fault no
software clears (an emergency button, a controller powered off) is said with what to do about it, and nothing is sent.
Any other is shown in its own sentence and acknowledged only after the person confirms: at the prompt, or with
``--yes`` typed by the person at the cell. A shell with no terminal and no ``--yes`` asks nobody and acknowledges
nothing. Never automatic, and never from the console (owner, 2026-09-19 and 2026-09-26).

Exit codes: 0 acknowledged, or no fault; 1 refused, declined, not acknowledgeable or not taken; 2 usage.
"""

from __future__ import annotations

import argparse
import sys

from src.contracts import UNSET


def _ask(sentence: str) -> bool:
    """The person at the cell, asked at the prompt; ``False`` where no terminal can ask anyone."""
    print(sentence, flush=True)
    if not sys.stdin.isatty():
        print("No terminal answers here, so nothing is acknowledged: pass --yes as the person at the cell.", flush=True)
        return False
    answer = input("Acknowledge it? You stand where you can see the arm. Type 'yes': ")
    return answer.strip().lower() == "yes"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.robot.faults",
        description="Acknowledge the fault of a cell, as the person standing at it. Never automatic.",
    )
    verbs = ap.add_subparsers(dest="verb", required=True)
    acknowledge = verbs.add_parser("acknowledge", help="read the arm's fault and acknowledge it after you confirm")
    acknowledge.add_argument("--profile", type=str, default=None,
                             help="config profile chain, e.g. 'ursim' (default: WILLY_PROFILE, else the base tree)")
    acknowledge.add_argument("--data-dir", type=str, default=None, help="config data root")
    acknowledge.add_argument("--yes", action="store_true",
                             help="you stand at the cell and confirm here instead of at the prompt")
    args = ap.parse_args(argv)

    from src.config import ConfigTree

    from .acknowledge import FaultAcknowledgement

    loaded = ConfigTree.from_directory(root=args.data_dir if args.data_dir else UNSET,
                                       profile=args.profile if args.profile else UNSET).load()
    if not loaded.ok:
        print(f"REFUSED: the config did not load: {loaded.error}", flush=True)
        return 1
    confirm = (lambda _sentence: True) if args.yes else _ask
    try:
        acknowledgement = FaultAcknowledgement.from_tree(loaded, confirm=confirm)
    except Exception as exc:  # noqa: BLE001 - an arm that does not build is a refusal, not a traceback
        print(f"REFUSED: the arm could not be built: {type(exc).__name__}: {exc}", flush=True)
        return 1
    report = acknowledgement.run()
    print(report.render(), flush=True)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
