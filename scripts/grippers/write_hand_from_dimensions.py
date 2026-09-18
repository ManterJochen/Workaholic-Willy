"""Write a gripper's collision bundle from the numbers its registry file carries, for a hand nobody baked.

    .venv/Scripts/python.exe scripts/grippers/write_hand_from_dimensions.py acme_2f --inflation-mm 10 --origin flange
    .venv/Scripts/python.exe scripts/grippers/write_hand_from_dimensions.py acme_2f --inflation-mm 10 --origin flange --measure-only

Run it with the project venv. No simulator, no GPU, no USD: numpy over one registry file.

The planner and the exact mesh guard both read a collision bundle, so a gripper this repository has never scanned
cannot be planned with until it has one. A customer describes theirs in ``config/grippers/<name>.yaml`` and this turns
those numbers into the same arrays a baked bundle carries: a housing box and two finger boxes, in the bundle frame,
approach +Y, closing X, binormal Z.

An envelope is not the hand. Measured against the two shipped hands that carry both a bundle and a full set of
dimensions, at each hand's own declared grasp centre, the Hand-E needs 5.73 mm of inflation before the envelope
encloses it and the EGU-50 needs 9.50 mm. So ``--inflation-mm`` is required and has no default, and the number belongs
to the hand rather than to this script. Ten millimetres is what the two measured hands ask for; a hand of a different
shape may ask for more, and ``--measure-only`` on a hand that has a bundle is how that is found out.

A hand whose ``palm_measured`` is false is refused by name, and a bundle that is already there is never overwritten,
because a scan describes a hand better than five numbers do.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BUNDLES = REPO / "src/robot/safety/data"
_WRITER = REPO / "src/robot/safety/planning/robot/hand_from_dimensions.py"


def _module():
    spec = importlib.util.spec_from_file_location("willy_hand_from_dimensions", _WRITER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["willy_hand_from_dimensions"] = module
    spec.loader.exec_module(module)
    return module


def needed_inflation_mm(module, hand: str, bundle: Path) -> "float | None":
    """The smallest inflation that encloses ``hand``'s baked bundle, or ``None`` where there is no bundle to ask.

    This is the only way to put a number on the envelope, and it is only possible for a hand that has both halves.
    Measured as the furthest any baked vertex lies outside the union of the boxes.
    """
    from src.config.grippers import load_gripper

    if not bundle.is_file():
        return None
    jaw = load_gripper(hand).jaw
    with np.load(bundle, allow_pickle=True) as held:
        points = np.concatenate([np.asarray(held[f"{part}__v"], dtype=np.float64)
                                 for part in ("gripper", "lfinger", "rfinger")])
    boxes = module.boxes_for_hand(jaw, inflation_mm=0.0)
    worst = np.full(len(points), np.inf)
    for box in boxes:
        gap = np.abs(points - np.asarray(box.centre_mm)) - np.asarray(box.half_extents_mm)
        worst = np.minimum(worst, np.linalg.norm(np.maximum(gap, 0.0), axis=1))
    return float(worst.max())


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Write a gripper's collision bundle from its registry dimensions.")
    parser.add_argument("hand", help="the registry name, robot.gripper.model")
    parser.add_argument("--inflation-mm", type=float, default=None,
                        help="how far every box grows past the measurements, required: an envelope is not the hand")
    parser.add_argument("--out", default=None, help="where to write; the committed bundle path by default")
    parser.add_argument("--origin", choices=("flange", "mounting_face"), default=None,
                        help="where grasp_centre_mm was measured from, required: the flange, or the hand's own mounting "
                             "face, which a cell then moves out by its declared plates")
    parser.add_argument("--measure-only", action="store_true",
                        help="report what the envelope would cost against a baked bundle, and write nothing")
    args = parser.parse_args(argv)

    from src.config.grippers import available_grippers

    module = _module()
    if args.hand not in available_grippers():
        raise SystemExit(f"{args.hand!r} is not a gripper this registry knows; it holds "
                         f"{', '.join(sorted(available_grippers()))}")

    from src.config.grippers import load_gripper

    jaw = load_gripper(args.hand).jaw
    said = module.refusal_for(jaw)
    bundle = Path(args.out) if args.out else BUNDLES / f"{args.hand}_hand_meshes.npz"

    needed = needed_inflation_mm(module, args.hand, BUNDLES / f"{args.hand}_hand_meshes.npz")
    if needed is None:
        print(f"[measure] {args.hand} has no baked bundle, so what the envelope costs cannot be measured here. "
              f"The two hands with a measured palm asked for 5.73 mm and 9.50 mm.")
    else:
        print(f"[measure] {args.hand} has a baked bundle: an envelope with no inflation leaves it "
              f"{needed:.2f} mm outside, so that is the smallest inflation that would enclose it.")

    if args.origin is None:
        raise SystemExit(
            "--origin is required and has no default: grasp_centre_mm is measured from the flange or from the hand's "
            "own mounting face, and a body written from the wrong one sits one plate away from the hand."
        )
    baked = BUNDLES / f"{args.hand}_hand_meshes.npz"
    if baked.is_file():
        with np.load(baked, allow_pickle=True) as held:
            baked_origin = str(held["gripper__origin"][0]) if "gripper__origin" in held.files else "flange"
        if baked_origin != args.origin:
            raise SystemExit(
                f"{args.hand}: --origin {args.origin}, and its baked bundle's numbers start at the {baked_origin}. The "
                f"registry numbers were measured off that bundle, so they start there too."
            )
    gap = module.housing_gap_mm(jaw)
    if gap > 0.0:
        print(f"[measure] {gap:.2f} mm between the {args.origin} and the housing's back is in no body of this "
              f"envelope: an adapter or a boss there is modelled by nothing unless a plate with a cross section "
              f"declares it.")
    if said is not None:
        raise SystemExit(f"{args.hand}: {said}")
    if args.measure_only:
        for box in module.boxes_for_hand(jaw, inflation_mm=args.inflation_mm or 0.0):
            print(f"  {box.render()}")
        return 0

    if args.inflation_mm is None:
        raise SystemExit(
            "--inflation-mm is required and has no default. An envelope built from measurements does not enclose "
            "the body it stands for: the two hands that could be measured asked for 5.73 mm and 9.50 mm. Measure "
            "yours with --measure-only against a baked bundle, or state the number your cell can live with."
        )

    try:
        written = module.write_bundle(args.hand, bundle, inflation_mm=args.inflation_mm, origin=args.origin)
    except ValueError as exc:
        raise SystemExit(f"{args.hand}: {exc}") from None
    print(f"wrote {written}")
    # The row, recorded by the writer: bundles.json refuses a bundle nobody recorded, and a customer does not
    # hand-edit it. The note carries no path, because a machine path would travel into committed data.
    from src.robot.safety.planning.bundle_index import INDEX_NAME, record_hand_bundle

    if (Path(written).parent / INDEX_NAME).is_file():
        row = record_hand_bundle(
            written, source="registry_dimensions",
            note=(f"an envelope built from the registry numbers of {args.hand}, starting at the {args.origin}, "
                  f"every box inflated {args.inflation_mm:g} mm"),
        )
        print(f"  {row.render()}")
    else:
        print(f"  no {INDEX_NAME} beside {Path(written).name}, so no row was recorded")
    print(f"  source dimensions, origin {args.origin}, inflation {args.inflation_mm:g} mm. Which arms it may be composed onto is "
          f"measured, not written: scripts/curobo/matrix_gate.py --hand {args.hand}")
    print("  next: fit its spheres for the planner, "
          f"scripts/curobo/fit_cover_spheres.py --hand {args.hand} --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
