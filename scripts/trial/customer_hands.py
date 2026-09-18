"""Two customer hands whose answers are known before the chain runs.

    .venv/Scripts/python.exe scripts/trial/customer_hands.py registry --hand acme_dims --out DIR
    .venv/Scripts/python.exe scripts/trial/customer_hands.py export-stl --from robotiq_hande --out DIR
    .venv/Scripts/python.exe scripts/trial/customer_hands.py compare acme_mesh robotiq_hande --tolerance-mm 0.001
    .venv/Scripts/python.exe scripts/trial/customer_hands.py bounds --hand acme_dims

A trial instrument, not product code, and nothing it makes is committed.

``acme_dims`` is invented and physically plausible, a hand nobody shipped, described only by the numbers a customer
would write into the registry. Its collision boxes were computed by hand from ``hand_from_dimensions.boxes_for_hand``
at a 10 mm inflation, with the finger boxes reaching ``finger_length_mm`` behind the grasp centre, so the trial
checks the writer against arithmetic and not against itself.

``acme_mesh`` is the committed Hand-E bundle exported as three binary STL files in a vendor frame whose rotation is not
one the catalogue uses, in metres, with the mounting face 12.5 mm up the vendor's approach axis. Its known answer is
the committed Hand-E: a customer who feeds those files to ``scripts/grippers/write_hand_from_mesh.py`` with the
recorded axes must get the Hand-E back. The export is Isaac-derived geometry and is written only under a trial log
folder, never into the tree.

Exit codes: 0 the answer is the known one, 1 it is not and the difference is named, 2 the question cannot be asked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = REPO / "src" / "robot" / "safety" / "data"
PARTS = ("gripper", "lfinger", "rfinger")

#: The acme_dims jaw, every field of ``ParallelJawSpec``, the grasp centre measured from the flange.
ACME_DIMS_JAW: dict[str, Any] = {
    "grasp_centre_mm": 140.0, "aperture_mm": 60.0, "min_width_mm": 4.0, "closed_width_mm": 1.0,
    "finger_ahead_mm": 11.0, "finger_behind_mm": 34.0, "finger_length_mm": 36.0, "finger_thickness_mm": 10.0,
    "finger_width_mm": 20.0, "finger_pad_overlap_mm": 2.0, "pad_length_mm": 22.0, "pad_ahead_mm": 11.0,
    "pad_behind_mm": 11.0,
    # The palm back sits on the flange: 140 - 34 - 106 = 0. Its thickness is the open fingers' outer face, 2 x 40.
    "palm_depth_mm": 106.0, "palm_width_mm": 70.0, "palm_thickness_mm": 80.0, "palm_measured": True,
    "friction_coefficient": 0.5, "friction_is_default": True,
}
ACME_DIMS_INFLATION_MM = 10.0
#: By hand: tip 151, housing back 106, finger back 140 - 36 = 104, palm back 0, outer face 30 + 10 = 40.
ACME_DIMS_BOXES = (
    "gripper: 100 x 126 x 90 mm centred at (0, 53, 0) mm",
    "lfinger: 30 x 67 x 40 mm centred at (-35, 127.5, 0) mm",
    "rfinger: 30 x 67 x 40 mm centred at (35, 127.5, 0) mm",
)
ACME_DIMS_BOUNDS = {
    "gripper": ((-50.0, -10.0, -45.0), (50.0, 116.0, 45.0)),
    "lfinger": ((-50.0, 94.0, -20.0), (-20.0, 161.0, 20.0)),
    "rfinger": ((20.0, 94.0, -20.0), (50.0, 161.0, 20.0)),
}

#: The vendor frame of acme_mesh: rows are the bundle's closing, approach and binormal in vendor axes, so
#: ``bundle = (vendor_mm - mount face along the approach) @ R.T``. Determinant +1, and not a permutation any catalogue
#: bake uses.
VENDOR_ROTATION = ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))
VENDOR_WORDS = {"closing": "-Y", "approach": "+Z", "binormal": "-X"}
VENDOR_MOUNT_FACE_MM = 12.5
VENDOR_SCALE_TO_MM = 1000.0


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def bundle_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    return candidate if candidate.suffix == ".npz" else DATA / f"{name_or_path}_hand_meshes.npz"


def _module(name: str, relative: str) -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / relative)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# --- acme_dims --------------------------------------------------------------------------------------------------------


def registry_text(hand: str = "acme_dims") -> str:
    """The registry file a customer writes for acme_dims, LF endings."""
    jaw = "\n".join(f"    {key}: {str(value).lower() if isinstance(value, bool) else value}"
                    for key, value in ACME_DIMS_JAW.items())
    return (
        f"# A trial hand: invented numbers, never committed.\n"
        f"gripper:\n  model: {hand}\n  kind: parallel_jaw\n"
        f"  source: >-\n    Invented for the customer chain trial, physically plausible, grasp centre from the flange.\n"
        f"  jaw:\n{jaw}\n"
    )


def acme_dims_boxes(*, inflation_mm: float = ACME_DIMS_INFLATION_MM) -> tuple[str, ...]:
    from src.config.schema.grippers import ParallelJawSpec
    from src.robot.safety.planning.robot.hand_from_dimensions import boxes_for_hand

    jaw = ParallelJawSpec.model_validate(ACME_DIMS_JAW)
    return tuple(box.render() for box in boxes_for_hand(jaw, inflation_mm=inflation_mm))


# --- acme_mesh --------------------------------------------------------------------------------------------------------


def export_stl(source: str, out: Path) -> dict[str, Any]:
    """The bundle ``source`` as three binary STL files in the acme_mesh vendor frame, and the answer that undoes them."""
    import trimesh

    path = bundle_path(source)
    arrays = _arrays(path)
    rotation = np.asarray(VENDOR_ROTATION, dtype=np.float64)
    approach = "XYZ".index(VENDOR_WORDS["approach"][1])
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for part in PARTS:
        vendor = np.asarray(arrays[f"{part}__v"], dtype=np.float64) @ rotation
        vendor[:, approach] += VENDOR_MOUNT_FACE_MM
        mesh = trimesh.Trimesh(vertices=vendor / VENDOR_SCALE_TO_MM, faces=np.asarray(arrays[f"{part}__f"]),
                               process=False)
        target = out / f"{part}.stl"
        mesh.export(str(target), file_type="stl")
        files[part] = str(target)
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    answer = {
        "source": source, "source_bundle": str(path), "source_arrays_digest": digest.hexdigest(),
        "files": files, "rotation_rows": VENDOR_ROTATION, "axes": VENDOR_WORDS,
        "mount_face_mm": VENDOR_MOUNT_FACE_MM, "scale_to_mm": VENDOR_SCALE_TO_MM,
        "origin": str(arrays["gripper__origin"].reshape(-1)[0]) if "gripper__origin" in arrays else "flange",
    }
    (out / "answer.json").write_text(json.dumps(answer, indent=1) + "\n", encoding="utf-8", newline="\n")
    return answer


# --- compare ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleComparison:
    a: str
    b: str
    tolerance_mm: float
    #: part -> the symmetric largest nearest-vertex distance, millimetres.
    distances_mm: tuple[tuple[str, float], ...]
    origins: tuple[str, str]
    sides_ok: bool
    jaw_difference_mm: "float | None" = None

    @property
    def worst_mm(self) -> float:
        return max(distance for _, distance in self.distances_mm)

    @property
    def exit_code(self) -> int:
        within = self.worst_mm <= self.tolerance_mm and self.origins[0] == self.origins[1] and self.sides_ok
        jaw = self.jaw_difference_mm is None or self.jaw_difference_mm <= max(self.tolerance_mm, 0.01)
        return 0 if within and jaw else 1

    def render(self) -> str:
        lines = [f"{self.a} against {self.b}: worst {self.worst_mm:.6g} mm, tolerance {self.tolerance_mm:g} mm, "
                 f"{'the same hand' if not self.exit_code else 'NOT the same hand'}"]
        lines += [f"  {part:<8} {distance:.6g} mm" for part, distance in self.distances_mm]
        lines.append(f"  origin   {self.origins[0]} / {self.origins[1]}")
        lines.append(f"  sides    {'left finger left of right' if self.sides_ok else 'the left finger is not on the left'}")
        if self.jaw_difference_mm is not None:
            lines.append(f"  jaw      largest difference {self.jaw_difference_mm:.6g} mm")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "tolerance_mm": self.tolerance_mm, "distances_mm": dict(self.distances_mm),
                "worst_mm": self.worst_mm, "origins": list(self.origins), "sides_ok": self.sides_ok,
                "jaw_difference_mm": self.jaw_difference_mm, "exit_code": self.exit_code}


def compare(a: str, b: str, *, tolerance_mm: float, jaw_centre_mm: "float | None" = None) -> BundleComparison:
    from scipy.spatial import cKDTree

    left, right = _arrays(bundle_path(a)), _arrays(bundle_path(b))
    distances = []
    for part in PARTS:
        pa = np.asarray(left[f"{part}__v"], dtype=np.float64)
        pb = np.asarray(right[f"{part}__v"], dtype=np.float64)
        worst = max(float(cKDTree(pb).query(pa)[0].max()), float(cKDTree(pa).query(pb)[0].max()))
        distances.append((part, worst))

    def origin(arrays: dict[str, np.ndarray]) -> str:
        return str(arrays["gripper__origin"].reshape(-1)[0]) if "gripper__origin" in arrays else "flange"

    sides = all(float(arrays["lfinger__v"][:, 0].mean()) < float(arrays["rfinger__v"][:, 0].mean())
                for arrays in (left, right))
    jaw = None
    if jaw_centre_mm is not None:
        measure = _module("_trial_measure_jaw", "scripts/grippers/measure_jaw_from_bundle.py")
        ma = measure.measure_parallel_jaw(bundle_path(a), centre_mm=jaw_centre_mm)
        mb = measure.measure_parallel_jaw(bundle_path(b), centre_mm=jaw_centre_mm)
        from dataclasses import fields

        numbers = [f.name for f in fields(ma) if isinstance(getattr(ma, f.name), float)]
        jaw = max(abs(getattr(ma, name) - getattr(mb, name)) for name in numbers)
    return BundleComparison(a=a, b=b, tolerance_mm=float(tolerance_mm), distances_mm=tuple(distances),
                            origins=(origin(left), origin(right)), sides_ok=sides, jaw_difference_mm=jaw)


def bounds_refusal(hand: str) -> "str | None":
    """Whether the written acme_dims bundle's bounds are the hand-computed ones, within a micrometre."""
    arrays = _arrays(bundle_path(hand))
    wrong = []
    for part, (low, high) in ACME_DIMS_BOUNDS.items():
        vertices = np.asarray(arrays[f"{part}__v"], dtype=np.float64)
        got_low, got_high = vertices.min(axis=0), vertices.max(axis=0)
        if np.abs(got_low - low).max() > 1e-3 or np.abs(got_high - high).max() > 1e-3:
            wrong.append(f"{part} spans {got_low.round(3).tolist()} to {got_high.round(3).tolist()}, "
                         f"by hand {list(low)} to {list(high)}")
    return "; ".join(wrong) or None


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Customer hands with known answers, for the customer chain trial.")
    verbs = parser.add_subparsers(dest="verb", required=True)
    registry = verbs.add_parser("registry", help="write acme_dims's registry file into a directory")
    registry.add_argument("--hand", default="acme_dims")
    registry.add_argument("--out", required=True)
    boxes = verbs.add_parser("boxes", help="print acme_dims's boxes, and exit 1 unless they are the hand-computed ones")
    boxes.add_argument("--inflation-mm", type=float, default=ACME_DIMS_INFLATION_MM)
    export = verbs.add_parser("export-stl", help="a committed bundle as STL files in the acme_mesh vendor frame")
    export.add_argument("--from", dest="source", required=True)
    export.add_argument("--out", required=True)
    comparison = verbs.add_parser("compare", help="two bundles, part by part")
    comparison.add_argument("a")
    comparison.add_argument("b")
    comparison.add_argument("--tolerance-mm", type=float, required=True)
    comparison.add_argument("--jaw-centre-mm", type=float, default=None)
    bounds = verbs.add_parser("bounds", help="the written acme_dims bundle against its hand-computed bounds")
    bounds.add_argument("--hand", default="acme_dims")
    args = parser.parse_args(argv)

    if args.verb == "registry":
        out = Path(args.out) / f"{args.hand}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(registry_text(args.hand), encoding="utf-8", newline="\n")
        print(f"wrote {out}")
        return 0
    if args.verb == "boxes":
        rendered = acme_dims_boxes(inflation_mm=args.inflation_mm)
        print("\n".join(rendered))
        return 0 if rendered == ACME_DIMS_BOXES else 1
    if args.verb == "export-stl":
        answer = export_stl(args.source, Path(args.out))
        print(json.dumps(answer, indent=1))
        return 0
    if args.verb == "compare":
        for name in (args.a, args.b):
            if not bundle_path(name).is_file():
                print(f"cannot ask: no bundle at {bundle_path(name)}", file=sys.stderr)
                return 2
        report = compare(args.a, args.b, tolerance_mm=args.tolerance_mm, jaw_centre_mm=args.jaw_centre_mm)
        print(report.render())
        return report.exit_code
    if not bundle_path(args.hand).is_file():
        print(f"cannot ask: no bundle at {bundle_path(args.hand)}", file=sys.stderr)
        return 2
    refused = bounds_refusal(args.hand)
    print(refused or f"{args.hand}: every part spans exactly the hand-computed bounds")
    return 0 if refused is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
