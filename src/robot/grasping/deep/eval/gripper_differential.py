"""Does a set generator's output actually depend on the hand it was told to grasp with?

The training run cannot answer this. An arm trained on a multi-gripper corpus reports one loss curve
and one held-out rate. Neither says whether the 14-number conditioning vector reached the head: a
network that ignores it entirely still trains, still converges, and still scores, because its corpus
is a mixture and the mixture's average is a perfectly learnable target. The only way to find out is
to hold the cloud fixed and vary the hand.

The control is not optional and it comes first. The same gripper twice must produce identical
proposals. Seeds are drawn from a generator and a stage-3 net crops the cloud, so "the two runs
differ" is a claim about the gripper only after it has been shown not to be a claim about the
sampler. Without that line, a nondeterministic pipeline reads as a conditioned model.

A pair that isolates one dimension is not the same as a pair the model has seen, and the only
single-axis pair in `JAW_GEOMETRY` is out of distribution. Over the registry, `2f85` and `slim_pad`
are identical in all eight numbers except `pad_span_mm` (38.0 against 19.01), which makes them the
sharpest possible test of one axis; no other pair differs in fewer than four. An arm trained on
`narrow_55 / slim_pad / wide_140` has never seen `2f85`, so a null result there is ambiguous: it may
mean the head ignores pad span, or it may mean the vector is off the manifold the head was fitted
on. Both pairs are therefore reported, and which is which is stated rather than implied. The seen
pair is the primary evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover (imported for annotations only)
    import numpy as np

__all__ = ["GripperPair", "PairDivergence", "DifferentialReport", "measure_gripper_differential"]

_NEWLINE = chr(10)

#: Below this, two proposal sets are the same answer. It is a position tolerance in millimetres and
#: it is deliberately small: the question is whether the head responds at all, not whether it
#: responds usefully, so anything a decoder could not have produced by rounding counts as a response.
_SAME_ANSWER_MM = 0.5


@dataclass(frozen=True, slots=True)
class GripperPair:
    """Two hands, and what a reader needs to know before believing the number beside them."""

    left: str
    right: str
    #: Both names were in the corpus this artifact trained on.
    both_seen: bool
    #: How many of the eight geometry numbers differ. 1 is the sharpest possible test of one axis.
    differing_dimensions: int
    note: str = ""


@dataclass(frozen=True, slots=True)
class PairDivergence:
    """How far apart the same cloud's proposals landed under two hands."""

    pair: GripperPair
    scenes: int
    median_position_mm: float
    max_position_mm: float
    median_width_mm: float
    max_width_mm: float

    @property
    def responded(self) -> bool:
        return self.median_position_mm > _SAME_ANSWER_MM or self.median_width_mm > _SAME_ANSWER_MM

    def render(self) -> str:
        seen = "SEEN" if self.pair.both_seen else "one UNSEEN"
        verdict = "responds" if self.responded else "identical: the vector did not reach the head"
        return (f"  {self.pair.left:>10s} vs {self.pair.right:<10s} "
                f"[{seen}, {self.pair.differing_dimensions} dim] "
                f"position {self.median_position_mm:6.2f} mm median / {self.max_position_mm:6.2f} max"
                f"   width {self.median_width_mm:6.2f} / {self.max_width_mm:6.2f}   {verdict}")


@dataclass(frozen=True, slots=True)
class DifferentialReport:
    """The control first, then the pairs, because the pairs mean nothing without it."""

    artifact: Path
    trained_grippers: tuple[str, ...]
    control_max_mm: float
    pairs: tuple[PairDivergence, ...]

    @property
    def control_holds(self) -> bool:
        """The same hand twice must give the same answer. If it does not, every number below is a
        measurement of the sampler and none of them is about the gripper."""
        return self.control_max_mm <= _SAME_ANSWER_MM

    @property
    def seen_pairs(self) -> tuple[PairDivergence, ...]:
        return tuple(p for p in self.pairs if p.pair.both_seen)

    @property
    def verdict(self) -> str:
        """Three outcomes, not two.

        A boolean answer renders "this model's output does not depend on the hand" whenever no seen
        pair existed to test it, which is absence of evidence reported as evidence of absence. An
        artifact stamped with one gripper makes `candidate_pairs` produce none, so untested has to
        be an outcome of its own beside dead.
        """
        if not self.control_holds:
            return ("INCONCLUSIVE: the same hand twice gave different answers, so nothing here is "
                    "about the gripper")
        if not self.seen_pairs:
            return ("UNTESTED: this artifact names " + ", ".join(self.trained_grippers)
                    + ", which yields no pair of hands it has both seen. Nothing below is evidence "
                      "about the grippers it was trained on. Pass `trained=` if the artifact lost "
                      "them")
        if all(p.responded for p in self.seen_pairs):
            return "LIVE: on hands it was trained on, the output depends on the gripper"
        return ("DEAD: on hands it was TRAINED on, this model's output does not depend on the "
                "gripper")

    @property
    def conditioning_is_live(self) -> bool:
        """False also when untested, so never read this alone. :attr:`verdict` is the answer."""
        return self.control_holds and bool(self.seen_pairs) and all(
            p.responded for p in self.seen_pairs)

    def render(self) -> str:
        lines = [f"gripper differential on {self.artifact}",
                 f"  artifact names: {', '.join(self.trained_grippers)}",
                 f"  CONTROL (same hand twice): max {self.control_max_mm:.4f} mm  "
                 + ("ok" if self.control_holds
                    else "failed: the pipeline is not deterministic, so nothing below is about "
                         "the gripper")]
        lines.extend(p.render() for p in self.pairs)
        lines.append(f"  => {self.verdict}")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"artifact": str(self.artifact),
                "trained_grippers": list(self.trained_grippers),
                "control_max_mm": self.control_max_mm,
                "control_holds": self.control_holds,
                "conditioning_is_live": self.conditioning_is_live,
                "verdict": self.verdict,
                "pairs": [{"left": p.pair.left, "right": p.pair.right,
                           "both_seen": p.pair.both_seen,
                           "differing_dimensions": p.pair.differing_dimensions,
                           "scenes": p.scenes,
                           "median_position_mm": p.median_position_mm,
                           "max_position_mm": p.max_position_mm,
                           "median_width_mm": p.median_width_mm,
                           "max_width_mm": p.max_width_mm,
                           "responded": p.responded} for p in self.pairs]}


def candidate_pairs(trained: Sequence[str]) -> list[GripperPair]:
    """The pairs worth running, widest-seen first, then the single-axis one.

    Ordered by evidential weight, not by sharpness. The widest pair the model actually saw is the
    primary test; the single-axis pair is sharper and weaker at the same time.
    """
    from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY  # noqa: PLC0415

    def differing(a: str, b: str) -> int:
        return sum(1 for key in JAW_GEOMETRY[a] if JAW_GEOMETRY[a][key] != JAW_GEOMETRY[b][key])

    seen = [g for g in trained if g in JAW_GEOMETRY]
    pairs: list[GripperPair] = []
    if len(seen) >= 2:
        widest = max(((a, b) for i, a in enumerate(seen) for b in seen[i + 1:]),
                     key=lambda ab: abs(JAW_GEOMETRY[ab[0]]["aperture_mm"]
                                        - JAW_GEOMETRY[ab[1]]["aperture_mm"]))
        pairs.append(GripperPair(
            left=widest[0], right=widest[1], both_seen=True,
            differing_dimensions=differing(*widest),
            note="the widest separation this artifact was trained across; a null here means the "
                 "conditioning is dead"))
    for a in JAW_GEOMETRY:
        for b in JAW_GEOMETRY:
            if a < b and differing(a, b) == 1:
                pairs.append(GripperPair(
                    left=a, right=b, both_seen=a in seen and b in seen,
                    differing_dimensions=1,
                    note="the only pair in the registry that isolates ONE dimension"))
    return pairs


def measure_gripper_differential(
    artifact: str | Path, files: Sequence[str | Path], *, seeds: int = 64,
    device: str | None = None, seed: int = 0, trained: Sequence[str] | None = None,
    report: Any = None) -> DifferentialReport:
    """Hold the cloud fixed, vary the hand, and measure whether the proposals move.

    The encoder runs once per scene. It never sees the gripper: only `net.propose` does. Re-encoding
    per hand would add the encoder's own nondeterminism to a difference that is supposed to be the
    head's, and would make the control weaker for no gain.
    """
    import dataclasses  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from src.robot.grasping.deep.corpus.sample import (  # noqa: PLC0415
        build_sample, load_scene,
    )
    from src.robot.grasping.deep.eval.propose import serving_shares  # noqa: PLC0415
    from src.robot.grasping.deep.net.gripper import gripper_vector  # noqa: PLC0415
    from src.robot.grasping.deep.net.set_targets import sample_seeds  # noqa: PLC0415
    from src.robot.grasping.deep.set_artifact import load_set_generator  # noqa: PLC0415

    say = report if report is not None else (lambda _line: None)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_set_generator(artifact, device=device)
    net = loaded.net.eval()
    # A current artifact carries every hand it trained across in `trained_grippers`, which
    # `loaded.grippers` reads back. An artifact written before that field existed carries the stamp
    # alone: a run whose `epochs.json` records `['narrow_55', 'slim_pad', 'wide_140']` is stamped
    # `slim_pad`, so a model fitted across three hands presents as belonging to one. `trained=` is
    # the caller's way to supply what such a file lost; it does not repair the artifact and is not
    # meant to. The runtime conditions on the cell's configured hand, checked against
    # `loaded.grippers`, and falls back to the stamp only when the cell configures no hand.
    stamped = tuple(getattr(loaded, "grippers", None) or (loaded.gripper,))
    known = tuple(trained) if trained is not None else stamped
    pairs = candidate_pairs(known)
    if trained is not None and set(known) != set(stamped):
        say(f"  artifact is stamped {', '.join(stamped)}; the caller says it trained on "
            f"{', '.join(known)}")
    say(f"  {len(pairs)} pair(s) to run")

    # The augmentation is off, for the reason `propose.py` records at length: `rotate_z` defaults
    # to True as a training device, and nothing rotates back at inference. Here it would also make
    # the control fail for a reason that has nothing to do with the gripper.
    spec = dataclasses.replace(loaded.sample, rotate_z=False)
    names = sorted({p.left for p in pairs} | {p.right for p in pairs})
    vectors = {name: gripper_vector(name).to(device) for name in names}

    # position and width per (gripper, scene), each an array over the seeds
    positions: dict[str, list[np.ndarray]] = {name: [] for name in names}
    widths: dict[str, list[np.ndarray]] = {name: [] for name in names}
    control: list[float] = []

    for path in files:
        scene = load_scene(path)
        sample = build_sample(scene, np.random.default_rng(seed), spec, target_instance=None)
        cloud = torch.as_tensor(np.asarray(sample["points_m"])[None], dtype=torch.float32,
                                device=device)
        features = torch.as_tensor(np.asarray(sample["features"])[None], dtype=torch.float32,
                                   device=device)[..., :net.backbone.config.in_features]
        with torch.no_grad():
            encoded = net.encode(cloud, features)
            field = net.graspability_logit(encoded)[0]
            candidate = torch.as_tensor(np.asarray(sample["supervise"]), dtype=torch.bool)
            if not bool(candidate.any()):
                candidate = torch.ones(cloud.shape[1], dtype=torch.bool)
            # One draw, reused for every hand. A fresh draw per gripper would put the sampler's
            # variance inside the difference this function exists to measure.
            draw = sample_seeds(count=seeds,
                                labelled=torch.zeros(cloud.shape[1], dtype=torch.bool),
                                score=field.detach().cpu(), candidate=candidate,
                                shares=serving_shares(loaded.step.shares),
                                generator=torch.Generator().manual_seed(seed))
            picks = draw.point_index.to(device)

            def _propose(vector: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
                """The offset, not a decoded pose. `offset_m` is the head's own output; decoding
                adds the sample frame back, which is identical for both hands and would only dilute
                the difference with a shared constant."""
                prediction = net.propose(encoded, torch.zeros_like(picks), picks,
                                         vector.expand(len(picks), vector.shape[-1]), cloud)
                return (prediction.offset_m.detach().cpu().numpy().reshape(len(picks), -1) * 1000.0,
                        prediction.width_m.detach().cpu().numpy().reshape(-1) * 1000.0)

            for name in names:
                pos, wid = _propose(vectors[name])
                positions[name].append(pos)
                widths[name].append(wid)
            # The control: the first hand, run a second time.
            again_pos, again_wid = _propose(vectors[names[0]])
            control.append(float(np.abs(again_pos - positions[names[0]][-1]).max()))
            control.append(float(np.abs(again_wid - widths[names[0]][-1]).max()))

    def _diverge(left: str, right: str) -> PairDivergence:
        per_scene_pos = [float(np.median(np.linalg.norm(a - b, axis=-1)))
                         for a, b in zip(positions[left], positions[right])]
        per_scene_wid = [float(np.median(np.abs(a - b)))
                         for a, b in zip(widths[left], widths[right])]
        pair = next(p for p in pairs if p.left == left and p.right == right)
        return PairDivergence(
            pair=pair, scenes=len(per_scene_pos),
            median_position_mm=float(np.median(per_scene_pos)) if per_scene_pos else 0.0,
            max_position_mm=float(np.max(per_scene_pos)) if per_scene_pos else 0.0,
            median_width_mm=float(np.median(per_scene_wid)) if per_scene_wid else 0.0,
            max_width_mm=float(np.max(per_scene_wid)) if per_scene_wid else 0.0)

    return DifferentialReport(
        artifact=Path(artifact), trained_grippers=known,
        control_max_mm=max(control) if control else 0.0,
        pairs=tuple(_diverge(p.left, p.right) for p in pairs))
