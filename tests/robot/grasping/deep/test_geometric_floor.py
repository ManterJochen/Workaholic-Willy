"""The training-free approach rules, which produced the sharpest number in the DL arc.

Worth pinning tightly, because the result they produced is one this project will act on: a per-object
constant reaches 0.8167 where the best global constant reaches 0.5355 and the trained head reaches
0.41. Every one of those is an `approach_hit` over a multi-hot admissible set, and each rule has a
different way of being quietly wrong: an oracle that does not actually peek understates the ceiling, a
per-object rule taken per BATCH understates it too, and a "straight down" bin hard-coded to index 0
would keep working right up until the spiral is reordered.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.robot.grasping.deep.corpus.grasp_encoding import approach_directions  # noqa: E402
from src.robot.grasping.deep.eval import (  # noqa: E402
    format_directions,
    format_geometric,
    object_direction_report,
    run_geometric_rules,
)

BINS = 8


def _batch(*, sets: list[list[list[int]]], normals: list[list[list[float]]] | None = None,
           bins: list[list[int]] | None = None) -> dict[str, torch.Tensor]:
    """One batch. `sets[item][point]` is the multi-hot of admissible bins for that point."""
    items, points = len(sets), len(sets[0])
    approach_set = torch.tensor(sets, dtype=torch.uint8)
    if normals is None:
        normals = [[[0.0, 0.0, 1.0]] * points for _ in range(items)]
    if bins is None:
        # The single-label bin is the FIRST admissible one, which is all these rules read it for.
        bins = [[int(np.argmax(row)) for row in item] for item in sets]
    features = torch.cat([torch.tensor(normals, dtype=torch.float32),
                          torch.zeros(items, points, 2)], dim=-1)
    return {
        "approach_set": approach_set,
        "approach_bin": torch.tensor(bins, dtype=torch.long),
        "supervise": torch.ones(items, points, dtype=torch.bool),
        "features": features,
        "points_m": torch.zeros(items, points, 3),
    }


def _by_name(rules) -> dict[str, float]:
    return {rule.name: rule.hit for rule in rules}


def test_the_constant_is_chosen_on_the_training_side() -> None:
    """⚠ THE HONEST CONSTANT MUST COME FROM THE TRAIN SPLIT. Choosing it on the test set is not a
    floor, it is a second model fitted to the answer, and `global_oracle` exists to show the gap."""
    # Training says bin 1 is most common; the held-out set actually prefers bin 2.
    train = _batch(sets=[[[0, 1, 0, 0, 0, 0, 0, 0]] * 4])
    test = _batch(sets=[[[0, 0, 1, 0, 0, 0, 0, 0]] * 4])
    hits = _by_name(run_geometric_rules([train], [test], approach_bins=BINS))
    assert hits["constant"] == pytest.approx(0.0)     # bin 1, never admissible on the test rows
    assert hits["global_oracle"] == pytest.approx(1.0)  # bin 2, chosen with hindsight


def test_the_object_oracle_is_taken_per_sample_not_per_batch() -> None:
    """⭑ THE MEASUREMENT THIS MODULE EXISTS FOR. Two objects in ONE batch, each wanting a different
    direction. Per object both are satisfied and the ceiling is 1.0; a per-batch constant would
    satisfy only one and report 0.5, understating exactly the headroom being measured."""
    test = _batch(sets=[
        [[1, 0, 0, 0, 0, 0, 0, 0]] * 4,
        [[0, 0, 0, 0, 1, 0, 0, 0]] * 4,
    ])
    hits = _by_name(run_geometric_rules([test], [test], approach_bins=BINS))
    assert hits["object_oracle"] == pytest.approx(1.0)
    assert hits["global_oracle"] == pytest.approx(0.5)


def test_the_object_oracle_never_beats_a_perfect_per_point_answer() -> None:
    """A per-object CONSTANT is one direction for the whole object, so on an object whose points
    genuinely disagree it cannot reach 1.0. That bound is the whole reason it is a useful ceiling."""
    test = _batch(sets=[[
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ]])
    hits = _by_name(run_geometric_rules([test], [test], approach_bins=BINS))
    assert hits["object_oracle"] == pytest.approx(2 / 3)


def test_the_down_bin_is_found_rather_than_hard_coded() -> None:
    """`down` must be the bin nearest (0, 0, -1) in whatever order the spiral produces.

    The construction happens to put its first sample nearest the pole today. A rule that assumed
    index 0 would keep passing until someone reorders `approach_directions`, and then it would be
    wrong everywhere with nothing reporting it.
    """
    directions = approach_directions(BINS, 135.0)
    expected = int(np.argmax(directions @ np.array([0.0, 0.0, -1.0])))
    sets = [[[1 if b == expected else 0 for b in range(BINS)]] * 3]
    hits = _by_name(run_geometric_rules([_batch(sets=sets)], [_batch(sets=sets)],
                                        approach_bins=BINS))
    assert hits["down"] == pytest.approx(1.0)


def test_the_normal_rule_reads_the_first_three_feature_channels() -> None:
    """The normals are features 0 to 2, exactly as `build_sample` stacks them. Reading the wrong
    slice would produce a number that looks like `the normal carries nothing`, which is the very
    conclusion this rule was built to test."""
    directions = approach_directions(BINS, 135.0)
    target = 3
    # A normal pointing exactly at bin `target`: the honest `normal` rule must pick it.
    normal = [float(x) for x in directions[target]]
    sets = [[[1 if b == target else 0 for b in range(BINS)]] * 3]
    batch = _batch(sets=sets, normals=[[normal] * 3])
    hits = _by_name(run_geometric_rules([batch], [batch], approach_bins=BINS))
    assert hits["normal"] == pytest.approx(1.0)
    assert hits["normal_inward"] == pytest.approx(0.0)


def test_unsupervised_and_unlabelled_points_are_excluded() -> None:
    """A point with no approach label has no correct answer, so scoring it would invent supervision.
    Here two of four points are excluded, one by `supervise` and one by a negative bin."""
    batch = _batch(sets=[[
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ]])
    batch["supervise"] = torch.tensor([[True, True, False, True]])
    batch["approach_bin"] = torch.tensor([[0, 0, 1, -1]])
    rules = run_geometric_rules([batch], [batch], approach_bins=BINS)
    assert rules[0].supervised_points == 2
    assert _by_name(rules)["global_oracle"] == pytest.approx(1.0)


def test_an_empty_held_out_set_returns_nothing_rather_than_dividing_by_zero() -> None:
    batch = _batch(sets=[[[1, 0, 0, 0, 0, 0, 0, 0]] * 2])
    batch["supervise"] = torch.zeros(1, 2, dtype=torch.bool)
    assert run_geometric_rules([batch], [batch], approach_bins=BINS) == []
    assert format_geometric([]) == "no rule measured"


def test_the_report_marks_oracles_and_stays_ascii() -> None:
    """⛔ Two failures in one test, because both have happened. An oracle printed without its mark
    becomes a reported result; a non-ASCII character raised UnicodeEncodeError on a cp1252 console
    and killed a finished run."""
    batch = _batch(sets=[[[1, 0, 0, 0, 0, 0, 0, 0]] * 3])
    rules = run_geometric_rules([batch], [batch], approach_bins=BINS, seed=0)
    text = format_geometric(rules, reference={"base": 0.4})
    text.encode("ascii")
    assert "ORACLE" in text
    assert "object_oracle" in text
    assert "must not be compared against a trained arm" in text
    assert "base" in text


def _down_bin() -> int:
    directions = approach_directions(BINS, 135.0)
    return int(np.argmax(directions @ np.array([0.0, 0.0, -1.0])))


def test_an_object_that_wants_down_is_counted_at_zero_tilt() -> None:
    """The tilt is measured from straight down, so an object whose best bin IS down must read 0."""
    down = _down_bin()
    sets = [[[1 if b == down else 0 for b in range(BINS)]] * 3]
    report = object_direction_report([_batch(sets=sets)], approach_bins=BINS)
    assert report is not None
    assert report.median_deg == pytest.approx(0.0, abs=1e-6)
    assert report.share_at_down == pytest.approx(1.0)
    assert report.within_10_deg == pytest.approx(1.0)


def test_the_tilt_is_the_real_angle_between_the_two_directions() -> None:
    """Not a bin-index distance. The spiral is not ordered by angle, so an index difference would be
    a plausible-looking number with no geometric meaning."""
    directions = approach_directions(BINS, 135.0)
    down = _down_bin()
    other = (down + 3) % BINS
    expected = float(np.degrees(np.arccos(np.clip(
        float(directions[other] @ directions[down]), -1.0, 1.0))))
    sets = [[[1 if b == other else 0 for b in range(BINS)]] * 3]
    report = object_direction_report([_batch(sets=sets)], approach_bins=BINS)
    assert report is not None
    assert report.median_deg == pytest.approx(expected, abs=1e-4)
    assert report.share_at_down == pytest.approx(0.0)


def test_the_gain_is_measured_only_on_objects_that_want_something_else() -> None:
    """⭑ THE DECOMPOSITION THIS REPORT EXISTS FOR. One object wants down, one wants elsewhere. The
    gain must describe ONLY the second: on the first, knowing the object's direction buys nothing,
    and averaging it in would dilute the very number that decides whether to build a per-object head.
    """
    down = _down_bin()
    other = (down + 3) % BINS
    wants_down = [[1 if b == down else 0 for b in range(BINS)]] * 4
    wants_other = [[1 if b == other else 0 for b in range(BINS)]] * 4
    report = object_direction_report([_batch(sets=[wants_down, wants_other])], approach_bins=BINS)
    assert report is not None
    assert report.share_at_down == pytest.approx(0.5)
    assert report.hit_objects_at_down == pytest.approx(1.0)
    assert report.hit_objects_off_down == pytest.approx(1.0)
    # `down` is never admissible on the second object, so it scores 0 there and the gain is the full
    # difference. Averaging the first object in would have halved it.
    assert report.down_hit_objects_off_down == pytest.approx(0.0)
    assert report.gain_where_off_down == pytest.approx(1.0)


def test_the_hit_columns_are_weighted_by_supervised_points() -> None:
    """Objects carry very different point counts, so an unweighted mean would let a two-point object
    count as much as a two-thousand-point one. The report's own text warns against combining the
    weighted hit columns with the unweighted object share; the weighting is what makes that true."""
    down = _down_bin()
    other = (down + 3) % BINS
    # Object A: 4 supervised points, all satisfied by its own bin. Object B: 1 supervised point.
    batch = _batch(sets=[[[1 if b == down else 0 for b in range(BINS)]] * 4,
                         [[1 if b == other else 0 for b in range(BINS)]] * 4])
    batch["supervise"] = torch.tensor([[True] * 4, [True, False, False, False]])
    report = object_direction_report([batch], approach_bins=BINS)
    assert report is not None
    assert report.objects == 2
    assert report.share_at_down == pytest.approx(0.5)   # objects, unweighted
    assert report.hit_all == pytest.approx(1.0)          # points, weighted 4 to 1


def test_an_object_whose_points_disagree_bounds_its_own_oracle() -> None:
    """A per-object CONSTANT cannot satisfy points that want different things, and the report has to
    show that rather than reporting the best bin as if it always worked."""
    down = _down_bin()
    other = (down + 3) % BINS
    rows = [[1 if b == down else 0 for b in range(BINS)],
            [1 if b == down else 0 for b in range(BINS)],
            [1 if b == other else 0 for b in range(BINS)]]
    report = object_direction_report([_batch(sets=[rows])], approach_bins=BINS)
    assert report is not None
    assert report.hit_all == pytest.approx(2 / 3)


def test_the_direction_report_is_ascii_and_survives_an_empty_input() -> None:
    down = _down_bin()
    sets = [[[1 if b == down else 0 for b in range(BINS)]] * 3]
    report = object_direction_report([_batch(sets=sets)], approach_bins=BINS)
    assert report is not None
    format_directions([report]).encode("ascii")
    assert format_directions([]) == "no object measured"

    empty = _batch(sets=sets)
    empty["supervise"] = torch.zeros(1, 3, dtype=torch.bool)
    assert object_direction_report([empty], approach_bins=BINS) is None
