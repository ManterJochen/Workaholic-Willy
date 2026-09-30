"""Which two links overlap, and by how much, derived from the config the planner loaded (B1, S14).

A cuRobo refusal says "no collision-free plan". The operator's next question is always the same: what touched what. The
planner knows it only as a number over a flat array of spheres, so the answer has to come from sphere OWNERSHIP in the
loaded descriptor: ``collision_link_names`` in order, the count per link from ``collision_spheres``, the spare slots
from ``extra_collision_spheres``, the padding from ``self_collision_buffer``, and the pairs ``self_collision_ignore``
takes out.

Pure arithmetic on plain data, so the python 3.11 suite tests exactly what the python 3.10 sidecar runs. What it cannot
prove here is that cuRobo lays its ``robot_spheres`` out in that order: only the box can, by naming a pair and its depth
where a probe measured the same one.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _pairs():
    path = _ROOT / "src" / "robot" / "safety" / "planning" / "_curobo_pairs.py"
    spec = importlib.util.spec_from_file_location("_curobo_pairs_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(*, buffer: float = 0.0, ignore: "dict | None" = None, links: "list[str] | None" = None) -> dict:
    """Two links, one sphere each, 12 mm inside one another: radii 0.05 m at 0.088 m apart."""
    return {
        "robot_cfg": {"kinematics": {
            "collision_link_names": list(links or ["wrist_1_link", "tool0"]),
            "collision_spheres": {
                "wrist_1_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}],
                "tool0": [{"center": [0.088, 0.0, 0.0], "radius": 0.05}],
            },
            "self_collision_buffer": {"wrist_1_link": buffer, "tool0": buffer},
            "self_collision_ignore": dict(ignore or {}),
        }},
    }


def _spheres(*centres_and_radii: "tuple[list[float], float]") -> np.ndarray:
    """One pose, one row per sphere: ``(1, S, 4)`` in metres, as cuRobo hands them over."""
    return np.asarray([[[*centre, radius] for centre, radius in centres_and_radii]], dtype=np.float64)


_TOUCHING = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], 0.05))


class TheDeepestPairIsNamedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _pairs()

    def test_two_overlapping_links_are_named_with_their_depth(self) -> None:
        layout = self.module.SphereLayout.from_robot_config(_config())
        (deepest,) = self.module.deepest_pairs(_TOUCHING, layout)
        self.assertEqual((deepest.link_a, deepest.link_b), ("tool0", "wrist_1_link"))
        self.assertAlmostEqual(deepest.depth_mm, 12.0, places=6)
        self.assertIn("12.0", deepest.render())
        self.assertEqual(deepest.to_dict()["depth_mm"], deepest.depth_mm)

    def test_the_guard_margin_in_the_buffers_deepens_it(self) -> None:
        """cuRobo subtracts both links' buffers from a pair's distance, so 5 mm each reads as 10 mm more overlap."""
        layout = self.module.SphereLayout.from_robot_config(_config(buffer=0.005))
        (deepest,) = self.module.deepest_pairs(_TOUCHING, layout)
        self.assertAlmostEqual(deepest.depth_mm, 22.0, places=6)

    def test_an_ignored_pair_a_single_link_and_an_empty_slot_are_not_collisions(self) -> None:
        module = self.module
        ignored = module.SphereLayout.from_robot_config(_config(ignore={"wrist_1_link": ["tool0"]}))
        self.assertEqual(module.deepest_pairs(_TOUCHING, ignored), (None,))

        one_link = module.SphereLayout.from_robot_config({
            "robot_cfg": {"kinematics": {
                "collision_link_names": ["wrist_1_link"],
                "collision_spheres": {"wrist_1_link": [{"center": [0, 0, 0], "radius": 0.05},
                                                       {"center": [0.088, 0, 0], "radius": 0.05}]},
            }},
        })
        self.assertEqual(module.deepest_pairs(_TOUCHING, one_link), (None,))

        empty = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], -100.0))
        self.assertEqual(module.deepest_pairs(empty, module.SphereLayout.from_robot_config(_config())), (None,))

    def test_the_owner_of_a_slot_follows_the_link_order(self) -> None:
        """⭐ THE CONTROL. The array is flat: only ``collision_link_names``, read in order and counted per link,
        says whose sphere slot 0 is. Here ``wrist_1_link`` owns two slots and ``tool0`` one, and the same three
        spheres are a single link's own business under one order and a real collision under the other, so an
        implementation that ignored the order could not answer both."""
        module = self.module
        two_and_one = {
            "collision_spheres": {
                "wrist_1_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.05},
                                 {"center": [0.088, 0.0, 0.0], "radius": 0.05}],
                "tool0": [{"center": [0.4, 0.0, 0.0], "radius": 0.05}],
            },
            "self_collision_ignore": {},
        }
        spheres = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], 0.05), ([0.4, 0.0, 0.0], 0.05))

        straight = module.SphereLayout.from_robot_config(
            {"robot_cfg": {"kinematics": {"collision_link_names": ["wrist_1_link", "tool0"], **two_and_one}}})
        self.assertEqual(straight.owners, ("wrist_1_link", "wrist_1_link", "tool0"))
        self.assertEqual(module.deepest_pairs(spheres, straight), (None,))

        swapped = module.SphereLayout.from_robot_config(
            {"robot_cfg": {"kinematics": {"collision_link_names": ["tool0", "wrist_1_link"], **two_and_one}}})
        self.assertEqual(swapped.owners, ("tool0", "wrist_1_link", "wrist_1_link"))
        (deepest,) = module.deepest_pairs(spheres, swapped)
        assert deepest is not None
        self.assertEqual((deepest.link_a, deepest.link_b), ("tool0", "wrist_1_link"))
        self.assertAlmostEqual(deepest.depth_mm, 12.0, places=6)

    def test_spare_slots_are_counted_so_a_payload_does_not_shift_every_owner(self) -> None:
        config = _config()
        config["robot_cfg"]["kinematics"]["collision_link_names"].append("attached_object")
        config["robot_cfg"]["kinematics"]["extra_collision_spheres"] = {"attached_object": 4}
        layout = self.module.SphereLayout.from_robot_config(config)
        self.assertEqual(layout.owners, ("wrist_1_link", "tool0", "attached_object", "attached_object",
                                         "attached_object", "attached_object"))
        self.assertEqual(layout.slots, 6)

    def test_a_descriptor_whose_spheres_are_a_file_has_no_layout_and_does_not_refuse(self) -> None:
        """A stock cuRobo descriptor resolves ``collision_spheres`` from a file as it loads, and the ownership is gone
        by the time anything here sees it. That costs the operator a pair NAME. It must not cost them a planner, so
        there is no layout and no exception."""
        config = _config()
        config["robot_cfg"]["kinematics"]["collision_spheres"] = "spheres/ur5e.yml"
        self.assertIsNone(self.module.SphereLayout.from_robot_config(config))
        self.assertIsNone(self.module.SphereLayout.from_robot_config({"robot_cfg": {}}))
        self.assertIsNone(self.module.SphereLayout.from_robot_config(
            {"robot_cfg": {"kinematics": {"collision_link_names": [], "collision_spheres": {}}}}))

    def test_a_sphere_array_the_layout_does_not_describe_is_refused(self) -> None:
        """⭐ THE CONTROL that a wrong layout cannot be papered over: cuRobo's array and the owners must agree in
        length, or every name after the mismatch is somebody else's link."""
        layout = self.module.SphereLayout.from_robot_config(_config())
        with self.assertRaises(self.module.SphereLayoutError):
            self.module.deepest_pairs(_spheres(([0.0, 0.0, 0.0], 0.05)), layout)

    def test_only_the_deepest_pair_of_a_pose_is_reported(self) -> None:
        """Three links overlap at once. The operator gets the pair that decides the refusal, not a list."""
        config = _config()
        config["robot_cfg"]["kinematics"]["collision_link_names"] = ["wrist_1_link", "tool0", "wrist_3_link"]
        config["robot_cfg"]["kinematics"]["collision_spheres"]["wrist_3_link"] = [
            {"center": [0.0, 0.0, 0.06], "radius": 0.05},
        ]
        layout = self.module.SphereLayout.from_robot_config(config)
        # wrist_1 and tool0 overlap by 12 mm, wrist_1 and wrist_3 by 40 mm, tool0 and wrist_3 not at all.
        spheres = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], 0.05), ([0.0, 0.0, 0.06], 0.05))
        (deepest,) = self.module.deepest_pairs(spheres, layout)
        assert deepest is not None
        self.assertEqual((deepest.link_a, deepest.link_b), ("wrist_1_link", "wrist_3_link"))
        self.assertAlmostEqual(deepest.depth_mm, 40.0, places=6)

    def test_one_verdict_per_pose(self) -> None:
        layout = self.module.SphereLayout.from_robot_config(_config())
        clear = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.5, 0.0, 0.0], 0.05))
        poses = np.concatenate([_TOUCHING, clear], axis=0)
        verdicts = self.module.deepest_pairs(poses, layout)
        self.assertEqual(len(verdicts), 2)
        self.assertIsNotNone(verdicts[0])
        self.assertIsNone(verdicts[1])


def _two_links(buffer_m: float, *, a: str = "forearm_link", b: str = "wrist_2_link") -> dict:
    """Two links, one sphere each, both padded by ``buffer_m``: the planner's forearm and wrist_2 at LOOK[0]."""
    return {"robot_cfg": {"kinematics": {
        "collision_link_names": [a, b],
        "collision_spheres": {a: [{"center": [0.0, 0.0, 0.0], "radius": 0.01571}],
                              b: [{"center": [0.03623, 0.0, 0.0], "radius": 0.01774}]},
        "self_collision_buffer": {a: buffer_m, b: buffer_m},
        "self_collision_ignore": {},
    }}}


#: The deepest sphere pair of forearm_link and wrist_2_link at the owner's LOOK[0] (2026-09-30 research): radii
#: 15.71 and 17.74 mm, centres 36.23 mm apart.
_LOOK0_PAIR = _spheres(([0.0, 0.0, 0.0], 0.01571), ([0.03623, 0.0, 0.0], 0.01774))


def _brute_force(spheres: np.ndarray, layout, within_mm: float) -> list:
    """Every link pair per pose, the long way: each sphere pair on its own, the kernel's validity rule, the max."""
    out = []
    for pose in np.asarray(spheres, dtype=np.float64):
        best: dict = {}
        for i in range(layout.slots):
            for j in range(i + 1, layout.slots):
                a, b = layout.owners[i], layout.owners[j]
                if a == b or (min(a, b), max(a, b)) in layout.ignored:
                    continue
                if pose[i, 3] + layout.pads_m[i] < 0.0 or pose[j, 3] + layout.pads_m[j] < 0.0:
                    continue
                gap = float(np.linalg.norm(pose[i, :3] - pose[j, :3]))
                depth = (pose[i, 3] + pose[j, 3] + layout.pads_m[i] + layout.pads_m[j] - gap) * 1000.0
                key = (min(a, b), max(a, b))
                best[key] = max(best.get(key, -np.inf), depth)
        out.append({key: depth for key, depth in best.items() if depth > -within_mm})
    return out


class EveryOverlappingPairIsNamedTests(unittest.TestCase):
    """F1 (the owner, 2026-09-30): the sidecar reports EVERY colliding self pair of a refused sample, with the depth
    the planner judged (its padding in) and the depth of the spheres alone, so the driver can leave each pair the exact
    guard judges to that guard. One missing pair would be a pair nobody judged, so the list is the kernel's: every
    pair of two different links not ignored, each sphere counted where its padded radius is not negative."""

    def setUp(self) -> None:
        self.module = _pairs()

    def test_look0_overlaps_by_1_2_mm_with_the_margin_and_is_2_8_mm_apart_without(self) -> None:
        layout = self.module.SphereLayout.from_robot_config(_two_links(0.002))
        (pairs,) = self.module.overlapping_pairs(_LOOK0_PAIR, layout)
        (pair,) = pairs
        self.assertEqual((pair.link_a, pair.link_b), ("forearm_link", "wrist_2_link"))
        self.assertAlmostEqual(pair.depth_mm, 1.22, places=6)
        self.assertAlmostEqual(pair.unpadded_depth_mm, -2.78, places=6)
        self.assertEqual(pair.to_row(), ["forearm_link", "wrist_2_link", pair.depth_mm, pair.unpadded_depth_mm])

    def test_the_same_spheres_without_the_margin_are_no_pair(self) -> None:
        """⭐ THE CONTROL: without the 2 + 2 mm the two spheres are 2.78 mm apart, and nothing is named."""
        layout = self.module.SphereLayout.from_robot_config(_two_links(0.0))
        self.assertEqual(self.module.overlapping_pairs(_LOOK0_PAIR, layout), ((),))

    def test_every_pair_is_listed_deepest_first_not_only_the_deepest(self) -> None:
        config = _config()
        config["robot_cfg"]["kinematics"]["collision_link_names"] = ["wrist_1_link", "tool0", "wrist_3_link"]
        config["robot_cfg"]["kinematics"]["collision_spheres"]["wrist_3_link"] = [
            {"center": [0.0, 0.0, 0.06], "radius": 0.05},
        ]
        layout = self.module.SphereLayout.from_robot_config(config)
        spheres = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], 0.05), ([0.0, 0.0, 0.06], 0.05))
        (pairs,) = self.module.overlapping_pairs(spheres, layout)
        self.assertEqual([(p.link_a, p.link_b) for p in pairs], [("wrist_1_link", "wrist_3_link"),
                                                                 ("tool0", "wrist_1_link")])
        self.assertAlmostEqual(pairs[0].depth_mm, 40.0, places=6)
        self.assertAlmostEqual(pairs[1].depth_mm, 12.0, places=6)

    def test_an_ignored_pair_a_single_link_and_an_empty_slot_are_never_named(self) -> None:
        module = self.module
        ignored = module.SphereLayout.from_robot_config(_config(ignore={"tool0": ["wrist_1_link"]}))
        self.assertEqual(module.overlapping_pairs(_TOUCHING, ignored), ((),))
        empty = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], -100.0))
        self.assertEqual(module.overlapping_pairs(empty, module.SphereLayout.from_robot_config(_config())), ((),))

    def test_a_sphere_counts_where_its_padded_radius_is_not_negative_as_in_the_kernel(self) -> None:
        """The kernel masks a sphere by its PADDED radius (``sph.w >= 0`` after the offset). A zero radius with 5 mm
        of padding is a sphere to it, so it is one here too: a pair the kernel counts and this missed would be a pair
        the driver never saw."""
        layout = self.module.SphereLayout.from_robot_config(_config(buffer=0.005))
        point = _spheres(([0.0, 0.0, 0.0], 0.0), ([0.05, 0.0, 0.0], 0.05))
        (pairs,) = self.module.overlapping_pairs(point, layout)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0].depth_mm, 10.0, places=6)

    def test_a_pair_within_a_hundredth_of_a_millimetre_of_touching_is_named_too(self) -> None:
        """Two arithmetics over one set of spheres: the kernel's float32 and this float64. A pair the kernel reads as
        just touching must not be one this reads as just apart, so a pair that close is named, conservatively."""
        layout = self.module.SphereLayout.from_robot_config(_config())
        near = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.100005, 0.0, 0.0], 0.05))
        (pairs,) = self.module.overlapping_pairs(near, layout)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0].depth_mm, -0.005, places=6)
        apart = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.10002, 0.0, 0.0], 0.05))
        self.assertEqual(self.module.overlapping_pairs(apart, layout), ((),))

    def test_it_agrees_with_the_long_way_on_random_robots(self) -> None:
        """⭐ The bounding-sphere shortcut never drops a pair: every pose of three random links, with random padding
        and an ignored pair, against each sphere pair on its own."""
        rng = np.random.default_rng(7)
        links = ["a_link", "b_link", "c_link", "d_link"]
        for trial in range(40):
            counts = rng.integers(1, 7, size=len(links))
            spheres_cfg = {name: [{"center": list(rng.uniform(-0.1, 0.1, 3)), "radius": float(rng.uniform(0.005, 0.04))}
                                  for _ in range(int(n))] for name, n in zip(links, counts)}
            config = {"robot_cfg": {"kinematics": {
                "collision_link_names": links, "collision_spheres": spheres_cfg,
                "self_collision_buffer": {name: float(rng.uniform(0.0, 0.004)) for name in links},
                "self_collision_ignore": {"a_link": ["b_link"]},
            }}}
            layout = self.module.SphereLayout.from_robot_config(config)
            poses = rng.uniform(-0.15, 0.15, size=(3, layout.slots, 4))
            poses[:, :, 3] = rng.uniform(-0.002, 0.05, size=(3, layout.slots))
            got = self.module.overlapping_pairs(poses, layout)
            want = _brute_force(poses, layout, self.module.NAMED_WITHIN_MM)
            for pose_pairs, expected in zip(got, want):
                with self.subTest(trial=trial):
                    self.assertEqual({(p.link_a, p.link_b) for p in pose_pairs}, set(expected))
                    for p in pose_pairs:
                        self.assertAlmostEqual(p.depth_mm, expected[(p.link_a, p.link_b)], places=9)

    def test_a_sphere_array_the_layout_does_not_describe_is_refused(self) -> None:
        layout = self.module.SphereLayout.from_robot_config(_config())
        with self.assertRaises(self.module.SphereLayoutError):
            self.module.overlapping_pairs(_spheres(([0.0, 0.0, 0.0], 0.05)), layout)


class TheReportRowsAreBuiltOnTheCpuTests(unittest.TestCase):
    """The rows a check_js report carries are built by ``refused_rows``, plain arithmetic both interpreters run, so the
    sidecar keeps only the reading of its tensors (review of F1, 2026-09-30: the sidecar's half had never run, because
    cuRobo's kernels load on no box of the development cell). What is pinned here runs; what the GPU run still owes is
    that cuRobo's own terms and spheres go in (``scripts/curobo/probe_band_admission.py``)."""

    def setUp(self) -> None:
        self.module = _pairs()
        self.layout = self.module.SphereLayout.from_robot_config(_two_links(0.002))
        self.asked: list[list[int]] = []

    def spheres_of(self, indices: "list[int]") -> np.ndarray:
        self.asked.append(list(indices))
        return np.repeat(_LOOK0_PAIR, len(indices), axis=0)

    def test_every_refused_sample_is_a_row_in_order_with_its_three_terms_and_its_pairs(self) -> None:
        passes = [True, False, False, True, False]
        bound = np.asarray([0.0, 0.0, 0.3, 0.0, 0.0])
        self_hit = np.asarray([0.0, 0.8, 0.0, 0.0, 0.2])
        world = np.asarray([0.0, 0.0, 0.0, 0.0, 0.5])
        rows, named = self.module.refused_rows(passes, bound, self_hit, world, self.spheres_of, self.layout,
                                               name_pairs=True)
        self.assertTrue(named)
        self.assertEqual([row["index"] for row in rows], [1, 2, 4])
        self.assertEqual([(row["bound_ok"], row["self_ok"], row["world_ok"]) for row in rows],
                         [(True, False, True), (False, True, True), (True, False, False)])
        self.assertEqual(self.asked, [[1, 4]], "the spheres of the self hits alone are read, once")
        self.assertEqual(rows[1]["pairs"], [], "a sample the self term did not refuse names no pair")
        (pair,) = rows[0]["pairs"]
        self.assertEqual(pair[:2], ["forearm_link", "wrist_2_link"])
        self.assertAlmostEqual(pair[2], 1.22, places=6)
        self.assertAlmostEqual(pair[3], -2.78, places=6)
        import json

        self.assertEqual(json.loads(json.dumps(rows)), rows, "the rows are JSON as they stand")

    def test_without_names_nothing_is_named_and_no_sphere_is_read(self) -> None:
        for layout, name_pairs in ((self.layout, False), (None, True)):
            with self.subTest(layout=layout is not None, name_pairs=name_pairs):
                self.asked.clear()
                rows, named = self.module.refused_rows([False], [0.0], [0.4], [0.0], self.spheres_of, layout,
                                                       name_pairs=name_pairs)
                self.assertFalse(named)
                self.assertEqual(rows, [{"index": 0, "bound_ok": True, "self_ok": False, "world_ok": True,
                                         "pairs": None}])
                self.assertEqual(self.asked, [])

    def test_a_self_hit_whose_spheres_name_nothing_says_so_with_an_empty_list(self) -> None:
        """The kernel's hit and the naming disagree: the row carries no pair, which the driver reads as a refusal that
        stands (``band.admission_refusal``: a self collision no pair was named for)."""
        apart = _spheres(([0.0, 0.0, 0.0], 0.01), ([0.2, 0.0, 0.0], 0.01))
        rows, named = self.module.refused_rows([False], [0.0], [0.1], [0.0], lambda indices: apart, self.layout,
                                               name_pairs=True)
        self.assertTrue(named)
        self.assertEqual(rows[0]["pairs"], [])

    def test_the_client_reads_the_rows_whole(self) -> None:
        """The round trip: what the sidecar would write, read by the client's own reader into the driver's rows."""
        from src.robot.safety.planning.curobo_client import _judgement_from_reply

        passes = [True, False, True]
        rows, named = self.module.refused_rows(passes, [0.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.0],
                                               self.spheres_of, self.layout, name_pairs=True)
        reply = {"success": True, "valid": False, "first_invalid": 1, "checked": 3, "clearance_m": 0.0,
                 "refused": rows, "pairs_named": named}
        judged = _judgement_from_reply(reply, sent=3, clearance_m=0.0, named=True)
        (row,) = judged.refused
        self.assertEqual((row.index, row.bound_ok, row.self_ok, row.world_ok), (1, True, False, True))
        (pair,) = row.pairs
        self.assertEqual((pair.link_a, pair.link_b), ("forearm_link", "wrist_2_link"))
        self.assertAlmostEqual(pair.depth_mm - pair.unpadded_depth_mm, 4.0, places=9)

    def test_a_count_of_passes_the_terms_do_not_match_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.module.refused_rows([False, True], [0.0], [0.4], [0.0], self.spheres_of, self.layout,
                                     name_pairs=True)


if __name__ == "__main__":
    unittest.main()
