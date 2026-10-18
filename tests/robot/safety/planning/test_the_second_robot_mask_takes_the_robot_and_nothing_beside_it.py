"""The second robot mask (owner, 2026-10-15): the robot as spheres inside its bodies, standing where the guard stands the
robot, and a point within their gap and a depth's noise of one the robot's.

"Die zweite Robotermaske wird aber interessant, wenn es zu ETH Sachen geht oder so zu mehreren Robotern ... das muss
schon mit drinnen sein", and "die Kugeln so fitten, dass diese nicht herausragen". The maps (``scripts/arms/fit.py
--mask``) hold every arm and hand inside out: no sphere past its body, every point of it within 8 mm of one. The robot's
envelope carries them on the frames the guard stands the robot on, its hand on the flange where the controller has it
(``self_envelope.mask_spheres``), and the self body takes a point within 10 mm of one for the robot's, whatever the self
filter's padding (``perceived.SelfBody.of``). This file holds:

* the committed mask maps: every sphere inside its body by the mesh's own winding number and distance, every point of a
  draw of its surface the fit never saw within the gap of one, an arm's spheres in its bundle's DH frames;
* a wrist camera's boxes filled from inside (``inner_box_spheres``), every point of a box within the gap of a sphere;
* the owner's cell carries its mask: the UR10's, the Hand-E's and the D415's spheres, 10 mm about them, and a cell
  whose maps are missing or disagree carries none;
* a robot a camera sees, drawn from its meshes (``tests/_seen_scenes``): every pixel of it is the robot's, read on it and
  a millimetre either side, no point farther from its meshes than the reach is, nor any of a pillar beside it;
* the mask hangs where the guard hangs the robot: a hand on a flange the controller puts 6.4 mm off the table goes
  with it, the links stay;
* the camera world takes a point the mask holds for the robot's and keeps one past its reach, and a view a pick holds
  loses the robot by the mask of its own shutter.

Why on the CPU and on the guard's own frames (2026-10-15): built first in the cuRobo sidecar on cuRobo's kinematics,
the mask stood the hand on the nominal table's flange, 6.4 mm and 0.6 degrees from where the owner's UR10 has it, and
would take points up to 18 mm from the real hand; the guard's frames carry the controller's flange. A k-d tree takes the
307,200 points of a 640 x 480 frame in 21 ms, where the sidecar took 44 ms and the frame's trip through the WSL mount.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from src.robot.safety.planning import self_envelope as envelopes
from src.robot.safety.planning._declared_body import Box, inner_box_spheres
from src.robot.safety.planning.live_world import DepthSnapshot
from src.robot.safety.planning.perceived import (
    DropReason,
    LinkCapsule,
    PerceptionGeometryError,
    SelfBody,
    SelfEnvelope,
    _within_spheres,
)
from src.robot.safety.planning.self_envelope import (
    MASK_NOISE_MM,
    arm_capsules,
    mask_spheres,
    self_envelope,
    yawed_link_transforms_mm,
)
from tests.robot.safety.planning.test_live_planner_world import (
    _CAMERA_HEIGHT_MM,
    _INTRINSICS,
    _SELF,
    _SHAPE,
    _Camera,
    _scene_with_a_block,
    _tool_at,
    _world,
    _wrist_world,
)
from tests.robot.safety.planning.test_the_planner_plans_on_a_lean_model_and_judges_on_the_evidence import _script

_ROOT = Path(__file__).resolve().parents[4]
_MAPS = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"
_DATA = _ROOT / "src" / "robot" / "safety" / "data"
#: A draw of every body's surface the fit never saw (it fitted on seed 0 and verified on 12345).
_SEED = 2026
_SAMPLES = 20_000
#: The maps are written to the micrometre and trimesh's nearest-surface query answers to a few of them: 3.6 um is the
#: most a sphere of the committed maps stands past its centre's distance to the surface. A hundredth of a millimetre is
#: a hundredth of what a D415 reads.
_WRITTEN_MM = 0.01
#: The probe's arm and its start pose (``scripts/curobo/probe_live_world.py``), degrees.
_UR5E_START_DEG = tuple(float(v) for v in np.degrees([0.0, -1.5, 1.5, -1.57, -1.57, 0.0]))


def _maps(kind: str) -> "list[str]":
    """Every arm (``arm``) or hand (``gripper``) with a committed cover map: each needs a mask map beside it."""
    return sorted(path.name.split(f"_{kind}_")[0] for path in _MAPS.glob(f"*_{kind}_spheres.yml"))


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _mask_map(name: str, kind: str) -> dict:
    return _read(_MAPS / f"{name}_{kind}_mask_spheres.yml")


class TheMaskMapsLieInsideTheirBodiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fit = _script("fit")
        cls.ruler = _script("_mesh_body")

    def test_every_arm_and_hand_with_a_cover_map_has_a_mask_map_at_the_gap(self) -> None:
        from src.robot.safety._ur_kinematics import ur_dh_rows

        arms, hands = _maps("arm"), _maps("gripper")
        self.assertIn("ur10", arms)
        self.assertIn("robotiq_hande", hands)
        for arm in arms:
            document = _mask_map(arm, "arm")
            blocks = document["collision_spheres"]
            with self.subTest(arm=arm):
                self.assertEqual(("mask", self.fit.MASK_GAP_MM), (document["_provenance"]["model"],
                                                                  document["_provenance"]["gap_mm"]))
                self.assertEqual([1, 2, 3, 4, 5, 6], [blocks[body]["frame"] for body in (
                    "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")])
                rows = ur_dh_rows(arm)
                assert rows is not None, f"{arm} has a mask map and no arm description"
                np.testing.assert_allclose(document["_provenance"]["dh_m"], [[row.a_m, row.d_m, row.alpha_rad]
                                                                             for row in rows], atol=1e-9)
        for hand in hands:
            document = _mask_map(hand, "gripper")
            cover = _read(_MAPS / f"{hand}_gripper_spheres.yml")
            with self.subTest(hand=hand):
                self.assertEqual(("mask", self.fit.MASK_GAP_MM), (document["_provenance"]["model"],
                                                                  document["_provenance"]["gap_mm"]))
                self.assertEqual(cover["_provenance"]["origin"], document["_provenance"]["origin"],
                                 "the mask's hand hangs where the cover map's hangs")

    def _judge(self, label: str, meshes: list, spheres: "list[dict]") -> None:
        """Every sphere inside the body, every point of a fresh draw of its surface within the gap of one."""
        body = self.ruler.MeshBody(meshes, samples=_SAMPLES, seed=_SEED)
        centres = np.asarray([sphere["center"] for sphere in spheres], dtype=np.float64)
        radii = np.asarray([sphere["radius"] for sphere in spheres], dtype=np.float64)
        to_surface = body.distance(centres)
        inside = body.inside(centres)
        # A flat sphere, of MASK_FLAT_M, is a point of the body: its centre inside it, or a seed of its surface within
        # its own radius of it (trimesh's query reads a seed on the UR3 upper arm's slivers up to 62 um off).
        flat = radii <= self.fit.MASK_FLAT_M + 1e-12
        deep = ~flat
        self.assertFalse((deep & ~inside).any(), f"{label}: {int((deep & ~inside).sum())} centre(s) outside the body")
        past = float(np.max(radii[deep] - to_surface[deep], initial=0.0)) * 1000.0
        self.assertLessEqual(past, _WRITTEN_MM, f"{label}: a sphere stands {past:.4f} mm past its body")
        off = float(np.max(to_surface[flat & ~inside], initial=0.0)) * 1000.0
        self.assertLessEqual(off, self.fit.MASK_FLAT_M * 1000.0, f"{label}: a flat sphere {off:.4f} mm off the surface")
        gaps = np.full(len(body.points), np.inf)
        for start in range(0, len(centres), 256):
            block = slice(start, start + 256)
            gaps = np.minimum(gaps, (np.linalg.norm(body.points[None, :, :] - centres[block][:, None, :], axis=2)
                                     - radii[block][:, None]).min(axis=0))
        worst = float(gaps.max()) * 1000.0
        self.assertLessEqual(worst, self.fit.MASK_GAP_MM, f"{label}: a surface point {worst:.2f} mm from every sphere")

    def test_every_sphere_lies_inside_its_body_and_every_surface_point_within_the_gap_of_one(self) -> None:
        """⭐ The owner's two conditions, on every committed map: nothing stands past the robot, so a point off the
        robot is never taken for a sphere's sake, and nothing of the robot lies farther than the gap from a sphere, so a
        point on it always is."""
        for arm in _maps("arm"):
            document = _mask_map(arm, "arm")
            for body, block in document["collision_spheres"].items():
                with self.subTest(arm=arm, body=body):
                    self._judge(f"{arm}/{body}", self.ruler.load_meshes(_DATA / f"{arm}_collision_meshes.npz",
                                                                       [body]), block["spheres"])
        for hand in _maps("gripper"):
            document = _mask_map(hand, "gripper")
            bodies = [row["body"] for row in document["_provenance"]["bodies"]]
            with self.subTest(hand=hand):
                self._judge(hand, self.ruler.load_meshes(_DATA / f"{hand}_hand_meshes.npz", bodies),
                            document["collision_spheres"]["tool0"])


def _box_points(box: Box, centres_mm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Every corner of every cell the spheres stand in, where a box's point lies farthest from its cell's sphere, and a
    draw of the box's inside and faces besides, millimetres."""
    centre, half = np.asarray(box.centre_mm), np.asarray(box.half_extents_mm)
    axes = [np.linspace(centre[i] - half[i], centre[i] + half[i], len(np.unique(np.round(centres_mm[:, i], 6))) + 1)
            for i in (0, 1, 2)]
    corners = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    drawn = rng.uniform(-1.0, 1.0, size=(26_000, 3))
    drawn[np.arange(6_000), rng.integers(0, 3, 6_000)] = rng.choice((-1.0, 1.0), 6_000)
    return np.concatenate([corners, centre + drawn * half])


class TheBoxesAreFilledFromInsideTests(unittest.TestCase):
    _BOXES = (
        Box(name="cube", centre_mm=(10.0, -20.0, 30.0), half_extents_mm=(20.0, 20.0, 20.0)),
        Box(name="d415", centre_mm=(35.0, 0.0, -12.9), half_extents_mm=(131.15 / 2.0, 35.15 / 2.0, 16.0)),
        Box(name="plate", centre_mm=(0.0, 0.0, 11.5), half_extents_mm=(50.0, 50.0, 1.0)),
        Box(name="bar", centre_mm=(0.0, 0.0, 0.0), half_extents_mm=(150.0, 5.0, 5.0)),
    )

    def test_every_sphere_lies_inside_the_box_and_every_point_of_it_within_the_gap(self) -> None:
        rng = np.random.default_rng(7)
        for box in self._BOXES:
            for gap in (8.0, 3.0):
                spheres = inner_box_spheres(box, gap_mm=gap)
                centres = np.asarray([sphere["center"] for sphere in spheres]) * 1000.0
                radii = np.asarray([sphere["radius"] for sphere in spheres]) * 1000.0
                with self.subTest(box=box.name, gap=gap):
                    reach = np.abs(centres - np.asarray(box.centre_mm)) + radii[:, None]
                    self.assertTrue(np.all(reach <= np.asarray(box.half_extents_mm) + 1e-9), "a sphere past the box")
                    points = _box_points(box, centres, rng)
                    near = np.full(len(points), np.inf)
                    for start in range(0, len(centres), 256):
                        block = slice(start, start + 256)
                        near = np.minimum(near, (np.linalg.norm(points[None] - centres[block][:, None], axis=2)
                                                 - radii[block][:, None]).min(axis=0))
                    self.assertLessEqual(float(near.max()), gap + 1e-9)

    def test_a_box_with_no_gap_or_no_volume_is_refused(self) -> None:
        for gap in (0.0, -1.0):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                inner_box_spheres(self._BOXES[0], gap_mm=gap)
        with self.assertRaises(ValueError):
            inner_box_spheres(Box(name="sheet", centre_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 0.0)), gap_mm=8.0)


def _probe_hand():  # noqa: ANN202
    """The probe cell's hand: the 2F-85 on the frame the Isaac cell declares (``tests/_wrist_body.wrist_cell``)."""
    from src.robot.safety.planning.hand import planner_hand
    from tests._wrist_body import wrist_cell

    return planner_hand(wrist_cell())


class TheOwnersCellCarriesItsMaskTests(unittest.TestCase):
    def test_the_ur10_the_hande_and_the_d415_inside_out_10_mm_about_them(self) -> None:
        from tests._owner_guard import HOME_DEG, camera_body, owner_cell

        preflight, arm = owner_cell()
        envelope = self_envelope(preflight, arm, np.radians(HOME_DEG))
        assert envelope is not None
        self.assertEqual(8.0 + MASK_NOISE_MM, envelope.mask_reach_mm)
        blocks = _mask_map("ur10", "arm")["collision_spheres"]
        hand = _mask_map("robotiq_hande", "gripper")["collision_spheres"]["tool0"]
        arm_count = sum(len(block["spheres"]) for block in blocks.values())
        frames = [sphere.frame for sphere in envelope.mask]
        self.assertEqual([frame for _, block in blocks.items() for frame in [block["frame"]] * len(block["spheres"])],
                         frames[:arm_count], "the arm's spheres in their links' DH frames, in the arm's order")
        rest = envelope.mask[arm_count + len(hand):]
        self.assertTrue(rest, "the D415's spheres")
        self.assertEqual({6}, set(frames[arm_count:]), "the hand and the camera on the flange")
        # The camera's spheres inside its enclosure, the rig's 3 mm of room about it taken off: in its optical frame.
        body = camera_body()
        to_optical = np.linalg.inv(body.placement())
        centres = np.asarray([sphere.start_mm for sphere in rest]) @ to_optical[:3, :3].T + to_optical[:3, 3]
        radii = np.asarray([sphere.radius_mm for sphere in rest])
        housing = body.boxes[0]
        enclosure = np.asarray(housing.half_extents_mm) - body.margin_mm
        self.assertTrue(np.all(np.abs(centres - housing.centre_mm) + radii[:, None] <= enclosure + 1e-6))

    def test_none_where_a_map_is_missing_or_the_maps_keep_two_gaps(self) -> None:
        """Each of these takes the robot by the self filter alone, as before."""
        hand = _probe_hand()
        self.assertIsNotNone(mask_spheres(hand, "ur5e"))
        self.assertIsNone(mask_spheres(hand, "ur99"), "an arm with no mask map")
        self.assertIsNone(mask_spheres(replace(hand, model="no_such_hand"), "ur5e"), "a hand with no mask map")
        read = envelopes._mask_map  # noqa: SLF001

        def two_gaps(path: str):  # noqa: ANN202
            gap, blocks = read(path)
            return (5.0 if "gripper" in path else gap), blocks

        envelopes._mask_spheres.cache_clear()  # noqa: SLF001
        self.addCleanup(envelopes._mask_spheres.cache_clear)  # noqa: SLF001
        with mock.patch.object(envelopes, "_mask_map", side_effect=two_gaps):
            self.assertIsNone(mask_spheres(hand, "ur5e"), "a hand fitted at another gap than the arm")


def _scene():  # noqa: ANN202
    """The probe's ur5e and its 2F-85 at the start pose with two pillars beside the hand, one 5 mm under it and one well
    clear, from a camera 1.4 m off, as 4a.11 drew it on the GPU: every point of the frame in BASE, which are the
    robot's, the far pillar, and the robot's triangles."""
    from tests import _seen_scenes as seen

    hand = _probe_hand()
    robot = seen.robot_triangles(_UR5E_START_DEG, model="ur5e", hand=hand)
    reach = robot[len(seen.robot_triangles(_UR5E_START_DEG, model="ur5e")):].reshape(-1, 3)
    low, high = reach.min(axis=0), reach.max(axis=0)
    centre = (low + high) / 2.0
    near = seen.Solid(centre=(centre[0], centre[1], (low[2] - 5.0) / 2.0), half=(30.0, 30.0, (low[2] - 5.0) / 2.0))
    far = seen.Solid(centre=(centre[0] + 120.0, centre[1] + 300.0, 100.0), half=(40.0, 40.0, 100.0))
    camera = seen.looking_at((500.0, 900.0, 1000.0), (centre[0] + 100.0, centre[1] + 100.0, 300.0))
    arm = seen.robot_depth(camera, robot)
    scene = seen.render(camera, (near, far))
    on_robot = arm < np.where(scene > 0.0, scene, np.inf)
    depth = np.where(on_robot, arm, scene)
    rows, cols = np.nonzero(depth > 0.0)

    def placed(z: np.ndarray) -> np.ndarray:
        return (np.column_stack(((cols - seen.K[0, 2]) * z / seen.FOCAL, (rows - seen.K[1, 2]) * z / seen.FOCAL, z))
                @ camera[:3, :3].T + camera[:3, 3])

    robots = on_robot[rows, cols]
    z = depth[rows, cols]
    return placed, z, robots, far, robot


class TheRobotACameraSeesIsTheMasksTests(unittest.TestCase):
    """4a.11, which asked the sidecar on the GPU, on the CPU: the mask on the guard's frames."""

    @classmethod
    def setUpClass(cls) -> None:
        placed, cls.z, cls.robots, cls.far, cls.triangles = _scene()
        cls.placed = staticmethod(placed)
        spheres, reach = mask_spheres(_probe_hand(), "ur5e")  # type: ignore[misc]
        frames = yawed_link_transforms_mm("ur5e", np.radians(_UR5E_START_DEG), 0.0)
        assert frames is not None
        cls.mask = SelfBody.of(SelfEnvelope(frames_mm=tuple(frames), capsules=arm_capsules("ur5e"), mask=spheres,
                                            mask_reach_mm=reach), padding_mm=0.0).mask_mm

    def test_every_pixel_of_the_robot_is_the_robots_read_on_it_or_a_millimetre_either_side(self) -> None:
        self.assertGreater(int(self.robots.sum()), 20_000)
        for by in (0.0, -1.0, 1.0):
            with self.subTest(read_mm=by):
                taken = _within_spheres(self.placed(self.z + np.where(self.robots, by, 0.0)), self.mask)
                self.assertTrue(bool(taken[self.robots].all()), f"{int((~taken[self.robots]).sum())} pixel(s) left")

    def test_no_point_farther_from_the_robot_than_the_reach_nor_any_of_a_pillar_beside_it(self) -> None:
        import trimesh

        points = self.placed(self.z)
        taken = _within_spheres(points, self.mask)
        beside = taken & ~self.robots
        if beside.any():
            body = trimesh.Trimesh(vertices=self.triangles.reshape(-1, 3),
                                   faces=np.arange(self.triangles.shape[0] * 3).reshape(-1, 3), process=False)
            self.assertLessEqual(float(body.nearest.on_surface(points[beside])[1].max()),
                                 8.0 + MASK_NOISE_MM + _WRITTEN_MM)
        on_far = self.far.distance_mm(points) <= 0.5
        self.assertGreater(int(on_far.sum()), 500)
        self.assertFalse(bool(taken[on_far].any()), "a pillar clear of the robot")


class TheMaskHangsWhereTheGuardHangsTheRobotTests(unittest.TestCase):
    def test_a_hand_on_a_flange_the_controller_puts_off_the_table_goes_with_it(self) -> None:
        """The owner's UR10 puts its flange 6.4 mm and 0.6 degrees from the nominal table's (2026-10-12), and the guard
        stands the hand where the controller has it (``arm_link_transforms_mm``): so does the mask, and its links stay."""
        spheres, reach = mask_spheres(_probe_hand(), "ur5e")  # type: ignore[misc]
        frames = yawed_link_transforms_mm("ur5e", np.radians(_UR5E_START_DEG), 0.0)
        assert frames is not None
        moved = list(frames)
        shift = np.eye(4)
        shift[:3, 3] = (6.4, 0.0, 0.0)
        moved[6] = shift @ frames[6]
        nominal, calibrated = (SelfBody.of(SelfEnvelope(frames_mm=tuple(chain), capsules=arm_capsules("ur5e"),
                                                        mask=spheres, mask_reach_mm=reach), padding_mm=0.0).mask_mm
                               for chain in (frames, moved))
        on_flange = np.asarray([sphere.frame == 6 for sphere in spheres])
        self.assertTrue(on_flange.any() and (~on_flange).any())
        np.testing.assert_allclose(calibrated[on_flange, :3] - nominal[on_flange, :3], [[6.4, 0.0, 0.0]] *
                                   int(on_flange.sum()), atol=1e-9)
        np.testing.assert_array_equal(calibrated[~on_flange], nominal[~on_flange])

    def test_the_hand_stands_inside_the_hand_the_guard_places(self) -> None:
        """The owner's Hand-E on its 23 mm plate, turned by the declared tool frame: every sphere of the hand's mask lies
        in the hand as the guard places it on the flange (``tests/_seen_scenes.robot_triangles``), by the winding number
        of its meshes; a flat one, of 0.1 mm, within its own radius of their surface."""
        import trimesh

        from tests import _seen_scenes as seen
        from tests._owner_guard import HOME_DEG, owner_cell

        preflight, arm = owner_cell()
        owners_hand = preflight.planner_hand(arm)
        spheres, reach = mask_spheres(owners_hand, "ur10")  # type: ignore[misc]
        frames = yawed_link_transforms_mm("ur10", np.radians(HOME_DEG), 0.0)
        assert frames is not None
        placed = SelfBody.of(SelfEnvelope(frames_mm=tuple(frames), capsules=arm_capsules("ur10"), mask=spheres,
                                          mask_reach_mm=reach), padding_mm=0.0).mask_mm
        # The arm's spheres first, in its links' order, then the hand's (``mask_spheres``); wrist_3 hangs on the flange too.
        arm_count = sum(len(block["spheres"]) for block in _mask_map("ur10", "arm")["collision_spheres"].values())
        hand = slice(arm_count, arm_count + len(_mask_map("robotiq_hande", "gripper")["collision_spheres"]["tool0"]))
        self.assertEqual(len(spheres), hand.stop, "the arm and the hand, no wrist camera handed")
        radii = np.asarray([sphere.radius_mm for sphere in spheres])[hand]
        centres = placed[hand, :3]
        triangles = seen.robot_triangles(HOME_DEG, model="ur10", hand=owners_hand)[
            len(seen.robot_triangles(HOME_DEG, model="ur10")):]
        mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(len(triangles) * 3).reshape(-1, 3),
                               process=False)
        inside = np.abs(_script("_mesh_body").winding(mesh, centres)) > 0.5
        flat = radii <= _script("fit").MASK_FLAT_M * 1000.0 + 1e-9
        self.assertTrue(bool(inside[~flat].all()), f"{int((~inside[~flat]).sum())} of the hand's spheres stand outside it")
        if (flat & ~inside).any():
            off = mesh.nearest.on_surface(centres[flat & ~inside])[1]
            self.assertLessEqual(float(off.max()), _script("fit").MASK_FLAT_M * 1000.0)

    def test_a_body_without_a_mask_is_the_self_filter_alone_and_a_mask_is_finite(self) -> None:
        body = SelfBody.of(_SELF, padding_mm=15.0)
        self.assertEqual((0, 4), body.mask_mm.shape)
        np.testing.assert_array_equal(SelfBody.from_frames(_SELF.frames_mm, _SELF.capsules, padding_mm=15.0).radii_mm,
                                      body.radii_mm)
        with self.assertRaises(PerceptionGeometryError):
            replace(body, mask_mm=np.array([[0.0, 0.0, np.nan, 1.0]]))


def _masked(envelope: SelfEnvelope, centre_mm: "tuple[float, float, float]", radius_mm: float) -> SelfEnvelope:
    """``envelope`` with a mask of one sphere on its base frame, 10 mm about it the robot's."""
    sphere = LinkCapsule(frame=0, start_mm=centre_mm, end_mm=centre_mm, radius_mm=radius_mm)
    return replace(envelope, mask=(sphere,), mask_reach_mm=8.0 + MASK_NOISE_MM)


def _moved(envelope: SelfEnvelope, by_mm: float) -> SelfEnvelope:
    shift = np.eye(4)
    shift[0, 3] = by_mm
    return replace(envelope, frames_mm=tuple(shift @ frame for frame in envelope.frames_mm))


def _frame(depth: np.ndarray, **kwargs: object) -> DepthSnapshot:
    return DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, **{"timestamp": 100.0, **kwargs})  # type: ignore[arg-type]


class TheCameraWorldTakesWhatTheMaskHoldsTests(unittest.TestCase):
    """The block of ``tests/test_live_planner_world``: 60 mm square, its top 80 mm over the bench, at x 150 mm."""

    def test_a_point_within_the_reach_of_a_mask_sphere_is_the_robots_and_one_past_it_stays(self) -> None:
        depth, _ = _scene_with_a_block()
        plain = _world(_Camera(_frame(depth))).world_for(self_envelope=_SELF, now=100.1)
        # The block's top, 42 mm round its middle at most, lies within 60 + 10 mm of a sphere 30 mm over the bench under
        # it, and 100 mm from one 20 mm under the bench.
        held = _world(_Camera(_frame(depth))).world_for(self_envelope=_masked(_SELF, (150.0, 0.0, 30.0), 60.0),
                                                        now=100.1)
        past = _world(_Camera(_frame(depth))).world_for(self_envelope=_masked(_SELF, (150.0, 0.0, -20.0), 60.0),
                                                        now=100.1)
        self.assertEqual((1, 0, 1), (plain.perceived_count, held.perceived_count, past.perceived_count), held.render())
        assert plain.perceived is not None and held.perceived is not None
        self.assertGreater(held.perceived.dropped_points[DropReason.SELF],
                           plain.perceived.dropped_points.get(DropReason.SELF, 0))

    def test_a_view_a_pick_holds_loses_the_robot_by_the_mask_of_its_own_shutter(self) -> None:
        """The first look's frame is held for the pick, the robot then standing over the block; the next look sees the
        bench alone, the robot and its mask moved off. The held frame keeps losing the block by its own shutter's mask;
        the control, no mask, holds the block that frame alone saw."""
        for masked in (False, True):
            with self.subTest(masked=masked):
                depth, _ = _scene_with_a_block()
                camera = _Camera(_frame(depth, tool_to_base_mm=_tool_at(0.0)))
                world = _wrist_world(camera)
                world.hold_pick_views()
                first_body = _masked(_SELF, (150.0, 0.0, 30.0), 60.0) if masked else _SELF
                first = world.world_for(self_envelope=first_body, now=100.1)
                camera.snapshot = _frame(np.full(_SHAPE, _CAMERA_HEIGHT_MM), timestamp=100.2,
                                         tool_to_base_mm=_tool_at(-92.0))
                later = world.world_for(self_envelope=_moved(first_body, -92.0), now=100.3)
                self.assertEqual((0, 0) if masked else (1, 1), (first.perceived_count, later.perceived_count),
                                 later.render())


if __name__ == "__main__":
    unittest.main()
