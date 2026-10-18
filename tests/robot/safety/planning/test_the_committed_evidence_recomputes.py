"""Every committed planner evidence file recomputes from committed files alone (P0f F0, the proof before any move).

The 50 files under ``planning/robot/evidence/`` admit a cell to plan with cuRobo, and they admit it by hashes: the
URDF the planner loaded, the arrays the exact mesh guard held, the hand's sphere map and the configurations judged.
aurora holds 47 of them; Willy's own three are the owner's Hand-E on a 23 mm coupling closing along -Y (margins 0, 3
and 4 mm). The seven measured with the 16 sphere slots a carried part reserved (``_a16``) left with the carried part
on 2026-10-17, and the others lost their ``_a0`` suffix and field, their hashes as they were. Until this file two of
the files were checked against anything (the desk tests' ur5e with the 2F-85, +Y and +Z); the
rest were checked for being 64 characters long. P0f moves and rewrites the scripts and the code these hashes come
from, so each is recomputed here, per file, twice where it can be: once through the function the stack calls, which
says the code still agrees with the files, and once through ``tests/_evidence_rulers.py``, which restates the recipe
and so survives the rewrite.

Measured 2026-09-23 on Windows (numpy 2.4.4, PyYAML 6.0.3) in aurora: 47 of 47 equal for every hash below; in Willy
57 of 57 (2026-10-13), 50 of 50 once the attach files left (2026-10-17). What each check
cannot see is said beside it. The two hashes a running sidecar reports about the descriptor it loaded
(``arm_descriptor_sha256`` and ``composed_sha256``) need the built descriptor and cuRobo, so they are proved on the box.

What this pins for P0f, measured: the header literal ``Written by scripts/arms/_ur_description.py`` is inside
``urdf_sha256`` (changed, none match); the six pin digests of the vendored description are too; the guard's part
names, frame integers and vertex and face dtypes are inside ``guard_sha256``, the 2F-85's float32 and int32 included.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType

from src.config.schema.robot import RobotConfig
from src.robot.safety._fcl_self_collision import composed_parts
from src.robot.safety.planning._hand_placement import HandPlacement, placement_quaternion_xyzw
from src.robot.safety.planning.evidence import (
    EVIDENCE_DIR,
    CombinationEvidence,
    desk_evidence_refusal,
    guard_sha256,
)
from src.robot.safety.planning.hand import COVER_FIT_WRITER, guard_variant_for
from src.robot.safety.planning.robot.retract_table import read_retract
from tests import _evidence_rulers as ruler

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts" / "arms"
#: How many files the repository ships per arm, 50 together: aurora's 47 and Willy's three on the ur10 (the owner's
#: Hand-E on 23 mm closing along -Y). A hand a customer measures adds files (the customer-hand trial adds one per hand
#: on ur10e); a shrunken directory is what must fail.
_SHIPPED_PER_ARM = {"ur3": 8, "ur3e": 8, "ur5": 7, "ur5e": 8, "ur10": 11, "ur10e": 8}
_EVIDENCED_ARMS = set(_SHIPPED_PER_ARM)


def _short_of_what_ships(arms: "list[str]") -> dict[str, int]:
    """Per arm, how many of the files the repository ships are missing; arms with none missing are left out."""
    counts = Counter(arms)
    return {arm: shipped - counts[arm] for arm, shipped in _SHIPPED_PER_ARM.items() if counts[arm] < shipped}


def _script(name: str) -> ModuleType:
    """A build script's module by path: the cuRobo environment cannot import ``src``, so these are not a package."""
    spec = importlib.util.spec_from_file_location(f"_f0_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # a dataclass module loaded by path looks itself up here
    spec.loader.exec_module(module)
    return module


def _placement(evidence: CombinationEvidence) -> HandPlacement:
    """How the file's hand was placed when it was swept: +Y+X undeclared, every other placement declared."""
    if (evidence.approach, evidence.closing) == ("+Y", "+X"):
        return HandPlacement.undeclared()
    return HandPlacement.from_quaternion_xyzw(placement_quaternion_xyzw(evidence.approach, evidence.closing))


def _parts(evidence: CombinationEvidence) -> dict:
    return composed_parts(evidence.arm, None, guard_variant_for(evidence.hand), float(evidence.coupling_mm),
                          placement=_placement(evidence), coupling_boxes=())


def _cell(evidence: CombinationEvidence) -> RobotConfig:
    """The cell a profile would declare for this file, so the desk check is asked what a cell would ask it."""
    gripper: dict = {
        "model": evidence.hand,
        "coupling_plates": ([{"name": "plate", "thickness_mm": float(evidence.coupling_mm)}]
                            if evidence.coupling_mm else []),
    }
    if (evidence.approach, evidence.closing) != ("+Y", "+X"):
        gripper["tool_frame"] = {
            "source": "willy",
            "offset_mm": [float(v) for v in _placement(evidence).approach_in_tool0],
            "rotation_quat_xyzw": [float(v) for v in placement_quaternion_xyzw(evidence.approach, evidence.closing)],
        }
    return RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": evidence.arm},
        "safety": {"self_collision": {"backend": "fcl", "planner_margin_mm": float(evidence.planner_margin_mm)}},
        "gripper": gripper,
    })


def _as_the_builder_writes(description: ModuleType, limits: ModuleType, arm_file: ModuleType,
                           arm: str) -> tuple[str, dict[str, bool]]:
    """The text ``descriptor.py`` writes, with which of its steps changed anything.

    The builder is module-level code that builds when imported, so its text steps are transcribed here: the package
    strip, the elbow clamp, and the tool0 and inertia checks that refuse. P0f F5 retired the parenthesis repair, the
    mesh rewrite and the tool0 graft, which served the Isaac source only. This proves the steps change nothing on
    today's render; the box proof rebuilds every descriptor and compares the files the builder wrote.
    """
    rendered = description.render_urdf(arm)
    stripped = rendered.replace("package://ur_description/", "")
    narrowed = limits.narrowed_rad(arm_file.joint_names(arm), arm_file.planner_limits_deg(arm),
                                   arm_file.limits_deg(arm))
    clamped, elbow_clamped = limits.clamp_to_planner_limits(stripped, narrowed)
    changed = {
        "package_strip": stripped != rendered,
        "elbow_clamp": bool(elbow_clamped) or clamped != stripped,
        "needs_tool0": '<link name="tool0"' not in clamped,
        "needs_inertia": any("<inertia " not in inertial.group(1)
                             for inertial in re.finditer(r"<inertial>(.*?)</inertial>", clamped, re.S)),
    }
    return clamped, changed


def _sidecar_urdf_sha256(text: str) -> str:
    """What the sidecar hashes: the file's bytes with CRLF folded (curobo_planner_server.py:315-319)."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8").replace(b"\r\n", b"\n")).hexdigest()


class TheEvidenceThisProvesTests(unittest.TestCase):
    def test_no_file_the_repository_ships_is_missing(self) -> None:
        """An emptied or shrunken directory fails here, rather than every per-file check passing on nothing. A hand a
        customer measured adds files, and each of those is recomputed by the checks below like the shipped ones."""
        files = ruler.evidence_files()
        self.assertEqual(sorted(EVIDENCE_DIR.glob("*.json")), files)
        self.assertEqual({}, _short_of_what_ships([ruler.held(path)["arm"] for path in files]))

    def test_the_check_sees_a_missing_file_and_admits_an_added_one(self) -> None:
        """The control. The exact count this replaced failed aurora's customer-hand trial of 2026-09-23 on the two files
        the trial measured for its hands on ur10e, 47 != 49, while every file it held recomputed."""
        shipped = [arm for arm, count in _SHIPPED_PER_ARM.items() for _ in range(count)]
        self.assertEqual(50, len(shipped))
        shrunk = list(shipped)
        shrunk.remove("ur5")
        self.assertEqual({"ur5": 1}, _short_of_what_ships(shrunk))
        self.assertEqual({}, _short_of_what_ships([*shipped, "ur10e", "ur10e"]))

    def test_each_arm_was_measured_against_one_urdf(self) -> None:
        by_arm: dict[str, set[str]] = {}
        for path in ruler.evidence_files():
            record = ruler.held(path)
            by_arm.setdefault(record["arm"], set()).add(record["urdf_sha256"])
        self.assertLessEqual(_EVIDENCED_ARMS, set(by_arm))
        self.assertEqual({arm: 1 for arm in by_arm}, {arm: len(held) for arm, held in by_arm.items()})

    def test_every_file_sits_under_the_name_its_record_derives(self) -> None:
        for path in ruler.evidence_files():
            with self.subTest(file=path.name):
                self.assertEqual(path.name, CombinationEvidence.from_dict(ruler.held(path)).path.name)


class TheUrdfHashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.description = _script("_ur_description")
        cls.limits = _script("_planner_limits")
        cls.arm_file = _script("_arm_file")

    def test_every_builder_text_step_changes_nothing_on_the_render(self) -> None:
        for arm in sorted(_EVIDENCED_ARMS):
            text, changed = _as_the_builder_writes(self.description, self.limits, self.arm_file, arm)
            with self.subTest(arm=arm):
                self.assertEqual(dict.fromkeys(changed, False), changed)
                self.assertEqual(self.description.render_urdf(arm), text)
                self.assertNotIn("\r", text)

    def test_every_file_names_the_urdf_the_render_gives(self) -> None:
        for path in ruler.evidence_files():
            record = ruler.held(path)
            text, _ = _as_the_builder_writes(self.description, self.limits, self.arm_file, record["arm"])
            for newline in ("\n", "\r\n"):  # what write_text puts on disk on Linux and on Windows
                with self.subTest(file=path.name, newline=repr(newline)):
                    self.assertEqual(record["urdf_sha256"], _sidecar_urdf_sha256(text.replace("\n", newline)))

    def test_the_writer_path_in_the_header_is_inside_the_hash(self) -> None:
        """The control, and the fact P0f's move has to respect: the header names the renderer's path."""
        text, _ = _as_the_builder_writes(self.description, self.limits, self.arm_file, "ur5e")
        literal = "Written by scripts/arms/_ur_description.py"
        self.assertIn(literal, text)
        moved = text.replace(literal, "Written by scripts/curobo/_ur_description.py")  # the path before P0f F10
        recorded = {ruler.held(path)["urdf_sha256"] for path in ruler.evidence_files()}
        self.assertNotIn(_sidecar_urdf_sha256(moved), recorded)


class TheGuardHashTests(unittest.TestCase):
    def test_every_file_is_the_arrays_the_guard_composes(self) -> None:
        for path in ruler.evidence_files():
            evidence = CombinationEvidence.from_dict(ruler.held(path))
            with self.subTest(file=path.name):
                self.assertEqual(evidence.guard_sha256, guard_sha256(_parts(evidence)))

    def test_every_file_is_what_the_outside_ruler_composes(self) -> None:
        for path in ruler.evidence_files():
            record = ruler.held(path)
            with self.subTest(file=path.name):
                self.assertEqual(record["guard_sha256"], ruler.guard_sha256(record))

    def test_the_hash_moves_with_one_vertex_and_with_the_placement(self) -> None:
        """The controls: a hash that a moved vertex or another placement leaves alone guards nothing."""
        z = CombinationEvidence.from_dict(ruler.held(EVIDENCE_DIR / "ur5e_robotiq_hande_c20mm_+Z+X_m4mm.json"))
        y = CombinationEvidence.from_dict(ruler.held(EVIDENCE_DIR / "ur5e_robotiq_hande_c20mm_+Y+X_m4mm.json"))
        parts = _parts(z)
        vertices, faces, frame = parts["lfinger"]
        vertices = vertices.copy()
        vertices[0, 0] += 1e-9
        parts["lfinger"] = (vertices, faces, frame)
        self.assertNotEqual(z.guard_sha256, guard_sha256(parts))
        self.assertNotEqual(z.guard_sha256, guard_sha256(_parts(y)))


class TheDeskAdmitsEveryFileTests(unittest.TestCase):
    def test_the_cell_each_file_was_measured_for_is_admitted_by_that_file(self) -> None:
        """The runtime half: the desk check computes the hand map, the guard geometry, the rung and the guard margin
        from the cell's own config and compares them, which is what a cell meets before it starts a planner."""
        for path in ruler.evidence_files():
            evidence = CombinationEvidence.from_dict(ruler.held(path))
            with self.subTest(file=path.name):
                refused, admitting = desk_evidence_refusal(_cell(evidence))
                self.assertIsNone(refused, refused)
                assert admitting is not None
                self.assertEqual(path.name, admitting.path.name)


class TheHandMapHashTests(unittest.TestCase):
    def test_every_file_is_its_hand_s_committed_map(self) -> None:
        for path in ruler.evidence_files():
            record = ruler.held(path)
            with self.subTest(file=path.name):
                self.assertEqual(record["hand_map_sha256"], ruler.hand_map_sha256(record["hand"]))

    def test_every_map_names_the_writer_the_planner_accepts(self) -> None:
        """No hash sees this string, and ``planner_hand`` refuses a map whose writer is not exactly it (hand.py:264).
        A move of the fit script has to keep the three maps and the constant equal to each other."""
        import yaml

        for hand in sorted({ruler.held(path)["hand"] for path in ruler.evidence_files()}):
            document = yaml.safe_load((ruler.PLANNING / f"{hand}_gripper_spheres.yml").read_text(encoding="utf-8"))
            with self.subTest(hand=hand):
                self.assertEqual(COVER_FIT_WRITER, document["_provenance"]["generated_by"])


_CONTENT_ROBOTS = Path(__file__).resolve().parents[4] / "ext_deps" / "curobo" / "curobo" / "content" / "configs" / "robot"


@unittest.skipUnless((_CONTENT_ROBOTS / "willy_ur5e.yml").is_file(),
                     "no installed cuRobo descriptors on this machine (ext_deps/curobo); a box proof, not a CI one")
class TheInstalledDescriptorsAreTheEvidencedOnesTests(unittest.TestCase):
    """The two sidecar hashes a descriptor decides, recomputed from the descriptors this box installed.

    Measured 2026-09-23 in aurora and 2026-10-13 in Willy: the files (47 there, 57 here, 50 since 2026-10-17) matched the
    descriptors built on 2026-09-16 and none matched a descriptor built from the same scripts that day, because a
    descriptor carried the whole retract table's byte hash in its provenance and the table had gained rows since.
    Planning worked on this box and on no freshly installed one. The provenance now hangs on the arm's own rows, the
    files were measured again, and this says so before a planner is started. ``composed_sha256`` is what
    ``compose_for_cell`` makes of the descriptor with the gate's own inputs, and the descriptor is read with
    ``yaml.safe_load``, which reads every installed descriptor as cuRobo's loader does.
    """

    def test_every_file_names_the_installed_descriptor(self) -> None:
        import yaml

        from src.robot.safety.planning._curobo_body_links import canonical_sha256, compose_for_cell
        from src.robot.safety.planning.body_link import HandLink, coupling_bodies
        from src.robot.safety.planning.hand import planner_hand

        gate = _script("gate")
        installed: dict = {}
        for path in ruler.evidence_files():
            evidence = CombinationEvidence.from_dict(ruler.held(path))
            if evidence.arm not in installed:
                installed[evidence.arm] = yaml.safe_load(
                    (_CONTENT_ROBOTS / f"willy_{evidence.arm}.yml").read_text(encoding="utf-8"))
            descriptor = installed[evidence.arm]
            rotation = (None if (evidence.approach, evidence.closing) == ("+Y", "+X")
                        else placement_quaternion_xyzw(evidence.approach, evidence.closing))
            resolved = planner_hand(gate._robot_config(evidence.arm, evidence.hand, evidence.coupling_mm,
                                                       evidence.planner_margin_mm, rotation))
            start = read_retract(evidence.arm, evidence.hand, float(resolved.coupling_mm),
                                 float(evidence.planner_margin_mm), placement=evidence.approach + evidence.closing)
            _, composed, *_ = compose_for_cell(
                descriptor, bodies=[HandLink.from_hand(resolved, arm=evidence.arm).to_dict(),
                                    *coupling_bodies(resolved, arm=evidence.arm)],
                margin_mm=float(evidence.planner_margin_mm), default_q=start)
            with self.subTest(file=path.name):
                self.assertEqual(evidence.arm_descriptor_sha256, canonical_sha256(descriptor.get("robot_cfg")))
                self.assertEqual(evidence.composed_sha256, canonical_sha256(composed))


class ThePoseHashTests(unittest.TestCase):
    def test_every_file_judged_the_rule_s_poses_from_its_own_retract(self) -> None:
        for path in ruler.evidence_files():
            record = ruler.held(path)
            with self.subTest(file=path.name):
                start = ruler.retract(record)
                self.assertEqual(start, read_retract(record["arm"], record["hand"], float(record["coupling_mm"]),
                                                     float(record["planner_margin_mm"]),
                                                     placement=record["approach"] + record["closing"]))
                poses = ruler.pose_set(start)
                self.assertEqual(record["measured"]["poses"], len(poses))
                self.assertEqual(record["measured"]["poses_sha256"], ruler.poses_sha256(poses))

    def test_the_live_pose_rule_is_the_frozen_one(self) -> None:
        """The gate's own rule against the ruler. Red here on a numpy that fuses ``low + (high - low) * u`` is a
        finding about that numpy, and the ruler above still says what the committed files were measured with."""
        gate = _script("_matrix_gate")
        start = [0.0, -1.0, 0.9, 0.0, 0.0, 0.0]
        live, _ = gate.pose_set(start, arm="ur5e")
        self.assertEqual(ruler.pose_set(start), live)
        self.assertEqual(ruler.poses_sha256(live), gate.poses_sha256(live))

    def test_another_seed_matches_no_file(self) -> None:
        recorded = {ruler.held(path)["measured"]["poses_sha256"] for path in ruler.evidence_files()}
        start = ruler.retract(ruler.held(ruler.evidence_files()[0]))
        self.assertNotIn(ruler.poses_sha256(ruler.pose_set(start, seed=ruler.POSE_SEED + 1)), recorded)


if __name__ == "__main__":
    unittest.main()
