"""The arm registry: one file per arm, and a name that always finds exactly one of them (P0d D2).

`src/config/arms.py` is the third hardware registry beside the hands and the cameras, and it is deliberately
the same shape as both: `<config tree>/arms/<model>.yaml` with a top-level `arm:` key, a refusal that names what
IS known, and a deployment tree that may repeat the repository's file and nothing else.

D2 built it, D3 moved every reader of the hand-kept arm tables onto it, and D4 deleted those tables. The tests here
are about the registry, and `test_arm_descriptions_are_urs_numbers.py` is about the numbers in it.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.config import ConfigError
from src.config.arms import available_arms, curobo_descriptor_name, load_arm, tree_arm_refusal

_DATA = Path(__file__).resolve().parents[1] / "config"

#: Every arm described today. Six a cell may name, and two this repository holds geometry for and does not admit.
_DESCRIBED = ("ur10", "ur10e", "ur16e", "ur20", "ur3", "ur3e", "ur5", "ur5e")


class _ScratchRegistry(unittest.TestCase):
    """A copy of the arm registry alone, so a broken description can be made without touching the tree.

    Registry-only rather than a copy of the whole tree: a scratch root that never reaches `ConfigTree` keeps
    these tests about the registry, which is what `tests/test_gripper_registry.py` does for the same reason.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="willy_arms_")
        self.root = Path(self._tmp.name)
        shutil.copytree(_DATA / "arms", self.root / "arms")
        self.addCleanup(self._tmp.cleanup)

    def _file(self, model: str) -> Path:
        return self.root / "arms" / f"{model}.yaml"


class TheRegistryReadsItsFilesTests(unittest.TestCase):
    def test_every_described_arm_is_there(self) -> None:
        self.assertEqual(sorted(_DESCRIBED), available_arms())

    def test_an_arm_answers_with_its_own_joints(self) -> None:
        arm = load_arm("ur5e")
        self.assertEqual("ur", arm.vendor)
        self.assertEqual("ur5e", arm.model)
        self.assertEqual(6, arm.dof)
        self.assertEqual(
            ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            [joint.name for joint in arm.joints],
        )

    def test_dof_is_the_joints_and_never_a_second_number(self) -> None:
        """The whole reason a joint is a row: five names beside six DH rows cannot be written down."""
        for model in _DESCRIBED:
            with self.subTest(model=model):
                arm = load_arm(model)
                self.assertEqual(len(arm.joints), arm.dof)

    def test_an_arm_no_simulator_can_stand_up_says_so_rather_than_guessing(self) -> None:
        """ur16e has committed exact-mesh geometry and no Isaac asset, which is why it is described."""
        described = load_arm("ur16e")
        self.assertIsNone(described.sim)
        self.assertIsNone(described.reach_mm, "a datasheet figure nobody here checked was written down")
        self.assertIsNone(described.payload_kg)
        self.assertIsNotNone(load_arm("ur5e").sim)

    def test_an_unknown_arm_is_refused_naming_what_is_known(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            load_arm("ur42")
        message = str(caught.exception)
        self.assertIn("ur42", message)
        self.assertIn("ur5e", message, "the refusal must name the arms that DO exist")

    def test_a_name_that_is_not_a_registry_name_is_refused_before_any_file_is_read(self) -> None:
        for name in ("../ur5e", "UR5e", "ur5e.yaml", ""):
            with self.subTest(name=name), self.assertRaises(ConfigError):
                load_arm(name)

    def test_the_descriptor_name_is_derived_rather_than_written_down(self) -> None:
        """Six hand-typed copies of one spelling are six chances to put willy_ur5.yml into ur5e.yaml."""
        self.assertEqual("willy_ur5e.yml", curobo_descriptor_name("ur5e"))
        self.assertEqual("willy_ur16e.yml", curobo_descriptor_name("ur16e"))
        with self.assertRaises(ConfigError):
            curobo_descriptor_name("../etc/passwd")


class ThePlannerIsToldADescriptorByTheArmNotBySimulatorTests(unittest.TestCase):
    """P0d D3a. The planner stack, the doctor and the desk checklist ask the config layer for a descriptor's name.

    Red before 2026-09-22, and wrong in two ways at once. The name came from the sim driver's model table,
    which refused every arm with no Isaac asset. Measured that day: ``python -m src.robot.safety.planning
    --check --model ur16e --hand robotiq_2f85`` answered "unsupported sim robot_model 'ur16e' ... Add a URModelSpec
    (Lula key + Isaac USD relpath) and build the matching cuRobo {key}.yml to support it", while ``willy_ur16e.yml``
    was built and in the content directory. So a planner probe for a real cell was refused by a simulator's table,
    in a simulator's words, and told to build a file that already existed.

    Measured through ``MotionStack.probe``, the path the doctor and the CLI take, rather than through the name
    function alone: the two callers turn a refusal into ``<none: ...>`` and would have reported the old message
    whatever the function underneath said.
    """

    @staticmethod
    def _named(model: str) -> str:
        from src.robot.safety.planning.stack import MotionStack

        return MotionStack.from_model(model=model, hand="robotiq_2f85").probe().environment.curobo.robot_config

    def test_a_described_arm_with_no_isaac_asset_is_named_its_descriptor(self) -> None:
        self.assertEqual("willy_ur16e.yml", self._named("ur16e"))

    def test_an_arm_nobody_describes_is_refused_in_the_arms_own_words(self) -> None:
        named = self._named("ur99")
        self.assertTrue(named.startswith("<none: "), named)
        self.assertIn("ur99", named)
        self.assertIn("Known arms", named)
        self.assertNotIn("sim robot_model", named, "an arm was refused by the simulator's table again")

    def test_the_command_line_still_takes_a_model_in_any_case(self) -> None:
        """``ur_model_spec`` lower-cased its input, so ``--model UR5e`` worked; a strict name would have broken it."""
        self.assertEqual("willy_ur5e.yml", self._named("UR5e"))


class AdmissionIsStatedBesideTheArmTests(unittest.TestCase):
    """P0d D3f. Owner, 2026-09-22: each description says whether a cell may name it; that replaces UR_MODEL_KEYS."""

    _ROOT = Path(__file__).resolve().parents[1]

    def test_every_admitted_arm_carries_its_committed_geometry(self) -> None:
        """THE Claim the flag makes, CHECKED. `admitted: true` is a statement by a person; this is what makes it
        cost something to write. An admitted arm with no bundle would validate at load and meet the exact-mesh guard's
        `no_bundle` at its first move, which is the silent degradation the old admitted-name module warned about.
        """
        from src.config.arms import admitted_arms

        for model in admitted_arms():
            with self.subTest(model=model):
                bundle = self._ROOT / "src/robot/safety/data" / f"{model}_collision_meshes.npz"
                spheres = self._ROOT / "src/robot/safety/planning/robot" / f"{model}_arm_spheres.yml"
                self.assertTrue(bundle.is_file(), f"{model} is admitted and has no collision bundle at {bundle}")
                self.assertTrue(spheres.is_file(), f"{model} is admitted and has no sphere map at {spheres}")

    def test_the_admitted_arms_are_the_six_the_owner_admitted(self) -> None:
        """Written out, because a list derived from the registry cannot tell whether the registry is right.

        Owner, 2026-09-20: exactly today's six. The old gate held the same six until D4 deleted it, so moving the
        gate admitted no arm and dropped none; this is the statement that survives it.
        """
        from src.config.arms import admitted_arms

        six = ["ur10", "ur10e", "ur3", "ur3e", "ur5", "ur5e"]
        self.assertEqual(six, admitted_arms())
        self.assertEqual(six, admitted_arms(simulated=True))

    def test_admission_is_a_statement_and_never_an_implied_value(self) -> None:
        """Which robots a cell may drive is a safety identity, so a description that says nothing does not load."""
        from pydantic import ValidationError

        from src.config.schema.arms import ArmSpec

        tree = load_arm("ur5e").model_dump(mode="json")
        del tree["admitted"]
        with self.assertRaises(ValidationError):
            ArmSpec.model_validate(tree)

    def test_the_validators_ask_the_registry_rather_than_a_list(self) -> None:
        """The discriminating case. Admit ur16e in its description and a real cell may name it; the table is
        unchanged throughout, so only a validator reading the registry can accept it.
        """
        from unittest import mock

        from src.config.schema.robot import RobotConfig
        from src.config import arms

        real = arms._registry(None)
        specs = dict(real[1])
        specs["ur16e"] = specs["ur16e"].model_copy(update={"admitted": True})
        with mock.patch.object(arms, "_registry", return_value=(real[0], specs)):
            RobotConfig.model_validate({"vendor": "ur", "ur": {"model": "ur16e"}})
            with self.assertRaises(Exception) as caught:
                RobotConfig.model_validate({"vendor": "sim", "sim": {"robot_model": "ur16e"}})
        self.assertNotIn("ur16e", arms.admitted_arms(), "the control: unpatched, the registry does not admit it")
        self.assertIn("sim block", str(caught.exception), "an admitted arm with no Isaac asset is still no sim robot")

    def test_an_arm_the_registry_does_not_admit_is_refused_naming_where_admission_lives(self) -> None:
        from src.config.schema.robot import RobotConfig

        with self.assertRaises(Exception) as caught:
            RobotConfig.model_validate({"vendor": "ur", "ur": {"model": "ur16e"}})
        message = str(caught.exception)
        self.assertIn("unknown UR model", message)
        self.assertIn("admitted: true", message)


class TheRegistryRefusesWhatItCannotResolveTests(_ScratchRegistry):
    def test_a_file_named_after_another_arm_is_refused(self) -> None:
        """A name has to find exactly one file, so the stem and the description have to agree."""
        text = self._file("ur5e").read_text(encoding="utf-8").replace('model: "ur5e"', 'model: "ur10e"')
        self._file("ur5e").write_text(text, encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            load_arm("ur5e", data_dir=self.root)
        self.assertIn("ur10e", str(caught.exception))
        self.assertIn("ur5e", str(caught.exception))

    def test_a_file_with_no_top_level_arm_key_is_refused(self) -> None:
        self._file("ur3").write_text("robot:\n  vendor: ur\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            load_arm("ur5e", data_dir=self.root)
        self.assertIn("arm", str(caught.exception))

    def test_one_broken_file_refuses_every_lookup(self) -> None:
        """A registry that is only partly readable cannot say which arm a name means, so it says none."""
        self._file("ur10").write_text("arm:\n  vendor: ur\n  model: \"ur10\"\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_arm("ur5e", data_dir=self.root)
        with self.assertRaises(ConfigError):
            available_arms(self.root)

    def test_a_file_that_is_not_model_yaml_is_refused_by_name(self) -> None:
        (self.root / "arms" / "ur5e.sim.yaml").write_text("arm: {}\n", encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            available_arms(self.root)
        self.assertIn("ur5e.sim.yaml", str(caught.exception))

    def test_a_registry_that_is_not_there_is_refused(self) -> None:
        shutil.rmtree(self.root / "arms")
        with self.assertRaises(ConfigError) as caught:
            available_arms(self.root)
        self.assertIn("no arm registry", str(caught.exception))


class TheSchemaRefusesADescriptionThatContradictsItselfTests(unittest.TestCase):
    """Each of these was writable as four parallel tables and is unwritable as one description."""

    @staticmethod
    def _joint(**over: object) -> dict:
        row = {"name": "shoulder_pan_joint", "link": "shoulder_link", "part": "shoulder", "a_m": 0.0, "d_m": 0.1625,
               "alpha_rad": 1.5707963, "limit_deg": [-360.0, 360.0], "home_rad": 0.0}
        row.update(over)
        # The planner's envelope follows the guard's unless a test states its own, so an envelope test still reaches
        # the validator it is about rather than the planner's.
        row.setdefault("planner_limit_deg", list(row["limit_deg"]))  # type: ignore[call-overload]
        return row

    def _arm(self, **over: object) -> None:
        from src.config.schema.arms import ArmSpec

        # `admitted`, `flange` and `upstream` are stated, so each test here reaches the validator it is about. Without them every
        # one of them raised on the missing field instead, and the three that only assert a ValidationError stayed
        # green for the wrong reason (P0d D3f made `admitted` required, P0e E12 `flange`; two message checks caught
        # it both times).
        tree = {"vendor": "ur", "model": "ur5e", "source": "a test", "admitted": True, "joints": [self._joint()],
                "flange": {"link": "tool0", "approach": "+Z"}, "base": {"link": "base_link", "chain_yaw_deg": 180,
                                                                  "controller_yaw_deg": 180},
                "planner": {"position_limit_clip_rad": 0.1},
                "upstream": {"adapter": "ur", "repo": "https://example.invalid/description.git", "commit": "0" * 40,
                             "licence": "BSD-3-Clause", "manifest_rows_sha256": "0" * 64}}
        tree.update(over)
        ArmSpec.model_validate(tree)

    def test_a_home_outside_its_own_envelope_is_refused(self) -> None:
        """A cell sent home would be refused by the guard before it moved, which is a cell nobody can start."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(limit_deg=[-90.0, 90.0], home_rad=3.14159)])
        self.assertIn("outside its own envelope", str(caught.exception))

    def test_an_empty_envelope_is_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self._arm(joints=[self._joint(limit_deg=[90.0, -90.0])])

    def test_two_joints_with_one_name_are_refused(self) -> None:
        """A joint vector is matched to a controller's ordering by these names; a repeat matches in silence."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(), self._joint()])
        self.assertIn("names a joint twice", str(caught.exception))

    def test_two_joints_moving_one_link_are_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(), self._joint(name="shoulder_lift_joint", part="upper_arm")])
        self.assertIn("names a link or a part twice", str(caught.exception))

    def test_a_part_the_bundle_keeps_for_the_hand_is_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(part="gripper")])
        self.assertIn("keeps for the hand", str(caught.exception))

    def test_a_part_that_splits_like_a_bundle_array_is_refused(self) -> None:
        """``__`` separates a part from its array in a bundle, so ``wrist__1`` would read as part ``wrist``."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(part="wrist__1")])
        self.assertIn(("string_pattern_mismatch", ("joints", 0, "part")),
                      [(error["type"], error["loc"]) for error in caught.exception.errors()])

    def test_a_flange_on_a_link_short_of_the_last_is_refused(self) -> None:
        from pydantic import ValidationError

        joints = [self._joint(), self._joint(name="shoulder_lift_joint", link="upper_arm_link", part="upper_arm")]
        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=joints, flange={"link": "shoulder_link", "approach": "+Z"})
        self.assertIn("only the last joint's link may be it", str(caught.exception))

    def test_a_planner_envelope_wider_than_the_guard_is_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(limit_deg=[-180.0, 180.0], planner_limit_deg=[-360.0, 360.0])])
        self.assertIn("the planner would propose what the guard refuses", str(caught.exception))

    def test_an_empty_planner_envelope_is_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(planner_limit_deg=[10.0, 10.0])])
        self.assertIn("planner_limit_deg", str(caught.exception))

    def test_a_clip_that_closes_a_joint_is_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(planner_limit_deg=[-5.0, 5.0])], planner={"position_limit_clip_rad": 0.1})
        self.assertIn("closes the range of shoulder_pan_joint", str(caught.exception))

    def test_an_arm_with_no_planner_block_is_refused(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.arms import ArmSpec

        with self.assertRaises(ValidationError) as caught:
            ArmSpec.model_validate({"vendor": "ur", "model": "ur5e"})
        self.assertIn("planner", [str(error["loc"][0]) for error in caught.exception.errors()])

    def _anchored(self, *anchors: "float | None", source: "str | None" = "Isaac Lula default_q, a test") -> None:
        names = ("shoulder_pan_joint", "shoulder_lift_joint")
        links = (("shoulder_link", "shoulder"), ("upper_arm_link", "upper_arm"))
        joints = [self._joint(name=name, link=link, part=part, retract_anchor_rad=anchor)
                  for name, (link, part), anchor in zip(names, links, anchors, strict=True)]
        self._arm(joints=joints, retract_anchor_source=source)

    def test_an_anchor_on_some_joints_is_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._anchored(0.0, None)
        self.assertIn("shoulder_lift_joint", str(caught.exception))

    def test_an_anchor_without_its_source_and_a_source_without_an_anchor_are_refused(self) -> None:
        from pydantic import ValidationError

        for anchors, source in (((0.0, -1.0), None), ((None, None), "Isaac Lula default_q, a test")):
            with self.subTest(anchors=anchors, source=source):
                with self.assertRaises(ValidationError) as caught:
                    self._anchored(*anchors, source=source)
                self.assertIn("retract_anchor_source", str(caught.exception))

    def test_an_anchor_outside_the_envelope_is_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as caught:
            self._arm(joints=[self._joint(limit_deg=[-90.0, 90.0], retract_anchor_rad=3.0)],
                      retract_anchor_source="Isaac Lula default_q, a test")
        self.assertIn("has its retract anchor at", str(caught.exception))

    def test_an_arm_with_no_anchor_and_no_source_is_described(self) -> None:
        """THE CONTROL: the key is optional, as ur20's description needs."""
        self._anchored(None, None, source=None)

    def test_an_arm_with_no_joints_is_refused(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self._arm(joints=[])

    def test_a_baked_variant_without_its_hand_is_refused(self) -> None:
        """The sim decides from the pair whether to select a variant or mount a hand standalone."""
        from pydantic import ValidationError

        sim = {"lula_key": "UR5e", "usd_relpath": "/x.usd", "wrist_link": "wrist_3_link",
               "baked_gripper_variant": "Robotiq_2f_85", "workspace_center_mm": [450.0, 0.0],
               "workspace_half_extents_mm": [150.0, 150.0]}
        with self.assertRaises(ValidationError) as caught:
            self._arm(sim=sim, reach_mm=850.0, payload_kg=5.0)
        self.assertIn("names the hand that variant is", str(caught.exception))

    def test_an_arm_a_simulator_stands_up_has_a_reach(self) -> None:
        """The patch a scene generator is handed is sized against the reach, so it cannot be absent."""
        from pydantic import ValidationError

        sim = {"lula_key": "UR5e", "usd_relpath": "/x.usd", "wrist_link": "wrist_3_link",
               "workspace_center_mm": [450.0, 0.0], "workspace_half_extents_mm": [150.0, 150.0]}
        with self.assertRaises(ValidationError) as caught:
            self._arm(sim=sim)
        self.assertIn("sizes its patch against the reach", str(caught.exception))

    def test_a_millimetre_written_where_a_metre_belongs_is_refused(self) -> None:
        """The one DH typo that still builds a plausible robot: 162.5 instead of 0.1625."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self._arm(joints=[self._joint(d_m=162.5)])


class ATreeMayRepeatTheRepositorysFileTests(_ScratchRegistry):
    """The same authority rule the hand and camera registries carry, and for the same reason."""

    def test_the_repositorys_own_tree_is_never_refused(self) -> None:
        self.assertIsNone(tree_arm_refusal("ur5e", data_dir=None))
        self.assertIsNone(tree_arm_refusal("ur5e", data_dir=_DATA))

    def test_a_byte_identical_copy_is_accepted(self) -> None:
        """The load-bearing one. A deployment tree is normally a copytree of the shipped tree, so it carries
        arms/ whether anybody meant it to. `python -m src.config --data <copy>` does exactly this on every
        run of `tests/test_config_cli.py`, and a registry that refused a tree-owned arms/ outright would make
        that exit 1.
        """
        for model in _DESCRIBED:
            with self.subTest(model=model):
                self.assertIsNone(tree_arm_refusal(model, data_dir=self.root))

    def test_a_changed_copy_is_refused_naming_what_differs(self) -> None:
        text = self._file("ur5e").read_text(encoding="utf-8").replace("reach_mm: 850.0", "reach_mm: 1850.0")
        self._file("ur5e").write_text(text, encoding="utf-8")
        said = tree_arm_refusal("ur5e", data_dir=self.root)
        self.assertIsNotNone(said)
        assert said is not None
        self.assertIn("1850.0", said)
        self.assertIn("850.0", said)
        self.assertIn("ur5e.yaml", said)

    def test_a_tree_holding_no_arm_registry_reads_through_the_repositorys(self) -> None:
        """As a tree with no cameras/ does: a description fills none of a tree's own numbers at load."""
        shutil.rmtree(self.root / "arms")
        self.assertIsNone(tree_arm_refusal("ur5e", data_dir=self.root))

    def test_a_tree_that_describes_an_arm_the_repository_does_not_is_refused(self) -> None:
        text = self._file("ur5e").read_text(encoding="utf-8").replace('model: "ur5e"', 'model: "ur42"')
        (self.root / "arms" / "ur42.yaml").write_text(text, encoding="utf-8")
        said = tree_arm_refusal("ur42", data_dir=self.root)
        self.assertIsNotNone(said)
        assert said is not None
        self.assertIn("ur42", said)
        self.assertIn("the repository registry", said)

    def test_a_tree_missing_the_arm_its_cell_names_is_told_which_file_to_copy(self) -> None:
        self._file("ur5e").unlink()
        said = tree_arm_refusal("ur5e", data_dir=self.root)
        self.assertIsNotNone(said)
        assert said is not None
        self.assertIn("holds no arm", said)
        self.assertIn("ur5e.yaml", said)


if __name__ == "__main__":
    unittest.main()
