"""Asking the config about itself — the three questions reading the YAML cannot answer.

  1. *Which layer set this?* The merge returns the value and forgets the file.
  2. *Why THIS number?* The comments carry the measured evidence and ``yaml.safe_load`` drops all 469
     of them.
  3. *What am I even allowed to set?* 107 schema fields appear in no shipped YAML — including
     ``robot.ur.model``, which decides whether a real UR3e is planned as a UR3e or as a UR5e.

Everything here is read-only and derived from the tree as it stands, so the tests double as a guarantee
that explaining cannot alter what is loaded.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.config.__main__ import main
from src.config._schema_index import field_doc, schema_index
from src.config.explain import explain, explain_key, find_keys

DATA = Path(__file__).resolve().parents[1] / "config"


class SchemaIndexTests(unittest.TestCase):
    def test_it_finds_fields_that_no_yaml_mentions(self) -> None:
        """The measured discoverability gap, and the reason `where` searches the schema not the files."""
        index = schema_index()
        self.assertIn("robot.ur.model", index)
        self.assertIn("robot.gripper.vacuum.vacuum_ok_input_pin", index)

    def test_a_dict_of_models_is_indexed_once_under_a_wildcard(self) -> None:
        """`robot.sim.cameras` is dict[str, SimCameraSchema]: every key accepts the same shape, so one
        entry describes them all rather than the index depending on which cameras happen to exist."""
        self.assertIn("robot.sim.cameras.*.hfov_deg", schema_index())

    def test_constraints_and_defaults_are_carried(self) -> None:
        field = schema_index()["robot.safety.self_collision.planner_margin_mm"]
        self.assertEqual(field.default, 0.0)
        self.assertIn(">= 0.0", field.constraints)

    def test_it_does_not_hang_on_the_real_schema(self) -> None:
        """Depth-capped and ref-cycle-guarded: a self-referencing model must not spin the CLI."""
        self.assertGreater(len(schema_index()), 400)


class FieldDocTests(unittest.TestCase):
    def test_the_source_comment_is_recovered(self) -> None:
        """Willy documents fields with `#:` comments, which Pydantic does NOT lift into the JSON schema
        — so the explanation existed, was maintained, and was invisible to every tool."""
        doc = field_doc("robot.ur.model")
        self.assertIn("Not cosmetic", doc)

    def test_an_unknown_path_returns_empty_rather_than_raising(self) -> None:
        self.assertEqual(field_doc("robot.nope.nope"), "")


class StructuredExplanationTests(unittest.TestCase):
    """The text and the structured view must be the same answer, because they now have two consumers.

    ``explain_key`` renders; ``explain`` returns the fields the operator console serialises. If those
    were two derivations, the browser and the terminal could disagree about which layer set a value --
    the exact confusion this module exists to remove. So there is one derivation and one renderer, and
    these tests hold that shape in place.
    """

    def test_the_cli_text_is_produced_from_the_structured_view_and_nothing_else(self) -> None:
        for key, layers in [
            ("robot.safety.self_collision.planner_margin_mm", ("sim", "ur3e")),
            ("robot.ur.model", ()),
            ("robot.sim.robot_modell", ("sim",)),
        ]:
            with self.subTest(key=key):
                self.assertEqual(explain(key, DATA, layers).render(), explain_key(key, DATA, layers))

    def test_the_structured_view_carries_the_facts_the_text_states(self) -> None:
        detail = explain("robot.safety.self_collision.planner_margin_mm", DATA, ("sim", "ur3e"))
        self.assertTrue(detail.known)
        self.assertIn("robot.ur3e.yaml", detail.set_in)
        self.assertEqual(len(detail.layers), 2)
        self.assertTrue(detail.layers[-1].winner)
        self.assertFalse(detail.layers[0].winner)
        self.assertIn("no plan at all above roughly 6 mm", detail.comment)

    def test_a_value_that_is_legitimately_none_can_still_be_shown_as_set(self) -> None:
        """``has_value`` exists because ``None`` is a real value here -- an unconfigured serial IS null.

        Defaulting to ``value is not None`` is right for the CLI, which passes the loaded value or
        nothing. A caller holding a genuine ``None`` says so and gets it rendered rather than omitted.
        """
        self.assertIn("= None", explain("robot.ur.model", DATA, (), None, has_value=True).render())
        self.assertNotIn("= None", explain("robot.ur.model", DATA, (), None).render())


class ExplainTests(unittest.TestCase):
    def test_it_names_the_winning_layer_and_the_whole_chain(self) -> None:
        """THE question the merge could not answer. `planner_margin_mm` is set by the sim layer and then
        overridden by the robot layer; a reader of the merged value sees neither."""
        text = explain_key(
            "robot.safety.self_collision.planner_margin_mm", DATA, ("sim", "ur3e"), value=4.0,
        )
        self.assertIn("robot.ur3e.yaml", text)
        self.assertIn("[layer: ur3e]", text)
        self.assertIn("layer chain", text)
        self.assertIn("robot.sim.yaml", text)          # the value it overrode
        self.assertIn("<- winner", text)

    def test_it_surfaces_the_measured_comment_from_the_winning_file(self) -> None:
        """The whole point. The WHY is written above the value and thrown away at load; showing it at
        the moment of confusion costs nothing and moves nothing."""
        text = explain_key(
            "robot.safety.self_collision.planner_margin_mm", DATA, ("sim", "ur3e"), value=4.0,
        )
        self.assertIn("why (comment above that line)", text)
        self.assertIn(
            "no plan at all above roughly 6 mm", text,
            "the measured evidence must survive into the answer",
        )

    def test_a_never_written_field_says_so_instead_of_pretending(self) -> None:
        text = explain_key("robot.ur.model", DATA, (), value="ur5e")
        self.assertIn("no YAML sets this", text)
        self.assertIn("Not cosmetic", text)  # ...but its meaning is still explained

    def test_an_unknown_key_is_named_as_such_with_a_suggestion(self) -> None:
        text = explain_key("robot.sim.robot_modell", DATA, ("sim",))
        self.assertIn("NOT A KNOWN KEY", text)
        self.assertIn("did you mean: robot.sim.robot_model?", text)

    def test_a_base_only_value_reports_the_base_file(self) -> None:
        text = explain_key("robot.workspace_limits.z_min", DATA, (), value=0.0)
        self.assertIn("robot.yaml", text)
        self.assertNotIn("[layer:", text)

    def test_a_camera_key_resolves_through_the_wildcard(self) -> None:
        """An explain of ONE camera must find the shape shared by all of them."""
        text = explain_key("robot.sim.cameras.overhead.hfov_deg", DATA, ("sim", "ur3e", "tiltcam"))
        self.assertNotIn("NOT A KNOWN KEY", text)
        self.assertIn("robot.tiltcam.yaml", text)


class FindKeysTests(unittest.TestCase):
    def test_searching_the_schema_beats_grepping_the_files(self) -> None:
        """Measured: grepping the YAML for "gripper" returns 5 hits and misses BOTH digital-I/O
        end-effectors, because the files only contain what someone chose.

        Searched per end-effector rather than through the broad "gripper" match on purpose. That
        broad search now returns 55 keys against a 40-line display limit, so which wiring block lands
        above the cut is an accident of alphabetical order -- and a test that depends on that accident
        fails the next time somebody adds a config block, which is exactly what happened when
        ``jaw_io`` landed (2026-08-17). The CLAIM is that the schema knows about wiring the YAML never
        mentions; asserting it per block is what actually tests that claim.
        """
        broad = find_keys("gripper")
        self.assertIn("key(s) match", broad)
        self.assertIn("robot.gripper.vacuum.vacuum_output_pin", find_keys("vacuum"))
        self.assertIn("robot.gripper.jaw_io.close_output_pin", find_keys("jaw_io"))

    def test_every_digital_io_wiring_key_is_discoverable(self) -> None:
        """A wiring number nobody can FIND is a number nobody will measure.

        Both I/O drivers are built on the trade "every wiring value is config, so the driver can ship
        before the hardware is chosen". That trade only pays if an operator can list what to measure
        without reading the source -- so this pins discoverability of the whole set, not a sample.
        """
        jaw = find_keys("jaw_io")
        for key in ("actuation", "close_output_pin", "open_output_pin", "part_present_input_pin",
                    "closed_confirm_input_pin", "open_confirm_input_pin", "io_port",
                    "close_timeout_s", "close_settle_s", "closed_below_mm", "pulse_s",
                    "open_on_connect_without_feedback"):
            with self.subTest(key=key):
                self.assertIn(f"robot.gripper.jaw_io.{key}", jaw)

    def test_a_miss_suggests_something_rather_than_shrugging(self) -> None:
        self.assertIn("did you mean", find_keys("grippr"))


class CliTests(unittest.TestCase):
    def test_explain_exits_zero(self) -> None:
        self.assertEqual(main(["explain", "robot.ur.model"]), 0)

    def test_where_needs_no_config_tree(self) -> None:
        """`where` is schema-only on purpose: it must still work when the tree is broken, which is
        exactly when someone is hunting for the right key name."""
        self.assertEqual(main(["where", "vacuum"]), 0)

    def test_the_profile_flag_works_on_both_sides_of_the_subcommand(self) -> None:
        """A tool nobody can invoke correctly is not an ergonomics improvement."""
        self.assertEqual(main(["explain", "robot.sim.robot_model", "--profile", "sim,ur3e"]), 0)
        self.assertEqual(main(["--profile", "sim,ur3e", "explain", "robot.sim.robot_model"]), 0)

    def test_plain_validation_still_works(self) -> None:
        self.assertEqual(main([]), 0)



class DecisionsTests(unittest.TestCase):
    """Only what someone CHOSE, not the 876 lines of restated defaults.

    Measured independently before this tool existed: 3 of 310 robot leaves differ from their schema
    default. A 448-line `robot.yaml` encodes three decisions, and `--print` gave no way to tell those
    three from the 307 restatements -- which is why reading the config did not tell you what the cell
    was configured to do.
    """

    def _decisions(self, layers=(), section="robot"):
        from src.config import load_config, reload_config
        from src.config.explain import decisions
        from src.config.loader import join_profiles, set_active_profile

        set_active_profile(join_profiles(*layers) if layers else None)
        reload_config()
        self.addCleanup(reload_config)
        self.addCleanup(set_active_profile, None)
        return decisions(load_config(None), DATA, layers, section=section)

    def test_the_production_tree_has_exactly_three_robot_decisions(self) -> None:
        """The headline measurement, now asserted. If this number moves, someone made a decision --
        which is precisely what the view is for."""
        text = self._decisions()
        self.assertIn("3 value(s)", text)
        self.assertIn("robot.workspace_limits.z_min", text)
        self.assertIn("robot.kuka.model", text)

    def test_every_decision_carries_its_origin(self) -> None:
        self.assertIn("robot.yaml:", self._decisions())

    def test_a_profile_chain_shows_what_the_layers_decided(self) -> None:
        text = self._decisions(("sim", "ur3e"))
        self.assertIn("robot.sim.robot_model", text)
        self.assertIn("[layer: ur3e]", text)

    def test_container_type_churn_is_not_reported_as_a_change(self) -> None:
        """`model_dump()` renders a tuple field whose default is written as a list; a plain `==` calls
        those different and would fill the view with values nobody set."""
        from src.config._schema_index import same_value

        self.assertTrue(same_value((1.0, 2.0), [1.0, 2.0]))
        self.assertFalse(same_value((1.0, 2.0), [1.0, 3.0]))

    def test_default_factory_fields_resolve_to_their_real_default(self) -> None:
        """The JSON schema omits `default` for every `default_factory` field, so a schema-only diff
        reports them all as changed -- turning the view back into noise."""
        from src.config._schema_index import field_default

        self.assertIsNotNone(field_default("robot.grasping.recovery.apply_modes"))

    def test_the_section_filter_narrows_it(self) -> None:
        text = self._decisions(("sim",), section="robot.safety")
        self.assertNotIn("robot.calibration", text)

    def test_cli_exits_zero(self) -> None:
        self.assertEqual(main(["decisions", "--section", "robot"]), 0)

if __name__ == "__main__":
    unittest.main()


class RobustnessTests(unittest.TestCase):
    """Two defects the config-legacy audit found in this very tool.

    Both matter more than their size: a tool whose entire job is to explain a confusing config must not
    crash on the confusing config, and must never contradict itself about whether a key exists.
    """

    def test_an_aliased_field_resolves_from_its_attribute_name(self) -> None:
        """The stereomatcher block keeps OpenCV's camelCase in YAML behind snake_case attributes -- the
        only aliased fields in the tree. Asked by attribute name, `explain` used to print the VALUE and
        say NOT A KNOWN KEY in the same breath."""
        text = explain_key("camera.stereomatcher.num_disparities", DATA, (), value=320)
        self.assertNotIn("NOT A KNOWN KEY", text)
        self.assertIn("type", text)

    def test_the_alias_map_only_covers_genuinely_aliased_fields(self) -> None:
        from src.config._schema_index import alias_for

        self.assertEqual(alias_for("camera.stereomatcher.num_disparities"),
                         "camera.stereomatcher.numDisparities")
        self.assertIsNone(alias_for("robot.sim.robot_model"))

    def test_output_survives_a_cp1252_console(self) -> None:
        """The config's own comments use box-drawing characters; a stock Windows console is cp1252, and
        printing an explanation raised UnicodeEncodeError."""
        from src.config.__main__ import _emit

        class _Cp1252Stdout:
            """A real cp1252 console: it rejects only what cp1252 genuinely cannot encode."""

            encoding = "cp1252"

            def __init__(self) -> None:
                self.written: list[str] = []

            def write(self, s: str) -> int:
                s.encode("cp1252")  # raises UnicodeEncodeError exactly where a real console does
                self.written.append(s)
                return len(s)

            def flush(self) -> None:
                pass

        import contextlib
        import io

        console = _Cp1252Stdout()
        with contextlib.redirect_stdout(console):
            try:
                _emit("box: ── dash: —")
            except UnicodeEncodeError:  # pragma: no cover - the regression this guards
                self.fail("_emit must not propagate a console encoding error")
        self.assertTrue(console.written, "the answer must still reach the console, dashes or not")
        self.assertIn("box:", "".join(console.written))
        _ = io

    def test_explaining_a_camera_key_with_a_decorated_comment_does_not_crash(self) -> None:
        self.assertEqual(main(["explain", "camera.cameras.primary_rig_id"]), 0)


class VendorCouplingTests(unittest.TestCase):
    """A safety gap the tiering review found that the plan itself had missed.

    Two cross-field rules coupled `safety.self_collision.kinematics_model` to `sim.robot_model` and to
    `ur.model`. Neither fires when the cell is neither -- and profiles COMPOSE, so that gap was
    reachable: `WILLY_PROFILE=sim,web` loaded cleanly with `vendor='kuka'` and `kinematics_model='ur5e'`,
    and the guard would then have evaluated a KUKA arm against UR5e link lengths. That is exactly the
    failure the sibling rules exist to prevent, arriving through the one door they did not watch.
    """

    @staticmethod
    def _cfg(*, vendor: str, kinematics_model: str | None):
        from src.config.schema.robot.robot_schema import RobotConfig

        return RobotConfig.model_validate({
            "vendor": vendor,
            "safety": {"self_collision": {"kinematics_model": kinematics_model}},
        })

    def test_a_ur_dh_table_is_rejected_on_a_non_ur_cell(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._cfg(vendor="kuka", kinematics_model="ur5e")
        self.assertIn("Universal Robots DH table", str(ctx.exception))

    def test_unset_is_always_fine(self) -> None:
        """The honest configuration for another vendor: leave it unset and let the guard use the capsule
        path. Accepting unset is also what keeps this change inert for every existing cell."""
        self.assertIsNone(self._cfg(vendor="kuka", kinematics_model=None).safety.self_collision.kinematics_model)

    def test_the_composed_profile_that_exposed_it_no_longer_loads(self) -> None:
        from src.config import ConfigError, load_config, reload_config
        from src.config.loader import set_active_profile

        set_active_profile("sim,web")
        reload_config()
        self.addCleanup(reload_config)
        self.addCleanup(set_active_profile, None)
        with self.assertRaises(ConfigError) as ctx:
            load_config(None)
        self.assertIn("vendor='kuka'", str(ctx.exception))


class AliasedProvenanceTests(unittest.TestCase):
    def test_an_aliased_key_finds_the_line_that_sets_it(self) -> None:
        """Measured: all 7 camelCase stereomatcher keys reported "no YAML sets this" while the file
        plainly set them -- the provenance index is keyed by the YAML spelling, not the attribute."""
        text = explain_key("camera.stereomatcher.num_disparities", DATA, (), value=320)
        self.assertIn("stereomatcher.yaml:", text)
        self.assertNotIn("no YAML sets this", text)
        self.assertIn("multiple of 16", text)  # ...and its comment comes with it


class EveryFieldIsExplainedTests(unittest.TestCase):
    """A documentation gate: no config field may be undocumented.

    Measured before this landed: 396 of 545 fields had no explanation ANY tool could reach. Almost none
    of that was missing prose -- it was prose the harvester could not see, for reasons that were all
    formatting or typing rather than content:

      * plain ``#`` comments above a field (the harvester took only ``#:``);
      * TRAILING comments on the field's own line (``name: str = "cube"  # identity handle``);
      * list-item paths (``objects[].color``) -- the walk SKIPPED the ``[]`` segment and then looked for
        the item's fields on the containing model;
      * unresolved ForwardRef annotations (``"X | None"``), which stopped the walk entirely;
      * aliased paths (``uniquenessRatio``), which the model walk keys by attribute name.

    So 389 of the 396 were fixed by looking properly, and only 7 needed writing. Keeping this at zero is
    what stops the gap reopening one convenient field at a time.
    """

    def test_no_config_field_is_without_an_explanation(self) -> None:
        from src.config._provenance import comment_above, index_chains
        from src.config._schema_index import field_doc, model_doc, schema_index

        index = schema_index()
        chains = index_chains(DATA, ("sim", "ur3e", "tiltcam"))
        undocumented = []
        for path, field in sorted(index.items()):
            chain = chains.get(path) or chains.get(field.path)
            explained = (
                field.description
                or field_doc(path)
                or field_doc(field.path)
                or (chain and comment_above(chain[-1]))
                or model_doc(path)
            )
            if not explained:
                undocumented.append(path)
        self.assertEqual(
            undocumented, [],
            "these config fields have no explanation a user could reach with `config explain`; "
            "add a comment above the field (or a block docstring) rather than leaving a bare type",
        )

    def test_the_harvester_reads_every_style_this_project_uses(self) -> None:
        from src.config._schema_index import field_doc

        # `#:` marker
        self.assertIn("Not cosmetic", field_doc("robot.ur.model"))
        # plain `#` block
        self.assertIn("Lula config", field_doc("robot.sim.robot_model"))
        # trailing comment on the field's own line, reached THROUGH a list-item segment
        self.assertIn("identity handle", field_doc("robot.sim.scene_setup.objects[].name"))
        # an aliased path resolves back to the attribute that declares it
        self.assertIn("runner-up", field_doc("camera.stereomatcher.uniquenessRatio"))

    def test_a_forward_ref_annotation_does_not_stop_the_walk(self) -> None:
        """`recovery.fixture` is annotated as the string "X | None"; an unresolved ForwardRef used to
        make the entire block read as undocumented for a typing reason."""
        from src.config._schema_index import model_doc

        self.assertIn("Operator-bounded envelope", model_doc("robot.grasping.recovery.fixture.center_mm"))


class TierTests(unittest.TestCase):
    """Tiers make the long tail invisible WITHOUT making it unsettable.

    That distinction is the whole design: removing a value from a FILE removes nothing; removing a
    FIELD removes a capability under `extra='forbid'`. A tier is a display filter and must never become
    the second thing by accident.
    """

    def test_a_safety_field_is_never_hidden_even_behind_a_disabled_block(self) -> None:
        """A safety bound under a switched-off block is still a safety bound."""
        from src.config._tiers import tier_for

        self.assertEqual(
            tier_for("robot.safety.self_collision.min_distance_mm", gated_off=True), "safety")

    def test_a_required_field_is_never_demoted(self) -> None:
        """63 fields have no default. Calling one "long tail" invites defaulting it later, which trades
        a hard load failure for a silent value in the FAIL-OPEN direction."""
        from src.config._tiers import tier_for

        self.assertEqual(tier_for("camera.stereomatcher.numDisparities", required=True), "site")
        self.assertEqual(
            tier_for("camera.stereomatcher.numDisparities", required=True, gated_off=True), "site")

    def test_a_decided_value_is_a_cell_fact(self) -> None:
        from src.config._tiers import tier_for

        self.assertEqual(tier_for("robot.sim.robot_model", decided=True), "site")

    def test_the_switched_off_long_tail_is_advanced(self) -> None:
        from src.config._tiers import tier_for

        self.assertEqual(tier_for("robot.grasping.ordering.strategy", gated_off=True), "advanced")
        self.assertEqual(tier_for("robot.grasping.ordering.strategy"), "tuned")

    def test_gate_state_follows_the_CHAIN_being_viewed_not_the_schema_default(self) -> None:
        """`robot.sim.enabled` defaults to false but the `sim` layer turns it on. Calling a sim cell's
        own fields "advanced" while looking at that cell would be exactly backwards."""
        from src.config._schema_index import schema_index
        from src.config._tiers import gate_state

        index = schema_index()
        self.assertTrue(gate_state("robot.sim.robot_model", index, None))          # no tree: default
        self.assertFalse(gate_state("robot.sim.robot_model", index, {"robot.sim.enabled": True}))

    def test_where_groups_by_tier(self) -> None:
        text = find_keys("gripper")
        self.assertIn("[advanced]", text)
        self.assertIn("[tuned]", text)

    def test_a_filtered_listing_says_out_loud_that_nothing_is_lost(self) -> None:
        """The note only appears when something IS hidden -- and then it must say plainly that hiding is
        a display choice, so nobody reads a short listing as a shrunken config."""
        text = find_keys("gripper", tier="advanced")
        self.assertIn("more in other tiers", text)
        self.assertIn("stays settable", text)

    def test_filtering_where_does_not_change_what_the_config_accepts(self) -> None:
        """The filter is cosmetic: a field hidden from a listing must still validate."""
        from src.config.schema.robot.sim_schema import SimConfig

        hidden = find_keys("gripper", tier="safety")
        self.assertNotIn("gripper_mount", hidden)
        self.assertEqual(SimConfig(gripper_mount="robotiq_2f85").gripper_mount, "robotiq_2f85")

    def test_cli_accepts_the_tier_flag(self) -> None:
        self.assertEqual(main(["where", "gripper", "--tier", "advanced"]), 0)
        self.assertEqual(main(["decisions", "--section", "robot.safety", "--tier", "safety"]), 0)


class CliContractTests(unittest.TestCase):
    """The five defects that writing the user guide exposed in this CLI, pinned against RECURRENCE.

    Every one of them survived an existing test, because that test asserted ``main([...]) == 0``. A flag
    that is parsed and then dropped on the floor still exits 0; a banner that lies about which profile it
    loaded still exits 0. So each test here asserts on the OUTPUT, never on the exit code alone.
    """

    @staticmethod
    def _run(argv: list[str]) -> tuple[int, str]:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_where_tier_actually_filters(self) -> None:
        """`--tier` was parsed and never forwarded, so the flag the help text advertised did nothing."""
        _, everything = self._run(["where", "grasping"])
        _, safety_only = self._run(["where", "grasping", "--tier", "safety"])
        self.assertIn("[safety]", safety_only)
        self.assertNotIn("[advanced]", safety_only)
        self.assertIn("[advanced]", everything)

    def test_where_limit_can_raise_the_per_tier_cap(self) -> None:
        """The listing stopped at 40 keys per tier with no CLI escape, hiding 102 of 188 matches."""
        _, capped = self._run(["where", "grasping"])
        _, raised = self._run(["where", "grasping", "--limit", "500"])
        self.assertIn("more", capped)
        self.assertNotIn("... and", raised)
        self.assertGreater(raised.count("\n        "), capped.count("\n        "))

    def test_the_validate_banner_reports_the_chain_it_actually_loaded(self) -> None:
        """It read only `--profile`, so WILLY_PROFILE=sim printed "(no profile)" while the sim overlays
        were applied -- the banner denying the layering it had just performed."""
        from src.config.loader import reload_config, set_active_profile

        try:
            set_active_profile("sim")
            reload_config()
            _, text = self._run([])
            self.assertIn("layers: sim", text)
            self.assertNotIn("(no profile)", text)
        finally:
            set_active_profile(None)
            reload_config()

    def test_the_banner_survives_a_console_that_cannot_encode_a_dash(self) -> None:
        """A bare print() with U+2014 crashed with UnicodeEncodeError -- exit 1 on a VALID config."""
        code, text = self._run([])
        self.assertEqual(code, 0)
        text.encode("cp850")  # raises UnicodeEncodeError if a non-cp850 character crept back in

    def test_profile_works_on_both_sides_of_the_subcommand(self) -> None:
        """`--profile` BEFORE the subcommand was silently discarded: argparse shares Action objects
        across `parents=`, and set_defaults MUTATES them, undoing the SUPPRESS that made it work."""
        _, before = self._run(["--profile", "sim", "explain", "robot.sim.assets_root"])
        _, after = self._run(["explain", "robot.sim.assets_root", "--profile", "sim"])
        _, neither = self._run(["explain", "robot.sim.assets_root"])
        self.assertIn("[layer: sim]", before)
        self.assertIn("[layer: sim]", after)
        self.assertIn("no YAML sets this", neither)
