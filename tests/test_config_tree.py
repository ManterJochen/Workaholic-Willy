"""The config tree as one value, and the two disagreements that made it necessary.

Both assertions below correspond to something measured on 2026-09-04, not to a shape I preferred.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.config.tree import ConfigTree, LoadedTree, default_data_dir


class DerivationTests(unittest.TestCase):
    def test_layers_are_derived_from_the_chain_and_cannot_be_supplied(self) -> None:
        """⛔ THE WHOLE POINT. `layers` and `profile` are one fact in two shapes. Every measured
        defect in this area came from a signature that let a caller pass both.
        """
        tree = ConfigTree.from_directory(profile="sim")
        self.assertEqual(tree.profile, "sim")
        self.assertEqual(tree.layers, ("sim",))

        import inspect

        params = inspect.signature(ConfigTree.from_directory).parameters
        self.assertNotIn("layers", params, "layers must be derived, never passed")

    def test_unset_and_none_are_different_profiles(self) -> None:
        """⚠ THREE STATES, ALL MEANINGFUL. `None` is "the base tree, ignore the variable"; UNSET is
        "the caller did not choose", which lets WILLY_PROFILE decide. Collapsing them silently
        disables an exported variable for an operator who set it deliberately.
        """
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"WILLY_PROFILE": "sim"}):
            self.assertEqual(ConfigTree.from_directory().layers, ("sim",))
            self.assertEqual(ConfigTree.from_directory(profile=None).layers, ())

    def test_the_default_root_has_a_public_name(self) -> None:
        """⛔ There was none, and four places rebuilt it independently, while `explain`, `decisions`
        and `set_keys` all take `root` as a REQUIRED argument whose wrongness is silent.
        """
        self.assertTrue(default_data_dir().is_dir())
        self.assertEqual(ConfigTree.from_directory().root, default_data_dir())


class ProvenanceTests(unittest.TestCase):
    def test_the_value_and_the_line_that_set_it_agree(self) -> None:
        """⛔⛔ THE MEASUREMENT THAT PAID FOR THIS CLASS.

            explain_in(load_config(profile="sim"), "robot.sim.enabled", <root>, ())
              ->  robot.sim.enabled = True
                  set in    robot/robot.yaml:31        <- that line sets `false`

        The value came out of the config and the provenance out of the empty `layers`, and nothing
        bound them. Through the tree the two cannot come from different places, so this asserts the
        pairing rather than the text: under the `sim` chain the winning file must be the sim overlay.
        """
        loaded = ConfigTree.from_directory(profile="sim").load()
        self.assertTrue(loaded.ok, loaded.error)
        text = loaded.explain("robot.sim.enabled").render()
        self.assertIn("robot.sim.enabled = True", text)
        self.assertIn("robot.sim.yaml", text)
        self.assertIn("[layer: sim]", text)

        base = ConfigTree.from_directory(profile=None).load()
        base_text = base.explain("robot.sim.enabled").render()
        self.assertIn("robot.sim.enabled = False", base_text)
        self.assertNotIn("robot.sim.yaml", base_text)


class VerdictTests(unittest.TestCase):
    def test_a_tree_that_does_not_load_is_a_verdict_not_an_exception(self) -> None:
        loaded = ConfigTree.from_directory(profile="nosuch").load()
        self.assertFalse(loaded.ok)
        self.assertEqual(loaded.exit_code, 1)
        self.assertIn("nosuch", loaded.error)

    def test_the_refusal_names_what_the_caller_typed(self) -> None:
        """⭐ `_validated_chain` carries a `source` argument for exactly one purpose, and the CLI's
        set-the-variable-then-restore-it dance defeated it: `--profile nosuch` blamed WILLY_PROFILE
        for a value nobody had exported. Loading with the chain as an ARGUMENT is what fixes it.
        """
        loaded = ConfigTree.from_directory(profile="nosuch").load()
        self.assertIn("profile='nosuch'", loaded.error)
        self.assertNotIn("WILLY_PROFILE", loaded.error)

    def test_render_reports_the_chain_that_was_loaded(self) -> None:
        """⚠ Not the one that was asked for. The banner used to print "(no profile)" while
        WILLY_PROFILE=sim was in force and its overlays had already been applied.
        """
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"WILLY_PROFILE": "sim"}):
            text = ConfigTree.from_directory().load().render()
        self.assertIn("layers: sim", text)
        self.assertFalse(text.endswith("\n"))

    def test_to_dict_is_json_safe_and_omits_the_config(self) -> None:
        import json

        payload = ConfigTree.from_directory(profile="sim").load().to_dict()
        json.dumps(payload)
        self.assertEqual(payload["layers"], ["sim"])
        self.assertEqual(payload["exit_code"], 0)
        self.assertNotIn("config", payload)


class OneDerivationTests(unittest.TestCase):
    def test_the_cli_no_longer_rebuilds_the_root_or_the_layers(self) -> None:
        """⛔ It built the same two expressions twice, four lines apart, one per query branch. This
        is the cheapest way to notice a third copy appearing.
        """
        body = (Path(__file__).resolve().parents[1] / "src" / "config" / "__main__.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('Path(__file__).resolve().parent / "data"', body)
        self.assertNotIn("set_active_profile", body, "the env dance came back")
        self.assertNotIn("profile_layers(", body, "the CLI derives layers again")

    def test_loaded_tree_binds_config_root_and_layers(self) -> None:
        """`explain` and `decisions` take only what they are about; the three facts they need come
        from the object that holds all three."""
        import inspect

        for name in ("explain", "decisions"):
            params = set(inspect.signature(getattr(LoadedTree, name)).parameters)
            self.assertNotIn("root", params)
            self.assertNotIn("layers", params)
            self.assertNotIn("cfg", params)



class WritePathTests(unittest.TestCase):
    """The two write-path defects, both measured on 2026-09-04 on a copy of the shipped tree."""

    def setUp(self) -> None:
        import shutil
        import tempfile

        from src.config.tree import default_data_dir

        self.tmp = Path(tempfile.mkdtemp(prefix="cfgtree-"))
        self.root = self.tmp / "data"
        shutil.copytree(default_data_dir(), self.root)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_write_cannot_land_in_a_file_a_different_chain_validates(self) -> None:
        """⛔⛔ MEASURED, AND IT REPORTED SUCCESS.

            set_key("robot.safety.payload.mass_kg", 3.25, root=T, layers=("ur3e",), profile=None)
              -> applied=True, refused=None, values={...: 0.0}

        It picked the target file from `layers[-1]` and wrote `mass_kg: 3.25` into robot.ur3e.yaml
        directly beneath a `max_mass_kg: 3.0` that forbids it, then validated the BASE tree because
        `profile` said None, so no rollback fired. The value read back was the base tree's 0.0, not
        the one written, and the tree no longer loaded under its own layer, against the module's own
        promise that this cannot happen.
        """
        from src.config.edit import set_key

        with self.assertRaises(ValueError) as caught:
            set_key(
                "robot.safety.payload.mass_kg", 3.25,
                root=self.root, layers=("ur3e",), profile=None,
            )
        self.assertIn("different chains", str(caught.exception))

    def test_the_tree_makes_that_call_unconstructible(self) -> None:
        """The guard above is for the raw function. Through the tree there is no way to say it."""
        import inspect

        params = inspect.signature(ConfigTree.write).parameters
        self.assertEqual(set(params) - {"self"}, {"items", "connected"})

    def test_the_write_reads_back_what_it_wrote(self) -> None:
        """The other half of the same defect: `_reload` validated the wrong tree, so the read-back
        reported the base tree's value while the overlay held the new one."""
        tree = ConfigTree(root=self.root, profile="ur3e", layers=("ur3e",))
        result = tree.write({"robot.safety.payload.mass_kg": 2.5}, connected=False)
        self.assertTrue(result.applied, result.message)
        self.assertEqual(result.values["robot.safety.payload.mass_kg"], 2.5)

    def test_connected_has_no_default_so_nobody_can_forget_it(self) -> None:
        """⛔⛔ THE SAFETY DEFECT. `set_keys` declares `connected: bool = False`, the permissive
        value, and its only production caller never passed it -- so the guard on the keys that decide
        WHICH MACHINE RECEIVES EVERY MOTION could not fire through the browser. A default meaning
        "no guard" is a guard nobody has to switch off.
        """
        import inspect

        param = inspect.signature(ConfigTree.write).parameters["connected"]
        self.assertIs(param.default, inspect.Parameter.empty)
        self.assertIs(param.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_the_guard_actually_refuses_a_live_cell(self) -> None:
        from src.config.edit import WriteRefused

        tree = ConfigTree(root=self.root, profile=None, layers=())
        result = tree.write({"robot.ur.ip": "10.0.0.9"}, connected=True)
        self.assertFalse(result.applied)
        self.assertIs(result.refused, WriteRefused.CELL_CONNECTED)

    def test_the_router_maps_that_refusal_instead_of_raising_KeyError(self) -> None:
        """⚠ It was missing from the status map, invisible for as long as the refusal could not fire.
        The moment `connected=` started being passed, a correct refusal would have become a 500."""
        body = (
            Path(__file__).resolve().parents[1] / "api" / "routers" / "config.py"
        ).read_text(encoding="utf-8")
        self.assertIn("WriteRefused.CELL_CONNECTED:", body)
        self.assertIn("connected=cell.session.connected", body)

class FlagsThatCannotApplyTests(unittest.TestCase):
    """⛔ A FLAG THAT CANNOT APPLY IS REJECTED, NOT IGNORED.

    The rule is the CLI's own, written at the `where` branch about `--tier`/`--limit`: "a filter that
    is accepted and ignored is worse than no filter: it answers a question you did not ask." Two more
    flags were breaking it, both measured on 2026-09-04, both exiting 0 while doing nothing.
    """

    def _refusal(self, argv: list[str]) -> str:
        import contextlib
        import io

        from src.config import __main__ as cli

        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            cli.main(argv)
        self.assertEqual(caught.exception.code, 2, "argparse's documented bad-arguments code")
        return err.getvalue()

    def test_print_is_refused_on_every_subcommand_and_on_both_sides(self) -> None:
        """It rode the shared parent onto all three and was read only after each had returned, so
        `explain KEY --print` printed the explanation, no JSON, exit 0."""
        for argv in (
            ["explain", "robot.sim.robot_model", "--print"],
            ["--print", "explain", "robot.sim.robot_model"],
            ["decisions", "--print"],
            ["where", "gripper", "--print"],
        ):
            with self.subTest(argv=argv):
                self.assertIn("--print dumps the whole validated config", self._refusal(argv))

    def test_where_refuses_the_two_flags_that_cannot_reach_it(self) -> None:
        """`find_keys` takes no root: it searches the SCHEMA compiled into this checkout. Pointing
        the tool at a customer's tree used to answer about ours, silently and with exit 0 -- where
        every other branch exits 1 on that same bad chain."""
        for argv in (
            ["where", "gripper", "--data", "D:/nonexistent"],
            ["where", "gripper", "--profile", "nosuch"],
        ):
            with self.subTest(argv=argv):
                self.assertIn("searches the SCHEMA compiled into this checkout", self._refusal(argv))

    def test_the_combinations_that_always_worked_still_do(self) -> None:
        """⚠ The refusals must be narrow. `--print` on the default command and `--tier` on `where`
        are the reason those flags exist."""
        from src.config import __main__ as cli

        for argv in (["--print"], ["where", "gripper", "--tier", "all"], ["explain", "robot.sim.enabled"]):
            with self.subTest(argv=argv):
                import contextlib
                import io

                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(argv), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
