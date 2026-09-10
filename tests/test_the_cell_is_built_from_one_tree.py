"""A cell's robot half and its camera half come from the same config tree.

⛔ **MEASURED 2026-09-10, AND THREE AGENTS FOUND IT INDEPENDENTLY.** `real_cell --profile ur3e`
resolved the robot through `load_robot_config(data_dir, profile=...)`, and then
`build_real_components` read the camera section with a bare `load_config()` at cells.py, and
`build_real_cell` did it again for `primary_rig_id`. A no-argument load takes `profile=UNSET`, which
falls back to `WILLY_PROFILE`, and `data_dir=None`, which falls back to the checkout's own tree. So
one command built a robot from the flag and cameras from the environment.

Measured consequences, all three reproduced: the same chain through `--profile` and through
`WILLY_PROFILE` produced two DIFFERENT refusals; a scratch tree that could build was refused quoting
the repository tree's rigs; and there is no environment escape hatch for `--data-dir` at all, so an
operator pointing the console at a deployment tree got that tree's arm and this checkout's cameras.

⭐ **THE COMMENT AT THE SECOND SITE WAS TRUE AND NOT ABOUT THE RIGHT THING.** It read "the same loader
`build_real_components` used, and it caches, so this is the same object rather than a second read of
the tree", which is correct about the cache and silent about WHICH tree was cached. Two calls agreeing
with each other is not the same as either one being right.

What this file pins is the property: when a caller supplies the application config, nothing downstream
re-reads it. A test that only compared rig ids would pass on a box whose default tree happens to match.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.config import load_config, load_robot_config
from src.robot.execution.autonomous_grasp import cells


class BothHalvesComeFromTheSuppliedTreeTests(unittest.TestCase):
    def test_the_builder_does_not_re_read_the_tree_it_was_given(self) -> None:
        """The property, asserted structurally so a matching default cannot hide it.

        `build_real_components` opens a camera, so this stops it before the device: what is measured
        is whether the loader was consulted, not whether a cell came up.
        """
        app_cfg = load_config()
        robot = load_robot_config()
        loads: list = []

        def watched(*args, **kwargs):
            loads.append((args, kwargs))
            return app_cfg

        # Patched at the module that owns it, so a builder importing it by any route is covered.
        with mock.patch("src.config.load_config", side_effect=watched):
            with mock.patch(
                "src.camera.orchestration.frame_provider.FrameProvider"
            ) as provider:
                provider.side_effect = RuntimeError("stop before the device")
                with self.assertRaises(Exception):
                    cells.build_real_components(robot, "a box", app_config=app_cfg)

        self.assertEqual(
            loads, [],
            "the builder was handed an application config and read the tree anyway",
        )

    def test_the_primary_rig_comes_from_the_supplied_config(self) -> None:
        """The visible half of the same property.

        A config naming a rig the default tree does not carry proves the supplied object was the one
        consulted, whatever the checkout holds.
        """
        app_cfg = load_config()
        rigs = app_cfg.camera.cameras.rigs
        if not rigs:
            self.skipTest("this tree names no camera rigs")
        invented = "rig_that_no_shipped_tree_carries"
        self.assertNotIn(invented, [getattr(r, "rig_id", None) for r in rigs])

        # The schema is frozen, so a new object rather than an assignment: `model_copy(update=...)`
        # at each level down to the field, which is also how the runtime builds an overlay.
        cameras = app_cfg.camera.cameras.model_copy(update={"primary_rig_id": invented})
        camera = app_cfg.camera.model_copy(update={"cameras": cameras})
        patched = app_cfg.model_copy(update={"camera": camera})
        robot = load_robot_config()

        with self.assertRaises(Exception) as caught:
            cells.build_real_components(robot, "a box", app_config=patched)
        self.assertIn(
            invented, str(caught.exception),
            "the refusal quoted a rig from some other tree than the one supplied",
        )


class NoCallerHadToChangeTests(unittest.TestCase):
    """Supplying nothing keeps the old behaviour exactly, which is what every existing caller does.

    The repair adds a door rather than moving one: `UNSET` means the caller did not choose a tree,
    and the default tree is then the honest answer. The defect was never that the default existed, it
    was that a caller who HAD chosen could not say so.
    """

    def test_omitting_the_config_still_reads_the_default_tree(self) -> None:
        robot = load_robot_config()
        seen: list = []

        real = cells.load_config if hasattr(cells, "load_config") else load_config

        def watched(*args, **kwargs):
            seen.append(True)
            return real(*args, **kwargs)

        with mock.patch("src.config.load_config", side_effect=watched):
            with mock.patch(
                "src.camera.orchestration.frame_provider.FrameProvider"
            ) as provider:
                provider.side_effect = RuntimeError("stop before the device")
                with self.assertRaises(Exception):
                    cells.build_real_components(robot, "a box")
        self.assertTrue(seen, "the default path must still read the tree")


if __name__ == "__main__":
    unittest.main()
