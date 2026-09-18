"""The sidecar loads a wrist camera on top of the config the combination evidence names, and hashes the evidence config.

As the owner decided, the evidence stays keyed by arm, hand, plate, placement and margin, and its hashes are formed
without the camera. So the camera travels in its own variable, ``compose_for_cell`` composes the evidence config
exactly as a cell without a camera does and the loaded config on top of it, and refuses unless removing the camera's
links gives the evidence config back. The planner start then proves the sidecar holds exactly this cell's camera, with
a complete cover (``body_link.wrist_body_refusal``). The camera stays checked against a carried part.

The sidecar runs in the cuRobo environment, so its half is read as source, as test_sidecar_composes_one_dict.py does.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest import mock

from src.contracts import UNSET
from src.robot.safety.planning._curobo_attach import ATTACHED_LINK_NAME
from src.robot.safety.planning._curobo_body_links import (
    ENV_BODY_LINKS,
    ENV_WRIST_BODY_LINKS,
    BodyLinkError,
    apply_body_links,
    body_report,
    canonical_sha256,
    compose_for_cell,
    compose_with_counts,
    without_links,
)
from src.robot.safety.planning.body_link import wrist_body_refusal
from src.robot.safety.planning.curobo_client import CuroboPlanClient, SidecarIdentity
from tests._wrist_body import wrist_body
from tests.test_sidecar_composes_one_dict import _descriptor

_SERVER = Path(__file__).resolve().parents[1] / "src/robot/safety/planning/curobo_planner_server.py"


def _arm() -> dict:
    """The per arm descriptor: the hand-carrying fixture with its hand stripped, as build_ur_config.py writes an arm."""
    config = copy.deepcopy(_descriptor())
    config.pop("_provenance")
    del config["robot_cfg"]["kinematics"]["collision_spheres"]["tool0"]
    config["robot_cfg"]["kinematics"]["collision_link_names"].remove("tool0")
    return config


def _hand() -> dict:
    return {"link": "hand", "parent": "tool0", "joint": "hand_joint", "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "spheres": [{"center": [0.0, 0.1, 0.0], "radius": 0.03}], "slots": 0, "buffer_m": 0.0,
            "ignore": ["tool0", "wrist_3_link", "wrist_2_link"]}


def _loaded_rows(wrist: dict, *, margin_mm: float = 10.0, attach: int = 16) -> tuple[list, list]:
    """What a sidecar composing this cell reports: its body rows and its wrist rows."""
    loaded, _, _, _, links = compose_for_cell(_arm(), bodies=[_hand()], wrist_bodies=[wrist], margin_mm=margin_mm,
                                              attach_spheres=attach)
    rows = [{**row, **{key: wrist.get(key) for key in ("rig_id", "model", "margin_mm", "boxes")}}
            for row in body_report(loaded, links)]
    return body_report(loaded, ["hand"]), rows


class TheEvidenceConfigIsTheCameraFreeConfigTests(unittest.TestCase):
    def test_without_a_camera_the_evidence_config_is_what_a_cell_composed_before(self) -> None:
        before, _, _ = compose_with_counts(_arm(), bodies=[_hand()], margin_mm=10.0, attach_spheres=16)
        loaded, evidence, _, _, links = compose_for_cell(_arm(), bodies=[_hand()], margin_mm=10.0, attach_spheres=16)
        self.assertEqual(links, [])
        self.assertEqual(evidence, before)
        self.assertIs(loaded, evidence)

    def test_with_a_camera_the_loaded_config_without_it_is_the_evidence_config(self) -> None:
        wrist = wrist_body().link()
        for margin, attach in ((0.0, 0), (10.0, 16)):
            with self.subTest(margin=margin, attach=attach):
                loaded, evidence, _, _, links = compose_for_cell(
                    _arm(), bodies=[_hand()], wrist_bodies=[wrist], margin_mm=margin, attach_spheres=attach)
                self.assertEqual(links, ["wrist_camera_wrist"])
                self.assertEqual(without_links(loaded, links), evidence)
                self.assertEqual(canonical_sha256(evidence),
                                 canonical_sha256(compose_with_counts(_arm(), bodies=[_hand()], margin_mm=margin,
                                                                      attach_spheres=attach)[0]))
                self.assertNotEqual(canonical_sha256(loaded), canonical_sha256(evidence))

    def test_the_camera_takes_the_guards_half_margin_and_stays_checked_against_the_part_and_the_wrist(self) -> None:
        loaded, _, _, _, _ = compose_for_cell(_arm(), bodies=[_hand()], wrist_bodies=[wrist_body().link()],
                                              margin_mm=10.0, attach_spheres=16)
        kinematics = loaded["robot_cfg"]["kinematics"]
        self.assertAlmostEqual(kinematics["self_collision_buffer"]["wrist_camera_wrist"], 0.005)
        ignored = kinematics["self_collision_ignore"]["wrist_camera_wrist"]
        for checked in (ATTACHED_LINK_NAME, "wrist_1_link", "wrist_2_link"):
            self.assertNotIn(checked, ignored)
        self.assertIn(ATTACHED_LINK_NAME, kinematics["self_collision_ignore"]["hand"],
                      "the hand holding the part still ignores it: only the camera is exempted")

    def test_a_step_that_writes_outside_the_camera_link_is_refused(self) -> None:
        """The control: a composition that touches another link's entries is caught, not cleaned away."""
        original = apply_body_links

        def leaky(config, bodies):  # noqa: ANN001, ANN202
            out, added = original(config, bodies)
            if any(str(body.get("link")).startswith("wrist_camera_") for body in bodies):
                out["robot_cfg"]["kinematics"]["self_collision_ignore"]["hand"].append("forearm_link")
            return out, added

        with mock.patch("src.robot.safety.planning._curobo_body_links.apply_body_links", leaky):
            with self.assertRaises(BodyLinkError) as caught:
                compose_for_cell(_arm(), bodies=[_hand()], wrist_bodies=[wrist_body().link()], margin_mm=10.0)
        self.assertIn("changed the config outside its own links, at", str(caught.exception))
        self.assertIn("self_collision_ignore.hand", str(caught.exception))

    def test_a_camera_sent_as_an_evidence_body_and_a_stranger_sent_as_a_camera_are_refused(self) -> None:
        with self.assertRaises(BodyLinkError) as caught:
            compose_for_cell(_arm(), bodies=[_hand(), wrist_body().link()])
        self.assertIn("so the evidence stays keyed without it", str(caught.exception))
        with self.assertRaises(BodyLinkError):
            compose_for_cell(_arm(), bodies=[_hand()], wrist_bodies=[{**wrist_body().link(), "link": "camera"}])

    def test_a_camera_blind_to_the_wrist_or_hung_elsewhere_is_refused(self) -> None:
        for blind in ("wrist_1_link", "wrist_2_link"):
            link = {**wrist_body().link(), "ignore": [*wrist_body().link()["ignore"], blind]}
            with self.subTest(blind=blind), self.assertRaises(BodyLinkError) as caught:
                apply_body_links(_arm(), [_hand(), link])
            self.assertIn("the arm links a camera on the flange can hit", str(caught.exception))
        hung = {**wrist_body().link(), "parent": "hand", "ignore": ["hand", "tool0", "wrist_3_link"]}
        with self.assertRaises(BodyLinkError) as caught:
            apply_body_links(_arm(), [_hand(), hung])
        self.assertIn("is placed on the flange by its calibration, so it hangs from the tool frame", str(caught.exception))


class TheServerHashesTheEvidenceConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")

    def test_the_server_composes_for_the_cell_and_hashes_the_evidence_config(self) -> None:
        self.assertIn("compose_for_cell(", self.source)
        self.assertIn("_composed_sha256 = canonical_sha256(_EVIDENCE)", self.source)
        self.assertNotIn("canonical_sha256(_COMPOSED)", self.source)
        self.assertIn("os.environ.get(ENV_WRIST_BODY_LINKS)", self.source)

    def test_a_camera_free_ready_line_carries_no_wrist_key(self) -> None:
        self.assertIn('_WRIST: dict = ({"wrist_bodies": _wrist_rows, "wrist_bodies_sha256": canonical_sha256(_wrist_rows)}\n'
                      "                    if _wrist_rows else {})", self.source)
        self.assertIn('"composed_sha256": _composed_sha256, "bodies": _body_rows, **_WRIST, **_MEASURING}', self.source)


class TheClientSendsAndReadsTheCameraTests(unittest.TestCase):
    def test_the_camera_travels_in_its_own_variable(self) -> None:
        link = wrist_body().link()
        with mock.patch.dict("os.environ", {ENV_WRIST_BODY_LINKS: "[\"inherited\"]"}):
            env = CuroboPlanClient(body_links=[_hand()], wrist_body_links=[link])._sidecar_env()
            self.assertIn("wrist_camera_wrist", env[ENV_WRIST_BODY_LINKS])
            self.assertNotIn("wrist_camera", env[ENV_BODY_LINKS])
            plain = CuroboPlanClient(body_links=[_hand()])._sidecar_env()
            self.assertNotIn(ENV_WRIST_BODY_LINKS, plain, "an inherited camera would load a housing nobody sent")

    def test_an_identity_reads_the_wrist_rows_and_says_nothing_when_they_are_absent(self) -> None:
        _, rows = _loaded_rows(wrist_body().link())
        identity = SidecarIdentity.from_ready({"wrist_bodies": rows, "wrist_bodies_sha256": canonical_sha256(rows)})
        self.assertEqual(identity.wrist_bodies, tuple(rows))
        self.assertIn("wrist cameras: wrist_camera_wrist (realsense_d435)", identity.render())
        silent = SidecarIdentity.from_ready({})
        self.assertIs(silent.wrist_bodies, UNSET)
        self.assertIs(silent.wrist_bodies_sha256, UNSET)
        self.assertNotIn("wrist cameras", silent.render())
        self.assertIsNone(silent.to_dict()["wrist_bodies"])


class ThePlannerStartProvesTheCameraTests(unittest.TestCase):
    """``wrist_body_refusal``, each refusal in its order, and the passing cells."""

    def setUp(self) -> None:
        self.body = wrist_body()
        _, self.rows = _loaded_rows(self.body.link())

    def _identity(self, rows: object = "same", sha: object = "matching") -> SidecarIdentity:
        held = self.rows if rows == "same" else rows
        if held is None:
            return SidecarIdentity.from_ready({})
        digest = canonical_sha256(held) if sha == "matching" else sha
        return SidecarIdentity.from_ready({"wrist_bodies": held, "wrist_bodies_sha256": digest})

    def test_the_sidecar_that_loaded_this_camera_passes(self) -> None:
        self.assertIsNone(wrist_body_refusal(self._identity(), [self.body]))

    def test_a_camera_free_cell_passes_a_camera_free_sidecar_and_refuses_one_with_a_camera(self) -> None:
        self.assertIsNone(wrist_body_refusal(SidecarIdentity.from_ready({}), []))
        refusal = wrist_body_refusal(self._identity(), [])
        assert refusal is not None
        self.assertIn("this cell carries none", refusal)

    def test_a_sidecar_that_reports_no_wrist_bodies_is_refused(self) -> None:
        refusal = wrist_body_refusal(self._identity(rows=None), [self.body])
        assert refusal is not None
        self.assertIn("the sidecar reported no wrist bodies", refusal)

    def test_rows_that_do_not_hash_to_the_reported_hash_are_refused(self) -> None:
        refusal = wrist_body_refusal(self._identity(sha="0" * 64), [self.body])
        assert refusal is not None
        self.assertIn("do not hash to the wrist_bodies_sha256 it reported", refusal)

    def test_another_camera_and_each_field_that_differs_is_refused(self) -> None:
        cases = {
            "link": ({"link": "wrist_camera_other"}, "and this cell sent ['wrist_camera_wrist']"),
            "parent": ({"parent": "hand"}, "with parent 'hand'"),
            "fixed_transform": ({"fixed_transform": [0.0] * 7}, "with fixed_transform"),
            "spheres_sha256": ({"spheres_sha256": "e" * 64}, "with spheres_sha256 eeeeeeeeeeee"),
            "ignore": ({"ignore": ["tool0", "wrist_3_link", "hand", "coupling", ATTACHED_LINK_NAME]},
                       "the planner skips a pair this camera must be checked against"),
        }
        for field, (change, said) in cases.items():
            with self.subTest(field=field):
                rows = [{**self.rows[0], **change}]
                refusal = wrist_body_refusal(self._identity(rows=rows), [self.body])
                assert refusal is not None
                self.assertIn(said, refusal)

    def test_a_body_whose_own_fill_leaves_a_hole_is_refused_as_a_defect(self) -> None:
        with mock.patch.object(type(self.body), "cover_refusal", return_value="'x' has cell (0, 0, 0) with a hole"):
            refusal = wrist_body_refusal(self._identity(), [self.body])
        assert refusal is not None
        self.assertIn("this is a defect, not a cell setting", refusal)


if __name__ == "__main__":
    unittest.main()
