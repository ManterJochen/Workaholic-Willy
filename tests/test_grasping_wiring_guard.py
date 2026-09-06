"""The wiring guard: a config flag that is ON must change something at the runtime.

WHY THIS EXISTS, and why it is not another hand-written wiring test.
``tests/test_grasping_config_wiring.py`` already asserts, block by block, that the
overlays reach the orchestrator. Every one of those assertions had to be *remembered*
by whoever added the block. That is the failure mode this file is aimed at: a new
``grasping.*`` sub-block ships with a schema field and a YAML default, the operator
turns it on, and the cell quietly keeps running the old setting because nobody wired
the value through -- and no test noticed, because no test was written for a block that
did not exist when the tests were.

So this guard does not name any block. It **enumerates** the flags out of the schema
and drives each one differentially: build the service with the flag off, build it again
with the flag on, and diff an identity-free digest of everything observable on the
built service. If nothing moved, the flag is decorative and the test says so by name.
A block added tomorrow is covered the day its schema field lands.

WHAT IT PROVES: the operator's value reaches a runtime carrier.
WHAT IT DOES NOT PROVE: that the pick path then *acts* on that carrier. A carrier that
is set and never read would pass here. That second half is covered per-block by the
behavioural tests (``test_t1_decision_integration``, ``test_g4_uncertainty_rerank``,
``test_pick_loop``, ...) -- this guard is the net under them, not a replacement.

Measured when this landed (2026-08-14): all 23 flags change something. There is
deliberately **no allow-list of known-unwired flags** -- an empty exception list is
only honest because it was measured, and the moment one is needed the failure names
the flag rather than hiding it.

⚠ BLIND SPOT, named rather than discovered: this enumerates BOOLEAN flags. A **string
selector** is just as capable of being inert, and one already was -- ``grasping.calculator``
has existed since WS0 with nothing reading it, so a cell could ask for ``deep`` and
silently get the analytic stack. This guard could not have seen it, and did not. Selectors
therefore need their own differential test, written by hand, the way
``test_deep_calculator.FactoryTests`` covers that one: each value builds a different
class, and an unknown value refuses. The same goes for a sub-block with no ``enabled``
field at all (``grasping.deep_generator`` is one) -- there is no flag here to drive.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
from pydantic import BaseModel

from src.config.schema.robot import RobotConfig
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame

# The mode every flag is measured in. ``dense_clutter`` is the widest ``apply_modes``
# bucket in the schema, so a mode filter cannot mask a wiring gap. A flag that only
# applies in a narrower mode declares it via ``_PREREQUISITES``.
_BASE_MODE = "dense_clutter"

# How many flags the schema walker is expected to find. This is not a contract on the
# schema -- it is a guard on the WALKER: a refactor that renames the convention (say
# ``*_enabled`` -> ``*_active``) would otherwise silently reduce this test to nothing,
# and a test that asserts over an empty set passes loudest of all.
_MIN_EXPECTED_FLAGS = 20


class _PrereqSpec(SimpleNamespace):
    """What else must be true before a flag can possibly do anything."""

    config: dict[str, Any]


def _prerequisites(flag: str, tmpdir: Path) -> dict[str, Any]:
    """The config a flag needs before it can act -- real dependency structure, not a map.

    This deliberately encodes only *preconditions*, never the flag-to-slot mapping. A
    mapping table rots the moment someone moves a slot; a precondition is a property of
    the config schema itself, and an UNDECLARED one shows up here as a failure -- which
    is right, because a sub-flag whose parent is off silently doing nothing is exactly
    the "falls back to another setting" bug this file guards.
    """

    if flag.startswith("feasibility.") and flag != "feasibility.enabled":
        # Every feasibility signal is gated on ``feas_active`` in build_effective_config.
        return {"feasibility": {"enabled": True}}
    if flag == "occlusion.hard_reject_enabled":
        # A hard reject cannot fire when the corridor analyzer that produces the score
        # is not running.
        return {"occlusion": {"directional_enabled": True}}
    if flag.startswith("ordering.blocker_graph."):
        # The blocker-graph signals only exist inside the ordering overlay.
        return {"ordering": {"enabled": True}}
    if flag == "support.container.wall_collision_enabled":
        # The schema itself refuses this without an interior box -- there is no default bin.
        return {"support": {"container": {"interior_min_mm": [0.0, 0.0, 0.0],
                                          "interior_max_mm": [300.0, 200.0, 150.0]}}}
    if flag.startswith("fusion.geometry.") and flag != "fusion.geometry.enabled":
        # The multi-camera block's sub-switches only reach the loop through the carrier that
        # ``fusion.geometry.enabled`` installs; with the parent off there is no carrier to put
        # them on.
        return {"fusion": {"geometry": {"enabled": True}}}
    if flag == "deep_ranker.enabled":
        # The ranker's TREES are gitignored (`.gitignore:52`), so no checkout carries them and
        # `DeepRankerContext.from_config` fail-safes to None on a missing artifact -- the flag is
        # wired (builders.py:1078) and would still look inert here. Supply a real artifact, the
        # same way `fusion.enabled` supplies real extrinsics, so the switch has something to load.
        return {"deep_ranker": {"artifact_dir": _write_ranker_artifact(tmpdir)}}
    if flag == "fusion.enabled":
        # The fusion substrate enforces a strict frame contract, so the overlay refuses
        # to wire without a CAMERA->BASE resolver. Supply a real persisted artifact.
        return {"fusion": {"extrinsics_artifact_path": _write_identity_extrinsics(tmpdir)}}
    return {}


def _write_ranker_artifact(directory: Path) -> str:
    """A real, loadable one-tree ranker under the spec the schema defaults to.

    Fitted on nothing and it does not need to be: this guard asks whether the operator's value
    reaches a runtime carrier, and the carrier is built only when the artifact loads.
    """
    import json

    from src.config.schema.robot.grasping_schema import GraspingDeepRankerConfig
    from src.robot.grasping.deep.ranker.features import spec_named
    from src.robot.grasping.deep.ranker.runtime import ARTIFACT_KIND, ARTIFACT_VERSION

    defaults = GraspingDeepRankerConfig()
    spec = spec_named(str(defaults.spec))
    artifact = directory / f"{spec.name}.json"
    if not artifact.exists():
        artifact.write_text(
            json.dumps({
                "kind": ARTIFACT_KIND,
                "artifact_version": ARTIFACT_VERSION,
                "spec": spec.name,
                "features": list(spec.features),
                "init_score": 0.0,
                "learning_rate": 0.1,
                "base_rate": 0.5,
                # One tree, one node, and that node is a leaf: `feature == -1`.
                "trees": [{"feature": [-1], "threshold": [0.0],
                           "left": [-1], "right": [-1], "value": [0.0]}],
            }),
            encoding="utf-8",
        )
    return str(directory)


def _write_identity_extrinsics(directory: Path) -> str:
    from src.calibration.extrinsics import Extrinsics
    from src.calibration.serialization import save_extrinsics
    from src.geometry import Frame, Transform

    artifact = directory / "eth_extrinsics.json"
    if not artifact.exists():
        save_extrinsics(
            artifact,
            Extrinsics(
                transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE),
                rmse_mm=1.0,
                max_error_mm=2.0,
                num_samples=10,
                captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                rig_id="wiring_guard",
            ),
        )
    return str(artifact)


# ---------------------------------------------------------------------------
# Schema enumeration
# ---------------------------------------------------------------------------


def _iter_flags(model: type[BaseModel] = RobotGraspingConfig, prefix: str = "") -> Iterator[str]:
    """Every boolean opt-in under ``robot.grasping``, as a dotted path."""

    for name, field in model.model_fields.items():
        annotation = field.annotation
        path = f"{prefix}{name}"
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from _iter_flags(annotation, path + ".")
        # NOTE (2026-08-22): widening this to EVERY bool -- not just `*enabled*` -- was measured
        # and reds on 9 flags. Several are artefacts of the differential rather than real
        # inertness (a flag whose enclosing block is off cannot move anything, and a flag whose
        # default is already True has no off->on edge to drive), so the widening needs
        # `_prerequisites` entries and a default-aware flip before it means what it says. Left
        # narrow deliberately, with the gap written down: `closed_loop.pregrasp_rescan` and
        # `verification.post_lift_vision_check` sat dead for months inside exactly this blind spot.
        elif annotation is bool and "enabled" in name:
            yield path


# ---------------------------------------------------------------------------
# Identity-free observation
# ---------------------------------------------------------------------------


def _shape(value: object, depth: int = 0) -> str:
    """A digest that captures VALUE and TYPE but never identity.

    Two builds of the same config produce different object addresses, so a naive
    ``repr`` would report a difference for every flag and the guard would pass
    vacuously. Falling back to the bare type name for anything unrecognised still
    catches the transition that matters here: ``None`` -> a wired carrier.
    """

    if depth > 5:
        return "<deep>"
    if value is None:
        return "None"
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, (int, float, str)):
        return repr(value)
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}"
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_shape(v, depth + 1) for v in value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_shape(v, depth + 1) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{k}:{_shape(v, depth + 1)}" for k, v in items) + "}"
    if is_dataclass(value) and not isinstance(value, type):
        body = ",".join(
            f"{f.name}={_shape(getattr(value, f.name, None), depth + 1)}" for f in fields(value)
        )
        return f"{type(value).__name__}({body})"
    if isinstance(value, BaseModel):
        body = ",".join(
            f"{k}={_shape(getattr(value, k, None), depth + 1)}"
            for k in sorted(type(value).model_fields)
        )
        return f"{type(value).__name__}({body})"
    return f"<{type(value).__name__}>"


def _observe(service: AutonomousGraspService) -> dict[str, str]:
    """Everything an operator's flag could plausibly have moved."""

    observed: dict[str, str] = {}
    effective = service.effective_config
    if effective is not None:
        for key, value in effective.to_dict().items():
            observed[f"effective_config.{key}"] = _shape(value)
    orchestrator = service.runtime.orchestrator
    for name in dir(orchestrator):
        if name.startswith("__"):
            continue
        try:
            value = getattr(orchestrator, name)
        except Exception:  # pragma: no cover - defensive; a property may refuse
            continue
        if callable(value) and not is_dataclass(value):
            continue
        observed[f"orchestrator.{name}"] = _shape(value)
    for name in ("refinement_policy", "verification_policy", "recovery_policy"):
        observed[f"service.{name}"] = _shape(getattr(service, name, None))
    return observed


# ---------------------------------------------------------------------------
# Test doubles (mirrors tests/test_grasping_config_wiring.py)
# ---------------------------------------------------------------------------


def _perception_frame() -> PerceptionFrame:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return PerceptionFrame(
        depth_map=np.full((32, 32), 500.0, dtype=np.float64),
        intrinsics=np.array(
            [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float64
        ),
        segmentations=(SimpleNamespace(mask=mask),),
    )


class _ScriptedCalculator:
    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        return GraspResult(
            candidates=(
                GraspPoint(
                    position=np.array([100.0, 50.0, 400.0]),
                    approach=np.array([0.0, 0.0, 1.0]),
                    axis=np.array([1.0, 0.0, 0.0]),
                    grip_width_mm=40.0,
                    score=0.9,
                    frame=GraspFrame.BASE,
                    label="wiring_guard",
                ),
            ),
            reasons=(),
            top_score=0.9,
        )


class _FakePerception:
    def acquire(self) -> PerceptionFrame:
        return _perception_frame()


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = _merge(existing, value)
        else:
            merged[key] = value
    return merged


def _nest(dotted: str, value: object) -> dict[str, Any]:
    parts = dotted.split(".")
    root: dict[str, Any] = {}
    cursor = root
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return root


def _build(grasping: dict[str, Any]) -> AutonomousGraspService:
    config = RobotConfig(
        vendor="dummy",
        gripper={"vendor": "none"},
        grasping=_merge({"default_mode": _BASE_MODE, "max_attempts": 5}, grasping),
    )
    return AutonomousGraspService.from_robot_config(
        config,
        calculator=_ScriptedCalculator(),  # type: ignore[arg-type]
        perception=_FakePerception(),
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


class GraspingFlagsReachTheRuntimeTests(unittest.TestCase):
    """Turning a flag on must be observable on the built service."""

    def test_the_walker_still_finds_the_flags(self) -> None:
        """A test that asserts over an empty set is the most dangerous kind of green."""

        flags = list(_iter_flags())
        self.assertGreaterEqual(
            len(flags),
            _MIN_EXPECTED_FLAGS,
            "the schema walker found fewer opt-in flags than expected -- either the "
            "naming convention changed (and this guard is now blind) or blocks were "
            f"removed. Found: {sorted(flags)}",
        )
        self.assertEqual(len(flags), len(set(flags)), "duplicate flag paths")

    def test_every_flag_either_moves_the_runtime_or_refuses_to_load(self) -> None:
        """The whole point: a config block the cell would silently ignore fails here.

        STRENGTHENED 2026-08-17. The original guard asserted that every flag moves the
        built service, and it passed on all 23 -- honestly, because it measured what it
        claimed. The on-box campaign then found three blocks (`feasibility`, `occlusion`,
        `ordering`) that move the service and are still ignored by the pick path: their
        value lands in ``EffectiveGraspingConfig``, and ``EffectiveGraspingConfig`` **is**
        a runtime carrier, so "moved something" was true and useless. That is the limit
        this file's own docstring names.

        So the assertion is now a disjunction, and it is strictly stronger: a flag must
        EITHER move the runtime OR be declared unwired in the schema and REFUSE TO LOAD.
        The one thing it may not be is quietly acceptable and inert. The docstring
        promised that the day an exception list was needed the failure would name the
        flag rather than hide it -- this is that day, and the list lives in the schema
        (``RobotGraspingConfig.UNWIRED_SWITCHES``) where an operator meets it, not here.
        """

        unwired = RobotGraspingConfig.UNWIRED_SWITCHES
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for flag in _iter_flags():
                with self.subTest(flag=flag):
                    prereq = _prerequisites(flag, tmpdir)
                    if flag in unwired:
                        # Declared unwired -> the schema must refuse it, and the refusal
                        # must SAY SO. A silent acceptance here is the failure mode.
                        with self.assertRaises(Exception) as ctx:
                            _build(_merge(prereq, _nest(flag, True)))
                        # grasping_schema.py:2075 now says "... never reaches the pick path.";
                        # the migration lower-cased the shout, the refusal is unchanged.
                        self.assertIn(
                            "never reaches the pick path",
                            str(ctx.exception),
                            f"`grasping.{flag}` is declared unwired but the refusal does "
                            "not explain itself to the operator who hit it.",
                        )
                        continue
                    before = _observe(_build(prereq))
                    after = _observe(_build(_merge(prereq, _nest(flag, True))))
                    changed = sorted(
                        key
                        for key in set(before) | set(after)
                        if before.get(key) != after.get(key)
                    )
                    self.assertTrue(
                        changed,
                        f"`grasping.{flag}` is ON and NOTHING changed on the built "
                        "service. The cell would run as if the operator had never set "
                        "it. Wire it through (builders.apply_orchestrator_overlays or "
                        "builders.build_effective_config), or -- if it genuinely needs "
                        "another flag first -- declare that in `_prerequisites`.",
                    )

    def test_the_unwired_list_names_only_flags_that_exist(self) -> None:
        """A stale entry would silently exempt a flag that was wired since -- or a typo."""

        flags = set(_iter_flags())
        stale = sorted(set(RobotGraspingConfig.UNWIRED_SWITCHES) - flags)
        self.assertEqual(
            stale, [],
            "RobotGraspingConfig.UNWIRED_SWITCHES names switches the schema walker does "
            "not find. Either they were renamed/removed, or the path is wrong -- both "
            "mean the exemption is silently covering nothing.",
        )


if __name__ == "__main__":
    unittest.main()
