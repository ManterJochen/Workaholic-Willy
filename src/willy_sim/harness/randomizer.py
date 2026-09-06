"""Domain randomization: one seeded, config-driven randomizer for the dense-pick collection.

A single :class:`DomainRandomizer` over five axes (position, orientation, colour, dome light, key
light) that stamps the applied parameters into ``extra["randomization"]``, so a recorded dataset is
auditable from the records rather than only by re-seeding.

The engine is split by axis:

  * Pose and orientation are sampled in numpy and applied via ``SingleRigidPrim.set_world_pose``.
    The caller zeroes the velocity, so the result is physics-clean, deterministic and reproducible
    without Isaac.
  * Colour is numpy-sampled from a named palette and applied via ``OmniPBR``. The name matches the
    applied colour, which is what keeps the ``object_set`` provenance honest and the leakage audit
    non-vacuous. Replicator's applied colour cannot be read back reliably enough to name it.
  * Lighting (dome and key intensity) goes through Isaac Replicator (``omni.replicator.core``),
    triggered via ``orchestrator.preview()``. ``orchestrator.step()`` takes over the timeline and
    breaks pick execution, so it must not be used here. Replicator is also the seam for richer
    texture and material randomization.

All Isaac imports (``omni.replicator``, ``OmniPBR``) are lazy, so this module imports where
``isaacsim`` is absent.

With ``RandomizationConfig.enabled=False``, the default, :meth:`DomainRandomizer.sample_pose`
returns the home poses unchanged and :meth:`DomainRandomizer.apply_colors` and
:meth:`DomainRandomizer.apply_lighting` do nothing, so the reset loop is a plain home reset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.utility.log_cfg import create_logger
from src.willy_sim.constants import RANDOMIZER_LOG_FILE, WILLY_SIM_LOG_DIR

#: Read against a dataset rather than a run: these lines say what the recorded episodes were varied
#: over. The provenance dict below is the ground truth, but it reaches only the records of episodes
#: that got as far as being logged, so a run that failed earlier would otherwise say nothing about
#: which axes were live.
_LOG = create_logger("DomainRandomizer", log_file=RANDOMIZER_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)

__all__ = [
    "RandomizationConfig",
    "EpisodePose",
    "DomainRandomizer",
    "DISTRACTOR_PALETTE",
    "nearest_palette_name",
]

# Named colour palette, RGB in [0, 1]. The object_set provenance maps a sampled or read-back colour
# to the nearest name, which keeps the ``build-dataset`` leakage audit non-vacuous.
DISTRACTOR_PALETTE: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("red", (0.90, 0.12, 0.12)), ("green", (0.10, 0.65, 0.18)), ("blue", (0.12, 0.22, 0.90)),
    ("yellow", (0.92, 0.86, 0.13)), ("purple", (0.55, 0.16, 0.78)), ("orange", (0.95, 0.52, 0.10)),
    ("cyan", (0.10, 0.78, 0.82)), ("magenta", (0.90, 0.15, 0.65)), ("lime", (0.55, 0.85, 0.18)),
    ("teal", (0.10, 0.55, 0.52)), ("pink", (0.96, 0.62, 0.74)), ("brown", (0.52, 0.36, 0.20)),
)


def nearest_palette_name(rgb: tuple[float, float, float]) -> str:
    """Map an RGB triple to the nearest :data:`DISTRACTOR_PALETTE` colour name (for object_set provenance)."""
    arr = np.asarray(rgb, dtype=np.float64)
    return min(DISTRACTOR_PALETTE, key=lambda kv: float(np.sum((np.asarray(kv[1]) - arr) ** 2)))[0]


@dataclass(frozen=True)
class RandomizationConfig:
    """The seeded domain-randomization knobs, built from a runner's CLI rather than from YAML.

    With ``enabled=False``, the default, the randomizer does nothing and the reset is a plain home
    reset. The ranges are centred on the shipped scene defaults (dome 150, key 300), so an enabled
    run stays in a realistic band.
    """

    enabled: bool = False
    seed: int = 0
    # --- pose axis (numpy, applied via set_world_pose) ---
    position_jitter_mm: float = 0.0          # per-object XY, uniform +/- this
    orientation_jitter_deg: float = 0.0      # per-object yaw about world Z, uniform +/- this
    # --- visual axis (colour via OmniPBR, lighting via Replicator) ---
    randomize_color: bool = False            # re-colour the distractors (keep the target's identity)
    dome_intensity_range: tuple[float, float] | None = None
    key_intensity_range: tuple[float, float] | None = None
    # which objects to re-colour: keep the prompted target stable so sim_target/prompt stay consistent
    recolor_target: bool = False

    @property
    def any_pose(self) -> bool:
        return self.enabled and (self.position_jitter_mm > 0.0 or self.orientation_jitter_deg > 0.0)

    @property
    def any_color(self) -> bool:
        # Colour goes through numpy and OmniPBR: the name is known before it is applied, so the
        # object_set provenance is honest. Replicator's applied colour cannot be read back reliably,
        # so it cannot name the object_set.
        return self.enabled and self.randomize_color

    @property
    def any_lighting(self) -> bool:
        return self.enabled and (
            self.dome_intensity_range is not None or self.key_intensity_range is not None
        )

    @property
    def any_visual(self) -> bool:
        return self.any_color or self.any_lighting

    # --- CLI factories (the single place the runner builds a config) ---
    @classmethod
    def disabled(cls) -> "RandomizationConfig":
        return cls(enabled=False)

    @classmethod
    def full(cls, *, seed: int, jitter_mm: float = 20.0) -> "RandomizationConfig":
        """The ``--randomize`` config: all five axes (position, orientation, colour, dome, key)."""
        return cls(
            enabled=True, seed=int(seed),
            position_jitter_mm=float(jitter_mm), orientation_jitter_deg=15.0,
            randomize_color=True, dome_intensity_range=(100.0, 200.0),
            key_intensity_range=(200.0, 450.0),
        )

    @classmethod
    def legacy(cls, *, seed: int, jitter_mm: float, recolor: bool) -> "RandomizationConfig":
        """The ``--seed`` and ``--collect`` axes: position jitter plus distractor colour.

        Orientation and lighting stay off, so a collection run varies only what those flags name.
        """
        return cls(
            enabled=True, seed=int(seed),
            position_jitter_mm=float(jitter_mm), orientation_jitter_deg=0.0,
            randomize_color=bool(recolor),
        )


@dataclass(frozen=True)
class EpisodePose:
    """The per-episode pose the randomizer chose (applied by the caller via set_world_pose)."""

    positions_mm: tuple[tuple[float, float, float], ...]
    orientations_wxyz: tuple[tuple[float, float, float, float], ...]
    yaw_deltas_deg: tuple[float, ...] = field(default_factory=tuple)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two WXYZ quaternions, in the order ``a`` then ``b``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _yaw_quat_wxyz(deg: float) -> np.ndarray:
    """WXYZ quaternion of a rotation by ``deg`` about world Z."""
    half = np.deg2rad(deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


class DomainRandomizer:
    """Seeded domain randomizer (pose+colour=numpy, lighting=Replicator). Constructed once per run."""

    def __init__(self, config: RandomizationConfig) -> None:
        self.config = config
        self._visual_ready = False
        self._light_paths: dict[str, str] = {}
        # Once per run, not per episode: which axes are live and off which seed. A disabled
        # randomizer is logged too, so a thin dataset can be explained from the log alone.
        _LOG.info(
            "randomizer built: enabled=%s seed=%d axes(pose=%s color=%s lighting=%s) "
            "jitter=%.1f mm yaw=%.1f deg",
            config.enabled, config.seed, config.any_pose, config.any_color, config.any_lighting,
            config.position_jitter_mm, config.orientation_jitter_deg,
        )

    # ------------------------------------------------------------------ pose (numpy, no Isaac needed)
    def sample_pose(
        self,
        episode_idx: int,
        home_positions_mm: list[tuple[float, float, float]],
        home_orientations_wxyz: list[tuple[float, float, float, float]],
    ) -> EpisodePose:
        """Per-episode object poses, computed in numpy alone.

        Seeded by ``config.seed + episode_idx``, so a run reproduces without Isaac. With
        ``enabled=False`` the home poses come back unchanged.
        """
        n = len(home_positions_mm)
        if not self.config.any_pose:
            return EpisodePose(
                positions_mm=tuple((float(p[0]), float(p[1]), float(p[2])) for p in home_positions_mm),
                orientations_wxyz=tuple(
                    (float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in home_orientations_wxyz
                ),
                yaw_deltas_deg=tuple(0.0 for _ in range(n)),
            )
        rng = np.random.default_rng(int(self.config.seed) + int(episode_idx))
        jit = float(self.config.position_jitter_mm)
        ydeg = float(self.config.orientation_jitter_deg)
        positions: list[tuple[float, float, float]] = []
        orients: list[tuple[float, float, float, float]] = []
        yaws: list[float] = []
        for pos_mm, quat in zip(home_positions_mm, home_orientations_wxyz):
            p = np.asarray(pos_mm, dtype=np.float64).copy()
            if jit > 0.0:
                p[:2] += rng.uniform(-jit, jit, size=2)
            yaw = float(rng.uniform(-ydeg, ydeg)) if ydeg > 0.0 else 0.0
            q = np.asarray(quat, dtype=np.float64)
            if yaw != 0.0:
                q = _quat_mul_wxyz(_yaw_quat_wxyz(yaw), q)
                q = q / float(np.linalg.norm(q))
            positions.append((float(p[0]), float(p[1]), float(p[2])))
            orients.append((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
            yaws.append(round(yaw, 3))
        # Debug rather than info: one line per episode, and a collection run has thousands. The same
        # numbers ride into ``extra["randomization"]`` on every logged record, so this copy matters
        # only for a run whose records did not survive.
        _LOG.debug(
            "episode %d: seeded %d, jittered %d object pose(s) (yaw deltas %s)",
            episode_idx, int(self.config.seed) + int(episode_idx), n, yaws,
        )
        return EpisodePose(
            positions_mm=tuple(positions), orientations_wxyz=tuple(orients), yaw_deltas_deg=tuple(yaws)
        )

    # ------------------------------------------------------------------ colour (numpy + OmniPBR, on-box)
    def sample_colors(
        self, episode_idx: int, n_objects: int, target_index: int
    ) -> list[tuple[str, tuple[float, float, float]] | None]:
        """Per-object ``(name, RGB)`` from :data:`DISTRACTOR_PALETTE`.

        Returns ``None`` for the kept target and for every object when colour is off. Seeded by
        ``seed + episode_idx``, in numpy alone, so it reproduces without Isaac.
        """
        if not self.config.any_color:
            return [None] * n_objects
        rng = np.random.default_rng(int(self.config.seed) + int(episode_idx) + 100_000)
        palette = list(DISTRACTOR_PALETTE)
        rng.shuffle(palette)
        out: list[tuple[str, tuple[float, float, float]] | None] = []
        pi = 0
        for j in range(n_objects):
            if j == target_index and not self.config.recolor_target:
                out.append(None)  # keep the prompted target's identity
                continue
            out.append(palette[pi % len(palette)])
            pi += 1
        return out

    def apply_colors(self, objs: list, episode_idx: int, target_index: int) -> list[str | None]:
        """Apply the sampled palette colours to the distractors via ``OmniPBR``.

        Returns the per-object colour name, or ``None`` where nothing was applied. The name matches
        the applied colour, which is what makes the object_set provenance honest. The ``OmniPBR``
        import is lazy, so it happens only when colour is on.
        """
        sampled = self.sample_colors(episode_idx, len(objs), target_index)
        if not self.config.any_color:
            return [None] * len(objs)
        from isaacsim.core.api.materials import OmniPBR  # type: ignore[import-not-found]

        names: list[str | None] = []
        # Counted rather than logged per object: a dense scene has a dozen or more prims, so one line
        # each would drown the log. One aggregate line follows the loop instead.
        failures = 0
        first_error = ""
        for j, (o, choice) in enumerate(zip(objs, sampled)):
            if choice is None:
                names.append(None)
                continue
            cname, crgb = choice
            try:
                o.apply_visual_material(
                    OmniPBR(
                        prim_path=f"/World/Looks/dr_mat_{episode_idx}_{j}",
                        name=f"dr_mat_{episode_idx}_{j}", color=np.array(crgb, dtype=np.float64),
                    )
                )
                names.append(cname)
            except Exception as exc:  # noqa: BLE001 (colour is best-effort; object_set degrades honestly to None)
                failures += 1
                if not first_error:
                    first_error = repr(exc)
                names.append(None)
        if failures:
            # A degraded colour axis is invisible downstream: the object_set provenance records
            # ``None``, which cannot be told apart from colour being off. Without this line the
            # leakage audit loses its signal for those episodes without saying so.
            _LOG.warning(
                "episode %d: %d of %d colour application(s) failed -> object_set provenance is None "
                "for them (first error: %s)",
                episode_idx, failures, sum(1 for c in sampled if c is not None), first_error,
            )
        return names

    # ------------------------------------------------------------------ lighting (Replicator, on-box)
    def setup_lighting(self, *, dome_path: str, key_path: str) -> None:
        """Register the Replicator light-intensity randomizers once, under an ``on_frame`` trigger.

        Each :meth:`apply_lighting` then fires them once through ``orchestrator.preview()``. The
        trigger must be ``on_frame``: ``on_custom_event`` with ``send_og_event`` leaves the sample
        nodes uninitialised and crashes. Colour is handled separately, in numpy and ``OmniPBR``. The
        ``omni.replicator.core`` import is lazy, so this module still imports without Isaac.
        """
        if not self.config.any_lighting:
            return
        import omni.replicator.core as rep  # type: ignore[import-not-found]

        self._light_paths = {"dome": dome_path, "key": key_path}
        with rep.trigger.on_frame():
            if self.config.dome_intensity_range is not None:
                with rep.get.prims(path_pattern=re.escape(dome_path)):
                    rep.modify.attribute(
                        "inputs:intensity",
                        rep.distribution.uniform(*self.config.dome_intensity_range),
                    )
            if self.config.key_intensity_range is not None:
                with rep.get.prims(path_pattern=re.escape(key_path)):
                    rep.modify.attribute(
                        "inputs:intensity",
                        rep.distribution.uniform(*self.config.key_intensity_range),
                    )
        self._visual_ready = True
        # Registration happens once, and every later apply_lighting depends on it. Without it,
        # apply_lighting returns an empty dict and the lighting axis is absent from the dataset;
        # this line is what tells that apart from a run that applied and read back.
        _LOG.info(
            "replicator light randomizers registered: dome=%s range=%s, key=%s range=%s",
            dome_path, self.config.dome_intensity_range,
            key_path, self.config.key_intensity_range,
        )

    def apply_lighting(self, episode_idx: int) -> dict:
        """Fire the registered light randomizers for this episode, then read back the intensities.

        Seeded by ``seed + episode_idx`` and applied through ``orchestrator.preview()``, which runs
        the ``on_frame`` randomizers without taking over the timeline. ``orchestrator.step()`` does
        take it over and breaks pick execution.
        """
        if not self._visual_ready:
            return {}
        import omni.replicator.core as rep  # type: ignore[import-not-found]

        rep.set_global_seed(int(self.config.seed) + int(episode_idx))
        rep.orchestrator.preview()
        applied = self._readback_lighting()
        # Debug, one line per episode. The read-back is stage ground truth, so ``None`` here means
        # the light prim was not found, which explains a dataset whose lighting column is all null.
        _LOG.debug("episode %d: lighting applied -> %s", episode_idx, applied)
        return applied

    def _readback_lighting(self) -> dict:
        """Read the applied light intensities from the live stage (ground truth)."""
        import omni.usd  # type: ignore[import-not-found]

        stage = omni.usd.get_context().get_stage()

        def _intensity(path: str) -> float | None:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return None
            for attr_name in ("inputs:intensity", "intensity"):
                attr = prim.GetAttribute(attr_name)
                if attr and attr.Get() is not None:
                    return round(float(attr.Get()), 1)
            return None

        out: dict = {}
        if self.config.dome_intensity_range is not None:
            out["dome_intensity"] = _intensity(self._light_paths.get("dome", ""))
        if self.config.key_intensity_range is not None:
            out["key_intensity"] = _intensity(self._light_paths.get("key", ""))
        return out

    # ------------------------------------------------------------------ provenance
    def provenance(
        self, episode_idx: int, pose: EpisodePose, color_names: list[str | None], lighting: dict
    ) -> dict:
        """The ``extra["randomization"]`` payload: the parameters that were applied.

        ``color_names`` are the palette names chosen in numpy, which match the colour ``OmniPBR``
        applied; ``lighting`` is the Replicator light-intensity read-back.
        """
        return {
            "enabled": True,
            "seed": int(self.config.seed) + int(episode_idx),
            "axes": {
                "position": self.config.position_jitter_mm > 0.0,
                "orientation": self.config.orientation_jitter_deg > 0.0,
                "color": self.config.randomize_color,
                "dome": self.config.dome_intensity_range is not None,
                "key": self.config.key_intensity_range is not None,
            },
            "positions_mm": [list(p) for p in pose.positions_mm],
            "yaw_deltas_deg": list(pose.yaw_deltas_deg),
            "object_set_colors": list(color_names),
            **lighting,
        }
