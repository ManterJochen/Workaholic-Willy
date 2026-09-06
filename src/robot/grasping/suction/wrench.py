"""Suction wrench-resistance model, the second score of the published analytic vacuum model.

Seal formation, in :mod:`.seal`, answers whether the cup can seal here. Wrench resistance answers whether
it holds the part once sealed. A compliant suction cup under gauge vacuum ``P`` over area ``A`` provides:

* a pull-off force along the normal up to ``F_vac = P * A``, resisting being pulled off the surface;
* a tangential force up to ``mu * F_vac``, the Coulomb friction under the vacuum preload;
* a bending moment up to ``M_max = F_vac * r_cup * k``, the elastic ring resisting tilt.

The part is held exactly when the external gravity wrench about the contact lies inside that resistible
set: the gravity force ``m*g`` decomposed into a normal and a shear component at the contact, and its
moment ``m*g * lever``, where ``lever`` is the offset of the centre of mass perpendicular to gravity. The
score is the tightest of the three feasibility margins, so a contact near the centre of mass, with a small
lever, on a light enough part scores high, while a far-offset contact or a too-heavy part scores low.

Implemented from the published statics in pure numpy, with no Isaac and no learned weights, and with no
code or data taken from the papers, which notice cites under "Prior art". Real air-leak and material
fidelity stays a real-hardware concern, and the Protocol in :mod:`.scorer` is where a data-driven scorer
would attach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WrenchConfig", "WrenchResult", "evaluate_wrench_resistance"]

_G_MPS2 = 9.81


@dataclass(frozen=True, slots=True)
class WrenchConfig:
    """Vacuum-cup wrench-resistance parameters, held close to SI units with millimetres and grams at the boundary."""

    vacuum_kpa: float = 50.0        # gauge vacuum pressure; a typical industrial cup pulls 40 to 70 kPa
    cup_radius_mm: float = 15.0
    friction_coef: float = 0.5      # rubber cup on a dry surface
    # The resistible moment as a fraction of F_vac * r_cup, the restoring moment of the elastic ring. 1.0
    # is a reasonable cup, and a smaller value means a stiffer or smaller cup that tips more easily.
    moment_arm_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.vacuum_kpa <= 0.0:
            raise ValueError(f"vacuum_kpa must be > 0, got {self.vacuum_kpa}")
        if self.cup_radius_mm <= 0.0:
            raise ValueError(f"cup_radius_mm must be > 0, got {self.cup_radius_mm}")
        if self.friction_coef < 0.0:
            raise ValueError(f"friction_coef must be >= 0, got {self.friction_coef}")
        if self.moment_arm_factor <= 0.0:
            raise ValueError(f"moment_arm_factor must be > 0, got {self.moment_arm_factor}")

    @property
    def vacuum_force_n(self) -> float:
        """Maximum pull-off force ``F_vac = P * A``, in newtons."""
        area_m2 = np.pi * (self.cup_radius_mm / 1000.0) ** 2
        return float(self.vacuum_kpa * 1000.0 * area_m2)


@dataclass(frozen=True, slots=True)
class WrenchResult:
    """Wrench-resistance evaluation at one contact, where ``resist_score`` is the tightest of the pull-off, shear and moment margins."""

    resist_score: float
    feasible: bool
    vacuum_force_n: float
    required_pull_n: float
    shear_n: float
    required_moment_nmm: float
    max_moment_nmm: float


def evaluate_wrench_resistance(
    contact_point_mm: np.ndarray,
    approach: np.ndarray,
    *,
    payload_mass_g: float,
    com_mm: np.ndarray,
    config: WrenchConfig | None = None,
    gravity_dir: np.ndarray | None = None,
) -> WrenchResult:
    """Whether a vacuum cup at ``contact_point_mm``, pressing along ``approach``, holds ``payload_mass_g`` given its ``com_mm`` and its ``gravity_dir``, which defaults to world ``-Z``."""
    cfg = config or WrenchConfig()
    p = np.asarray(contact_point_mm, dtype=np.float64).reshape(3)
    a = np.asarray(approach, dtype=np.float64).reshape(3)
    a = a / float(np.linalg.norm(a))
    com = np.asarray(com_mm, dtype=np.float64).reshape(3)
    g = np.asarray(gravity_dir if gravity_dir is not None else [0.0, 0.0, -1.0], dtype=np.float64)
    g = g / float(np.linalg.norm(g))

    mass_kg = max(float(payload_mass_g), 0.0) / 1000.0
    f_grav = mass_kg * _G_MPS2  # N, gravity magnitude
    f_vac = cfg.vacuum_force_n

    # Gravity force decomposed at the contact. The cup sealed on the face whose outward normal is -approach,
    # since it pressed in along approach, and the part separates by moving along +approach, away from the
    # cup. The pull-off demand is therefore the gravity component along +approach: a top suction, with the
    # approach down and gravity down, demands the full weight. The shear is the in-plane remainder, and on
    # a side suction all of gravity is shear.
    grav_vec = f_grav * g
    pull_off = max(0.0, float(np.dot(grav_vec, a)))    # component pulling the part off the cup (N)
    shear = float(np.linalg.norm(grav_vec - np.dot(grav_vec, a) * a))  # tangential component (N)

    # Moment about the contact from gravity acting at the CoM: |(com - p) x F_grav|.
    lever = (com - p) / 1000.0  # m
    moment_nm = float(np.linalg.norm(np.cross(lever, grav_vec)))
    moment_nmm = moment_nm * 1000.0
    max_moment_nmm = f_vac * cfg.cup_radius_mm * cfg.moment_arm_factor

    # Feasibility ratios (limit / demand); inf when there is no demand.
    def _ratio(limit: float, demand: float) -> float:
        if demand <= 1e-9:
            return float("inf")
        return limit / demand

    ratios = [
        _ratio(f_vac, pull_off),                    # pull-off vs vacuum
        _ratio(cfg.friction_coef * f_vac, shear),   # shear vs friction
        _ratio(max_moment_nmm, moment_nmm),         # tilt vs elastic moment
    ]
    tightest = min(ratios)
    resist_score = float(np.clip(tightest, 0.0, 1.0))
    feasible = bool(tightest >= 1.0)
    return WrenchResult(
        resist_score=resist_score,
        feasible=feasible,
        vacuum_force_n=f_vac,
        required_pull_n=pull_off,
        shear_n=shear,
        required_moment_nmm=moment_nmm,
        max_moment_nmm=max_moment_nmm,
    )
