"""The UR models Willy can drive: one list, shared by every block that names a model.

The canonical lower-case key ("ur5e", "ur3e", ...) is the single string the whole stack keys off:
the safety DH table (``safety/_ur_kinematics.py``), the exact-mesh bundle
(``{key}_collision_meshes.npz``), the cuRobo robot config (``{key}.yml``), the Isaac USD and Lula
config, and ``safety.self_collision.kinematics_model``. The simulated cell and the real UR cell
each name a model and must agree on which names exist, so the set lives here rather than inside
one vendor's schema. A UR3e whose config says "ur5e" resolves another robot's link lengths in
every one of those lookups, without a word, on real hardware.

The list sits in the config layer, the bottom of the dependency stack, so it must not import the
driver package. Keep it in lockstep with ``src.robot.drivers.sim.robot_models._UR_MODELS``.
"""

from __future__ import annotations

__all__ = ["UR_MODEL_KEYS"]

#: Every UR from the UR3 to the UR10, both series. The CB-series arms joined on 2026-09-09.
#:
#: Four registries answer "which UR can this stack drive", and until then none checked another:
#: this gate, the DH table, the joint limits and the sim spec registry. Measured before they were
#: reconciled, the DH table held seven models, the joint limits eight, and this gate and the spec
#: registry three. So ur10 had correct kinematics and could not be configured at all.
#:
#: The two directions fail differently and one of them is silent. A model with a DH row and no key
#: here is refused at load with "unknown UR model", which is loud and harmless. A key with no DH
#: row degrades without an error: the exact-mesh guard reports ``unknown_model`` and drops to the
#: capsule proxy, the arm-against-arm capsules disappear, and ``reach.shoulder_height_mm`` returns
#: 0.0, over-stating the working sphere by the shoulder height. A cell in that state plans and
#: picks and reports nothing. ``tests/test_ur_model_family.py`` compares all four.
#:
#: ur16e, ur20 and ur30 are deliberately absent: Isaac ships assets and this repository has no DH
#: row for them, so admitting them would be inventing kinematics.
UR_MODEL_KEYS: tuple[str, ...] = (
    "ur3", "ur3e",
    "ur5", "ur5e",
    "ur10", "ur10e",
)
