"""The typed grasp sampling mode contract.

This is the operator-facing grasping mode surface. The internal switch of the
calculator is a tri-state ``dense_sampling`` of ``True``, ``False`` or ``None``, and a
caller should not have to remember which bool means what. A caller configures the
pipeline with a typed enum and the bridge here maps the enum onto the bool, so the
calculator core is unchanged.

The contract is locked:

* ``GraspSamplingMode.AUTO`` becomes ``dense_sampling=None``, the heuristic.
* ``GraspSamplingMode.SINGLE_OBJECT`` becomes ``dense_sampling=False``.
* ``GraspSamplingMode.DENSE_CLUTTER`` becomes ``dense_sampling=True``.

A bool caller keeps working: ``True`` maps to ``DENSE_CLUTTER`` and ``False`` to
``SINGLE_OBJECT``. A string is accepted from a closed allow-list, for YAML and CLI
usability, and anything else raises ``ValueError`` with an explicit hint.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "GraspSamplingMode",
    "GraspSamplingModeInput",
    "mode_to_dense_sampling",
    "resolve_grasp_sampling_mode",
]


class GraspSamplingMode(StrEnum):
    """The operator-facing grasp sampling mode.

    The values are stable strings, so they travel through YAML config, JSON telemetry
    and a wire payload without translation.
    """

    AUTO = "auto"
    SINGLE_OBJECT = "single_object"
    DENSE_CLUTTER = "dense_clutter"


# The public input alias, used in a signature that accepts the typed enum, the bool
# form, a stable string alias, or ``None``, which means ``AUTO``.
GraspSamplingModeInput = "GraspSamplingMode | bool | str | None"


_STR_ALIASES: dict[str, GraspSamplingMode] = {
    "auto": GraspSamplingMode.AUTO,
    "single": GraspSamplingMode.SINGLE_OBJECT,
    "single_object": GraspSamplingMode.SINGLE_OBJECT,
    "dense": GraspSamplingMode.DENSE_CLUTTER,
    "dense_clutter": GraspSamplingMode.DENSE_CLUTTER,
}


def resolve_grasp_sampling_mode(
    value: GraspSamplingMode | bool | str | None,
) -> GraspSamplingMode:
    """Coerce any accepted input into a :class:`GraspSamplingMode`.

    The accepted inputs are:

    * an existing :class:`GraspSamplingMode`, returned as it is;
    * ``None``, which becomes :attr:`GraspSamplingMode.AUTO`;
    * ``True``, the bool form of :attr:`GraspSamplingMode.DENSE_CLUTTER`;
    * ``False``, the bool form of :attr:`GraspSamplingMode.SINGLE_OBJECT`;
    * a case-insensitive string from ``{"auto", "single", "single_object", "dense",
      "dense_clutter"}``.

    Any other value raises :class:`ValueError` listing the allowed forms, so a caller
    sees exactly what they passed and what is accepted.
    """
    if isinstance(value, GraspSamplingMode):
        return value
    if value is None:
        return GraspSamplingMode.AUTO
    # ``bool`` is a subclass of ``int``, so it is tested before any generic int
    # handling. An arbitrary int is deliberately not accepted, so a stray ``0`` or ``1``
    # cannot silently mean a mode.
    if isinstance(value, bool):
        return (
            GraspSamplingMode.DENSE_CLUTTER if value else GraspSamplingMode.SINGLE_OBJECT
        )
    if isinstance(value, str):
        key = value.strip().lower()
        mapped = _STR_ALIASES.get(key)
        if mapped is not None:
            return mapped
    allowed = (
        "GraspSamplingMode enum, None, True/False, or one of "
        + ", ".join(sorted(_STR_ALIASES))
    )
    raise ValueError(
        f"Invalid grasp_sampling_mode value {value!r}; expected {allowed}."
    )


def mode_to_dense_sampling(mode: GraspSamplingMode) -> bool | None:
    """Bridge the enum onto the internal ``dense_sampling`` tri-state of the calculator.

    * ``AUTO`` becomes ``None``, which lets the auto heuristic decide.
    * ``SINGLE_OBJECT`` becomes ``False``, which forces the silhouette path.
    * ``DENSE_CLUTTER`` becomes ``True``, which forces the dense surface sampler.
    """
    if mode is GraspSamplingMode.AUTO:
        return None
    if mode is GraspSamplingMode.SINGLE_OBJECT:
        return False
    if mode is GraspSamplingMode.DENSE_CLUTTER:
        return True
    raise ValueError(f"Unhandled GraspSamplingMode: {mode!r}")
