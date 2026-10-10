""""The caller did not choose this", as data: the one sentinel and the one precedence rule.

The sentinel is not `None`, because `None` is itself a chosen value for several
settings here: `train_units=None` means every unit, `control=None` means the
corpus labels, `labelled_unit_share=None` means no rebalancing,
`baseline_pick_rate=None` means skip the non-regression key, `profile=None`
means the base tree with no overlays. A named bundle that could not separate
"train on everything" from "I said nothing" would overwrite the first while
believing it was filling in the second.

Nor is it a comparison against the default, which cannot answer the question: a
setting that is chosen and happens to equal the default is indistinguishable
from one nobody touched. Inspecting `sys.argv` answers only from a shell, since
under pytest or inside a service it holds the runner's arguments and the flag
never appears.

The argparse rule is `default=UNSET`, not `argparse.SUPPRESS`. The namespace
then always carries the attribute, holding `UNSET` when nobody typed the flag,
so the CLI hands the same data to the same resolver as a Python caller.
`SUPPRESS` removes the attribute, turning every unmigrated read into an
`AttributeError`, and combined with `parents=` and `set_defaults` it mutates the
shared Action objects, the defect `src/config/__main__.py` documents.

This is the only `UNSET` in the tree: import it rather than defining a private
sentinel. A private copy is a defect no identity check can see, because nothing
imports the copy, so catching one takes a source sweep over `src/`, `datagen/`
and `api/`.

`x is UNSET` is correct at run time and narrows nothing for a type checker,
because `_Unset` is a plain class rather than a singleton mypy can reason about.
Use :func:`chosen`, which is a `TypeGuard`, or :func:`resolve`, which is the
same test written once.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Final, TypeGuard, TypeVar, Union

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = ["UNSET", "Maybe", "chosen", "merged_overrides", "resolve"]


class _Unset:
    """The type of :data:`UNSET`. One value, compared by identity, never by equality.

    Private so `UNSET` is the only name a caller needs, and still importable for
    an annotation such as `int | _Unset`.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover (debugging aid)
        return "UNSET"

    def __bool__(self) -> bool:
        # False rather than a raised TypeError, this project's usual instinct, which would turn
        # every `if value:` into a loud failure instead of a silent wrong answer. False matches
        # the sentinel that shipped first, since two sentinels disagreeing here would make the
        # same expression mean different things depending on the module. Truthiness still cannot
        # separate an unchosen field from a chosen 0 or 0.0, and both occur here: a 0.0 threshold
        # means "do not apply it". Use chosen() for the question truthiness cannot ask.
        return False


#: "The caller did not choose this." Compare with `is`, never with `==` or truthiness.
UNSET: Final[_Unset] = _Unset()

_T = TypeVar("_T")

#: A value that may not have been chosen. `Maybe[int]` is `int | _Unset`.
Maybe = Union[_T, _Unset]


def chosen(value: "Maybe[_T]") -> "TypeGuard[_T]":
    """Did the caller actually choose this? Narrows the value for a type checker when true.

    Holds the identity test in one place, so nobody writes `!= UNSET`, which
    works until something defines `__eq__`.

    A `TypeGuard` rather than a bare `x is UNSET`, which narrows nothing:
    identity narrowing needs a singleton a type checker can reason about, and
    `_Unset` is a plain class, so a value stays `str | _Unset | None` in the
    branch where it provably cannot be `UNSET`. `isinstance(value, _Unset)`
    narrows correctly but would force every caller to import a private name to
    use a public sentinel. This is the public form of that test.

    `TypeGuard` narrows where it is true and not where it is false, so write the
    `if chosen(x):` arm as the one that uses the value::

        if chosen(profile):
            chain = validate(profile)      # profile is str | None here
        else:
            chain = from_environment()

    The reversed shape narrows no better than `is UNSET`. `TypeIs`, which
    narrows both arms, needs Python 3.13 or `typing_extensions`; this project is
    on 3.11 with neither.
    """
    return not isinstance(value, _Unset)


def resolve(name: str, *layers: "Maybe[_T]") -> _T:
    """The first layer that was actually chosen, most significant first.

    The precedence of this repository, in one function so it cannot be written
    down twice with two answers::

        max_attempts = resolve("max_attempts", explicit, from_config, 5)
        #                                      ^^^^^^^^  ^^^^^^^^^^^  ^
        #                                      caller    YAML         code default

    `AutonomousGraspService.from_robot_config` resolves `max_attempts` in that
    order: the sentinel first, then the config block, then a literal.

    The last layer must be a real value. Falling off the end is a programming
    error rather than a runtime condition, so it raises instead of returning
    `None`, which a caller would carry into a plan and fail on somewhere
    unrelated with no trace of the missing setting. `name` exists so the refusal
    can say which one.
    """
    for layer in layers:
        if not isinstance(layer, _Unset):
            return layer
    raise ValueError(
        f"nothing supplied a value for {name!r}: every layer was UNSET. The last layer passed to "
        f"resolve() is the code default and must always be a real value."
    )


_O = TypeVar("_O", bound="DataclassInstance")


def merged_overrides(overrides: _O | None, empty: Callable[[], _O], **arguments: Any) -> _O | None:
    """An overrides bundle with the settings a factory also takes as plain keywords laid into it.

    ``DetectorTraining.from_dataset(..., epochs=80)`` and ``overrides=DetectorPlanOverrides(epochs=80)`` are one
    choice written two ways; this turns the first into the second, so the recipe, the tier and the plan see one
    bundle. A setting given both ways is refused rather than one of them silently winning.

    Args:
        overrides (_O | None): The bundle the caller passed, or ``None``.
        empty (Callable[[], _O]): Builds a bundle with nothing chosen, for a caller who passed none.
        **arguments (Any): The keywords by field name; ``None`` is "not given".

    Returns:
        _O | None: The bundle with every given keyword in it; ``overrides`` unchanged where none was given.

    Raises:
        ValueError: A setting given as a keyword and in ``overrides`` as well.
    """
    given = {name: value for name, value in arguments.items() if value is not None}
    if not given:
        return overrides
    bundle = overrides if overrides is not None else empty()
    twice = sorted(name for name in given if chosen(getattr(bundle, name)))
    if twice:
        raise ValueError(f"{', '.join(twice)} given twice, as an argument and in overrides; give each setting once")
    return dataclasses.replace(bundle, **given)
