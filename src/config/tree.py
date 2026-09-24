"""A config tree as one value: the directory, the chain, and the layers that chain splits into.

The root, the profile and the layers have to agree, and `explain_in`, `decisions`, `index_chains`
and `set_keys` each take them as independent arguments beside an already-loaded config. Nothing
binds those arguments to one another, so a config loaded under one chain can be walked for
provenance under another and the answer comes back confidently wrong, with no error raised:

  * on the read path the value comes from the config and the origin from `layers`, so the tool
    whose whole purpose is "which layer set this" reports `robot.sim.enabled = True` above
    `robot/robot.yaml:31`, a line that sets `false`.
  * on the write path `set_keys` picks the target file from `layers[-1]` and validates under
    `profile`, so `layers=("ur3e",)` with `profile=None` writes `mass_kg: 3.25` into
    `robot.ur3e.yaml` beneath the `max_mass_kg: 3.0` that forbids it, validates the base tree
    instead, reports `applied=True`, and leaves the tree unloadable under the layer it wrote.

So the ask methods hang off the tree and take only a key, and `layers` is derived from `profile`
rather than passed beside it: passing both is what lets them disagree, and deriving one from the
other is what stops it. Holding the three facts in one object is not by itself the fix:
`api/cell.py`'s `Console` carries root, profile and layers together, and
`api/routers/config.py:56` and `:138` unpack it into four and five separate arguments by hand.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover, typing only
    from .edit import WriteResult
    from .explain import KeyExplanation
    from .schema import AppConfig, RobotConfig

__all__ = ["ConfigTree", "LoadedTree", "default_data_dir", "load_tree"]


def default_data_dir() -> Path:
    """The directory `load_config()` reads when nobody names one.

    The public accessor for `loader.py:83`'s private `_DEFAULT_DATA_DIR`, a `parents[2]` walk no
    other file can repeat from its own location. It stands in for four independent rebuilds of that
    walk: the constant itself, `src/config/__main__.py` twice in one function, `api/cell.py:45` and
    `willy_sim/config.py:53`. `explain`, `decisions` and `set_keys` take `root` as a required
    argument, and a wrong root gives a silently wrong answer rather than an error, so a caller asks
    here instead of deriving one.
    """
    from .loader import _DEFAULT_DATA_DIR  # noqa: PLC0415

    return _DEFAULT_DATA_DIR


def _registry_hands(root: Path, config: Any) -> tuple[str, ...]:
    """The hands in the tree's gripper registry, with the cell's hand resolved, or a ConfigError.

    The registry is read whenever the tree has a ``grippers/`` directory, so a hand file that cannot
    be read, an alias two hands claim, or a cell naming a hand no file describes refuses the tree it
    sits in. A tree without one validates while it names no hand; naming a hand there is the refusal
    ``load_gripper`` gives. The hand is resolved with ``aliases=False``, the lookup behind
    ``robot.gripper.model``, so a short name is refused here too, naming its model. Refusing a cell
    that names no hand at all is the build's job, not the validator's.

    The hand the cell names must then equal the repository's: a tree that describes it differently,
    or describes a hand the repository does not, is refused naming both places, because the hand's
    committed body, sphere map, retract rows and evidence were written from the repository's file.
    A hand the cell does not name is not compared.
    """
    from .grippers import available_grippers, load_gripper, tree_hand_refusal  # noqa: PLC0415
    from .loader import ConfigError  # noqa: PLC0415

    hands = tuple(available_grippers(root)) if (root / "grippers").is_dir() else ()
    model = getattr(getattr(getattr(config, "robot", None), "gripper", None), "model", None)
    if model:
        load_gripper(model, data_dir=root, aliases=False)
        refusal = tree_hand_refusal(model, data_dir=root)
        if refusal is not None:
            raise ConfigError(refusal)
    return hands


def _registry_cameras(root: Path, config: Any) -> None:
    """Refuse an unreadable camera registry, or a rig's camera the repository does not hold alike.

    The camera registry is read whenever the tree has a ``cameras/`` directory, so a broken camera file
    refuses the tree it sits in, as a broken hand file does. Every camera a rig's ``body`` names,
    enabled or not, is then looked up in the repository's registry, the one authority, and a tree copy
    that describes it differently is refused naming both files. A tree whose rigs declare no body and
    that holds no ``cameras/`` reads nothing.
    """
    from .cameras import available_cameras, load_camera, tree_camera_refusal  # noqa: PLC0415
    from .loader import ConfigError  # noqa: PLC0415

    if (root / "cameras").is_dir():
        available_cameras(root)
    rigs = getattr(getattr(getattr(config, "camera", None), "cameras", None), "rigs", None) or ()
    for model in sorted({rig.body.model for rig in rigs if getattr(rig, "body", None) is not None}):
        load_camera(model)
        refusal = tree_camera_refusal(model, data_dir=root)
        if refusal is not None:
            raise ConfigError(refusal)


def _no_values() -> Mapping[str, Any]:
    return MappingProxyType({})


def _flattened(values: Mapping[str, object], prefix: str = "") -> dict[str, Any]:
    """``values`` as dotted keys to leaves, every key checked and every value copied.

    A mapping value merges key by key, so ``{"robot.gripper": {"model": m}}`` sets the one key
    ``{"robot.gripper.model": m}`` sets. A list, an empty mapping and a scalar are leaves, set
    whole.

    A key that is not a dotted string, or a value that is a model object rather than the plain
    data a YAML file holds, is a programmer error and raises. A model object would reach the named
    hand fill as a block it cannot read, and the fill would be skipped without a word.
    """
    from .loader import _value_segments  # noqa: PLC0415

    if not isinstance(values, Mapping):
        raise TypeError(
            f"with_values takes a mapping of dotted keys to values, not {type(values).__name__}"
        )
    out: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"a config key is a dotted string such as 'robot.gripper.model', not {key!r}")
        dotted = f"{prefix}.{key}" if prefix else key
        _value_segments(dotted)
        if hasattr(value, "model_dump"):
            raise TypeError(
                f"{dotted} is given a {type(value).__name__}; with_values takes plain data as a YAML file "
                "holds it, so name the keys that change"
            )
        if isinstance(value, Mapping) and value:
            out.update(_flattened(value, dotted))
        else:
            out[dotted] = copy.deepcopy(value)
    return out


def _merged_values(earlier: Mapping[str, Any], later: Mapping[str, Any]) -> dict[str, Any]:
    """``earlier`` with ``later`` on top: a key of ``later`` replaces the same key, every key inside
    it and every block holding it, so the last value given for a place is the one in force."""
    from .loader import _holds, _in_memory_keys, _normal_key, _within  # noqa: PLC0415

    replaced = _in_memory_keys(later)
    kept = {
        key: value for key, value in earlier.items()
        if not (_within(_normal_key(key), replaced) or _holds(_normal_key(key), replaced))
    }
    return {**kept, **later}


def _plain(value: Any) -> Any:
    """A value given in memory as data `json.dumps` takes with no encoder: the view for `to_dict`."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


@dataclass(frozen=True, slots=True)
class LoadedTree:
    """A validated tree, and the three facts it was validated under.

    The ask methods live here rather than on `ConfigTree` because they need the loaded config as
    well as the root and the layers, and this is the only object that holds all three. A method
    taking the config as an argument would let the three disagree again.

    A noun that spans sections takes this object and reads it under names that stay: `app_config`
    and `robot` for the validated sections, `root` for the directory the registries and the
    evidence are read from, `profile` and `layers` for the chain, and `values` for what
    `with_values` gave in memory. A door reads the sections from here and never loads `root` under
    `profile` a second time: the values given in memory live only in this object, and a second
    load drops them without a word.
    """

    tree: "ConfigTree"
    #: The validated `AppConfig`, or `None` when the tree did not load.
    config: Any = None
    #: The loader's refusal text verbatim, empty when the tree loaded.
    error: str = ""
    #: The hands in the tree's gripper registry, sorted. Empty when the tree has no registry.
    hands: tuple[str, ...] = ()
    #: What `with_values` gave in memory on top of the files, dotted key to value in the caller's
    #: spelling. Empty for a tree read from its files alone. Out of the hash: a value may be a list.
    values: Mapping[str, Any] = field(default_factory=_no_values, hash=False)

    @property
    def ok(self) -> bool:
        return self.config is not None

    @property
    def exit_code(self) -> int:
        """The process exit code: 0 when the tree loaded, 1 when it did not.

        The same 1 as the CLI's `except ConfigError: return 1`, which is the only source of a 1 in
        that entry point.
        """
        return 0 if self.ok else 1

    @property
    def chain(self) -> str:
        """The layers as a person reads them, or ``(no profile)``.

        This is the chain that was loaded, including whatever `WILLY_PROFILE` contributed, not the
        one that was asked for.
        """
        return " -> ".join(self.tree.layers) or "(no profile)"

    # --- what a door reads ---------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The directory this tree was read from, resolved: the repository's `config/` when none
        was named. Every `data_dir=` downstream takes it, the registry checks read `grippers/`
        and `cameras/` from it, and every relative path the tree holds was read against it
        (`src.config.paths`), so the validated sections carry absolute paths."""
        return self.tree.root

    @property
    def profile(self) -> str | None:
        """The chain as `load_config(profile=...)` takes it, such as `"sim,ur3e"`, or `None` for
        the base tree. `chain` is the same fact for a person."""
        return self.tree.profile

    @property
    def layers(self) -> tuple[str, ...]:
        """The chain split into its layers, in the order they merge."""
        return self.tree.layers

    @property
    def app_config(self) -> "AppConfig":
        """The validated `AppConfig`, or the tree's own refusal raised as `ConfigError`.

        `config` is the same object, or `None` when the tree did not load, which the report needs.
        A door needs the config or a refusal, and this gives one of the two.
        """
        if self.config is None:
            from .loader import ConfigError  # noqa: PLC0415

            raise ConfigError(self.error)
        return self.config

    @property
    def robot(self) -> "RobotConfig":
        """The validated `robot` section, or `ConfigError`: the tree's own refusal when it did
        not load, and the sentence `load_robot_config` gives when it loaded without a robot block."""
        from .loader import _NO_ROBOT_BLOCK, ConfigError  # noqa: PLC0415

        robot = getattr(self.app_config, "robot", None)
        if robot is None:
            raise ConfigError(_NO_ROBOT_BLOCK)
        return robot

    # --- a change in memory --------------------------------------------------------------------

    def with_values(self, values: Mapping[str, object]) -> "LoadedTree":
        """This tree with `values` given in memory, loaded as a load loads it.

        Nothing is written. `values` maps a dotted key to its value,
        `{"robot.gripper.model": "robotiq_hande"}`, with `[n]` for an item of a list the tree holds
        (`"camera.cameras.rigs[1].enabled"`). The files are read again under this tree's root and
        chain, so a file edited since this tree loaded is read as it is now. The values merge on
        top of every layer, and then the load runs as it always runs: the named hand fills the
        thirteen numbers it supplies, the schema validates the whole tree, and the registries are
        checked. A value that does not validate comes back as a tree that did not load, with the
        load's own refusal, where the key is said to be written in `LoadedTree.with_values` rather
        than at a file line that holds another value.

        The values this tree already holds stay, and a key given again, a key inside it or a block
        holding it takes the new value. A mapping value merges key by key; a list, an empty
        mapping and a scalar replace; `None` sets `None`. A relative path given here is read against
        this tree's folder, as the same value written in a layer would be (`src.config.paths`), so
        a path a program computed goes in absolute. A key that is not a dotted string, or a
        model object given as a value, is a programmer error and raises `ValueError` or
        `TypeError`.

        `model_copy` is no substitute: it runs no validator and skips the named hand fill, so
        naming the Hand-E that way keeps the 2F-85's widths and finger geometry and raises nothing.
        """
        return self.tree._load(_merged_values(self.values, _flattened(values)))

    # --- the questions, each taking only what it is about --------------------------------------

    def explain(self, key: str) -> "KeyExplanation":
        """Everything known about one key: its value here, and which file decided it.

        `explain_in` reads the value out of a config and walks `root` plus `layers` for the origin.
        All three come from this tree, so the value and the provenance cannot describe two
        different loads.

        A key given in memory is said to be set in `LoadedTree.with_values`. The file lines that
        also write it stay in the chain, and none of them wins.
        """
        from .explain import Layer, _yaml_path, explain_in  # noqa: PLC0415

        explanation = explain_in(self.config, key, self.tree.root, self.tree.layers)
        if not self.values or not explanation.known:
            return explanation
        from .loader import _IN_MEMORY_ORIGIN, _normal_key  # noqa: PLC0415

        try:
            spelled = {_normal_key(key), _normal_key(_yaml_path(key))}
        except ValueError:  # a key that is no dotted path was given nothing in memory
            return explanation
        given = [name for name in self.values if _normal_key(name) in spelled]
        if not given:
            return explanation
        # Printing a file line as `set in` here would put the value given in memory above a line
        # that sets another one, the disagreement the module docstring describes.
        stated = Layer(location=_IN_MEMORY_ORIGIN, raw=repr(self.values[given[-1]]), winner=True)
        files = tuple(replace(layer, winner=False) for layer in explanation.layers)
        return replace(explanation, layers=(*files, stated), comment="", derived_from="")

    def decisions(
        self, *, section: "Maybe[str | None]" = UNSET, tier: "Maybe[str | None]" = UNSET
    ) -> str:
        """Only the values that differ from their schema default: what someone actually decided.

        Neither filter is defaulted here: `explain.decisions` already declares `section=None` and
        `tier=None`, and repeating them in this signature would declare one fact twice.
        """
        from .explain import decisions as _decisions  # noqa: PLC0415

        extra: dict[str, Any] = {}
        if chosen(section):
            extra["section"] = section
        if chosen(tier):
            extra["tier"] = tier
        if self.values:
            from .loader import _in_memory_keys  # noqa: PLC0415

            # A value given in memory is said to be set there, never at the file line it replaced.
            extra["given"] = _in_memory_keys(self.values)
        return _decisions(self.config, self.tree.root, self.tree.layers, **extra)

    # --- the report halves ---------------------------------------------------------------------

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The verdict as one ASCII line, no trailing newline, taking no arguments.

        A tree given values in memory names their keys, so its verdict is never read as the files'.
        """
        memory = ", ".join(self.values)
        if not self.ok:
            if memory:
                return f"config error, with {memory} given in memory:\n{self.error}"
            return f"config error:\n{self.error}"
        named = str(self.tree.named_root) if self.tree.named_root is not None else "<default>"
        line = f"OK: config under {named} validates.  layers: {self.chain}"
        if memory:
            line += f"  in memory: {memory}"
        return f"{line}  hands: {', '.join(self.hands)}" if self.hands else line

    def to_dict(self) -> dict[str, Any]:
        """Plain data: the tree, the chain, the values given in memory and the verdict.

        The config itself is not in here. It is a Pydantic model with its own `model_dump_json`,
        and a second serialisation of it would be a second answer.
        """
        return {
            "root": str(self.tree.root),
            "profile": self.tree.profile,
            "layers": list(self.tree.layers),
            "chain": self.chain,
            "hands": list(self.hands),
            "ok": self.ok,
            "exit_code": self.exit_code,
            "error": self.error,
            "values": {key: _plain(value) for key, value in self.values.items()},
        }


@dataclass(frozen=True, slots=True)
class ConfigTree:
    """Where the YAML lives and which overlays are in force, as one value.

        from src.config import ConfigTree

        loaded = ConfigTree.from_directory(profile="sim").load()
        print(loaded.render())
        print(loaded.explain("robot.sim.enabled").render())

    `layers` is derived from `profile` and never passed: the two are the same fact in two shapes,
    and a caller allowed to supply both is allowed to make them disagree.
    """

    root: Path
    #: The chain string. `None` means the base tree with no overlays, and it is a real value: a
    #: caller who has not chosen is `UNSET` at the factory and gets whatever `WILLY_PROFILE` says.
    profile: str | None
    layers: tuple[str, ...] = ()
    #: The root as the caller named it, or `None` when they named none. Only for `render()`, which
    #: echoes the operator's own words rather than the resolved path.
    named_root: str | None = field(default=None, compare=False)
    #: What to call `profile` in a refusal: `"profile"` when the caller chose it, `"WILLY_PROFILE"`
    #: when this class read it out of the environment. Wording only, so it is out of `compare`.
    #:
    #: Without it the refusal blamed a flag nobody typed. Resolving an unchosen profile here and
    #: then passing the result as `load_config(profile=...)` makes an environment chain arrive
    #: through the argument door, and the argument door's refusal says `profile=`. Measured
    #: 2026-09-10: `WILLY_PROFILE=nosuch python -m src.config` and `python -m src.config
    #: --profile nosuch` printed the same sentence.
    profile_source: str = field(default="profile", compare=False)

    @classmethod
    def from_directory(
        cls,
        *,
        root: "Maybe[str | Path | None]" = UNSET,
        profile: "Maybe[str | None]" = UNSET,
    ) -> "ConfigTree":
        """The tree at ``root`` under ``profile``, with the layers derived from the chain.

        All three states of `profile` are meaningful: `UNSET` is "the caller did not choose" and
        lets `WILLY_PROFILE` decide, `None` is "the base tree, ignore the variable", and a string
        is that chain. Collapsing the first two silently disables an exported variable for an
        operator who set it deliberately.
        """
        from .loader import _PROFILE_ENV_VAR, active_profile, profile_layers  # noqa: PLC0415

        named = None if not chosen(root) or root is None else str(root)
        resolved_root = Path(named).resolve() if named is not None else default_data_dir()
        # The origin is recorded where it is still known. One line down the chain it is just a
        # string and the two origins are indistinguishable, which is how the refusal came to name a
        # flag the operator had not passed.
        chain = profile if chosen(profile) else active_profile()
        return cls(
            root=resolved_root,
            profile=chain,
            layers=profile_layers(chain),
            named_root=named,
            profile_source="profile" if chosen(profile) else _PROFILE_ENV_VAR,
        )

    def load(self) -> LoadedTree:
        """Read and validate. A tree that does not load is a verdict, not an exception.

        The profile travels as an argument rather than through the environment, which keeps the
        `source` that `_validated_chain` carries pointing at what the operator typed: a bad
        `--profile` is reported against the flag, not against `WILLY_PROFILE`.

        And then it blamed the flag instead, because this class resolves the variable itself. Every
        chain leaves here through the argument door, whoever chose it, and that door's refusal says
        `profile=`. Measured 2026-09-10, `WILLY_PROFILE=nosuch python -m src.config` printed the
        same sentence as `--profile nosuch`, so the operator was sent looking for a flag they had
        not passed. `profile_source` carries the origin the rest of the way.
        """
        return self._load({})

    def _load(self, values: Mapping[str, Any]) -> LoadedTree:
        """The load with `values` given in memory on top of the files; `load` gives none.

        One path serves both, so a tree given values meets the checks a tree read from its files
        alone meets, in the same order.
        """
        from .loader import ConfigError, _load_with_values, _validated_chain, load_config  # noqa: PLC0415

        given: Mapping[str, Any] = MappingProxyType(dict(values))
        try:
            # The same validator, called first, only so the refusal is worded honestly.
            # `load_config` runs it with `source="profile"` because that is the door it owns, and it
            # cannot know that the chain reaching it came out of the environment. Calling it here
            # with the origin this object recorded means a mistyped layer is refused naming what the
            # operator actually did. On a good chain this is one `rglob` per layer and the load then
            # validates identically; on a bad one it raises before `load_config` is reached, so
            # there is exactly one refusal either way.
            _validated_chain(self.root, self.profile, source=self.profile_source)
            config = (
                _load_with_values(self.named_root, profile=self.profile, values=given) if given
                else load_config(self.named_root, profile=self.profile)
            )
            hands = _registry_hands(self.root, config)
            _registry_cameras(self.root, config)
        except ConfigError as exc:
            return LoadedTree(tree=self, error=str(exc), values=given)
        return LoadedTree(tree=self, config=config, hands=hands, values=given)

    def write(self, items: "Mapping[str, Any]", *, connected: bool) -> "WriteResult":
        """Write measured values into this tree as one transaction: all land, or none do.

        `connected` has no default. `set_keys` declares `connected: bool = False`, the permissive
        value, and its one production caller, `api/routers/config.py:138`, has
        `CellSession.connected` one attribute away. A caller that leaves it out skips the
        `requires_disconnected` guard on the two controller-address keys, `robot.ur.ip` and
        `robot.kuka.controller_ip`: the machine that receives every motion is repointable from the
        browser while the cell is live, with nothing in the result saying the guard did not run.

        `root`, `layers` and `profile` come from the tree, so the write cannot land in a file that
        a different chain then validates. `set_keys` raises on that disagreement; here it is
        unconstructible.
        """
        from .edit import set_keys  # noqa: PLC0415

        return set_keys(
            items,
            root=self.root,
            layers=self.layers,
            profile=self.profile,
            connected=connected,
        )


def load_tree(
    profile: "Maybe[str | None]" = UNSET, *, root: "Maybe[str | Path | None]" = UNSET
) -> LoadedTree:
    """The tree at ``root`` under ``profile``, loaded in one call.

        tree = load_tree()               # the cell WILLY_PROFILE names
        tree = load_tree("console_dummy")
        tree = load_tree(None, root="D:/cells/line3")

    It is ``ConfigTree.from_directory(root=root, profile=profile).load()``. Unset ``profile`` is
    the chain ``WILLY_PROFILE`` names, so a program run as ``WILLY_PROFILE=<cell> python ...``
    names no robot itself; ``None`` is the base tree; a string is that chain. Unset ``root`` is the
    repository's tree. A tree that does not load comes back with ``ok`` false and its refusal, as
    ``ConfigTree.load`` returns it.
    """
    return ConfigTree.from_directory(root=root, profile=profile).load()
