"""Every field the schema accepts, including the ones no YAML mentions.

The YAML shows what someone chose, not what may be chosen: measured, 107 schema fields appear in no
shipped YAML at all, among them ``robot.ur.model``, which decides whether a real UR3e is planned and
collision-checked as a UR3e or as a UR5e. Searching the YAML for "gripper" likewise finds five hits
and misses the entire suction end-effector.

This module is the other half: a flat, dotted index of the schema (type, constraints, default and
description per field), derived from ``AppConfig.model_json_schema()`` rather than hand-maintained,
so it cannot fall out of date with the models.

Nested models are followed through ``$ref``; a ``dict[str, Model]`` (e.g. ``robot.sim.cameras``) is
indexed once under a ``*`` segment because every key under it accepts the same shape; a list of
models is indexed under ``[]``. Recursion is depth-capped and ref-cycle-guarded so a self-referencing
schema cannot hang the CLI.

Stdlib plus the schema package: no robot-runtime import, nothing to keep in sync by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "SchemaField",
    "alias_for",
    "field_default",
    "field_doc",
    "model_doc",
    "same_value",
    "schema_index",
]

_MAX_DEPTH = 12


@dataclass(frozen=True, slots=True)
class SchemaField:
    """One accepted config key, as the schema describes it."""

    path: str
    type_name: str
    default: Any
    constraints: str
    description: str
    required: bool

    def summary(self) -> str:
        bits = [self.type_name]
        if self.constraints:
            bits.append(self.constraints)
        return ", ".join(bits)


def schema_index() -> dict[str, SchemaField]:
    """Flat ``dotted.path -> SchemaField`` for the whole :class:`AppConfig` surface."""
    from .schema.app import AppConfig

    schema = AppConfig.model_json_schema()
    defs = schema.get("$defs", {})
    out: dict[str, SchemaField] = {}
    _walk(schema, defs, "", out, 0, frozenset())
    return out


def _resolve(node: dict[str, Any], defs: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Follow a ``$ref`` (returning the def name so cycles can be detected)."""
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.split("/")[-1]
        return defs.get(name, {}), name
    return node, None


def _unwrap_optional(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """``X | None`` renders as ``anyOf: [X, null]``; index the real branch, not the union."""
    options = node.get("anyOf")
    if not isinstance(options, list):
        return node
    real = [o for o in options if isinstance(o, dict) and o.get("type") != "null"]
    return real[0] if len(real) == 1 else node


def _type_name(node: dict[str, Any]) -> str:
    if "enum" in node:
        return " | ".join(repr(v) for v in node["enum"])
    if "const" in node:
        return repr(node["const"])
    kind = node.get("type")
    if kind == "array":
        return "list"
    if isinstance(kind, str):
        return {"integer": "int", "number": "float", "string": "str", "boolean": "bool"}.get(kind, kind)
    if "anyOf" in node:
        return " | ".join(_type_name(o) for o in node["anyOf"] if isinstance(o, dict))
    return "object"


def _constraints(node: dict[str, Any]) -> str:
    parts = []
    for key, template in (
        ("minimum", ">= {}"), ("maximum", "<= {}"),
        ("exclusiveMinimum", "> {}"), ("exclusiveMaximum", "< {}"),
        ("minLength", "min length {}"), ("maxLength", "max length {}"),
    ):
        if key in node:
            parts.append(template.format(node[key]))
    return ", ".join(parts)


def _walk(
    node: dict[str, Any],
    defs: dict[str, Any],
    prefix: str,
    out: dict[str, SchemaField],
    depth: int,
    seen: frozenset[str],
) -> None:
    if depth > _MAX_DEPTH:
        return
    node, name = _resolve(node, defs)
    if name is not None:
        if name in seen:  # a self-referencing model would otherwise recurse forever
            return
        seen = seen | {name}
    properties = node.get("properties")
    if isinstance(properties, dict):
        required = set(node.get("required", []))
        for key, raw in properties.items():
            if not isinstance(raw, dict):
                continue
            path = f"{prefix}.{key}" if prefix else key
            child = _unwrap_optional(raw, defs)
            resolved, _ = _resolve(child, defs)
            out[path] = SchemaField(
                path=path,
                type_name=_type_name(resolved if "properties" in resolved else child),
                default=raw.get("default", resolved.get("default")),
                constraints=_constraints(child) or _constraints(resolved),
                description=(raw.get("description") or resolved.get("description") or "").strip(),
                required=key in required,
            )
            _walk(child, defs, path, out, depth + 1, seen)
        return
    # dict[str, Model]: every key under it accepts the same shape, so index it once under `*`.
    extra = node.get("additionalProperties")
    if isinstance(extra, dict):
        _walk(_unwrap_optional(extra, defs), defs, f"{prefix}.*", out, depth + 1, seen)
        return
    items = node.get("items")
    if isinstance(items, dict):
        _walk(_unwrap_optional(items, defs), defs, f"{prefix}[]", out, depth + 1, seen)


def _model_at(path_parts: list[str]) -> tuple[type, str] | None:
    """Resolve a dotted path to ``(owning model class, field name)``, or ``None``.

    Walks the real Pydantic models rather than the JSON schema, which has flattened away the one
    thing wanted here: which class declares the field.
    """
    import typing

    from pydantic import BaseModel

    from .schema.app import AppConfig

    def _unwrap(annotation: Any, owner: Any = None) -> Any:
        """Strip Optional / dict[str, M] / list[M] down to a BaseModel, if one is in there.

        Some annotations arrive as an unresolved ForwardRef string (``"X | None"``): the schema
        modules use ``from __future__ import annotations`` and pydantic never needed to rebuild them.
        Resolving that string against the declaring module's namespace is what lets the walk reach
        blocks such as ``recovery.fixture``, which otherwise read as undocumented for a typing reason.
        """
        import sys as _sys

        raw = getattr(annotation, "__forward_arg__", annotation)
        if isinstance(raw, str) and owner is not None:
            namespace = vars(_sys.modules.get(getattr(owner, "__module__", ""), object))
            for candidate in (part.strip().strip("'\"") for part in raw.split("|")):
                found = namespace.get(candidate)
                if isinstance(found, type) and issubclass(found, BaseModel):
                    return found
            return None
        for _ in range(4):
            args = typing.get_args(annotation)
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                return annotation
            model_args = [a for a in args if isinstance(a, type) and issubclass(a, BaseModel)]
            if not model_args:
                # ``list[M] | None`` arrives as ``(list[M], NoneType)``: neither branch is a model,
                # and the model sits one level further in. The walk descends through the branch
                # that still has arguments, or an optional list of models such as
                # ``robot.gripper.coupling_plates`` reads as undocumented for a typing reason.
                nested = [a for a in args if a is not type(None) and typing.get_args(a)]
                if not nested:
                    return None
                annotation = nested[0]
                continue
            annotation = model_args[0]
        return None

    reverse = {json: attr.rsplit(".", 1)[-1] for attr, json in _alias_pairs().items()}
    current: Any = AppConfig
    for i, part in enumerate(path_parts):
        # `*` marks a dict-of-models level and is not a field of its own: the descent already happened
        # on the segment before it. `objects[]` is a field whose items are models, so it is descended
        # through; skipping it would look for the item's fields on the containing model and find none,
        # leaving every `objects[].*` field undocumented.
        if part == "*":
            continue
        name = part[:-2] if part.endswith("[]") else part
        fields = getattr(current, "model_fields", None)
        if fields and name not in fields:
            # The path may be spelled with the YAML alias (`uniquenessRatio`); the models are keyed by
            # the attribute (`uniqueness_ratio`).
            name = reverse.get(f"{'.'.join(path_parts[:i])}.{name}".lstrip("."), name)
        if not fields or name not in fields:
            return None
        if i == len(path_parts) - 1 and not part.endswith("[]"):
            return current, name
        current = _unwrap(fields[name].annotation, current)
        if current is None:
            return None
    return None


def field_doc(path: str) -> str:
    """The comment block written immediately above a field in its schema source, dedented.

    Both ``#:`` (the Sphinx-style marker) and plain ``#`` count, because the schema sources use both:
    ``robot.sim.robot_model`` carries a three-line plain-``#`` explanation of the de-locking that field
    does, and accepting only ``#:`` reports such a field as undocumented.

    Pydantic does not lift these comments into the JSON schema, so reading the block from the source
    at query time is what surfaces it, without copying the prose into a ``description=`` that would
    then have to be kept in sync.
    """
    import inspect

    resolved = _model_at(path.split("."))
    if resolved is None:
        return ""
    model, name = resolved
    try:
        source, start = inspect.getsourcelines(model)
    except (OSError, TypeError):
        return ""
    for offset, line in enumerate(source):
        stripped = line.strip()
        if not (stripped.startswith(f"{name}:") or stripped.startswith(f"{name} :")):
            continue
        trailing = line.split("#", 1)[1].strip() if "#" in line else ""
        out: list[str] = []
        i = offset - 1
        while i >= 0 and source[i].strip().startswith("#"):
            text = source[i].strip().lstrip("#")
            out.append(text[1:].strip() if text.startswith(":") else text.strip())
            i -= 1
        _ = start  # line numbers not needed; the block is what carries the meaning
        # A trailing comment on the field's own line is the third style the schema sources use
        # (`name: str = "cube"  # identity handle for prompt-based selection`). The block above wins
        # when both exist; the trailing comment is returned when it is all there is.
        return "\n".join(reversed(out)) or trailing
    return ""


def field_default(path: str, fallback: Any = None) -> Any:
    """The real default of a field, including ``default_factory`` values.

    The JSON schema omits a ``default`` for any field built by a ``default_factory`` (every list or
    nested-model default in this tree), so a schema-only comparison reports all of those as changed
    whatever they hold. Pydantic is asked instead; the schema value stays as the fallback.
    """
    resolved = _model_at(path.split("."))
    if resolved is None:
        return fallback
    model, name = resolved
    field = getattr(model, "model_fields", {}).get(name)
    if field is None:
        return fallback
    try:
        default = field.get_default(call_default_factory=True)
    except TypeError:  # older pydantic signature
        default = field.get_default()
    return fallback if default is None and not field.is_required() and fallback is not None else default


def same_value(a: Any, b: Any) -> bool:
    """Config-equality that ignores container-type churn.

    ``model_dump()`` renders a ``tuple[float, float, float]`` field as a tuple while its default is
    written as a list, so a plain ``==`` calls identical values different.
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return bool(a == b)


def _alias_pairs() -> dict[str, str]:
    """``attribute.path -> json.path`` for every field whose YAML key differs from its Python name.

    The stereomatcher block keeps OpenCV's camelCase in YAML (``numDisparities``) behind snake_case
    attributes (``num_disparities``), the only aliased fields in the tree. The JSON schema knows only
    the alias, so without this mapping ``explain`` reports an attribute name as an unknown key while
    printing that key's value.
    """
    import typing

    from pydantic import BaseModel

    from .schema.app import AppConfig

    out: dict[str, str] = {}

    def _model(annotation: Any) -> Any:
        for _ in range(4):
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                return annotation
            models = [a for a in typing.get_args(annotation)
                      if isinstance(a, type) and issubclass(a, BaseModel)]
            if not models:
                return None
            annotation = models[0]
        return None

    def _walk_models(model: type, attr_prefix: str, json_prefix: str, depth: int, seen: frozenset) -> None:
        if depth > _MAX_DEPTH or model in seen:
            return
        seen = seen | {model}
        for name, field in getattr(model, "model_fields", {}).items():
            json_name = field.alias or name
            attr_path = f"{attr_prefix}.{name}" if attr_prefix else name
            json_path = f"{json_prefix}.{json_name}" if json_prefix else json_name
            if attr_path != json_path:
                out[attr_path] = json_path
            child = _model(field.annotation)
            if child is not None:
                _walk_models(child, attr_path, json_path, depth + 1, seen)

    _walk_models(AppConfig, "", "", 0, frozenset())
    return out


def alias_for(path: str) -> str | None:
    """The JSON/YAML path for an attribute-named ``path``, or ``None`` if it is not aliased."""
    return _alias_pairs().get(path)


def model_doc(path: str) -> str:
    """The docstring of the model that declares ``path``, a fallback when the field has none.

    A block's class docstring usually explains what the whole block is for: ``VacuumGripperConfig``
    states that every one of its numbers has to be measured on the actual cell. Only the first
    paragraph is returned, so an unexplained field answers with that instead of a bare type and range.
    """
    import inspect

    resolved = _model_at(path.split("."))
    if resolved is None:
        return ""
    doc = inspect.getdoc(resolved[0]) or ""
    paragraphs = doc.split("\n\n")
    return paragraphs[0].strip() if paragraphs else ""
