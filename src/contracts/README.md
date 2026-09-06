# `src/contracts/`: the calling convention, named once

Two agreements the whole repository follows: what a verb hands back, and what "the caller did not
choose this" means as data.

`stdlib only` `adds no edge to the dependency stack` `structural: implementers never import it`

Every capability here reaches an operator through `python -m <pkg>` and through Python, and those
two callers have to be the same answer in two costumes. That takes an agreement about the shape of a
result and an agreement about the shape of an unset argument. Both live here, each under one name.

## What is in it

| File | Role |
|---|---|
| [`reporting.py`](reporting.py) | `Rendered` (`render() -> str`) and `Structured` (`to_dict() -> dict`), two `@runtime_checkable` Protocols |
| [`options.py`](options.py) | `UNSET`, `Maybe[T]`, `chosen()`, `resolve()`: "not chosen" as data, and the precedence rule in one function |

The package imports `typing` and nothing else. It does not import from `src`, from `datagen` or from
any third-party package, which is what lets it sit below everything without adding an edge to the
dependency stack. Anything that needs an import is a utility and belongs in
[`src/utility/`](../utility/README.md).

## The five rules

**1. The noun.** A package publishes classes named for a domain noun an operator can say out loud:
`ConfigTree`, `Cell`, `Scene`, `PickRun`, `RecordLog`, `SoakGate`, `GeneratorTraining`. No suffix,
and never a bare `Runner`, `Session`, `Trainer`, `Result` or `Report` at package top level, because
those collide the moment a top-level namespace exists. If the noun cannot be said to an operator, it
is not a noun and it does not get a class.

**2. Two factories, never a bare `__init__`.** `from_config(...)` or `from_robot_config(...)` is the
YAML door and takes an already validated schema object, never a path it loads itself.
`from_<python-input>(...)` is the plain-Python door: `from_components`, `from_cloud`, `from_records`,
`from_recipe`, `from_plan`. Both keyword-only. A factory that reads a source rather than a Python
value names the source instead, as `ConfigTree.from_directory`, `RecordLog.from_jsonl` and
`SoakGate.over_records` do.

**2b. One builder, not two.** The YAML door resolves config into arguments and then calls the Python
door. This is not style. Two doors that each assemble their own object are two objects that drift,
and the drift is silent: one path applies an overlay the other does not, and the difference surfaces
as a runtime filter that quietly rejects nothing. `RuntimePickService.from_robot_config` returns
`cls.from_components(...)`; copy that shape.

**3. One verb, and options are an object.** `tree.load()`, `cell.preflight()`, `scene.grasps()`,
`run.execute()`, `log.kpis()`, `training.train()`. Everything overridable goes into one frozen
options object whose fields default to [`UNSET`](options.py).

Side effects are separate methods, never constructor booleans. `training.train()` computes;
`training.write_report(report)` writes. A boolean in a constructor hides an action inside a noun.

**4. The verb returns a report, and a report has two independent halves.** See below.

**5. Givenness is data.** `default=UNSET` at the argparse boundary, `is UNSET` inside, resolved by
[`resolve()`](options.py) so the precedence rule, explicit before YAML before code default, exists
in exactly one place.

## Why two Protocols and not one

`render()` and `to_dict()` are separate obligations, and a class legitimately carries only one.

| Class | `render()` | `to_dict()` |
|---|:---:|:---:|
| `KeyExplanation` (`src/config/explain.py`) | yes | no |
| `IOSnapshot`, `TransitionResult` (`src/robot/drivers/ur/io_bench.py`) | yes | no |
| `TrainingRunReport` (`src/robot/grasping/deep/train/report.py`) | yes | no |
| `DecisionReport` (`src/robot/grasping/decision.py`) | no | yes |
| `PreflightReport` (`src/robot/execution/real_cell/preflight.py`) | yes | yes |
| `PromotionReport` (`src/robot/grasping/calibration/model_promotion.py`) | yes | yes |
| `AutonomousGraspReport` (`src/robot/execution/autonomous_grasp/report.py`) | yes | yes |

A combined protocol would owe a method at each of the first four rows, and two of those are not
reports at all: `IOSnapshot` is a bench reading and `TransitionResult` is one measured edge. Both
describe themselves to a person and neither owes anyone a wire format. So the halves are separate
and compose structurally: a class with both satisfies both, with no base class, no registration and
no import.

### What each half promises

`render()` returns the whole object, takes zero arguments, is ASCII, and ends without a trailing
newline so a caller can nest it inside a larger block.

A renderer that takes options has two outputs, and then one caller prints the first and another the
second, which is the drift the contract exists to stop. Options belong in the object before it is
rendered. ASCII is not a preference: rendered text is printed, a Windows console is cp1252, and one
decorated glyph raises `UnicodeEncodeError` instead of printing.

A method that describes one field is a `summary` or a `describe`, keeps that name, and is not this.
`src/robot/safety/planning/environment.py` holds one-line `@property` fragments that its own
whole-object method composes out of, and one name for all three would make that composition
unreadable.

`to_dict()` returns plain data that survives `json.dumps` with no custom encoder: no dataclass, no
`StrEnum` member (use `.value`), no `Path`, no numpy scalar. A dict that serialises only by accident
breaks the first time a field is added, and consumers here write bytes that other tools compare.

It is a view, never a second computation. The moment it derives a number `render()` does not derive,
the two halves are two answers, which is the defect the contract is against.

## The traps

**`render` means one thing.** Two other meanings were plausible and are both settled: a method that
draws a debug image is `draw()` (`src/robot/grasping/generation/_debug_renderer.py`), and a function
that serialises a report to the HTTP wire is `to_wire()` (`api/routers/preflight.py`). A convention
that tolerates two meanings for one name is decoration.

**`runtime_checkable` checks presence, never signatures.** `isinstance(x, Rendered)` is true for a
class whose `render` takes four required arguments. That is a limit of the language, not a gap you
can close with a stricter Protocol.

**`UNSET` is not `None`.** `None` is a chosen value for several settings in this tree:
`train_units=None` means every unit, `control=None` means the corpus labels, `profile=None` means
the base tree with no overlays. A sentinel that could not separate "all of them" from "I said
nothing" would overwrite the first while believing it was filling in the second.

**`UNSET` is not a comparison against the default either.** A setting that is chosen and happens to
equal the default is indistinguishable from one nobody touched. Inspecting `sys.argv` answers only
from a shell: under a test runner or inside a service it holds the runner's arguments and the flag
never appears.

**At an argparse boundary the rule is `default=UNSET`, not `argparse.SUPPRESS`.** With `UNSET` the
namespace always carries the attribute, so the CLI hands the same data to the same resolver a Python
caller does. `SUPPRESS` removes the attribute, turning every unmigrated read into an
`AttributeError`, and combined with `parents=` and `set_defaults` it mutates shared `Action`
objects, which is the defect `src/config/__main__.py` documents at the parser it had to work around.

**Import the sentinel, never define a private one.** There is exactly one `UNSET` in the tree, and a
private copy is a defect no identity check can catch, because nothing imports the copy. Catching one
takes a source sweep over `src/`, `datagen/` and `api/`, and that sweep is part of what the contract
guards. A convention spreads faster than its own contract, so a copy written after this file existed
is the ordinary case, not the surprising one.

**`x is UNSET` narrows nothing for a type checker.** Identity narrowing needs a singleton the
checker can reason about, and `_Unset` is a plain class, so a value stays `str | _Unset | None` in
the branch where it provably cannot be `UNSET`. Use `chosen()`, which is a `TypeGuard`, and write
the `if chosen(x):` arm as the one that uses the value: `TypeGuard` narrows where it is true and not
where it is false.

**The last layer of `resolve()` must be a real value.** Falling off the end is a programming error,
so it raises rather than returning `None`, which a caller would carry into a plan and fail on
somewhere unrelated with no trace of the missing setting.

## Where to read the convention working

A contract with no adopter is a wish. The reference implementations are deliberately not in this
package.

- [`src/robot/grasping/deep/train/api.py`](../robot/grasping/deep/train/api.py) is the whole shape
  end to end: the noun `GeneratorTraining`, keyword-only factories, one verb `train()`, a frozen
  typed report, and side effects as separate methods.
- [`src/robot/execution/real_cell/preflight.py`](../robot/execution/real_cell/preflight.py) is the
  smallest complete report: a `StrEnum` status, a frozen check, a frozen report, a derived `.ok`,
  one `render()`, one pure producer function.
