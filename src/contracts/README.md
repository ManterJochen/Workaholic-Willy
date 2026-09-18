# The calling convention (`src/contracts`)

Two agreements every public noun of the library follows: what a verb hands back, and what "the caller
did not choose this" means as data. You never import this package to use the library; you meet the
convention on every name `willy` exports, and you import it when you add a noun of your own.

```python
import json

from willy import Cell, PickRun, Recording, load_tree

tree = load_tree("console_dummy")  # a chain named here; load_tree() leaves it UNSET, and WILLY_PROFILE decides
cell = Cell.rehearsal(tree.robot)  # the noun, from a factory; nothing is built or connected yet
run = PickRun.from_cell(cell, runs=1, recording=Recording.off())  # the input first, every option by keyword
report = run.execute()             # the one verb, returning a frozen report

print(report)                      # render(): the whole report as ASCII text for a person
wire = json.dumps(report.to_dict())  # the same facts as plain data, with no custom encoder
```

The package holds two files and imports `typing` alone. It imports nothing from `src`, `datagen` or a
third-party package, which is what lets it sit below everything without adding an edge to the
dependency stack. Anything that needs an import is a utility and belongs in
[`src/utility/`](../utility/README.md).

| File | Holds |
|---|---|
| [`reporting.py`](reporting.py) | `Rendered` (`render() -> str`) and `Structured` (`to_dict() -> dict`), two `@runtime_checkable` Protocols |
| [`options.py`](options.py) | `UNSET`, `Maybe[T]`, `chosen()`, `resolve()`: "not chosen" as data, and the precedence rule in one function |

## The rules

**1. The noun.** A package publishes classes named for a domain noun an operator can say out loud:
`ConfigTree`, `Cell`, `Scene`, `PickRun`, `RecordLog`, `SoakGate`, `GeneratorTraining`. No suffix,
and never a bare `Runner`, `Session`, `Trainer`, `Result` or `Report` at package top level, because
those collide the moment a top-level namespace such as `willy` exists. If the noun cannot be said to an
operator, it does not get a class.

**2. Factories, never a bare `__init__`.** `from_config(...)` or `from_robot_config(...)` is the YAML
door and takes an already validated schema object, never a path it loads itself. `from_tree(tree)` is
the door from a loaded tree: it refuses a tree that did not load with that tree's own `ConfigError`,
then hands the section it needs to `from_config`. `from_<python-input>(...)` is the plain-Python door:
`from_parts`, `from_cloud`, `from_cell`, `from_recipe`, `from_plan`. The input a factory is named for
may come first, positionally; every option after it is keyword-only. A factory that reads a source
rather than a Python value names the source instead, as `ConfigTree.from_directory`,
`RecordLog.from_jsonl` and `SoakGate.over_records` do.

**3. One builder, not two.** The YAML door resolves config into arguments and then calls the Python
door. Two doors that each assemble their own object are two objects that drift, and the drift is
silent: one path applies an overlay the other does not, and the difference surfaces as a runtime
filter that quietly rejects nothing. `RuntimePickService.from_robot_config` resolves the arm and the
gripper through [`robot_parts.py`](../robot/execution/robot_parts.py) and then returns its one
`cls.from_components(...)` call, and `Robot.from_tree` returns what `Robot.from_config` builds; copy
that shape.

**4. One verb, and options are an object.** `tree.load()`, `cell.preflight()`, `scene.grasps()`,
`run.execute()`, `log.kpis()`, `training.train()`. Everything overridable goes into one frozen options
object whose fields default to [`UNSET`](options.py), as `GraspMotion`, `SweepOptions` and
`PlanOverrides` do.

Side effects are separate methods, never constructor booleans. `training.train()` computes;
`training.write_report(report)` writes. A boolean in a constructor hides an action inside a noun.

**5. The verb returns a report, and a report has two independent halves.** See
[Why two Protocols and not one](#why-two-protocols-and-not-one).

**6. Givenness is data.** `default=UNSET` at the argparse boundary, `is UNSET` inside, resolved by
[`resolve()`](options.py) so the precedence rule, explicit before YAML before code default, exists in
exactly one place:

```python
from src.contracts import UNSET, Maybe, chosen, resolve


def attempts(explicit: Maybe[int] = UNSET, from_config: Maybe[int] = UNSET) -> int:
    return resolve("max_attempts", explicit, from_config, 5)  # the last layer is the code default


def chain(profile: Maybe[str | None] = UNSET) -> str:
    if chosen(profile):              # narrows to str | None; None is a real choice, the base tree
        return profile or "<base tree>"
    return "the chain WILLY_PROFILE names"
```

## Why two Protocols and not one

`render()` and `to_dict()` are separate obligations, and a class legitimately carries only one.

| Class | `render()` | `to_dict()` |
|---|:---:|:---:|
| `KeyExplanation` ([`src/config/explain.py`](../config/explain.py)) | yes | no |
| `IOSnapshot`, `TransitionResult` ([`src/robot/drivers/ur/io_bench.py`](../robot/drivers/ur/io_bench.py)) | yes | no |
| `TrainingRunReport` ([`src/robot/grasping/deep/train/report.py`](../robot/grasping/deep/train/report.py)) | yes | no |
| `DecisionReport` ([`src/robot/grasping/decision.py`](../robot/grasping/decision.py)) | no | yes |
| `PreflightReport` ([`src/robot/execution/real_cell/preflight.py`](../robot/execution/real_cell/preflight.py)) | yes | yes |
| `PickRunReport` ([`src/robot/execution/pick_run.py`](../robot/execution/pick_run.py)) | yes | yes |
| `AutonomousGraspReport` ([`src/robot/execution/autonomous_grasp/report.py`](../robot/execution/autonomous_grasp/report.py)) | yes | yes |

A combined protocol would owe a method at each of the first four rows, and two of those are not
reports at all: `IOSnapshot` is a bench reading and `TransitionResult` is one measured edge. Both
describe themselves to a person and neither owes anyone a wire format. So the halves are separate and
compose structurally: a class with both satisfies both, with no base class, no registration and no
import.

### What each half promises

`render()` returns the whole object, takes zero arguments, is ASCII, and ends without a trailing
newline so a caller can nest it inside a larger block. Every class `willy` exports that renders also
prints as its render, so `print(report)` and `print(report.render())` show the same text.

A renderer that takes options has two outputs, and then one caller prints the first and another the
second, which is the drift the contract exists to stop. Options belong in the object before it is
rendered. ASCII is not a preference: rendered text is printed, a Windows console is cp1252, and one
decorated glyph raises `UnicodeEncodeError` instead of printing.

A method that describes one field is a `summary` or a `describe`, keeps that name, and is not this.
[`src/robot/safety/planning/environment.py`](../robot/safety/planning/environment.py) holds one-line
`@property` fragments that its own whole-object method composes out of, and one name for all three
would make that composition unreadable.

`to_dict()` returns plain data that survives `json.dumps` with no custom encoder: no dataclass, no
`StrEnum` member (use `.value`), no `Path`, no numpy scalar. A dict that serialises only by accident
breaks the first time a field is added, and consumers here write bytes that other tools compare.

It is a view, never a second computation. The moment it derives a number `render()` does not derive,
the two halves are two answers, which is the defect the contract is against.

## The traps

**`render` means one thing.** A method that draws a debug image is `draw()`
([`src/robot/grasping/generation/_debug_renderer.py`](../robot/grasping/generation/_debug_renderer.py)),
and a function that serialises a report to the HTTP wire is `to_wire()`
([`api/routers/preflight.py`](../../api/routers/preflight.py)). A convention that tolerates two
meanings for one name is decoration.

**`runtime_checkable` checks presence, never signatures.** `isinstance(x, Rendered)` is true for a
class whose `render` takes four required arguments. That is a limit of the language, not a gap you can
close with a stricter Protocol.

**`UNSET` is not `None`.** `None` is a chosen value for several settings in this tree:
`train_units=None` means every unit, `control=None` means the corpus labels, `profile=None` means the
base tree with no overlays. A sentinel that could not separate "all of them" from "I said nothing"
would overwrite the first while believing it was filling in the second.

**`UNSET` is not a comparison against the default either.** A setting that is chosen and happens to
equal the default is indistinguishable from one nobody touched. Inspecting `sys.argv` answers only
from a shell: under a test runner or inside a service it holds the runner's arguments and the flag
never appears.

**At an argparse boundary the rule is `default=UNSET`, not `argparse.SUPPRESS`.** With `UNSET` the
namespace always carries the attribute, so the command line hands the same data to the same resolver a
Python caller does. `SUPPRESS` removes the attribute, turning every unmigrated read into an
`AttributeError`, and combined with `parents=` and `set_defaults` it mutates shared `Action` objects,
which is the defect [`src/config/__main__.py`](../config/__main__.py) documents at the parser it had
to work around.

**Import the sentinel, never define a private one.** There is exactly one `UNSET` in the tree, and a
private copy is a defect no identity check can catch, because nothing imports the copy. Catching one
takes a source sweep over `src/`, `datagen/` and `api/`, and that sweep is part of what the contract
guards.

**`x is UNSET` narrows nothing for a type checker.** Identity narrowing needs a singleton the checker
can reason about, and `_Unset` is a plain class, so a value stays `str | _Unset | None` in the branch
where it provably cannot be `UNSET`. Use `chosen()`, which is a `TypeGuard`, and write the
`if chosen(x):` arm as the one that uses the value: `TypeGuard` narrows where it is true and not where
it is false.

**The last layer of `resolve()` must be a real value.** Falling off the end is a programming error, so
it raises rather than returning `None`, which a caller would carry into a plan and fail on somewhere
unrelated with no trace of the missing setting.

## Where to read the convention working

The reference implementations are deliberately not in this package.

- [`src/robot/execution/robot.py`](../robot/execution/robot.py) is the door a user meets first:
  `from_tree` over `from_config` over `from_parts`, verbs that each return a report, and a refusal
  that is a report rather than an exception.
- [`src/robot/grasping/deep/train/api.py`](../robot/grasping/deep/train/api.py) is the whole shape
  end to end: the noun `GeneratorTraining`, keyword-only factories, one verb `train()`, a frozen typed
  report, and side effects as separate methods.
- [`src/robot/execution/real_cell/preflight.py`](../robot/execution/real_cell/preflight.py) is the
  smallest complete report: a `StrEnum` status, a frozen check, a frozen report, a derived `.ok`, one
  `render()`, one pure producer function.
