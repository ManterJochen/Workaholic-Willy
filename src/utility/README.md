# `src/utility/`: small cross-cutting runtime helpers

Device and dtype selection, atomic IO, path resolution, timing, unit scaling and colour conversion.
Shared by everything, dependent on almost nothing.

`the bottom of the stack` `leaf submodules are stdlib-only` `atomic writes by default`
`no import-time global state`

This package owns the generic primitives that belong to no camera, no calibration, no geometry, no
robot vendor and no application workflow. Every other layer imports up from here, and it stays
dependency-aware, so pulling in one helper never drags in an optional dependency the caller does not
need.

## What is in it

| Module | Owns |
|---|---|
| [`device.py`](device.py) | Torch device and dtype selection: `get_device`, `is_cuda`, `resolve_torch_dtype`, `move_inputs_to_device` |
| [`io.py`](io.py) | UTF-8 JSON and text reads, and atomic writes: `load_json`, `dump_json`, `atomic_write_text`, `atomic_write_bytes` |
| [`paths.py`](paths.py) | Project, log and debug path resolution plus bounded rotation: `project_root`, `logs_dir`, `debug_dir`, `ensure_dir`, `rotate_files` |
| [`log_cfg.py`](log_cfg.py) | Rotating-file and console logger construction: `create_logger` |
| [`constants.py`](constants.py) | The package log directory, the per-module log-file names, and the lazy `utility_logger` accessor |
| [`timing.py`](timing.py) | Monotonic timing: `now_ms` and the `timed` context manager |
| [`unit_scaling.py`](unit_scaling.py) | The canonical millimetre scale factors: `unit_scaling`, `SUPPORTED_DISTANCE_UNITS` |
| [`vision.py`](vision.py) | Generic OpenCV BGR and RGB conversion: `bgr_to_rgb`, `rgb_to_bgr` |
| [`__init__.py`](__init__.py) | Eager re-exports of the small helpers; the four torch helpers resolve on first attribute access |

## Usage

```python
from src.utility import dump_json, unit_scaling, now_ms
from src.utility.paths import debug_dir

dump_json({"ok": True}, "out.json")     # atomic UTF-8 write, so no reader sees a partial file
value_m = 1234.567 * unit_scaling("m")  # mm to m, exact, never rounded
png_dir = debug_dir("detector")         # logs/debug/detector/, created and rotated to a cap

# The torch helpers are lazy at the package root: only touching one imports torch.
from src.utility import get_device
device = get_device()                   # cuda, then mps, then cpu, unless WILLY_DEVICE forces one
```

## What it guarantees

**Dependency hygiene.** Importing a leaf submodule directly is stdlib-only and never requires torch:
`src.utility.io`, `src.utility.unit_scaling`, `src.utility.timing`, `src.utility.paths`. The package
root is nearly as cheap, because the four torch helpers named in `_DEVICE_EXPORTS` are resolved
through a module `__getattr__` on first access rather than imported eagerly. The exception is
`vision`, which the root does import eagerly and which pulls in `cv2` and NumPy. Import the leaf when
you want a helper with no third-party cost at all.

**Atomic by default.** `dump_json(..., atomic=True)`, the default, writes a temp sibling and
`os.replace`s it into place, so a reader never observes a half-written file. `os.replace` is atomic
on Windows and Linux alike when source and target sit on one filesystem, and the temp file is
created in the destination's own parent directory so that they always do. Text writes are UTF-8 with
LF line endings on every platform. `atomic_write_bytes` is the same guarantee for a producer rather
than a finished string, for a torch checkpoint that a string would have to hold twice; it fsyncs
before the rename, which is what makes the guarantee survive a power loss and not only a crashed
process.

**Logging is opt-in.** Importing a utility module never configures global logging. `create_logger` is
explicit, and it is idempotent per logger name. File handlers are keyed by absolute path, so every
logger pointing at one file shares a single `RotatingFileHandler`: a handler per logger gives each
its own file descriptor, and on Windows their rotations race. The console handler is shared once
process-wide, so a record reaches stdout exactly once.

**No import-time global state.** `create_logger` opens its file handler with `delay=True`, so the
file appears when a line is first written rather than at construction. On top of that,
`utility_logger` in `constants.py` is a cached accessor rather than the usual module-scope
`logger = create_logger(...)`, because `paths` is imported by nearly every process here, including
ones that only resolve a path and never log. Together they mean no log directory is created and no
empty `paths.log` appears until something has something to say.

**`unit_scaling` never rounds** and raises `ValueError` on an unknown unit. Calibration re-exports a
thin wrapper in `src/calibration/helpers.py` that converts that failure into a
`CalibrationDataError` for its own callers.

**`now_ms` is monotonic.** It is `perf_counter`, not wall clock, so a duration measured across an
NTP correction stays a duration. Only differences are meaningful.

**Boundaries.** These are generic primitives. They own no domain schema, no calibration policy, no
model loading and no camera orchestration.

## Two helpers log about themselves

`device.py` and `paths.py` each write to their own rotating file under `logs/utility/`, named in
[`constants.py`](constants.py). One file per helper rather than one aggregate: a device question and
a filesystem question are never read together.

They log only what is otherwise silent to the caller. `get_device` records the device each model got,
and warns when auto-selection lands on CPU. `resolve_torch_dtype` warns when a configured
half-precision dtype is served as `float32`, which happens on every non-CUDA backend and explains
later accuracy and latency differences that would otherwise look inexplicable. `rotate_files` records
what it deleted and warns when a locked file leaves a bucket above its cap.

Nothing is logged on a hot happy path, and an error that is raised is not also logged. Every other
module here, and every value object, logs nothing on purpose.

## Environment variables

| Variable | Effect |
|---|---|
| `WILLY_PROJECT_ROOT` | Override project-root detection |
| `WILLY_LOG_DIR` | Override the logs directory |
| `WILLY_DEBUG_DIR` | Override the debug-image directory |
| `WILLY_DEVICE` | Force `auto`, `cuda`, `mps` or `cpu` for the torch device helpers |

`WILLY_DEVICE=cuda` and `WILLY_DEVICE=mps` raise when that backend is unavailable, rather than
falling back: an explicit choice that silently degrades is worse than a refusal.

## Where to look next

- [`../calibration/README.md`](../calibration/README.md), which wraps `unit_scaling` into a
  `CalibrationDataError`-raising helper
- [`../geometry/README.md`](../geometry/README.md), the NumPy-only transforms that lean on the same
  millimetre scaling
- [`../models/README.md`](../models/README.md), the perception layer that consumes the device and
  dtype helpers
