# Runtime helpers: paths, devices, atomic files, logs (`src/utility`)

The small primitives every other layer shares: where the project, its logs and its weights live, which
torch device a model gets, a file write no reader sees half done, a logger, a monotonic clock, the
millimetre scale factors and the BGR to RGB swap. You do not call this package to make a cell pick; the
library calls it for you. Reach for it in your own code to put a file or a log where the library does.

```python
from src.utility import dump_json, now_ms, unit_scaling
from src.utility.paths import debug_dir, weights_root

dump_json({"ok": True}, "out.json")     # atomic UTF-8 write: a reader never sees a partial file
value_m = 1234.567 * unit_scaling("m")  # millimetres to metres, exact
png_dir = debug_dir("detector")         # logs/debug/detector/, created and rotated to a cap
weights = weights_root()                # assets/models/hf/, where every downloaded model lives

from src.utility import get_device      # the torch helpers load torch on first use only
device = get_device()                   # cuda, then mps, then cpu, unless WILLY_DEVICE forces one
```

## Environment variables

| Variable | Effect |
| --- | --- |
| `WILLY_PROJECT_ROOT` | overrides project-root detection |
| `WILLY_LOG_DIR` | overrides the logs directory |
| `WILLY_DEBUG_DIR` | overrides the debug-image directory |
| `WILLY_DEVICE` | forces `auto`, `cuda`, `mps` or `cpu` for the torch helpers |

`WILLY_DEVICE=cuda` and `WILLY_DEVICE=mps` raise when that backend is missing rather than fall back: an
explicit choice that quietly degrades is worse than a refusal. `unit_scaling` raises `ValueError` on a
unit other than `mm`, `cm` or `m`.

## What it guarantees

**No optional dependency you did not ask for.** The leaf modules `src.utility.io`, `unit_scaling`,
`timing` and `paths` import the standard library only. The package root imports eagerly except the four
torch helpers named in `_DEVICE_EXPORTS`, which resolve on first access. `vision` is the exception: the
root imports it, and it pulls in `cv2` and NumPy. Import a leaf when you want a helper at no third-party
cost.

**Atomic writes.** `dump_json(..., atomic=True)`, the default, writes a temporary sibling in the same
directory and `os.replace`s it into place, which is atomic on Windows and Linux alike. Text is UTF-8
with LF line endings on every platform. `atomic_write_bytes` gives the same guarantee to a producer,
such as a torch checkpoint, and fsyncs before the rename so the write survives a power loss.

**Logging only when asked.** Importing a module here never configures logging. `create_logger` is
explicit and idempotent per name. Every logger pointed at one file shares one `RotatingFileHandler`,
so rotations do not race on Windows, and the console handler is shared so a record reaches stdout
once. File handlers open on the first line written, so no empty log file appears, and the helpers here
build their own logger only when they first log, so resolving a path creates no log directory.

**`device.py` and `paths.py` log what is otherwise silent**, each to its own file under `logs/utility/`:
the device each model got, a warning when auto-selection lands on the CPU or a half-precision dtype is
served as `float32` (every non-CUDA backend), and what `rotate_files` deleted. Nothing logs on a hot
path, and an error that is raised is not also logged.

**`now_ms` is monotonic.** It is `perf_counter`, not the wall clock, so only differences are meaningful.

## Files

| File | Holds |
| --- | --- |
| [`paths.py`](paths.py) | `project_root`, `logs_dir`, `debug_dir`, `ensure_dir`, `rotate_files`, `weights_root`, `fence_model_downloads` |
| [`io.py`](io.py) | `load_json`, `dump_json`, `atomic_write_text`, `atomic_write_bytes` |
| [`device.py`](device.py) | `get_device`, `is_cuda`, `resolve_torch_dtype`, `move_inputs_to_device` |
| [`log_cfg.py`](log_cfg.py) | `create_logger`, and the one place `WILLY_LOG_DIR` reaches the logs |
| [`constants.py`](constants.py) | the package log directory, its log file names, and the lazy `utility_logger` |
| [`timing.py`](timing.py) | `now_ms` and the `timed` context manager |
| [`unit_scaling.py`](unit_scaling.py) | `unit_scaling` and `SUPPORTED_DISTANCE_UNITS`, the millimetre scale factors |
| [`vision.py`](vision.py) | `bgr_to_rgb` and `rgb_to_bgr` |

`fence_model_downloads` points Hugging Face and torch hub at `weights_root()`. Call it before importing
`huggingface_hub`, `transformers` or `torch.hub`, because they read their cache location once, at import.

## Details

- [`../calibration/README.md`](../calibration/README.md) wraps `unit_scaling` in a helper that raises `CalibrationDataError`
- [`../geometry/README.md`](../geometry/README.md) holds the transforms that use the same millimetres
- [`../models/README.md`](../models/README.md) is the main user of the device and dtype helpers
- Tests: `tests/test_utility_boundaries.py`, `tests/test_logs_go_where_configured.py`
