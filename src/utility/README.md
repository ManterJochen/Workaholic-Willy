# Runtime helpers: paths, devices, atomic files, logs (`src/utility`)


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
