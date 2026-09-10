"""Which backend settles and renders your scenes, and what each one costs.

`isaac` is a multi-gigabyte GPU install and the only path-traced RGB; `mujoco` is a pip wheel with a
real solver and no RGB; `none` needs nothing beyond this repository, seats objects analytically and
refuses the `pile` family by name rather than faking a settle. What the steps downstream consume is
a file format rather than an API, so a geometry corpus is one `none` can build on a laptop.
"""

import json
import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 06_datagen, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from datagen.api import DatasetBuild  # noqa: E402
from datagen.cost import ENGINE_COSTS, estimate, format_estimate  # noqa: E402
from datagen.render.engine import ENGINES, engine_is_available  # noqa: E402

# 1. Which of the three this interpreter can reach, and why not. The probe imports nothing heavy.
for name, what in ENGINES.items():
    print(name, engine_is_available(name), what)

# 2. The same request priced on each. Read `requested` and `usable` together: they differ because
#    `none` refuses a whole family, which at equal family weights is a quarter of every request.
for name in ENGINES:
    plan = estimate(8, engine=name, jobs=1, meshes=0)
    print(name, plan.requested_scenes, plan.usable_scenes, round(plan.hours, 2),
          ENGINE_COSTS[name].yield_fraction)
print(format_estimate(estimate(8, engine="none")))

# 3. Build with the one that needs nothing installed. Ask for `requested_scenes` to end up with 8.
build = DatasetBuild.from_file(None, name="example_corpus", engine="none", out_root="logs/examples",
                              scenes=estimate(8, engine="none").requested_scenes)
print(build.describe())

# 4. Render. `refused_family` among the status counts is the yield arriving as data, not an error.
try:
    rendered = build.render()
except OSError as unwritable:
    # The first line here that touches the disk: the writer creates logs/examples/<name>/
    # scenes before the engine seats anything.
    print(f"cannot write under logs/examples ({unwritable}); a render needs a writable "
          f"working directory")
    raise SystemExit
print(rendered.ok, dict(rendered.summary.get("by_status", {})), rendered.reason)

# 5. The engine is stamped into the dataset. Two engines never settle a scene identically, and the
#    factory refuses rather than substituting, so a corpus can always say which one made it.
print(json.loads((build.root / "provenance.json").read_text(encoding="utf-8"))["renderer"])
