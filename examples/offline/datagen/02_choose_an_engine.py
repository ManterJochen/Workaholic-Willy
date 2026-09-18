"""Choose the engine that settles and renders a dataset's scenes, and build a small dataset on the one that needs
nothing installed.

The dataset is written to a temporary directory; your own build names its `out_root`.
"""

import json
import tempfile

from willy import DatasetBuild, engine_is_available

# isaac path-traces RGB and needs its own GPU install. mujoco settles scenes with a real solver and
# renders depth and masks, no RGB. none seats objects analytically and needs nothing beyond this library.
for engine in ("isaac", "mujoco", "none"):
    available, why_not = engine_is_available(engine)
    print(f"{engine:7s}", "available" if available else why_not)

with tempfile.TemporaryDirectory() as work:
    # No file here: from_file lays these settings over the defaults, as it would over your dataset's JSON.
    build = DatasetBuild.from_file(name="first_dataset", scenes=8, seed=0, engine="none", out_root=work)
    print(build.describe())

    rendered = build.render()
    # none refuses the pile family by name rather than fake a settle, so at equal family weights a quarter
    # of the requested scenes come back refused: ask for a third more scenes than you need.
    print("rendered:", rendered.ok, dict(rendered.summary["by_status"]))

    # The engine is stamped into the dataset, because two engines never settle a scene identically.
    provenance = json.loads((build.root / "provenance.json").read_text(encoding="utf-8"))
    print("rendered by", provenance["renderer"])
