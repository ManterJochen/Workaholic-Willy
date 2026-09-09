"""The labels are geometry and the screen is physics. Which of the two you ship.

Labelling enumerates the grasps each object's geometry admits, from antipodal structure rather than
by searching, in minutes and with no GPU. What it does not know is short: no inverse kinematics, so
a grasp it calls real may be unreachable, and no dynamics, which is what the screen below adds.
"""

import json
import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 06_datagen, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from datagen.api import DatasetBuild  # noqa: E402
from datagen.config import DatagenConfig  # noqa: E402
from datagen.corpus.clouds import physics_verdicts  # noqa: E402
from datagen.grasps.service import PhysicsSampling  # noqa: E402
from datagen.render.engine import engine_is_available  # noqa: E402

# 1. Address a dataset already on disk: a name and a root address one, and `from_config` because
#    nothing here re-renders it. Re-labelling is minutes where a re-render is a night.
dataset = Path("logs/examples/example_corpus")
if not (dataset / "index.jsonl").is_file():
    print(f"no dataset at {dataset} -- choose_an_engine.py builds it in about ten seconds")
    raise SystemExit
build = DatasetBuild.from_config(DatagenConfig(scenes=1, seed=0), name=dataset.name,
                                 out_root=dataset.parent)

# 2. Label every scene. Closed form, CPU only, writes grasps.jsonl into the dataset.
labelled = build.label(density="default")
print(labelled.ok, labelled.summary["jaw"], labelled.summary["suction"], labelled.summary["objects"])

# 3. The jaw model and the resolved density are in the report because a label COUNT means nothing
#    without them: the same object earns an order of magnitude more labels at `grid`.
summary = json.loads((dataset / "grasp_label_report.json").read_text(encoding="utf-8"))
print(summary["jaw_model"], summary["scenes"], summary["partial"], summary["density"])

# 4. Why it refused what it refused. `finger_collision` is the one about your scene.
for reason, count in sorted(summary["rejected"].items(), key=lambda row: -row[1])[:6]:
    print(reason, count)

# 5. The screen: teleport a jaw to a label, close it, take the table away, see whether the object
#    stayed. A separate referee over jaw rows, needing mujoco or isaac in this interpreter.
available, why_not = engine_is_available("mujoco")
if not available:
    print("no physics screen:", why_not)
    raise SystemExit
verdict = PhysicsSampling.from_dataset(dataset, engine="mujoco").sample(per_class=4)
print(verdict.render())

# 6. The join back, keyed by (file, line) and never by pose. A refusal is dropped, not scored 0.
verdicts = physics_verdicts(dataset / verdict.out_name)
print(len(verdicts), sum(verdicts.values()))
