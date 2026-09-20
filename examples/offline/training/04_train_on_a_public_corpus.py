"""Train a grasp generator on a published corpus: somebody else's data, through this library's own loop.

The route for a user with no simulator and no cell. The import writes the same `.npz` scene files
`datagen` writes, so nothing after the fetch knows where the data came from: the folds, the probes
and every metric run on it unchanged. Two producers, one consumer, and the consumer cannot tell
them apart.

The fetch is by range request, not a download. The source is 229 GB and its clouds are one Zip64
split across five parts, so looking at 24 scenes costs about 5 MB rather than the archive. The
licence is the source's own and travels with the data: a model trained on it inherits the obligation.
"""

import shutil
import tempfile
from pathlib import Path

from willy import GeneratorTraining, PublicCorpus

with tempfile.TemporaryDirectory() as work:
    # Describing it opens no connection: the title, the licence, the published size, and how many
    # scenes `out_dir` already holds, which is what makes a second call resume rather than restart.
    corpus = PublicCorpus.from_source(out_dir=Path(work) / "public")
    print(corpus.describe())

    # The smoke tier is about a minute on a GPU and many times longer on a CPU alone, so the GPU is
    # checked before the network is touched.
    if shutil.which("nvidia-smi") is None:
        print("\nno NVIDIA GPU driver on this machine, and training the generator wants one. The "
              "import above needs none: corpus.fetch(limit=24) runs anywhere with a network.")
        raise SystemExit

    try:
        # `validate` reads written scenes back through the corpus loader and puts them through the
        # sample contract. Checking before training is the cheapest place to catch a frame error,
        # which is the failure that trains happily to a confident wrong answer.
        imported = corpus.fetch(limit=24, validate=3, report=print)
    except (OSError, RuntimeError, ValueError) as refusal:
        print(f"\nthe corpus could not be read ({type(refusal).__name__}: {refusal}); it needs the "
              "network, and the first call also builds an index of the remote archive.")
        raise SystemExit from None
    print(imported)
    if not imported.ok:
        raise SystemExit("nothing usable was imported; see the counts above")

    # From here it is example 02 with a different corpus directory. The hand the artifact claims is
    # the one the labels belong to, stamped into every imported scene, so nothing names it again.
    run = GeneratorTraining.from_recipe(corpus=corpus.out_dir, recipe="v1", tier="smoke",
                                        out_dir=Path(work) / "model")
    print(run.describe())
    result = run.train()
    print(result)
    print("report written to", run.write_report(result))
