"""Which objects a corpus is made of: a public collection, or the parts your cell handles.

The shipped default is `procedural_weight: 1.0` and every other source weight `0.0`, so an untouched
corpus is parametric boxes. `import_from_directory` has no default licence and raises without one,
because `audit_asset_rows` reads that string before any render, by clause and not by prefix:
`cc-by-sa` and `cc-by-nd` both start with `cc-by`, and both are refused.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 06_datagen, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from datagen.assets.library import (  # noqa: E402
    MESH_LIBRARY_DIR, MeshLibrary, declared_license, import_from_directory)
from datagen.assets.licensing import audit_asset_rows  # noqa: E402
from datagen.assets.service import MeshPreparation, available_sources  # noqa: E402

# 1. What this machine already holds, per source. Zero everywhere is normal: meshes are fetched
#    into `assets/meshes`, never vendored into the repository.
print(MESH_LIBRARY_DIR, MeshLibrary().counts())

# 2. What the public collections offer. `licence_verified` False means terms published for the
#    collection and nothing per object, so a dataset built on it inherits a claim, not a check.
for row in available_sources():
    print(row["key"], row["objects"], row["approx_gb"], row["licence"], row["licence_verified"])

# 3. Fetch one, small. `limit` takes the first n of a collection, which is a trial and not a
#    sample. An already-present mesh is skipped, so this is a no-op once the slice is on disk.
print(MeshPreparation.from_sources(["thingi10k"]).fetch(library=MESH_LIBRARY_DIR, limit=5).render())

# 4. Your own parts. A tetrahedron stands in for the CAD you would point this at. `license=` is
#    required and refused rather than defaulted: nobody but you knows what your parts are under.
parts = Path("logs/examples/my_cad")
parts.mkdir(parents=True, exist_ok=True)
(parts / "bracket.obj").write_text("v 0 0 0\nv 40 0 0\nv 0 40 0\nv 0 0 40\n"
                                   "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n", encoding="utf-8")
entries = import_from_directory("custom", parts, destination="logs/examples/meshes",
                                license="own", attribution="ACME GmbH")

# 5. The declaration travels WITH the meshes, as a file beside them, because the manifest and the
#    CI audit read it in a later session than the one that typed it.
print(len(entries), declared_license("logs/examples/meshes/custom"))

# 6. The gate every manifest passes before a render. An empty list is clean; every other row comes
#    back with its own reason, all of them at once rather than one per run.
print(audit_asset_rows([{"id": "bracket", "source": "custom", "license": "own"}]))
print(audit_asset_rows([{"id": "mug", "source": "gso", "license": "cc-by-nd-4.0"},
                        {"id": "tray", "source": "gso", "license": "CC-BY-4.0"}]))
