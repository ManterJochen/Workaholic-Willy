"""Bring your own parts: import your meshes under the licence you declare, and describe a dataset drawn from
them alone.

The import lands in a temporary directory here. With no `destination` it lands in `assets/meshes` under the
working directory, which is the library a build draws its `custom` objects from.
"""

import tempfile
from pathlib import Path

from willy import DatasetBuild, import_from_directory

with tempfile.TemporaryDirectory() as work:
    cad = Path(work) / "cad"
    cad.mkdir()
    # A 40 mm tetrahedron stands in for your CAD export; obj, stl, ply, off, glb and gltf are read.
    (cad / "bracket.obj").write_text("v 0 0 0\nv 40 0 0\nv 0 40 0\nv 0 0 40\n"
                                     "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n", encoding="utf-8")

    # The licence has no default and the import refuses without one: nobody but you knows what your parts
    # are under, and the licence gate every build passes before it renders audits this string.
    parts = import_from_directory("custom", cad, destination=Path(work) / "meshes", license="own",
                                  attribution="your company")
    print("imported", [part.asset_id for part in parts])

    # The declaration is written beside the meshes, so the audit of a build in a later session reads it.
    print((Path(work) / "meshes" / "custom" / "LICENSE.txt").read_text(encoding="utf-8"))

# The default draws generated shapes only. Zeroing their weight still leaves them as the fallback when none
# of your parts is placeable, so refuse the fallback too: the build then stops rather than fill the dataset
# with generated shapes under your parts' name.
yours = {"procedural_weight": 0.0, "custom_weight": 1.0, "refuse_procedural_fallback": True}
build = DatasetBuild.from_file(name="my_parts", scenes=300, engine="mujoco", overrides={"assets": yours})
print(build.describe())
