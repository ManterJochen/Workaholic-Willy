"""Download the public meshes a dataset draws from: what exists, what it costs, what it obliges you to.

A build with no mesh library still runs, on generated shapes: parametric kinds drawn as boxes,
cylinders and spheres. A grasp number measured over those is a number about convex primitives, so
real scanned meshes are what makes a corpus mean anything, and they are not in this repository --
they are gigabytes under somebody else's licence, fetched into `assets/meshes/`, which is gitignored.

The download itself is the one line here that is not run: it is a network call of up to gigabytes,
and an example that starts one when you were reading it would be a bad example. Everything around it
runs, including the part that says what this machine already holds.
"""

from willy import DatasetBuild, MeshPreparation, available_sources

# What can be downloaded, and on what terms. `licence_verified` is the column that decides what you
# may claim: False means the collection publishes terms for itself and states nothing per object, so
# nothing here checked an individual mesh and a dataset shipped on that basis inherits the claim.
print(f"{'source':10} {'objects':>8} {'GB':>8}  licence")
for source in available_sources():
    checked = "per model" if source["licence_verified"] else "collection terms, not per model"
    print(f"{source['key']:10} {source['objects']:8d} {source['approx_gb']:8.1f}  "
          f"{source['licence']} ({checked})")

# What is on this machine. `entries()` walks the library rather than the published list, so it counts
# meshes that are actually here, under the working directory, after any earlier fetch.
public = MeshPreparation.from_sources(["gso", "ycb"])
print()
print(public.describe())
present = public.entries()
print(f"  on disk        {len(present)} mesh(es)")

if not present:
    # One call, and it resumes: a mesh already on disk is skipped, so an interrupted fetch continues.
    # `limit` takes the first n of a collection, which is a trial run and not a sample of it.
    print("\nnothing fetched yet. This downloads a trial slice of GSO, about 70 MB:")
    print("    MeshPreparation.from_sources(['gso']).fetch(limit=20)")
    print("the whole of GSO and YCB, about 4.8 GB:")
    print("    MeshPreparation.from_sources(['gso', 'ycb']).fetch(report=print)")
    print("or from a shell:  python -m datagen.assets.fetch gso --limit 20")
    print("\nCC-BY is obligating: `python -m datagen.assets --attribution` prints the text that has "
          "to ship with any dataset built from these meshes.")
    raise SystemExit

# With meshes present, a build can draw from them instead of from generated shapes. Zeroing the
# procedural weight is not enough on its own: generated shapes remain the fallback for a scene none
# of the meshes fit, so the fallback is refused too, and the build stops rather than quietly filling
# a "scanned meshes" dataset with primitives.
meshes = {"procedural_weight": 0.0, "gso_weight": 1.0, "ycb_weight": 1.0,
          "refuse_procedural_fallback": True}
build = DatasetBuild.from_file(name="from_public_meshes", scenes=500, engine="mujoco",
                               overrides={"assets": meshes})
print()
print(build.describe())
# Normalising first is not optional for a fetched collection: every mesh is read as metres, and a
# collection exported in millimetres arrives a thousand times too large.
print("\nbefore a build: MeshPreparation.from_sources(['gso']).normalise(), then .screen(<out>) to "
      "see which of them earn a grasp label at all")
