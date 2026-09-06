# `scripts/`: the things you install or run from outside Python

One folder per thing that is installed or driven from a shell, rather than a flat pile where you have
to read a filename to guess what it belongs to.

| Folder | What it does | Start here |
|---|---|---|
| [`examples/`](examples/) | seven runnable examples that drive the library API, in dependency order, from an empty cell to a pick | [`examples/README.md`](examples/README.md) |
| [`ext_deps/`](ext_deps/) | installs Coal and cuRobo into `ext_deps/`: micromamba, both environments from the committed lockfiles, the pinned cuRobo clone with both kernel backends, the UR5e and UR3e descriptors, then the doctor | [`ext_deps/README.md`](../ext_deps/README.md) |
| [`model_weights/`](model_weights/) | fetches the perception model weights (detector, segmenter, VLM) into the shared Hugging Face cache | `python scripts/model_weights/fetch.py --list` |
| [`ursim/`](ursim/) | brings URSim up in Docker under WSL and proves this stack talks to real UR controller software: the SDK, the driver, both I/O grippers, the gripper socket, the protective-stop path | [`ursim/README.md`](ursim/README.md) |
| [`curobo/`](curobo/) | builds the two cuRobo artifacts that are not shipped: the compiled kernel backend on Windows, and the per-model `{model}.yml` robot config both drivers ask the planner for. Both must run under the cuRobo environment's own interpreter | `python scripts/curobo/build_ur_config.py ur5e` |
| [`isaac/`](isaac/) | bakes `{model}_collision_meshes.npz`, the per-link collision geometry the self-collision guard checks against, out of the Isaac UR USD into `src/robot/safety/data/`. Needs an Isaac install; the baked artifact is what ships | `bake_ur_collision_meshes.py` |

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ext_deps\install.ps1   # exit code means "this box can plan"
python scripts\model_weights\fetch.py --list                            # what is fetchable, and what is cached
python -m datagen.assets.fetch --list                                   # the object meshes, and what they cost
python -m datagen cost --scenes 2000 --engine mujoco                    # what a corpus costs before you start it
```

## The object meshes are fetched by the generator, not from here

There is no `scripts/meshes/`. The mesh fetcher belongs to the thing that consumes meshes and lives at
[`datagen/assets/fetch.py`](../datagen/assets/fetch.py); the probes that grade a mesh library are
[`datagen/eval/`](../datagen/eval/). See [`datagen/README.md`](../datagen/README.md).

## Two properties both fetchers hold

**TLS verification stays fully on, and the trusted roots can come from the OS.** With `truststore`
installed (it is in `requirements.txt`), both `scripts/model_weights/fetch.py` and
`datagen/assets/fetch.py` inject the operating system trust store. That is the answer to a corporate
proxy whose CA lives in the Windows certificate store and not in `certifi`'s bundle. Only the set of
trusted roots changes, never whether the certificate is checked, which matters because these weights
run in a robot's perception path.

**The mesh fetcher checks the licence at the source, per model.** Anything that is not CC0, CC-BY or
your own is skipped before it is downloaded, rather than after. Non-commercial, ShareAlike and
NoDerivatives are all refused. Where a collection publishes no per-object licence, the meshes carry the
collection's published terms and the fetcher says so plainly rather than implying it verified something
it did not.

To bring your own parts instead:

```bash
python -m datagen.assets --fetch --source custom --from <dir> --license own --attribution-text "<you>"
```

The licence is refused rather than defaulted, because nobody but you knows what your parts are licensed
as. It is written to `LICENSE.txt` beside the meshes, where the manifest and the audit read it back
later.
