#!/usr/bin/env bash
# Build cuRobo's COMPILED (pybind) kernel backend on Linux, the twin of build_compiled_backend.bat (P0h h5).
#
# The environment's lock carries nvcc and the GCC it accepts, so nothing here needs root. What the bare
# `CUROBO_USE_PYBIND=1 python setup.py build_ext --inplace` lacks was measured in Ubuntu-24.04 on 2026-09-26, one
# addition at a time, against a distribution with no compiler of its own:
#
#   1. nvcc and torch look for `gcc`, `g++` and `c++` on PATH ("gcc: No such file or directory", then "nvcc fatal:
#      Failed to preprocess host compiler properties"). The environment's bin/ holds them; nothing activates it.
#   2. conda-forge puts the CUDA headers under targets/x86_64-linux/include, not under CUDA_HOME/include where
#      torch's cpp_extension looks, so g++ fails on "cuda_runtime_api.h: No such file or directory". The Windows
#      twin of this is the .bat's trap 4 (import libraries flat in Library\lib).
#
# The link needs nothing more: torch falls back from CUDA_HOME/lib64, which conda-forge does not make, to
# CUDA_HOME/lib and passes it with -L itself.
#
# CUROBO_USE_PYBIND=1 is the switch: without it setup.py compiles nothing and reports success.
#
# Usage (from anywhere):
#     bash scripts/curobo/build_compiled_backend.sh
#
# Optional: set CUROBO_ENV / CUROBO_SRC first to override the in-repo defaults.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUROBO_ENV=${CUROBO_ENV:-$here/../../ext_deps/curobo_env}
CUROBO_SRC=${CUROBO_SRC:-$here/../../ext_deps/curobo}

if [ ! -x "$CUROBO_ENV/bin/python" ]; then
  echo "[error] no python at $CUROBO_ENV/bin/python" >&2
  echo "        create the env first: python3 scripts/ext_deps/install.py --component curobo" >&2
  exit 2
fi
if [ ! -f "$CUROBO_SRC/setup.py" ]; then
  echo "[error] no cuRobo source at $CUROBO_SRC" >&2
  echo "        clone it first: python3 scripts/ext_deps/install.py --component curobo" >&2
  exit 2
fi
for tool in nvcc gcc g++; do
  if [ ! -x "$CUROBO_ENV/bin/$tool" ]; then
    echo "[error] no $tool in $CUROBO_ENV/bin: the environment was not made from curobo_env.linux-64.lock" >&2
    exit 2
  fi
done

export CUROBO_USE_PYBIND=1
export CUDA_HOME=$CUROBO_ENV
export PATH=$CUROBO_ENV/bin:$PATH
export CPATH=$CUROBO_ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}

echo "[1/2] compiling CUDA kernels (this takes several minutes)"
(cd "$CUROBO_SRC" && "$CUROBO_ENV/bin/python" setup.py build_ext --inplace)

echo "[2/2] verifying that the compiled backend is selectable"
# torch first, as on Windows: cuRobo imports it before it touches a backend, and the check should see what real
# usage sees. The name is checked, not only printed: a check that prints another backend's name and exits 0 would
# be a green without the spare.
"$CUROBO_ENV/bin/python" -c "import torch; import curobo._src.runtime as rt; rt.kernel_backend='pybind'; from curobo._src.curobolib.backends import get_backend_name; name = get_backend_name(); print('   backend ->', name); raise SystemExit(0 if name == 'pybind' else 1)"

echo
echo "done. cuRobo now has two independent kernel backends."
