"""The doctor asks each of cuRobo's kernel backends in an interpreter of its own (P0h h6).

Measured 2026-09-26, cuRobo 8e734f3, on Windows and in Ubuntu-24.04: ``get_backend_name()`` keeps the first backend it
resolves. Asked ``cuda_core`` and then ``pybind`` in one process, it answered ``cuda_core`` twice; in the other order,
``pybind`` twice. The doctor asked both in one process, so it could never see two, and it warned "1 of 2 -- only
'cuda_core' resolves" on every box, the ones with both backends built included, with a remedy to build what was
built. Its reason came from importing ``curobo._src.curobolib.geom_cu``, a module this cuRobo does not have, so the
reason was wrong as well. The redundancy check the outage of 2026-08-10 asked for was a warning that always fired.

The cuRobo here is a stand-in written for the test, which keeps its first backend exactly as the real one does, and
the probe runs through the doctor's own sidecar call against it.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from src.robot.safety.planning import doctor
from src.robot.safety.planning.doctor import ProbeStatus

_STAND_IN = {
    "torch/__init__.py": """
        __version__ = "0.0-stand-in"
        class cuda:
            @staticmethod
            def is_available():
                return False
    """,
    "curobo/__init__.py": '__version__ = "0.0-stand-in"\n',
    "curobo/_src/__init__.py": "",
    "curobo/_src/runtime.py": 'kernel_backend = "cuda_core"\n',
    "curobo/_src/curobolib/__init__.py": "",
    "curobo/_src/curobolib/backends/__init__.py": """
        from curobo._src import runtime as _runtime
        _resolved = None
        def get_backend_name():
            # as cuRobo 8e734f3: the first backend resolved is the one kept; pybind falls back when it cannot load
            global _resolved
            if _resolved is None:
                wanted = _runtime.kernel_backend
                if wanted == "pybind":
                    try:
                        import curobo._src.curobolib.backends.pybind.geometry  # noqa: F401
                    except ImportError:
                        wanted = "cuda_core"
                _resolved = wanted
            return _resolved
    """,
    "curobo/_src/curobolib/backends/pybind/__init__.py": "",
    "cuda/__init__.py": "",
    "cuda/core/__init__.py": "",
    "cuda/core/experimental/__init__.py": "",
}
_BUILT = {"curobo/_src/curobolib/backends/pybind/geometry.py": "BUILT = True\n"}


def _backend_probe(files: dict[str, str]) -> doctor.Probe:
    with tempfile.TemporaryDirectory() as root:
        for name, text in files.items():
            path = Path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text), encoding="utf-8")
        with mock.patch.object(doctor, "curobo_python_path", return_value=sys.executable), \
                mock.patch.dict(os.environ, {"PYTHONPATH": root}):
            probes = doctor._probe_curobo((), "willy_ur5e.yml")  # noqa: SLF001
    return next(probe for probe in probes if probe.name == "cuRobo kernel backends")


class EachBackendIsAskedAloneTests(unittest.TestCase):
    def test_a_box_with_both_backends_is_told_it_has_both(self) -> None:
        probe = _backend_probe({**_STAND_IN, **_BUILT})
        self.assertIs(probe.status, ProbeStatus.OK, probe.detail)
        self.assertIn("2 independent backends: cuda_core, pybind", probe.detail)

    def test_a_box_without_the_compiled_backend_is_told_why_by_the_module_that_is_missing(self) -> None:
        probe = _backend_probe(_STAND_IN)
        self.assertIs(probe.status, ProbeStatus.WARN)
        self.assertIn("only 'cuda_core' resolves", probe.detail)
        self.assertIn("No module named 'curobo._src.curobolib.backends.pybind.geometry'", probe.detail)
        self.assertNotIn("geom_cu", probe.detail)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
