"""The two requirements files describe one dependency set, and drift between them is silent.

`requirements.txt` is the supported install and `requirements-cpu.txt` is the same set for a host
that cannot take the CUDA wheels. They are allowed to differ in exactly three lines: the extra
index URL and the two torch pins. Everything else is the same package at the same version, in the
same order.

That parity is what makes it safe for continuous integration to install the CPU file. The runner
has no GPU, the CUDA wheels are several gigabytes it would never execute, and the coverage gate
imports `api` and `datagen`, so a package present in one file and missing from the other would
either fail the job for a reason that has nothing to do with the change, or pass the job against a
smaller set than the one that ships.

The failure this guards is adding a dependency to one file and forgetting the other. Nothing about
that produces an error until a host installs the file that lost the line.
"""

from __future__ import annotations

import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GPU = _ROOT / "requirements.txt"
_CPU = _ROOT / "requirements-cpu.txt"

#: The three lines each file is allowed to hold that the other does not.
_GPU_ONLY = ("--extra-index-url https://download.pytorch.org/whl/cu128",
             "torch==2.7.1+cu128",
             "torchvision==0.22.1+cu128")
_CPU_ONLY = ("--extra-index-url https://download.pytorch.org/whl/cpu",
             "torch==2.7.1",
             "torchvision==0.22.1")


def _requirements(path: pathlib.Path) -> list[str]:
    """The file's requirement lines, without comments and blank lines."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


class RequirementsParityTests(unittest.TestCase):
    def test_both_files_exist(self) -> None:
        self.assertTrue(_GPU.is_file(), _GPU)
        self.assertTrue(_CPU.is_file(), _CPU)

    def test_the_torch_lines_are_the_declared_ones(self) -> None:
        gpu, cpu = _requirements(_GPU), _requirements(_CPU)
        self.assertEqual(tuple(gpu[:3]), _GPU_ONLY)
        self.assertEqual(tuple(cpu[:3]), _CPU_ONLY)

    def test_everything_else_is_identical_and_in_order(self) -> None:
        gpu = [line for line in _requirements(_GPU) if line not in _GPU_ONLY]
        cpu = [line for line in _requirements(_CPU) if line not in _CPU_ONLY]
        self.assertEqual(gpu, cpu, "the two requirements files have drifted apart")

    def test_no_second_torch_pin_hides_further_down(self) -> None:
        for path in (_GPU, _CPU):
            names = [line.split("==")[0].strip().lower()
                     for line in _requirements(path) if "==" in line]
            for package in ("torch", "torchvision"):
                self.assertEqual(names.count(package), 1, f"{package} in {path.name}")

    def test_the_gate_tools_are_pinned(self) -> None:
        """Continuous integration installs one of these files and then runs ruff, mypy and pytest."""
        for path in (_GPU, _CPU):
            names = {line.split("==")[0].strip().lower()
                     for line in _requirements(path) if "==" in line}
            for tool in ("ruff", "mypy", "pytest", "pytest-cov"):
                self.assertIn(tool, names, f"{tool} is not pinned in {path.name}")


if __name__ == "__main__":
    unittest.main()
