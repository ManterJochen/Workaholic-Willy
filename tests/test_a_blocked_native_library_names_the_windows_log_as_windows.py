"""A native library the system refused to load names the Windows event log as Windows' only (P0h h2, owner 7a).

Two refusals tell an operator that ur_rtde or CoACD IS installed and could not be loaded, and both ended in "check
the CodeIntegrity event log", which is Windows' application control log: on Linux there is none, and the fault is a
missing system library or an ABI mismatch. Each now puts that log under "On Windows" and says what to look for
elsewhere. The rest of each sentence is as it was, "IS installed" and "reinstalling will not help" included.
"""

from __future__ import annotations

import builtins
import unittest
from unittest import mock


class TheCodeIntegrityLogIsWindowsOnlyTests(unittest.TestCase):

    def test_ur_rtde_blocked_puts_the_event_log_under_windows(self) -> None:
        from src.robot.drivers.ur import connection

        said = connection._unloadable(OSError("[WinError 4551] a policy blocked this file"))  # noqa: SLF001
        self.assertIn("On Windows, check the CodeIntegrity event log before changing anything.", said)
        self.assertNotIn("mismatch. Check the CodeIntegrity", said)
        self.assertIn("reinstalling will not help", said)

    def test_coacd_blocked_puts_the_event_log_under_windows_and_says_what_else_it_is(self) -> None:
        from datagen.render import convex_decomposition

        real = builtins.__import__

        def refuse(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202 - an __import__ double
            if name == "coacd":
                raise OSError("[WinError 4551] a policy blocked this file")
            return real(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", refuse):
            said = convex_decomposition.decomposition_refusal()
        assert said is not None
        self.assertIn("On Windows this is usually Smart App Control refusing an unsigned build whose reputation it "
                      "re-evaluated: read the CodeIntegrity event log before changing anything.", said)
        self.assertIn("Elsewhere it is a missing system library or an ABI mismatch.", said)
        self.assertIn("reinstalling will not help", said)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
