"""``python -m api``: the two things it says before it has done them.

⛔ THE DEFECT. The entry point printed

    serving   http://127.0.0.1:99999   (localhost only, no authentication)

and then died inside uvicorn with an eleven-frame asyncio traceback ending in
``OverflowError: bind(): port must be 0-65535`` (measured 2026-09-10, exit code 1). Two separate
faults in one line: nothing checked that the number IS a port, and the banner claimed the server was
serving before anything had bound. `OverflowError` is not an `OSError`, so uvicorn's own bind-failure
handler (the one that turns "port already in use" into a single line) never saw it either.

The same sentence is a lie in the ordinary case too: start a second console on a port the first one
holds and the banner still says "serving", because the claim is printed several frames before the
socket exists.

⚠ NOTHING HERE STARTS A SERVER. ``uvicorn.run`` is replaced by a recorder, so the assertions are
about what ``main`` decided, and the one socket these tests open is opened by the test itself.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import socket
import unittest


_HAS_UVICORN = importlib.util.find_spec("uvicorn") is not None


@unittest.skipIf(not _HAS_UVICORN, "requirements.txt is not installed (optional extra)")
class EntryPointPortTests(unittest.TestCase):
    """What the CLI does with a port before it prints anything about one."""

    def setUp(self) -> None:
        import uvicorn

        from api.cell import console, set_console

        #: Every (host, port) uvicorn was asked to serve. Empty is the assertion that matters when
        #: the port is refused: a guard that reports an error and starts the server anyway is not
        #: a guard.
        self.served: list[tuple[object, object]] = []

        def _record(app: object, **kwargs: object) -> None:
            self.served.append((kwargs.get("host"), kwargs.get("port")))

        self._real_run, uvicorn.run = uvicorn.run, _record
        self.addCleanup(setattr, uvicorn, "run", self._real_run)
        # `main` installs its own Console process-wide. Put the previous one back, or every later
        # test in this process is talking to a console built here.
        self.addCleanup(set_console, console())

    def _main(self, argv: list[str]) -> tuple[int, str, str]:
        from api.__main__ import main

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_a_number_that_is_not_a_port_is_refused_instead_of_reaching_the_socket(self) -> None:
        """65536 and up, 0, and negatives. `--port 99999` is a typo, and it read as a crash."""
        for port in ("99999", "0", "-1", "70000"):
            with self.subTest(port=port):
                self.served.clear()
                code, out, err = self._main(["--port", port])
                self.assertEqual(code, 2, f"--port {port} was accepted")
                self.assertNotIn("serving", out, "it announced a server on an impossible port")
                self.assertIn("65535", err, "the refusal does not say what a port is")
                self.assertEqual(self.served, [], "uvicorn was handed the impossible port anyway")

    def test_a_port_already_in_use_is_not_announced_as_serving(self) -> None:
        """The everyday case: a console is already running, and this one is started on top of it.

        The banner is the operator's evidence that the thing they started is the thing in their
        browser, so it must not be printed until the port is actually theirs.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            port = taken.getsockname()[1]

            code, out, err = self._main(["--port", str(port)])

        self.assertNotEqual(code, 0, "a console that could not bind reported success")
        self.assertNotIn("serving", out, "it announced a server it had not bound")
        self.assertIn(str(port), err, "the refusal does not name the port")
        self.assertEqual(self.served, [], "uvicorn was started on a port that was already taken")

    def test_a_free_port_is_announced_and_served(self) -> None:
        """The other half: the guard must not refuse the ordinary start."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        code, out, _ = self._main(["--port", str(port)])

        self.assertEqual(code, 0)
        self.assertIn(f"serving   http://127.0.0.1:{port}", out)
        self.assertEqual(self.served, [("127.0.0.1", port)])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
