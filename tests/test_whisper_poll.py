"""Iteration-2 regression test: the Whisper ``listen_once`` loop is bounded and
no longer spins forever on silence / mic failure / blank output.

The bounded loop lives in the torch-free ``_poll`` helper so it is testable on
any host (the parent ``speech_to_text`` module imports torch).
"""

from __future__ import annotations

import unittest

from src.models.speech._poll import poll_until_text


class PollUntilTextTests(unittest.TestCase):
    def test_returns_first_truthy_text(self) -> None:
        seq = iter([None, None, "hello"])
        sleeps = []

        def drain() -> str | None:
            return next(seq)

        def sleep(ms: int) -> None:
            sleeps.append(ms)

        text = poll_until_text(
            drain=drain, sleep=sleep, now=lambda: 0.0, max_attempts=10
        )
        self.assertEqual(text, "hello")
        self.assertEqual(sleeps, [100, 100, 100])

    def test_returns_none_on_max_attempts(self) -> None:
        attempts = {"n": 0}

        def drain() -> str | None:
            attempts["n"] += 1
            return None

        text = poll_until_text(
            drain=drain, sleep=lambda ms: None, now=lambda: 0.0, max_attempts=5
        )
        self.assertIsNone(text)
        self.assertEqual(attempts["n"], 5)

    def test_returns_none_on_timeout(self) -> None:
        clock = {"t": 0.0}

        def now() -> float:
            return clock["t"]

        def sleep(ms: int) -> None:
            clock["t"] += 1.0  # each poll advances the monotonic clock by 1 s

        text = poll_until_text(
            drain=lambda: None, sleep=sleep, now=now, timeout_s=3.0
        )
        self.assertIsNone(text)
        self.assertGreaterEqual(clock["t"], 3.0)

    def test_text_wins_before_bound(self) -> None:
        seq = iter([None, "found", "later"])
        text = poll_until_text(
            drain=lambda: next(seq),
            sleep=lambda ms: None,
            now=lambda: 0.0,
            max_attempts=10,
        )
        self.assertEqual(text, "found")


if __name__ == "__main__":
    unittest.main()
