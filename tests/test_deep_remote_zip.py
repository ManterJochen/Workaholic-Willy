"""Reading single members from a large remote zip without downloading it.

⛔ WHY THESE TESTS ARE WORTH THEIR LENGTH. The reader turns byte offsets into HTTP ranges against a
203 GB archive. Every failure mode here is silent: a wrong offset does not raise, it returns somebody
else's bytes, and those bytes then become training data. The two that matter most are pinned first.

    a split archive's offsets are CONCATENATED, so a read may straddle a part boundary
    the classic 32-bit fields OVERFLOW past 4 GB, and the Zip64 extra field is POSITIONAL

⚠ These tests already changed the design once. The short-range guard lived inside the transport
method, so the fake transport below dropped the guard along with the request, and the test for the
guard could not fail for the right reason. Transport and check are separate now.

The second is the nastier one. The first entries of a big archive parse correctly from the 32-bit
fields, so a reader that ignores the extra field passes any test written against the head of the
archive and fetches garbage from everything after 4 GB.
"""

from __future__ import annotations

import io
import struct
import unittest
import unittest.mock
import zipfile

from src.robot.grasping.deep.foreign.remote_zip import (
    RemoteZip, ZipMember, _parse_directory, _zip64_values)


class _LocalZip(RemoteZip):
    """The reader against bytes in memory, split into parts exactly as the real archive is."""

    def __init__(self, blob: bytes, pieces: int = 3) -> None:
        step = len(blob) // pieces + 1
        self.blob = blob
        chunks = [blob[i:i + step] for i in range(0, len(blob), step)]
        super().__init__([f"part{i}" for i in range(len(chunks))], [len(c) for c in chunks])
        self._chunks = chunks

    def _get(self, url: str, start: int, end: int) -> bytes:
        return self._chunks[self.parts.index(url)][start:end + 1]


def _archive(names_to_bytes: dict[str, bytes], *, method: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=method) as handle:
        for name, payload in names_to_bytes.items():
            handle.writestr(name, payload)
    return buffer.getvalue()


class SplitTests(unittest.TestCase):
    """⛔ Offsets are in concatenated coordinates and reads straddle part boundaries."""

    def test_a_member_is_read_correctly_across_the_parts(self) -> None:
        payloads = {f"scene_{i}.bin": bytes([i % 251]) * 40_000 for i in range(12)}
        archive = _LocalZip(_archive(payloads), pieces=5)
        members = {m.name: m for m in archive.directory()}
        self.assertEqual(len(members), len(payloads))
        for name, expected in payloads.items():
            with self.subTest(name):
                self.assertEqual(archive.read(members[name]), expected)

    def test_a_span_that_crosses_two_parts_is_stitched_in_order(self) -> None:
        blob = bytes(range(256)) * 40
        archive = _LocalZip(blob, pieces=4)
        boundary = archive._bounds[1][0]
        self.assertEqual(archive._span(boundary - 30, 60), blob[boundary - 30:boundary + 30])

    def test_a_span_outside_the_archive_refuses(self) -> None:
        archive = _LocalZip(bytes(1000), pieces=2)
        with self.assertRaises(ValueError):
            archive._span(900, 500)

    def test_a_host_that_ignores_the_range_header_is_named(self) -> None:
        """A server answering 200 with a whole 48 GB part would otherwise look like a hang."""

        class _Ignores(_LocalZip):
            def _get(self, url: str, start: int, end: int) -> bytes:
                return self.blob                       # a host that answers 200 with the whole file

        with self.assertRaises(OSError) as caught:
            _Ignores(_archive({"a": b"x" * 100})).directory()
        self.assertIn("range requests", str(caught.exception))


class BackoffTests(unittest.TestCase):
    """⛔⛔ A RATE LIMIT IS A SLOWDOWN, NOT A FAILURE, and treating it as one cost 41 % of an import.

    MEASURED: fetching with eight workers, 153 of 369 scenes came back `HTTP 429` and were counted as
    permanently failed; the serial run before it had none.

    ⚠ And the benchmark that chose eight workers did not predict it. It read 0.8, 3.2, 6.5, 10.6 and
    13.5 requests per second at one to twenty-four workers, which looked like clean scaling. It ran
    for three seconds. A short benchmark against a rate-limited service measures the BURST ALLOWANCE
    rather than the sustained rate, and those are different numbers.
    """

    def test_a_429_is_retried_and_then_succeeds(self) -> None:
        import urllib.error

        blob = _archive({"a.bin": b"x" * 400})

        class _Throttles(_LocalZip):
            calls = 0

            def _get(self, url: str, start: int, end: int) -> bytes:
                type(self).calls += 1
                if type(self).calls <= 2:
                    raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)
                return super()._get(url, start, end)

        with unittest.mock.patch("time.sleep"):
            archive = _Throttles(blob)
            self.assertEqual(archive.read(archive.directory()[0]), b"x" * 400)
        self.assertGreater(_Throttles.calls, 2, "it did not retry")

    def test_a_404_is_NOT_retried(self) -> None:
        """A missing member is a real answer. Retrying it five times just wastes half a minute."""
        import urllib.error

        class _Missing(_LocalZip):
            calls = 0

            def _get(self, url: str, start: int, end: int) -> bytes:
                type(self).calls += 1
                raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

        with unittest.mock.patch("time.sleep"), self.assertRaises(urllib.error.HTTPError):
            _Missing(_archive({"a.bin": b"y" * 100})).directory()
        self.assertEqual(_Missing.calls, 1, "a 404 was retried")

    def test_retry_after_is_honoured_over_our_own_curve(self) -> None:
        """A server that says how long to wait has told us something better than a backoff curve."""
        import email.message
        import urllib.error

        headers = email.message.Message()
        headers["Retry-After"] = "7"
        slept: list[float] = []

        class _Says(_LocalZip):
            calls = 0

            def _get(self, url: str, start: int, end: int) -> bytes:
                type(self).calls += 1
                if type(self).calls == 1:
                    raise urllib.error.HTTPError(url, 429, "slow down", headers, None)
                return super()._get(url, start, end)

        with unittest.mock.patch("time.sleep", slept.append):
            archive = _Says(_archive({"a.bin": b"z" * 200}))
            archive.directory()
        self.assertTrue(slept)
        self.assertGreater(max(slept), 3.0, f"the stated 7 s was ignored: {slept}")


class Zip64Tests(unittest.TestCase):
    """⛔⛔ THE POSITIONAL EXTRA FIELD. Only the overflowed values are present, in a fixed order:
    uncompressed size, compressed size, header offset. Which one comes first therefore depends on
    which classic fields hold the sentinel."""

    @staticmethod
    def _extra(*values: int) -> bytes:
        body = b"".join(struct.pack("<Q", v) for v in values)
        return struct.pack("<HH", 0x0001, len(body)) + body

    def _entry(self, compressed: int, uncompressed: int, offset: int, extra: bytes) -> ZipMember:
        name = b"scene.npy"
        blob = (b"PK\x01\x02" + bytes(6) + struct.pack("<H", 8) + bytes(8)
                + struct.pack("<II", compressed, uncompressed)
                + struct.pack("<HHH", len(name), len(extra), 0) + bytes(8)
                + struct.pack("<I", offset) + name + extra)
        return next(iter(_parse_directory(blob)))

    def test_only_the_offset_overflows_which_is_the_common_case(self) -> None:
        """The case a naive reader gets wrong: it takes the first eight bytes as the offset only
        when all three overflowed, and here the first eight bytes ARE the offset."""
        entry = self._entry(1000, 2000, 0xFFFFFFFF, self._extra(9_000_000_000))
        self.assertEqual(entry.header_offset, 9_000_000_000)
        self.assertEqual(entry.compressed, 1000)
        self.assertEqual(entry.uncompressed, 2000)

    def test_sizes_and_offset_all_overflow(self) -> None:
        entry = self._entry(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                            self._extra(8_000_000_001, 8_000_000_002, 8_000_000_003))
        self.assertEqual(entry.uncompressed, 8_000_000_001)
        self.assertEqual(entry.compressed, 8_000_000_002)
        self.assertEqual(entry.header_offset, 8_000_000_003)

    def test_the_uncompressed_size_alone_overflows(self) -> None:
        entry = self._entry(1234, 0xFFFFFFFF, 5678, self._extra(7_000_000_000))
        self.assertEqual(entry.uncompressed, 7_000_000_000)
        self.assertEqual(entry.compressed, 1234)
        self.assertEqual(entry.header_offset, 5678)

    def test_a_missing_extra_field_leaves_the_sentinel_rather_than_inventing_a_value(self) -> None:
        entry = self._entry(1000, 2000, 0xFFFFFFFF, b"")
        self.assertEqual(entry.header_offset, 0xFFFFFFFF)

    def test_the_extra_field_is_found_past_other_fields(self) -> None:
        """Zip64 is rarely the first extra field. A reader that only checks the first one misses it
        whenever a timestamp or a unix-attributes field was written before it."""
        padding = struct.pack("<HH", 0x5455, 5) + b"\x03\x00\x00\x00\x00"
        entry = self._entry(1, 2, 0xFFFFFFFF, padding + self._extra(6_000_000_000))
        self.assertEqual(entry.header_offset, 6_000_000_000)

    def test_a_truncated_extra_field_returns_nothing_rather_than_half_a_number(self) -> None:
        self.assertIsNone(_zip64_values(struct.pack("<HH", 0x0001, 4) + b"\x00" * 4, 1))

    def test_a_real_zip64_archive_round_trips(self) -> None:
        """Built by the standard library with Zip64 forced, so the layout is not one I invented."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as handle:
            for index in range(4):
                with handle.open(f"member_{index}", "w", force_zip64=True) as target:
                    target.write(bytes([index]) * 20_000)
        archive = _LocalZip(buffer.getvalue(), pieces=3)
        members = {m.name: m for m in archive.directory()}
        self.assertEqual(len(members), 4)
        for index in range(4):
            with self.subTest(index):
                self.assertEqual(archive.read(members[f"member_{index}"]),
                                 bytes([index]) * 20_000)


class MemberTests(unittest.TestCase):

    def test_stored_members_are_read(self) -> None:
        archive = _LocalZip(_archive({"plain.bin": b"abc" * 500}, method=zipfile.ZIP_STORED))
        member = archive.directory()[0]
        self.assertEqual(member.method, 0)
        self.assertEqual(archive.read(member), b"abc" * 500)

    def test_the_local_header_lengths_are_read_and_not_assumed(self) -> None:
        """The local header's name and extra lengths may differ from the directory's, so the data
        does not begin at a fixed distance past the header. Assuming otherwise returns shifted bytes
        on the archives where it differs, and nothing raises."""
        source = open("src/robot/grasping/deep/foreign/remote_zip.py",
                      encoding="utf-8").read()
        self.assertIn('struct.unpack_from("<HH", header, 26)', source)

    def test_an_unsupported_compression_method_refuses_by_name(self) -> None:
        archive = _LocalZip(_archive({"a.bin": b"x" * 100}))
        member = archive.directory()[0]
        with self.assertRaises(ValueError) as caught:
            archive.read(ZipMember(member.name, 99, member.compressed, member.uncompressed,
                                   member.header_offset))
        self.assertIn("compression method 99", str(caught.exception))

    def test_a_wrong_header_offset_refuses_rather_than_returning_bytes(self) -> None:
        archive = _LocalZip(_archive({"a.bin": b"x" * 100}))
        member = archive.directory()[0]
        with self.assertRaises(ValueError) as caught:
            archive.read(ZipMember(member.name, member.method, member.compressed,
                                   member.uncompressed, member.header_offset + 7))
        self.assertIn("local file header", str(caught.exception))

    def test_a_directory_whose_count_disagrees_with_the_end_record_refuses(self) -> None:
        """A truncated archive is the failure this catches, and it is the one that would otherwise
        produce a corpus quietly missing its tail."""
        blob = bytearray(_archive({f"m{i}": b"y" * 200 for i in range(5)}))
        end = blob.rfind(b"PK\x05\x06")
        struct.pack_into("<H", blob, end + 10, 9)
        with self.assertRaises(ValueError) as caught:
            _LocalZip(bytes(blob)).directory()
        self.assertIn("claims 9 entries", str(caught.exception))


class CacheTests(unittest.TestCase):

    def test_the_directory_is_written_and_read_back(self) -> None:
        """152 MB of range requests is not a cost to pay twice, and a fetch that re-reads it looks
        to the host like a scraper rather than a reader."""
        import tempfile
        from pathlib import Path

        payloads = {f"m{i}.bin": bytes([i]) * 300 for i in range(6)}
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "nested" / "directory.json"
            first = _LocalZip(_archive(payloads)).directory(cache=cache)
            self.assertTrue(cache.exists())

            class _Refuses(_LocalZip):
                def _get(self, url: str, start: int, end: int) -> bytes:
                    raise AssertionError("the cache should have answered this")

            second = _Refuses(_archive(payloads)).directory(cache=cache)
        self.assertEqual([m.name for m in first], [m.name for m in second])
        self.assertEqual([m.header_offset for m in first], [m.header_offset for m in second])


class ShapeTests(unittest.TestCase):

    def test_parts_and_sizes_must_agree(self) -> None:
        with self.assertRaises(ValueError):
            RemoteZip(["a", "b"], [10])

    def test_an_empty_archive_list_refuses(self) -> None:
        with self.assertRaises(ValueError):
            RemoteZip([], [])

    def test_something_that_is_not_a_zip_refuses(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _LocalZip(b"this is not a zip" * 100).directory()
        self.assertIn("not a zip", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
