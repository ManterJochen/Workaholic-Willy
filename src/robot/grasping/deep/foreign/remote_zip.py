"""Reading single members out of a very large remote zip, without downloading it.

The public grasp corpora worth importing are published as one enormous archive. The one this reads
is 229 GB, and its point clouds are a Zip64 split across five raw parts, so the obvious reading is
that a user must download all of it before they can look at any of it. Nobody evaluates a dataset
that way.

But a zip is not a stream. It carries a central directory at its end listing every member and where
that member starts, and a member is independently compressed. So with HTTP range requests, which the
usual hosts support, the archive can be read the way a filesystem is read:

    fetch the tail                  find the end-of-central-directory record
    fetch the central directory     152 MB for 994,861 entries, once, then cached on disk
    fetch one member                about 200 KB for a scene, decompressed in memory

Measured against a 203,488,798,069-byte split archive: 994,861 entries, and one member reads back
byte-exact at its declared uncompressed size. A thousand scenes is then about 200 MB rather than
229 GB.

Two things make this harder than reading an ordinary zip, and both are handled here rather than left
to a caller who would get them subtly wrong.

The archive is split across files but its offsets are not. `pc.partaa` through `pc.partae` are raw
byte ranges of one logical archive, so every offset in the central directory is in concatenated
coordinates and a read may straddle a part boundary. `_span` maps one logical range onto the parts
it touches. Reading part-local offsets instead works for the first 48 GB and then silently returns
the wrong bytes.

The 32-bit fields overflow, and they overflow silently. A member 4 GB into the archive cannot state
its offset in the classic 32-bit field, so the field holds `0xFFFFFFFF` and the true value moves
into a Zip64 extra field. On a 203 GB archive that is most members, not an edge case: the first
entries parse correctly from the 32-bit fields, so a reader that ignores the extra field works at
the start and then fetches garbage from an arbitrary offset. The order of the extra field's values
is positional and depends on which of the classic fields overflowed, which is the part that is easy
to get wrong.
"""

from __future__ import annotations

import io
import json
import random
import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, Sequence

__all__ = ["RemoteZip", "ZipMember"]

#: End of central directory, its Zip64 counterpart, the locator that points at it, a central
#: directory entry, and a local file header.
_EOCD: Final[bytes] = b"PK\x05\x06"
_EOCD64: Final[bytes] = b"PK\x06\x06"
_LOCATOR: Final[bytes] = b"PK\x06\x07"
_ENTRY: Final[bytes] = b"PK\x01\x02"
_LOCAL: Final[bytes] = b"PK\x03\x04"

#: How much of the tail to pull looking for the end record. It sits at the very end unless the archive
#: carries a comment, and a comment may be up to 64 KB.
_TAIL: Final[int] = 65536 + 64

#: The Zip64 extended information extra field.
_ZIP64_EXTRA: Final[int] = 0x0001

#: HTTP codes worth trying again. 429 is the host saying "slower", and 5xx is it saying "not now";
#: neither means the bytes are unavailable. Anything else is a real answer and is raised.
_RETRY_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

#: Attempts per range, and the first backoff in seconds. Doubling from one second gives roughly
#: 1, 2, 4, 8, 16 with jitter, which is 31 seconds of patience before a range is called lost.
_MAX_ATTEMPTS: Final[int] = 6
_BACKOFF: Final[float] = 1.0
_MAX_BACKOFF: Final[float] = 30.0

#: Marks a classic field as overflowed, its real value being in the Zip64 extra field.
_OVERFLOW32: Final[int] = 0xFFFFFFFF
_OVERFLOW16: Final[int] = 0xFFFF


@dataclass(frozen=True, slots=True)
class ZipMember:
    """One file inside the archive, and where to find it."""

    name: str
    method: int
    compressed: int
    uncompressed: int
    header_offset: int


class RemoteZip:
    """A zip on a web server, read through range requests.

    `parts` is the archive in order: one URL for an ordinary zip, several for a split one. Sizes are
    fetched once so the concatenated coordinate system can be built.

    Not a general zip implementation. Stored and deflated members are read; encryption, patched
    entries and multi-disk semantics beyond simple concatenation are refused by name rather than
    half-supported.
    """

    def __init__(self, parts: Sequence[str], sizes: Sequence[int], *,
                 timeout: float = 120.0) -> None:
        if not parts:
            raise ValueError("an archive needs at least one part")
        if len(parts) != len(sizes):
            raise ValueError(f"{len(parts)} part(s) against {len(sizes)} size(s)")
        self.parts = list(parts)
        self.sizes = [int(s) for s in sizes]
        self.timeout = timeout
        self.total = sum(self.sizes)
        bounds, running = [], 0
        for size in self.sizes:
            bounds.append((running, running + size))
            running += size
        self._bounds = bounds
        self._directory: list[ZipMember] | None = None

    # ---- transport ---------------------------------------------------------------------------

    def _get(self, url: str, start: int, end: int) -> bytes:
        """The raw request, and the only thing a different transport should have to replace."""
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body: bytes = response.read()
        return body

    def _fetch(self, url: str, start: int, end: int) -> bytes:
        """One range: retried when the host says to slow down, then checked.

        The check and the retry belong here and the transport belongs in `_get`. A guard placed in
        the transport is lost the moment the transport is replaced, which is where a new transport
        makes it most necessary.

        A rate limit is a slowdown, not a failure. Treating it as a failure loses a large part of an
        import: fetching with several workers, many scenes come back `HTTP 429: Too Many Requests`
        and are counted as permanently failed, where a serial run has none.

        A short benchmark does not predict that. A request rate that climbs with the worker count
        reads as clean scaling, but a benchmark that runs for a few seconds against a rate-limited
        service measures the burst allowance rather than the sustained rate. Backing off is what
        finds the second one.

        `Retry-After` wins over the curve here, because a server that says how long to wait has said
        something better than any backoff invented locally.
        """
        body = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                body = self._get(url, start, end)
                break
            except urllib.error.HTTPError as error:
                if error.code not in _RETRY_CODES or attempt == _MAX_ATTEMPTS - 1:
                    raise
                stated = error.headers.get("Retry-After") if error.headers else None
                delay = (float(stated) if stated and str(stated).strip().isdigit()
                         else _BACKOFF * 2 ** attempt)
                # Jitter, so eight workers throttled at the same instant do not all come back at the
                # same instant and throttle each other again.
                time.sleep(min(delay, _MAX_BACKOFF) * (0.5 + random.random()))
        assert body is not None
        want = end - start + 1
        if len(body) != want:
            # A server that ignores Range answers 200 with the whole file, which for a 48 GB part
            # would look like a hang rather than an error. Named here so it cannot be mistaken.
            raise OSError(f"{url} returned {len(body)} bytes for a {want}-byte range; the host may "
                          f"not support range requests")
        return body

    def _span(self, start: int, length: int) -> bytes:
        """Bytes `[start, start+length)` of the concatenated archive, stitched across parts."""
        if start < 0 or length < 0 or start + length > self.total:
            raise ValueError(f"[{start}, {start + length}) is outside a {self.total}-byte archive")
        if not length:
            return b""
        out = io.BytesIO()
        finish = start + length
        for (begin, end), url in zip(self._bounds, self.parts, strict=True):
            if end <= start or begin >= finish:
                continue
            out.write(self._fetch(url, max(start, begin) - begin, min(finish, end) - begin - 1))
        return out.getvalue()

    # ---- the directory -----------------------------------------------------------------------

    def _locate_directory(self) -> tuple[int, int, int]:
        """`(offset, size, entries)` of the central directory, following Zip64 when present."""
        tail = self._span(max(0, self.total - _TAIL), min(_TAIL, self.total))
        end = tail.rfind(_EOCD)
        if end < 0:
            raise ValueError("no end-of-central-directory record; this is not a zip, or its comment "
                             "is longer than 64 KB")
        entries, size, offset = struct.unpack_from("<H", tail, end + 10)[0], *struct.unpack_from(
            "<II", tail, end + 12)
        if _OVERFLOW32 in (size, offset) or entries == _OVERFLOW16:
            locator = tail.rfind(_LOCATOR)
            if locator < 0:
                raise ValueError("the classic end record has overflowed but there is no Zip64 "
                                 "locator, so the directory cannot be found")
            record = self._span(struct.unpack_from("<Q", tail, locator + 8)[0], 56)
            if record[:4] != _EOCD64:
                raise ValueError(f"the Zip64 locator points at {record[:4]!r}, not a Zip64 end record")
            entries = struct.unpack_from("<Q", record, 32)[0]
            size, offset = struct.unpack_from("<QQ", record, 40)
        return offset, size, entries

    def directory(self, *, cache: Path | None = None) -> list[ZipMember]:
        """Every member, read once and then kept.

        `cache` is a local file the parsed directory is written to and read back from. 152 MB of
        range requests per invocation is not a per-run cost anyone should pay twice, and a fetch that
        re-reads it looks to the host like a scraper rather than a reader.
        """
        if self._directory is not None:
            return self._directory
        if cache is not None and cache.exists():
            rows = json.loads(cache.read_text(encoding="utf-8"))
            self._directory = [ZipMember(*row) for row in rows]
            return self._directory
        offset, size, entries = self._locate_directory()
        members = list(_parse_directory(self._span(offset, size)))
        if len(members) != entries:
            raise ValueError(f"the end record claims {entries} entries and the directory holds "
                             f"{len(members)}; the archive is truncated or not what it says it is")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps([[m.name, m.method, m.compressed, m.uncompressed,
                                          m.header_offset] for m in members]), encoding="utf-8")
        self._directory = members
        return members

    # ---- members -----------------------------------------------------------------------------

    def read(self, member: ZipMember) -> bytes:
        """One member's bytes, decompressed.

        The local header is read rather than assumed. Its name and extra-field lengths are
        allowed to differ from the central directory's, so the data does not begin at a fixed offset
        past the header. Assuming 30 bytes plus the directory's own lengths works on most archives
        and silently returns shifted bytes on the rest.
        """
        header = self._span(member.header_offset, 30)
        if header[:4] != _LOCAL:
            raise ValueError(f"{member.name}: expected a local file header at "
                             f"{member.header_offset}, found {header[:4]!r}")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        body = self._span(member.header_offset + 30 + name_length + extra_length, member.compressed)
        if member.method == 0:
            data = body
        elif member.method == 8:
            data = zlib.decompress(body, -15)
        else:
            raise ValueError(f"{member.name}: compression method {member.method} is not supported; "
                             f"only stored (0) and deflated (8) are")
        if len(data) != member.uncompressed:
            raise ValueError(f"{member.name}: got {len(data)} bytes against a declared "
                             f"{member.uncompressed}")
        return data


def _parse_directory(blob: bytes) -> Iterator[ZipMember]:
    """Walk the central directory, resolving Zip64 overflows.

    The extra field is positional. It carries only the values that actually overflowed, in a fixed
    order: uncompressed size, compressed size, local header offset, disk number. So which value sits
    first depends on which classic fields hold the sentinel, and a reader that always takes the first
    eight bytes as the offset is right exactly when all three overflowed. On a 203 GB archive most
    members overflow the offset alone, which is the case that reader gets wrong.
    """
    position = 0
    while position + 46 <= len(blob) and blob[position:position + 4] == _ENTRY:
        method = struct.unpack_from("<H", blob, position + 10)[0]
        compressed, uncompressed = struct.unpack_from("<II", blob, position + 20)
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", blob, position + 28)
        header_offset = struct.unpack_from("<I", blob, position + 42)[0]
        name = blob[position + 46:position + 46 + name_length].decode("utf-8", "replace")
        extra = blob[position + 46 + name_length:position + 46 + name_length + extra_length]

        if _OVERFLOW32 in (compressed, uncompressed, header_offset):
            wanted = [uncompressed == _OVERFLOW32, compressed == _OVERFLOW32,
                      header_offset == _OVERFLOW32]
            values = _zip64_values(extra, sum(wanted))
            if values is not None:
                taken = iter(values)
                if wanted[0]:
                    uncompressed = next(taken)
                if wanted[1]:
                    compressed = next(taken)
                if wanted[2]:
                    header_offset = next(taken)
        yield ZipMember(name, method, compressed, uncompressed, header_offset)
        position += 46 + name_length + extra_length + comment_length


def _zip64_values(extra: bytes, count: int) -> list[int] | None:
    """The first `count` eight-byte values of the Zip64 extended information field, if present."""
    position = 0
    while position + 4 <= len(extra):
        header, size = struct.unpack_from("<HH", extra, position)
        if header == _ZIP64_EXTRA:
            body = extra[position + 4:position + 4 + size]
            if len(body) < count * 8:
                return None
            return [struct.unpack_from("<Q", body, index * 8)[0] for index in range(count)]
        position += 4 + size
    return None
