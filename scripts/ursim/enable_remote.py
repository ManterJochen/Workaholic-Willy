"""Put a URSim controller into Remote Control without a person at the pendant (2026-09-22).

Why this exists. Remote Control is two things, and only the first is a setting: the feature enable
(``/ursim/.polyscope/remotecontrol.properties``, which survives a restart) and the top-right Local/Remote
selector, which boots to Local every time. Neither the dashboard protocol nor RTDE can move that selector,
which is true and was measured on 2026-08-19; the conclusion drawn from it, that only a human at the pendant
can, is not. The pendant is a Java UI on an Xvfb display inside the container, and x11vnc serves it on
127.0.0.1:5900. A pointer event over RFB is the same click, so this script performs it: menu, Settings,
System, Remote Control, Enable, Exit, then the selector.

The oracle is the controller, not the screen. Every run ends by asking the dashboard ``is in remote control``
and exits 1 unless it answers true, because a click that landed on the wrong pixel still looks like a click.
On failure the last framebuffer is written next to the log so a person can see what the pendant showed.

Where it runs. Inside the container, on its own Python 3, because x11vnc and the dashboard both listen on the
container's loopback: ``ursim.sh remote`` copies this file in and runs it. It imports nothing but the standard
library and writes nothing into the repository.

The coordinates are PolyScope 5.26's on the 1280x800 display this image starts. A different image may move
them; the failure then names the step that did not take, and the screenshot shows where the click went.
"""

from __future__ import annotations

import socket
import struct
import time

VNC = ("127.0.0.1", 5900)
DASHBOARD = ("127.0.0.1", 29999)
SHOT = "/tmp/enable_remote_last.png"

# Each step is (label, x, y, seconds to wait afterwards). The safety-configuration dialog of a fresh
# controller sits over the first click and is dismissed by it; on a controller that has no such dialog the
# same point is empty canvas, which is why this sequence is safe to repeat.
STEPS = (
    ("confirm the safety configuration", 639, 706, 1.5),
    ("open the hamburger menu", 1248, 31, 1.5),
    ("choose Settings", 1159, 178, 2.5),
    ("open System", 122, 348, 1.5),
    ("open Remote Control", 148, 406, 1.5),
    ("press Enable", 617, 466, 1.5),
    ("leave Settings", 145, 661, 2.0),
    ("open the local/remote selector", 1116, 32, 1.5),
    ("choose Remote Control", 1053, 88, 2.0),
)


def ask_dashboard(command: str) -> str:
    """One dashboard question, or a sentence saying why there was no answer."""
    try:
        with socket.create_connection(DASHBOARD, timeout=5) as sock:
            sock.recv(4096)
            sock.sendall((command + "\n").encode())
            return sock.recv(4096).decode(errors="replace").strip()
    except OSError as exc:
        return f"<no answer: {type(exc).__name__}: {exc}>"


def _recvall(sock: socket.socket, count: int) -> bytes:
    buf = b""
    while len(buf) < count:
        chunk = sock.recv(count - len(buf))
        if not chunk:
            raise RuntimeError(f"the VNC server closed the connection with {len(buf)} of {count} bytes read")
        buf += chunk
    return buf


class Screen:
    """The pendant's display over RFB: click on it, and photograph it."""

    def __init__(self) -> None:
        self.sock = socket.create_connection(VNC, timeout=30)
        self.sock.settimeout(30)
        _recvall(self.sock, 12)
        self.sock.sendall(b"RFB 003.008\n")
        offered = _recvall(self.sock, 1)[0]
        if offered == 0:
            reason = _recvall(self.sock, struct.unpack(">I", _recvall(self.sock, 4))[0])
            raise RuntimeError(f"the VNC server refused the connection: {reason.decode(errors='replace')}")
        types = _recvall(self.sock, offered)
        if 1 not in types:
            raise RuntimeError(f"the VNC server wants authentication (types {list(types)}), and this script has none")
        self.sock.sendall(bytes([1]))
        if struct.unpack(">I", _recvall(self.sock, 4))[0] != 0:
            raise RuntimeError("the VNC server rejected the None security type")
        self.sock.sendall(bytes([1]))  # ClientInit, shared, so a watching browser is not thrown off
        self.width, self.height = struct.unpack(">HH", _recvall(self.sock, 4))
        _recvall(self.sock, 16)
        _recvall(self.sock, struct.unpack(">I", _recvall(self.sock, 4))[0])
        # 32 bpp true colour, so a pixel is B G R X in memory, and raw encoding only.
        self.sock.sendall(b"\x00\x00\x00\x00" + struct.pack(">BBBBHHHBBBxxx", 32, 24, 0, 1, 255, 255, 255, 16, 8, 0))
        self.sock.sendall(struct.pack(">BxHi", 2, 1, 0))

    def click(self, x: int, y: int) -> None:
        for mask in (0, 1, 0):
            self.sock.sendall(struct.pack(">BBHH", 5, mask, x, y))
            time.sleep(0.25)

    def photograph(self, path: str) -> str:
        """Write the whole framebuffer as a PNG and return the path, for a person to look at."""
        import zlib

        self.sock.sendall(struct.pack(">BBHHHH", 3, 0, 0, 0, self.width, self.height))
        rows = [bytearray(self.width * 4) for _ in range(self.height)]
        deadline, seen = time.time() + 30, 0
        while seen == 0 and time.time() < deadline:
            # Every message is consumed whole, or the next read starts mid-message and the rectangle header
            # that follows is nonsense: the first version of this skipped the type byte of a ServerCutText and
            # then reported "encoding 49", which is the tail of a clipboard string read as a number.
            kind = _recvall(self.sock, 1)[0]
            if kind == 1:  # SetColourMapEntries: 1 pad, first colour, count, then three shorts each
                _recvall(self.sock, 3)
                _recvall(self.sock, 6 * struct.unpack(">H", _recvall(self.sock, 2))[0])
                continue
            if kind == 2:  # Bell
                continue
            if kind == 3:  # ServerCutText: 3 pad, length, text
                _recvall(self.sock, 3)
                _recvall(self.sock, struct.unpack(">I", _recvall(self.sock, 4))[0])
                continue
            if kind != 0:
                raise RuntimeError(f"the server sent message type {kind}, which this script does not read")
            _recvall(self.sock, 1)
            for _ in range(struct.unpack(">H", _recvall(self.sock, 2))[0]):
                x, y, w, h, encoding = struct.unpack(">HHHHi", _recvall(self.sock, 12))
                if encoding != 0:
                    raise RuntimeError(f"the server sent encoding {encoding}, and this script reads raw only")
                data = _recvall(self.sock, w * h * 4)
                for row in range(h):
                    rows[y + row][x * 4:(x + w) * 4] = data[row * w * 4:(row + 1) * w * 4]
                seen += 1
        pixels = bytearray()
        for line in rows:
            pixels += b"\x00"
            rgb = bytearray(self.width * 3)
            rgb[0::3], rgb[1::3], rgb[2::3] = line[2::4], line[1::4], line[0::4]
            pixels += rgb

        def chunk(tag: bytes, payload: bytes) -> bytes:
            return (struct.pack(">I", len(payload)) + tag + payload
                    + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        with open(path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                         + chunk(b"IDAT", zlib.compress(bytes(pixels), 6)) + chunk(b"IEND", b""))
        return path

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def main() -> int:
    before = ask_dashboard("is in remote control")
    print(f"is in remote control: {before}", flush=True)
    if before == "true":
        print("already in remote control: nothing clicked", flush=True)
        return 0
    if before.startswith("<no answer"):
        print("the dashboard did not answer, so PolyScope is not up yet: nothing clicked", flush=True)
        return 1

    screen = Screen()
    print(f"pendant display {screen.width}x{screen.height}", flush=True)
    try:
        # The sequence is run more than once on purpose. A fresh controller opens with the safety
        # confirmation and then the getting-started box, and each one swallows the clicks meant for the menu
        # behind it. Every step is harmless where its target is absent, and the dashboard says when to stop,
        # so passes are cheaper than reading the screen to decide which dialog is up.
        for attempt in (1, 2, 3):
            print(f"pass {attempt}:", flush=True)
            for label, x, y, wait in STEPS:
                print(f"  click ({x:>4},{y:>3})  {label}", flush=True)
                screen.click(x, y)
                time.sleep(wait)
            after = ask_dashboard("is in remote control")
            print(f"is in remote control: {after}", flush=True)
            if after == "true":
                print(f"model: {ask_dashboard('get robot model')} | {ask_dashboard('robotmode')}", flush=True)
                return 0
        print(f"the selector did not move in 3 passes; the pendant as it looks now: {screen.photograph(SHOT)}",
              flush=True)
        return 1
    finally:
        screen.close()


if __name__ == "__main__":
    raise SystemExit(main())
