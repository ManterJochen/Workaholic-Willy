"""Click the corners of the ArUco marker in one frame of the wrist camera, and read where they lie in BASE.

A check of the camera's calibration that moves nothing. Stand the arm still where the camera sees the marker, run this
and click the marker's four corners in turn, round it, then close the window. Each click is one pixel, found in the
robot's base frame from the depth about it, never from one pixel's depth, and placed by the TCP read at the shutter and
the camera's hand-eye calibration. The corners' middle should lie a few millimetres from where the marker lies, about
the calibration's own error, each edge should measure the printed edge, and the surface should face up. A pixel that
cannot be found says why.

No detector and no console: a program of yours finds its point in frame.color itself and calls the same functions;
tcp_above proposes a pose over it and moves nothing. The capture takes the cell's lock and opens the camera, which a
running operator console holds both of: stop the console first. Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/18_find_clicked_pixels_in_base.py
"""

import tempfile
import tkinter
from pathlib import Path

import numpy as np

from willy import Camera, PixelRefused, Robot, capture, load_tree, pixel_to_base, tcp_above

MARKER_MIDDLE_MM = (-130.0, -700.0, 50.0)  # where your marker's middle lies in BASE, as you noted it
MARKER_EDGE_MM = 150.0  # the printed black square's edge

tree = load_tree()
robot = Robot.from_tree(tree, gripper=None)  # the arm alone: no hand is built, so nothing is said to it
with Camera.from_tree(tree) as camera, robot.connected():  # the cell's lock, then the arm; nothing moves
    frame = capture(camera, robot)  # refused while the arm moves
print(frame)

# The colour image in a window of its own, which tkinter reads as PPM; each click is marked where it fell.
picture = Path(tempfile.gettempdir()) / "willy_capture.ppm"
height, width = frame.color.shape[:2]
picture.write_bytes(f"P6 {width} {height} 255\n".encode() + frame.color[..., ::-1].tobytes())
window = tkinter.Tk()
window.title("click the marker's corners in turn, round it, then close this window")
image = tkinter.PhotoImage(file=str(picture))
canvas = tkinter.Canvas(window, width=width, height=height)
canvas.create_image(0, 0, image=image, anchor="nw")
clicks: list[tuple[int, int]] = []


def click(event: "tkinter.Event[tkinter.Canvas]") -> None:
    clicks.append((event.x, event.y))
    canvas.create_oval(event.x - 4, event.y - 4, event.x + 4, event.y + 4, outline="red", width=2)


canvas.bind("<Button-1>", click)
canvas.pack()
window.mainloop()

corners = []
for u, v in clicks:
    try:
        corners.append(pixel_to_base(frame, u, v))
    except PixelRefused as refused:  # outside the image, no depth about it, no one surface: never a silent 0
        print(f"{refused.reason}: {refused}")
        continue
    print(corners[-1])  # the point, its surface's normal, the window's measured share and the plane's spread
if len(corners) == 4:
    middle = np.mean([corner.base_mm for corner in corners], axis=0)
    apart = float(np.linalg.norm(middle - np.asarray(MARKER_MIDDLE_MM)))
    print(f"the corners' middle {np.round(middle, 1).tolist()} mm, {apart:.1f} mm from where the marker lies")
    edges = [float(np.linalg.norm(corners[i].base_mm - corners[i - 1].base_mm)) for i in range(4)]
    print("its edges " + ", ".join(f"{edge:.1f}" for edge in edges) + f" mm; printed {MARKER_EDGE_MM:g} mm")
if corners:  # a TCP pose 50 mm over the first corner, square to its surface: proposed, never driven
    print(tcp_above(corners[0].base_mm, corners[0].normal, 50.0))
