"""Willy the Workaholic: the optional operator console.

The only tree in this repo that may import a web framework, and nothing under ``src/`` may import it
back; the library stays a library.

    pip install -r requirements.txt
    python -m api                       # 127.0.0.1:8000, localhost only, no auth

What it is for: preparing a cell, driving a pick by prompt (typed or spoken), watching what the stack
actually did, and showing it to someone. It exposes what the library already computes, the preflight
verdicts, the config with its provenance and the grasp records, rather than re-deriving any of it.

The rule this package is built around: the console issues tasks, not parameters. A pick can be
started; a prompt can be given; a run can be told not to start the next attempt. Nothing that changes
how a motion executes (speed, acceleration, workspace limits, safety toggles) is writable from a
browser, and nothing at all is writable while a run is active. The console is also deliberately
missing an E-stop button: a stop that travels over a socket is an illusion (latency, a closed tab, a
sleeping laptop), and offering one would invite someone to rely on it instead of the physical
mushroom.
"""

__all__ = ["__version__"]

#: Console version, independent of the library's. Bumped when the HTTP surface changes shape.
__version__ = "0.1.0"
