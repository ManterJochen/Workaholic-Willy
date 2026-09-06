"""The network: a serialized point-cloud backbone and the heads that read it.

This package re-exports nothing, and that is load-bearing. A package `__init__` executes on every
import of anything beneath it, so a name re-exported here drags its module into every run that
touches any sibling, and a module nothing uses can then no longer be deleted without breaking
imports elsewhere. Consumers import submodules directly.

`net.serialized_backbone` is the backbone, `net.set_generator` the net, `net.slot_head` the K-slot
head, `net.set_loss` and `net.set_targets` the loss and its targets, `net.gripper` the conditioning
vector. A package that re-exports is a package with two addresses per symbol.
"""
