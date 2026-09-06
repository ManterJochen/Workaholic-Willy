"""Shared CLI arguments for selecting which cell a willy_sim runner boots.

A runner's own flags describe the experiment: how many runs, which prompt, which grasp mode. These
flags describe the cell the experiment runs in: which robot is in it and which optional config
layers are stacked on top. They live here rather than being repeated in each runner that offers
them, so the spelling, the choices and the help text cannot drift apart, and so adding a new robot
or rig layer reaches all of them at once.

Both map onto the loader's profile chain (see ``src/config/loader.py``):

    run_m1_pick                                  ->  WILLY_PROFILE=sim
    run_m1_pick --robot-model ur3e               ->  WILLY_PROFILE=sim,ur3e
    run_m1_pick --robot-model ur3e --profile tiltcam  ->  WILLY_PROFILE=sim,ur3e,tiltcam

Omitting both loads the plain ``sim`` profile, with no extra layer.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.config.schema.robot.sim_schema import UR_MODEL_KEYS

__all__ = ["add_cell_arguments", "cell_profile_kwargs"]


def add_cell_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--robot-model`` + ``--profile`` to a runner's parser."""
    parser.add_argument(
        "--robot-model", type=str, default=None, choices=list(UR_MODEL_KEYS),
        help="which UR this cell drives (default: the config's, a UR5e). Selects the matching config "
             "layer, Lula/Isaac USD, cuRobo config and self-collision kinematics together.",
    )
    parser.add_argument(
        "--radial-closing", action="store_true",
        help="when a silhouette is near-isotropic its principal axis is numerically degenerate, so aim "
             "the jaw along the RADIAL direction (base -> grasp) instead of following the noise. "
             "MEASURED: a UR3e cannot plan a TANGENTIAL top-down close at any azimuth (0/4) while a "
             "radial one plans 4/4; a UR5e absorbs both. Default off -> byte-identical.",
    )
    parser.add_argument(
        "--profile", action="append", default=None, metavar="LAYER",
        help="extra config layer stacked on top, repeatable (e.g. --profile tiltcam for the real "
             "cell's tilted D435 pair instead of the nadir camera).",
    )
    return parser


def cell_profile_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The ``bootstrap_sim_cell`` kwargs implied by the parsed cell arguments."""
    return {
        "robot_model": getattr(args, "robot_model", None),
        "extra_profiles": tuple(getattr(args, "profile", None) or ()),
    }
