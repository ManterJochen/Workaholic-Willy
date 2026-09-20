"""Workaholic-Willy as a library: the public nouns under one name.

    from willy import Pose, Robot, load_tree

    robot = Robot.from_tree(load_tree())          # the cell WILLY_PROFILE names
    with robot.connected():
        print(robot.home())
        print(robot.move(Pose.tool_down(450.0, 100.0, 300.0)))

Every name here is the library's own object, imported on first use: importing ``willy`` loads nothing, and a name
resolves to exactly what its home module defines. The home modules stay where they are and keep their own imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

#: Each public name and the module that defines it, grouped as a program meets them.
_HOME: dict[str, str] = {
    # The config tree of a cell.
    "ConfigError": "src.config",
    "ConfigTree": "src.config",
    "LoadedTree": "src.config",
    "load_speech_section": "src.config",
    "load_tree": "src.config",
    # Poses and frames, in millimetres.
    "Frame": "src.geometry",
    "Pose": "src.geometry",
    # The robot: an arm and its hand, the motion and hand verbs and what they report.
    "Robot": "src.robot.execution.robot",
    "MotionOutcome": "src.robot.execution.motion",
    "MotionReport": "src.robot.execution.motion",
    "HandOutcome": "src.robot.execution.handling",
    "HandReport": "src.robot.execution.handling",
    "HandlingOutcome": "src.robot.execution.handling",
    "HandlingReport": "src.robot.execution.handling",
    "HoldEvidence": "src.robot.core.gripper",
    "JointPositions": "src.robot.core",
    "SafetyPreflight": "src.robot.safety",
    "create_arm": "src.robot.drivers",
    # The whole cell and its pick service.
    "Cell": "src.robot.execution.cell",
    "AutonomousGraspOutcome": "src.robot.execution.autonomous_grasp.report",
    "AutonomousGraspReport": "src.robot.execution.autonomous_grasp.report",
    "AutonomousGraspService": "src.robot.execution.autonomous_grasp.service",
    "GraspMode": "src.robot.execution.autonomous_grasp.config",
    "GraspMotion": "src.robot.grasping.motion.grasp_motion",
    "PickPrompt": "src.robot.execution.autonomous_grasp.prompt",
    "PassRule": "src.robot.execution.pick_run",
    "PickRun": "src.robot.execution.pick_run",
    "PickRunReport": "src.robot.execution.pick_run",
    "PlannerStart": "src.robot.execution.planner_start",
    "RecordLog": "src.robot.grasping.replay.runs",
    "Recording": "src.robot.execution.pick_run",
    # What a build or a connect refuses with.
    "CameraWorldPlan": "src.robot.execution.camera_world_wiring",
    "CameraWorldRequired": "src.robot.execution.camera_world_wiring",
    "CellBusy": "src.robot.execution.cell_lock",
    "CellNotBuilt": "src.robot.execution.cell",
    "LockKeyRequired": "src.robot.execution.robot",
    "NoRealGripper": "src.robot.execution.lifecycle",
    "WristBodyRequired": "src.robot.execution.wrist_bodies",
    # Cameras, calibration and where an object is.
    "Camera": "src.camera",
    "CameraRefused": "src.camera",
    "HandEyeCalibration": "src.robot.execution.hand_eye",
    "Located": "src.robot.perception.locator",
    "Locator": "src.robot.perception.locator",
    "LocatorRefused": "src.robot.perception.locator",
    "RGBDFrame": "src.camera",
    "RigNotCalibrated": "src.camera",
    "SweepOptions": "src.robot.execution.hand_eye",
    # Hands seen by a camera: MediaPipe, optional and standalone.
    "HandGesture": "src.models.handdetection",
    "build_palm_detector": "src.models.handdetection",
    "build_gesture_recognizer": "src.models.handdetection",
    "build_hand_finder_on_camera": "src.models.handdetection",
    # Grasps, and the stacks a desk can evaluate without a robot.
    "MotionStack": "src.robot.safety.planning.stack",
    "PerceptionSpec": "src.models.perception_spec",
    "RuleBasedRouter": "src.models.routing",
    "Scene": "src.robot.grasping.scene",
    "build_calculator": "src.robot.grasping.calculator_factory",
    "preflight_calculator": "src.robot.grasping.calculator_factory",
    "synthesize_suction_grasps": "src.robot.grasping.suction",
    # Speech: push to talk, and a person confirms.
    "Confirmation": "src.models.speech",
    "Listener": "src.models.speech",
    "PushToTalkSource": "src.models.speech",
    "TalkButton": "src.models.speech",
    "TerminalConfirmer": "src.models.speech",
    "shared_speech": "src.models.speech",
    # Isaac Sim, under Isaac's own interpreter: a known-pose pick rate, and one wrist-camera pick filmed.
    "record_demo": "src.willy_sim.run_eih_demo",
    "run_gate": "src.willy_sim.run_m1_pick",
    # Offline: scenes, a dataset from simulation, parts of your own, and a generator trained on them.
    "DatasetBuild": "datagen.api",
    "GeneratorTraining": "src.robot.grasping.deep.train.api",
    "MeshPreparation": "datagen.assets.service",
    "PhysicsSampling": "datagen.grasps.service",
    "PublicCorpus": "src.robot.grasping.deep.foreign.service",
    "PlanOverrides": "src.robot.grasping.deep.train.plan",
    "available_sources": "datagen.assets.service",
    "engine_is_available": "datagen.render.engine",
    "held_out_assets": "datagen.heldout",
    "import_from_directory": "datagen.assets.library",
    "layout_scene": "datagen.scenes",
    "verify_dataset": "datagen.verify",
}

__all__ = sorted(_HOME)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_HOME))


if not TYPE_CHECKING:
    # Defined for the interpreter only: mypy reads the imports below, so a name this door does not export is an error
    # there rather than an ``Any``.
    def __getattr__(name: str) -> Any:
        home = _HOME.get(name)
        if home is None:
            raise AttributeError(f"module 'willy' has no attribute {name!r}")
        value = getattr(import_module(home), name)
        globals()[name] = value
        return value

else:  # pragma: no cover (the names as mypy reads them, each from the module that defines it)
    # Not from `src.config` or `src.models.speech`: those hand their names out through a module `__getattr__`, and
    # mypy reads every name reached that way as Any, so no call through them would be checked.
    from src.config.loader import ConfigError, load_speech_section
    from src.config.tree import ConfigTree, LoadedTree, load_tree
    from src.camera import Camera, CameraRefused, RGBDFrame, RigNotCalibrated
    from src.geometry import Frame, Pose
    from src.models.handdetection import (
        HandGesture, build_gesture_recognizer, build_hand_finder_on_camera, build_palm_detector)
    from src.models.perception_spec import PerceptionSpec
    from src.models.routing import RuleBasedRouter
    from src.models.speech.confirm import Confirmation, TerminalConfirmer
    from src.models.speech.holder import shared_speech
    from src.models.speech.listener import Listener
    from src.models.speech.push_to_talk import PushToTalkSource, TalkButton
    from src.robot.core import JointPositions
    from src.robot.core.gripper import HoldEvidence
    from src.robot.drivers import create_arm
    from src.robot.execution.autonomous_grasp.config import GraspMode
    from src.robot.execution.autonomous_grasp.prompt import PickPrompt
    from src.robot.execution.autonomous_grasp.report import (
        AutonomousGraspOutcome, AutonomousGraspReport)
    from src.robot.execution.autonomous_grasp.service import AutonomousGraspService
    from src.robot.execution.camera_world_wiring import CameraWorldPlan, CameraWorldRequired
    from src.robot.execution.cell import Cell, CellNotBuilt
    from src.robot.execution.cell_lock import CellBusy
    from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions
    from src.robot.execution.handling import HandlingOutcome, HandlingReport, HandOutcome, HandReport
    from src.robot.execution.lifecycle import NoRealGripper
    from src.robot.execution.motion import MotionOutcome, MotionReport
    from src.robot.execution.pick_run import PassRule, PickRun, PickRunReport, Recording
    from src.robot.execution.planner_start import PlannerStart
    from src.robot.execution.robot import LockKeyRequired, Robot
    from src.robot.execution.wrist_bodies import WristBodyRequired
    from src.robot.grasping.calculator_factory import build_calculator, preflight_calculator
    from src.robot.grasping.deep.foreign.service import PublicCorpus
    from src.robot.grasping.deep.train.api import GeneratorTraining
    from src.robot.grasping.deep.train.plan import PlanOverrides
    from src.robot.grasping.motion.grasp_motion import GraspMotion
    from src.robot.grasping.replay.runs import RecordLog
    from src.robot.grasping.scene import Scene
    from src.robot.grasping.suction import synthesize_suction_grasps
    from src.robot.perception.locator import Located, Locator, LocatorRefused
    from src.robot.safety import SafetyPreflight
    from src.robot.safety.planning.stack import MotionStack
    from src.willy_sim.run_eih_demo import record_demo
    from src.willy_sim.run_m1_pick import run_gate
    from datagen.api import DatasetBuild
    from datagen.assets.library import import_from_directory
    from datagen.assets.service import MeshPreparation, available_sources
    from datagen.grasps.service import PhysicsSampling
    from datagen.heldout import held_out_assets
    from datagen.render.engine import engine_is_available
    from datagen.scenes import layout_scene
    from datagen.verify import verify_dataset
