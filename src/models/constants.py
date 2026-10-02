"""Shared constants for the models package.

The log directory and the per-model file names live here, so no model
class carries either as a string literal of its own.
"""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: The one directory holding every model class's rotating log file.
#: Resolved relative to the process working directory by
#: :func:`src.utility.log_cfg.create_logger`.
MODELS_LOG_DIR: Final[str] = "logs/models"

#: Per-model log-file names. One file per subsystem, rather than the
#: robot package's single consolidated file, because these subsystems
#: run independently and are read one at a time.
DETECTOR_LOG_FILE: Final[str] = "GroundingDINOObjectDetector.log"
RTDETR_LOG_FILE: Final[str] = "rtdetr_detector.log"
SEGMENTER_LOG_FILE: Final[str] = "sam2_segmenter.log"
ONEFORMER_LOG_FILE: Final[str] = "oneformer_segmenter.log"
HAND_FINDER_LOG_FILE: Final[str] = "hand_finder.log"
WHISPER_LOG_FILE: Final[str] = "whisper_model.log"

#: The perception seam (:class:`~src.models.perception_backend.TwoStageBackend`). Its own
#: file because it is the only place that sees detector and segmenter together: a run where every
#: detection lost its mask is invisible in either model's log and obvious in this one.
PERCEPTION_BACKEND_LOG_FILE: Final[str] = "perception_backend.log"

#: The VLM grounding parser. Separate from the model wrapper's log: these lines record which boxes the
#: parser dropped and why, a different question from what the model said.
VLM_PARSING_LOG_FILE: Final[str] = "vlm_parsing.log"

#: RT-DETR fine-tuning. An offline, long-running job rather than a runtime component, so mixing it
#: into the detector's log would bury a pick-time inference line under a training run's history.
RTDETR_TRAIN_LOG_FILE: Final[str] = "rtdetr_train.log"

# ----------------------------------------------------------------------
# The VLM route + the routing seam
# ----------------------------------------------------------------------
# A bare ``logging.getLogger(__name__)`` formats its lines and then drops them, because no CLI in this
# repo calls ``basicConfig``. The four constants below give those modules the same rotating file every
# other model writes to, leaving their call sites where they are.

#: The grounding VLM wrapper (``vlm/qwen.py``): which checkpoint loaded, what it cost, and how many
#: boxes each answer produced. Separate from the parser's file below, because "the model said nothing"
#: and "the model said something this parser could not use" are different diagnoses.
VLM_GROUNDER_LOG_FILE: Final[str] = "vlm_grounder.log"

#: The refuse-vs-degrade guard (``vlm/availability.py``). Its own file because a degraded run is a fact
#: about the whole cell rather than a detail of one model: every pick after that warning is grounded by
#: a detector that returns a confident box for the wrong object.
VLM_AVAILABILITY_LOG_FILE: Final[str] = "vlm_availability.log"

#: Which route each prompt took (``routed_backend.py``). Read to answer "why was that pick slow": the
#: answer is usually a word in the prompt that sent it to the VLM.
PERCEPTION_ROUTING_LOG_FILE: Final[str] = "perception_routing.log"

#: The process's one VLM copy (``vlm/holder.py``): which weights are held, when a person or a command loaded them
#: and what a failed load said. Its own file because the copy outlives any one cell build, and "why does the card
#: say the VLM is not loaded" is answered by the holder's history, not by a model's.
VLM_HOLDER_LOG_FILE: Final[str] = "vlm_holder.log"

#: The command reader (``vlm/command.py``): each sentence, what the model answered, the one corrective retry and
#: why it was asked. Separate from the grounder's file: a wrong card is a reading problem, a wrong box a seeing one.
VLM_COMMAND_LOG_FILE: Final[str] = "vlm_command.log"

#: What :func:`~src.models.factory.build_perception` assembled. The models mirror of the robot
#: package's ``grasp_builders.log``: config alone does not say what a cell is running, because presets
#: and legacy keys both feed into the result.
MODELS_FACTORY_LOG_FILE: Final[str] = "models_factory.log"

# ``_inference.py`` has no entry here and keeps its bare ``getLogger``. It is a shared load-time
# helper, so its three warnings describe whichever model called it, and the datagen evaluator imports
# it inside ``ProcessPoolExecutor`` workers, where a module-scope handler means one rotating file
# opened per worker process on a single path. ``log_cfg`` serialises writers within a process, not
# across processes.
