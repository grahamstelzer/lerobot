# inference/config.py
#   code for InferenceConfig, REST_POSITION
#   TODO: moved here for organization but ROBOT_CONFIG should probably be somewhere else

import time
import logging
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class InferenceConfig:
    policy_path: str  = "grahamwichhh/pi05_30k"
    dataset_path: str = "grahamwichhh/v5_pick-up-cube"
    task: str         = "Pick up the yellow cube."
    device: str       = "cuda"
    fps: int          = 30
    run_time_s: int   = 120
    using_xvla: bool  = False # xvla has annoying renaming scheme for their datasets
    use_autocast: bool = False
    use_amp: bool      = False

    camera_video_1: Path = Path("/dev/video0")
    camera_video_2: Path = Path("/dev/video2")
    camera_video_3: Path = Path("/dev/video4")

    @property
    def xvla_rename_map(self) -> dict:

        if not self.using_xvla:
            return {}

        return {
            "observation.images.camera1": "observation.images.image",
            "observation.images.camera2": "observation.images.image2",
            "observation.images.camera3": "observation.images.image3",
        }

# used for so101 ending (prevents abrupt stop)
REST_POSITION = {
    "shoulder_pan.pos":  1.4,
    "shoulder_lift.pos": -99.0,
    "elbow_flex.pos":    97.0,
    "wrist_flex.pos":    72.0,
    "wrist_roll.pos":    -3.0,
    "gripper.pos":       3.2,
}



# runtime state for timing plots
TIMING_PLOT_NAME = f"testing_{time.strftime('%Y%m%d_%H%M%S')}"
timing_history = {
    "camera_capture": [],
    "obs_processing": [],
    "predict_action": [],
}
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("iteration")
ax.set_ylabel("ms")
ax.set_title(TIMING_PLOT_NAME)