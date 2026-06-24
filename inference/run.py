import gc
import torch
import numpy as np
import logging

from lerobot.utils.utils import init_logging
from lerobot.utils.robot_utils import precise_sleep
from lerobot.robots import make_robot_from_config

from inference.config import InferenceConfig, REST_POSITION, TIMING_PLOT_NAME, timing_history, fig, ax
from inference.model import load_dataset, load_model, load_pipeline
from inference.loop import run_inference
from inference.visualization import save_timing_plot, save_attention_video


def main():

    gc.collect()
    torch.cuda.empty_cache()
    init_logging()

    cfg = InferenceConfig()

    dataset = load_dataset(cfg.dataset_path, rename_map=cfg.xvla_rename_map)
    policy, policy_cfg = load_model(cfg.policy_path, cfg.device, dataset.meta)
    preprocessor, postprocessor = load_pipeline(policy_cfg, dataset.meta)

    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()

    viz_frames = []

    try:
        run_inference(
            robot=robot,
            policy=policy,
            policy_cfg=policy_cfg,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=cfg.task,
            fps=cfg.fps,
            run_time_s=cfg.run_time_s,
            dataset=dataset,
            timing_history=timing_history,
            ax=ax,
            viz_frames=viz_frames,
        )

    finally:
        logging.info("Attempting to return to resting position.")
        rest_start_t = time.perf_counter()
        rest_duration_s = 5.0

        current_obs = robot.get_observation()
        current_pos = {k: current_obs[k] for k in REST_POSITION}

        while time.perf_counter() - rest_start_t < rest_duration_s:
            alpha = (time.perf_counter() - rest_start_t) / rest_duration_s
            alpha = min(alpha, 1.0)
            interpolated = {
                k: current_pos[k] + alpha * (REST_POSITION[k] - current_pos[k])
                for k in REST_POSITION
            }
            robot.send_action(interpolated)
            precise_sleep(1.0 / cfg.fps)

        save_timing_plot(timing_history, fig, ax, TIMING_PLOT_NAME)
        save_attention_video(viz_frames, cfg.fps)
        robot.disconnect()
        logging.info("Done.")


if __name__ == "__main__":
    main()