# attach out into lerobot/ ????
from inference.config import InferenceConfig, REST_POSITION, timing_history, fig, ax
from inference.model import load_dataset, load_model, load_pipeline, build_robot
from inference.loop import run_inference
from inference.visualization import save_timing_plot, save_attention_video

def main():
    cfg = InferenceConfig()  # change defaults here when needed
    dataset = load_dataset(cfg.dataset_path, rename_map=cfg.xvla_rename_map)
    policy, policy_cfg = load_model(cfg.policy_path, cfg.device, dataset.meta)
    preprocessor, postprocessor = load_pipeline(policy_cfg, dataset.meta)
    robot = build_robot(ROBOT_CONFIG)
    robot.connect()
    try:
        run_inference(robot, policy, policy_cfg, preprocessor, postprocessor, cfg)
    finally:
        save_timing_plot(timing_history, fig, ax)
        save_attention_video(viz_frames)
        robot.disconnect()