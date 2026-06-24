# inference/loop.y
#   code for run_inference()






def run_inference(robot, policy, policy_cfg, preprocessor, postprocessor, task, fps, run_time_s, dataset):

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()


    # TODO: this is redundant with renaming in load_dataset()
    if USING_XVLA:
        robot_to_policy_key_map = {
            "camera1": "observation.images.image",
            "camera2": "observation.images.image2",
            "camera3": "observation.images.image3",
        }
    else:
        robot_to_policy_key_map = {
            "camera1": "observation.images.camera1",
            "camera2": "observation.images.camera2",
            "camera3": "observation.images.camera3",
        }





    device = get_safe_torch_device(policy_cfg.device)
    logging.info(f"Starting inference loop | task='{task}' | fps={fps} | duration={run_time_s}s")

    start_t = time.perf_counter()
    timestamp = 0.0


    # velocity clamping (attempted smoothing)
    # prev_action: dict | None = None
    # max_joint_vel_deg_per_s = 125.0     # delta defines max movement PER TICK (can be multiple per second)
    # max_delta_deg = max_joint_vel_deg_per_s / FPS    # a possible result is moving 15 (units?) rather than 34 in a single second





    while timestamp < run_time_s:
        loop_start_t = time.perf_counter()

        # --- OBSERVE ---

        t0 = time.perf_counter() # start time

        raw_obs = robot.get_observation()
        t1 = time.perf_counter() # check how long it takes to get the robot state

        obs = robot_observation_processor(raw_obs)
        t2 = time.perf_counter() # check how long observation processing took


        # bypass build_dataset_frame (dataset issues)
        #   the observation batch can be tensors directly from policy_cfg.input_features.
        observation_frame = {}



        # joint state: robot outputs short keys like 'shoulder_pan.pos',
        # policy expects them aggregated under 'observation.state'
        state_keys = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ]

        
        # observation_frame["observation.state"] = state_tensor.to(device) # TODO: purpose? use for triton?

        # state_tensor = torch.tensor(
        #     [obs[k] for k in state_keys], dtype=torch.float32
        # ).unsqueeze(0)  # shape: (1, 6) - batch dim required


        # Camera images - just remap the key, no conversion needed.
        # predict_action calls prepare_observation_for_inference internally,
        # which converts numpy arrays to tensors itself.
        for robot_key, policy_key in robot_to_policy_key_map.items():

            # BELOW IS TESTING TO OPTIMIZE IMAGE PREPROC Y SENDING TO GPU
            # if robot_key in obs and policy_key in policy_cfg.input_features:
            #     img_np = obs[robot_key]                           # (H,W,3) uint8 numpy
            #     buf = _pinned_buffers[policy_key]
            #     buf.copy_(torch.from_numpy(img_np))               # CPU pinned, one copy
            #     img_gpu = buf.cuda(non_blocking=True)             # async DMA, no bounce buffer
            #     img_gpu = img_gpu.permute(2,0,1).float().div_(255.0).unsqueeze(0)
            #     # (1, 3, H, W) float32 on GPU - preprocessor will skip re-transfer
            #     observation_frame[policy_key] = img_gpu


            if robot_key in obs and policy_key in policy_cfg.input_features:
                observation_frame[policy_key] = obs[robot_key]  # raw numpy array, HWC uint8

        
        # Joint state - same, just pass the numpy array
        observation_frame["observation.state"] = np.array(
            [obs[k] for k in state_keys], dtype=np.float32
        )

    

        # --- PREDICT ---
        """
            call predict_action(), will return a base tensor
                expects np.ndarray
                returns torch.Tensor

            example: 
                action_values are tensor([[  2.3541, -17.7406,  46.0053,  58.2234,  26.3015,  15.2118]])
        """




        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy_cfg.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )

        t3 = time.perf_counter() # check how long it takes to prediction the action





        # extract_attention_heatmaps lives in modeling_pi05 and reads the buffer
        # snapshot that sample_actions stored on the model after the denoising loop.
        # We pass the raw camera frames from obs so the overlay is on original resolution.
        # TODO: heatmaps toggle!!
        from lerobot.policies.pi05.modeling_pi05 import extract_attention_heatmaps

        attn_snapshot = getattr(policy.model, "last_attn_buffer_snapshot", None)
        if attn_snapshot:
            # Build raw_camera_frames list in the same order as the model's camera keys
            raw_frames = [
                obs[robot_key]                        # HWC uint8 numpy, original resolution
                for robot_key in ["camera1", "camera2", "camera3"]
                if robot_key in obs
            ]

            heatmaps = extract_attention_heatmaps(
                raw_camera_frames=raw_frames,
                attn_buffer=attn_snapshot,
            )

            if heatmaps is not None:
                composite = np.concatenate(heatmaps, axis=1)          # [H, W*3, 3] BGR
                viz_frames.append(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))  # imageio wants RGB






        # must convert it to "action_processed_policy" via make_robot_action using dataset as well
        """
            should be like: 
                {'shoulder_pan.pos': 2.354058265686035, 
                'shoulder_lift.pos': -17.740571975708008, 
                'elbow_flex.pos': 46.00529479980469, 
                'wrist_flex.pos': 58.223419189453125, 
                'wrist_roll.pos': 26.301464080810547, 
                'gripper.pos': 15.21180248260498}

        """

        action_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)

        # --- SEND ---
        robot_action_to_send = robot_action_processor((action_processed_policy, obs))

        robot.send_action(robot_action_to_send)

        # --- PACE TO FPS ---

        # loop_start_t ─────────────────────────────────────> now
        #       [observe → predict → send]  [sleep]
        #       |←────── dt_s ─────────────|←──────→|
        #       |←──────────── 1/fps (33.3ms) ──────→|

        dt_s = time.perf_counter() - loop_start_t
        sleep_time_s = 1.0 / fps - dt_s
        # if sleep_time_s < 0:
        #     logging.warning(f"Loop running slow: {1/dt_s:.1f} Hz vs target {fps} Hz")
        precise_sleep(max(sleep_time_s, 0.0))






        # matplot:
        timing_history["camera_capture"].append(1000 * (t1 - t0))
        timing_history["obs_processing"].append(1000 * (t2 - t1))
        timing_history["predict_action"].append(1000 * (t3 - t2))

        # remove initial outlier of model being sent to device:
        if len(timing_history["predict_action"]) == 1:
            # set to 0
            timing_history["predict_action"][0] = 0.0

        if len(timing_history["camera_capture"]) % 10 == 0:
            ax.clear()
            iterations = range(len(timing_history["camera_capture"]))
            ax.plot(iterations, timing_history["camera_capture"], label="camera_capture")
            ax.plot(iterations, timing_history["obs_processing"], label="obs_processing")
            ax.plot(iterations, timing_history["predict_action"], label="predict_action")
            ax.legend()
            ax.set_xlabel("iteration")
            ax.set_ylabel("ms")
            ax.set_title("per-iteration timing")
            plt.pause(0.001)  # non-blocking draw - 1ms, won't affect FPS meaningfully


        timestamp = time.perf_counter() - start_t
        

        # check if at rest position every n seconds
        # buffer after minimum seconds so dont end the moment loop starts:
        if timestamp > 20.0 and timestamp % 100.0 < 1.0:
            if check_at_rest_position(obs):
                logging.info("At resting position...")
                break


        # exit()

