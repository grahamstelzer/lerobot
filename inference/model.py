# inference/model.py 
#   code for load_dataset(), load_model(), load_pipeline()
import logging



# ─────────────────────────────────────────────
# 1. LOAD DATASET METADATA
# ─────────────────────────────────────────────
"""
    Fetch metadata from the training dataset without downloading any episode frames.
    LeRobotDataset(repo_id) pulls only the dataset card and stats JSON from HuggingFace.
    What we get back:
    ds_meta.features: shapes + dtypes the model expects (action, observations, etc.)
    ds_meta.stats: per-feature mean/std used for normalization during training
    Both are required downstream:
    make_policy()             needs ds_meta to validate input/output feature shapes
    make_pre_post_processors() needs ds_meta.stats to build the normalization layers
"""
def load_dataset(dataset_path: str, rename_map=None):
    logging.info(f"Loading dataset metadata from: {dataset_path}")
    dataset=LeRobotDataset(dataset_path)
    logging.info(f"Dataset meta loaded:\n{dataset.meta}")

    # xvla uses different naming scheme. if using xvla we need to rename things
    if USING_XVLA:
        changed_features = []

        for feature in dataset.meta.features:
            if feature in rename_map:
                new_key = rename_map[feature]
                temp_feature_dict = {new_key: dict(dataset.meta.features[feature])}
                changed_features.append((feature, new_key))

        for old_key, new_key in changed_features:
            dataset.meta.features[new_key] = dataset.meta.features.pop(old_key)
            logging.info("Renamed feature '%s' -> '%s'", old_key, new_key)  

        print("renamed dataset features:")
        print(dataset.meta.features)


    return dataset


# ─────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────


def load_model(policy_path: str, device: str, ds_meta):
    """
    Load the pretrained VLA policy.


    PreTrainedConfig.from_pretrained() reads the model architecture config
    (e.g. Pi0, ACT, Diffusion) stored alongside the weights on HuggingFace.


    make_policy() instantiates the correct model class, loads weights, and uses
    ds_meta to validate that the model's expected input/output features match
    the dataset the policy was trained on.


    ds_meta comes from load_dataset_meta() - it must be the dataset the policy
    was trained on, not an arbitrary dataset.
    """
    logging.info(f"Loading policy from: {policy_path}")





    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device


    # 
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    policy_cfg.rtc_config = RTCConfig()  # or with custom params


    # "Missing key(s) in state_dict" weight loading errors seen with ds_meta=None,
    # because make_policy uses ds_meta.features to correctly configure the model
    # head dimensions before loading weights.

    policy = make_policy(policy_cfg, ds_meta=ds_meta)

    # NOTE: prints layers and torch.Sizes but states many are empty
    # for name, param in policy.named_parameters():
    #     print(name, param.shape)

    # for name, module in policy.named_modules():
    #     if isinstance(module, type(module)) and not list(module.parameters(recurse=False)):
    #         print(f"No params: {name} ({type(module).__name__})")

    # alternate way to print policy
    # total = sum(p.numel() for p in policy.parameters())
    # total_buf = sum(b.numel() for b in policy.buffers())
    # print(f"Policy: {type(policy).__name__} | params={total:,} | buffers={total_buf:,}")

    # spot-check one weight that should exist
    # for name, buf in policy.named_buffers():
    #     print(name, buf.shape)
    #     break





    policy.eval()   # disable dropout etc. required for deterministic inference





    # ATTEMPTED OPTIMS

    # halve weights to fp16:
    # policy = policy.half()

    # policy = torch.compile(policy, mode="reduce-overhead")
    # mode options:
    #   "default"         - balanced
    #   "reduce-overhead" - best for repeated same-shape inputs (your case)
    #   "max-autotune"    - slowest to compile, fastest at runtime

    policy_cfg.use_amp = USE_AMP

    logging.info("Policy loaded successfully.")
    return policy, policy_cfg




# ─────────────────────────────────────────────
# 3. LOAD PIPELINE (pre/postprocessors)
# ─────────────────────────────────────────────


def load_pipeline(policy_cfg, ds_meta, rename_map=None):

    # NOTE: at this point in time, rename_map is not needed since the dataset features
    #       have been remapped in load_dataset()
    # TODO: double check this does not break anything in the preprocessing

    """
    Build the normalization pipelines that wrap the model.


    Preprocessor:  raw obs dict → normalized tensors the model expects
    Postprocessor: raw model output tensors → denormalized joint targets


    ds_meta.stats contains the per-feature mean/std saved at training time.
    rename_stats() applies RENAME_MAP to those stat keys so they match
    whatever observation key names your robot produces at runtime.


    Mirrors lerobot_record.py lines 484-492.
    """

    rename_map = rename_map if rename_map is not None else {}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),  # lerobot_record.py line 487
        preprocessor_overrides={
            "device_processor":              {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return preprocessor, postprocessor

