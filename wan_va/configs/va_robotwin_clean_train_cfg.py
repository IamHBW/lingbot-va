# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


va_robotwin_clean_train_cfg = EasyDict(
    __name__="Config: VA RoboTwin raw-HDF5 clean-only post-training"
)
# Start from the official RoboTwin post-training configuration, then override
# the local data paths and requested run-scale parameters below.
va_robotwin_clean_train_cfg.update(va_robotwin_train_cfg)

va_robotwin_clean_train_cfg.dataset_backend = "robotwin_hdf5_cache"
va_robotwin_clean_train_cfg.raw_dataset_path = os.getenv(
    "ROBOTWIN_ROOT",
    "/mnt/data/public_data/robotwin",
)
va_robotwin_clean_train_cfg.dataset_path = os.getenv(
    "ROBOTWIN_LATENT_CACHE",
    "/mnt/data/users/tianyu/workspace/outputs/.cache/lingbot-va/robotwin-clean-official",
)
va_robotwin_clean_train_cfg.empty_emb_path = os.path.join(
    va_robotwin_clean_train_cfg.dataset_path,
    "empty_emb.pt",
)
va_robotwin_clean_train_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "LINGBOT_VA_BASE",
    "/mnt/data/users/bowen/workspace/ckpt/lingbot-va-base",
)
va_robotwin_clean_train_cfg.robotwin_cache_manifest_sha256 = (
    "c80df58d6490cfe72a8e80875d00cc3859938b9dcbb7fe28655bcb1da4018067"
)

va_robotwin_clean_train_cfg.enable_wandb = _env_bool(
    "ENABLE_WANDB",
    va_robotwin_train_cfg.enable_wandb,
)
va_robotwin_clean_train_cfg.load_worker = int(
    os.getenv("LOAD_WORKER", str(va_robotwin_train_cfg.load_worker))
)
va_robotwin_clean_train_cfg.save_interval = int(
    os.getenv("SAVE_INTERVAL", str(va_robotwin_train_cfg.save_interval))
)

va_robotwin_clean_train_cfg.batch_size = int(
    os.getenv(
        "PER_DEVICE_BATCH_SIZE",
        str(va_robotwin_train_cfg.batch_size),
    )
)
va_robotwin_clean_train_cfg.global_batch_size = int(
    os.getenv("GLOBAL_BATCH_SIZE", "32")
)
va_robotwin_clean_train_cfg.gradient_accumulation_steps = int(
    os.getenv("GRADIENT_ACCUMULATION_STEPS", "1")
)
va_robotwin_clean_train_cfg.num_steps = int(
    os.getenv("NUM_STEPS", "30000")
)
