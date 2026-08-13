# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.


def build_train_dataset(config):
    backend = getattr(config, "dataset_backend", "lerobot")
    if backend == "lerobot":
        from .lerobot_latent_dataset import MultiLatentLeRobotDataset

        return MultiLatentLeRobotDataset(config=config)
    if backend == "robotwin_hdf5_cache":
        from .robotwin_hdf5_dataset import RoboTwinLatentCacheDataset

        return RoboTwinLatentCacheDataset(config=config)
    raise ValueError(f"Unsupported dataset_backend={backend!r}")


def __getattr__(name):
    if name == "MultiLatentLeRobotDataset":
        from .lerobot_latent_dataset import MultiLatentLeRobotDataset

        return MultiLatentLeRobotDataset
    raise AttributeError(name)


__all__ = ["build_train_dataset", "MultiLatentLeRobotDataset"]
