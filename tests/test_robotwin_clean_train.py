import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from wan_va.dataset import build_train_dataset
from wan_va.dataset.robotwin_hdf5_dataset import (
    RoboTwinLatentCacheDataset,
    validate_robotwin_cache_manifest,
)
from wan_va.train import (
    _require_finite_across_ranks,
    resolve_batch_configuration,
)


def clean50_manifest():
    source_root = "/raw"
    samples = []
    for task_index in range(50):
        task = f"task_{task_index:02d}"
        for episode_index in range(50):
            samples.append(
                {
                    "task": task,
                    "episode_index": episode_index,
                    "source": (
                        f"{source_root}/{task}/demo_clean/data/"
                        f"episode{episode_index}.hdf5"
                    ),
                    "cache": (
                        f"samples/{task}/episode_{episode_index:06d}.pt"
                    ),
                }
            )
    return {
        "format_version": 1,
        "dataset_backend": "robotwin_hdf5_cache",
        "source_root": source_root,
        "phase": "demo_clean",
        "instruction_type": "seen",
        "cameras": ["head_camera", "left_camera", "right_camera"],
        "num_samples": 2500,
        "samples": samples,
    }


class RoboTwinCleanTrainTest(unittest.TestCase):
    def test_cache_manifest_contract(self):
        manifest = clean50_manifest()
        records = validate_robotwin_cache_manifest(manifest)
        self.assertEqual(len(records), 2500)

        duplicate = copy.deepcopy(manifest)
        duplicate["samples"][-1] = duplicate["samples"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_robotwin_cache_manifest(duplicate)

        cache_paths = [Path("/cache") / item["cache"] for item in records[:-1]]
        with self.assertRaisesRegex(ValueError, "missing=1"):
            validate_robotwin_cache_manifest(
                manifest,
                actual_cache_paths=cache_paths,
                cache_dir=Path("/cache"),
            )

        with self.assertRaisesRegex(ValueError, "PER_DEVICE_BATCH_SIZE"):
            RoboTwinLatentCacheDataset(
                SimpleNamespace(batch_size=2, dataset_path="/unused")
            )

    def test_dataset_factory(self):
        config = SimpleNamespace(dataset_backend="robotwin_hdf5_cache")
        sentinel = object()
        with mock.patch(
            "wan_va.dataset.robotwin_hdf5_dataset.RoboTwinLatentCacheDataset",
            return_value=sentinel,
        ) as constructor:
            self.assertIs(build_train_dataset(config), sentinel)
            constructor.assert_called_once_with(config=config)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            build_train_dataset(SimpleNamespace(dataset_backend="unknown"))

    def test_global_batch_32_and_conflicts(self):
        self.assertEqual(resolve_batch_configuration(32, 8, 1, 4), (32, 4))
        self.assertEqual(resolve_batch_configuration(32, 32, 1, 1), (32, 1))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_batch_configuration(32, 8, 1, 8)
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            resolve_batch_configuration(30, 8, 1, None)

    def test_non_finite_values_fail(self):
        _require_finite_across_ranks("loss", torch.tensor(1.0))
        with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
            _require_finite_across_ranks("loss", torch.tensor(float("nan")))

        fake_dtensor = mock.Mock()
        fake_dtensor.detach.return_value = fake_dtensor
        fake_dtensor.to_local.return_value = torch.tensor(1.0)
        _require_finite_across_ranks("gradient norm", fake_dtensor)


if __name__ == "__main__":
    unittest.main()
