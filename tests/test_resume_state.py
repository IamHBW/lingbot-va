import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wan_va.utils.resume import (
    CHECKPOINT_VERSION,
    capture_runtime_state,
    dataset_signature,
    load_optimizer_state,
    restore_runtime_state,
    resume_data_iterator,
    save_optimizer_state,
    validate_checkpoint,
)


class RandomDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return torch.tensor(
            [index, random.random(), np.random.random(), torch.rand(()).item()],
            dtype=torch.float64,
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _optimizer_state(optimizer):
    return [
        {
            name: value.clone() if torch.is_tensor(value) else value
            for name, value in state.items()
        }
        for state in optimizer.state.values()
    ]


def _assert_optimizer_state_equal(left, right):
    assert len(left) == len(right)
    for left_state, right_state in zip(left, right):
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            if torch.is_tensor(left_state[key]):
                torch.testing.assert_close(left_state[key], right_state[key])
            else:
                assert left_state[key] == right_state[key]


def _write_checkpoint(root, topology, signature):
    checkpoint = root / "checkpoint_step_3"
    (checkpoint / "transformer").mkdir(parents=True)
    (checkpoint / "optimizer").mkdir()
    (checkpoint / "transformer" / "config.json").write_text("{}")
    (checkpoint / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"model")
    (checkpoint / "optimizer" / ".metadata").write_bytes(b"optimizer")
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "global_step": 3,
            "scheduler_state_dict": {},
            "topology": topology,
            "data_signature": signature,
        },
        checkpoint / "trainer_state.pt",
    )
    generator = torch.Generator().manual_seed(42)
    state = capture_runtime_state(
        generator, generator.get_state(), 0, 1, torch.device("cpu")
    )
    state.update(
        format_version=CHECKPOINT_VERSION,
        rank=0,
        world_size=1,
        global_step=3,
    )
    torch.save(state, checkpoint / "rank_0.pt")
    (checkpoint / "training_config.json").write_text(
        json.dumps(
            {
                "format_version": CHECKPOINT_VERSION,
                "config": {},
                "dependencies": {},
                "datasets": [{"repo_id": "toy", "length": 16}],
                "data_signature": signature,
            }
        )
    )
    return checkpoint


class ResumeStateTest(unittest.TestCase):
    def test_next_batch_and_rng_match_after_resume(self):
        for num_workers in (0, 2):
            with self.subTest(num_workers=num_workers):
                seed_everything(7)
                generator = torch.Generator().manual_seed(42)
                loader = DataLoader(
                    RandomDataset(),
                    batch_size=2,
                    shuffle=True,
                    num_workers=num_workers,
                    generator=generator,
                )
                epoch_generator_state = generator.get_state()
                iterator = iter(loader)
                for _ in range(3):
                    next(iterator)
                state = capture_runtime_state(
                    generator,
                    epoch_generator_state,
                    0,
                    3,
                    torch.device("cpu"),
                )

                expected_batch = next(iterator)
                expected_rng = (random.random(), np.random.random(), torch.rand(()))

                resumed_generator = torch.Generator().manual_seed(999)
                resumed_loader = DataLoader(
                    RandomDataset(),
                    batch_size=2,
                    shuffle=True,
                    num_workers=num_workers,
                    generator=resumed_generator,
                )
                resumed_iterator = resume_data_iterator(
                    resumed_loader, resumed_generator, state
                )
                restore_runtime_state(
                    state, resumed_generator, torch.device("cpu")
                )

                torch.testing.assert_close(
                    next(resumed_iterator), expected_batch, rtol=0, atol=0
                )
                actual_rng = (random.random(), np.random.random(), torch.rand(()))
                self.assertEqual(actual_rng[0], expected_rng[0])
                self.assertEqual(actual_rng[1], expected_rng[1])
                torch.testing.assert_close(
                    actual_rng[2], expected_rng[2], rtol=0, atol=0
                )

    def test_model_adamw_scheduler_and_step_match_after_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "optimizer"
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / 3)
            )
            inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)

            for _ in range(2):
                model(inputs).sum().backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            saved_model = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            saved_scheduler = scheduler.state_dict()
            save_optimizer_state(model, optimizer, checkpoint_dir)
            step = 2

            model(inputs).sum().backward()
            optimizer.step()
            scheduler.step()
            expected_model = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            expected_optimizer = _optimizer_state(optimizer)
            expected_lr = scheduler.get_last_lr()

            resumed_model = torch.nn.Linear(3, 2)
            resumed_model.load_state_dict(saved_model)
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.parameters(), lr=0.1
            )
            resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
                resumed_optimizer,
                lr_lambda=lambda value: min(1.0, (value + 1) / 3),
            )
            resumed_scheduler.load_state_dict(saved_scheduler)
            load_optimizer_state(
                resumed_model, resumed_optimizer, checkpoint_dir
            )

            resumed_model(inputs).sum().backward()
            resumed_optimizer.step()
            resumed_scheduler.step()
            step += 1

            for key, value in resumed_model.state_dict().items():
                torch.testing.assert_close(value, expected_model[key])
            _assert_optimizer_state_equal(
                _optimizer_state(resumed_optimizer), expected_optimizer
            )
            self.assertEqual(resumed_scheduler.get_last_lr(), expected_lr)
            self.assertEqual(step, 3)

    def test_checkpoint_validation_rejects_invalid_state(self):
        with tempfile.TemporaryDirectory() as directory:
            topology = {
                "world_size": 1,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "num_workers": 0,
                "prefetch_factor": None,
            }
            signature = dataset_signature([{"repo_id": "toy", "length": 16}])
            checkpoint = _write_checkpoint(Path(directory), topology, signature)
            validate_checkpoint(checkpoint, topology, signature, rank=0)

            for changed in (
                {"world_size": 2},
                {"batch_size": 1},
                {"gradient_accumulation_steps": 2},
                {"num_workers": 2},
                {"prefetch_factor": 2},
            ):
                with self.subTest(changed=changed), self.assertRaisesRegex(
                    ValueError, "topology mismatch"
                ):
                    validate_checkpoint(
                        checkpoint, topology | changed, signature, rank=0
                    )
            with self.assertRaisesRegex(ValueError, "dataset signature"):
                validate_checkpoint(checkpoint, topology, "changed", rank=0)

            (checkpoint / "optimizer" / ".metadata").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete checkpoint"):
                validate_checkpoint(checkpoint, topology, signature, rank=0)
            (checkpoint / "optimizer" / ".metadata").write_bytes(b"optimizer")
            (checkpoint / "rank_0.pt").write_bytes(b"corrupt")
            with self.assertRaises(Exception):
                validate_checkpoint(checkpoint, topology, signature, rank=0)


if __name__ == "__main__":
    unittest.main()
