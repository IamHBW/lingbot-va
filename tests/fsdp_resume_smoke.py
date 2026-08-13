"""Run with: PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/fsdp_resume_smoke.py."""

import copy
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from wan_va.utils.resume import (
    capture_runtime_state,
    load_optimizer_state,
    restore_runtime_state,
    resume_data_iterator,
    save_optimizer_state,
)


class RandomDataset(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, index):
        return torch.tensor(
            [index, random.random(), np.random.random(), torch.rand(()).item()],
            dtype=torch.float32,
        )


def make_loader(rank, world_size, generator):
    sampler = DistributedSampler(
        RandomDataset(), world_size, rank, shuffle=True, seed=42
    )
    return DataLoader(
        RandomDataset(),
        batch_size=2,
        sampler=sampler,
        num_workers=2,
        generator=generator,
    )


def train_step(model, optimizer, scheduler, batch, device):
    optimizer.zero_grad()
    model(batch.to(device)).sum().backward()
    optimizer.step()
    scheduler.step()


def full_model_state(model):
    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    random.seed(7 + rank)
    np.random.seed(7 + rank)
    torch.manual_seed(7 + rank)
    torch.cuda.manual_seed(7 + rank)

    paths = [tempfile.mkdtemp(prefix="lingbot-resume-") if rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    checkpoint_dir = Path(paths[0])

    try:
        generator = torch.Generator().manual_seed(42 + rank)
        loader = make_loader(rank, world_size, generator)
        epoch_generator_state = generator.get_state()
        iterator = iter(loader)
        first_batch = next(iterator)

        torch.manual_seed(123)
        model = torch.nn.Linear(4, 2)
        torch.manual_seed(7 + rank)
        model = fully_shard(model.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / 3)
        )
        train_step(model, optimizer, scheduler, first_batch, device)

        runtime_state = capture_runtime_state(
            generator, epoch_generator_state, 0, 1, device
        )
        model_state = full_model_state(model)
        if rank == 0:
            save_file(model_state, checkpoint_dir / "model.safetensors")
        save_optimizer_state(model, optimizer, checkpoint_dir / "optimizer")
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        restore_runtime_state(runtime_state, generator, device)
        dist.barrier()

        expected_batch = next(iterator)
        expected_rng = (
            random.random(),
            np.random.random(),
            torch.rand(()),
            torch.rand((), device=device).cpu(),
        )
        train_step(model, optimizer, scheduler, expected_batch, device)
        expected_state = full_model_state(model)
        expected_lr = scheduler.get_last_lr()

        loaded_state = load_file(checkpoint_dir / "model.safetensors")
        resumed_model = torch.nn.Linear(4, 2)
        resumed_model.load_state_dict(loaded_state)
        resumed_model = fully_shard(resumed_model.to(device))
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.1)
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
            resumed_optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / 3),
        )
        resumed_scheduler.load_state_dict(scheduler_state)
        load_optimizer_state(
            resumed_model, resumed_optimizer, checkpoint_dir / "optimizer"
        )

        resumed_generator = torch.Generator().manual_seed(999 + rank)
        resumed_loader = make_loader(rank, world_size, resumed_generator)
        resumed_iterator = resume_data_iterator(
            resumed_loader, resumed_generator, runtime_state
        )
        restore_runtime_state(runtime_state, resumed_generator, device)
        actual_batch = next(resumed_iterator)
        actual_rng = (
            random.random(),
            np.random.random(),
            torch.rand(()),
            torch.rand((), device=device).cpu(),
        )
        torch.testing.assert_close(actual_batch, expected_batch, rtol=0, atol=0)
        for actual, expected in zip(actual_rng, expected_rng):
            torch.testing.assert_close(
                torch.as_tensor(actual), torch.as_tensor(expected), rtol=0, atol=0
            )

        train_step(
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            actual_batch,
            device,
        )
        actual_state = full_model_state(resumed_model)
        if rank == 0:
            for key, value in actual_state.items():
                torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
        assert resumed_scheduler.get_last_lr() == expected_lr
        dist.barrier()
        if rank == 0:
            print("FSDP2 resume smoke test passed")
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
