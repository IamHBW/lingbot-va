import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


CHECKPOINT_VERSION = 1


def dataset_signature(datasets):
    payload = json.dumps(datasets, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def capture_runtime_state(generator, epoch_generator_state, epoch, batch_offset, device):
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "data_loader_generator_state": generator.get_state(),
        "data_epoch_generator_state": epoch_generator_state,
        "data_epoch": epoch,
        "data_batch_offset": batch_offset,
    }


def restore_runtime_state(state, generator, device):
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    generator.set_state(state["data_loader_generator_state"])


def resume_data_iterator(loader, generator, state):
    generator.set_state(state["data_epoch_generator_state"])
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(state["data_epoch"])
    iterator = iter(loader)
    for _ in range(state["data_batch_offset"]):
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError("checkpoint data batch offset exceeds the saved epoch") from exc
    if not torch.equal(generator.get_state(), state["data_loader_generator_state"]):
        raise ValueError("DataLoader generator state does not match the checkpoint")
    return iterator


def save_optimizer_state(model, optimizer, checkpoint_dir):
    state = {"optimizer": get_optimizer_state_dict(model, optimizer)}
    dcp.save(state, checkpoint_id=str(checkpoint_dir))


def load_optimizer_state(model, optimizer, checkpoint_dir):
    state = {"optimizer": get_optimizer_state_dict(model, optimizer)}
    dcp.load(state, checkpoint_id=str(checkpoint_dir))
    set_optimizer_state_dict(model, optimizer, optim_state_dict=state["optimizer"])


def validate_checkpoint(checkpoint_dir, topology, data_signature_value, rank):
    checkpoint_dir = Path(checkpoint_dir)
    required = [
        checkpoint_dir / "transformer" / "config.json",
        checkpoint_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        checkpoint_dir / "trainer_state.pt",
        checkpoint_dir / "training_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    optimizer_dir = checkpoint_dir / "optimizer"
    if not optimizer_dir.is_dir() or not any(optimizer_dir.iterdir()):
        missing.append(str(optimizer_dir))
    if missing:
        raise ValueError(f"incomplete checkpoint; missing: {', '.join(missing)}")

    trainer_state = torch.load(
        checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    with (checkpoint_dir / "training_config.json").open(encoding="utf-8") as file:
        training_config = json.load(file)

    if trainer_state.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported trainer checkpoint format")
    if training_config.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported training_config checkpoint format")
    if {"global_step", "scheduler_state_dict", "topology", "data_signature"}.difference(
        trainer_state
    ):
        raise ValueError("trainer checkpoint state is incomplete")
    if not isinstance(trainer_state["global_step"], int) or trainer_state["global_step"] < 0:
        raise ValueError("trainer checkpoint global step is invalid")
    if {"config", "dependencies", "datasets", "data_signature"}.difference(
        training_config
    ):
        raise ValueError("training_config checkpoint state is incomplete")
    if checkpoint_dir.name != f"checkpoint_step_{trainer_state.get('global_step')}":
        raise ValueError("checkpoint directory name does not match its global step")

    saved_topology = trainer_state.get("topology", {})
    mismatches = [
        f"{key}: checkpoint={saved_topology.get(key)!r}, current={value!r}"
        for key, value in topology.items()
        if saved_topology.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"checkpoint topology mismatch: {'; '.join(mismatches)}")

    rank_states = []
    for saved_rank in range(topology["world_size"]):
        rank_file = checkpoint_dir / f"rank_{saved_rank}.pt"
        if not rank_file.is_file():
            raise ValueError(f"incomplete checkpoint; missing: {rank_file}")
        rank_state = torch.load(rank_file, map_location="cpu", weights_only=False)
        required_rank_keys = {
            "python_rng_state",
            "numpy_rng_state",
            "torch_rng_state",
            "cuda_rng_state",
            "data_loader_generator_state",
            "data_epoch_generator_state",
            "data_epoch",
            "data_batch_offset",
        }
        if rank_state.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported rank {saved_rank} checkpoint format")
        if required_rank_keys.difference(rank_state):
            raise ValueError(f"rank {saved_rank} checkpoint state is incomplete")
        if (
            rank_state.get("rank") != saved_rank
            or rank_state.get("world_size") != topology["world_size"]
        ):
            raise ValueError(f"rank state does not match rank {saved_rank}")
        if rank_state.get("global_step") != trainer_state.get("global_step"):
            raise ValueError("trainer and rank checkpoint steps differ")
        if rank_state["data_epoch"] < 0 or rank_state["data_batch_offset"] < 0:
            raise ValueError(f"rank {saved_rank} data position is invalid")
        rank_states.append(rank_state)

    positions = {
        (state["data_epoch"], state["data_batch_offset"]) for state in rank_states
    }
    if len(positions) != 1:
        raise ValueError("rank checkpoint data positions differ")
    if trainer_state.get("data_signature") != data_signature_value:
        raise ValueError("checkpoint dataset signature does not match the current dataset")
    if training_config.get("data_signature") != data_signature_value:
        raise ValueError("training_config dataset signature is inconsistent")
    if dataset_signature(training_config["datasets"]) != data_signature_value:
        raise ValueError("training_config dataset manifest is inconsistent")

    return trainer_state, rank_states[rank]
