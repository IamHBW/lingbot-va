#!/usr/bin/env bash

set -euo pipefail

umask 007

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export NNODES=4
export NGPU=8
export TOTAL_GPUS=$((NNODES * NGPU))
if [[ -n "${PET_NNODES:-}" ]] && [[ "${PET_NNODES}" != "${NNODES}" ]]; then
    echo "Expected PET_NNODES=${NNODES}, got ${PET_NNODES}" >&2
    exit 1
fi
export NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-}}"
if [[ -z "${NODE_RANK}" ]]; then
    if [[ "${HOSTNAME:-}" =~ -master-([0-9]+)$ ]]; then
        NODE_RANK="${BASH_REMATCH[1]}"
    elif [[ "${HOSTNAME:-}" =~ -worker-([0-9]+)$ ]]; then
        NODE_RANK="$((BASH_REMATCH[1] + 1))"
    else
        echo "Cannot infer NODE_RANK from scheduler environment" >&2
        exit 1
    fi
    export NODE_RANK
fi
export MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
if [[ -z "${MASTER_ADDR}" ]]; then
    if [[ "${HOSTNAME:-}" =~ -worker-[0-9]+$ ]]; then
        MASTER_ADDR="${HOSTNAME%-worker-*}-master-0"
    elif [[ "${HOSTNAME:-}" =~ -master-[0-9]+$ ]]; then
        MASTER_ADDR="${HOSTNAME%-master-*}-master-0"
    else
        echo "Cannot infer MASTER_ADDR from scheduler environment" >&2
        exit 1
    fi
    export MASTER_ADDR
fi
export MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29501}}"
if [[ ! "${NODE_RANK}" =~ ^[0-9]+$ ]] || ((NODE_RANK >= NNODES)); then
    echo "Invalid NODE_RANK=${NODE_RANK} for NNODES=${NNODES}" >&2
    exit 1
fi

export TMPDIR="/mnt/data/users/bowen/workspace/tmp/lingbot-va"
mkdir -p "${TMPDIR}"

source /mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh
conda activate lingbot-va
source /mnt/data/users/bowen/workspace/tokens.sh

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_TEAM_NAME="${WANDB_TEAM_NAME:-${WANDB_ENTITY:-}}"
: "${WANDB_API_KEY:?WANDB_API_KEY is missing from bowen/workspace/tokens.sh}"
: "${WANDB_BASE_URL:?WANDB_BASE_URL is missing from bowen/workspace/tokens.sh}"
: "${WANDB_TEAM_NAME:?WANDB_ENTITY is missing from bowen/workspace/tokens.sh}"

export RUN_PRECOMPUTE=0
export GLOBAL_BATCH_SIZE=32
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export LOAD_WORKER="${LOAD_WORKER:-16}"
export ROBOTWIN_ROOT="/mnt/data/public_data/robotwin"
export ROBOTWIN_DESCRIPTION_ROOT="/mnt/data/users/tianyu/workspace/code/RoboTwin/description"
export ROBOTWIN_LATENT_CACHE="/mnt/data/users/tianyu/workspace/outputs/.cache/lingbot-va/robotwin-clean-official"
export LINGBOT_VA_BASE="/mnt/data/users/bowen/workspace/ckpt/lingbot-va-base"
export ENABLE_WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-va_robotwin}"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_SOURCE="/mnt/data/users/tianyu/workspace/checkpoints/lingbot-va-base"
if [[ ! -L "${LINGBOT_VA_BASE}" ]] || [[ "$(readlink -f "${LINGBOT_VA_BASE}")" != "${BASE_SOURCE}" ]]; then
    echo "Invalid base checkpoint link: ${LINGBOT_VA_BASE} must point to ${BASE_SOURCE}" >&2
    exit 1
fi

CACHE_MANIFEST="${ROBOTWIN_LATENT_CACHE}/manifest.json"
EXPECTED_CACHE_SHA256="c80df58d6490cfe72a8e80875d00cc3859938b9dcbb7fe28655bcb1da4018067"
if [[ "$(sha256sum "${CACHE_MANIFEST}" | awk '{print $1}')" != "${EXPECTED_CACHE_SHA256}" ]]; then
    echo "RoboTwin cache manifest SHA256 mismatch" >&2
    exit 1
fi

RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_NAME="${ROBOTWIN_RUN_NAME:-robotwin-gb32-${RUN_STAMP}}"
RUN_ID="${ROBOTWIN_RUN_ID:-robotwin-gb32-${RUN_STAMP}}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/data/users/bowen/workspace/outputs/lingbot-va/${RUN_NAME}}"
export RUN_NAME RUN_ID SAVE_ROOT
export TORCHINDUCTOR_CACHE_DIR="/tmp/bowen/torchinductor/${RUN_ID}/node-${NODE_RANK}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
mkdir -p "${SAVE_ROOT}/logs" "${SAVE_ROOT}/wandb"
export WANDB_DIR="${SAVE_ROOT}/wandb"

echo "RUN_NAME=${RUN_NAME}"
echo "SAVE_ROOT=${SAVE_ROOT}"
echo "TOPOLOGY=${NNODES}x${NGPU} world_size=${TOTAL_GPUS} global_batch=${GLOBAL_BATCH_SIZE} accumulation=${GRADIENT_ACCUMULATION_STEPS} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"

if ((NODE_RANK == 0)) && [[ -z "${HTRAIN_JOB_ID:-}" ]] && command -v submit >/dev/null 2>&1; then
    HTRAIN_JOB_ID="$(submit --status "${HTRAIN_JOB_NAME:-}" 2>/dev/null | awk '/^id:/ {print $2; exit}' || true)"
    export HTRAIN_JOB_ID
fi

if ((NODE_RANK == 0)); then
python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

import torch

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.robotwin_hdf5_dataset import (
    RoboTwinLatentCacheDataset,
    RoboTwinRawHDF5Dataset,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


config = VA_CONFIGS["robotwin_clean_train"]
raw = RoboTwinRawHDF5Dataset(
    config.raw_dataset_path,
    os.environ["ROBOTWIN_DESCRIPTION_ROOT"],
    phases=("demo_clean",),
    instruction_type="seen",
)
counts = Counter(episode.task for episode in raw.episodes)
if len(raw) != 2500 or len(counts) != 50 or set(counts.values()) != {50}:
    raise ValueError(
        f"Raw RoboTwin coverage is not 50x50: tasks={len(counts)}, episodes={len(raw)}"
    )
cache = RoboTwinLatentCacheDataset(config)
cache_index = {
    (record["task"], record["episode_index"]): index
    for index, record in enumerate(cache.sample_records)
}
ordered = sorted(raw.episodes, key=lambda episode: episode.length)
selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
sample_checks = []
for episode in selected:
    sample = cache[cache_index[(episode.task, episode.episode_index)]]
    sample_checks.append(
        {
            "task": episode.task,
            "episode_index": episode.episode_index,
            "raw_length": episode.length,
            "latents": list(sample["latents"].shape),
            "actions": list(sample["actions"].shape),
            "actions_mask": list(sample["actions_mask"].shape),
            "text_emb": list(sample["text_emb"].shape),
        }
    )

repo = Path.cwd()
base_link = Path(os.environ["LINGBOT_VA_BASE"])
base = base_link.resolve(strict=True)
weight_files = sorted(base.rglob("*.safetensors"))
base_weights = [
    {
        "path": str(path.relative_to(base)),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }
    for path in weight_files
]

key_files = [
    repo / "requirements.txt",
    repo / "wan_va/train.py",
    repo / "wan_va/configs/__init__.py",
    repo / "wan_va/configs/va_robotwin_clean_train_cfg.py",
    repo / "wan_va/dataset/__init__.py",
    repo / "wan_va/dataset/robotwin_hdf5_dataset.py",
    repo / "script/run_robotwin_clean_gb32.sh",
    repo / "tests/test_robotwin_clean_train.py",
]
external_repo = Path("/mnt/data/users/tianyu/workspace/code/lingbot-va")
adapter_source = external_repo / "wan_va/dataset/robotwin_hdf5_dataset.py"
precompute = external_repo / "script/precompute_robotwin_clean_latents.py"
packages = (
    "torch",
    "numpy",
    "diffusers",
    "transformers",
    "lerobot",
    "safetensors",
    "h5py",
    "opencv-python",
    "wandb",
)
dependencies = {"python": platform.python_version()}
for package in packages:
    try:
        dependencies[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        dependencies[package] = None

try:
    gpu_info = command(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader",
    ).splitlines()
except (FileNotFoundError, subprocess.CalledProcessError):
    gpu_info = []

diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
manifest = {
    "format_version": 1,
    "created_utc": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat(),
    "experiment": {
        "type": "post-training",
        "dataset": "RoboTwin demo_clean/seen clean50",
        "global_batch_size": 32,
        "world_size": int(os.environ["TOTAL_GPUS"]),
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer_steps": 30000,
        "checkpoint_interval": 1000,
        "precompute_ran": False,
        "evaluation_in_scope": False,
    },
    "code": {
        "repo": str(repo),
        "head": command("git", "rev-parse", "HEAD"),
        "status": command("git", "status", "--short"),
        "tracked_diff": diff.decode("utf-8", errors="replace"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "key_files": {
            str(path.relative_to(repo)): sha256(path) for path in key_files
        },
    },
    "external_adapter": {
        "owner": "tianyu",
        "repo": str(external_repo),
        "source_commit": "4f2aaae17998c1c480438295870b66b890162956",
        "source_file": str(adapter_source),
        "source_sha256": sha256(adapter_source),
        "precompute_file": str(precompute),
        "precompute_sha256": sha256(precompute),
    },
    "data": {
        "raw_root": str(Path(os.environ["ROBOTWIN_ROOT"]).resolve()),
        "description_root": str(
            Path(os.environ["ROBOTWIN_DESCRIPTION_ROOT"]).resolve()
        ),
        "cache_root": str(Path(os.environ["ROBOTWIN_LATENT_CACHE"]).resolve()),
        "cache_manifest": str(Path(os.environ["ROBOTWIN_LATENT_CACHE"]) / "manifest.json"),
        "cache_manifest_sha256": sha256(
            Path(os.environ["ROBOTWIN_LATENT_CACHE"]) / "manifest.json"
        ),
        "tasks": len(counts),
        "episodes": len(raw),
        "episodes_per_task": sorted(set(counts.values())),
        "raw_length_min": ordered[0].length,
        "raw_length_median": ordered[len(ordered) // 2].length,
        "raw_length_max": ordered[-1].length,
        "sample_checks": sample_checks,
    },
    "base_checkpoint": {
        "link": str(base_link),
        "resolved": str(base),
        "total_weight_bytes": sum(item["size"] for item in base_weights),
        "weights": base_weights,
    },
    "config": json_safe(dict(config)),
    "dependencies": dependencies,
    "runtime": {
        "tmpdir": os.environ["TMPDIR"],
        "torchinductor_cache_dir": os.environ["TORCHINDUCTOR_CACHE_DIR"],
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": gpu_info,
    },
    "wandb": {
        "entity": os.environ["WANDB_TEAM_NAME"],
        "project": os.environ["WANDB_PROJECT"],
        "run_name": os.environ["RUN_NAME"],
        "train_id": os.environ["RUN_ID"] + "-train",
        "mode": "online",
    },
    "htrain": {
        "job_name": os.getenv("HTRAIN_JOB_NAME"),
        "job_id": os.getenv("HTRAIN_JOB_ID") or os.getenv("JOB_ID"),
        "project": os.getenv("HTRAIN_PROJECT"),
        "nodes": int(os.environ["NNODES"]),
        "gpus_per_node": int(os.environ["NGPU"]),
    },
}
path = Path(os.environ["SAVE_ROOT"]) / "run_manifest.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
temporary.replace(path)
print(f"Preflight passed; manifest written to {path}")
PY
fi

run_training() {
    local output_root="$1"
    local log_file="$2"
    local command=(
        python -m torch.distributed.run
        --nnodes="${NNODES}"
        --nproc_per_node="${NGPU}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
        --tee=3
        -m wan_va.train
        --config-name=robotwin_clean_train
        --save-root="${output_root}"
    )
    if ((NODE_RANK == 0)); then
        "${command[@]}" 2>&1 | tee "${log_file}"
    else
        "${command[@]}" 2>&1 | tee "${log_file%.log}-node-${NODE_RANK}.log"
    fi
}

verify_checkpoint() {
    local checkpoint="$1"
    python - "${checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

import torch

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import build_train_dataset
from wan_va.modules import load_transformer
from wan_va.utils.resume import dataset_signature, validate_checkpoint

checkpoint = Path(sys.argv[1])
trainer_state = torch.load(
    checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
)
dataset = build_train_dataset(VA_CONFIGS["robotwin_clean_train"])
validate_checkpoint(
    checkpoint,
    trainer_state["topology"],
    dataset_signature(dataset.dataset_manifest),
    rank=0,
)
model = load_transformer(
    str(checkpoint / "transformer"),
    torch_dtype=torch.float32,
    torch_device="cpu",
    attn_mode="flex",
)
del model
print(f"Strict checkpoint load passed: {checkpoint}")
PY
}

export NUM_STEPS=30000
export SAVE_INTERVAL=1000
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_ID="${RUN_ID}-train"
run_training "${SAVE_ROOT}/train" "${SAVE_ROOT}/logs/train.log"
if ((NODE_RANK == 0)); then
verify_checkpoint "${SAVE_ROOT}/train/checkpoints/checkpoint_step_30000" \
    2>&1 | tee "${SAVE_ROOT}/logs/final-checkpoint-load.log"

python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["SAVE_ROOT"]) / "run_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
manifest["status"] = "completed"
manifest["final_checkpoint"] = str(
    Path(os.environ["SAVE_ROOT"])
    / "train/checkpoints/checkpoint_step_30000"
)
path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
fi
