# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""RoboTwin raw-HDF5 and precomputed-latent datasets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


DEFAULT_CAMERAS = ("head_camera", "left_camera", "right_camera")


def _natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    ]


def _episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected RoboTwin episode filename: {path.name}")
    return int(match.group(1))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stable_pick(items: Sequence[str], seed: str) -> str:
    if not items:
        raise ValueError("Cannot choose from an empty sequence.")
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big") % len(items)
    return str(items[index])


class RoboTwinInstructionResolver:
    """Build deterministic seen instructions from RoboTwin scene metadata."""

    def __init__(
        self,
        description_root: str | Path,
        instruction_type: str = "seen",
    ):
        self.description_root = Path(description_root).expanduser()
        self.task_instruction_dir = self.description_root / "task_instruction"
        self.objects_description_dir = (
            self.description_root / "objects_description"
        )
        self.instruction_type = str(instruction_type)
        if self.instruction_type not in {"seen", "unseen"}:
            raise ValueError(
                "instruction_type must be either 'seen' or 'unseen', "
                f"got {instruction_type!r}"
            )
        if not self.task_instruction_dir.exists():
            raise FileNotFoundError(
                f"RoboTwin task instructions not found: {self.task_instruction_dir}"
            )
        self._task_cache: dict[str, dict[str, Any]] = {}
        self._object_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _extract_placeholders(instruction: str) -> set[str]:
        return set(re.findall(r"{([^}]+)}", instruction))

    @classmethod
    def _filter_templates(
        cls,
        templates: Sequence[str],
        episode_params: dict[str, str],
    ) -> list[str]:
        param_keys = {key.strip("{}") for key in episode_params}
        arm_keys = {
            key
            for key in param_keys
            if len(key) == 1 and "a" <= key <= "z"
        }
        output = []
        for template in templates:
            placeholders = cls._extract_placeholders(str(template))
            if placeholders == param_keys:
                output.append(str(template))
            elif (
                arm_keys
                and placeholders.union(arm_keys) == param_keys
                and not placeholders.intersection(arm_keys)
            ):
                output.append(str(template))
        return output

    def _load_task(self, task_name: str) -> dict[str, Any]:
        if task_name not in self._task_cache:
            path = self.task_instruction_dir / f"{task_name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Task instruction file not found: {path}")
            self._task_cache[task_name] = _load_json(path)
        return self._task_cache[task_name]

    def _replace_placeholders(
        self,
        template: str,
        episode_params: dict[str, str],
        seed: str,
    ) -> str:
        result = str(template)
        for raw_key, raw_value in episode_params.items():
            key = str(raw_key).strip("{}")
            value = str(raw_value)
            object_path = self.objects_description_dir / f"{value}.json"
            if object_path.exists():
                cache_key = str(object_path)
                if cache_key not in self._object_cache:
                    self._object_cache[cache_key] = _load_json(object_path)
                descriptions = self._object_cache[cache_key].get(
                    self.instruction_type, []
                )
                if not descriptions and self.instruction_type == "unseen":
                    descriptions = self._object_cache[cache_key].get("seen", [])
                if not descriptions:
                    raise ValueError(
                        f"No {self.instruction_type!r} descriptions in {object_path}"
                    )
                value = "the " + _stable_pick(
                    [str(item) for item in descriptions],
                    f"{seed}|{key}|{raw_value}",
                )
            elif "/" in value or "\\" in value:
                raise FileNotFoundError(
                    f"Object description file not found: {object_path}"
                )
            elif len(key) == 1 and "a" <= key <= "z":
                value = f"the {value} arm"
            result = result.replace("{" + key + "}", value)
        return result

    def resolve(
        self,
        task_name: str,
        phase: str,
        episode_index: int,
        episode_params: dict[str, str],
    ) -> str:
        task_payload = self._load_task(task_name)
        templates = self._filter_templates(
            task_payload.get(self.instruction_type, []),
            episode_params,
        )
        if not templates:
            raise ValueError(
                "No compatible RoboTwin instruction template for "
                f"{task_name}/{phase}/episode{episode_index}; "
                f"scene keys={sorted(episode_params)}"
            )
        seed = f"{task_name}|{phase}|{episode_index}|{self.instruction_type}"
        template = _stable_pick(templates, seed)
        return self._replace_placeholders(
            template,
            episode_params,
            seed=f"{seed}|{template}",
        )


@dataclass(frozen=True)
class RoboTwinEpisode:
    path: Path
    task: str
    phase: str
    episode_index: int
    length: int
    prompt: str


def decode_robotwin_rgb(encoded: bytes | np.bytes_) -> np.ndarray:
    """Decode a frame while preserving RoboTwin simulator RGB channel order."""

    encoded_array = np.frombuffer(bytes(encoded).rstrip(b"\0"), dtype=np.uint8)
    image = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV failed to decode a RoboTwin RGB frame.")
    # RoboTwin writes simulator RGB arrays directly through cv2.imencode.
    # cv2.imdecode therefore restores the original numeric channel ordering.
    return np.ascontiguousarray(image)


def read_robotwin_eef_actions(h5_file: h5py.File) -> np.ndarray:
    left_pose = np.asarray(h5_file["endpose/left_endpose"], dtype=np.float32)
    left_gripper = np.asarray(
        h5_file["endpose/left_gripper"], dtype=np.float32
    ).reshape(-1, 1)
    right_pose = np.asarray(h5_file["endpose/right_endpose"], dtype=np.float32)
    right_gripper = np.asarray(
        h5_file["endpose/right_gripper"], dtype=np.float32
    ).reshape(-1, 1)
    lengths = {
        len(left_pose),
        len(left_gripper),
        len(right_pose),
        len(right_gripper),
    }
    if len(lengths) != 1:
        raise ValueError(f"RoboTwin action fields have unequal lengths: {lengths}")
    return np.concatenate(
        [left_pose, left_gripper, right_pose, right_gripper],
        axis=1,
    )


def get_relative_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    rotation = R.from_quat(pose[:, 3:7])
    first_rotation = R.from_quat(
        np.repeat(pose[:1, 3:7], repeats=pose.shape[0], axis=0)
    )
    relative_translation = pose[:, :3] - pose[:1, :3]
    relative_quaternion = (first_rotation.inv() * rotation).as_quat()
    return np.concatenate(
        [relative_translation, relative_quaternion],
        axis=1,
    ).astype(np.float32)


def build_lingbot_action_tensors(
    actions: np.ndarray,
    frame_ids: Sequence[int] | np.ndarray | torch.Tensor,
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the official RoboTwin EEF alignment and 30-channel mapping."""

    actions = np.asarray(actions, dtype=np.float32)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise ValueError(f"Expected raw EEF actions [T,16], got {actions.shape}")
    if frame_ids.ndim != 1 or len(frame_ids) < 2:
        raise ValueError(f"Expected at least two sampled frame ids, got {frame_ids}")

    frame_stride = int(frame_ids[1] - frame_ids[0])
    if frame_stride <= 0 or not np.all(np.diff(frame_ids) == frame_stride):
        raise ValueError("RoboTwin frame_ids must have a constant positive stride.")

    actions = actions[int(frame_ids[0]) :]
    left_action = get_relative_pose(actions[:, :7])
    right_action = get_relative_pose(actions[:, 8:15])
    actions = np.concatenate(
        [left_action, actions[:, 7:8], right_action, actions[:, 15:16]],
        axis=1,
    )

    actions_per_latent = frame_stride * 4
    actions = np.pad(
        actions,
        pad_width=((actions_per_latent, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    latent_frame_num = (len(frame_ids) - 1) // 4 + 1
    required_action_num = latent_frame_num * actions_per_latent
    if len(actions) < required_action_num:
        raise ValueError(
            f"Episode has only {len(actions)} aligned actions; "
            f"{required_action_num} are required."
        )
    actions = actions[:required_action_num]
    action_mask = np.ones_like(actions, dtype=bool)

    actions = np.pad(actions, ((0, 0), (0, 1)), mode="constant")
    action_mask = np.pad(action_mask, ((0, 0), (0, 1)), mode="constant")
    inverse_ids = np.asarray(config.inverse_used_action_channel_ids)
    actions = actions[:, inverse_ids]
    action_mask = action_mask[:, inverse_ids]

    q01 = np.asarray(config.norm_stat["q01"], dtype=np.float32)[None]
    q99 = np.asarray(config.norm_stat["q99"], dtype=np.float32)[None]
    actions = (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    actions = np.clip(actions, -1.5, 1.5)
    actions *= action_mask

    actions = actions.reshape(
        latent_frame_num,
        actions_per_latent,
        int(config.action_dim),
    ).transpose(2, 0, 1)
    action_mask = action_mask.reshape(
        latent_frame_num,
        actions_per_latent,
        int(config.action_dim),
    ).transpose(2, 0, 1)
    return (
        torch.from_numpy(actions[..., None]).float(),
        torch.from_numpy(action_mask[..., None]).bool(),
    )


class RoboTwinRawHDF5Dataset(torch.utils.data.Dataset):
    """Episode-level clean RoboTwin RGB observations and 16D EEF actions."""

    def __init__(
        self,
        root_dir: str | Path,
        description_root: str | Path,
        phases: Sequence[str] = ("demo_clean",),
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        task_names: Sequence[str] | None = None,
        frame_stride: int = 4,
        max_episodes_per_task: int | None = None,
        instruction_type: str = "seen",
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.description_root = Path(description_root).expanduser()
        self.phases = tuple(str(phase) for phase in phases)
        self.cameras = tuple(str(camera) for camera in cameras)
        self.task_names = None if task_names is None else set(task_names)
        self.frame_stride = int(frame_stride)
        self.max_episodes_per_task = max_episodes_per_task
        if not self.root_dir.exists():
            raise FileNotFoundError(f"RoboTwin data root not found: {self.root_dir}")
        if self.frame_stride != 4:
            raise ValueError(
                "LingBot-VA RoboTwin action alignment requires frame_stride=4."
            )
        if len(self.cameras) != 3:
            raise ValueError(
                "Official RoboTwin T-shape input requires exactly three cameras."
            )

        resolver = RoboTwinInstructionResolver(
            self.description_root,
            instruction_type=instruction_type,
        )
        self.episodes = self._build_index(resolver)
        if not self.episodes:
            raise FileNotFoundError(
                f"No RoboTwin episodes found under {self.root_dir} "
                f"for phases={self.phases}."
            )

    def _build_index(
        self,
        resolver: RoboTwinInstructionResolver,
    ) -> list[RoboTwinEpisode]:
        episodes = []
        for task_dir in sorted(self.root_dir.iterdir(), key=_natural_key):
            if not task_dir.is_dir() or task_dir.name.startswith("."):
                continue
            if self.task_names is not None and task_dir.name not in self.task_names:
                continue
            for phase in self.phases:
                phase_dir = task_dir / phase
                data_dir = phase_dir / "data"
                scene_info_path = phase_dir / "scene_info.json"
                if not data_dir.exists():
                    continue
                if not scene_info_path.exists():
                    raise FileNotFoundError(
                        f"RoboTwin scene metadata not found: {scene_info_path}"
                    )
                scene_info = _load_json(scene_info_path)
                paths = sorted(data_dir.glob("episode*.hdf5"), key=_natural_key)
                if self.max_episodes_per_task is not None:
                    paths = paths[: int(self.max_episodes_per_task)]
                for path in paths:
                    episode_index = _episode_index(path)
                    episode_payload = scene_info.get(f"episode_{episode_index}", {})
                    params = (
                        episode_payload.get("info", {})
                        if isinstance(episode_payload, dict)
                        else {}
                    )
                    if not isinstance(params, dict):
                        raise ValueError(
                            f"Invalid scene info for {task_dir.name}/{path.name}"
                        )
                    prompt = resolver.resolve(
                        task_name=task_dir.name,
                        phase=phase,
                        episode_index=episode_index,
                        episode_params={
                            str(key): str(value) for key, value in params.items()
                        },
                    )
                    with h5py.File(path, "r") as h5_file:
                        length = self._validate_episode(h5_file, path)
                    episodes.append(
                        RoboTwinEpisode(
                            path=path,
                            task=task_dir.name,
                            phase=phase,
                            episode_index=episode_index,
                            length=length,
                            prompt=prompt,
                        )
                    )
        return episodes

    def _validate_episode(self, h5_file: h5py.File, path: Path) -> int:
        lengths = []
        for camera in self.cameras:
            key = f"observation/{camera}/rgb"
            if key not in h5_file:
                raise KeyError(f"Missing `{key}` in {path}")
            lengths.append(int(h5_file[key].shape[0]))
        for key in (
            "endpose/left_endpose",
            "endpose/left_gripper",
            "endpose/right_endpose",
            "endpose/right_gripper",
        ):
            if key not in h5_file:
                raise KeyError(f"Missing `{key}` in {path}")
            lengths.append(int(h5_file[key].shape[0]))
        if len(set(lengths)) != 1:
            raise ValueError(f"Unequal RGB/action lengths in {path}: {lengths}")
        if lengths[0] < 17:
            raise ValueError(f"RoboTwin episode is too short ({lengths[0]}): {path}")
        return lengths[0]

    def frame_ids_for_episode(self, episode: RoboTwinEpisode) -> np.ndarray:
        # The official cache keeps 16k+1 raw timesteps before 4x RGB sampling.
        last_frame = ((episode.length - 1) // 16) * 16
        frame_ids = np.arange(
            0,
            last_frame + 1,
            self.frame_stride,
            dtype=np.int64,
        )
        if len(frame_ids) % 4 != 1:
            raise AssertionError(
                f"Wan VAE source length must be 4k+1, got {len(frame_ids)}"
            )
        return frame_ids

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        frame_ids = self.frame_ids_for_episode(episode)
        with h5py.File(episode.path, "r") as h5_file:
            observations = {}
            for camera in self.cameras:
                encoded_frames = h5_file[f"observation/{camera}/rgb"][frame_ids]
                frames = np.stack(
                    [decode_robotwin_rgb(encoded) for encoded in encoded_frames],
                    axis=0,
                )
                observations[camera] = (
                    torch.from_numpy(frames)
                    .permute(3, 0, 1, 2)
                    .contiguous()
                )
            actions = torch.from_numpy(read_robotwin_eef_actions(h5_file))
        return {
            "observations": observations,
            "actions": actions,
            "frame_ids": torch.from_numpy(frame_ids),
            "prompt": episode.prompt,
            "task": episode.task,
            "phase": episode.phase,
            "episode_index": episode.episode_index,
            "episode_path": str(episode.path),
            "raw_length": episode.length,
        }


class RoboTwinLatentCacheDataset(torch.utils.data.Dataset):
    """Training dataset for caches produced from RoboTwinRawHDF5Dataset."""

    def __init__(self, config):
        if int(config.batch_size) != 1:
            raise ValueError(
                "RoboTwin episode latents have variable temporal lengths, so "
                "PER_DEVICE_BATCH_SIZE must remain 1. Increase the global "
                "batch with GRADIENT_ACCUMULATION_STEPS."
            )
        self.cache_dir = Path(config.dataset_path).expanduser()
        self.samples_dir = self.cache_dir / "samples"
        if not self.samples_dir.exists():
            raise FileNotFoundError(
                f"RoboTwin latent cache not found: {self.samples_dir}. "
                "Run script/precompute_robotwin_clean_latents.py first."
            )
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"RoboTwin cache manifest not found: {self.manifest_path}"
            )
        manifest_bytes = self.manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        expected_sha256 = getattr(
            config, "robotwin_cache_manifest_sha256", None
        )
        if expected_sha256 and manifest_sha256 != expected_sha256:
            raise ValueError(
                "RoboTwin cache manifest SHA256 mismatch: "
                f"expected={expected_sha256}, actual={manifest_sha256}"
            )
        manifest = json.loads(manifest_bytes)
        actual_cache_paths = sorted(self.samples_dir.glob("*/episode_*.pt"))
        source_root = Path(str(manifest.get("source_root", ""))).expanduser()
        actual_source_paths = sorted(
            source_root.glob("*/demo_clean/data/episode*.hdf5")
        )
        self.sample_records = validate_robotwin_cache_manifest(
            manifest,
            actual_cache_paths=actual_cache_paths,
            actual_source_paths=actual_source_paths,
            cache_dir=self.cache_dir,
        )
        self.sample_paths = [
            self.cache_dir / str(record["cache"])
            for record in self.sample_records
        ]
        self.dataset_manifest = [
            {
                "backend": "robotwin_hdf5_cache",
                "cache_dir": str(self.cache_dir.resolve()),
                "manifest_path": str(self.manifest_path.resolve()),
                "manifest_sha256": manifest_sha256,
                "source_root": str(source_root.resolve()),
                "phase": "demo_clean",
                "instruction_type": "seen",
                "num_tasks": 50,
                "episodes_per_task": 50,
                "length": len(self.sample_records),
            }
        ]
        self.cfg_prob = float(config.cfg_prob)
        empty_emb_path = Path(config.empty_emb_path).expanduser()
        if not empty_emb_path.exists():
            raise FileNotFoundError(
                f"Empty text embedding not found: {empty_emb_path}"
            )
        self.empty_emb = torch.load(
            empty_emb_path,
            map_location="cpu",
            weights_only=False,
        )

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = torch.load(
            self.sample_paths[index],
            map_location="cpu",
            weights_only=False,
        )
        record = self.sample_records[index]
        for key, expected in (
            ("task", record["task"]),
            ("phase", "demo_clean"),
            ("episode_index", record["episode_index"]),
            ("episode_path", record["source"]),
        ):
            if sample.get(key) != expected:
                raise ValueError(
                    f"Cached sample metadata {key!r} mismatch in "
                    f"{self.sample_paths[index]}"
                )
        text_emb = sample["text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb
        output = {
            "latents": sample["latents"],
            "actions": sample["actions"].float(),
            "actions_mask": sample["actions_mask"].bool(),
            "text_emb": text_emb,
        }
        if output["latents"].ndim != 4:
            raise ValueError(
                f"Expected cached latents [C,F,H,W], got "
                f"{tuple(output['latents'].shape)}"
            )
        if output["actions"].ndim != 4:
            raise ValueError(
                f"Expected cached actions [C,F,N,1], got "
                f"{tuple(output['actions'].shape)}"
            )
        if output["actions_mask"].shape != output["actions"].shape:
            raise ValueError(
                "Cached action/action-mask shapes do not match in "
                f"{self.sample_paths[index]}"
            )
        if output["latents"].shape[1] != output["actions"].shape[1]:
            raise ValueError(
                "Cached video/action latent frame counts do not match in "
                f"{self.sample_paths[index]}"
            )
        return output


def validate_robotwin_cache_manifest(
    manifest: dict[str, Any],
    *,
    actual_cache_paths: Sequence[Path] | None = None,
    actual_source_paths: Sequence[Path] | None = None,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate the fixed clean50 cache contract and return sorted records."""

    required = {
        "format_version": 1,
        "dataset_backend": "robotwin_hdf5_cache",
        "phase": "demo_clean",
        "instruction_type": "seen",
        "num_samples": 2500,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Invalid RoboTwin cache manifest fields: {mismatches}")
    if tuple(manifest.get("cameras", ())) != DEFAULT_CAMERAS:
        raise ValueError(
            f"RoboTwin cache cameras must be {DEFAULT_CAMERAS!r}"
        )

    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != 2500:
        raise ValueError("RoboTwin cache manifest must contain 2500 samples")
    source_root = Path(str(manifest.get("source_root", ""))).expanduser()
    tasks: dict[str, set[int]] = {}
    cache_names: set[str] = set()
    source_names: set[str] = set()
    normalized = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("RoboTwin cache sample records must be objects")
        try:
            task = str(record["task"])
            episode_index = int(record["episode_index"])
            cache_name = str(record["cache"])
            source_name = str(record["source"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid RoboTwin cache sample record: {record}") from exc
        if not task or Path(task).name != task or not 0 <= episode_index < 50:
            raise ValueError(f"Invalid RoboTwin task/episode: {task}/{episode_index}")
        expected_cache = f"samples/{task}/episode_{episode_index:06d}.pt"
        expected_source = str(
            source_root
            / task
            / "demo_clean"
            / "data"
            / f"episode{episode_index}.hdf5"
        )
        if cache_name != expected_cache or source_name != expected_source:
            raise ValueError(
                f"RoboTwin cache/source path mismatch for {task}/{episode_index}"
            )
        if cache_name in cache_names or source_name in source_names:
            raise ValueError(f"Duplicate RoboTwin sample: {task}/{episode_index}")
        cache_names.add(cache_name)
        source_names.add(source_name)
        tasks.setdefault(task, set()).add(episode_index)
        normalized.append(record)

    expected_episodes = set(range(50))
    if len(tasks) != 50 or any(value != expected_episodes for value in tasks.values()):
        raise ValueError("RoboTwin cache must cover exactly 50 tasks x 50 episodes")

    if actual_cache_paths is not None:
        if cache_dir is None:
            raise ValueError("cache_dir is required with actual_cache_paths")
        actual = {
            path.relative_to(cache_dir).as_posix() for path in actual_cache_paths
        }
        if actual != cache_names:
            raise ValueError(
                "RoboTwin cache files differ from manifest: "
                f"missing={len(cache_names - actual)}, extra={len(actual - cache_names)}"
            )
    if actual_source_paths is not None:
        actual = {str(path) for path in actual_source_paths}
        if actual != source_names:
            raise ValueError(
                "RoboTwin raw files differ from manifest: "
                f"missing={len(source_names - actual)}, extra={len(actual - source_names)}"
            )

    return sorted(
        normalized,
        key=lambda record: (str(record["task"]), int(record["episode_index"])),
    )
