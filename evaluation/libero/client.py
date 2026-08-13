import argparse
import json
import os
import socket
import time
from pathlib import Path

import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
    WebsocketClientPolicy,
)


MAX_STEPS = 800
EXPECTED_ACTION_SHAPE = (7, 4, 4)
VIDEO_NAMES = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def write_json_atomic(data, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_video(real_obs_list, save_path, fps=15, video_names=VIDEO_NAMES):
    if not real_obs_list:
        raise ValueError("No observation frames to save")

    save_path = Path(save_path)
    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    final_frames = []
    for obs in real_obs_list:
        images = [obs[name] for name in video_names]
        if any(image.shape[:2] != (base_h, width_base) for image in images):
            raise ValueError("Observation cameras have inconsistent frame sizes")
        final_frames.append(np.hstack(images).astype(np.uint8))

    temp_path = save_path.with_name(f"{save_path.stem}.tmp{save_path.suffix}")
    try:
        imageio.mimsave(temp_path, final_frames, fps=fps)
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"Video encoder produced an empty file: {temp_path}")
        os.replace(temp_path, save_path)
    finally:
        temp_path.unlink(missing_ok=True)


def construct_single_env(env_args):
    return OffScreenRenderEnv(**env_args)


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(model, benchmark_instance, task_idx, video_path):
    task = benchmark_instance.get_task(task_idx)
    prompt = task.language
    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = None
    try:
        cur_env = construct_single_env(env_args)
        first_obs = init_single_env(cur_env, init_states[0])
        model.infer(dict(reset=True, prompt=prompt))

        full_obs_list = [first_obs]
        done = False
        first = True
        while cur_env.env.timestep < MAX_STEPS:
            ret = model.infer(dict(obs=first_obs, prompt=prompt))
            action = np.asarray(ret["action"])
            if action.shape != EXPECTED_ACTION_SHAPE:
                raise ValueError(
                    f"Expected action shape {EXPECTED_ACTION_SHAPE}, got {action.shape}"
                )
            if not np.isfinite(action).all():
                raise ValueError("Inference returned a non-finite action")

            key_frame_list = []
            action_per_frame = action.shape[2] // 4
            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    if cur_env.env.timestep >= MAX_STEPS:
                        break
                    observes, done = env_one_step(cur_env, action[:, i, j])
                    if done or (j + 1) % action_per_frame == 0:
                        full_obs_list.append(observes)
                        key_frame_list.append(observes)
                    if done:
                        break
                if done or cur_env.env.timestep >= MAX_STEPS:
                    break

            first = False
            if done or cur_env.env.timestep >= MAX_STEPS:
                break
            model.infer(
                dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action)
            )

        save_video(full_obs_list, video_path, fps=60)
        return int(cur_env.env.timestep) if done else None
    finally:
        if cur_env is not None:
            cur_env.close()


def _is_complete_record(result_path, task_idx, out_dir, suite):
    try:
        record = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(record, dict):
        return False
    required = {
        "classification_id",
        "elapsed_seconds",
        "episode_index",
        "init_state_index",
        "prompt",
        "status",
        "success@520",
        "success@800",
        "success_step",
        "suite",
        "task_index",
        "task_name",
        "video_path",
    }
    video_path = Path(out_dir) / record.get("video_path", "")
    return (
        required <= record.keys()
        and record["status"] == "completed"
        and record["suite"] == suite
        and record["task_index"] == task_idx
        and record["classification_id"] == task_idx + 1
        and video_path.is_file()
        and video_path.stat().st_size > 0
    )


def run(libero_benchmark, port, out_dir, test_num, task_range=None):
    if test_num != 1:
        raise ValueError("This evaluator records exactly one episode per task")

    benchmark_instance = benchmark.get_benchmark_dict()[libero_benchmark]()
    total_tasks = benchmark_instance.get_num_tasks()
    start, end = (0, total_tasks) if task_range is None else task_range
    if not (0 <= start < end <= total_tasks):
        raise ValueError(
            f"Task range [{start}, {end}) is outside [0, {total_tasks})"
        )

    out_dir = Path(out_dir)
    task_indices = range(start, end)
    print(
        f"Using {libero_benchmark}: tasks [{start}, {end}) of {total_tasks}"
    )
    model = None

    for task_idx in tqdm(task_indices, total=len(task_indices)):
        task_dir = out_dir / "tasks" / f"{task_idx:04d}"
        result_path = task_dir / "result.json"
        if _is_complete_record(result_path, task_idx, out_dir, libero_benchmark):
            continue

        task = benchmark_instance.get_task(task_idx)
        video_relative = Path("tasks") / f"{task_idx:04d}" / "episode_0.mp4"
        video_path = out_dir / video_relative
        video_path.parent.mkdir(exist_ok=True, parents=True)
        record = {
            "classification_id": task_idx + 1,
            "elapsed_seconds": None,
            "episode_index": 0,
            "init_state_index": 0,
            "prompt": task.language,
            "status": "error",
            "success@520": False,
            "success@800": False,
            "success_step": None,
            "suite": libero_benchmark,
            "task_index": task_idx,
            "task_name": task.name,
            "video_path": video_relative.as_posix(),
        }
        started = time.monotonic()

        try:
            if model is None:
                with socket.create_connection(("127.0.0.1", port), timeout=5):
                    pass
                model = WebsocketClientPolicy(host="127.0.0.1", port=port)
            success_step = run_one(model, benchmark_instance, task_idx, video_path)
        except Exception as exc:
            record["elapsed_seconds"] = round(time.monotonic() - started, 6)
            record["error"] = {
                "message": str(exc),
                "type": type(exc).__name__,
            }
            write_json_atomic(record, result_path)
            raise

        record.update(
            {
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "status": "completed",
                "success@520": success_step is not None and success_step <= 520,
                "success@800": success_step is not None and success_step <= MAX_STEPS,
                "success_step": success_step,
            }
        )
        write_json_atomic(record, result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs=2,
        default=None,
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=1,
        help="Number of episodes per task (must be 1)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
