import argparse
import csv
import json
from pathlib import Path


TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
TOPOLOGIES = ("htrain-local", "5090-remote")
PHASE_CONFIGS = {"clean": "demo_clean", "randomized": "demo_randomized"}
START_SEED = 100000
EPISODES_PER_TASK = 20


def validate_topology(topology: str) -> str:
    if topology not in TOPOLOGIES:
        raise ValueError(f"ROBOTWIN_EVAL_TOPOLOGY must be one of {TOPOLOGIES}, got {topology!r}")
    return topology


def topology_for_network_mode(network_mode: str) -> str:
    if network_mode == "local":
        return "htrain-local"
    if network_mode in {"direct", "relay"}:
        return "5090-remote"
    raise ValueError(f"Unsupported NETWORK_MODE: {network_mode}")


def load_attempt(path: Path, task: str, phase: str, episodes: int = EPISODES_PER_TASK) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(row.get("task"), row.get("phase"), row.get("episode_index")) for row in rows]
    if len(rows) != episodes or len(set(keys)) != episodes:
        raise ValueError(f"{path}: expected {episodes} unique rows, got rows={len(rows)} unique={len(set(keys))}")
    if {row.get("task") for row in rows} != {task} or {row.get("phase") for row in rows} != {phase}:
        raise ValueError(f"{path}: task/phase mismatch")
    if {row.get("episode_index") for row in rows} != set(range(episodes)):
        raise ValueError(f"{path}: episode indexes must be exactly 0..{episodes - 1}")
    if len({row.get("seed") for row in rows}) != episodes or any(row.get("seed", -1) < START_SEED for row in rows):
        raise ValueError(f"{path}: seeds must be unique integers starting at {START_SEED} or later")
    for row in rows:
        if row.get("status") != "finished" or not isinstance(row.get("success"), bool):
            raise ValueError(f"{path}: every row must be a finished rollout with boolean success")
        if not row.get("instruction") or "elapsed_seconds" not in row or "error" not in row:
            raise ValueError(f"{path}: incomplete result row")
    return sorted(rows, key=lambda row: row["episode_index"])


def _write_outputs(output: Path, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results_detailed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    task_rows = []
    for phase in PHASE_CONFIGS:
        for task in TASKS:
            subset = [row for row in rows if row["task"] == task and row["phase"] == phase]
            if not subset:
                continue
            successes = sum(row["success"] for row in subset)
            task_rows.append(
                {"phase": phase, "task": task, "successes": successes, "finished": len(subset),
                 "success_rate": successes / len(subset)}
            )
    aggregates = {}
    for phase in PHASE_CONFIGS:
        subset = [row for row in rows if row["phase"] == phase]
        if subset:
            successes = sum(row["success"] for row in subset)
            aggregates[phase] = {
                "successes": successes,
                "finished": len(subset),
                "success_rate": successes / len(subset),
            }
    summary = {"aggregates": aggregates, "tasks": task_rows, "total_finished": len(rows)}
    (output / "results_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("phase", "task", "successes", "finished", "success_rate"))
        writer.writeheader()
        writer.writerows(task_rows)


def aggregate_phase(root: Path, phase: str, episodes: int = EPISODES_PER_TASK) -> list[dict]:
    rows = []
    for task in TASKS:
        task_root = root / "tasks" / task
        attempt = int((task_root / "completed_attempt.txt").read_text(encoding="utf-8").strip())
        rows.extend(load_attempt(task_root / f"attempt-{attempt}" / "results_detailed.jsonl", task, phase, episodes))
    expected = len(TASKS) * episodes
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows for {phase}, got {len(rows)}")
    _write_outputs(root, rows)
    return rows


def merge_phases(clean_root: Path, randomized_root: Path, output: Path) -> list[dict]:
    rows = []
    for root, phase in ((clean_root, "clean"), (randomized_root, "randomized")):
        path = root / "results_detailed.jsonl"
        phase_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(phase_rows) != len(TASKS) * EPISODES_PER_TASK or {row["phase"] for row in phase_rows} != {phase}:
            raise ValueError(f"Invalid completed {phase} result set: {path}")
        rows.extend(phase_rows)
    keys = {(row["task"], row["phase"], row["episode_index"]) for row in rows}
    if len(rows) != 2000 or len(keys) != 2000:
        raise ValueError(f"Expected 2000 unique finished rollouts, got rows={len(rows)} unique={len(keys)}")
    _write_outputs(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    attempt = subparsers.add_parser("audit-attempt")
    attempt.add_argument("--path", type=Path, required=True)
    attempt.add_argument("--task", required=True)
    attempt.add_argument("--phase", choices=PHASE_CONFIGS, required=True)
    attempt.add_argument("--episodes", type=int, default=EPISODES_PER_TASK)
    phase = subparsers.add_parser("aggregate-phase")
    phase.add_argument("--root", type=Path, required=True)
    phase.add_argument("--phase", choices=PHASE_CONFIGS, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--clean-root", type=Path, required=True)
    merge.add_argument("--randomized-root", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit-attempt":
        load_attempt(args.path, args.task, args.phase, args.episodes)
    elif args.command == "aggregate-phase":
        aggregate_phase(args.root, args.phase)
    else:
        merge_phases(args.clean_root, args.randomized_root, args.output)


if __name__ == "__main__":
    main()
