import json
import tempfile
import unittest
from pathlib import Path

from evaluation.robotwin.eval_protocol import (
    EPISODES_PER_TASK,
    START_SEED,
    TASKS,
    aggregate_phase,
    load_attempt,
    merge_phases,
    topology_for_network_mode,
    validate_topology,
)


def rows(task, phase, count=EPISODES_PER_TASK):
    return [
        {
            "task": task,
            "phase": phase,
            "episode_index": index,
            "seed": START_SEED + index,
            "instruction": f"instruction {index}",
            "success": index % 2 == 0,
            "elapsed_seconds": 1.0,
            "status": "finished",
            "error": None,
        }
        for index in range(count)
    ]


def write_rows(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8")


class RoboTwinEvalProtocolTest(unittest.TestCase):
    def test_topologies_are_explicit(self):
        self.assertEqual(validate_topology("htrain-local"), "htrain-local")
        self.assertEqual(topology_for_network_mode("local"), "htrain-local")
        self.assertEqual(topology_for_network_mode("direct"), "5090-remote")
        self.assertEqual(topology_for_network_mode("relay"), "5090-remote")
        with self.assertRaises(ValueError):
            validate_topology("auto")

    def test_task_set_is_exactly_fifty_unique_tasks(self):
        self.assertEqual(len(TASKS), 50)
        self.assertEqual(len(set(TASKS)), 50)
        self.assertEqual(TASKS[0], "adjust_bottle")
        self.assertEqual(TASKS[-1], "turn_switch")

    def test_attempt_rejects_missing_duplicate_and_old_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            write_rows(path, rows(TASKS[0], "clean")[:-1])
            with self.assertRaises(ValueError):
                load_attempt(path, TASKS[0], "clean")
            duplicate = rows(TASKS[0], "clean")
            duplicate[-1]["episode_index"] = 0
            write_rows(path, duplicate)
            with self.assertRaises(ValueError):
                load_attempt(path, TASKS[0], "clean")
            invalid_seed = rows(TASKS[0], "clean")
            invalid_seed[0]["seed"] = START_SEED - 1
            write_rows(path, invalid_seed)
            with self.assertRaises(ValueError):
                load_attempt(path, TASKS[0], "clean")

    def test_aggregate_and_merge_require_1000_per_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase_roots = {}
            for phase in ("clean", "randomized"):
                phase_root = root / phase
                phase_roots[phase] = phase_root
                for task in TASKS:
                    task_root = phase_root / "tasks" / task
                    write_rows(task_root / "attempt-1" / "results_detailed.jsonl", rows(task, phase))
                    (task_root / "completed_attempt.txt").write_text("1\n", encoding="utf-8")
                self.assertEqual(len(aggregate_phase(phase_root, phase)), 1000)
            merged = merge_phases(phase_roots["clean"], phase_roots["randomized"], root / "merged")
            self.assertEqual(len(merged), 2000)
            summary = json.loads((root / "merged" / "results_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total_finished"], 2000)


if __name__ == "__main__":
    unittest.main()
