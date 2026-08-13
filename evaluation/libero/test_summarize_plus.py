import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.libero import summarize_plus


class SummarizePlusTest(unittest.TestCase):
    def test_complete_run_and_missing_task(self):
        categories = [f"Category {index}" for index in range(7)]
        classifications = []
        for category_index, category in enumerate(categories):
            for difficulty in range(1, 6):
                task_index = category_index * 5 + difficulty - 1
                classifications.append(
                    {
                        "id": task_index + 1,
                        "name": f"task_{task_index}",
                        "category": category,
                        "difficulty_level": difficulty,
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            classification_path = Path(temp_dir) / "classification.json"
            classification_path.write_text(
                json.dumps({summarize_plus.SUITE: classifications}),
                encoding="utf-8",
            )

            for task_index, classification in enumerate(classifications):
                success_step = (None, 500, 700)[task_index % 3]
                relative_video = (
                    Path("tasks") / f"{task_index:04d}" / "episode_0.mp4"
                )
                task_dir = run_dir / relative_video.parent
                task_dir.mkdir(parents=True)
                (run_dir / relative_video).write_bytes(b"video")
                record = {
                    "classification_id": task_index + 1,
                    "elapsed_seconds": 1.0,
                    "episode_index": 0,
                    "init_state_index": 0,
                    "prompt": f"prompt {task_index}",
                    "status": "completed",
                    "success@520": success_step == 500,
                    "success@800": success_step is not None,
                    "success_step": success_step,
                    "suite": summarize_plus.SUITE,
                    "task_index": task_index,
                    "task_name": classification["name"],
                    "video_path": relative_video.as_posix(),
                }
                (task_dir / "result.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )

            with mock.patch.object(
                summarize_plus, "EXPECTED_TASKS", len(classifications)
            ):
                smoke_dir = run_dir / "smoke" / "tasks" / "0000"
                smoke_dir.mkdir(parents=True)
                (smoke_dir / "result.json").write_text("{}", encoding="utf-8")
                summary = summarize_plus.summarize(run_dir, classification_path)
                self.assertEqual(summary["task_count"], 35)
                self.assertEqual(
                    summary["overall_micro"]["successes"]["success@520"], 12
                )
                self.assertEqual(
                    summary["overall_micro"]["successes"]["success@800"], 23
                )
                self.assertEqual(summary["category_macro"]["category_count"], 7)
                self.assertIn("Category × difficulty", summarize_plus.render_report(summary))

                (run_dir / "tasks" / "0000" / "result.json").unlink()
                with self.assertRaisesRegex(ValueError, "missing task indices: 0"):
                    summarize_plus.summarize(run_dir, classification_path)
