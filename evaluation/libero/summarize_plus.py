import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SUITE = "libero_10"
EXPECTED_TASKS = 2519
METRICS = ("success@520", "success@800")


def load_classification(path):
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read classification file {path}: {exc}") from exc
    rows = document.get(SUITE) if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Classification file has no {SUITE} list")
    if len(rows) != EXPECTED_TASKS:
        raise ValueError(
            f"Expected {EXPECTED_TASKS} classifications, found {len(rows)}"
        )

    by_id = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Classification entry {position} is not an object")
        classification_id = row.get("id")
        if type(classification_id) is not int or classification_id in by_id:
            raise ValueError(f"Invalid or duplicate classification id: {classification_id}")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError(f"Classification {classification_id} has no name")
        if not isinstance(row.get("category"), str) or not row["category"]:
            raise ValueError(f"Classification {classification_id} has no category")
        if row.get("difficulty_level") not in range(1, 6):
            raise ValueError(
                f"Classification {classification_id} has invalid difficulty"
            )
        by_id[classification_id] = row

    expected_ids = set(range(1, EXPECTED_TASKS + 1))
    if set(by_id) != expected_ids:
        raise ValueError("Classification ids must be exactly 1..2519")
    return path, by_id


def _format_indices(values):
    values = sorted(values)
    head = ", ".join(map(str, values[:12]))
    return head if len(values) <= 12 else f"{head}, ... (+{len(values) - 12})"


def _validate_record(record, result_path, run_dir, metadata):
    if not isinstance(record, dict):
        raise ValueError("result is not a JSON object")
    task_index = record.get("task_index")
    if type(task_index) is not int or not 0 <= task_index < EXPECTED_TASKS:
        raise ValueError(f"invalid task_index {task_index}")
    classification = metadata[task_index + 1]

    expected_result = Path("tasks") / f"{task_index:04d}" / "result.json"
    if result_path.relative_to(run_dir) != expected_result:
        raise ValueError(
            f"result path must be {expected_result}, got {result_path.relative_to(run_dir)}"
        )

    expected_fields = {
        "status": "completed",
        "suite": SUITE,
        "classification_id": task_index + 1,
        "episode_index": 0,
        "init_state_index": 0,
        "task_name": classification["name"],
    }
    for field, expected in expected_fields.items():
        if record.get(field) != expected:
            raise ValueError(
                f"task {task_index}: {field}={record.get(field)!r}, expected {expected!r}"
            )

    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"task {task_index}: prompt is empty")
    elapsed = record.get("elapsed_seconds")
    if (
        type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError(f"task {task_index}: invalid elapsed_seconds")

    success_step = record.get("success_step")
    if success_step is not None and (
        type(success_step) is not int or not 1 <= success_step <= 800
    ):
        raise ValueError(f"task {task_index}: invalid success_step")

    for metric, limit in zip(METRICS, (520, 800)):
        value = record.get(metric)
        expected = success_step is not None and success_step <= limit
        if type(value) is not bool or value != expected:
            raise ValueError(
                f"task {task_index}: {metric}={value!r}, expected {expected}"
            )

    video_text = record.get("video_path")
    expected_video = Path("tasks") / f"{task_index:04d}" / "episode_0.mp4"
    if not isinstance(video_text, str) or Path(video_text) != expected_video:
        raise ValueError(f"task {task_index}: invalid video_path {video_text!r}")
    video_path = run_dir / expected_video
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise ValueError(f"task {task_index}: video is missing or empty")
    if "error" in record:
        raise ValueError(f"task {task_index}: completed result still contains error")

    merged = dict(record)
    merged["category"] = classification["category"]
    merged["difficulty_level"] = classification["difficulty_level"]
    return merged

def load_results(run_dir, metadata):
    run_dir = Path(run_dir).resolve()
    result_files = sorted((run_dir / "tasks").glob("*/result.json"))
    seen = {}
    records = {}
    issues = []

    for result_path in result_files:
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{result_path}: unreadable JSON ({exc})")
            continue
        if not isinstance(record, dict):
            issues.append(f"{result_path}: result is not an object")
            continue
        task_index = record.get("task_index")
        if type(task_index) is int and task_index in seen:
            issues.append(
                f"duplicate task {task_index}: {seen[task_index]} and {result_path}"
            )
            continue
        if type(task_index) is int:
            seen[task_index] = result_path
        try:
            merged = _validate_record(record, result_path, run_dir, metadata)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        records[task_index] = merged

    expected = set(range(EXPECTED_TASKS))
    found = set(records)
    missing = expected - found
    extra = found - expected
    if missing:
        issues.append(f"missing task indices: {_format_indices(missing)}")
    if extra:
        issues.append(f"unexpected task indices: {_format_indices(extra)}")
    if len(result_files) != EXPECTED_TASKS:
        issues.append(
            f"expected {EXPECTED_TASKS} result files, found {len(result_files)}"
        )
    if issues:
        shown = issues[:20]
        if len(issues) > len(shown):
            shown.append(f"... {len(issues) - len(shown)} more validation errors")
        raise ValueError("Result validation failed:\n- " + "\n- ".join(shown))
    return run_dir, [records[index] for index in range(EXPECTED_TASKS)]

def _stats(records):
    task_count = len(records)
    successes = {
        metric: sum(record[metric] for record in records) for metric in METRICS
    }
    return {
        "task_count": task_count,
        "successes": successes,
        "success_rates": {
            metric: successes[metric] / task_count for metric in METRICS
        },
    }


def summarize(run_dir, classification_path):
    classification_path, metadata = load_classification(classification_path)
    run_dir, records = load_results(run_dir, metadata)
    categories = defaultdict(list)
    difficulties = defaultdict(list)
    category_difficulties = defaultdict(list)
    for record in records:
        category = record["category"]
        difficulty = record["difficulty_level"]
        categories[category].append(record)
        difficulties[difficulty].append(record)
        category_difficulties[(category, difficulty)].append(record)

    if len(categories) != 7 or set(difficulties) != set(range(1, 6)):
        raise ValueError("Expected seven categories and difficulty levels 1..5")
    by_category = {
        category: _stats(categories[category]) for category in sorted(categories)
    }
    by_difficulty = {
        str(difficulty): _stats(difficulties[difficulty])
        for difficulty in sorted(difficulties)
    }
    by_category_and_difficulty = {
        category: {
            str(difficulty): _stats(category_difficulties[(category, difficulty)])
            for difficulty in range(1, 6)
        }
        for category in sorted(categories)
    }
    category_macro = {
        "category_count": len(by_category),
        "success_rates": {
            metric: sum(
                stats["success_rates"][metric] for stats in by_category.values()
            )
            / len(by_category)
            for metric in METRICS
        },
    }

    return {
        "schema_version": 1,
        "suite": SUITE,
        "task_count": len(records),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "classification_file": str(classification_path.resolve()),
        "total_elapsed_seconds": sum(
            record["elapsed_seconds"] for record in records
        ),
        "overall_micro": _stats(records),
        "category_macro": category_macro,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_category_and_difficulty": by_category_and_difficulty,
        "validation": {
            "missing": 0,
            "duplicates": 0,
            "infrastructure_errors": 0,
            "nonempty_videos": len(records),
        },
    }


def _percent(value):
    return f"{value * 100:.2f}%"


def render_report(summary_data):
    overall = summary_data["overall_micro"]
    macro = summary_data["category_macro"]
    lines = [
        "# LingBot-VA × LIBERO-Plus Run Report",
        "",
        f"- Suite: `{summary_data['suite']}`",
        f"- Validated tasks: {summary_data['task_count']}",
        f"- Generated (UTC): {summary_data['generated_at_utc']}",
        "- Upstream 98.5% is background only, not an environment-matched control.",
        "",
        "## Aggregate",
        "",
        "| Scope | Count | success@520 | success@800 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Overall micro | {overall['task_count']} | "
            f"{_percent(overall['success_rates']['success@520'])} | "
            f"{_percent(overall['success_rates']['success@800'])} |"
        ),
        (
            f"| Seven-category macro | {macro['category_count']} categories | "
            f"{_percent(macro['success_rates']['success@520'])} | "
            f"{_percent(macro['success_rates']['success@800'])} |"
        ),
        "",
        "## By category",
        "",
        "| Category | Tasks | success@520 | success@800 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, stats in summary_data["by_category"].items():
        lines.append(
            f"| {category} | {stats['task_count']} | "
            f"{_percent(stats['success_rates']['success@520'])} | "
            f"{_percent(stats['success_rates']['success@800'])} |"
        )

    lines.extend(
        [
            "",
            "## By difficulty",
            "",
            "| Difficulty | Tasks | success@520 | success@800 |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for difficulty, stats in summary_data["by_difficulty"].items():
        lines.append(
            f"| {difficulty} | {stats['task_count']} | "
            f"{_percent(stats['success_rates']['success@520'])} | "
            f"{_percent(stats['success_rates']['success@800'])} |"
        )

    lines.extend(
        [
            "",
            "## Category × difficulty",
            "",
            "Each cell is `success@520 / success@800 (tasks)`.",
            "",
            "| Category | 1 | 2 | 3 | 4 | 5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, difficulties in summary_data["by_category_and_difficulty"].items():
        cells = []
        for difficulty in map(str, range(1, 6)):
            stats = difficulties[difficulty]
            cells.append(
                f"{_percent(stats['success_rates']['success@520'])} / "
                f"{_percent(stats['success_rates']['success@800'])} "
                f"({stats['task_count']})"
            )
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_atomic(path, text):
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Strictly summarize a complete 2519-task LIBERO-Plus run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary_data = summarize(args.run_dir, args.classification)
    except ValueError as exc:
        parser.error(str(exc))

    _write_atomic(
        args.run_dir / "summary.json",
        json.dumps(summary_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_atomic(args.run_dir / "RUN_REPORT.md", render_report(summary_data))
    print(
        f"Validated {summary_data['task_count']} tasks: "
        f"success@520={_percent(summary_data['overall_micro']['success_rates']['success@520'])}, "
        f"success@800={_percent(summary_data['overall_micro']['success_rates']['success@800'])}"
    )


if __name__ == "__main__":
    main()
