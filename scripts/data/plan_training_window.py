#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


STEP_RE = re.compile(r"^step(?P<step>[0-9]+)$")
WINDOW_RE = re.compile(r"^(?P<start>step[0-9]+)(?:->|:|-|_to_)(?P<end>step[0-9]+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "window_id",
        "from_revision",
        "to_revision",
        "batch_idx_start",
        "batch_idx_end_exclusive",
        "parquet_files",
        "sample_size",
        "sample_seed",
        "priority",
        "split",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "window_id": row["window_id"],
                    "from_revision": row["from_revision"],
                    "to_revision": row["to_revision"],
                    "batch_idx_start": row["batch_idx_start"],
                    "batch_idx_end_exclusive": row["batch_idx_end_exclusive"],
                    "parquet_files": ";".join(row["parquet_files"]),
                    "sample_size": row["sample_policy"]["sample_size"],
                    "sample_seed": row["sample_policy"]["seed"],
                    "priority": row.get("priority", ""),
                    "split": row.get("split", ""),
                }
            )


def parse_step(revision: str) -> int:
    match = STEP_RE.match(revision)
    if not match:
        raise ValueError(f"invalid checkpoint revision {revision!r}; expected step<number>")
    return int(match.group("step"))


def parse_window(text: str) -> tuple[str, str]:
    compact = text.strip().replace(" ", "")
    match = WINDOW_RE.match(compact)
    if not match:
        raise ValueError(f"invalid window {text!r}; expected stepA->stepB")
    return match.group("start"), match.group("end")


def shard_end_steps(start: int, end_exclusive: int, shard_size: int) -> list[int]:
    if start < 0 or end_exclusive <= start:
        raise ValueError("window must satisfy 0 <= start < end")
    first_block = start // shard_size
    last_block = (end_exclusive - 1) // shard_size
    return [(block + 1) * shard_size for block in range(first_block, last_block + 1)]


def file_name(pattern: str, shard_end_step: int) -> str:
    return pattern.format(shard_end_step=shard_end_step)


def configured_windows(config: dict[str, Any], include_splits: set[str]) -> list[dict[str, Any]]:
    selected = config.get("selected_attribution_windows", {})
    if not isinstance(selected, dict):
        raise ValueError("dataset config selected_attribution_windows must be a mapping")
    windows: list[dict[str, Any]] = []
    for split in ["primary", "secondary", "controls"]:
        if split not in include_splits:
            continue
        for item in selected.get(split, []):
            if not isinstance(item, dict):
                raise ValueError(f"invalid selected window item: {item!r}")
            windows.append({**item, "split": split})
    return windows


def build_plan_record(
    *,
    from_revision: str,
    to_revision: str,
    config: dict[str, Any],
    dataset_config_path: Path,
    sample_size: int,
    seed: int,
    split: str,
    priority: Any,
    rationale: str,
) -> dict[str, Any]:
    start = parse_step(from_revision)
    end = parse_step(to_revision)
    if end <= start:
        raise ValueError(f"invalid window {from_revision}->{to_revision}: end must be greater than start")

    mapping = config["window_mapping"]
    shard_size = int(mapping["shard_size_steps"])
    practical = config["practical_repo"]
    pattern = str(practical["file_pattern"])
    shard_ends = shard_end_steps(start, end, shard_size)
    files = [file_name(pattern, shard_end) for shard_end in shard_ends]
    window_id = f"{from_revision}_to_{to_revision}"
    return {
        "schema_version": "0.1",
        "window_id": window_id,
        "from_revision": from_revision,
        "to_revision": to_revision,
        "batch_idx_start": start,
        "batch_idx_end_exclusive": end,
        "batch_count": end - start,
        "parquet_files": files,
        "row_filter": {
            "batch_idx_gte": start,
            "batch_idx_lt": end,
        },
        "sample_policy": {
            "sample_size": sample_size,
            "seed": seed,
            "mode": "uniform_without_replacement",
            "note": "Sampler may return fewer rows if the filtered window has fewer available rows.",
        },
        "split": split,
        "priority": priority,
        "rationale": rationale,
        "source": {
            "dataset_config": str(dataset_config_path),
            "repo_id": practical["repo_id"],
            "repo_type": practical.get("repo_type", "dataset"),
            "format": practical["format"],
            "data_dir": practical.get("data_dir", "data"),
            "file_pattern": pattern,
            "official_fallback_repo_id": config.get("official_fallback", {}).get("repo_id"),
        },
    }


def resolve_run_dir(args: argparse.Namespace) -> Path:
    run_id = args.run_id or default_run_id()
    return (
        args.output_root
        / args.experiment_name
        / args.model_name
        / args.dataset_name
        / args.probe_set
        / args.output_variant
        / run_id
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Pythia packed-training-stream files for attribution windows.")
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/datasets/pythia_preshuffled_stream.yaml"))
    parser.add_argument("--windows", default=None, help="Comma-separated windows like step128->step256,step256->step512. Defaults to selected windows from dataset config.")
    parser.add_argument("--include-splits", default="primary,secondary,controls", help="Config-selected splits to include when --windows is omitted.")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="assistant-axis-attribution-v0")
    parser.add_argument("--output-variant", default="training-window-plan")
    parser.add_argument("--run-id", default="selected-attribution-windows-v0")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be positive")
    config = load_yaml(args.dataset_config)
    run_dir = resolve_run_dir(args)
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    for directory in [meta_dir, checkpoints_dir, results_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "run.log"
    append_jsonl(log_path, {"time_utc": utc_now(), "event": "start", "dataset_config": str(args.dataset_config)})

    if args.windows:
        requested = []
        for text in args.windows.split(","):
            from_revision, to_revision = parse_window(text)
            requested.append(
                {
                    "from_revision": from_revision,
                    "to_revision": to_revision,
                    "split": "manual",
                    "priority": "manual",
                    "rationale": "manual CLI selection",
                }
            )
    else:
        include_splits = {item.strip() for item in args.include_splits.split(",") if item.strip()}
        requested = configured_windows(config, include_splits)

    plans = [
        build_plan_record(
            from_revision=str(item["from_revision"]),
            to_revision=str(item["to_revision"]),
            config=config,
            dataset_config_path=args.dataset_config,
            sample_size=args.sample_size,
            seed=args.seed,
            split=str(item.get("split", "")),
            priority=item.get("priority", ""),
            rationale=str(item.get("rationale", "")),
        )
        for item in requested
    ]

    summary = {
        "schema_version": "0.1",
        "planner": "TrainingWindowPlanner",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "dataset_config": str(args.dataset_config),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "window_count": len(plans),
        "windows": plans,
    }

    write_json(results_dir / "window_plan.json", summary)
    write_jsonl(results_dir / "window_plan.jsonl", plans)
    write_csv(results_dir / "window_plan.csv", plans)
    write_json(
        meta_dir / "run_manifest.json",
        {
            "schema_version": "0.1",
            "runner": "TrainingWindowPlanner",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "dataset_config": str(args.dataset_config),
            "output_paths": {
                "window_plan_json": str(results_dir / "window_plan.json"),
                "window_plan_jsonl": str(results_dir / "window_plan.jsonl"),
                "window_plan_csv": str(results_dir / "window_plan.csv"),
            },
        },
    )
    write_json(
        checkpoints_dir / "progress.json",
        {
            "schema_version": "0.1",
            "state": "completed",
            "updated_at_utc": utc_now(),
            "planned_windows": [plan["window_id"] for plan in plans],
        },
    )
    write_json(
        meta_dir / "status.json",
        {
            "schema_version": "0.1",
            "state": "completed",
            "message": "training window plan completed",
            "updated_at_utc": utc_now(),
            "counts": {"windows": len(plans)},
        },
    )
    append_jsonl(log_path, {"time_utc": utc_now(), "event": "completed", "window_count": len(plans)})
    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "window_count": len(plans),
                "window_plan": str(results_dir / "window_plan.json"),
                "window_plan_jsonl": str(results_dir / "window_plan.jsonl"),
                "window_plan_csv": str(results_dir / "window_plan.csv"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
