#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no} in {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no} in {path} must be a JSON object")
            records.append(record)
    return records


def resolve_artifact_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def describe_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def maybe_load_tensor_shape(path: Path, enabled: bool) -> list[int] | None:
    if not enabled or not path.exists():
        return None
    import torch

    tensor = torch.load(path, map_location="cpu")
    return list(tensor.shape)


def inspect_run(run_dir: Path, repo_root: Path, load_tensors: bool) -> dict[str, Any]:
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    status_path = meta_dir / "status.json"
    manifest_path = meta_dir / "run_manifest.json"
    progress_path = checkpoints_dir / "progress.json"
    index_path = results_dir / "activation_index.jsonl"

    status = load_json(status_path)
    manifest = load_json(manifest_path)
    progress = load_json(progress_path)
    index_rows = load_jsonl(index_path)

    rollout_ids = [str(row.get("rollout_id", "")) for row in index_rows]
    duplicate_rollout_ids = duplicate_values(rollout_ids)
    missing_activation_paths: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    bad_spans: list[dict[str, Any]] = []
    activation_paths_seen: set[str] = set()

    for row in index_rows:
        activation_path_text = str(row.get("activation_path", ""))
        activation_path = resolve_artifact_path(activation_path_text, repo_root)
        if not activation_path.exists():
            missing_activation_paths.append(activation_path_text)
        else:
            activation_paths_seen.add(activation_path_text)

        start = row.get("response_token_start")
        end = row.get("response_token_end")
        count = row.get("response_token_count")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(count, int) or end <= start or count <= 0:
            bad_spans.append(
                {
                    "rollout_id": row.get("rollout_id"),
                    "response_token_start": start,
                    "response_token_end": end,
                    "response_token_count": count,
                }
            )
        elif end - start != count:
            bad_spans.append(
                {
                    "rollout_id": row.get("rollout_id"),
                    "response_token_start": start,
                    "response_token_end": end,
                    "response_token_count": count,
                    "expected_count": end - start,
                }
            )

        recorded_shape = row.get("activation_shape")
        loaded_shape = maybe_load_tensor_shape(activation_path, load_tensors)
        if loaded_shape is not None and recorded_shape != loaded_shape:
            shape_mismatches.append(
                {
                    "rollout_id": row.get("rollout_id"),
                    "activation_path": activation_path_text,
                    "recorded_shape": recorded_shape,
                    "loaded_shape": loaded_shape,
                }
            )

    selected_count = None
    if progress:
        selected_count = progress.get("selected_count")
    if selected_count is None and manifest:
        selected_count = manifest.get("selection", {}).get("selected_count")

    response_counts = [
        int(row["response_token_count"])
        for row in index_rows
        if isinstance(row.get("response_token_count"), int)
    ]
    prompt_counts = [
        int(row["prompt_token_count"])
        for row in index_rows
        if isinstance(row.get("prompt_token_count"), int)
    ]

    completed_by_index_and_file = len(index_rows) - len(missing_activation_paths)
    remaining_by_index = None
    if isinstance(selected_count, int):
        remaining_by_index = max(selected_count - completed_by_index_and_file, 0)

    checks = {
        "status_json_exists": status_path.exists(),
        "run_manifest_exists": manifest_path.exists(),
        "progress_json_exists": progress_path.exists(),
        "activation_index_exists": index_path.exists(),
        "no_duplicate_rollout_ids": not duplicate_rollout_ids,
        "all_activation_paths_exist": not missing_activation_paths,
        "all_response_spans_valid": not bad_spans,
        "tensor_shapes_match_index": not shape_mismatches,
    }

    return {
        "run_dir": str(run_dir),
        "paths": {
            "status": str(status_path),
            "manifest": str(manifest_path),
            "progress": str(progress_path),
            "activation_index": str(index_path),
        },
        "status": status,
        "manifest_summary": {
            "runner": manifest.get("runner") if manifest else None,
            "model_id": manifest.get("model_id") if manifest else None,
            "revision": manifest.get("revision") if manifest else None,
            "layer": manifest.get("layer") if manifest else None,
            "pooling_policy": manifest.get("pooling_policy") if manifest else None,
        },
        "progress": progress,
        "counts": {
            "selected": selected_count,
            "index_rows": len(index_rows),
            "activation_files_existing": len(activation_paths_seen),
            "missing_activation_files": len(missing_activation_paths),
            "completed_by_index_and_file": completed_by_index_and_file,
            "remaining_by_index": remaining_by_index,
            "duplicate_rollout_ids": len(duplicate_rollout_ids),
            "bad_response_spans": len(bad_spans),
            "shape_mismatches": len(shape_mismatches),
        },
        "breakdowns": {
            "record_type": dict(Counter(str(row.get("record_type")) for row in index_rows)),
            "role_group": dict(Counter(str(row.get("role_group")) for row in index_rows if row.get("role_group") is not None)),
            "checkpoint_revision": dict(Counter(str(row.get("checkpoint_revision")) for row in index_rows)),
            "layer": dict(Counter(str(row.get("layer")) for row in index_rows)),
            "activation_shape": dict(Counter(json.dumps(row.get("activation_shape")) for row in index_rows)),
        },
        "span_stats": {
            "prompt_token_count": describe_numbers(prompt_counts),
            "response_token_count": describe_numbers(response_counts),
        },
        "checks": checks,
        "problems": {
            "duplicate_rollout_ids": duplicate_rollout_ids[:20],
            "missing_activation_paths": missing_activation_paths[:20],
            "bad_response_spans": bad_spans[:20],
            "shape_mismatches": shape_mismatches[:20],
        },
    }


def print_human(report: dict[str, Any]) -> None:
    print("Activation Run Inspection")
    print("-------------------------")
    print(f"run_dir: {report['run_dir']}")
    status = report.get("status") or {}
    print(f"state: {status.get('state')}")
    print(f"message: {status.get('message')}")
    print("")
    print("Counts")
    print("------")
    for key, value in report["counts"].items():
        print(f"{key}: {value}")
    print("")
    print("Span Stats")
    print("----------")
    for key, stats in report["span_stats"].items():
        print(f"{key}: {stats}")
    print("")
    print("Checks")
    print("------")
    for key, value in report["checks"].items():
        print(f"{key}: {value}")
    problems = report["problems"]
    if any(problems.values()):
        print("")
        print("Problem Samples")
        print("---------------")
        for key, values in problems.items():
            if values:
                print(f"{key}: {json.dumps(values, indent=2, sort_keys=True)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an activation cache run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--load-tensors", action="store_true", help="Load .pt files to verify actual tensor shapes.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    report = inspect_run(run_dir=run_dir, repo_root=repo_root, load_tensors=args.load_tensors)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    critical_checks = [
        "status_json_exists",
        "run_manifest_exists",
        "progress_json_exists",
        "activation_index_exists",
        "no_duplicate_rollout_ids",
        "all_activation_paths_exist",
        "all_response_spans_valid",
        "tensor_shapes_match_index",
    ]
    return 0 if all(report["checks"][check] for check in critical_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
