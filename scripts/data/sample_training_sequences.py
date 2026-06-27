#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(record)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "window_id",
        "planned_sample_size",
        "existing_records",
        "new_records",
        "total_records",
        "candidate_rows",
        "parquet_files",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas/pyarrow are required for Parquet sampling. Install with: "
            "pip install -r requirements-data.txt"
        ) from exc
    return pd


def planned_sample_size(plan: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    policy = plan.get("sample_policy", {})
    return int(policy.get("sample_size", 1000))


def planned_seed(plan: dict[str, Any], seed_offset: int) -> int:
    policy = plan.get("sample_policy", {})
    return int(policy.get("seed", 17)) + seed_offset


def file_path_for_plan(plan: dict[str, Any], parquet_file: str, args: argparse.Namespace) -> Path:
    if args.local_data_dir:
        candidate = args.local_data_dir / parquet_file
        if candidate.exists():
            return candidate
        nested = args.local_data_dir / str(plan["source"].get("data_dir", "data")) / parquet_file
        if nested.exists():
            return nested
        raise FileNotFoundError(f"could not find {parquet_file} under {args.local_data_dir}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for dataset download. Install requirements-data.txt") from exc

    filename = f"{plan['source'].get('data_dir', 'data').strip('/')}/{parquet_file}"
    return Path(
        hf_hub_download(
            repo_id=str(plan["source"]["repo_id"]),
            repo_type=str(plan["source"].get("repo_type", "dataset")),
            filename=filename,
            cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
            token=args.hf_token,
        )
    )


def normalize_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def iter_candidate_rows(plan: dict[str, Any], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    pd = import_pandas()
    start = int(plan["row_filter"]["batch_idx_gte"])
    end = int(plan["row_filter"]["batch_idx_lt"])
    for parquet_file in plan["parquet_files"]:
        path = file_path_for_plan(plan, str(parquet_file), args)
        frame = pd.read_parquet(path, columns=["uid", "batch_idx", "token_ids"])
        frame = frame[(frame["batch_idx"] >= start) & (frame["batch_idx"] < end)]
        for row in frame.itertuples(index=False):
            yield {
                "uid": str(row.uid),
                "batch_idx": int(row.batch_idx),
                "token_ids": normalize_token_ids(row.token_ids),
                "source_file": str(parquet_file),
            }


def sample_rows(candidates: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size >= len(candidates):
        return list(candidates)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(candidates)), sample_size))
    return [candidates[index] for index in indices]


def build_sample_record(row: dict[str, Any], plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    token_ids = row["token_ids"]
    window_id = str(plan["window_id"])
    uid = str(row["uid"])
    return {
        "schema_version": "0.1",
        "sample_id": f"{window_id}__{uid}",
        "window_id": window_id,
        "uid": uid,
        "batch_idx": int(row["batch_idx"]),
        "source_file": row["source_file"],
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "source": {
            "sampler_run_dir": str(run_dir),
            "window_plan_id": window_id,
            "from_revision": plan["from_revision"],
            "to_revision": plan["to_revision"],
            "repo_id": plan["source"]["repo_id"],
            "row_filter": plan["row_filter"],
        },
    }


def existing_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        window_id = str(row["window_id"])
        counts[window_id] = counts.get(window_id, 0) + 1
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample packed Pythia training sequences from planned windows.")
    parser.add_argument("--window-plan-jsonl", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None, help="Override per-window sample size from the plan.")
    parser.add_argument("--max-windows", type=int, default=None, help="Only process the first N windows from the plan.")
    parser.add_argument("--local-data-dir", type=Path, default=None, help="Directory containing Parquet files, optionally nested under data/.")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Plan and report required files without reading Parquet.")
    parser.add_argument("--force-completed", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="assistant-axis-attribution-v0")
    parser.add_argument("--output-variant", default="training-sequence-sample")
    parser.add_argument("--run-id", default="selected-attribution-windows-sample-v0")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.sample_size is not None and args.sample_size < 1:
        raise SystemExit("--sample-size must be positive")
    if args.max_windows is not None and args.max_windows < 1:
        raise SystemExit("--max-windows must be positive")

    plans = load_jsonl(args.window_plan_jsonl)
    if args.max_windows is not None:
        plans = plans[: args.max_windows]
    if not plans:
        raise SystemExit("no window plans selected")

    run_dir = resolve_run_dir(args)
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    for directory in [meta_dir, checkpoints_dir, results_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    status_path = meta_dir / "status.json"
    if status_path.exists() and not args.force_completed:
        status = load_json(status_path)
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    samples_path = results_dir / "sampled_sequences.jsonl"
    existing_rows = load_jsonl(samples_path)
    counts = existing_counts(existing_rows)
    summary_rows: list[dict[str, Any]] = []
    log_path = logs_dir / "run.log"
    append_jsonl(log_path, {"time_utc": utc_now(), "event": "start", "plans": len(plans), "dry_run": args.dry_run})

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    iterator = tqdm(plans, desc="sample windows", unit="window") if tqdm else plans

    failed: dict[str, Any] | None = None
    try:
        for index, plan in enumerate(iterator):
            window_id = str(plan["window_id"])
            target = planned_sample_size(plan, args.sample_size)
            existing = counts.get(window_id, 0)
            if existing >= target and not args.force_completed:
                summary_rows.append(
                    {
                        "window_id": window_id,
                        "planned_sample_size": target,
                        "existing_records": existing,
                        "new_records": 0,
                        "total_records": existing,
                        "candidate_rows": "",
                        "parquet_files": ";".join(plan["parquet_files"]),
                        "status": "skipped_existing",
                    }
                )
                continue

            if args.dry_run:
                summary_rows.append(
                    {
                        "window_id": window_id,
                        "planned_sample_size": target,
                        "existing_records": existing,
                        "new_records": 0,
                        "total_records": existing,
                        "candidate_rows": "",
                        "parquet_files": ";".join(plan["parquet_files"]),
                        "status": "dry_run",
                    }
                )
                continue

            candidates = list(iter_candidate_rows(plan, args))
            chosen = sample_rows(candidates, max(0, target - existing), planned_seed(plan, index))
            for row in chosen:
                append_jsonl(samples_path, build_sample_record(row, plan, run_dir))
            total = existing + len(chosen)
            counts[window_id] = total
            summary_rows.append(
                {
                    "window_id": window_id,
                    "planned_sample_size": target,
                    "existing_records": existing,
                    "new_records": len(chosen),
                    "total_records": total,
                    "candidate_rows": len(candidates),
                    "parquet_files": ";".join(plan["parquet_files"]),
                    "status": "completed" if total >= target else "partial",
                }
            )
            write_json(
                checkpoints_dir / "progress.json",
                {
                    "schema_version": "0.1",
                    "state": "running",
                    "updated_at_utc": utc_now(),
                    "completed_windows": [row["window_id"] for row in summary_rows],
                    "counts": counts,
                },
            )

        final_state = "dry_run" if args.dry_run else "completed"
        final_message = "training sequence sampling dry run completed" if args.dry_run else "training sequence sampling completed"
    except Exception as exc:
        final_state = "failed"
        final_message = f"training sequence sampling failed: {type(exc).__name__}: {exc}"
        failed = {"error_type": type(exc).__name__, "message": str(exc)}
        append_jsonl(log_path, {"time_utc": utc_now(), "event": "error", **failed})

    write_json(
        meta_dir / "run_manifest.json",
        {
            "schema_version": "0.1",
            "runner": "TrainingSequenceSampler",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "window_plan_jsonl": str(args.window_plan_jsonl),
            "sample_size_override": args.sample_size,
            "local_data_dir": str(args.local_data_dir) if args.local_data_dir else None,
            "hf_cache_dir": str(args.hf_cache_dir) if args.hf_cache_dir else None,
            "dry_run": args.dry_run,
        },
    )
    write_json(
        results_dir / "window_sample_summary.json",
        {
            "schema_version": "0.1",
            "run_dir": str(run_dir),
            "sampled_sequences_jsonl": str(samples_path),
            "windows": summary_rows,
            "failed": failed,
        },
    )
    write_csv(results_dir / "window_sample_summary.csv", summary_rows)
    write_json(
        checkpoints_dir / "progress.json",
        {
            "schema_version": "0.1",
            "state": final_state,
            "updated_at_utc": utc_now(),
            "completed_windows": [row["window_id"] for row in summary_rows],
            "counts": counts,
        },
    )
    write_json(
        status_path,
        {
            "schema_version": "0.1",
            "state": final_state,
            "message": final_message,
            "updated_at_utc": utc_now(),
            "counts": {
                "windows": len(plans),
                "sampled_records": sum(counts.values()),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": final_state,
                "message": final_message,
                "run_dir": str(run_dir),
                "sampled_sequences": str(samples_path),
                "summary": str(results_dir / "window_sample_summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_state in {"completed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
