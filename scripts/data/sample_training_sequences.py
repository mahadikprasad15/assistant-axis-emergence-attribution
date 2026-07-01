#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import secrets
import time
from array import array
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


def import_tqdm(enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm


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


def progress_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def read_filtered_parquet(
    pd: Any,
    path: Path,
    start: int,
    end: int,
    log_path: Path,
    parquet_file: str,
) -> Any:
    columns = ["uid", "batch_idx", "token_ids"]
    filters = [("batch_idx", ">=", start), ("batch_idx", "<", end)]
    try:
        return pd.read_parquet(path, columns=columns, filters=filters)
    except Exception as exc:
        append_jsonl(
            log_path,
            {
                "time_utc": utc_now(),
                "event": "parquet_filter_read_fallback",
                "parquet_file": parquet_file,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        frame = pd.read_parquet(path, columns=columns)
        return frame[(frame["batch_idx"] >= start) & (frame["batch_idx"] < end)]


def iter_candidate_rows(
    plan: dict[str, Any],
    args: argparse.Namespace,
    log_path: Path,
    tqdm: Any | None,
    progress_enabled: bool,
) -> Iterable[dict[str, Any]]:
    pd = import_pandas()
    start = int(plan["row_filter"]["batch_idx_gte"])
    end = int(plan["row_filter"]["batch_idx_lt"])
    window_id = str(plan["window_id"])
    parquet_files = [str(item) for item in plan["parquet_files"]]
    file_iter = (
        tqdm(parquet_files, desc=f"{window_id} parquet files", unit="file", leave=False)
        if tqdm
        else parquet_files
    )
    for parquet_file in file_iter:
        progress_print(
            progress_enabled,
            f"[{window_id}] resolving/loading {parquet_file} for batch_idx [{start}, {end})",
        )
        read_started = time.perf_counter()
        append_jsonl(
            log_path,
            {
                "time_utc": utc_now(),
                "event": "parquet_read_start",
                "window_id": window_id,
                "parquet_file": parquet_file,
                "batch_idx_gte": start,
                "batch_idx_lt": end,
            },
        )
        path = file_path_for_plan(plan, str(parquet_file), args)
        frame = read_filtered_parquet(pd, path, start, end, log_path, parquet_file)
        elapsed = time.perf_counter() - read_started
        append_jsonl(
            log_path,
            {
                "time_utc": utc_now(),
                "event": "parquet_read_done",
                "window_id": window_id,
                "parquet_file": parquet_file,
                "local_path": str(path),
                "rows_after_filter": int(len(frame)),
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        progress_print(
            progress_enabled,
            f"[{window_id}] loaded {parquet_file}: {len(frame)} candidate rows in {elapsed:.1f}s",
        )
        row_iter = (
            tqdm(frame.itertuples(index=False), total=len(frame), desc=f"{parquet_file} rows", unit="row", leave=False)
            if tqdm and len(frame) >= 1000
            else frame.itertuples(index=False)
        )
        for row in row_iter:
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


def reservoir_sample_parquet(
    plan: dict[str, Any],
    args: argparse.Namespace,
    log_path: Path,
    sample_size: int,
    seed: int,
    tqdm: Any | None,
    progress_enabled: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Two-pass uniform sampling without decoding unselected token rows."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for streaming Parquet sampling") from exc

    start = int(plan["row_filter"]["batch_idx_gte"])
    end = int(plan["row_filter"]["batch_idx_lt"])
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    candidate_count = 0
    parquet_files = [str(item) for item in plan["parquet_files"]]
    resolved_paths: dict[str, Path] = {}

    # Pass 1: select row coordinates using only the small scalar columns.
    for parquet_file in parquet_files:
        progress_print(
            progress_enabled,
            f"[{plan['window_id']}] pass 1/2 selecting rows from {parquet_file}",
        )
        path = file_path_for_plan(plan, parquet_file, args)
        resolved_paths[parquet_file] = path
        started = time.perf_counter()
        parquet = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
        file_candidates = 0
        row_group_iter = range(parquet.metadata.num_row_groups)
        if tqdm:
            row_group_iter = tqdm(
                row_group_iter,
                total=parquet.metadata.num_row_groups,
                desc=f"{parquet_file} scalar row groups",
                unit="group",
                leave=False,
            )
        for row_group in row_group_iter:
            row_offset = 0
            for batch in parquet.iter_batches(
                batch_size=8192,
                row_groups=[row_group],
                columns=["uid", "batch_idx"],
                use_threads=False,
            ):
                uid_column = batch.column(batch.schema.get_field_index("uid"))
                batch_idx_column = batch.column(batch.schema.get_field_index("batch_idx"))
                for batch_row in range(batch.num_rows):
                    batch_idx = int(batch_idx_column[batch_row].as_py())
                    if not start <= batch_idx < end:
                        continue
                    stream_index = candidate_count
                    candidate_count += 1
                    file_candidates += 1
                    if len(reservoir) < sample_size:
                        slot = len(reservoir)
                    else:
                        slot = rng.randrange(candidate_count)
                        if slot >= sample_size:
                            continue
                    record = {
                        "uid": str(uid_column[batch_row].as_py()),
                        "batch_idx": batch_idx,
                        "source_file": parquet_file,
                        "_row_group": row_group,
                        "_row_index": row_offset + batch_row,
                        "_stream_index": stream_index,
                    }
                    if slot == len(reservoir):
                        reservoir.append(record)
                    else:
                        reservoir[slot] = record
                row_offset += batch.num_rows
                del uid_column, batch_idx_column, batch
            if row_group % 16 == 0:
                pa.default_memory_pool().release_unused()
                gc.collect()
        pa.default_memory_pool().release_unused()
        append_jsonl(
            log_path,
            {
                "time_utc": utc_now(),
                "event": "parquet_stream_done",
                "window_id": str(plan["window_id"]),
                "parquet_file": parquet_file,
                "local_path": str(path),
                "candidate_rows": file_candidates,
                "reservoir_records": len(reservoir),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )

    # Pass 2: decode token IDs only for the selected row coordinates.
    selected_groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for slot, record in enumerate(reservoir):
        key = (str(record["source_file"]), int(record["_row_group"]))
        selected_groups.setdefault(key, []).append((int(record["_row_index"]), slot))
    group_items = sorted(selected_groups.items())
    if tqdm:
        group_items = tqdm(group_items, desc="selected token row groups", unit="group")
    open_file: str | None = None
    parquet = None
    for (parquet_file, row_group), selected in group_items:
        if parquet_file != open_file:
            parquet = pq.ParquetFile(
                resolved_paths[parquet_file], memory_map=True, pre_buffer=False
            )
            open_file = parquet_file
        assert parquet is not None
        selected.sort()
        cursor = 0
        row_offset = 0
        for batch in parquet.iter_batches(
            batch_size=32,
            row_groups=[row_group],
            columns=["token_ids"],
            use_threads=False,
        ):
            batch_end = row_offset + batch.num_rows
            token_column = batch.column(batch.schema.get_field_index("token_ids"))
            while cursor < len(selected) and selected[cursor][0] < batch_end:
                row_index, slot = selected[cursor]
                if row_index >= row_offset:
                    reservoir[slot]["token_ids"] = array(
                        "I", token_column[row_index - row_offset].as_py()
                    )
                cursor += 1
            row_offset = batch_end
            del token_column, batch
            if cursor == len(selected):
                break
        if cursor != len(selected):
            raise ValueError(
                f"failed to recover {len(selected) - cursor} selected token rows "
                f"from {parquet_file} row group {row_group}"
            )
        pa.default_memory_pool().release_unused()
        gc.collect()

    if any("token_ids" not in record for record in reservoir):
        raise ValueError("selected reservoir contains records without recovered token IDs")
    reservoir.sort(key=lambda row: int(row["_stream_index"]))
    for row in reservoir:
        for key in ["_stream_index", "_row_group", "_row_index"]:
            row.pop(key)
    return reservoir, candidate_count


def build_sample_record(row: dict[str, Any], plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    token_ids = normalize_token_ids(row["token_ids"])
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
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm bars and progress messages.")
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

    progress_enabled = not args.no_progress
    tqdm = import_tqdm(progress_enabled)
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

            chosen, candidate_count = reservoir_sample_parquet(
                plan, args, log_path, target, planned_seed(plan, index), tqdm, progress_enabled
            )
            existing_ids = {str(row["sample_id"]) for row in existing_rows if str(row["window_id"]) == window_id}
            chosen_ids = {f"{window_id}__{row['uid']}" for row in chosen}
            if existing_ids - chosen_ids:
                raise ValueError(
                    f"existing sample for {window_id} does not match deterministic reservoir selection; "
                    "use a new run id"
                )
            new_rows = [row for row in chosen if f"{window_id}__{row['uid']}" not in existing_ids]
            for row in new_rows:
                append_jsonl(samples_path, build_sample_record(row, plan, run_dir))
            total = len(existing_ids | chosen_ids)
            counts[window_id] = total
            summary_rows.append(
                {
                    "window_id": window_id,
                    "planned_sample_size": target,
                    "existing_records": existing,
                    "new_records": len(new_rows),
                    "total_records": total,
                    "candidate_rows": candidate_count,
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
