#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} contains a non-object JSONL row")
                rows.append(record)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "window_id",
        "uid",
        "batch_idx",
        "source_file",
        "token_count",
        "eos_token_count",
        "decoded_char_count",
        "preview_start",
        "preview_middle",
        "preview_end",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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


def normalize_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("token_ids must be a list-like value")
    return [int(item) for item in value]


def existing_sample_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["sample_id"]) for row in rows if "sample_id" in row}


def import_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for decoding. Install requirements-model.txt "
            "or a compatible transformers build."
        ) from exc
    return AutoTokenizer


def load_tokenizer(args: argparse.Namespace) -> Any:
    AutoTokenizer = import_tokenizer()
    kwargs: dict[str, Any] = {
        "cache_dir": str(args.hf_cache_dir) if args.hf_cache_dir else None,
        "local_files_only": args.local_files_only,
        "revision": args.tokenizer_revision,
        "token": args.hf_token,
    }
    return AutoTokenizer.from_pretrained(args.tokenizer_id, **{k: v for k, v in kwargs.items() if v is not None})


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def compact_text(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def clipped(text: str, max_chars: int) -> str:
    text = compact_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def build_previews(decoded: str, preview_chars: int) -> dict[str, str]:
    midpoint = max(0, len(decoded) // 2 - preview_chars // 2)
    return {
        "preview_start": clipped(decoded[:preview_chars], preview_chars),
        "preview_middle": clipped(decoded[midpoint : midpoint + preview_chars], preview_chars),
        "preview_end": clipped(decoded[-preview_chars:], preview_chars),
    }


def count_token(token_ids: list[int], token_id: int | None) -> int:
    if token_id is None:
        return 0
    return sum(1 for item in token_ids if item == token_id)


def build_decoded_record(
    sample: dict[str, Any],
    tokenizer: Any,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    run_dir: Path,
    preview_chars: int,
    include_full_text: bool,
) -> dict[str, Any]:
    token_ids = normalize_token_ids(sample["token_ids"])
    decoded = decode_token_ids(tokenizer, token_ids)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    record = {
        "schema_version": "0.1",
        "sample_id": str(sample["sample_id"]),
        "window_id": str(sample["window_id"]),
        "uid": str(sample["uid"]),
        "batch_idx": int(sample["batch_idx"]),
        "source_file": str(sample["source_file"]),
        "token_count": int(sample.get("token_count", len(token_ids))),
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "eos_token_id": eos_token_id,
        "eos_token_count": count_token(token_ids, eos_token_id),
        "decoded_char_count": len(decoded),
        **build_previews(decoded, preview_chars),
        "source": {
            "sample_source": sample.get("source", {}),
            "decode_run_dir": str(run_dir),
        },
    }
    if include_full_text:
        record["decoded_text"] = decoded
    return record


def write_progress(
    path: Path,
    state: str,
    total_records: int,
    completed_ids: set[str],
    failed: dict[str, Any] | None = None,
) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "state": state,
            "updated_at_utc": utc_now(),
            "total_records": total_records,
            "completed_records": len(completed_ids),
            "completed_sample_ids": sorted(completed_ids),
            "failed": failed,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode sampled packed Pythia training sequences for inspection.")
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default="EleutherAI/pythia-410m-deduped")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--preview-chars", type=int, default=600)
    parser.add_argument("--include-full-text", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate selected sample records without loading a tokenizer.")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--force-completed", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="assistant-axis-attribution-v0")
    parser.add_argument("--output-variant", default="training-sequence-decode")
    parser.add_argument("--run-id", default="selected-attribution-windows-decode-v0")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.max_records is not None and args.max_records < 1:
        raise SystemExit("--max-records must be positive")
    if args.preview_chars < 1:
        raise SystemExit("--preview-chars must be positive")
    if args.save_every < 1:
        raise SystemExit("--save-every must be positive")

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

    samples = load_jsonl(args.sample_jsonl)
    if args.max_records is not None:
        samples = samples[: args.max_records]
    if not samples:
        raise SystemExit("no sample records selected")
    selected_ids = {str(sample["sample_id"]) for sample in samples}
    if len(selected_ids) != len(samples):
        raise SystemExit("sample records contain duplicate sample_id values")

    decoded_path = results_dir / "decoded_sequences.jsonl"
    existing_rows = load_jsonl(decoded_path)
    existing_rows = [row for row in existing_rows if str(row.get("sample_id", "")) in selected_ids]
    completed_ids = existing_sample_ids(existing_rows)
    log_path = logs_dir / "run.log"
    append_jsonl(
        log_path,
        {
            "time_utc": utc_now(),
            "event": "start",
            "sample_jsonl": str(args.sample_jsonl),
            "selected_records": len(samples),
            "existing_records": len(completed_ids),
            "dry_run": args.dry_run,
        },
    )

    write_json(
        meta_dir / "run_manifest.json",
        {
            "schema_version": "0.1",
            "runner": "TrainingSequenceDecoder",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "sample_jsonl": str(args.sample_jsonl),
            "tokenizer_id": args.tokenizer_id,
            "tokenizer_revision": args.tokenizer_revision,
            "hf_cache_dir": str(args.hf_cache_dir) if args.hf_cache_dir else None,
            "local_files_only": args.local_files_only,
            "max_records": args.max_records,
            "preview_chars": args.preview_chars,
            "include_full_text": args.include_full_text,
            "dry_run": args.dry_run,
        },
    )

    failed: dict[str, Any] | None = None
    summary_rows: list[dict[str, Any]] = [
        {
            "sample_id": str(row["sample_id"]),
            "window_id": str(row["window_id"]),
            "uid": str(row["uid"]),
            "batch_idx": int(row["batch_idx"]),
            "source_file": str(row["source_file"]),
            "token_count": int(row.get("token_count", len(row.get("token_ids", [])))),
            "eos_token_count": row.get("eos_token_count", ""),
            "decoded_char_count": row.get("decoded_char_count", ""),
            "preview_start": row.get("preview_start", ""),
            "preview_middle": row.get("preview_middle", ""),
            "preview_end": row.get("preview_end", ""),
        }
        for row in existing_rows
    ]

    try:
        if args.dry_run:
            final_state = "dry_run"
            final_message = "training sequence decode dry run completed"
        else:
            tokenizer = load_tokenizer(args)
            try:
                from tqdm.auto import tqdm
            except ImportError:
                tqdm = None
            pending_samples = [
                sample
                for sample in samples
                if args.force_completed or str(sample["sample_id"]) not in completed_ids
            ]
            initial_completed = 0 if args.force_completed else len(completed_ids)
            iterator = (
                tqdm(pending_samples, desc="decode samples", unit="sample", initial=initial_completed, total=len(samples))
                if tqdm
                else pending_samples
            )
            processed_since_save = 0
            for sample in iterator:
                sample_id = str(sample["sample_id"])
                record = build_decoded_record(
                    sample=sample,
                    tokenizer=tokenizer,
                    tokenizer_id=args.tokenizer_id,
                    tokenizer_revision=args.tokenizer_revision,
                    run_dir=run_dir,
                    preview_chars=args.preview_chars,
                    include_full_text=args.include_full_text,
                )
                append_jsonl(decoded_path, record)
                completed_ids.add(sample_id)
                summary_rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "window_id": record["window_id"],
                        "uid": record["uid"],
                        "batch_idx": record["batch_idx"],
                        "source_file": record["source_file"],
                        "token_count": record["token_count"],
                        "eos_token_count": record["eos_token_count"],
                        "decoded_char_count": record["decoded_char_count"],
                        "preview_start": record["preview_start"],
                        "preview_middle": record["preview_middle"],
                        "preview_end": record["preview_end"],
                    }
                )
                processed_since_save += 1
                if processed_since_save >= args.save_every:
                    write_progress(checkpoints_dir / "progress.json", "running", len(samples), completed_ids)
                    processed_since_save = 0
            final_state = "completed"
            final_message = "training sequence decode completed"
    except Exception as exc:
        final_state = "failed"
        final_message = f"training sequence decode failed: {type(exc).__name__}: {exc}"
        failed = {"error_type": type(exc).__name__, "message": str(exc)}
        append_jsonl(log_path, {"time_utc": utc_now(), "event": "error", **failed})

    write_progress(checkpoints_dir / "progress.json", final_state, len(samples), completed_ids, failed)
    write_csv(results_dir / "decoded_preview.csv", summary_rows)
    write_json(
        results_dir / "decode_summary.json",
        {
            "schema_version": "0.1",
            "run_dir": str(run_dir),
            "sample_jsonl": str(args.sample_jsonl),
            "decoded_sequences_jsonl": str(decoded_path),
            "decoded_preview_csv": str(results_dir / "decoded_preview.csv"),
            "selected_records": len(samples),
            "decoded_records": len(completed_ids),
            "dry_run": args.dry_run,
            "include_full_text": args.include_full_text,
            "failed": failed,
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
                "selected_records": len(samples),
                "decoded_records": len(completed_ids),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": final_state,
                "message": final_message,
                "run_dir": str(run_dir),
                "decoded_sequences": str(decoded_path),
                "summary": str(results_dir / "decode_summary.json"),
                "preview_csv": str(results_dir / "decoded_preview.csv"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_state in {"completed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
