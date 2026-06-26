#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_progress_bar(total: int, initial: int, enabled: bool) -> Any:
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        initial=initial,
        desc="activation records",
        unit="record",
        dynamic_ncols=True,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    append_jsonl(path, {"event": event, "payload": payload, "time_utc": utc_now()})


def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def checkpoint_step_from_revision(revision: str) -> int | None:
    match = re.fullmatch(r"step(\d+)", revision)
    if not match:
        return None
    return int(match.group(1))


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_run_dir:
        return args.resume_run_dir
    run_id = args.run_id or default_run_id()
    return (
        args.output_root
        / args.experiment_name
        / args.model_name
        / args.dataset_name
        / args.probe_set
        / args.variant
        / run_id
    )


def torch_dtype_from_name(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def selected_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    if limit < 1:
        raise ValueError("--limit must be positive when provided")
    return records[:limit]


def load_completed_ids(index_path: Path) -> set[str]:
    completed: set[str] = set()
    if not index_path.exists():
        return completed
    for row in load_jsonl(index_path):
        activation_path = Path(str(row.get("activation_path", "")))
        if activation_path.exists():
            completed.add(str(row["rollout_id"]))
    return completed


def build_texts(record: dict[str, Any], separator: str) -> tuple[str, str]:
    prompt_prefix = str(record["prompt_text"]) + separator
    response = str(record["generated_response"]).strip()
    if not response:
        raise ValueError(f"empty generated_response for rollout_id={record['rollout_id']}")
    return prompt_prefix, prompt_prefix + response


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device_map,
    )
    model.eval()
    return model, tokenizer


def write_status(path: Path, state: str, message: str, counts: dict[str, int]) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "state": state,
            "message": message,
            "updated_at_utc": utc_now(),
            "counts": counts,
        },
    )


def write_progress(path: Path, selected_ids: list[str], completed_ids: set[str], cursor: int) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "updated_at_utc": utc_now(),
            "cursor": cursor,
            "selected_count": len(selected_ids),
            "completed_count": len(completed_ids),
            "remaining_count": len(set(selected_ids) - completed_ids),
            "completed_rollout_ids": sorted(completed_ids),
        },
    )


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    response_jsonl: Path,
    index_path: Path,
    selected_count: int,
    completed_count: int,
) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "runner": "ActivationCacheRunner",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "model_id": args.model_id,
            "revision": args.revision,
            "layer": args.layer,
            "pooling_policy": "response_token_mean",
            "response_separator": args.response_separator,
            "response_jsonl": {
                "path": str(response_jsonl),
                "sha256": file_sha256(response_jsonl),
            },
            "results": {
                "activation_index_jsonl": {
                    "path": str(index_path),
                    "sha256": file_sha256(index_path),
                }
            },
            "selection": {
                "limit": args.limit,
                "selected_count": selected_count,
                "completed_count": completed_count,
            },
            "execution": {
                "batch_size": args.batch_size,
                "save_every": args.save_every,
                "progress_enabled": not args.no_progress,
            },
        },
    )


def cache_batch(
    batch: list[dict[str, Any]],
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    activations_dir: Path,
    index_path: Path,
) -> list[str]:
    import torch

    prefixes: list[str] = []
    full_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for record in batch:
        prefix, full_text = build_texts(record, args.response_separator)
        prompt_len = token_count(tokenizer, prefix)
        full_len = token_count(tokenizer, full_text)
        if full_len <= prompt_len:
            raise ValueError(
                f"non-positive response span for rollout_id={record['rollout_id']}: "
                f"prompt_len={prompt_len}, full_len={full_len}"
            )
        prefixes.append(prefix)
        full_texts.append(full_text)
        spans.append((prompt_len, full_len))

    encoded = tokenizer(full_texts, return_tensors="pt", padding=True, add_special_tokens=True)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[args.layer + 1].detach().cpu()

    completed_ids: list[str] = []
    for row_idx, record in enumerate(batch):
        response_start, response_end = spans[row_idx]
        pooled = hidden[row_idx, response_start:response_end, :].mean(dim=0)
        rollout_id = str(record["rollout_id"])
        activation_path = (
            activations_dir
            / f"{sanitize_id(rollout_id)}__{sanitize_id(args.revision)}__layer{args.layer:02d}.pt"
        )
        torch.save(pooled, activation_path)
        activation_id = f"{rollout_id}__{args.revision}__layer{args.layer:02d}__response_token_mean"
        index_row = {
            "schema_version": "0.1",
            "activation_id": activation_id,
            "rollout_id": rollout_id,
            "record_type": record.get("record_type"),
            "role_id": record.get("role_id"),
            "role_group": record.get("role_group"),
            "default_prompt_id": record.get("default_prompt_id"),
            "question_id": record.get("question_id"),
            "question_category": record.get("question_category"),
            "model_id": args.model_id,
            "model_repo_id": args.model_id,
            "checkpoint_revision": args.revision,
            "checkpoint_step": checkpoint_step_from_revision(args.revision),
            "layer": args.layer,
            "hook_name": f"hidden_states[{args.layer + 1}]",
            "pooling_policy": "response_token_mean",
            "padding_side": "right",
            "prompt_token_count": response_start,
            "full_token_count": response_end,
            "response_token_count": response_end - response_start,
            "response_token_start": response_start,
            "response_token_end": response_end,
            "activation_shape": list(pooled.shape),
            "pooled_shape": list(pooled.shape),
            "activation_path": str(activation_path),
        }
        append_jsonl(index_path, index_row)
        completed_ids.append(rollout_id)
    return completed_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache response-token mean activations for fixed responses.")
    parser.add_argument("--response-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--variant", default="response-token-mean-layer12")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--model-id", default="EleutherAI/pythia-410m-deduped")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--response-separator", default="\n\n")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.layer < 0:
        raise SystemExit("--layer must be non-negative")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.save_every < 1:
        raise SystemExit("--save-every must be positive")

    run_dir = resolve_run_dir(args)
    inputs_dir = run_dir / "inputs"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    activations_dir = results_dir / "activations"
    logs_dir = run_dir / "logs"
    meta_dir = run_dir / "meta"
    index_path = results_dir / "activation_index.jsonl"
    progress_path = checkpoints_dir / "progress.json"
    status_path = meta_dir / "status.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"

    for directory in [inputs_dir, checkpoints_dir, results_dir, activations_dir, logs_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if status_path.exists() and not args.force_completed:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    records = selected_records(load_jsonl(args.response_jsonl), args.limit)
    selected_ids = [str(record["rollout_id"]) for record in records]
    completed_ids = load_completed_ids(index_path).intersection(selected_ids)

    write_status(
        status_path,
        "running",
        "activation caching started",
        {"selected": len(selected_ids), "completed": len(completed_ids)},
    )
    append_log(log_path, "start", {"run_dir": str(run_dir), "selected": len(selected_ids)})
    write_json(inputs_dir / "selected_rollout_ids.json", {"rollout_ids": selected_ids})

    cursor = 0
    progress_bar = make_progress_bar(
        total=len(selected_ids),
        initial=len(completed_ids),
        enabled=not args.no_progress,
    )
    try:
        model, tokenizer = load_model_and_tokenizer(args)
        pending_batch: list[dict[str, Any]] = []
        for cursor, record in enumerate(records, start=1):
            rollout_id = str(record["rollout_id"])
            if rollout_id in completed_ids:
                continue
            pending_batch.append(record)
            if len(pending_batch) < args.batch_size:
                continue
            batch_completed_ids = cache_batch(pending_batch, args, model, tokenizer, activations_dir, index_path)
            completed_ids.update(batch_completed_ids)
            if progress_bar is not None:
                progress_bar.update(len(batch_completed_ids))
                progress_bar.set_postfix(
                    {
                        "batch": len(batch_completed_ids),
                        "done": len(completed_ids),
                    },
                    refresh=True,
                )
            pending_batch = []
            if len(completed_ids) % args.save_every == 0:
                write_progress(progress_path, selected_ids, completed_ids, cursor)
                append_log(log_path, "progress", {"cursor": cursor, "completed": len(completed_ids)})

        if pending_batch:
            batch_completed_ids = cache_batch(pending_batch, args, model, tokenizer, activations_dir, index_path)
            completed_ids.update(batch_completed_ids)
            if progress_bar is not None:
                progress_bar.update(len(batch_completed_ids))
                progress_bar.set_postfix(
                    {
                        "batch": len(batch_completed_ids),
                        "done": len(completed_ids),
                    },
                    refresh=True,
                )
        final_state = "completed" if len(completed_ids) == len(selected_ids) else "failed"
        final_message = "activation caching completed" if final_state == "completed" else "missing activation records"
    except Exception as exc:
        final_state = "failed"
        final_message = f"activation caching failed: {type(exc).__name__}: {exc}"
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
    finally:
        if progress_bar is not None:
            progress_bar.close()

    write_progress(progress_path, selected_ids, completed_ids, cursor)
    write_status(
        status_path,
        final_state,
        final_message,
        {"selected": len(selected_ids), "completed": len(completed_ids)},
    )
    write_manifest(
        manifest_path,
        args,
        run_dir,
        args.response_jsonl,
        index_path,
        selected_count=len(selected_ids),
        completed_count=len(completed_ids),
    )
    append_log(log_path, final_state, {"completed": len(completed_ids), "selected": len(selected_ids)})

    print(
        json.dumps(
            {
                "status": final_state,
                "message": final_message,
                "run_dir": str(run_dir),
                "activation_index_jsonl": str(index_path),
                "selected": len(selected_ids),
                "completed": len(completed_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_state == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
