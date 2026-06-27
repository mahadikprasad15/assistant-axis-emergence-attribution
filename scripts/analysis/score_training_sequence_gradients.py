#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import secrets
from collections import defaultdict
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
    return tqdm(total=total, initial=initial, desc="gradient attribution", unit="sample", dynamic_ncols=True)


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} contains a non-object JSONL row")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    append_jsonl(path, {"event": event, "payload": payload, "time_utc": utc_now()})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "window_id",
        "uid",
        "batch_idx",
        "source_file",
        "loss",
        "local_aa_score",
        "final_aa_score",
        "gradient_norm",
        "update_pressure_norm",
        "gradient_pressure_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


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
        / args.output_variant
        / run_id
    )


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def normalize_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("token_ids must be a list-like value")
    return [int(item) for item in value]


def selected_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    if limit < 1:
        raise ValueError("--limit must be positive when provided")
    return records[:limit]


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


def load_axis_vector(axis_run_dir: Path | None, axis_vector_path: Path | None, repo_root: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    if axis_vector_path is None:
        if axis_run_dir is None:
            raise ValueError("axis run dir or axis vector path is required")
        axis_vector_path = axis_run_dir / "results" / "assistant_axis_vector.pt"
    axis_vector_path = resolve_path(axis_vector_path, repo_root)
    if not axis_vector_path.exists():
        raise FileNotFoundError(f"axis vector not found: {axis_vector_path}")

    vector = torch.load(axis_vector_path, map_location="cpu").float()
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm.item()) <= 0:
        raise ValueError(f"axis vector has invalid norm: {axis_vector_path}")
    vector = vector / norm
    summary: dict[str, Any] = {"vector_path": str(axis_vector_path), "vector_sha256": file_sha256(axis_vector_path)}
    if axis_run_dir is not None:
        summary_path = axis_run_dir / "results" / "assistant_axis_summary.json"
        if summary_path.exists():
            summary["summary"] = load_json(summary_path)
        summary["run_dir"] = str(axis_run_dir)
    return vector, summary


def load_completed_ids(index_path: Path, vectors_dir: Path, require_vectors: bool, selected_ids: set[str]) -> set[str]:
    completed: set[str] = set()
    for row in load_jsonl(index_path):
        sample_id = str(row.get("sample_id"))
        if sample_id not in selected_ids:
            continue
        if require_vectors:
            vector_path = Path(str(row.get("gradient_pressure_path", "")))
            if not vector_path.exists():
                continue
        completed.add(sample_id)
    return completed


def latest_rows_by_sample_id(rows: list[dict[str, Any]], selected_ids: set[str]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id in selected_ids:
            latest[sample_id] = row
    return list(latest.values())


def safe_cosine(a: Any, b: Any) -> float:
    import torch

    value = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0)
    score = float(value.item())
    if not math.isfinite(score):
        raise ValueError("non-finite cosine score")
    return score


def prepare_batch(batch: list[dict[str, Any]], pad_token_id: int, max_input_tokens: int | None) -> dict[str, Any]:
    import torch

    prepared: list[dict[str, Any]] = []
    for sample in batch:
        token_ids = normalize_token_ids(sample["token_ids"])
        if len(token_ids) < 2:
            raise ValueError(f"sample_id={sample['sample_id']} has fewer than 2 token ids")
        if max_input_tokens is not None:
            token_ids = token_ids[: max_input_tokens + 1]
        input_ids = token_ids[:-1]
        targets = token_ids[1:]
        if not input_ids:
            raise ValueError(f"sample_id={sample['sample_id']} has empty model input after truncation")
        prepared.append({"sample": sample, "input_ids": input_ids, "targets": targets})

    max_len = max(len(row["input_ids"]) for row in prepared)
    input_tensor = torch.full((len(prepared), max_len), int(pad_token_id), dtype=torch.long)
    target_tensor = torch.full((len(prepared), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(prepared), max_len), dtype=torch.long)
    valid_mask = torch.zeros((len(prepared), max_len), dtype=torch.bool)

    for row_idx, row in enumerate(prepared):
        length = len(row["input_ids"])
        input_tensor[row_idx, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        target_tensor[row_idx, :length] = torch.tensor(row["targets"], dtype=torch.long)
        attention_mask[row_idx, :length] = 1
        valid_mask[row_idx, :length] = True

    return {
        "samples": [row["sample"] for row in prepared],
        "input_ids": input_tensor,
        "targets": target_tensor,
        "attention_mask": attention_mask,
        "valid_mask": valid_mask,
        "lengths": [len(row["input_ids"]) for row in prepared],
    }


def score_batch(
    batch: list[dict[str, Any]],
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    local_axis: Any,
    final_axis: Any | None,
    run_dir: Path,
    vectors_dir: Path,
) -> list[dict[str, Any]]:
    import torch

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    prepared = prepare_batch(batch, int(pad_token_id), args.max_input_tokens)
    device = next(model.parameters()).device
    input_ids = prepared["input_ids"].to(device)
    targets = prepared["targets"].to(device)
    attention_mask = prepared["attention_mask"].to(device)
    valid_mask = prepared["valid_mask"].to(device)

    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[args.layer + 1]
    hidden.retain_grad()
    logits = outputs.logits
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(targets.shape)
    masked_losses = losses * valid_mask.float()
    valid_counts = valid_mask.sum(dim=1).clamp_min(1)
    per_sample_loss = masked_losses.sum(dim=1) / valid_counts
    loss = masked_losses.sum() / valid_counts.sum().clamp_min(1)
    loss.backward()

    grad = hidden.grad
    if grad is None:
        raise RuntimeError("hidden-state gradient was not retained")
    grad = grad.detach().float().cpu()
    valid_mask_cpu = valid_mask.detach().cpu()
    local_axis = local_axis.cpu()
    final_axis = final_axis.cpu() if final_axis is not None else None

    records: list[dict[str, Any]] = []
    for row_idx, sample in enumerate(prepared["samples"]):
        mask = valid_mask_cpu[row_idx]
        pooled_grad = grad[row_idx, mask, :].mean(dim=0)
        update_pressure = -pooled_grad
        gradient_norm = float(torch.linalg.vector_norm(pooled_grad).item())
        update_norm = float(torch.linalg.vector_norm(update_pressure).item())
        local_score = safe_cosine(update_pressure, local_axis)
        final_score = safe_cosine(update_pressure, final_axis) if final_axis is not None else None
        sample_id = str(sample["sample_id"])
        vector_path: Path | None = None
        if args.save_gradient_vectors:
            vector_path = vectors_dir / f"{sanitize_id(sample_id)}__{sanitize_id(args.revision)}__layer{args.layer:02d}.pt"
            torch.save(update_pressure, vector_path)
        record = {
            "schema_version": "0.1",
            "sample_id": sample_id,
            "window_id": str(sample["window_id"]),
            "uid": str(sample["uid"]),
            "batch_idx": int(sample["batch_idx"]),
            "source_file": str(sample["source_file"]),
            "token_count": int(sample.get("token_count", len(normalize_token_ids(sample["token_ids"])))),
            "model_id": args.model_id,
            "checkpoint_revision": args.revision,
            "layer": args.layer,
            "hook_name": f"hidden_states[{args.layer + 1}]",
            "input_token_count": int(prepared["lengths"][row_idx]),
            "target_token_count": int(prepared["lengths"][row_idx]),
            "loss": float(per_sample_loss[row_idx].detach().cpu().item()),
            "local_aa_score": local_score,
            "final_aa_score": final_score,
            "gradient_norm": gradient_norm,
            "update_pressure_norm": update_norm,
            "gradient_pooling": "token_mean",
            "sign_convention": "positive_score_means_aa_amplifying_pressure",
            "source": {
                "sample_source": sample.get("source", {}),
                "scorer_run_dir": str(run_dir),
            },
        }
        if vector_path is not None:
            record["gradient_pressure_path"] = str(vector_path)
        records.append(record)

    del outputs, hidden, logits, losses, masked_losses, loss
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


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
            "completed_sample_ids": sorted(completed_ids),
        },
    )


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


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[str(row["window_id"])].append(row)

    def stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        return {
            "count": len(values),
            "min": min(values),
            "mean": mean,
            "max": max(values),
            "std": math.sqrt(variance),
        }

    windows = {}
    for window_id, window_rows in sorted(by_window.items()):
        windows[window_id] = {
            "local_aa_score": stats([float(row["local_aa_score"]) for row in window_rows]),
            "final_aa_score": stats([float(row["final_aa_score"]) for row in window_rows if row["final_aa_score"] is not None]),
            "loss": stats([float(row["loss"]) for row in window_rows]),
        }
    top_local = sorted(rows, key=lambda row: float(row["local_aa_score"]), reverse=True)[:20]
    bottom_local = sorted(rows, key=lambda row: float(row["local_aa_score"]))[:20]
    return {
        "schema_version": "0.1",
        "records": len(rows),
        "windows": windows,
        "top_local_aa": [
            {
                "sample_id": row["sample_id"],
                "window_id": row["window_id"],
                "local_aa_score": row["local_aa_score"],
                "loss": row["loss"],
            }
            for row in top_local
        ],
        "bottom_local_aa": [
            {
                "sample_id": row["sample_id"],
                "window_id": row["window_id"],
                "local_aa_score": row["local_aa_score"],
                "loss": row["loss"],
            }
            for row in bottom_local
        ],
    }


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    sample_jsonl: Path,
    local_axis_summary: dict[str, Any],
    final_axis_summary: dict[str, Any] | None,
    score_path: Path,
    completed_count: int,
    selected_count: int,
) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "runner": "TrainingSequenceGradientScorer",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "model_id": args.model_id,
            "checkpoint_revision": args.revision,
            "layer": args.layer,
            "hook_name": f"hidden_states[{args.layer + 1}]",
            "loss": {
                "type": "manual_next_token_cross_entropy",
                "input_rule": "input_ids=token_ids[:-1]",
                "target_rule": "targets=token_ids[1:]",
            },
            "gradient": {
                "object": "dL/dh_layer",
                "pooling": "token_mean",
                "update_pressure": "-mean_tokens(dL/dh_layer)",
                "score": "cosine(update_pressure, assistant_axis)",
                "sign_convention": "positive_score_means_aa_amplifying_pressure",
            },
            "sample_jsonl": {"path": str(sample_jsonl), "sha256": file_sha256(sample_jsonl)},
            "local_axis": local_axis_summary,
            "final_axis": final_axis_summary,
            "selection": {"limit": args.limit, "selected_count": selected_count, "completed_count": completed_count},
            "execution": {
                "batch_size": args.batch_size,
                "save_every": args.save_every,
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
                "max_input_tokens": args.max_input_tokens,
                "save_gradient_vectors": args.save_gradient_vectors,
                "progress_enabled": not args.no_progress,
            },
            "results": {"attribution_scores_jsonl": str(score_path), "sha256": file_sha256(score_path)},
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score packed training sequences by activation-gradient alignment with Assistant Axis.")
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--local-axis-run-dir", type=Path, default=None)
    parser.add_argument("--local-axis-vector", type=Path, default=None)
    parser.add_argument("--final-axis-run-dir", type=Path, default=None)
    parser.add_argument("--final-axis-vector", type=Path, default=None)
    parser.add_argument("--model-id", default="EleutherAI/pythia-410m-deduped")
    parser.add_argument("--revision", default="step512")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--save-gradient-vectors", action="store_true")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force-completed", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="assistant-axis-attribution-v0")
    parser.add_argument("--output-variant", default="gradient-attribution-layer12")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.layer < 0:
        raise SystemExit("--layer must be non-negative")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.save_every < 1:
        raise SystemExit("--save-every must be positive")
    if args.max_input_tokens is not None and args.max_input_tokens < 1:
        raise SystemExit("--max-input-tokens must be positive")

    repo_root = Path(".").resolve()
    sample_jsonl = resolve_path(args.sample_jsonl, repo_root)
    run_dir = resolve_run_dir(args)
    inputs_dir = run_dir / "inputs"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    vectors_dir = results_dir / "gradient_pressure_vectors"
    logs_dir = run_dir / "logs"
    meta_dir = run_dir / "meta"
    score_path = results_dir / "attribution_scores.jsonl"
    csv_path = results_dir / "attribution_scores.csv"
    summary_path = results_dir / "attribution_summary.json"
    progress_path = checkpoints_dir / "progress.json"
    status_path = meta_dir / "status.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"

    for directory in [inputs_dir, checkpoints_dir, results_dir, logs_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    if args.save_gradient_vectors:
        vectors_dir.mkdir(parents=True, exist_ok=True)

    if status_path.exists() and not args.force_completed:
        status = load_json(status_path)
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    samples = selected_records(load_jsonl(sample_jsonl), args.limit)
    if not samples:
        raise SystemExit("no sample records selected")
    selected_ids = [str(sample["sample_id"]) for sample in samples]
    selected_id_set = set(selected_ids)
    if len(selected_id_set) != len(selected_ids):
        raise SystemExit("sample records contain duplicate sample_id values")

    completed_ids = load_completed_ids(score_path, vectors_dir, args.save_gradient_vectors, selected_id_set)
    if args.force_completed:
        completed_ids = set()
    write_status(status_path, "running", "gradient attribution scoring started", {"selected": len(selected_ids), "completed": len(completed_ids)})
    append_log(log_path, "start", {"run_dir": str(run_dir), "selected": len(selected_ids), "completed": len(completed_ids)})
    write_json(inputs_dir / "selected_sample_ids.json", {"sample_ids": selected_ids})

    final_state = "failed"
    final_message = "gradient attribution did not complete"
    cursor = 0
    progress_bar = make_progress_bar(len(selected_ids), len(completed_ids), not args.no_progress)
    try:
        local_axis, local_axis_summary = load_axis_vector(args.local_axis_run_dir, args.local_axis_vector, repo_root)
        final_axis = None
        final_axis_summary = None
        if args.final_axis_run_dir is not None or args.final_axis_vector is not None:
            final_axis, final_axis_summary = load_axis_vector(args.final_axis_run_dir, args.final_axis_vector, repo_root)
        model, tokenizer = load_model_and_tokenizer(args)

        pending_batch: list[dict[str, Any]] = []
        for cursor, sample in enumerate(samples, start=1):
            sample_id = str(sample["sample_id"])
            if sample_id in completed_ids and not args.force_completed:
                continue
            pending_batch.append(sample)
            if len(pending_batch) < args.batch_size:
                continue
            score_rows = score_batch(pending_batch, args, model, tokenizer, local_axis, final_axis, run_dir, vectors_dir)
            for row in score_rows:
                append_jsonl(score_path, row)
                completed_ids.add(str(row["sample_id"]))
            if progress_bar is not None:
                progress_bar.update(len(score_rows))
                progress_bar.set_postfix({"batch": len(score_rows), "done": len(completed_ids)}, refresh=True)
            pending_batch = []
            if len(completed_ids) % args.save_every == 0:
                write_progress(progress_path, selected_ids, completed_ids, cursor)
                append_log(log_path, "progress", {"cursor": cursor, "completed": len(completed_ids)})

        if pending_batch:
            score_rows = score_batch(pending_batch, args, model, tokenizer, local_axis, final_axis, run_dir, vectors_dir)
            for row in score_rows:
                append_jsonl(score_path, row)
                completed_ids.add(str(row["sample_id"]))
            if progress_bar is not None:
                progress_bar.update(len(score_rows))
                progress_bar.set_postfix({"batch": len(score_rows), "done": len(completed_ids)}, refresh=True)

        final_state = "completed" if len(completed_ids) == len(selected_ids) else "failed"
        final_message = "gradient attribution scoring completed" if final_state == "completed" else "missing attribution score records"
    except Exception as exc:
        final_state = "failed"
        final_message = f"gradient attribution scoring failed: {type(exc).__name__}: {exc}"
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
        local_axis_summary = locals().get("local_axis_summary", None)
        final_axis_summary = locals().get("final_axis_summary", None)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    rows = latest_rows_by_sample_id(load_jsonl(score_path), selected_id_set)
    write_csv(csv_path, rows)
    summary = summarize_scores(rows)
    summary.update(
        {
            "run_dir": str(run_dir),
            "sample_jsonl": str(sample_jsonl),
            "attribution_scores_jsonl": str(score_path),
            "attribution_scores_csv": str(csv_path),
            "gradient_pressure_vectors_dir": str(vectors_dir) if args.save_gradient_vectors else None,
            "state": final_state,
            "message": final_message,
        }
    )
    write_json(summary_path, summary)
    write_progress(progress_path, selected_ids, completed_ids, cursor)
    write_status(status_path, final_state, final_message, {"selected": len(selected_ids), "completed": len(completed_ids)})
    write_manifest(
        manifest_path,
        args,
        run_dir,
        sample_jsonl,
        local_axis_summary or {},
        final_axis_summary,
        score_path,
        completed_count=len(completed_ids),
        selected_count=len(selected_ids),
    )
    append_log(log_path, final_state, {"completed": len(completed_ids), "selected": len(selected_ids)})

    print(
        json.dumps(
            {
                "status": final_state,
                "message": final_message,
                "run_dir": str(run_dir),
                "scores": str(score_path),
                "summary": str(summary_path),
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
