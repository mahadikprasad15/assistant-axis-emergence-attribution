#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    append_jsonl(path, {"event": event, "payload": payload, "time_utc": utc_now()})


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def portable_artifact_path(path_text: str, bundle_path: Path, fallback_dir: str) -> Path:
    declared = Path(path_text)
    candidates = [
        declared,
        Path.cwd() / declared if not declared.is_absolute() else declared,
        bundle_path.parent / fallback_dir / declared.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"artifact from target bundle not found: {path_text}")


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_run_dir:
        return args.resume_run_dir
    return (
        args.output_root
        / args.experiment_name
        / args.model_name
        / args.dataset_name
        / args.probe_set
        / args.output_variant
        / (args.run_id or default_run_id())
    )


def torch_dtype_from_name(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(args: argparse.Namespace, model_id: str, revision: str, dtype_name: str) -> Any:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
        local_files_only=args.local_files_only,
        dtype=torch_dtype_from_name(dtype_name),
        device_map=args.device_map,
    )
    model.eval()
    return model


def normalize_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def prepare_batch(batch: list[dict[str, Any]], max_input_tokens: int | None) -> dict[str, Any]:
    import torch

    prepared = []
    for sample in batch:
        token_ids = normalize_token_ids(sample["token_ids"])
        if len(token_ids) < 2:
            raise ValueError(f"sample_id={sample['sample_id']} has fewer than 2 token ids")
        input_ids = token_ids[:-1]
        if max_input_tokens is not None:
            input_ids = input_ids[:max_input_tokens]
        prepared.append((sample, input_ids))

    max_len = max(len(input_ids) for _, input_ids in prepared)
    input_tensor = torch.zeros((len(prepared), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(prepared), max_len), dtype=torch.long)
    for row_idx, (_, input_ids) in enumerate(prepared):
        length = len(input_ids)
        input_tensor[row_idx, :length] = torch.tensor(input_ids, dtype=torch.long)
        attention_mask[row_idx, :length] = 1
    return {
        "samples": [sample for sample, _ in prepared],
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "lengths": [len(input_ids) for _, input_ids in prepared],
    }


def projection_summary(scores: Any, top_k: int) -> dict[str, float | int]:
    import torch

    if scores.numel() == 0 or not torch.isfinite(scores).all():
        raise ValueError("token projections are empty or non-finite")
    k = min(top_k, int(scores.numel()))
    return {
        "mean_raw_projection": float(scores.mean().item()),
        "max_raw_projection": float(scores.max().item()),
        "top_k_mean_raw_projection": float(torch.topk(scores, k=k).values.mean().item()),
        "positive_token_fraction": float((scores > 0).float().mean().item()),
        "top_k_tokens": k,
    }


def score_batch(
    batch: list[dict[str, Any]],
    model: Any,
    layer: int,
    axes: dict[str, Any],
    reference_offsets: dict[str, float] | None,
    top_k: int,
    max_input_tokens: int | None,
    run_dir: Path,
    model_id: str,
    revision: str,
) -> list[dict[str, Any]]:
    import torch

    prepared = prepare_batch(batch, max_input_tokens)
    device = next(model.parameters()).device
    input_ids = prepared["input_ids"].to(device)
    attention_mask = prepared["attention_mask"].to(device)
    captured: dict[str, Any] = {}

    def capture_hidden(_module: Any, _inputs: Any, output: Any) -> None:
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    if layers is None:
        raise AttributeError("could not locate transformer layers on loaded model")
    handle = layers[layer].register_forward_hook(capture_hidden)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("layer hook did not capture hidden states")

    hidden = captured["hidden"].detach().float()
    axis_names = list(axes)
    axis_matrix = torch.stack([axes[name].to(hidden.device).float() for name in axis_names], dim=0)
    projections = torch.einsum("btd,ad->bta", hidden, axis_matrix)
    hidden_norms = torch.linalg.vector_norm(hidden, dim=-1)

    records = []
    for row_idx, sample in enumerate(prepared["samples"]):
        length = prepared["lengths"][row_idx]
        axis_scores: dict[str, dict[str, float | int]] = {}
        centered_scores: dict[str, dict[str, float | int]] = {}
        for axis_idx, axis_name in enumerate(axis_names):
            token_scores = projections[row_idx, :length, axis_idx]
            axis_scores[axis_name] = projection_summary(token_scores, top_k)
            if reference_offsets is not None:
                centered = token_scores - reference_offsets[axis_name]
                values = projection_summary(centered, top_k)
                centered_scores[axis_name] = {
                    "mean_centered_projection": values["mean_raw_projection"],
                    "max_centered_projection": values["max_raw_projection"],
                    "top_k_mean_centered_projection": values["top_k_mean_raw_projection"],
                    "positive_token_fraction": values["positive_token_fraction"],
                    "top_k_tokens": values["top_k_tokens"],
                    "reference_projection": reference_offsets[axis_name],
                }
        records.append(
            {
                "schema_version": "0.1",
                "sample_id": str(sample["sample_id"]),
                "window_id": str(sample["window_id"]),
                "uid": str(sample["uid"]),
                "batch_idx": int(sample["batch_idx"]),
                "source_file": str(sample["source_file"]),
                "model_id": model_id,
                "checkpoint_revision": revision,
                "layer": layer,
                "hook_name": f"layers[{layer}]",
                "valid_token_count": length,
                "token_scope": "all_valid_training_input_tokens",
                "axis_scores": axis_scores,
                "centered_axis_scores": centered_scores if reference_offsets is not None else None,
                "mean_hidden_norm": float(hidden_norms[row_idx, :length].mean().item()),
                "source": {"sample_source": sample.get("source", {}), "scorer_run_dir": str(run_dir)},
            }
        )
    return records


def scalar_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": mean,
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
        "std": math.sqrt(variance),
    }


def summarize(rows: list[dict[str, Any]], axis_names: list[str]) -> dict[str, Any]:
    metrics = [
        "mean_raw_projection",
        "max_raw_projection",
        "top_k_mean_raw_projection",
        "positive_token_fraction",
    ]
    summary = {
        axis_name: {
            metric: scalar_stats([float(row["axis_scores"][axis_name][metric]) for row in rows])
            for metric in metrics
        }
        for axis_name in axis_names
    }
    centered = {}
    if rows and rows[0].get("centered_axis_scores"):
        centered_metrics = [
            "mean_centered_projection",
            "max_centered_projection",
            "top_k_mean_centered_projection",
            "positive_token_fraction",
        ]
        centered = {
            axis_name: {
                metric: scalar_stats(
                    [float(row["centered_axis_scores"][axis_name][metric]) for row in rows]
                )
                for metric in centered_metrics
            }
            for axis_name in axis_names
        }
    return {
        "schema_version": "0.1",
        "records": len(rows),
        "axis_names": axis_names,
        "raw_axis_scores": summary,
        "centered_axis_scores": centered,
        "mean_hidden_norm": scalar_stats([float(row["mean_hidden_norm"]) for row in rows]),
    }


def write_progress(path: Path, selected_ids: list[str], completed_ids: set[str]) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "updated_at_utc": utc_now(),
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
            "counts": counts,
            "updated_at_utc": utc_now(),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score forward-only Vector Filter projections.")
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--target-bundle", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/pythia_410m_concept_attribution_256_512_v0.yaml"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="concept-attribution-256-512-v0")
    parser.add_argument("--output-variant", default="vector-filter-layer12")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--include-diagnostic-targets", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.save_every < 1:
        raise SystemExit("--save-every must be positive")
    repo_root = Path(".").resolve()
    sample_jsonl = resolve_path(args.sample_jsonl, repo_root)
    bundle_path = resolve_path(args.target_bundle, repo_root)
    config_path = resolve_path(args.experiment_config, repo_root)
    config = load_yaml(config_path)
    bundle = load_json(bundle_path)
    if bundle is None:
        raise SystemExit(f"target bundle not found: {bundle_path}")

    model_config = config["model"]
    filter_config = config["vector_filter"]
    model_id = str(model_config["model_id"])
    revision = str(filter_config["checkpoint_revision"])
    layer = int(model_config["layer"])
    dtype_name = args.torch_dtype or str(model_config["torch_dtype"])
    batch_size = args.batch_size or int(filter_config["batch_size"])
    if batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    target_names = list(config["axis_targets"]["primary"])
    if args.include_diagnostic_targets:
        target_names.extend(config["axis_targets"].get("diagnostics", []))
    available_targets = {str(row["axis_name"]): row for row in bundle["targets"]}
    missing_targets = [name for name in target_names if name not in available_targets]
    if missing_targets:
        raise SystemExit(f"target bundle is missing configured targets: {missing_targets}")

    run_dir = resolve_run_dir(args)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    inputs_dir = run_dir / "inputs"
    for directory in [results_dir, meta_dir, checkpoints_dir, logs_dir, inputs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    scores_path = results_dir / "vector_filter_scores.jsonl"
    summary_path = results_dir / "vector_filter_summary.json"
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"

    samples = load_jsonl(sample_jsonl)
    if args.limit is not None:
        samples = samples[: args.limit]
    selected_ids = [str(row["sample_id"]) for row in samples]
    if len(selected_ids) != len(set(selected_ids)):
        raise SystemExit("sample JSONL contains duplicate sample_id values")
    expected_window = str(config["training_window"]["window_id"])
    if {str(row.get("window_id")) for row in samples} != {expected_window}:
        raise SystemExit(f"sample JSONL must contain only window {expected_window}")

    existing = load_jsonl(scores_path)
    completed_ids = {str(row["sample_id"]) for row in existing}
    if len(completed_ids) != len(existing):
        raise SystemExit("existing Vector Filter scores contain duplicate sample ids")
    if status_path.exists() and not args.force_completed:
        status = load_json(status_path) or {}
        if status.get("state") == "completed" and completed_ids == set(selected_ids) and summary_path.exists():
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0
    pending = [row for row in samples if str(row["sample_id"]) not in completed_ids]

    axis_paths = {
        name: portable_artifact_path(str(available_targets[name]["vector_path"]), bundle_path, "vectors")
        for name in target_names
    }
    import torch

    axes = {name: torch.load(path, map_location="cpu").float() for name, path in axis_paths.items()}
    shapes = {tuple(vector.shape) for vector in axes.values()}
    if len(shapes) != 1:
        raise SystemExit(f"target vectors have mixed shapes: {sorted(map(str, shapes))}")
    for name, vector in axes.items():
        if not torch.isfinite(vector).all():
            raise SystemExit(f"target vector is non-finite: {name}")

    reference_offsets: dict[str, float] | None = None
    reference_path: Path | None = None
    if bool(filter_config.get("centered_projection_sensitivity", False)):
        reference = bundle.get("reference_mean")
        if not isinstance(reference, dict) or not reference.get("vector_path"):
            raise SystemExit("centered projection requested but target bundle has no reference_mean")
        reference_path = portable_artifact_path(str(reference["vector_path"]), bundle_path, "means")
        reference_vector = torch.load(reference_path, map_location="cpu").float()
        reference_offsets = {
            name: float(torch.dot(reference_vector, vector).item()) for name, vector in axes.items()
        }

    manifest = {
        "schema_version": "0.1",
        "runner": "VectorFilterRunner",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "inputs": {
            "sample_jsonl": {"path": str(sample_jsonl), "sha256": file_sha256(sample_jsonl)},
            "target_bundle": {"path": str(bundle_path), "sha256": file_sha256(bundle_path)},
            "experiment_config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "axis_vectors": {
                name: {"path": str(path), "sha256": file_sha256(path)} for name, path in axis_paths.items()
            },
            "reference_mean": (
                {"path": str(reference_path), "sha256": file_sha256(reference_path)} if reference_path else None
            ),
        },
        "model": {"model_id": model_id, "revision": revision, "layer": layer, "torch_dtype": dtype_name},
        "execution": {
            "batch_size": batch_size,
            "save_every": args.save_every,
            "selected_records": len(samples),
            "resume_completed_records": len(completed_ids),
            "max_input_tokens": args.max_input_tokens,
            "progress_enabled": not args.no_progress,
        },
        "scoring": {
            "axis_names": target_names,
            "primary_score": filter_config["primary_score"],
            "top_k_tokens": int(filter_config["top_k_tokens"]),
            "token_scope": filter_config["token_scope"],
            "centered_projection_sensitivity": reference_offsets is not None,
            "reference_offsets": reference_offsets,
        },
        "outputs": {"scores": str(scores_path), "summary": str(summary_path)},
    }
    write_json(manifest_path, manifest)
    write_status(
        status_path,
        "running",
        "Vector Filter scoring started",
        {"selected": len(samples), "completed": len(completed_ids)},
    )
    write_progress(progress_path, selected_ids, completed_ids)
    append_log(log_path, "start", {"selected": len(samples), "pending": len(pending), "batch_size": batch_size})

    try:
        model = load_model(args, model_id, revision, dtype_name) if pending else None
        batch_iter = range(0, len(pending), batch_size)
        if not args.no_progress:
            try:
                from tqdm.auto import tqdm

                batch_iter = tqdm(
                    batch_iter,
                    total=math.ceil(len(pending) / batch_size),
                    desc="vector filter",
                    unit="batch",
                    initial=0,
                )
            except ImportError:
                pass
        since_save = 0
        for start in batch_iter:
            batch = pending[start : start + batch_size]
            scored = score_batch(
                batch,
                model,
                layer,
                axes,
                reference_offsets,
                int(filter_config["top_k_tokens"]),
                args.max_input_tokens,
                run_dir,
                model_id,
                revision,
            )
            for row in scored:
                append_jsonl(scores_path, row)
                completed_ids.add(str(row["sample_id"]))
            since_save += len(scored)
            if since_save >= args.save_every:
                write_progress(progress_path, selected_ids, completed_ids)
                write_status(
                    status_path,
                    "running",
                    "Vector Filter scoring in progress",
                    {"selected": len(samples), "completed": len(completed_ids)},
                )
                since_save = 0

        rows = load_jsonl(scores_path)
        by_id = {str(row["sample_id"]): row for row in rows}
        if set(by_id) != set(selected_ids) or len(by_id) != len(rows):
            raise ValueError("final Vector Filter scores do not exactly match selected sample ids")
        ordered_rows = [by_id[sample_id] for sample_id in selected_ids]
        summary = summarize(ordered_rows, target_names)
        summary.update(
            {
                "model_id": model_id,
                "checkpoint_revision": revision,
                "layer": layer,
                "primary_score": filter_config["primary_score"],
                "scores_jsonl": str(scores_path),
            }
        )
        write_json(summary_path, summary)
        write_json(results_dir / "results.json", summary)
        write_progress(progress_path, selected_ids, completed_ids)
        write_status(
            status_path,
            "completed",
            "Vector Filter scoring completed",
            {"selected": len(samples), "completed": len(completed_ids)},
        )
        append_log(log_path, "completed", {"selected": len(samples), "completed": len(completed_ids)})
        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "scores": str(scores_path),
                    "summary": str(summary_path),
                    "selected": len(samples),
                    "completed": len(completed_ids),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        write_progress(progress_path, selected_ids, completed_ids)
        write_status(
            status_path,
            "failed",
            f"Vector Filter scoring failed: {type(exc).__name__}: {exc}",
            {"selected": len(samples), "completed": len(completed_ids)},
        )
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
        print(
            json.dumps(
                {
                    "status": "failed",
                    "run_dir": str(run_dir),
                    "message": f"{type(exc).__name__}: {exc}",
                    "selected": len(samples),
                    "completed": len(completed_ids),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
