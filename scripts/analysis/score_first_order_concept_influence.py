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

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_no} in {path} must be a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def append_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    append_jsonl(path, {"time_utc": utc_now(), "event": event, "payload": payload})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def portable_bundle_path(path_text: str, bundle_path: Path, fallback_dir: str) -> Path:
    declared = Path(path_text)
    candidates = [declared]
    if not declared.is_absolute():
        candidates.extend([Path.cwd() / declared, bundle_path.parent / fallback_dir / declared.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"bundle artifact not found: {path_text}")


def torch_dtype_from_name(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError("token_ids must be a non-empty list")
    return [int(token_id) for token_id in value]


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


def parameter_layer_index(name: str) -> int | None:
    match = re.search(r"(?:gpt_neox\.layers|layers)\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def resolve_parameter_scope(
    model: Any,
    scope: str,
    layer: int,
    every_nth_layer: int,
) -> tuple[list[str], list[Any], dict[str, Any]]:
    candidates = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    layer_indices = [value for name, _ in candidates if (value := parameter_layer_index(name)) is not None]
    max_layer = max(layer_indices) if layer_indices else layer
    selected = []
    for name, parameter in candidates:
        parameter_layer = parameter_layer_index(name)
        keep = scope == "all_parameters"
        if scope == "layer12_only":
            keep = parameter_layer == layer
        elif scope == "upper_half_layers":
            keep = parameter_layer is not None and parameter_layer >= (max_layer + 1) // 2
        elif scope == "every_nth_layer":
            keep = parameter_layer is not None and parameter_layer % every_nth_layer == 0
        if keep:
            selected.append((name, parameter))
    if not selected:
        raise ValueError(f"parameter scope selected no parameters: {scope}")
    names = [name for name, _ in selected]
    parameters = [parameter for _, parameter in selected]
    scope_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    summary = {
        "parameter_scope_id": scope,
        "parameter_names": names,
        "parameter_tensor_count": len(parameters),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "scope_hash": scope_hash,
        "layer": layer,
        "every_nth_layer": every_nth_layer if scope == "every_nth_layer" else None,
    }
    return names, parameters, summary


def load_target_bundle(bundle_path: Path, target_names: list[str]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    import torch

    bundle = load_json(bundle_path)
    target_rows = {str(row["axis_name"]): row for row in bundle["targets"]}
    missing = [name for name in target_names if name not in target_rows]
    if missing:
        raise ValueError(f"target bundle is missing targets: {missing}")
    targets = {}
    target_sources = {}
    for name in target_names:
        path = portable_bundle_path(str(target_rows[name]["vector_path"]), bundle_path, "vectors")
        vector = torch.load(path, map_location="cpu").float().detach()
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm) or float(norm.item()) <= 0:
            raise ValueError(f"invalid target vector: {name}")
        targets[name] = vector / norm
        target_sources[name] = {"path": str(path), "sha256": file_sha256(path)}
    evaluation_path = portable_bundle_path(
        str(bundle["evaluation_records_jsonl"]), bundle_path, "."
    )
    return targets, {"bundle": bundle, "targets": target_sources}, evaluation_path


def response_projection(
    record: dict[str, Any],
    model: Any,
    tokenizer: Any,
    layer: int,
    targets: dict[str, Any],
    response_separator: str,
    max_eval_tokens: int,
) -> dict[str, Any]:
    import torch

    prefix = str(record["prompt_text"]) + response_separator
    full_text = prefix + str(record["generated_response"]).strip()
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_eval_tokens)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    response_start = len(prefix_ids)
    if response_start >= encoded["input_ids"].shape[1]:
        raise ValueError(
            f"max eval token limit removed the response span: {record['rollout_id']}"
        )
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[layer + 1][0, response_start:, :]
    if hidden.shape[0] < 1:
        raise ValueError(f"empty response-token span: {record['rollout_id']}")
    pooled = hidden.mean(dim=0)
    return {name: torch.dot(pooled.float(), target.to(pooled.device)) for name, target in targets.items()}


def gradients_for_scalar(scalar: Any, parameters: list[Any], retain_graph: bool) -> list[Any]:
    import torch

    gradients = torch.autograd.grad(
        scalar,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    return [torch.zeros_like(parameter) if gradient is None else gradient for parameter, gradient in zip(parameters, gradients)]


def build_query_gradients(
    evaluation_records: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    parameters: list[Any],
    layer: int,
    targets: dict[str, Any],
    response_separator: str,
    max_eval_tokens: int,
    progress_callback: Any | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    import torch

    defaults = [row for row in evaluation_records if row.get("record_type") == "default"]
    contrasts = [row for row in evaluation_records if row.get("record_type") == "role"]
    if not defaults or not contrasts:
        raise ValueError("evaluation split must contain default and role records")
    accumulators = {
        name: [torch.zeros_like(parameter, device="cpu", dtype=torch.float32) for parameter in parameters]
        for name in targets
    }
    for index, record in enumerate(defaults + contrasts, start=1):
        sign_weight = 1.0 / len(defaults) if record.get("record_type") == "default" else -1.0 / len(contrasts)
        projections = response_projection(
            record, model, tokenizer, layer, targets, response_separator, max_eval_tokens
        )
        names = list(targets)
        for target_index, name in enumerate(names):
            gradients = gradients_for_scalar(
                projections[name] * sign_weight,
                parameters,
                retain_graph=target_index < len(names) - 1,
            )
            for parameter_index, gradient in enumerate(gradients):
                accumulators[name][parameter_index].add_(gradient.detach().float().cpu())
        model.zero_grad(set_to_none=True)
        if progress_callback:
            progress_callback(index, len(defaults) + len(contrasts))
    norms = {}
    for name, gradients in accumulators.items():
        squared = sum(float(torch.sum(gradient.double() ** 2).item()) for gradient in gradients)
        norm = math.sqrt(squared)
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"query gradient is non-finite or zero: {name}")
        norms[name] = norm
    return accumulators, {"default_records": len(defaults), "contrast_records": len(contrasts), "query_gradient_norms": norms}


def save_query_bundle(
    path: Path,
    gradients: dict[str, list[Any]],
    parameter_names: list[str],
    scope_summary: dict[str, Any],
    source: dict[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "0.1",
            "parameter_names": parameter_names,
            "scope": scope_summary,
            "gradients": gradients,
            "source": source,
        },
        path,
    )


def load_query_bundle(path: Path, parameter_names: list[str], scope_hash: str) -> dict[str, list[Any]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if payload.get("parameter_names") != parameter_names:
        raise ValueError("cached query gradient parameter names do not match current scope")
    if payload.get("scope", {}).get("scope_hash") != scope_hash:
        raise ValueError("cached query gradient scope hash does not match current scope")
    return payload["gradients"]


def sequence_loss(sample: dict[str, Any], model: Any, max_input_tokens: int) -> Any:
    import torch

    token_ids = normalize_token_ids(sample["token_ids"])[: max_input_tokens + 1]
    if len(token_ids) < 2:
        raise ValueError(f"sample has fewer than two tokens: {sample['sample_id']}")
    device = next(model.parameters()).device
    inputs = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)
    outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    return torch.nn.functional.cross_entropy(
        outputs.logits.reshape(-1, outputs.logits.shape[-1]), targets.reshape(-1), reduction="mean"
    )


def score_sequence(
    sample: dict[str, Any],
    model: Any,
    parameters: list[Any],
    query_gradients: dict[str, list[Any]],
    query_norms: dict[str, float],
    scope_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    loss = sequence_loss(sample, model, args.max_input_tokens)
    sequence_gradients = gradients_for_scalar(loss, parameters, retain_graph=False)
    sequence_squared_norm = sum(float(torch.sum(gradient.detach().double() ** 2).item()) for gradient in sequence_gradients)
    sequence_norm = math.sqrt(sequence_squared_norm)
    if not math.isfinite(sequence_norm) or sequence_norm <= 0:
        raise ValueError(f"sequence gradient is non-finite or zero: {sample['sample_id']}")
    axis_scores = {}
    for name, query_parts in query_gradients.items():
        dot = 0.0
        for gradient, query in zip(sequence_gradients, query_parts):
            dot += float(torch.sum(gradient.detach().float() * query.to(gradient.device)).item())
        negative_dot = -dot
        cosine = negative_dot / (sequence_norm * query_norms[name])
        if not math.isfinite(negative_dot) or not math.isfinite(cosine):
            raise ValueError(f"non-finite FOPCI score: {sample['sample_id']} target={name}")
        axis_scores[name] = {
            "query_gradient_norm": query_norms[name],
            "negative_gradient_dot": negative_dot,
            "gradient_cosine": cosine,
        }
    model.zero_grad(set_to_none=True)
    return {
        "schema_version": "0.1",
        "sample_id": str(sample["sample_id"]),
        "window_id": str(sample["window_id"]),
        "uid": str(sample["uid"]),
        "batch_idx": int(sample["batch_idx"]),
        "checkpoint_revision": args.revision,
        "parameter_scope_id": scope_summary["parameter_scope_id"],
        "parameter_count": scope_summary["parameter_count"],
        "loss": float(loss.detach().cpu().item()),
        "sequence_gradient_norm": sequence_norm,
        "axis_scores": axis_scores,
        "subset_kind": str(sample.get("subset_kind", "random")),
        "subset_stratum": str(sample.get("subset_stratum", "preregistered_random")),
        "curvature": "identity",
        "primary_score": "negative_parameter_gradient_dot",
        "torch_dtype": args.torch_dtype,
        "source": {
            "source_file": sample.get("source_file"),
            "parameter_scope_hash": scope_summary["scope_hash"],
            "parameter_tensor_count": scope_summary["parameter_tensor_count"],
        },
    }


def completed_ids(path: Path) -> set[str]:
    result = set()
    for row in load_jsonl(path):
        if row.get("schema_version") == "0.1" and row.get("axis_scores"):
            result.add(str(row["sample_id"]))
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_axis: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, scores in row["axis_scores"].items():
            by_axis[name].append(float(scores["negative_gradient_dot"]))
    axes = {}
    for name, values in sorted(by_axis.items()):
        mean = sum(values) / len(values)
        axes[name] = {
            "count": len(values),
            "min": min(values),
            "mean": mean,
            "max": max(values),
            "positive_fraction": sum(value > 0 for value in values) / len(values),
        }
    return {"schema_version": "0.1", "records": len(rows), "axes": axes}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    axis_names = sorted({name for row in rows for name in row.get("axis_scores", {})})
    fields = ["sample_id", "window_id", "batch_idx", "subset_kind", "subset_stratum", "loss", "sequence_gradient_norm"]
    fields.extend(f"{name}__negative_gradient_dot" for name in axis_names)
    fields.extend(f"{name}__gradient_cosine" for name in axis_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row.get(field) for field in fields}
            for name in axis_names:
                flat[f"{name}__negative_gradient_dot"] = row["axis_scores"][name]["negative_gradient_dot"]
                flat[f"{name}__gradient_cosine"] = row["axis_scores"][name]["gradient_cosine"]
            writer.writerow(flat)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score first-order parameter-space concept influence (FOPCI).")
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--target-bundle", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiments/pythia_410m_concept_attribution_256_512_v0.yaml"))
    parser.add_argument("--parameter-scope", choices=["all_parameters", "layer12_only", "upper_half_layers", "every_nth_layer"], default="layer12_only")
    parser.add_argument("--every-nth-layer", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-eval-tokens", type=int, default=512)
    parser.add_argument("--response-separator", default="\n\n")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["float32"], default="float32")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rebuild-query-gradient", action="store_true")
    parser.add_argument("--force-completed", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="concept-attribution-256-512-v0")
    parser.add_argument("--output-variant", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path.cwd().resolve()
    config_path = resolve_path(args.experiment_config, repo_root)
    config = load_yaml(config_path)
    args.model_id = args.model_id or str(config["model"]["model_id"])
    args.revision = args.revision or str(config["fopci"]["checkpoint_revision"])
    args.layer = args.layer if args.layer is not None else int(config["model"]["layer"])
    if args.every_nth_layer < 1 or args.save_every < 1:
        raise SystemExit("--every-nth-layer and --save-every must be positive")
    sample_path = resolve_path(args.sample_jsonl, repo_root)
    bundle_path = resolve_path(args.target_bundle, repo_root)
    output_variant = args.output_variant or f"fopci-{args.parameter_scope}"
    run_dir = args.resume_run_dir or (
        args.output_root / args.experiment_name / args.model_name / args.dataset_name /
        args.probe_set / output_variant / (args.run_id or default_run_id())
    )
    run_dir = resolve_path(run_dir, repo_root)
    inputs_dir = run_dir / "inputs"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    logs_dir = run_dir / "logs"
    for directory in [inputs_dir, checkpoints_dir, results_dir, meta_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    score_path = results_dir / "fopci_scores.jsonl"
    query_path = results_dir / "query_gradient_bundle.pt"
    query_summary_path = results_dir / "query_gradient_summary.json"
    summary_path = results_dir / "results.json"
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"
    if status_path.exists() and not args.force_completed:
        status = load_json(status_path)
        if status.get("state") == "completed" and summary_path.exists():
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    write_json(status_path, {"schema_version": "0.1", "state": "running", "updated_at_utc": utc_now()})
    append_log(log_path, "start", {"run_dir": str(run_dir)})
    final_state = "failed"
    try:
        samples = load_jsonl(sample_path)
        if args.limit is not None:
            samples = samples[: args.limit]
        if not samples:
            raise ValueError("no sequence samples selected")
        sample_ids = [str(row["sample_id"]) for row in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample JSONL contains duplicate sample IDs")
        target_names = list(config["axis_targets"]["primary"])
        targets, target_source, evaluation_path = load_target_bundle(bundle_path, target_names)
        evaluation_records = load_jsonl(evaluation_path)
        construction_ids = set(target_source["bundle"]["construction_question_ids"])
        if construction_ids & {int(row["question_id"]) for row in evaluation_records}:
            raise ValueError("evaluation records overlap target construction questions")
        model, tokenizer = load_model_and_tokenizer(args)
        parameter_names, parameters, scope_summary = resolve_parameter_scope(
            model, args.parameter_scope, args.layer, args.every_nth_layer
        )
        if query_path.exists() and not args.rebuild_query_gradient:
            query_gradients = load_query_bundle(query_path, parameter_names, scope_summary["scope_hash"])
            query_norms = {
                name: math.sqrt(sum(float((part.double() ** 2).sum().item()) for part in parts))
                for name, parts in query_gradients.items()
            }
            query_summary = load_json(query_summary_path)
            append_log(log_path, "query_gradient_resumed", {"path": str(query_path)})
        else:
            query_gradients, query_summary = build_query_gradients(
                evaluation_records, model, tokenizer, parameters, args.layer, targets,
                args.response_separator, args.max_eval_tokens,
                lambda done, total: append_log(log_path, "query_gradient_progress", {"completed": done, "total": total}),
            )
            query_norms = query_summary["query_gradient_norms"]
            query_source = {
                "target_bundle": str(bundle_path),
                "target_bundle_sha256": file_sha256(bundle_path),
                "evaluation_records": str(evaluation_path),
                "evaluation_records_sha256": file_sha256(evaluation_path),
                "checkpoint_revision": args.revision,
                "layer": args.layer,
            }
            save_query_bundle(query_path, query_gradients, parameter_names, scope_summary, query_source)
            query_summary.update({"path": str(query_path), "sha256": file_sha256(query_path), "source": query_source})
            write_json(query_summary_path, query_summary)
        done = completed_ids(score_path)
        selected_ids = set(sample_ids)
        for index, sample in enumerate(samples, start=1):
            sample_id = str(sample["sample_id"])
            if sample_id in done and not args.force_completed:
                continue
            row = score_sequence(sample, model, parameters, query_gradients, query_norms, scope_summary, args)
            append_jsonl(score_path, row)
            done.add(sample_id)
            if len(done) % args.save_every == 0:
                write_json(progress_path, {
                    "schema_version": "0.1", "state": "running", "cursor": index,
                    "selected_count": len(samples), "completed_count": len(done & selected_ids),
                    "completed_sample_ids": sorted(done & selected_ids), "updated_at_utc": utc_now(),
                })
                append_log(log_path, "progress", {"cursor": index, "completed": len(done & selected_ids)})
        rows_by_id = {str(row["sample_id"]): row for row in load_jsonl(score_path) if str(row["sample_id"]) in selected_ids}
        rows = [rows_by_id[sample_id] for sample_id in sample_ids if sample_id in rows_by_id]
        write_csv(results_dir / "fopci_scores.csv", rows)
        summary = summarize(rows)
        summary.update({"run_dir": str(run_dir), "parameter_scope": scope_summary, "query_gradient_summary": query_summary})
        write_json(summary_path, summary)
        complete = len(rows) == len(samples)
        final_state = "completed" if complete else "failed"
        write_json(progress_path, {
            "schema_version": "0.1", "state": final_state, "selected_count": len(samples),
            "completed_count": len(rows), "completed_sample_ids": sorted(rows_by_id), "updated_at_utc": utc_now(),
        })
        write_json(manifest_path, {
            "schema_version": "0.1", "runner": "FirstOrderConceptInfluenceRunner", "created_at_utc": utc_now(),
            "run_dir": str(run_dir), "model_id": args.model_id, "checkpoint_revision": args.revision,
            "layer": args.layer, "torch_dtype": args.torch_dtype, "curvature": "identity",
            "parameter_scope": scope_summary,
            "inputs": {
                "sample_jsonl": {"path": str(sample_path), "sha256": file_sha256(sample_path)},
                "target_bundle": {"path": str(bundle_path), "sha256": file_sha256(bundle_path)},
                "evaluation_records": {"path": str(evaluation_path), "sha256": file_sha256(evaluation_path)},
            },
            "targets": target_source["targets"],
            "outputs": {"query_gradient_bundle": str(query_path), "scores": str(score_path), "summary": str(summary_path)},
            "selection": {"limit": args.limit, "selected_count": len(samples), "completed_count": len(rows)},
        })
        write_json(status_path, {"schema_version": "0.1", "state": final_state, "updated_at_utc": utc_now(), "counts": {"selected": len(samples), "completed": len(rows)}})
        append_log(log_path, final_state, {"selected": len(samples), "completed": len(rows)})
        print(json.dumps({"status": final_state, "run_dir": str(run_dir), "records": len(rows)}, indent=2))
        return 0 if complete else 2
    except Exception as exc:
        write_json(status_path, {"schema_version": "0.1", "state": "failed", "updated_at_utc": utc_now(), "message": f"{type(exc).__name__}: {exc}"})
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
        print(json.dumps({"status": "failed", "run_dir": str(run_dir), "message": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
