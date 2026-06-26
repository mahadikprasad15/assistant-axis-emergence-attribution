#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from collections import Counter
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


def resolve_run_dir(args: argparse.Namespace, variant_id: str) -> Path:
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


def resolve_artifact_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def select_variant(config: dict[str, Any], variant_id: str) -> dict[str, Any]:
    variants = config.get("axis_construction_variants", [])
    for variant in variants:
        if variant.get("variant_id") == variant_id:
            return variant
    known = [variant.get("variant_id") for variant in variants]
    raise ValueError(f"unknown axis variant {variant_id!r}; known variants: {known}")


def validate_single_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("activation index contains no rows")
    fields = ["model_id", "checkpoint_revision", "layer", "pooling_policy"]
    context: dict[str, Any] = {}
    for field in fields:
        values = {row.get(field) for row in rows}
        if len(values) != 1:
            raise ValueError(f"activation index mixes multiple {field} values: {sorted(map(str, values))}")
        context[field] = next(iter(values))
    return context


def selected_rows(rows: list[dict[str, Any]], variant: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    default_prompt_ids = set(map(str, variant.get("default_prompt_ids", [])))
    contrast_role_groups = set(map(str, variant.get("contrast_role_groups", [])))
    default_rows = [
        row
        for row in rows
        if row.get("record_type") == "default" and str(row.get("default_prompt_id")) in default_prompt_ids
    ]
    contrast_rows = [
        row
        for row in rows
        if row.get("record_type") == "role" and str(row.get("role_group")) in contrast_role_groups
    ]
    return default_rows, contrast_rows


def load_vectors(rows: list[dict[str, Any]], repo_root: Path) -> tuple[Any, list[str]]:
    import torch

    vectors = []
    rollout_ids: list[str] = []
    missing_paths: list[str] = []
    for row in rows:
        activation_path = resolve_artifact_path(str(row.get("activation_path", "")), repo_root)
        if not activation_path.exists():
            missing_paths.append(str(row.get("activation_path", "")))
            continue
        vector = torch.load(activation_path, map_location="cpu")
        vectors.append(vector.float())
        rollout_ids.append(str(row["rollout_id"]))
    if missing_paths:
        raise FileNotFoundError(f"missing activation tensor paths: {missing_paths[:10]}")
    if not vectors:
        raise ValueError("no activation vectors selected")
    shapes = {tuple(vector.shape) for vector in vectors}
    if len(shapes) != 1:
        raise ValueError(f"selected activation vectors have mixed shapes: {sorted(map(str, shapes))}")
    return torch.stack(vectors, dim=0), rollout_ids


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


def write_progress(path: Path, state: str, completed_steps: list[str]) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "updated_at_utc": utc_now(),
            "state": state,
            "completed_steps": completed_steps,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an Assistant Axis vector from an activation run.")
    parser.add_argument("--activation-run-dir", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiments/pythia_410m_mvp_v0.yaml"))
    parser.add_argument("--axis-variant-id", default="aa_main")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--output-variant", default="aa-main-layer12")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = Path(".").resolve()
    activation_run_dir = args.activation_run_dir
    if not activation_run_dir.is_absolute():
        activation_run_dir = repo_root / activation_run_dir
    activation_index_path = activation_run_dir / "results" / "activation_index.jsonl"
    activation_manifest_path = activation_run_dir / "meta" / "run_manifest.json"

    config = load_yaml(args.experiment_config)
    variant = select_variant(config, args.axis_variant_id)
    run_dir = resolve_run_dir(args, args.axis_variant_id)
    inputs_dir = run_dir / "inputs"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    meta_dir = run_dir / "meta"
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"

    for directory in [inputs_dir, checkpoints_dir, results_dir, logs_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if status_path.exists() and not args.force_completed:
        status = load_json(status_path) or {}
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    completed_steps: list[str] = []
    write_status(status_path, "running", "assistant axis build started", {"default_records": 0, "contrast_records": 0})
    append_log(log_path, "start", {"run_dir": str(run_dir), "activation_run_dir": str(activation_run_dir)})

    try:
        rows = load_jsonl(activation_index_path)
        context = validate_single_context(rows)
        completed_steps.append("loaded_activation_index")
        default_rows, contrast_rows = selected_rows(rows, variant)
        if not default_rows:
            raise ValueError(f"no default activation rows selected for {variant.get('default_prompt_ids')}")
        if not contrast_rows:
            raise ValueError(f"no contrast activation rows selected for {variant.get('contrast_role_groups')}")
        completed_steps.append("selected_rows")

        default_vectors, default_rollout_ids = load_vectors(default_rows, repo_root)
        contrast_vectors, contrast_rollout_ids = load_vectors(contrast_rows, repo_root)
        completed_steps.append("loaded_vectors")

        import torch

        default_mean = default_vectors.mean(dim=0)
        contrast_mean = contrast_vectors.mean(dim=0)
        raw_axis = default_mean - contrast_mean
        raw_norm = float(torch.linalg.vector_norm(raw_axis).item())
        if raw_norm <= 0:
            raise ValueError("axis norm is zero before normalization")
        axis = raw_axis / raw_norm
        completed_steps.append("computed_axis")

        axis_path = results_dir / "assistant_axis_vector.pt"
        default_mean_path = results_dir / "default_mean.pt"
        contrast_mean_path = results_dir / "contrast_mean.pt"
        summary_path = results_dir / "assistant_axis_summary.json"
        torch.save(axis, axis_path)
        torch.save(default_mean, default_mean_path)
        torch.save(contrast_mean, contrast_mean_path)

        summary = {
            "schema_version": "0.1",
            "axis_id": f"{args.axis_variant_id}__{context['checkpoint_revision']}__layer{context['layer']}",
            "variant_id": args.axis_variant_id,
            "model_id": context["model_id"],
            "checkpoint_revision": context["checkpoint_revision"],
            "layer": context["layer"],
            "pooling_policy": context["pooling_policy"],
            "vector_path": str(axis_path),
            "default_mean_path": str(default_mean_path),
            "contrast_mean_path": str(contrast_mean_path),
            "vector_shape": list(axis.shape),
            "default_prompt_ids": variant.get("default_prompt_ids", []),
            "contrast_role_groups": variant.get("contrast_role_groups", []),
            "counts": {
                "default_records": len(default_rows),
                "contrast_records": len(contrast_rows),
            },
            "norm": raw_norm,
            "source": {
                "activation_run_dir": str(activation_run_dir),
                "activation_index_jsonl": str(activation_index_path),
                "experiment_config": str(args.experiment_config),
                "default_rollout_ids": default_rollout_ids,
                "contrast_rollout_ids": contrast_rollout_ids,
            },
            "breakdowns": {
                "default_prompt_id": dict(Counter(str(row.get("default_prompt_id")) for row in default_rows)),
                "contrast_role_group": dict(Counter(str(row.get("role_group")) for row in contrast_rows)),
                "contrast_role_id": dict(Counter(str(row.get("role_id")) for row in contrast_rows)),
            },
        }
        write_json(summary_path, summary)
        completed_steps.append("wrote_outputs")

        activation_manifest = load_json(activation_manifest_path)
        manifest = {
            "schema_version": "0.1",
            "builder": "AssistantAxisBuilder",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "experiment_config": {
                "path": str(args.experiment_config),
                "sha256": file_sha256(args.experiment_config),
            },
            "activation_run": {
                "run_dir": str(activation_run_dir),
                "activation_index_jsonl": str(activation_index_path),
                "activation_index_sha256": file_sha256(activation_index_path),
                "run_manifest": activation_manifest,
            },
            "variant": {
                "variant_id": args.axis_variant_id,
                "default_prompt_ids": variant.get("default_prompt_ids", []),
                "contrast_role_groups": variant.get("contrast_role_groups", []),
                "diagnostic_role_groups": variant.get("diagnostic_role_groups", []),
            },
            "outputs": {
                "assistant_axis_vector": str(axis_path),
                "default_mean": str(default_mean_path),
                "contrast_mean": str(contrast_mean_path),
                "summary": str(summary_path),
            },
            "validation": {
                "passed": True,
                "errors": [],
                "warnings": [],
            },
        }
        write_json(manifest_path, manifest)
        write_progress(progress_path, "completed", completed_steps)
        write_status(
            status_path,
            "completed",
            "assistant axis build completed",
            {"default_records": len(default_rows), "contrast_records": len(contrast_rows)},
        )
        append_log(log_path, "completed", {"default_records": len(default_rows), "contrast_records": len(contrast_rows)})
        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "summary": str(summary_path),
                    "axis_vector": str(axis_path),
                    "default_records": len(default_rows),
                    "contrast_records": len(contrast_rows),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        write_progress(progress_path, "failed", completed_steps)
        write_status(status_path, "failed", f"assistant axis build failed: {type(exc).__name__}: {exc}", {"default_records": 0, "contrast_records": 0})
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
        print(
            json.dumps(
                {
                    "status": "failed",
                    "run_dir": str(run_dir),
                    "message": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
