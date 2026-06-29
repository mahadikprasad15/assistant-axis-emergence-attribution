#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


STAGES = [
    "preflight",
    "vector_filter",
    "activation_gradient_batch1",
    "activation_gradient_batch8",
    "activation_gradient_comparison",
    "fopci_layer12",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": utc_now(), "event": event, "payload": payload}, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_run_dir(output_root: Path, probe_set: str, variant: str, run_id: str) -> Path:
    return (
        output_root
        / "assistant_axis_attribution"
        / "pythia-410m-deduped"
        / "pile-deduped-pythia-preshuffled"
        / probe_set
        / variant
        / run_id
    )


def validate_inputs(
    sample_path: Path,
    bundle_path: Path,
    config_path: Path,
    pilot_size: int,
) -> dict[str, Any]:
    missing = [str(path) for path in [sample_path, bundle_path, config_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("missing pilot inputs: " + ", ".join(missing))
    config = load_yaml(config_path)
    bundle = load_json(bundle_path)
    samples = load_jsonl(sample_path)
    if len(samples) < pilot_size:
        raise ValueError(f"pilot needs {pilot_size} records, but sample contains {len(samples)}")
    selected = samples[:pilot_size]
    sample_ids = [str(row.get("sample_id", "")) for row in selected]
    if any(not sample_id for sample_id in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("pilot records require unique, non-empty sample IDs")
    required_fields = {"sample_id", "window_id", "uid", "batch_idx", "token_ids"}
    for row in selected:
        missing_fields = sorted(required_fields - set(row))
        if missing_fields:
            raise ValueError(f"sample {row.get('sample_id')} is missing fields: {missing_fields}")
        if str(row["window_id"]) != str(config["training_window"]["window_id"]):
            raise ValueError(f"unexpected window for sample {row['sample_id']}: {row['window_id']}")
    target_names = {str(row["axis_name"]) for row in bundle.get("targets", [])}
    required_targets = set(config["axis_targets"]["primary"])
    if not required_targets <= target_names:
        raise ValueError(f"target bundle is missing: {sorted(required_targets - target_names)}")
    construction = set(bundle["construction_question_ids"])
    evaluation = set(bundle["evaluation_question_ids"])
    if construction & evaluation:
        raise ValueError("target bundle construction and evaluation questions overlap")
    evaluation_path = Path(str(bundle["evaluation_records_jsonl"]))
    if not evaluation_path.exists():
        candidate = bundle_path.parent / evaluation_path.name
        if candidate.exists():
            evaluation_path = candidate
        else:
            raise FileNotFoundError(f"evaluation records not found: {bundle['evaluation_records_jsonl']}")
    return {
        "pilot_size": pilot_size,
        "sample_ids": sample_ids,
        "sample_jsonl": {"path": str(sample_path), "sha256": file_sha256(sample_path)},
        "target_bundle": {"path": str(bundle_path), "sha256": file_sha256(bundle_path)},
        "evaluation_records": {"path": str(evaluation_path), "sha256": file_sha256(evaluation_path)},
        "experiment_config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "primary_targets": sorted(required_targets),
        "checkpoint_revision": str(config["model"]["scoring_revision"]),
        "torch_dtype": "float32",
    }


def run_command(command: list[str], log_path: Path, stage: str, dry_run: bool) -> None:
    append_log(log_path, "stage_command", {"stage": stage, "command": command, "dry_run": dry_run})
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"stage {stage} failed with exit code {completed.returncode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the shared three-method concept-attribution pilot.")
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--target-bundle", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/pythia_410m_concept_attribution_256_512_v0.yaml"),
    )
    parser.add_argument("--pilot-size", type=int, default=10)
    parser.add_argument("--vector-filter-batch-size", type=int, default=10)
    parser.add_argument("--activation-candidate-batch-size", type=int, default=8)
    parser.add_argument("--fopci-parameter-scope", default="layer12_only", choices=["layer12_only", "all_parameters", "upper_half_layers", "every_nth_layer"])
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--probe-set", default="concept-attribution-256-512-v0")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.pilot_size < 2 or args.activation_candidate_batch_size < 2:
        raise SystemExit("pilot size and activation candidate batch size must be at least 2")
    repo_root = Path.cwd().resolve()
    sample_path = args.sample_jsonl.resolve()
    bundle_path = args.target_bundle.resolve()
    config_path = args.experiment_config.resolve()
    output_root = args.output_root.resolve()
    run_id = args.run_id or default_run_id()
    run_dir = args.resume_run_dir.resolve() if args.resume_run_dir else child_run_dir(
        output_root, args.probe_set, "three-method-pilot", run_id
    )
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    inputs_dir = run_dir / "inputs"
    for directory in [results_dir, meta_dir, checkpoints_dir, logs_dir, inputs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    results_path = results_dir / "results.json"
    log_path = logs_dir / "run.log"
    if status_path.exists() and not args.force_completed:
        status = load_json(status_path)
        if status.get("state") == "completed" and results_path.exists():
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0
    completed_stages = []
    write_json(status_path, {"schema_version": "0.1", "state": "running", "updated_at_utc": utc_now()})
    try:
        inputs = validate_inputs(sample_path, bundle_path, config_path, args.pilot_size)
        completed_stages.append("preflight")
        write_json(inputs_dir / "pilot_inputs.json", inputs)
        common = ["--sample-jsonl", str(sample_path), "--target-bundle", str(bundle_path), "--experiment-config", str(config_path), "--output-root", str(output_root), "--probe-set", args.probe_set, "--device-map", args.device_map, "--limit", str(args.pilot_size)]
        cache_args = ["--hf-cache-dir", str(args.hf_cache_dir)] if args.hf_cache_dir else []
        local_args = ["--local-files-only"] if args.local_files_only else []
        vector_run_id = f"{run_id}-vector"
        activation_one_id = f"{run_id}-activation-b1"
        activation_many_id = f"{run_id}-activation-b{args.activation_candidate_batch_size}"
        comparison_id = f"{run_id}-activation-comparison"
        fopci_id = f"{run_id}-fopci-{args.fopci_parameter_scope}"
        commands = {
            "vector_filter": [sys.executable, "scripts/analysis/score_vector_filter.py", *common, "--batch-size", str(args.vector_filter_batch_size), "--torch-dtype", "float32", "--run-id", vector_run_id, *cache_args, *local_args],
            "activation_gradient_batch1": [sys.executable, "scripts/analysis/score_training_sequence_gradients.py", *common, "--batch-size", "1", "--torch-dtype", "float32", "--output-variant", "activation-gradient-pilot-batch1", "--run-id", activation_one_id, *cache_args, *local_args],
            "activation_gradient_batch8": [sys.executable, "scripts/analysis/score_training_sequence_gradients.py", *common, "--batch-size", str(args.activation_candidate_batch_size), "--torch-dtype", "float32", "--output-variant", f"activation-gradient-pilot-batch{args.activation_candidate_batch_size}", "--run-id", activation_many_id, *cache_args, *local_args],
        }
        activation_one_dir = child_run_dir(output_root, args.probe_set, "activation-gradient-pilot-batch1", activation_one_id)
        activation_many_dir = child_run_dir(output_root, args.probe_set, f"activation-gradient-pilot-batch{args.activation_candidate_batch_size}", activation_many_id)
        commands["activation_gradient_comparison"] = [
            sys.executable, "scripts/analysis/compare_gradient_attribution_runs.py",
            "--reference-run-dir", str(activation_one_dir), "--candidate-run-dir", str(activation_many_dir),
            "--score-type", "dot", "--max-absolute-delta", "1e-6", "--output-root", str(output_root),
            "--probe-set", args.probe_set, "--output-variant", "activation-gradient-pilot-comparison", "--run-id", comparison_id,
        ]
        commands["fopci_layer12"] = [
            sys.executable, "scripts/analysis/score_first_order_concept_influence.py", *common,
            "--parameter-scope", args.fopci_parameter_scope, "--torch-dtype", "float32",
            "--run-id", fopci_id, *cache_args, *local_args,
        ]
        child_runs = {
            "vector_filter": child_run_dir(output_root, args.probe_set, "vector-filter-layer12", vector_run_id),
            "activation_gradient_batch1": activation_one_dir,
            "activation_gradient_batch8": activation_many_dir,
            "activation_gradient_comparison": child_run_dir(output_root, args.probe_set, "activation-gradient-pilot-comparison", comparison_id),
            "fopci_layer12": child_run_dir(output_root, args.probe_set, f"fopci-{args.fopci_parameter_scope}", fopci_id),
        }
        write_json(manifest_path, {
            "schema_version": "0.1", "runner": "ConceptAttributionPilotOrchestrator",
            "created_at_utc": utc_now(), "run_dir": str(run_dir), "run_id": run_id,
            "pilot_size": args.pilot_size, "inputs": inputs,
            "child_runs": {name: str(path) for name, path in child_runs.items()}, "commands": commands,
        })
        for stage in STAGES[1:]:
            child_status = child_runs[stage] / "meta" / "status.json"
            if child_status.exists() and load_json(child_status).get("state") == "completed" and not args.force_completed:
                append_log(log_path, "stage_resumed", {"stage": stage, "run_dir": str(child_runs[stage])})
            else:
                run_command(commands[stage], log_path, stage, args.dry_run)
            completed_stages.append(stage)
            write_json(progress_path, {"schema_version": "0.1", "state": "running", "completed_stages": completed_stages, "updated_at_utc": utc_now()})
        state = "dry_run_completed" if args.dry_run else "completed"
        result = {"schema_version": "0.1", "state": state, "pilot_size": args.pilot_size, "completed_stages": completed_stages, "child_runs": {name: str(path) for name, path in child_runs.items()}}
        write_json(results_path, result)
        write_json(status_path, {"schema_version": "0.1", "state": state, "updated_at_utc": utc_now()})
        write_json(progress_path, {"schema_version": "0.1", "state": state, "completed_stages": completed_stages, "updated_at_utc": utc_now()})
        append_log(log_path, state, {"completed_stages": completed_stages})
        print(json.dumps({"status": state, "run_dir": str(run_dir), "child_runs": result["child_runs"]}, indent=2))
        return 0
    except Exception as exc:
        write_json(status_path, {"schema_version": "0.1", "state": "failed", "message": f"{type(exc).__name__}: {exc}", "updated_at_utc": utc_now()})
        write_json(progress_path, {"schema_version": "0.1", "state": "failed", "completed_stages": completed_stages, "updated_at_utc": utc_now()})
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
        print(json.dumps({"status": "failed", "run_dir": str(run_dir), "message": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
