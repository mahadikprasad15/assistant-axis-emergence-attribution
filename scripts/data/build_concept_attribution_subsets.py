#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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


def selection_hash(seed: int, namespace: str, sample_id: str) -> str:
    value = f"{seed}:{namespace}:{sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


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


def validate_samples(records: list[dict[str, Any]], expected_window: str) -> None:
    if not records:
        raise ValueError("sample JSONL is empty")
    sample_ids = [str(record.get("sample_id", "")) for record in records]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("every sampled sequence must have a non-empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample JSONL contains duplicate sample_id values")
    windows = {str(record.get("window_id")) for record in records}
    if windows != {expected_window}:
        raise ValueError(f"expected only window {expected_window}, found {sorted(windows)}")
    for record in records:
        if "token_ids" not in record or not record["token_ids"]:
            raise ValueError(f"sample has no token_ids: {record['sample_id']}")


def select_by_hash(
    records: list[dict[str, Any]], count: int, seed: int, namespace: str
) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(f"cannot select {count} records from population of {len(records)}")
    ranked = sorted(
        records,
        key=lambda record: (
            selection_hash(seed, namespace, str(record["sample_id"])),
            str(record["sample_id"]),
        ),
    )
    return ranked[:count]


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


def write_progress(path: Path, state: str, completed_steps: list[str]) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "state": state,
            "completed_steps": completed_steps,
            "updated_at_utc": utc_now(),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stable nested concept-attribution subsets.")
    parser.add_argument("--sample-jsonl", type=Path, required=True)
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
    parser.add_argument("--output-variant", default="concept-attribution-subsets")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = Path(".").resolve()
    sample_jsonl = resolve_path(args.sample_jsonl, repo_root)
    config_path = resolve_path(args.experiment_config, repo_root)
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

    membership_path = results_dir / "subset_membership.jsonl"
    master_path = results_dir / "master_sequences.jsonl"
    activation_path = results_dir / "activation_gradient_sequences.jsonl"
    fopci_random_path = results_dir / "fopci_random_sequences.jsonl"
    summary_path = results_dir / "subset_summary.json"
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"

    if status_path.exists() and not args.force_completed:
        status = load_json(status_path) or {}
        required = [membership_path, master_path, activation_path, fopci_random_path, summary_path]
        if status.get("state") == "completed" and all(path.exists() for path in required):
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    completed_steps: list[str] = []
    counts = {"master": 0, "activation_gradient": 0, "fopci_random": 0}
    write_status(status_path, "running", "concept attribution subset build started", counts)
    append_log(log_path, "start", {"sample_jsonl": str(sample_jsonl), "run_dir": str(run_dir)})

    try:
        config = load_yaml(config_path)
        policy = config["sampling"]
        window_id = str(config["training_window"]["window_id"])
        seed = int(policy["seed"])
        master_size = int(policy["master_sample_size"])
        activation_size = int(policy["activation_gradient_sample_size"])
        fopci_random_size = int(policy["fopci_random_size"])
        if not (0 < fopci_random_size <= activation_size <= master_size):
            raise ValueError("expected fopci_random_size <= activation_size <= master_size")
        completed_steps.append("loaded_config")

        candidates = load_jsonl(sample_jsonl)
        validate_samples(candidates, window_id)
        completed_steps.append("validated_samples")

        master = select_by_hash(candidates, master_size, seed, "master")
        activation = select_by_hash(master, activation_size, seed, "activation_gradient")
        fopci_random = select_by_hash(activation, fopci_random_size, seed, "fopci_random")
        master_ids = {str(record["sample_id"]) for record in master}
        activation_ids = {str(record["sample_id"]) for record in activation}
        fopci_random_ids = {str(record["sample_id"]) for record in fopci_random}
        if not fopci_random_ids <= activation_ids <= master_ids:
            raise ValueError("nested subset invariant failed")
        completed_steps.append("selected_nested_subsets")

        membership = []
        for record in sorted(master, key=lambda row: str(row["sample_id"])):
            sample_id = str(record["sample_id"])
            is_activation = sample_id in activation_ids
            is_fopci_random = sample_id in fopci_random_ids
            membership.append(
                {
                    "schema_version": "0.1",
                    "sample_id": sample_id,
                    "window_id": window_id,
                    "master_sample_member": True,
                    "vector_filter_member": True,
                    "activation_gradient_member": is_activation,
                    "fopci_member": is_fopci_random,
                    "fopci_subset_kind": "random" if is_fopci_random else None,
                    "fopci_stratum": "preregistered_random" if is_fopci_random else None,
                    "adaptive_selection_status": "pending",
                    "selection_seed": seed,
                    "stable_hash": selection_hash(seed, "membership", sample_id),
                    "source": {
                        "sample_jsonl": str(sample_jsonl),
                        "sample_jsonl_sha256": file_sha256(sample_jsonl),
                        "experiment_config": str(config_path),
                    },
                }
            )

        write_jsonl(master_path, sorted(master, key=lambda row: str(row["sample_id"])))
        write_jsonl(activation_path, sorted(activation, key=lambda row: str(row["sample_id"])))
        write_jsonl(fopci_random_path, sorted(fopci_random, key=lambda row: str(row["sample_id"])))
        write_jsonl(membership_path, membership)
        completed_steps.append("wrote_subsets")

        counts = {
            "candidate_records": len(candidates),
            "master": len(master),
            "vector_filter": len(master),
            "activation_gradient": len(activation),
            "fopci_random": len(fopci_random),
            "fopci_adaptive": 0,
            "fopci_total_current": len(fopci_random),
            "fopci_total_planned": int(policy["fopci_sample_size"]),
        }
        summary = {
            "schema_version": "0.1",
            "selection_stage": "preregistered_random_subsets",
            "adaptive_selection_status": "pending_cheaper_method_scores",
            "window_id": window_id,
            "seed": seed,
            "stable_key": policy["stable_key"],
            "counts": counts,
            "nested_invariants": {
                "fopci_random_within_activation_gradient": fopci_random_ids <= activation_ids,
                "activation_gradient_within_master": activation_ids <= master_ids,
                "vector_filter_equals_master": True,
            },
            "outputs": {
                "membership_jsonl": str(membership_path),
                "master_sequences_jsonl": str(master_path),
                "activation_gradient_sequences_jsonl": str(activation_path),
                "fopci_random_sequences_jsonl": str(fopci_random_path),
            },
        }
        write_json(summary_path, summary)
        write_json(results_dir / "results.json", summary)
        completed_steps.append("validated_outputs")

        manifest = {
            "schema_version": "0.1",
            "builder": "ConceptAttributionSubsetBuilder",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "inputs": {
                "sample_jsonl": {"path": str(sample_jsonl), "sha256": file_sha256(sample_jsonl)},
                "experiment_config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            },
            "selection": {
                "algorithm": "sha256_rank",
                "seed": seed,
                "namespaces": ["master", "activation_gradient", "fopci_random"],
                "adaptive_selection_status": "pending",
            },
            "outputs": summary["outputs"],
            "validation": {"passed": True, "counts": counts, **summary["nested_invariants"]},
        }
        write_json(manifest_path, manifest)
        write_progress(progress_path, "completed", completed_steps)
        write_status(status_path, "completed", "concept attribution subset build completed", counts)
        append_log(log_path, "completed", counts)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "summary": str(summary_path),
                    "master": len(master),
                    "activation_gradient": len(activation),
                    "fopci_random": len(fopci_random),
                    "fopci_adaptive_status": "pending",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        write_progress(progress_path, "failed", completed_steps)
        write_status(
            status_path,
            "failed",
            f"concept attribution subset build failed: {type(exc).__name__}: {exc}",
            counts,
        )
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
        print(
            json.dumps(
                {"status": "failed", "run_dir": str(run_dir), "message": f"{type(exc).__name__}: {exc}"},
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
