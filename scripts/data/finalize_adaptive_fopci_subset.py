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
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("sample_id"):
            raise ValueError(f"invalid record at {path}:{line_number}")
        rows.append(row)
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample IDs in {path}")
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(seed: int, namespace: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode()).hexdigest()


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda sample_id: (values[sample_id], sample_id))
    count = len(ordered)
    return {
        sample_id: (rank / (count - 1) if count > 1 else 0.5)
        for rank, sample_id in enumerate(ordered)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize the adaptive 250 plus fixed random 250 FOPCI subset.")
    parser.add_argument("--subset-run-dir", type=Path, required=True)
    parser.add_argument("--vector-run-dir", type=Path, required=True)
    parser.add_argument("--activation-run-dir", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiments/pythia_410m_concept_attribution_256_512_v0.yaml"))
    parser.add_argument("--force-completed", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="concept-attribution-256-512-v0")
    parser.add_argument("--output-variant", default="fopci-adaptive-subset")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = (
        args.output_root / args.experiment_name / args.model_name / args.dataset_name /
        args.probe_set / args.output_variant / (args.run_id or default_run_id())
    ).resolve()
    results_dir, meta_dir = run_dir / "results", run_dir / "meta"
    checkpoints_dir, logs_dir, inputs_dir = run_dir / "checkpoints", run_dir / "logs", run_dir / "inputs"
    for directory in [results_dir, meta_dir, checkpoints_dir, logs_dir, inputs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    status_path, summary_path = meta_dir / "status.json", results_dir / "results.json"
    if status_path.exists() and summary_path.exists() and not args.force_completed:
        if load_json(status_path).get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0
    write_json(status_path, {"schema_version": "0.1", "state": "running", "updated_at_utc": utc_now()})

    config_path = args.experiment_config.resolve()
    config = load_yaml(config_path)
    policy = config["sampling"]
    seed = int(policy["seed"])
    axes = list(config["axis_targets"]["primary"])
    stratum_counts = {str(name): int(count) for name, count in policy["adaptive_strata"].items()}
    expected_adaptive = int(policy["fopci_adaptive_size"])
    expected_random = int(policy["fopci_random_size"])
    expected_total = int(policy["fopci_sample_size"])
    if sum(stratum_counts.values()) != expected_adaptive:
        raise ValueError("adaptive stratum counts do not sum to fopci_adaptive_size")

    subset_results = args.subset_run_dir.resolve() / "results"
    activation_samples_path = subset_results / "activation_gradient_sequences.jsonl"
    random_samples_path = subset_results / "fopci_random_sequences.jsonl"
    vector_scores_path = args.vector_run_dir.resolve() / "results" / "vector_filter_scores.jsonl"
    activation_scores_path = args.activation_run_dir.resolve() / "results" / "attribution_scores.jsonl"
    activation_samples = load_jsonl(activation_samples_path)
    random_samples = load_jsonl(random_samples_path)
    vector_rows = load_jsonl(vector_scores_path)
    activation_rows = load_jsonl(activation_scores_path)
    sample_by_id = {str(row["sample_id"]): row for row in activation_samples}
    random_ids = {str(row["sample_id"]) for row in random_samples}
    vector_by_id = {str(row["sample_id"]): row for row in vector_rows}
    activation_by_id = {str(row["sample_id"]): row for row in activation_rows}
    activation_ids = set(sample_by_id)
    if len(random_ids) != expected_random or not random_ids <= activation_ids:
        raise ValueError("fixed random subset count or containment is invalid")
    if not activation_ids <= set(vector_by_id):
        raise ValueError(f"Vector Filter missing {len(activation_ids - set(vector_by_id))} activation-subset records")
    if set(activation_by_id) != activation_ids:
        raise ValueError("Activation Gradient score IDs do not exactly match the frozen activation subset")

    # Compute per-axis ranks, then aggregate each method across all primary targets.
    vector_ranks: dict[str, dict[str, float]] = {}
    activation_ranks: dict[str, dict[str, float]] = {}
    for axis in axes:
        vector_values = {
            sample_id: finite(vector_by_id[sample_id]["axis_scores"][axis]["mean_raw_projection"], f"vector {axis}")
            for sample_id in activation_ids
        }
        activation_values = {
            sample_id: finite(activation_by_id[sample_id]["axis_metrics"][axis]["dot"], f"activation {axis}")
            for sample_id in activation_ids
        }
        vector_ranks[axis] = percentile_ranks(vector_values)
        activation_ranks[axis] = percentile_ranks(activation_values)
    vector_level = {sample_id: sum(vector_ranks[axis][sample_id] for axis in axes) / len(axes) for sample_id in activation_ids}
    activation_level = {sample_id: sum(activation_ranks[axis][sample_id] for axis in axes) / len(axes) for sample_id in activation_ids}
    available = activation_ids - random_ids
    selected: dict[str, str] = {}
    selection_scores: dict[str, dict[str, float]] = {}

    endpoint, final = "endpoint_step512", "final_step143000"
    scoring = {
        "concordant_high": lambda sid: min(vector_level[sid], activation_level[sid]),
        "concordant_low": lambda sid: -max(vector_level[sid], activation_level[sid]),
        "vector_filter_high_activation_gradient_low": lambda sid: vector_level[sid] - activation_level[sid],
        "activation_gradient_high_vector_filter_low": lambda sid: activation_level[sid] - vector_level[sid],
        "endpoint_final_target_disagreement": lambda sid: (
            abs(vector_ranks[endpoint][sid] - vector_ranks[final][sid])
            + abs(activation_ranks[endpoint][sid] - activation_ranks[final][sid])
        ),
        "near_zero_control": lambda sid: -sum(
            abs(vector_ranks[axis][sid] - 0.5) + abs(activation_ranks[axis][sid] - 0.5)
            for axis in axes
        ),
    }
    if set(stratum_counts) != set(scoring):
        raise ValueError(f"unsupported adaptive strata: {sorted(set(stratum_counts) - set(scoring))}")
    for stratum, count in stratum_counts.items():
        candidates = [sample_id for sample_id in available if sample_id not in selected]
        ranked = sorted(
            candidates,
            key=lambda sid: (-scoring[stratum](sid), stable_hash(seed, stratum, sid), sid),
        )
        chosen = ranked[:count]
        if len(chosen) != count:
            raise ValueError(f"insufficient candidates for {stratum}: {len(chosen)} < {count}")
        for sample_id in chosen:
            selected[sample_id] = stratum
            selection_scores[sample_id] = {
                "stratum_score": scoring[stratum](sample_id),
                "vector_level_percentile": vector_level[sample_id],
                "activation_level_percentile": activation_level[sample_id],
            }
    if len(selected) != expected_adaptive or set(selected) & random_ids:
        raise ValueError("adaptive selection count or disjointness invariant failed")

    adaptive_rows = [sample_by_id[sample_id] for sample_id in sorted(selected)]
    combined_ids = random_ids | set(selected)
    combined_rows = [sample_by_id[sample_id] for sample_id in sorted(combined_ids)]
    if len(combined_rows) != expected_total:
        raise ValueError(f"combined FOPCI count {len(combined_rows)} != {expected_total}")
    membership = []
    for sample_id in sorted(combined_ids):
        kind = "random" if sample_id in random_ids else "adaptive"
        stratum = "preregistered_random" if kind == "random" else selected[sample_id]
        membership.append({
            "schema_version": "0.1", "sample_id": sample_id,
            "window_id": str(sample_by_id[sample_id]["window_id"]),
            "fopci_subset_kind": kind, "fopci_stratum": stratum,
            "adaptive_selection_status": "finalized", "selection_seed": seed,
            "stable_hash": stable_hash(seed, stratum, sample_id),
            "selection_scores": selection_scores.get(sample_id),
        })
    write_jsonl(results_dir / "adaptive_sequences.jsonl", adaptive_rows)
    write_jsonl(results_dir / "combined_fopci_sequences.jsonl", combined_rows)
    write_jsonl(results_dir / "fopci_membership.jsonl", membership)
    actual_counts = {name: sum(value == name for value in selected.values()) for name in stratum_counts}
    summary = {
        "schema_version": "0.1", "selection_stage": "adaptive_finalized", "seed": seed,
        "counts": {"random": len(random_ids), "adaptive": len(selected), "combined": len(combined_ids), "adaptive_strata": actual_counts},
        "selection_policy": {
            "method_level": "mean percentile rank across primary axes",
            "concordant_high": "largest minimum of Vector Filter and Activation Gradient method levels",
            "concordant_low": "smallest maximum of the two method levels",
            "cross_method_disagreement": "largest signed method-level percentile difference",
            "endpoint_final_target_disagreement": "largest summed endpoint-vs-final percentile gap",
            "near_zero_control": "smallest total distance from median percentile across methods and axes",
            "stratum_order": list(stratum_counts),
            "random_subset_excluded_before_adaptive_selection": True,
        },
        "outputs": {
            "adaptive_sequences": str(results_dir / "adaptive_sequences.jsonl"),
            "combined_fopci_sequences": str(results_dir / "combined_fopci_sequences.jsonl"),
            "membership": str(results_dir / "fopci_membership.jsonl"),
        },
    }
    write_json(summary_path, summary)
    input_manifest = {
        "experiment_config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "activation_samples": {"path": str(activation_samples_path), "sha256": file_sha256(activation_samples_path)},
        "random_samples": {"path": str(random_samples_path), "sha256": file_sha256(random_samples_path)},
        "vector_scores": {"path": str(vector_scores_path), "sha256": file_sha256(vector_scores_path)},
        "activation_scores": {"path": str(activation_scores_path), "sha256": file_sha256(activation_scores_path)},
    }
    write_json(inputs_dir / "source_artifacts.json", input_manifest)
    write_json(meta_dir / "run_manifest.json", {
        "schema_version": "0.1", "builder": "AdaptiveFOPCISubsetFinalizer", "created_at_utc": utc_now(),
        "run_dir": str(run_dir), "inputs": input_manifest, "selection_policy": summary["selection_policy"],
        "outputs": summary["outputs"], "validation": {"passed": True, "counts": summary["counts"]},
    })
    write_json(checkpoints_dir / "progress.json", {
        "schema_version": "0.1", "state": "completed", "selected_ids": sorted(combined_ids), "updated_at_utc": utc_now(),
    })
    write_json(status_path, {
        "schema_version": "0.1", "state": "completed", "counts": summary["counts"], "updated_at_utc": utc_now(),
    })
    (logs_dir / "run.log").write_text(json.dumps({"time_utc": utc_now(), "event": "completed", "counts": summary["counts"]}) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "run_dir": str(run_dir), **summary["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
