#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("sample_id"):
            raise ValueError(f"invalid record at {path}:{line_number}")
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        rows[sample_id] = row
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value for {label}: {value!r}")
    return result


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_norm = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_norm = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_norm * y_norm) if x_norm > 0 and y_norm > 0 else None


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in ordered[cursor:end]:
            result[index] = average_rank
        cursor = end
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_interval(
    xs: list[float],
    ys: list[float],
    statistic: Callable[[list[float], list[float]], float | None],
    samples: int,
    rng: random.Random,
) -> dict[str, float] | None:
    estimates = []
    for _ in range(samples):
        indices = [rng.randrange(len(xs)) for _ in xs]
        estimate = statistic([xs[i] for i in indices], [ys[i] for i in indices])
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None
    return {"low": percentile(estimates, 0.025), "high": percentile(estimates, 0.975)}


def top_overlap(xs: list[float], ys: list[float], top_k: int) -> dict[str, Any]:
    left = set(sorted(range(len(xs)), key=xs.__getitem__, reverse=True)[:top_k])
    right = set(sorted(range(len(ys)), key=ys.__getitem__, reverse=True)[:top_k])
    overlap = len(left & right)
    denominator = len(left | right)
    population = len(xs)
    tail_probability = sum(
        math.comb(top_k, k) * math.comb(population - top_k, top_k - k)
        for k in range(overlap, top_k + 1)
        if top_k - k <= population - top_k
    ) / math.comb(population, top_k)
    return {
        "count": overlap,
        "fraction": overlap / top_k,
        "jaccard": overlap / denominator if denominator else 1.0,
        "random_expected_count": top_k * top_k / population,
        "hypergeometric_tail_p_uncorrected": tail_probability,
    }


def comparison(
    xs: list[float], ys: list[float], top_k: int, bootstrap_samples: int, rng: random.Random
) -> dict[str, Any]:
    return {
        "pearson": pearson(xs, ys),
        "pearson_bootstrap_95ci": bootstrap_interval(xs, ys, pearson, bootstrap_samples, rng),
        "spearman": spearman(xs, ys),
        "spearman_bootstrap_95ci": bootstrap_interval(xs, ys, spearman, bootstrap_samples, rng),
        "sign_agreement": sum((x >= 0) == (y >= 0) for x, y in zip(xs, ys)) / len(xs),
        "top_overlap": top_overlap(xs, ys, top_k),
    }


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
        "positive_fraction": sum(value > 0 for value in values) / len(values),
    }


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze normalized agreement across Vector Filter, Activation Gradient, and FOPCI.")
    parser.add_argument("--vector-run-dir", type=Path, required=True)
    parser.add_argument("--activation-run-dir", type=Path, required=True)
    parser.add_argument("--fopci-run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--allow-partial-overlap", action="store_true")
    parser.add_argument("--force-completed", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="concept-attribution-256-512-v0")
    parser.add_argument("--output-variant", default="three-method-normalized-analysis")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_samples < 1 or args.top_k < 1:
        raise SystemExit("--bootstrap-samples and --top-k must be positive")
    run_id = args.run_id or default_run_id()
    run_dir = (
        args.output_root / args.experiment_name / args.model_name / args.dataset_name /
        args.probe_set / args.output_variant / run_id
    ).resolve()
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    inputs_dir = run_dir / "inputs"
    for directory in [results_dir, meta_dir, checkpoints_dir, logs_dir, inputs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    status_path = meta_dir / "status.json"
    summary_path = results_dir / "results.json"
    if status_path.exists() and summary_path.exists() and not args.force_completed:
        if read_json(status_path).get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0
    write_json(status_path, {"schema_version": "0.1", "state": "running", "updated_at_utc": utc_now()})

    source_paths = {
        "vector": args.vector_run_dir.resolve() / "results" / "vector_filter_scores.jsonl",
        "activation": args.activation_run_dir.resolve() / "results" / "attribution_scores.jsonl",
        "fopci": args.fopci_run_dir.resolve() / "results" / "fopci_scores.jsonl",
    }
    try:
        sources = {name: read_jsonl_by_id(path) for name, path in source_paths.items()}
        id_sets = {name: set(rows) for name, rows in sources.items()}
        shared_ids = set.intersection(*id_sets.values())
        if not args.allow_partial_overlap and any(ids != shared_ids for ids in id_sets.values()):
            raise ValueError(f"source sample sets differ: { {name: len(ids) for name, ids in id_sets.items()} }")
        sample_ids = sorted(shared_ids)
        if not sample_ids:
            raise ValueError("the three runs have no shared sample IDs")
        if args.top_k > len(sample_ids):
            raise ValueError(f"top-k {args.top_k} exceeds shared records {len(sample_ids)}")
        vector, activation, fopci = sources["vector"], sources["activation"], sources["fopci"]
        axes = sorted(
            set(vector[sample_ids[0]]["axis_scores"])
            & set(activation[sample_ids[0]]["axis_metrics"])
            & set(fopci[sample_ids[0]]["axis_scores"])
        )
        if not axes:
            raise ValueError("the three runs have no shared axes")
        rng = random.Random(args.seed)
        rows: list[dict[str, Any]] = []
        axis_summaries: dict[str, Any] = {}
        for axis in axes:
            metrics: dict[str, list[float]] = {
                "vector_raw": [], "vector_centered": [], "vector_centered_normalized": [],
                "activation_dot": [], "activation_cosine": [], "fopci_dot": [], "fopci_cosine": [],
                "mean_hidden_norm": [], "activation_gradient_norm": [], "fopci_gradient_norm": [],
                "loss": [],
            }
            for sample_id in sample_ids:
                v, a, f = vector[sample_id], activation[sample_id], fopci[sample_id]
                hidden_norm = finite_float(v["mean_hidden_norm"], "mean_hidden_norm")
                centered = finite_float(v["centered_axis_scores"][axis]["mean_centered_projection"], "vector_centered")
                values = {
                    "vector_raw": finite_float(v["axis_scores"][axis]["mean_raw_projection"], "vector_raw"),
                    "vector_centered": centered,
                    "vector_centered_normalized": centered / hidden_norm,
                    "activation_dot": finite_float(a["axis_metrics"][axis]["dot"], "activation_dot"),
                    "activation_cosine": finite_float(a["axis_metrics"][axis]["cosine"], "activation_cosine"),
                    "fopci_dot": finite_float(f["axis_scores"][axis]["negative_gradient_dot"], "fopci_dot"),
                    "fopci_cosine": finite_float(f["axis_scores"][axis]["gradient_cosine"], "fopci_cosine"),
                    "mean_hidden_norm": hidden_norm,
                    "activation_gradient_norm": finite_float(a["update_pressure_norm"], "activation_gradient_norm"),
                    "fopci_gradient_norm": finite_float(f["sequence_gradient_norm"], "fopci_gradient_norm"),
                    "loss": finite_float(f["loss"], "loss"),
                }
                for name, value in values.items():
                    metrics[name].append(value)
                rows.append({"sample_id": sample_id, "axis_name": axis, **values})
            primary_pairs = [
                ("vector_centered_normalized", "activation_cosine"),
                ("vector_centered_normalized", "fopci_cosine"),
                ("activation_cosine", "fopci_cosine"),
            ]
            raw_pairs = [
                ("vector_centered", "activation_dot"),
                ("vector_centered", "fopci_dot"),
                ("activation_dot", "fopci_dot"),
            ]
            confound_pairs = [
                ("vector_centered", "mean_hidden_norm"),
                ("activation_dot", "activation_gradient_norm"),
                ("fopci_dot", "fopci_gradient_norm"),
                ("activation_dot", "loss"),
                ("fopci_dot", "loss"),
            ]
            axis_summaries[axis] = {
                "distributions": {name: distribution(values) for name, values in metrics.items()},
                "normalized_comparisons": {
                    f"{left}__vs__{right}": comparison(
                        metrics[left], metrics[right], args.top_k, args.bootstrap_samples, rng
                    ) for left, right in primary_pairs
                },
                "raw_comparisons": {
                    f"{left}__vs__{right}": comparison(
                        metrics[left], metrics[right], args.top_k, args.bootstrap_samples, rng
                    ) for left, right in raw_pairs
                },
                "confound_correlations": {
                    f"{left}__vs__{right}": {
                        "pearson": pearson(metrics[left], metrics[right]),
                        "spearman": spearman(metrics[left], metrics[right]),
                    } for left, right in confound_pairs
                },
                "top_sequences": {
                    name: [sample_ids[i] for i in sorted(
                        range(len(sample_ids)), key=values.__getitem__, reverse=True
                    )[:args.top_k]]
                    for name, values in metrics.items()
                    if name in {"vector_centered_normalized", "activation_cosine", "fopci_cosine"}
                },
            }
        summary = {
            "schema_version": "0.1", "records": len(sample_ids), "axes": axis_summaries,
            "primary_metrics": {
                "vector": "mean_centered_projection / mean_hidden_norm",
                "activation": "update_pressure_cosine",
                "fopci": "parameter_gradient_cosine",
            },
            "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed, "interval": "percentile_95"},
            "top_k": args.top_k,
            "caveats": [
                "Hypergeometric tail probabilities are exploratory and uncorrected for multiple comparisons.",
                "Bootstrap intervals quantify sampling instability within this 50-sequence smoke sample only.",
            ],
        }
        with (results_dir / "normalized_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(summary_path, summary)
        input_manifest = {
            name: {"path": str(path), "sha256": file_sha256(path), "records": len(sources[name])}
            for name, path in source_paths.items()
        }
        write_json(inputs_dir / "source_runs.json", input_manifest)
        write_json(meta_dir / "run_manifest.json", {
            "schema_version": "0.1", "runner": "ThreeMethodNormalizedAttributionAnalyzer",
            "created_at_utc": utc_now(), "git_commit": git_commit(), "run_dir": str(run_dir),
            "parameters": {"top_k": args.top_k, "bootstrap_samples": args.bootstrap_samples, "seed": args.seed},
            "inputs": input_manifest,
            "outputs": {"summary": str(summary_path), "scores_csv": str(results_dir / "normalized_scores.csv")},
        })
        write_json(checkpoints_dir / "progress.json", {
            "schema_version": "0.1", "state": "completed", "completed_records": len(sample_ids),
            "sample_ids": sample_ids, "updated_at_utc": utc_now(),
        })
        write_json(status_path, {
            "schema_version": "0.1", "state": "completed", "records": len(sample_ids),
            "updated_at_utc": utc_now(),
        })
        (logs_dir / "run.log").write_text(
            json.dumps({"time_utc": utc_now(), "event": "completed", "records": len(sample_ids)}) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "completed", "run_dir": str(run_dir), "records": len(sample_ids), "summary": str(summary_path)}, indent=2))
        return 0
    except Exception as exc:
        write_json(status_path, {
            "schema_version": "0.1", "state": "failed", "updated_at_utc": utc_now(),
            "message": f"{type(exc).__name__}: {exc}",
        })
        (logs_dir / "run.log").write_text(
            json.dumps({"time_utc": utc_now(), "event": "failed", "message": str(exc)}) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
