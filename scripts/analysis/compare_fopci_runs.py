#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_norm = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_norm = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_norm * y_norm) if x_norm > 0 and y_norm > 0 else None


def run_score_path(run_dir: Path) -> Path:
    path = run_dir / "results" / "fopci_scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare optimized FOPCI raw dots with a reference run.")
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="concept-attribution-256-512-v0")
    parser.add_argument("--output-variant", default="fopci-batch-validation")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        raise SystemExit("tolerances must be nonnegative")
    reference_path = run_score_path(args.reference_run_dir.resolve())
    candidate_path = run_score_path(args.candidate_run_dir.resolve())
    reference = {str(row["sample_id"]): row for row in load_jsonl(reference_path)}
    candidate = {str(row["sample_id"]): row for row in load_jsonl(candidate_path)}
    shared = sorted(set(reference) & set(candidate))
    if not shared:
        raise ValueError("runs have no shared sample IDs")
    axes = sorted(
        set.intersection(
            *[
                set(reference[sample_id]["axis_scores"]) & set(candidate[sample_id]["axis_scores"])
                for sample_id in shared
            ]
        )
    )
    if not axes:
        raise ValueError("runs have no shared axes")
    details = []
    summaries = []
    passed = True
    for axis_name in axes:
        reference_values = []
        candidate_values = []
        axis_deltas = []
        axis_allowed = []
        for sample_id in shared:
            ref = float(reference[sample_id]["axis_scores"][axis_name]["negative_gradient_dot"])
            cand = float(candidate[sample_id]["axis_scores"][axis_name]["negative_gradient_dot"])
            if not math.isfinite(ref) or not math.isfinite(cand):
                raise ValueError(f"non-finite score: sample={sample_id} axis={axis_name}")
            delta = abs(cand - ref)
            allowed = args.absolute_tolerance + args.relative_tolerance * abs(ref)
            row_passed = delta <= allowed
            passed = passed and row_passed
            reference_values.append(ref)
            candidate_values.append(cand)
            axis_deltas.append(delta)
            axis_allowed.append(allowed)
            details.append(
                {
                    "sample_id": sample_id,
                    "axis_name": axis_name,
                    "reference": ref,
                    "candidate": cand,
                    "absolute_delta": delta,
                    "allowed_delta": allowed,
                    "passed": row_passed,
                }
            )
        summaries.append(
            {
                "axis_name": axis_name,
                "records": len(shared),
                "max_absolute_delta": max(axis_deltas),
                "mean_absolute_delta": sum(axis_deltas) / len(axis_deltas),
                "max_allowed_delta": max(axis_allowed),
                "pearson": pearson(reference_values, candidate_values),
                "sign_agreement": sum(
                    (ref > 0) == (cand > 0) for ref, cand in zip(reference_values, candidate_values)
                )
                / len(reference_values),
                "passed": all(delta <= allowed for delta, allowed in zip(axis_deltas, axis_allowed)),
            }
        )
    run_dir = (
        args.output_root
        / args.experiment_name
        / args.model_name
        / args.dataset_name
        / args.probe_set
        / args.output_variant
        / (args.run_id or default_run_id())
    )
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for directory in [results_dir, meta_dir, checkpoints_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    details_path = results_dir / "fopci_score_agreement.csv"
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    summary = {
        "schema_version": "0.1",
        "passed": passed,
        "shared_records": len(shared),
        "axes": summaries,
        "thresholds": {
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
        },
        "reference": {"run_dir": str(args.reference_run_dir.resolve()), "scores_sha256": file_sha256(reference_path)},
        "candidate": {"run_dir": str(args.candidate_run_dir.resolve()), "scores_sha256": file_sha256(candidate_path)},
    }
    write_json(results_dir / "fopci_score_agreement_summary.json", summary)
    write_json(results_dir / "results.json", summary)
    write_json(meta_dir / "status.json", {"schema_version": "0.1", "state": "completed", "passed": passed, "updated_at_utc": utc_now()})
    write_json(checkpoints_dir / "progress.json", {"schema_version": "0.1", "state": "completed", "shared_records": len(shared), "updated_at_utc": utc_now()})
    write_json(meta_dir / "run_manifest.json", {"schema_version": "0.1", "runner": "FOPCIRunComparator", "created_at_utc": utc_now(), "run_dir": str(run_dir.resolve()), "summary": summary})
    (logs_dir / "run.log").write_text(json.dumps({"time_utc": utc_now(), "event": "completed", "passed": passed}) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "passed": passed, "run_dir": str(run_dir)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
