#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["sample_id"])] = row
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def axis_scores(row: dict[str, Any], score_type: str) -> dict[str, float]:
    if score_type == "dot":
        if not row.get("axis_dot_scores"):
            return {}
        return {str(name): float(value) for name, value in row["axis_dot_scores"].items()}
    if row.get("axis_scores"):
        return {str(name): float(value) for name, value in row["axis_scores"].items()}
    scores = {"local_aa": float(row["local_aa_score"])}
    if row.get("final_aa_score") is not None:
        scores["final_aa"] = float(row["final_aa_score"])
    return scores


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    centered_a = [value - mean_a for value in a]
    centered_b = [value - mean_b for value in b]
    denom = math.sqrt(sum(value * value for value in centered_a) * sum(value * value for value in centered_b))
    return sum(x * y for x, y in zip(centered_a, centered_b)) / denom if denom > 0 else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare score agreement between two gradient-attribution runs.")
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--axis-name", action="append", default=[])
    parser.add_argument("--score-type", choices=["cosine", "dot"], default="cosine")
    parser.add_argument("--max-absolute-delta", type=float, default=1e-4)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="assistant-axis-attribution-v0")
    parser.add_argument("--output-variant", default="gradient-attribution-comparison-layer12")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.max_absolute_delta <= 0:
        raise SystemExit("--max-absolute-delta must be positive")
    run_id = args.run_id or default_run_id()
    run_dir = args.output_root / args.experiment_name / args.model_name / args.dataset_name / args.probe_set / args.output_variant / run_id
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for directory in [results_dir, meta_dir, checkpoints_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    reference_path = args.reference_run_dir / "results" / "attribution_scores.jsonl"
    candidate_path = args.candidate_run_dir / "results" / "attribution_scores.jsonl"
    reference = load_jsonl(reference_path)
    candidate = load_jsonl(candidate_path)
    sample_ids = sorted(set(reference) & set(candidate))
    if not sample_ids:
        raise SystemExit("runs have no shared sample ids")
    common_axes = set(axis_scores(reference[sample_ids[0]], args.score_type)) & set(
        axis_scores(candidate[sample_ids[0]], args.score_type)
    )
    if not common_axes:
        raise SystemExit(f"runs have no shared {args.score_type} axis scores")
    selected_axes = args.axis_name or sorted(common_axes)
    missing_axes = [name for name in selected_axes if name not in common_axes]
    if missing_axes:
        raise SystemExit(f"requested axes are not common to both runs: {missing_axes}")

    detail_rows: list[dict[str, Any]] = []
    summaries = []
    for name in selected_axes:
        ref_values = [axis_scores(reference[sample_id], args.score_type)[name] for sample_id in sample_ids]
        candidate_values = [axis_scores(candidate[sample_id], args.score_type)[name] for sample_id in sample_ids]
        deltas = [candidate_value - ref_value for ref_value, candidate_value in zip(ref_values, candidate_values)]
        absolute = [abs(value) for value in deltas]
        for sample_id, ref_value, candidate_value, delta in zip(sample_ids, ref_values, candidate_values, deltas):
            detail_rows.append(
                {
                    "sample_id": sample_id,
                    "axis_name": name,
                    "reference_score": ref_value,
                    "candidate_score": candidate_value,
                    "delta": delta,
                    "absolute_delta": abs(delta),
                }
            )
        summaries.append(
            {
                "axis_name": name,
                "records": len(sample_ids),
                "mean_absolute_delta": sum(absolute) / len(absolute),
                "max_absolute_delta": max(absolute),
                "rmse": math.sqrt(sum(value * value for value in deltas) / len(deltas)),
                "pearson": pearson(ref_values, candidate_values),
                "sign_agreement": sum((a >= 0) == (b >= 0) for a, b in zip(ref_values, candidate_values)) / len(ref_values),
                "passed": max(absolute) <= args.max_absolute_delta,
            }
        )

    details_path = results_dir / "score_agreement.csv"
    with details_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    passed = all(item["passed"] for item in summaries)
    summary = {
        "schema_version": "0.1",
        "reference_run_dir": str(args.reference_run_dir),
        "candidate_run_dir": str(args.candidate_run_dir),
        "shared_records": len(sample_ids),
        "axes": summaries,
        "score_type": args.score_type,
        "thresholds": {"max_absolute_delta": args.max_absolute_delta},
        "passed": passed,
    }
    summary_path = results_dir / "score_agreement_summary.json"
    write_json(summary_path, summary)
    write_json(
        meta_dir / "run_manifest.json",
        {
            "schema_version": "0.1",
            "runner": "GradientAttributionRunComparator",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "reference_scores": str(reference_path),
            "candidate_scores": str(candidate_path),
            "axes": selected_axes,
            "score_type": args.score_type,
            "thresholds": summary["thresholds"],
            "results": {"summary": str(summary_path), "details_csv": str(details_path)},
        },
    )
    write_json(meta_dir / "status.json", {"schema_version": "0.1", "state": "completed", "passed": passed, "updated_at_utc": utc_now()})
    write_json(checkpoints_dir / "progress.json", {"schema_version": "0.1", "state": "completed", "shared_records": len(sample_ids)})
    (logs_dir / "run.log").write_text(json.dumps({"time_utc": utc_now(), "event": "completed", "passed": passed}) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "passed": passed, "run_dir": str(run_dir), "summary": str(summary_path)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
