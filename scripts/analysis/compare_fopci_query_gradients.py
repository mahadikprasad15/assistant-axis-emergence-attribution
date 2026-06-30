#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare batched and sequential FOPCI query-gradient bundles.")
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="concept-attribution-256-512-v0")
    parser.add_argument("--output-variant", default="fopci-query-batch-validation")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    import torch

    args = build_parser().parse_args()
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        raise SystemExit("tolerances must be nonnegative")
    reference = torch.load(args.reference_bundle, map_location="cpu")
    candidate = torch.load(args.candidate_bundle, map_location="cpu")
    for key in ["parameter_names", "scope"]:
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"query bundles differ in {key}")
    reference_gradients = reference.get("gradients", {})
    candidate_gradients = candidate.get("gradients", {})
    if set(reference_gradients) != set(candidate_gradients):
        raise ValueError("query bundles have different targets")
    axes = []
    passed = True
    for axis_name in sorted(reference_gradients):
        reference_parts = reference_gradients[axis_name]
        candidate_parts = candidate_gradients[axis_name]
        if len(reference_parts) != len(candidate_parts):
            raise ValueError(f"parameter tensor count differs for {axis_name}")
        dot = 0.0
        reference_squared = 0.0
        candidate_squared = 0.0
        difference_squared = 0.0
        maximum_absolute_delta = 0.0
        maximum_reference = 0.0
        for reference_part, candidate_part in zip(reference_parts, candidate_parts):
            reference_part = reference_part.double()
            candidate_part = candidate_part.double()
            if reference_part.shape != candidate_part.shape:
                raise ValueError(f"parameter shape differs for {axis_name}")
            difference = candidate_part - reference_part
            dot += float(torch.sum(reference_part * candidate_part).item())
            reference_squared += float(torch.sum(reference_part**2).item())
            candidate_squared += float(torch.sum(candidate_part**2).item())
            difference_squared += float(torch.sum(difference**2).item())
            maximum_absolute_delta = max(maximum_absolute_delta, float(torch.max(torch.abs(difference)).item()))
            maximum_reference = max(maximum_reference, float(torch.max(torch.abs(reference_part)).item()))
        reference_norm = math.sqrt(reference_squared)
        candidate_norm = math.sqrt(candidate_squared)
        difference_norm = math.sqrt(difference_squared)
        relative_l2_error = difference_norm / max(reference_norm, 1e-30)
        cosine = dot / (reference_norm * candidate_norm) if reference_norm > 0 and candidate_norm > 0 else float("nan")
        allowed_maximum_delta = args.absolute_tolerance + args.relative_tolerance * maximum_reference
        axis_passed = (
            math.isfinite(cosine)
            and maximum_absolute_delta <= allowed_maximum_delta
            and relative_l2_error <= args.relative_tolerance
        )
        passed = passed and axis_passed
        axes.append(
            {
                "axis_name": axis_name,
                "reference_norm": reference_norm,
                "candidate_norm": candidate_norm,
                "norm_ratio": candidate_norm / reference_norm if reference_norm > 0 else None,
                "cosine": cosine,
                "difference_norm": difference_norm,
                "relative_l2_error": relative_l2_error,
                "max_absolute_delta": maximum_absolute_delta,
                "max_allowed_delta": allowed_maximum_delta,
                "passed": axis_passed,
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
    summary = {
        "schema_version": "0.1",
        "passed": passed,
        "reference_bundle": str(args.reference_bundle.resolve()),
        "candidate_bundle": str(args.candidate_bundle.resolve()),
        "thresholds": {
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
        },
        "axes": axes,
    }
    write_json(run_dir / "results" / "query_gradient_agreement.json", summary)
    write_json(run_dir / "results" / "results.json", summary)
    write_json(run_dir / "meta" / "status.json", {"schema_version": "0.1", "state": "completed", "passed": passed, "updated_at_utc": utc_now()})
    write_json(run_dir / "meta" / "run_manifest.json", {"schema_version": "0.1", "runner": "FOPCIQueryGradientComparator", "created_at_utc": utc_now(), "run_dir": str(run_dir.resolve()), "summary": summary})
    write_json(run_dir / "checkpoints" / "progress.json", {"schema_version": "0.1", "state": "completed", "targets": len(axes), "updated_at_utc": utc_now()})
    log_path = run_dir / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"time_utc": utc_now(), "event": "completed", "passed": passed}) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "passed": passed, "run_dir": str(run_dir)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
