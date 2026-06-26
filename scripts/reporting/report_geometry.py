#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["source_record_count", "pc1_abs_loading_rank", "aa_abs_loading_rank"]:
            if row.get(key) not in (None, ""):
                row[key] = int(row[key])
        for key in ["pc1_loading", "aa_loading"]:
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


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


def resolve_run_dir(args: argparse.Namespace) -> Path:
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


def score_band(value: float | None, proceed_threshold: float, caution_threshold: float) -> str:
    if value is None:
        return "missing"
    if value >= proceed_threshold:
        return "pass"
    if value >= caution_threshold:
        return "caution"
    return "fail"


def decide_gate(artifact_complete: bool, alignment_band: str, pc1_band: str) -> str:
    if not artifact_complete:
        return "stop"
    if alignment_band == "fail" or pc1_band == "fail":
        return "stop"
    if alignment_band == "caution" or pc1_band == "caution":
        return "caution"
    if alignment_band == "pass" and pc1_band == "pass":
        return "proceed"
    return "caution"


def top_rows(rows: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: abs(float(row.get(key, 0.0))), reverse=True)[:limit]


def format_loading_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| group_type | group_id | records | pc1_loading | aa_loading |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('group_type')} | {row.get('group_id')} | {row.get('source_record_count')} | "
            f"{float(row.get('pc1_loading', 0.0)):.4f} | {float(row.get('aa_loading', 0.0)):.4f} |"
        )
    return "\n".join(lines)


def build_metrics(
    aa_summary: dict[str, Any],
    geometry_summary: dict[str, Any],
    loadings: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    aa_pc1_cosine = geometry_summary.get("aa_pc1_cosine")
    pc1_evr = geometry_summary.get("pc1_explained_variance_ratio")
    alignment_band = score_band(aa_pc1_cosine, args.alignment_proceed, args.alignment_caution)
    pc1_band = score_band(pc1_evr, args.pc1_evr_proceed, args.pc1_evr_caution)
    counts = aa_summary.get("counts", {})
    artifact_complete = bool(
        counts.get("default_records", 0) > 0
        and counts.get("contrast_records", 0) > 0
        and geometry_summary.get("group_count", 0) >= 2
        and loadings
    )
    gate = decide_gate(artifact_complete, alignment_band, pc1_band)
    return {
        "schema_version": "0.1",
        "gate": gate,
        "artifact_complete": artifact_complete,
        "thresholds": {
            "alignment_proceed": args.alignment_proceed,
            "alignment_caution": args.alignment_caution,
            "pc1_evr_proceed": args.pc1_evr_proceed,
            "pc1_evr_caution": args.pc1_evr_caution,
        },
        "metrics": {
            "aa_pc1_cosine": aa_pc1_cosine,
            "aa_pc1_cosine_band": alignment_band,
            "pc1_explained_variance_ratio": pc1_evr,
            "pc1_explained_variance_band": pc1_band,
            "aa_default_records": counts.get("default_records"),
            "aa_contrast_records": counts.get("contrast_records"),
            "geometry_group_count": geometry_summary.get("group_count"),
            "geometry_record_count": geometry_summary.get("record_count"),
        },
        "context": {
            "model_id": geometry_summary.get("model_id") or aa_summary.get("model_id"),
            "checkpoint_revision": geometry_summary.get("checkpoint_revision") or aa_summary.get("checkpoint_revision"),
            "layer": geometry_summary.get("layer") or aa_summary.get("layer"),
            "pooling_policy": geometry_summary.get("pooling_policy") or aa_summary.get("pooling_policy"),
            "axis_variant_id": aa_summary.get("variant_id"),
        },
        "top_pc1_loadings": top_rows(loadings, "pc1_loading", args.top_k),
        "top_aa_loadings": top_rows(loadings, "aa_loading", args.top_k),
    }


def build_markdown(metrics: dict[str, Any], aa_summary: dict[str, Any], geometry_summary: dict[str, Any]) -> str:
    gate = str(metrics["gate"]).upper()
    metric_values = metrics["metrics"]
    context = metrics["context"]
    lines = [
        "# Geometry Sanity Report",
        "",
        f"Gate decision: **{gate}**",
        "",
        "## Context",
        "",
        f"- model: `{context.get('model_id')}`",
        f"- checkpoint: `{context.get('checkpoint_revision')}`",
        f"- layer: `{context.get('layer')}`",
        f"- pooling: `{context.get('pooling_policy')}`",
        f"- axis variant: `{context.get('axis_variant_id')}`",
        "",
        "## Metrics",
        "",
        f"- AA-PC1 cosine: `{metric_values.get('aa_pc1_cosine')}` ({metric_values.get('aa_pc1_cosine_band')})",
        f"- PC1 explained variance ratio: `{metric_values.get('pc1_explained_variance_ratio')}` ({metric_values.get('pc1_explained_variance_band')})",
        f"- AA default records: `{metric_values.get('aa_default_records')}`",
        f"- AA contrast records: `{metric_values.get('aa_contrast_records')}`",
        f"- geometry groups: `{metric_values.get('geometry_group_count')}`",
        f"- geometry source records: `{metric_values.get('geometry_record_count')}`",
        "",
        "## Top PC1 Loadings",
        "",
        format_loading_table(metrics["top_pc1_loadings"]),
        "",
        "## Top AA Loadings",
        "",
        format_loading_table(metrics["top_aa_loadings"]),
        "",
        "## Source Files",
        "",
        f"- AA vector: `{aa_summary.get('vector_path')}`",
        f"- role PC1: `{geometry_summary.get('pc1_path')}`",
        f"- loadings CSV: `{geometry_summary.get('role_loadings_csv')}`",
        "",
        "## Interpretation Boundary",
        "",
        "This report only gates whether the final-checkpoint direction is coherent enough to sweep. It does not establish emergence over training.",
        "",
    ]
    return "\n".join(lines)


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final-checkpoint geometry sanity report.")
    parser.add_argument("--assistant-axis-run-dir", type=Path, required=True)
    parser.add_argument("--role-geometry-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--output-variant", default="geometry-report-layer12")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alignment-proceed", type=float, default=0.30)
    parser.add_argument("--alignment-caution", type=float, default=0.15)
    parser.add_argument("--pc1-evr-proceed", type=float, default=0.20)
    parser.add_argument("--pc1-evr-caution", type=float, default=0.10)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = Path(".").resolve()
    aa_run_dir = args.assistant_axis_run_dir
    geometry_run_dir = args.role_geometry_run_dir
    if not aa_run_dir.is_absolute():
        aa_run_dir = repo_root / aa_run_dir
    if not geometry_run_dir.is_absolute():
        geometry_run_dir = repo_root / geometry_run_dir

    run_dir = resolve_run_dir(args)
    results_dir = run_dir / "results"
    checkpoints_dir = run_dir / "checkpoints"
    meta_dir = run_dir / "meta"
    logs_dir = run_dir / "logs"
    for directory in [results_dir, checkpoints_dir, meta_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    status_path = meta_dir / "status.json"
    manifest_path = meta_dir / "run_manifest.json"
    progress_path = checkpoints_dir / "progress.json"
    log_path = logs_dir / "run.log"
    metrics_path = results_dir / "geometry_metrics.json"
    report_path = results_dir / "geometry_report.md"

    if status_path.exists() and not args.force_completed:
        status = load_json(status_path)
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    write_status(status_path, "running", "geometry report started", {"reports": 0})
    append_log(log_path, "start", {"run_dir": str(run_dir)})

    try:
        aa_summary_path = aa_run_dir / "results" / "assistant_axis_summary.json"
        geometry_summary_path = geometry_run_dir / "results" / "role_geometry_summary.json"
        loadings_path = geometry_run_dir / "results" / "role_loadings.csv"
        aa_summary = load_json(aa_summary_path)
        geometry_summary = load_json(geometry_summary_path)
        loadings = load_csv(loadings_path)
        metrics = build_metrics(aa_summary, geometry_summary, loadings, args)
        report = build_markdown(metrics, aa_summary, geometry_summary)
        write_json(metrics_path, metrics)
        report_path.write_text(report, encoding="utf-8")
        write_json(progress_path, {"schema_version": "0.1", "state": "completed", "completed_steps": ["loaded_inputs", "wrote_report"], "updated_at_utc": utc_now()})
        write_json(
            manifest_path,
            {
                "schema_version": "0.1",
                "builder": "GeometryReportBuilder",
                "created_at_utc": utc_now(),
                "run_dir": str(run_dir),
                "assistant_axis_run": {
                    "run_dir": str(aa_run_dir),
                    "summary": str(aa_summary_path),
                    "summary_sha256": file_sha256(aa_summary_path),
                },
                "role_geometry_run": {
                    "run_dir": str(geometry_run_dir),
                    "summary": str(geometry_summary_path),
                    "summary_sha256": file_sha256(geometry_summary_path),
                    "loadings_csv": str(loadings_path),
                    "loadings_sha256": file_sha256(loadings_path),
                },
                "outputs": {
                    "metrics": str(metrics_path),
                    "report": str(report_path),
                },
                "validation": {
                    "passed": True,
                    "errors": [],
                    "warnings": [],
                },
            },
        )
        write_status(status_path, "completed", f"geometry report completed: {metrics['gate']}", {"reports": 1})
        append_log(log_path, "completed", {"gate": metrics["gate"]})
        print(
            json.dumps(
                {
                    "status": "completed",
                    "gate": metrics["gate"],
                    "run_dir": str(run_dir),
                    "metrics": str(metrics_path),
                    "report": str(report_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        write_json(progress_path, {"schema_version": "0.1", "state": "failed", "completed_steps": [], "updated_at_utc": utc_now()})
        write_status(status_path, "failed", f"geometry report failed: {type(exc).__name__}: {exc}", {"reports": 0})
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
        print(
            json.dumps(
                {
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                    "run_dir": str(run_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
