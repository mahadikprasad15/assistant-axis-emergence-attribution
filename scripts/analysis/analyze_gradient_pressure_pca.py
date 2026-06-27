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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} contains a non-object JSONL row")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def resolve_path(path_text: str | Path | None, repo_root: Path) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


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


def write_progress(path: Path, state: str, completed_scopes: list[str]) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "state": state,
            "updated_at_utc": utc_now(),
            "completed_scopes": completed_scopes,
            "completed_count": len(completed_scopes),
        },
    )


def latest_rows_by_sample_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            latest[sample_id] = row
    return list(latest.values())


def infer_axis_path(manifest: dict[str, Any], axis_key: str) -> str | None:
    axis = manifest.get(axis_key)
    if isinstance(axis, dict):
        value = axis.get("vector_path")
        if isinstance(value, str):
            return value
    return None


def load_axis(path: Path | None) -> Any | None:
    if path is None:
        return None
    import torch

    if not path.exists():
        raise FileNotFoundError(f"axis vector not found: {path}")
    vector = torch.load(path, map_location="cpu").float()
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm.item()) <= 0:
        raise ValueError(f"axis vector has invalid norm: {path}")
    return vector / norm


def safe_cosine(a: Any, b: Any | None) -> float | None:
    if b is None:
        return None
    import torch

    score = float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0).item())
    if not math.isfinite(score):
        raise ValueError("non-finite cosine")
    return score


def load_pressure_rows(rows: list[dict[str, Any]], repo_root: Path, limit: int | None) -> tuple[list[dict[str, Any]], Any]:
    import torch

    selected = rows if limit is None else rows[:limit]
    metadata: list[dict[str, Any]] = []
    vectors = []
    for row in selected:
        path = resolve_path(row.get("gradient_pressure_path"), repo_root)
        if path is None or not path.exists():
            raise FileNotFoundError(f"missing gradient pressure vector for sample_id={row.get('sample_id')}: {path}")
        vector = torch.load(path, map_location="cpu").float()
        if vector.ndim != 1:
            raise ValueError(f"gradient pressure vector must be 1D: {path}")
        vectors.append(vector)
        metadata.append(row)
    if not vectors:
        raise ValueError("no gradient pressure vectors selected")
    shapes = {tuple(vector.shape) for vector in vectors}
    if len(shapes) != 1:
        raise ValueError(f"mixed gradient vector shapes: {sorted(map(str, shapes))}")
    return metadata, torch.stack(vectors, dim=0)


def orient_component(component: Any, local_axis: Any | None, final_axis: Any | None) -> Any:
    reference = local_axis if local_axis is not None else final_axis
    if reference is None:
        return component
    cosine = safe_cosine(component, reference)
    if cosine is not None and cosine < 0:
        return -component
    return component


def run_pca(
    matrix: Any,
    scope: str,
    scope_id: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    local_axis: Any | None,
    final_axis: Any | None,
    pcs_dir: Path,
    singular_values_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    if matrix.shape[0] < args.min_records:
        return [], {"scope": scope, "scope_id": scope_id, "records": int(matrix.shape[0]), "skipped": "too_few_records"}
    work = matrix.float()
    mean = work.mean(dim=0)
    if args.center:
        work = work - mean
    if torch.linalg.vector_norm(work) <= 0:
        return [], {"scope": scope, "scope_id": scope_id, "records": int(matrix.shape[0]), "skipped": "zero_variance"}

    _, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    energy = singular_values.square()
    total_energy = float(energy.sum().item())
    if total_energy <= 0 or not math.isfinite(total_energy):
        return [], {"scope": scope, "scope_id": scope_id, "records": int(matrix.shape[0]), "skipped": "invalid_energy"}

    max_components = min(args.top_k, vh.shape[0])
    scope_slug = sanitize_id(f"{scope}_{scope_id}")
    singular_values_path = singular_values_dir / f"{scope_slug}__singular_values.pt"
    mean_path = singular_values_dir / f"{scope_slug}__mean.pt"
    torch.save(singular_values, singular_values_path)
    torch.save(mean, mean_path)

    component_rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for component_index in range(max_components):
        component = orient_component(vh[component_index].detach().cpu(), local_axis, final_axis)
        explained = float((energy[component_index] / energy.sum()).item())
        cumulative += explained
        pc_path = pcs_dir / f"{scope_slug}__pc{component_index + 1:02d}.pt"
        torch.save(component, pc_path)
        component_rows.append(
            {
                "schema_version": "0.1",
                "scope": scope,
                "scope_id": scope_id,
                "component_index": component_index,
                "records": int(matrix.shape[0]),
                "vector_dim": int(matrix.shape[1]),
                "centered": args.center,
                "singular_value": float(singular_values[component_index].item()),
                "explained_variance_ratio": explained,
                "cumulative_explained_variance_ratio": cumulative,
                "local_aa_cosine": safe_cosine(component, local_axis),
                "final_aa_cosine": safe_cosine(component, final_axis),
                "pc_vector_path": str(pc_path),
                "singular_values_path": str(singular_values_path),
                "mean_vector_path": str(mean_path),
                "top_local_aa_score_samples": [
                    str(row["sample_id"])
                    for row in sorted(rows, key=lambda item: float(item.get("local_aa_score", 0.0)), reverse=True)[:10]
                ],
            }
        )
    summary = {
        "scope": scope,
        "scope_id": scope_id,
        "records": int(matrix.shape[0]),
        "vector_dim": int(matrix.shape[1]),
        "centered": args.center,
        "pc1_explained_variance_ratio": component_rows[0]["explained_variance_ratio"],
        "pc1_local_aa_cosine": component_rows[0]["local_aa_cosine"],
        "pc1_final_aa_cosine": component_rows[0]["final_aa_cosine"],
        "top_k_explained_variance_ratio": component_rows[-1]["cumulative_explained_variance_ratio"],
        "singular_values_path": str(singular_values_path),
    }
    return component_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scope",
        "scope_id",
        "component_index",
        "records",
        "vector_dim",
        "centered",
        "singular_value",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
        "local_aa_cosine",
        "final_aa_cosine",
        "pc_vector_path",
        "singular_values_path",
        "mean_vector_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PCA/SVD over saved gradient-pressure vectors.")
    parser.add_argument("--attribution-run-dir", type=Path, required=True)
    parser.add_argument("--local-axis-vector", type=Path, default=None)
    parser.add_argument("--final-axis-vector", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-records", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-center", dest="center", action="store_false")
    parser.set_defaults(center=True)
    parser.add_argument("--force-completed", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="assistant-axis-attribution-v0")
    parser.add_argument("--output-variant", default="gradient-pressure-pca-layer12")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if args.min_records < 2:
        raise SystemExit("--min-records must be at least 2")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    repo_root = Path(".").resolve()
    attribution_run_dir = resolve_path(args.attribution_run_dir, repo_root)
    if attribution_run_dir is None:
        raise SystemExit("--attribution-run-dir is required")
    scores_path = attribution_run_dir / "results" / "attribution_scores.jsonl"
    attribution_manifest_path = attribution_run_dir / "meta" / "run_manifest.json"
    attribution_manifest = load_json(attribution_manifest_path) if attribution_manifest_path.exists() else {}

    local_axis_path = resolve_path(args.local_axis_vector, repo_root) or resolve_path(infer_axis_path(attribution_manifest, "local_axis"), repo_root)
    final_axis_path = resolve_path(args.final_axis_vector, repo_root) or resolve_path(infer_axis_path(attribution_manifest, "final_axis"), repo_root)

    run_dir = resolve_run_dir(args)
    inputs_dir = run_dir / "inputs"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    pcs_dir = results_dir / "pcs"
    singular_values_dir = results_dir / "singular_values"
    logs_dir = run_dir / "logs"
    meta_dir = run_dir / "meta"
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"
    components_path = results_dir / "gradient_pressure_components.jsonl"
    csv_path = results_dir / "gradient_pressure_pca.csv"
    summary_path = results_dir / "gradient_pressure_pca_summary.json"

    for directory in [inputs_dir, checkpoints_dir, results_dir, pcs_dir, singular_values_dir, logs_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if status_path.exists() and not args.force_completed:
        status = load_json(status_path)
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    if components_path.exists():
        components_path.unlink()

    write_status(status_path, "running", "gradient pressure PCA started", {"components": 0, "scopes": 0})
    append_log(log_path, "start", {"attribution_run_dir": str(attribution_run_dir), "run_dir": str(run_dir)})

    completed_scopes: list[str] = []
    final_state = "failed"
    final_message = "gradient pressure PCA did not complete"
    all_component_rows: list[dict[str, Any]] = []
    scope_summaries: list[dict[str, Any]] = []

    try:
        rows = latest_rows_by_sample_id(load_jsonl(scores_path))
        rows, matrix = load_pressure_rows(rows, repo_root, args.limit)
        local_axis = load_axis(local_axis_path)
        final_axis = load_axis(final_axis_path)
        write_json(
            inputs_dir / "selected_sample_ids.json",
            {"sample_ids": [str(row["sample_id"]) for row in rows]},
        )

        component_rows, scope_summary = run_pca(
            matrix=matrix,
            scope="global",
            scope_id="all",
            rows=rows,
            args=args,
            local_axis=local_axis,
            final_axis=final_axis,
            pcs_dir=pcs_dir,
            singular_values_dir=singular_values_dir,
        )
        all_component_rows.extend(component_rows)
        scope_summaries.append(scope_summary)
        completed_scopes.append("global:all")
        write_progress(progress_path, "running", completed_scopes)

        by_window: dict[str, list[int]] = defaultdict(list)
        for row_idx, row in enumerate(rows):
            by_window[str(row["window_id"])].append(row_idx)

        import torch

        for window_id, indices in sorted(by_window.items()):
            window_matrix = matrix[torch.tensor(indices, dtype=torch.long), :]
            window_rows = [rows[index] for index in indices]
            component_rows, scope_summary = run_pca(
                matrix=window_matrix,
                scope="window",
                scope_id=window_id,
                rows=window_rows,
                args=args,
                local_axis=local_axis,
                final_axis=final_axis,
                pcs_dir=pcs_dir,
                singular_values_dir=singular_values_dir,
            )
            all_component_rows.extend(component_rows)
            scope_summaries.append(scope_summary)
            completed_scopes.append(f"window:{window_id}")
            write_progress(progress_path, "running", completed_scopes)

        for row in all_component_rows:
            append_jsonl(components_path, row)
        if not all_component_rows:
            raise ValueError("no PCA component rows were produced; check record counts and --min-records")
        write_csv(csv_path, all_component_rows)
        summary = {
            "schema_version": "0.1",
            "run_dir": str(run_dir),
            "attribution_run_dir": str(attribution_run_dir),
            "records": len(rows),
            "vector_dim": int(matrix.shape[1]),
            "centered": args.center,
            "top_k": args.top_k,
            "min_records": args.min_records,
            "local_axis_vector": str(local_axis_path) if local_axis_path else None,
            "final_axis_vector": str(final_axis_path) if final_axis_path else None,
            "component_rows": len(all_component_rows),
            "scopes": scope_summaries,
            "outputs": {
                "components_jsonl": str(components_path),
                "components_csv": str(csv_path),
                "pcs_dir": str(pcs_dir),
                "singular_values_dir": str(singular_values_dir),
            },
        }
        write_json(summary_path, summary)
        final_state = "completed"
        final_message = "gradient pressure PCA completed"
    except Exception as exc:
        final_state = "failed"
        final_message = f"gradient pressure PCA failed: {type(exc).__name__}: {exc}"
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})

    write_progress(progress_path, final_state, completed_scopes)
    write_status(
        status_path,
        final_state,
        final_message,
        {"components": len(all_component_rows), "scopes": len(scope_summaries)},
    )
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "runner": "GradientPressurePCAAnalyzer",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "attribution_run": {
                "run_dir": str(attribution_run_dir),
                "scores_jsonl": str(scores_path),
                "scores_sha256": file_sha256(scores_path),
                "run_manifest": attribution_manifest,
            },
            "axes": {
                "local_axis_vector": str(local_axis_path) if local_axis_path else None,
                "local_axis_sha256": file_sha256(local_axis_path) if local_axis_path else None,
                "final_axis_vector": str(final_axis_path) if final_axis_path else None,
                "final_axis_sha256": file_sha256(final_axis_path) if final_axis_path else None,
            },
            "pca": {
                "centered": args.center,
                "top_k": args.top_k,
                "min_records": args.min_records,
                "orientation": "local_axis_nonnegative_else_final_axis_nonnegative",
            },
            "results": {
                "summary": str(summary_path),
                "components_jsonl": str(components_path),
                "components_csv": str(csv_path),
                "pcs_dir": str(pcs_dir),
                "singular_values_dir": str(singular_values_dir),
            },
        },
    )
    append_log(log_path, final_state, {"components": len(all_component_rows), "scopes": len(scope_summaries)})

    print(
        json.dumps(
            {
                "status": final_state,
                "message": final_message,
                "run_dir": str(run_dir),
                "summary": str(summary_path),
                "components": str(components_path),
                "component_rows": len(all_component_rows),
                "scopes": len(scope_summaries),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_state == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
