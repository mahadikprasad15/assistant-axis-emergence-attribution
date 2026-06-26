#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
import sys
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


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


def cosine(a: Any, b: Any) -> float:
    import torch

    a = a.float()
    b = b.float()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 0:
        return float("nan")
    return float((torch.dot(a, b) / denom).item())


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    denom = math.sqrt(sum(x * x for x in x_centered) * sum(y * y for y in y_centered))
    if denom <= 0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denom


def loading_map(path: Path, key: str) -> dict[tuple[str, str], float]:
    rows = load_csv(path)
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        out[(str(row["group_type"]), str(row["group_id"]))] = float(row[key])
    return out


def loading_corr(a: dict[tuple[str, str], float], b: dict[tuple[str, str], float]) -> float | None:
    keys = sorted(set(a).intersection(b))
    return pearson([a[key] for key in keys], [b[key] for key in keys])


def load_checkpoint_artifacts(sweep_summary: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    import torch

    checkpoints = sweep_summary.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("sweep summary has no checkpoints")

    artifacts: list[dict[str, Any]] = []
    for item in checkpoints:
        paths = item.get("paths", {})
        if not isinstance(paths, dict):
            raise ValueError(f"checkpoint missing paths: {item}")
        aa_dir = resolve_path(str(paths["aa_dir"]), repo_root)
        role_dir = resolve_path(str(paths["role_dir"]), repo_root)
        report_dir = resolve_path(str(paths["report_dir"]), repo_root)

        aa_summary_path = aa_dir / "results" / "assistant_axis_summary.json"
        role_summary_path = role_dir / "results" / "role_geometry_summary.json"
        report_metrics_path = report_dir / "results" / "geometry_metrics.json"
        aa_summary = load_json(aa_summary_path)
        role_summary = load_json(role_summary_path)
        report_metrics = load_json(report_metrics_path) if report_metrics_path.exists() else {}

        aa_vector_path = resolve_path(str(aa_summary["vector_path"]), repo_root)
        pc1_path = resolve_path(str(role_summary["pc1_path"]), repo_root)
        loadings_path = resolve_path(str(role_summary["role_loadings_csv"]), repo_root)

        artifacts.append(
            {
                "revision": str(item["revision"]),
                "step": int(item["step"]) if item.get("step") is not None else None,
                "reason": item.get("reason"),
                "aa_dir": aa_dir,
                "role_dir": role_dir,
                "report_dir": report_dir,
                "aa_summary_path": aa_summary_path,
                "role_summary_path": role_summary_path,
                "report_metrics_path": report_metrics_path,
                "aa_vector_path": aa_vector_path,
                "pc1_path": pc1_path,
                "loadings_path": loadings_path,
                "aa_summary": aa_summary,
                "role_summary": role_summary,
                "report_metrics": report_metrics,
                "aa_vector": torch.load(aa_vector_path, map_location="cpu").float(),
                "pc1_vector": torch.load(pc1_path, map_location="cpu").float(),
                "aa_loadings": loading_map(loadings_path, "aa_loading"),
                "pc1_loadings": loading_map(loadings_path, "pc1_loading"),
            }
        )
    return sorted(artifacts, key=lambda row: (-1 if row["step"] is None else row["step"]))


def transition_score(row: dict[str, Any]) -> float:
    score = 0.0
    if row.get("aa_adjacent_cosine") is not None:
        score += 1.0 - float(row["aa_adjacent_cosine"])
    if row.get("pc1_adjacent_cosine") is not None:
        score += 1.0 - float(row["pc1_adjacent_cosine"])
    if row.get("aa_loading_adjacent_corr") is not None:
        score += 1.0 - float(row["aa_loading_adjacent_corr"])
    if row.get("aa_pc1_cosine_delta") is not None:
        score += abs(float(row["aa_pc1_cosine_delta"]))
    if row.get("pc1_evr_delta") is not None:
        score += abs(float(row["pc1_evr_delta"]))
    return score


def top_moving_roles(artifacts: list[dict[str, Any]], final_artifact: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    final_loadings = final_artifact["aa_loadings"]
    first = artifacts[0]["aa_loadings"]
    rows = []
    for group_type, group_id in sorted(set(first).intersection(final_loadings)):
        if group_type != "role":
            continue
        start = first[(group_type, group_id)]
        final = final_loadings[(group_type, group_id)]
        rows.append(
            {
                "group_type": group_type,
                "group_id": group_id,
                "first_aa_loading": start,
                "final_aa_loading": final,
                "delta_to_final": final - start,
                "abs_delta_to_final": abs(final - start),
            }
        )
    return sorted(rows, key=lambda row: row["abs_delta_to_final"], reverse=True)[:top_k]


def build_markdown(
    trajectory_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    moving_roles: list[dict[str, Any]],
    final_revision: str,
) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Axis Trajectory Report",
        "",
        f"Final reference checkpoint: `{final_revision}`",
        "",
        "## Checkpoint Metrics",
        "",
        "| revision | step | AA vs final | AA vs previous | PC1 vs final | PC1 vs previous | AA-PC1 | PC1 EVR | AA loading corr vs final |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in trajectory_rows:
        lines.append(
            f"| {row['revision']} | {row['step']} | {fmt(row['aa_to_final_cosine'])} | "
            f"{fmt(row['aa_adjacent_cosine'])} | {fmt(row['pc1_to_final_cosine'])} | "
            f"{fmt(row['pc1_adjacent_cosine'])} | {fmt(row['aa_pc1_cosine'])} | "
            f"{fmt(row['pc1_explained_variance_ratio'])} | {fmt(row['aa_loading_to_final_corr'])} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Transition Windows",
            "",
            "| from | to | score | AA adjacent | PC1 adjacent | AA loading adjacent corr | AA-PC1 delta | PC1 EVR delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in transition_rows[:10]:
        lines.append(
            f"| {row['from_revision']} | {row['to_revision']} | {fmt(row['transition_score'])} | "
            f"{fmt(row['aa_adjacent_cosine'])} | {fmt(row['pc1_adjacent_cosine'])} | "
            f"{fmt(row['aa_loading_adjacent_corr'])} | {fmt(row['aa_pc1_cosine_delta'])} | "
            f"{fmt(row['pc1_evr_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Moving Roles By AA Loading",
            "",
            "| role | first AA loading | final AA loading | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in moving_roles:
        lines.append(
            f"| {row['group_id']} | {fmt(row['first_aa_loading'])} | "
            f"{fmt(row['final_aa_loading'])} | {fmt(row['delta_to_final'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This report identifies geometric transitions in the fixed-response activation geometry. It does not yet attribute those transitions to training sequences or validate steering effects.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze cross-checkpoint Assistant Axis and role-geometry trajectory.")
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--final-revision", default="step143000")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--output-variant", default="axis-trajectory-layer12")
    parser.add_argument("--run-id", default="coarse8-full-v0")
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--top-k-moving-roles", type=int, default=12)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = Path(".").resolve()
    run_dir = resolve_run_dir(args)
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for directory in [results_dir, meta_dir, checkpoints_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    status_path = meta_dir / "status.json"
    if status_path.exists() and not args.force_completed:
        status = load_json(status_path)
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    try:
        sweep_summary = load_json(args.sweep_summary)
        artifacts = load_checkpoint_artifacts(sweep_summary, repo_root)
        final_matches = [item for item in artifacts if item["revision"] == args.final_revision]
        if not final_matches:
            raise ValueError(f"final revision {args.final_revision!r} not found in sweep summary")
        final_artifact = final_matches[0]
        final_aa = final_artifact["aa_vector"]
        final_pc1 = final_artifact["pc1_vector"]
        final_aa_loadings = final_artifact["aa_loadings"]
        final_pc1_loadings = final_artifact["pc1_loadings"]

        trajectory_rows: list[dict[str, Any]] = []
        transition_rows: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        for item in artifacts:
            role_summary = item["role_summary"]
            row = {
                "revision": item["revision"],
                "step": item["step"],
                "aa_to_final_cosine": cosine(item["aa_vector"], final_aa),
                "aa_adjacent_cosine": cosine(previous["aa_vector"], item["aa_vector"]) if previous else None,
                "pc1_to_final_cosine": cosine(item["pc1_vector"], final_pc1),
                "pc1_adjacent_cosine": cosine(previous["pc1_vector"], item["pc1_vector"]) if previous else None,
                "aa_pc1_cosine": role_summary.get("aa_pc1_cosine"),
                "pc1_explained_variance_ratio": role_summary.get("pc1_explained_variance_ratio"),
                "aa_loading_to_final_corr": loading_corr(item["aa_loadings"], final_aa_loadings),
                "pc1_loading_to_final_corr": loading_corr(item["pc1_loadings"], final_pc1_loadings),
                "aa_loading_adjacent_corr": loading_corr(previous["aa_loadings"], item["aa_loadings"]) if previous else None,
                "pc1_loading_adjacent_corr": loading_corr(previous["pc1_loadings"], item["pc1_loadings"]) if previous else None,
            }
            trajectory_rows.append(row)

            if previous is not None:
                transition = {
                    "from_revision": previous["revision"],
                    "to_revision": item["revision"],
                    "from_step": previous["step"],
                    "to_step": item["step"],
                    "aa_adjacent_cosine": row["aa_adjacent_cosine"],
                    "pc1_adjacent_cosine": row["pc1_adjacent_cosine"],
                    "aa_loading_adjacent_corr": row["aa_loading_adjacent_corr"],
                    "pc1_loading_adjacent_corr": row["pc1_loading_adjacent_corr"],
                    "aa_pc1_cosine_delta": (
                        None
                        if previous["role_summary"].get("aa_pc1_cosine") is None or row["aa_pc1_cosine"] is None
                        else float(row["aa_pc1_cosine"]) - float(previous["role_summary"]["aa_pc1_cosine"])
                    ),
                    "pc1_evr_delta": (
                        None
                        if previous["role_summary"].get("pc1_explained_variance_ratio") is None or row["pc1_explained_variance_ratio"] is None
                        else float(row["pc1_explained_variance_ratio"]) - float(previous["role_summary"]["pc1_explained_variance_ratio"])
                    ),
                }
                transition["transition_score"] = transition_score(transition)
                transition_rows.append(transition)
            previous = item

        transition_rows = sorted(transition_rows, key=lambda row: row["transition_score"], reverse=True)
        moving_roles = top_moving_roles(artifacts, final_artifact, args.top_k_moving_roles)

        trajectory_csv = results_dir / "axis_trajectory.csv"
        transitions_csv = results_dir / "checkpoint_transitions.csv"
        moving_roles_csv = results_dir / "top_moving_roles.csv"
        summary_json = results_dir / "trajectory_summary.json"
        report_md = results_dir / "trajectory_report.md"

        write_csv(
            trajectory_csv,
            trajectory_rows,
            [
                "revision",
                "step",
                "aa_to_final_cosine",
                "aa_adjacent_cosine",
                "pc1_to_final_cosine",
                "pc1_adjacent_cosine",
                "aa_pc1_cosine",
                "pc1_explained_variance_ratio",
                "aa_loading_to_final_corr",
                "pc1_loading_to_final_corr",
                "aa_loading_adjacent_corr",
                "pc1_loading_adjacent_corr",
            ],
        )
        write_csv(
            transitions_csv,
            transition_rows,
            [
                "from_revision",
                "to_revision",
                "from_step",
                "to_step",
                "transition_score",
                "aa_adjacent_cosine",
                "pc1_adjacent_cosine",
                "aa_loading_adjacent_corr",
                "pc1_loading_adjacent_corr",
                "aa_pc1_cosine_delta",
                "pc1_evr_delta",
            ],
        )
        write_csv(
            moving_roles_csv,
            moving_roles,
            ["group_type", "group_id", "first_aa_loading", "final_aa_loading", "delta_to_final", "abs_delta_to_final"],
        )
        summary = {
            "schema_version": "0.1",
            "final_revision": args.final_revision,
            "checkpoint_count": len(artifacts),
            "trajectory_csv": str(trajectory_csv),
            "transitions_csv": str(transitions_csv),
            "top_moving_roles_csv": str(moving_roles_csv),
            "report": str(report_md),
            "candidate_transition_windows": transition_rows[:5],
        }
        write_json(summary_json, summary)
        report_md.write_text(build_markdown(trajectory_rows, transition_rows, moving_roles, args.final_revision), encoding="utf-8")
        write_json(
            meta_dir / "run_manifest.json",
            {
                "schema_version": "0.1",
                "analyzer": "AxisTrajectoryAnalyzer",
                "created_at_utc": utc_now(),
                "run_dir": str(run_dir),
                "sweep_summary": str(args.sweep_summary),
                "final_revision": args.final_revision,
                "outputs": summary,
            },
        )
        write_json(
            checkpoints_dir / "progress.json",
            {
                "schema_version": "0.1",
                "state": "completed",
                "updated_at_utc": utc_now(),
                "completed_steps": ["loaded_sweep", "computed_trajectory", "wrote_outputs"],
            },
        )
        write_json(
            status_path,
            {
                "schema_version": "0.1",
                "state": "completed",
                "message": "axis trajectory analysis completed",
                "updated_at_utc": utc_now(),
                "counts": {"checkpoints": len(artifacts), "transitions": len(transition_rows)},
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "summary": str(summary_json),
                    "report": str(report_md),
                    "trajectory_csv": str(trajectory_csv),
                    "transitions_csv": str(transitions_csv),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        write_json(
            status_path,
            {
                "schema_version": "0.1",
                "state": "failed",
                "message": f"axis trajectory analysis failed: {type(exc).__name__}: {exc}",
                "updated_at_utc": utc_now(),
                "counts": {},
            },
        )
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
