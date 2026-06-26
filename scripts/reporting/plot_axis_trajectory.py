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


BLUE = "#2f6fae"
LIGHT_BLUE = "#79b7e8"
GREEN = "#4f9a5b"
GOLD = "#c8aa00"
ORANGE = "#d9902f"
RED = "#b84a4a"
GRAY = "#6b7280"
DARK = "#1f2937"
GRID = "#e7eaf0"
SUBTLE = "#f6f7f9"


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


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


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


def setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
        }
    )
    return plt


def clean_axes(ax: Any) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cfd5dd")
    ax.spines["bottom"].set_color("#cfd5dd")


def checkpoint_x(step: int) -> float:
    return math.log2(step + 1)


def checkpoint_tick_labels(steps: list[int], labels: list[str]) -> tuple[list[float], list[str]]:
    return [checkpoint_x(step) for step in steps], labels


def add_checkpoint_bands(ax: Any, xs: list[float]) -> None:
    if len(xs) < 3:
        return
    for index in range(0, len(xs) - 1, 2):
        ax.axvspan(xs[index], xs[index + 1], color=SUBTLE, alpha=0.75, zorder=0)


def save_line_plot(
    plt: Any,
    path: Path,
    steps: list[int],
    labels: list[str],
    series: list[tuple[str, list[float | None], str, str]],
    title: str,
    ylabel: str,
    subtitle: str | None = None,
) -> None:
    xs, tick_labels = checkpoint_tick_labels(steps, labels)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    add_checkpoint_bands(ax, xs)
    for name, values, color, marker in series:
        y = [float("nan") if value is None else value for value in values]
        ax.plot(xs, y, marker=marker, linewidth=2.2, markersize=5.5, color=color, label=name, zorder=3)
    ax.set_title(title)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, color=GRAY, fontsize=9, va="bottom")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Pythia checkpoint, spaced by log2(step + 1)")
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=28, ha="right", fontsize=9)
    ax.margins(x=0.025)
    if "cosine" in ylabel.lower() or "correlation" in ylabel.lower():
        ax.axhline(0.0, color="#a9b2bd", linewidth=0.9, zorder=1)
        ax.set_ylim(bottom=-1.05, top=1.05)
    clean_axes(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_transition_plot(plt: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    plot_rows = list(reversed(rows))
    labels = [f"{row['from_revision']} -> {row['to_revision']}" for row in plot_rows]
    values = [as_float(row.get("transition_score")) or 0.0 for row in plot_rows]
    colors = [ORANGE if index >= len(values) - 3 else "#d7dee8" for index in range(len(values))]
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.38 * len(labels))))
    ax.barh(labels, values, color=colors)
    ax.set_title("Candidate Transition Windows")
    ax.set_xlabel("transition score; higher means more geometry change")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    for label, value in zip(labels, values, strict=True):
        ax.text(value, label, f" {value:.2f}", va="center", ha="left", fontsize=8, color=GRAY)
    clean_axes(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_moving_roles_plot(plt: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    plot_rows = list(reversed(rows))
    labels = [row["group_id"] for row in plot_rows]
    values = [as_float(row.get("delta_to_final")) or 0.0 for row in plot_rows]
    colors = [BLUE if value >= 0 else RED for value in values]
    fig, ax = plt.subplots(figsize=(9, max(4.8, 0.36 * len(labels))))
    ax.barh(labels, values, color=colors)
    ax.axvline(0.0, color="#374151", linewidth=1)
    ax.set_title("Top Moving Roles By AA Loading")
    ax.set_xlabel("final AA loading - first AA loading")
    clean_axes(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_markdown(plot_paths: dict[str, Path], summary: dict[str, Any]) -> str:
    lines = [
        "# Axis Trajectory Plot Pack",
        "",
        f"Source trajectory run: `{summary.get('source_trajectory_run_dir')}`",
        "",
        "## Plots",
        "",
    ]
    for label, path in plot_paths.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Reading Order",
            "",
            "1. Start with cosine trajectory to see when AA/PC1 align with the final checkpoint.",
            "2. Check geometry quality to see whether AA-PC1 and PC1 EVR are strong at each checkpoint.",
            "3. Use loading correlations and transition scores to identify candidate emergence or refinement windows.",
            "4. Inspect moving roles to understand which role groups reorganize most along the final AA.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot cross-checkpoint Assistant Axis trajectory artifacts.")
    parser.add_argument("--trajectory-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--output-variant", default="axis-trajectory-plots-layer12")
    parser.add_argument("--run-id", default="coarse8-full-v0")
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = Path(".").resolve()
    trajectory_run_dir = args.trajectory_run_dir
    if not trajectory_run_dir.is_absolute():
        trajectory_run_dir = repo_root / trajectory_run_dir
    trajectory_results = trajectory_run_dir / "results"
    trajectory_csv = trajectory_results / "axis_trajectory.csv"
    transitions_csv = trajectory_results / "checkpoint_transitions.csv"
    moving_roles_csv = trajectory_results / "top_moving_roles.csv"
    trajectory_summary = trajectory_results / "trajectory_summary.json"

    run_dir = resolve_run_dir(args)
    results_dir = run_dir / "results"
    plots_dir = results_dir / "plots"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    for directory in [results_dir, plots_dir, meta_dir, checkpoints_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    status_path = meta_dir / "status.json"
    if status_path.exists() and not args.force_completed:
        status = load_json(status_path)
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    trajectory_rows = load_csv(trajectory_csv)
    transition_rows = load_csv(transitions_csv)
    moving_role_rows = load_csv(moving_roles_csv)
    summary = load_json(trajectory_summary)

    trajectory_rows = sorted(trajectory_rows, key=lambda row: as_int(row.get("step")) or -1)
    transition_rows = sorted(transition_rows, key=lambda row: as_float(row.get("transition_score")) or 0.0, reverse=True)
    moving_role_rows = sorted(moving_role_rows, key=lambda row: as_float(row.get("abs_delta_to_final")) or 0.0, reverse=True)

    xs = [as_int(row["step"]) or 0 for row in trajectory_rows]
    labels = [row["revision"] for row in trajectory_rows]
    plt = setup_matplotlib()
    final_revision = summary.get("final_revision", "final")
    subtitle = f"Reference endpoint: {final_revision}; x-axis uses log spacing so early checkpoints remain readable."

    plot_paths = {
        "cosine trajectory": plots_dir / "cosine_trajectory.png",
        "geometry quality": plots_dir / "geometry_quality.png",
        "loading correlations": plots_dir / "loading_correlations.png",
        "transition scores": plots_dir / "transition_scores.png",
        "top moving roles": plots_dir / "top_moving_roles.png",
    }
    save_line_plot(
        plt,
        plot_paths["cosine trajectory"],
        xs,
        labels,
        [
            ("AA vs final", [as_float(row.get("aa_to_final_cosine")) for row in trajectory_rows], BLUE, "o"),
            ("AA vs previous", [as_float(row.get("aa_adjacent_cosine")) for row in trajectory_rows], LIGHT_BLUE, "s"),
            ("PC1 vs final", [as_float(row.get("pc1_to_final_cosine")) for row in trajectory_rows], GREEN, "^"),
            ("PC1 vs previous", [as_float(row.get("pc1_adjacent_cosine")) for row in trajectory_rows], GOLD, "D"),
        ],
        "AA and PC1 Cosine Trajectory",
        "cosine",
        subtitle,
    )
    save_line_plot(
        plt,
        plot_paths["geometry quality"],
        xs,
        labels,
        [
            ("AA-PC1 cosine", [as_float(row.get("aa_pc1_cosine")) for row in trajectory_rows], BLUE, "o"),
            ("PC1 explained variance", [as_float(row.get("pc1_explained_variance_ratio")) for row in trajectory_rows], ORANGE, "s"),
        ],
        "Per-Checkpoint Geometry Quality",
        "metric value",
        subtitle,
    )
    save_line_plot(
        plt,
        plot_paths["loading correlations"],
        xs,
        labels,
        [
            ("AA loadings vs final", [as_float(row.get("aa_loading_to_final_corr")) for row in trajectory_rows], BLUE, "o"),
            ("PC1 loadings vs final", [as_float(row.get("pc1_loading_to_final_corr")) for row in trajectory_rows], GREEN, "s"),
            ("AA loadings vs previous", [as_float(row.get("aa_loading_adjacent_corr")) for row in trajectory_rows], LIGHT_BLUE, "^"),
            ("PC1 loadings vs previous", [as_float(row.get("pc1_loading_adjacent_corr")) for row in trajectory_rows], GOLD, "D"),
        ],
        "Role Loading Correlations",
        "Pearson correlation",
        subtitle,
    )
    save_transition_plot(plt, plot_paths["transition scores"], transition_rows)
    save_moving_roles_plot(plt, plot_paths["top moving roles"], moving_role_rows[:20])

    summary["source_trajectory_run_dir"] = str(trajectory_run_dir)
    report_path = results_dir / "plot_report.md"
    report_path.write_text(build_markdown(plot_paths, summary), encoding="utf-8")
    manifest = {
        "schema_version": "0.1",
        "plotter": "AxisTrajectoryPlotter",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "source_trajectory_run_dir": str(trajectory_run_dir),
        "inputs": {
            "trajectory_csv": str(trajectory_csv),
            "transitions_csv": str(transitions_csv),
            "moving_roles_csv": str(moving_roles_csv),
            "trajectory_summary": str(trajectory_summary),
        },
        "outputs": {label: str(path) for label, path in plot_paths.items()} | {"plot_report": str(report_path)},
    }
    write_json(meta_dir / "run_manifest.json", manifest)
    write_json(
        checkpoints_dir / "progress.json",
        {
            "schema_version": "0.1",
            "state": "completed",
            "updated_at_utc": utc_now(),
            "completed_steps": ["loaded_trajectory", "wrote_plots", "wrote_manifest"],
        },
    )
    write_json(
        status_path,
        {
            "schema_version": "0.1",
            "state": "completed",
            "message": "axis trajectory plots completed",
            "updated_at_utc": utc_now(),
            "counts": {"plots": len(plot_paths)},
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "plot_report": str(report_path),
                "plots": {label: str(path) for label, path in plot_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
