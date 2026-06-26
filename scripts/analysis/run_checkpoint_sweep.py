#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def make_progress_bar(total: int, enabled: bool) -> Any:
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc="checkpoint sweep", unit="stage", dynamic_ncols=True)


def run_dir(
    output_root: Path,
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    probe_set: str,
    variant: str,
    run_id: str,
) -> Path:
    return output_root / experiment_name / model_name / dataset_name / probe_set / variant / run_id


def subprocess_run(command: list[str], log_path: Path, dry_run: bool) -> None:
    append_jsonl(log_path, {"time_utc": utc_now(), "event": "command_start", "command": command})
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        append_jsonl(log_path, {"time_utc": utc_now(), "event": "command_dry_run", "command": command})
        return
    result = subprocess.run(command)
    append_jsonl(
        log_path,
        {
            "time_utc": utc_now(),
            "event": "command_finish",
            "returncode": result.returncode,
            "command": command,
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")


def checkpoint_specs(config: dict[str, Any], requested_revisions: set[str] | None) -> list[dict[str, Any]]:
    raw = config.get("checkpoint_sweep", {}).get("checkpoints", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("experiment config must define checkpoint_sweep.checkpoints")
    specs = []
    for item in raw:
        if not isinstance(item, dict) or "revision" not in item:
            raise ValueError(f"invalid checkpoint item: {item!r}")
        if requested_revisions is None or str(item["revision"]) in requested_revisions:
            specs.append(item)
    if not specs:
        raise ValueError("no checkpoints selected")
    return specs


def stage_paths(args: argparse.Namespace, revision: str) -> dict[str, Path | str]:
    layer_variant = f"response-token-mean-layer{args.layer}"
    aa_variant = f"aa-main-layer{args.layer}"
    role_variant = f"role-geometry-layer{args.layer}"
    report_variant = f"geometry-report-layer{args.layer}"

    activation_run_id = f"activation-{revision}-layer{args.layer}-full-v0"
    aa_run_id = f"aa-main-{revision}-layer{args.layer}-full-v0"
    role_run_id = f"role-geometry-{revision}-layer{args.layer}-full-v0"
    report_run_id = f"geometry-report-{revision}-layer{args.layer}-full-v0"

    activation_dir = run_dir(
        args.output_root,
        args.experiment_name,
        args.model_name,
        args.dataset_name,
        args.probe_set,
        layer_variant,
        activation_run_id,
    )
    aa_dir = run_dir(
        args.output_root,
        args.experiment_name,
        args.model_name,
        args.dataset_name,
        args.probe_set,
        aa_variant,
        aa_run_id,
    )
    role_dir = run_dir(
        args.output_root,
        args.experiment_name,
        args.model_name,
        args.dataset_name,
        args.probe_set,
        role_variant,
        role_run_id,
    )
    report_dir = run_dir(
        args.output_root,
        args.experiment_name,
        args.model_name,
        args.dataset_name,
        args.probe_set,
        report_variant,
        report_run_id,
    )
    return {
        "activation_run_id": activation_run_id,
        "aa_run_id": aa_run_id,
        "role_run_id": role_run_id,
        "report_run_id": report_run_id,
        "activation_dir": activation_dir,
        "aa_dir": aa_dir,
        "role_dir": role_dir,
        "report_dir": report_dir,
    }


def status_is_completed(run_path: Path) -> bool:
    status = load_json(run_path / "meta" / "status.json")
    return bool(status and status.get("state") == "completed")


def command_with_optional_cache(command: list[str], hf_cache_dir: Path | None) -> list[str]:
    if hf_cache_dir is None:
        return command
    return command + ["--hf-cache-dir", str(hf_cache_dir)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run activation, AA, role-geometry, and report stages over checkpoints.")
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiments/pythia_410m_mvp_v0.yaml"))
    parser.add_argument("--response-jsonl", type=Path, default=None)
    parser.add_argument("--checkpoints", default=None, help="Comma-separated checkpoint revisions. Defaults to config coarse list.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--sweep-variant", default="checkpoint-sweep-layer12")
    parser.add_argument("--sweep-run-id", default="coarse8-full-v0")
    parser.add_argument("--model-id", default="EleutherAI/pythia-410m-deduped")
    parser.add_argument("--axis-variant-id", default="aa_main")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--activation-batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.layer < 0:
        raise SystemExit("--layer must be non-negative")
    if args.activation_batch_size < 1:
        raise SystemExit("--activation-batch-size must be positive")
    if args.save_every < 1:
        raise SystemExit("--save-every must be positive")

    config = load_yaml(args.experiment_config)
    response_jsonl = args.response_jsonl or Path(config["fixed_response_policy"]["generated_response_jsonl"])
    requested = set(args.checkpoints.split(",")) if args.checkpoints else None
    specs = checkpoint_specs(config, requested)

    sweep_dir = run_dir(
        args.output_root,
        args.experiment_name,
        args.model_name,
        args.dataset_name,
        args.probe_set,
        args.sweep_variant,
        args.sweep_run_id,
    )
    meta_dir = sweep_dir / "meta"
    checkpoints_dir = sweep_dir / "checkpoints"
    results_dir = sweep_dir / "results"
    logs_dir = sweep_dir / "logs"
    for directory in [meta_dir, checkpoints_dir, results_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"
    summary_path = results_dir / "checkpoint_sweep_summary.json"

    if status_is_completed(sweep_dir) and not args.force_completed:
        print(json.dumps({"status": "skipped_completed", "run_dir": str(sweep_dir)}, indent=2))
        return 0

    write_json(
        status_path,
        {
            "schema_version": "0.1",
            "state": "running",
            "message": "checkpoint sweep started",
            "updated_at_utc": utc_now(),
            "counts": {"checkpoints": len(specs), "completed_checkpoints": 0},
        },
    )
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "runner": "CheckpointSweepRunner",
            "created_at_utc": utc_now(),
            "run_dir": str(sweep_dir),
            "experiment_config": str(args.experiment_config),
            "response_jsonl": str(response_jsonl),
            "model_id": args.model_id,
            "layer": args.layer,
            "activation_batch_size": args.activation_batch_size,
            "save_every": args.save_every,
            "checkpoints": specs,
        },
    )

    total_stages = len(specs) * 5
    progress_bar = make_progress_bar(total_stages, enabled=not args.no_progress)
    completed: list[dict[str, Any]] = []
    failed_stage: dict[str, Any] | None = None
    try:
        for spec in specs:
            revision = str(spec["revision"])
            paths = stage_paths(args, revision)
            stage_record: dict[str, Any] = {
                "revision": revision,
                "step": spec.get("step"),
                "reason": spec.get("reason"),
                "paths": {key: str(value) for key, value in paths.items()},
                "stages": {},
            }

            activation_cmd = [
                sys.executable,
                "scripts/activations/cache_rollout_activations.py",
                "--response-jsonl",
                str(response_jsonl),
                "--model-id",
                args.model_id,
                "--revision",
                revision,
                "--layer",
                str(args.layer),
                "--batch-size",
                str(args.activation_batch_size),
                "--save-every",
                str(args.save_every),
                "--torch-dtype",
                args.torch_dtype,
                "--device-map",
                args.device_map,
                "--run-id",
                str(paths["activation_run_id"]),
            ]
            activation_cmd = command_with_optional_cache(activation_cmd, args.hf_cache_dir)
            if args.no_progress:
                activation_cmd.append("--no-progress")
            subprocess_run(activation_cmd, log_path, args.dry_run)
            stage_record["stages"]["activation"] = "completed" if status_is_completed(paths["activation_dir"]) else "submitted"
            if progress_bar:
                progress_bar.update(1)

            inspect_cmd = [
                sys.executable,
                "scripts/activations/inspect_activation_run.py",
                "--run-dir",
                str(paths["activation_dir"]),
            ]
            subprocess_run(inspect_cmd, log_path, args.dry_run)
            stage_record["stages"]["activation_inspection"] = "completed"
            if progress_bar:
                progress_bar.update(1)

            aa_cmd = [
                sys.executable,
                "scripts/analysis/build_assistant_axis.py",
                "--activation-run-dir",
                str(paths["activation_dir"]),
                "--axis-variant-id",
                args.axis_variant_id,
                "--run-id",
                str(paths["aa_run_id"]),
            ]
            subprocess_run(aa_cmd, log_path, args.dry_run)
            stage_record["stages"]["assistant_axis"] = "completed" if status_is_completed(paths["aa_dir"]) else "submitted"
            if progress_bar:
                progress_bar.update(1)

            role_cmd = [
                sys.executable,
                "scripts/analysis/build_role_geometry.py",
                "--activation-run-dir",
                str(paths["activation_dir"]),
                "--assistant-axis-run-dir",
                str(paths["aa_dir"]),
                "--run-id",
                str(paths["role_run_id"]),
            ]
            subprocess_run(role_cmd, log_path, args.dry_run)
            stage_record["stages"]["role_geometry"] = "completed" if status_is_completed(paths["role_dir"]) else "submitted"
            if progress_bar:
                progress_bar.update(1)

            report_cmd = [
                sys.executable,
                "scripts/reporting/report_geometry.py",
                "--assistant-axis-run-dir",
                str(paths["aa_dir"]),
                "--role-geometry-run-dir",
                str(paths["role_dir"]),
                "--run-id",
                str(paths["report_run_id"]),
            ]
            subprocess_run(report_cmd, log_path, args.dry_run)
            metrics_path = Path(paths["report_dir"]) / "results" / "geometry_metrics.json"
            metrics = load_json(metrics_path) if metrics_path.exists() else None
            stage_record["stages"]["geometry_report"] = "completed" if status_is_completed(paths["report_dir"]) else "submitted"
            stage_record["metrics"] = metrics or {}
            completed.append(stage_record)
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({"checkpoint": revision}, refresh=True)

            write_json(
                progress_path,
                {
                    "schema_version": "0.1",
                    "updated_at_utc": utc_now(),
                    "completed_checkpoints": [item["revision"] for item in completed],
                    "remaining_checkpoints": [str(item["revision"]) for item in specs if str(item["revision"]) not in {c["revision"] for c in completed}],
                },
            )
            write_json(summary_path, {"schema_version": "0.1", "checkpoints": completed})

        if args.dry_run:
            final_state = "dry_run"
            final_message = "checkpoint sweep dry run completed; no stage commands were executed"
        else:
            final_state = "completed"
            final_message = "checkpoint sweep completed"
    except Exception as exc:
        final_state = "failed"
        final_message = f"checkpoint sweep failed: {type(exc).__name__}: {exc}"
        failed_stage = {"error_type": type(exc).__name__, "message": str(exc)}
        append_jsonl(log_path, {"time_utc": utc_now(), "event": "error", **failed_stage})
    finally:
        if progress_bar:
            progress_bar.close()

    write_json(summary_path, {"schema_version": "0.1", "checkpoints": completed, "failed_stage": failed_stage})
    write_json(
        status_path,
        {
            "schema_version": "0.1",
            "state": final_state,
            "message": final_message,
            "updated_at_utc": utc_now(),
            "counts": {"checkpoints": len(specs), "completed_checkpoints": len(completed)},
        },
    )
    print(
        json.dumps(
            {
                "status": final_state,
                "message": final_message,
                "run_dir": str(sweep_dir),
                "summary": str(summary_path),
                "completed_checkpoints": len(completed),
                "selected_checkpoints": len(specs),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_state == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
