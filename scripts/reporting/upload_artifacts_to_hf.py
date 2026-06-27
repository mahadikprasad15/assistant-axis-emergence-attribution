#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS: list[dict[str, Any]] = [
    {
        "kind": "validated_fixed_responses",
        "path": "data/rollouts/assistant_axis_rollouts_v0_responses.jsonl",
        "required": True,
    },
    {
        "kind": "validated_fixed_response_manifest",
        "path": "data/rollouts/assistant_axis_rollouts_v0_responses_manifest.json",
        "required": True,
    },
    {
        "kind": "rolefaithful_fixed_response_generation_run",
        "path": "artifacts/runs/assistant_axis_attribution/fixed-response-generator/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/llama-3.2-1b-instruct-rolefaithful/llama-3.2-1b-rolefaithful-full-v1",
        "required": True,
    },
    {
        "kind": "final_checkpoint_activation_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/response-token-mean-layer12/activation-step143000-layer12-full-v0",
        "required": True,
    },
    {
        "kind": "final_checkpoint_assistant_axis_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/aa-main-layer12/aa-main-step143000-layer12-full-v0",
        "required": True,
    },
    {
        "kind": "final_checkpoint_role_geometry_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/role-geometry-layer12/role-geometry-step143000-layer12-full-v0",
        "required": True,
    },
    {
        "kind": "final_checkpoint_geometry_report_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/geometry-report-layer12/geometry-report-step143000-layer12-full-v0",
        "required": True,
    },
    {
        "kind": "checkpoint_sweep_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/coarse8-full-v0",
        "required": True,
    },
    {
        "kind": "axis_trajectory_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/coarse8-full-v0",
        "required": True,
    },
    {
        "kind": "axis_trajectory_plots_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/coarse8-full-v0",
        "required": True,
    },
    {
        "kind": "early_dense_checkpoint_sweep_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/early-dense-0-1000-full-v0",
        "required": False,
    },
    {
        "kind": "early_dense_axis_trajectory_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/early-dense-0-1000-full-v0",
        "required": False,
    },
    {
        "kind": "early_dense_axis_trajectory_plots_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/early-dense-0-1000-full-v0",
        "required": False,
    },
    {
        "kind": "dense_1000_5000_checkpoint_sweep_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/checkpoint-sweep-layer12/dense-1000-5000-full-v1",
        "required": False,
    },
    {
        "kind": "dense_1000_5000_axis_trajectory_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-layer12/dense-1000-5000-full-v1",
        "required": False,
    },
    {
        "kind": "dense_1000_5000_axis_trajectory_plots_run",
        "path": "artifacts/runs/assistant_axis_attribution/pythia-410m-deduped/fixed-aa-rollouts-v0/assistant-axis-rollouts-v0/axis-trajectory-plots-layer12/dense-1000-5000-full-v1",
        "required": False,
    },
]

SKIP_PARTS = {".cache", ".venv", "__pycache__"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not should_skip(child):
            total += child.stat().st_size
    return total


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def sweep_dependency_records(sweep_path: Path) -> list[dict[str, Any]]:
    summary_path = sweep_path / "results" / "checkpoint_sweep_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    stage_keys = {
        "activation_dir": "activation_run",
        "aa_dir": "assistant_axis_run",
        "role_dir": "role_geometry_run",
        "report_dir": "geometry_report_run",
    }
    for checkpoint in summary.get("checkpoints", []):
        revision = str(checkpoint.get("revision", "unknown"))
        paths = checkpoint.get("paths", {})
        for path_key, stage_name in stage_keys.items():
            if paths.get(path_key):
                records.append(
                    {
                        "kind": f"sweep_{revision}_{stage_name}",
                        "path": str(paths[path_key]),
                        "required": True,
                        "discovered_from": str(summary_path),
                    }
                )
    return records


def expand_sweep_dependencies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = list(records)
    for item in records:
        if "checkpoint_sweep_run" not in str(item["kind"]):
            continue
        sweep_path = Path(item["path"])
        if sweep_path.exists():
            expanded.extend(sweep_dependency_records(sweep_path))
    deduplicated: dict[str, dict[str, Any]] = {}
    for item in expanded:
        deduplicated[str(Path(item["path"]))] = item
    return list(deduplicated.values())


def collect_artifacts(allow_missing: bool, artifact_specs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing_required: list[str] = []
    for item in expand_sweep_dependencies(artifact_specs or DEFAULT_ARTIFACTS):
        path = Path(item["path"])
        exists = path.exists()
        if item["required"] and not exists:
            missing_required.append(str(path))
        if exists:
            records.append(
                {
                    **item,
                    "exists": True,
                    "is_dir": path.is_dir(),
                    "size_bytes": file_size(path),
                }
            )
        else:
            records.append({**item, "exists": False, "is_dir": False, "size_bytes": 0})
    if missing_required and not allow_missing:
        joined = "\n".join(f"- {path}" for path in missing_required)
        raise FileNotFoundError(f"missing required upload artifacts:\n{joined}")
    return records


def expected_repo_files(prefix: str, local_path: Path) -> list[str]:
    repo_path = path_in_repo(prefix, local_path)
    if local_path.is_file():
        return [repo_path]
    return [
        f"{repo_path}/{child.relative_to(local_path).as_posix()}"
        for child in local_path.rglob("*")
        if child.is_file() and not should_skip(child) and child.name != ".DS_Store"
    ]


def verify_remote_artifacts(args: argparse.Namespace, artifact_records: list[dict[str, Any]]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN")
    remote_files = set(HfApi(token=token).list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type))
    expected: list[str] = []
    for record in artifact_records:
        if record["exists"]:
            expected.extend(expected_repo_files(args.path_in_repo, Path(record["path"])))
    missing = sorted(set(expected) - remote_files)
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:20])
        raise RuntimeError(f"remote verification failed; {len(missing)} uploaded files are missing:\n{preview}")
    return {"expected_file_count": len(set(expected)), "missing_file_count": 0, "verified": True}


def path_in_repo(prefix: str, local_path: Path) -> str:
    local = local_path.as_posix().lstrip("/")
    prefix = prefix.strip("/")
    return f"{prefix}/{local}" if prefix else local


def upload_artifacts(args: argparse.Namespace, artifact_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, create_repo

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF token missing. Set HF_TOKEN or pass --token.")

    if args.create_repo:
        create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
            token=token,
        )

    api = HfApi(token=token)
    uploaded: list[dict[str, Any]] = []
    for record in artifact_records:
        if not record["exists"]:
            uploaded.append({**record, "uploaded": False, "reason": "missing"})
            continue
        local_path = Path(record["path"])
        repo_path = path_in_repo(args.path_in_repo, local_path)
        if args.dry_run:
            uploaded.append({**record, "uploaded": False, "dry_run": True, "path_in_repo": repo_path})
            continue
        if local_path.is_dir():
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(local_path),
                path_in_repo=repo_path,
                ignore_patterns=["*/__pycache__/*", "__pycache__/*", ".DS_Store"],
                commit_message=f"Upload {record['kind']}",
            )
        else:
            api.upload_file(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                commit_message=f"Upload {record['kind']}",
            )
        uploaded.append({**record, "uploaded": True, "path_in_repo": repo_path})
    return uploaded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload curated Assistant Axis MVP artifacts to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example username/dataset-name.")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--path-in-repo", default="pythia410m-mvp-v0")
    parser.add_argument("--token", default=None, help="Optional token. Prefer HF_TOKEN env var.")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-repo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--sweep-run-dir",
        type=Path,
        action="append",
        default=[],
        help="Upload only this sweep run and every checkpoint stage it references. Repeat for multiple sweeps.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        action="append",
        default=[],
        help="Upload only this explicit file or directory. Repeat for multiple artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--output-variant", default="hf-upload")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_id = args.run_id or default_run_id()
    run_dir = (
        args.output_root
        / args.experiment_name
        / args.model_name
        / args.dataset_name
        / args.probe_set
        / args.output_variant
        / run_id
    )
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    for directory in [results_dir, meta_dir, checkpoints_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    status_path = meta_dir / "status.json"
    manifest_path = results_dir / "hf_upload_manifest.json"
    try:
        artifact_specs = None
        if args.sweep_run_dir or args.artifact_path:
            artifact_specs = [
                {"kind": f"explicit_checkpoint_sweep_run_{index}", "path": str(path), "required": True}
                for index, path in enumerate(args.sweep_run_dir)
            ]
            artifact_specs.extend(
                {"kind": f"explicit_artifact_{index}", "path": str(path), "required": True}
                for index, path in enumerate(args.artifact_path)
            )
        artifact_records = collect_artifacts(args.allow_missing, artifact_specs)
        uploaded = upload_artifacts(args, artifact_records)
        remote_verification = None if args.dry_run else verify_remote_artifacts(args, artifact_records)
        total_bytes = sum(item["size_bytes"] for item in artifact_records if item["exists"])
        manifest = {
            "schema_version": "0.1",
            "uploader": "HFArtifactUploader",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "repo_id": args.repo_id,
            "repo_type": args.repo_type,
            "path_in_repo": args.path_in_repo,
            "private": args.private,
            "dry_run": args.dry_run,
            "artifact_count": len(artifact_records),
            "total_size_bytes": total_bytes,
            "artifacts": uploaded,
            "remote_verification": remote_verification,
        }
        write_json(manifest_path, manifest)
        write_json(
            meta_dir / "run_manifest.json",
            {
                "schema_version": "0.1",
                "runner": "HFArtifactUploader",
                "created_at_utc": utc_now(),
                "run_dir": str(run_dir),
                "repo_id": args.repo_id,
                "repo_type": args.repo_type,
                "path_in_repo": args.path_in_repo,
                "manifest": str(manifest_path),
            },
        )
        write_json(
            checkpoints_dir / "progress.json",
            {
                "schema_version": "0.1",
                "state": "completed",
                "updated_at_utc": utc_now(),
                "completed_artifacts": [item["kind"] for item in uploaded if item.get("uploaded") or item.get("dry_run")],
            },
        )
        write_json(
            status_path,
            {
                "schema_version": "0.1",
                "state": "completed",
                "message": "HF artifact upload completed" if not args.dry_run else "HF artifact upload dry run completed",
                "updated_at_utc": utc_now(),
                "counts": {"artifacts": len(uploaded)},
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "dry_run": args.dry_run,
                    "repo_id": args.repo_id,
                    "repo_type": args.repo_type,
                    "path_in_repo": args.path_in_repo,
                    "run_dir": str(run_dir),
                    "manifest": str(manifest_path),
                    "artifact_count": len(uploaded),
                    "total_size_bytes": total_bytes,
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
                "message": f"HF artifact upload failed: {type(exc).__name__}: {exc}",
                "updated_at_utc": utc_now(),
                "counts": {},
            },
        )
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
