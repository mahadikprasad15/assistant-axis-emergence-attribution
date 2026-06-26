#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no} in {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no} in {path} must be a JSON object")
            records.append(record)
    return records


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


def resolve_artifact_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def validate_single_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("activation index contains no rows")
    context: dict[str, Any] = {}
    for field in ["model_id", "checkpoint_revision", "layer", "pooling_policy"]:
        values = {row.get(field) for row in rows}
        if len(values) != 1:
            raise ValueError(f"activation index mixes multiple {field} values: {sorted(map(str, values))}")
        context[field] = next(iter(values))
    return context


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("record_type") == "default":
        return "default_prompt", str(row.get("default_prompt_id"))
    if row.get("record_type") == "role":
        return "role", str(row.get("role_id"))
    raise ValueError(f"unsupported record_type for rollout_id={row.get('rollout_id')}: {row.get('record_type')}")


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    return dict(grouped)


def load_group_vectors(grouped: dict[tuple[str, str], list[dict[str, Any]]], repo_root: Path) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], list[str]]]:
    import torch

    group_vectors: dict[tuple[str, str], Any] = {}
    group_rollout_ids: dict[tuple[str, str], list[str]] = {}
    for key, rows in grouped.items():
        vectors = []
        rollout_ids = []
        missing_paths = []
        for row in rows:
            activation_path = resolve_artifact_path(str(row.get("activation_path", "")), repo_root)
            if not activation_path.exists():
                missing_paths.append(str(row.get("activation_path", "")))
                continue
            vectors.append(torch.load(activation_path, map_location="cpu").float())
            rollout_ids.append(str(row["rollout_id"]))
        if missing_paths:
            raise FileNotFoundError(f"missing activation tensor paths for {key}: {missing_paths[:10]}")
        if not vectors:
            raise ValueError(f"group has no vectors: {key}")
        shapes = {tuple(vector.shape) for vector in vectors}
        if len(shapes) != 1:
            raise ValueError(f"group {key} has mixed activation shapes: {sorted(map(str, shapes))}")
        group_vectors[key] = torch.stack(vectors, dim=0).mean(dim=0)
        group_rollout_ids[key] = rollout_ids
    return group_vectors, group_rollout_ids


def unit(vector: Any) -> Any:
    import torch

    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) <= 0:
        raise ValueError("cannot normalize zero vector")
    return vector / norm


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


def write_progress(path: Path, state: str, completed_steps: list[str]) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "updated_at_utc": utc_now(),
            "state": state,
            "completed_steps": completed_steps,
        },
    )


def write_role_vectors_jsonl(
    path: Path,
    group_vectors: dict[tuple[str, str], Any],
    group_rollout_ids: dict[tuple[str, str], list[str]],
    role_vector_dir: Path,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch

    records = []
    for (group_type, group_id), vector in sorted(group_vectors.items()):
        vector_path = role_vector_dir / f"{group_type}__{group_id}.pt"
        torch.save(vector, vector_path)
        record = {
            "schema_version": "0.1",
            "group_id": group_id,
            "group_type": group_type,
            "model_id": context["model_id"],
            "checkpoint_revision": context["checkpoint_revision"],
            "layer": context["layer"],
            "pooling_policy": context["pooling_policy"],
            "vector_path": str(vector_path),
            "vector_shape": list(vector.shape),
            "source_record_count": len(group_rollout_ids[(group_type, group_id)]),
            "source_rollout_ids": group_rollout_ids[(group_type, group_id)],
        }
        append_jsonl(path, record)
        records.append(record)
    return records


def write_loadings_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_type",
        "group_id",
        "source_record_count",
        "pc1_loading",
        "aa_loading",
        "pc1_abs_loading_rank",
        "aa_abs_loading_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build role/default geometry from activation and AA runs.")
    parser.add_argument("--activation-run-dir", type=Path, required=True)
    parser.add_argument("--assistant-axis-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--output-variant", default="role-geometry-layer12")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = Path(".").resolve()
    activation_run_dir = args.activation_run_dir
    assistant_axis_run_dir = args.assistant_axis_run_dir
    if not activation_run_dir.is_absolute():
        activation_run_dir = repo_root / activation_run_dir
    if not assistant_axis_run_dir.is_absolute():
        assistant_axis_run_dir = repo_root / assistant_axis_run_dir

    activation_index_path = activation_run_dir / "results" / "activation_index.jsonl"
    activation_manifest_path = activation_run_dir / "meta" / "run_manifest.json"
    aa_summary_path = assistant_axis_run_dir / "results" / "assistant_axis_summary.json"

    run_dir = resolve_run_dir(args)
    inputs_dir = run_dir / "inputs"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    role_vector_dir = results_dir / "role_vectors"
    logs_dir = run_dir / "logs"
    meta_dir = run_dir / "meta"
    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"

    for directory in [inputs_dir, checkpoints_dir, results_dir, role_vector_dir, logs_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if status_path.exists() and not args.force_completed:
        status = load_json(status_path) or {}
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    completed_steps: list[str] = []
    write_status(status_path, "running", "role geometry build started", {"groups": 0})
    append_log(log_path, "start", {"run_dir": str(run_dir), "activation_run_dir": str(activation_run_dir)})

    try:
        import torch

        rows = load_jsonl(activation_index_path)
        context = validate_single_context(rows)
        grouped = group_rows(rows)
        if len(grouped) < 2:
            raise ValueError("need at least two role/default groups for geometry")
        completed_steps.append("grouped_activation_rows")

        group_vectors, group_rollout_ids = load_group_vectors(grouped, repo_root)
        shapes = {tuple(vector.shape) for vector in group_vectors.values()}
        if len(shapes) != 1:
            raise ValueError(f"group mean vectors have mixed shapes: {sorted(map(str, shapes))}")
        completed_steps.append("built_group_vectors")

        aa_summary = load_json(aa_summary_path)
        if not aa_summary:
            raise FileNotFoundError(f"missing assistant axis summary: {aa_summary_path}")
        aa_vector_path = resolve_artifact_path(str(aa_summary["vector_path"]), repo_root)
        if not aa_vector_path.exists():
            raise FileNotFoundError(f"missing assistant axis vector: {aa_vector_path}")
        assistant_axis = torch.load(aa_vector_path, map_location="cpu").float()
        if tuple(assistant_axis.shape) not in shapes:
            raise ValueError(f"assistant axis shape {list(assistant_axis.shape)} does not match group vector shapes {shapes}")
        assistant_axis = unit(assistant_axis)
        completed_steps.append("loaded_assistant_axis")

        ordered_keys = sorted(group_vectors)
        matrix = torch.stack([group_vectors[key].float() for key in ordered_keys], dim=0)
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        pc1 = unit(vh[0].float())
        aa_pc1_cosine = float(torch.dot(assistant_axis, pc1).item())
        if aa_pc1_cosine < 0:
            pc1 = -pc1
            aa_pc1_cosine = -aa_pc1_cosine
        completed_steps.append("computed_pc1")

        role_vectors_path = results_dir / "role_vectors.jsonl"
        if role_vectors_path.exists():
            role_vectors_path.unlink()
        role_vector_records = write_role_vectors_jsonl(
            role_vectors_path,
            group_vectors,
            group_rollout_ids,
            role_vector_dir,
            context,
        )

        pc1_path = results_dir / "role_pc1.pt"
        summary_path = results_dir / "role_geometry_summary.json"
        loadings_path = results_dir / "role_loadings.csv"
        torch.save(pc1, pc1_path)

        loading_rows = []
        for key, vector in group_vectors.items():
            group_type, group_id = key
            vector_unit = unit(vector.float())
            loading_rows.append(
                {
                    "group_type": group_type,
                    "group_id": group_id,
                    "source_record_count": len(group_rollout_ids[key]),
                    "pc1_loading": float(torch.dot(vector_unit, pc1).item()),
                    "aa_loading": float(torch.dot(vector_unit, assistant_axis).item()),
                }
            )
        pc1_ranked = sorted(loading_rows, key=lambda row: abs(row["pc1_loading"]), reverse=True)
        aa_ranked = sorted(loading_rows, key=lambda row: abs(row["aa_loading"]), reverse=True)
        pc1_ranks = {(row["group_type"], row["group_id"]): idx + 1 for idx, row in enumerate(pc1_ranked)}
        aa_ranks = {(row["group_type"], row["group_id"]): idx + 1 for idx, row in enumerate(aa_ranked)}
        for row in loading_rows:
            key = (row["group_type"], row["group_id"])
            row["pc1_abs_loading_rank"] = pc1_ranks[key]
            row["aa_abs_loading_rank"] = aa_ranks[key]
        write_loadings_csv(loadings_path, loading_rows)
        completed_steps.append("wrote_outputs")

        explained_variance_ratio = None
        if len(singular_values) > 0 and float((singular_values**2).sum().item()) > 0:
            explained_variance_ratio = float(((singular_values[0] ** 2) / (singular_values**2).sum()).item())

        summary = {
            "schema_version": "0.1",
            "builder": "RoleGeometryBuilder",
            "model_id": context["model_id"],
            "checkpoint_revision": context["checkpoint_revision"],
            "layer": context["layer"],
            "pooling_policy": context["pooling_policy"],
            "group_count": len(group_vectors),
            "record_count": len(rows),
            "group_type_counts": dict(Counter(group_type for group_type, _ in group_vectors)),
            "vector_shape": list(matrix.shape[1:]),
            "pc1_path": str(pc1_path),
            "role_vectors_jsonl": str(role_vectors_path),
            "role_loadings_csv": str(loadings_path),
            "assistant_axis_summary": str(aa_summary_path),
            "assistant_axis_vector": str(aa_vector_path),
            "aa_pc1_cosine": aa_pc1_cosine,
            "pc1_explained_variance_ratio": explained_variance_ratio,
            "top_pc1_loadings": pc1_ranked[:10],
            "top_aa_loadings": aa_ranked[:10],
        }
        write_json(summary_path, summary)

        activation_manifest = load_json(activation_manifest_path)
        manifest = {
            "schema_version": "0.1",
            "builder": "RoleGeometryBuilder",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "activation_run": {
                "run_dir": str(activation_run_dir),
                "activation_index_jsonl": str(activation_index_path),
                "activation_index_sha256": file_sha256(activation_index_path),
                "run_manifest": activation_manifest,
            },
            "assistant_axis_run": {
                "run_dir": str(assistant_axis_run_dir),
                "summary": str(aa_summary_path),
                "summary_sha256": file_sha256(aa_summary_path),
                "vector_path": str(aa_vector_path),
            },
            "outputs": {
                "role_vectors_jsonl": str(role_vectors_path),
                "role_vector_records": len(role_vector_records),
                "role_pc1": str(pc1_path),
                "role_loadings_csv": str(loadings_path),
                "summary": str(summary_path),
            },
            "validation": {
                "passed": True,
                "errors": [],
                "warnings": [],
            },
        }
        write_json(manifest_path, manifest)
        write_progress(progress_path, "completed", completed_steps)
        write_status(status_path, "completed", "role geometry build completed", {"groups": len(group_vectors)})
        append_log(log_path, "completed", {"groups": len(group_vectors), "aa_pc1_cosine": aa_pc1_cosine})

        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "summary": str(summary_path),
                    "aa_pc1_cosine": aa_pc1_cosine,
                    "groups": len(group_vectors),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        write_progress(progress_path, "failed", completed_steps)
        write_status(status_path, "failed", f"role geometry build failed: {type(exc).__name__}: {exc}", {"groups": 0})
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})
        print(
            json.dumps(
                {
                    "status": "failed",
                    "run_dir": str(run_dir),
                    "message": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
