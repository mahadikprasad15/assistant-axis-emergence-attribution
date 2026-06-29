#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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


def stable_hash(values: list[Any]) -> str:
    payload = json.dumps(values, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_run_dir:
        return args.resume_run_dir
    return (
        args.output_root
        / args.experiment_name
        / args.model_name
        / args.dataset_name
        / args.probe_set
        / args.output_variant
        / (args.run_id or default_run_id())
    )


def select_axis_variant(axis_config: dict[str, Any], variant_id: str) -> dict[str, Any]:
    for variant in axis_config.get("axis_construction_variants", []):
        if variant.get("variant_id") == variant_id:
            return variant
    raise ValueError(f"axis variant not found: {variant_id}")


def validate_question_split(config: dict[str, Any]) -> tuple[set[int], set[int]]:
    split = config["question_split"]
    construction = {int(value) for value in split["construction_question_ids"]}
    evaluation = {int(value) for value in split["evaluation_question_ids"]}
    if not construction or not evaluation:
        raise ValueError("construction and evaluation question sets must be non-empty")
    overlap = construction & evaluation
    if overlap:
        raise ValueError(f"construction/evaluation question overlap: {sorted(overlap)}")
    return construction, evaluation


def activation_context(rows: list[dict[str, Any]], expected_revision: str, expected_layer: int) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"activation index for {expected_revision} is empty")
    context: dict[str, Any] = {}
    for field in ["model_id", "checkpoint_revision", "layer", "pooling_policy"]:
        values = {row.get(field) for row in rows}
        if len(values) != 1:
            raise ValueError(f"{expected_revision} activation index mixes {field}: {sorted(map(str, values))}")
        context[field] = next(iter(values))
    if context["checkpoint_revision"] != expected_revision:
        raise ValueError(
            f"expected activation revision {expected_revision}, found {context['checkpoint_revision']}"
        )
    if int(context["layer"]) != expected_layer:
        raise ValueError(f"expected layer {expected_layer}, found {context['layer']} at {expected_revision}")
    return context


def select_contrast_rows(
    rows: list[dict[str, Any]],
    question_ids: set[int],
    variant: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    default_ids = {str(value) for value in variant.get("default_prompt_ids", [])}
    contrast_groups = {str(value) for value in variant.get("contrast_role_groups", [])}
    scoped = [row for row in rows if int(row.get("question_id", -1)) in question_ids]
    defaults = [
        row
        for row in scoped
        if row.get("record_type") == "default" and str(row.get("default_prompt_id")) in default_ids
    ]
    contrasts = [
        row
        for row in scoped
        if row.get("record_type") == "role" and str(row.get("role_group")) in contrast_groups
    ]
    if not defaults:
        raise ValueError("question split selected no default records")
    if not contrasts:
        raise ValueError("question split selected no contrast records")
    return defaults, contrasts


def load_activation_matrix(rows: list[dict[str, Any]], repo_root: Path) -> tuple[Any, list[str]]:
    import torch

    vectors = []
    rollout_ids = []
    for row in rows:
        path = resolve_path(Path(str(row.get("activation_path", ""))), repo_root)
        if not path.exists():
            raise FileNotFoundError(f"activation tensor not found: {path}")
        vector = torch.load(path, map_location="cpu").float()
        vectors.append(vector)
        rollout_ids.append(str(row["rollout_id"]))
    shapes = {tuple(vector.shape) for vector in vectors}
    if len(shapes) != 1:
        raise ValueError(f"activation vectors have mixed shapes: {sorted(map(str, shapes))}")
    return torch.stack(vectors), rollout_ids


def build_axis(
    rows: list[dict[str, Any]],
    question_ids: set[int],
    variant: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    import torch

    default_rows, contrast_rows = select_contrast_rows(rows, question_ids, variant)
    default_vectors, default_ids = load_activation_matrix(default_rows, repo_root)
    contrast_vectors, contrast_ids = load_activation_matrix(contrast_rows, repo_root)
    default_mean = default_vectors.mean(dim=0)
    contrast_mean = contrast_vectors.mean(dim=0)
    raw_axis = default_mean - contrast_mean
    raw_norm = torch.linalg.vector_norm(raw_axis)
    if not torch.isfinite(raw_norm) or float(raw_norm.item()) <= 0:
        raise ValueError("Assistant Axis has a non-finite or zero norm")
    return {
        "axis": raw_axis / raw_norm,
        "raw_norm": float(raw_norm.item()),
        "default_mean": default_mean,
        "contrast_mean": contrast_mean,
        "default_rollout_ids": default_ids,
        "contrast_rollout_ids": contrast_ids,
        "counts": {"default_records": len(default_ids), "contrast_records": len(contrast_ids)},
    }


def build_innovation(endpoint_axis: Any, native_axis: Any) -> tuple[Any, float, float]:
    import torch

    projection = torch.dot(endpoint_axis, native_axis) * native_axis
    residual = endpoint_axis - projection
    residual_norm = torch.linalg.vector_norm(residual)
    if not torch.isfinite(residual_norm) or float(residual_norm.item()) <= 1e-8:
        raise ValueError("endpoint axis has no stable component orthogonal to native axis")
    innovation = residual / residual_norm
    orthogonality = float(torch.dot(innovation, native_axis).item())
    return innovation, float(residual_norm.item()), orthogonality


def validate_response_categories(
    responses: list[dict[str, Any]], construction: set[int], evaluation: set[int]
) -> dict[str, list[str]]:
    categories: dict[int, str] = {}
    for row in responses:
        question_id = int(row.get("question_id", -1))
        category = str(row.get("question_category", ""))
        if question_id in categories and categories[question_id] != category:
            raise ValueError(f"question {question_id} has inconsistent categories")
        categories[question_id] = category
    missing = sorted((construction | evaluation) - set(categories))
    if missing:
        raise ValueError(f"fixed responses are missing configured question ids: {missing}")
    construction_categories = sorted({categories[value] for value in construction})
    evaluation_categories = sorted({categories[value] for value in evaluation})
    if set(construction_categories) != set(evaluation_categories):
        raise ValueError(
            "construction/evaluation category coverage differs: "
            f"{construction_categories} vs {evaluation_categories}"
        )
    return {"construction": construction_categories, "evaluation": evaluation_categories}


def target_metadata(
    axis_name: str,
    revision: str,
    vector_path: Path,
    question_ids: list[int],
    raw_norm: float,
) -> dict[str, Any]:
    return {
        "axis_name": axis_name,
        "source_revision": revision,
        "vector_path": str(vector_path),
        "vector_norm": 1.0,
        "raw_axis_norm": raw_norm,
        "construction_split_hash": stable_hash(question_ids),
    }


def write_status(path: Path, state: str, message: str, counts: dict[str, int]) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "state": state,
            "message": message,
            "counts": counts,
            "updated_at_utc": utc_now(),
        },
    )


def write_progress(path: Path, state: str, completed_steps: list[str]) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "state": state,
            "completed_steps": completed_steps,
            "updated_at_utc": utc_now(),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build held-out Assistant Axis concept targets.")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/pythia_410m_concept_attribution_256_512_v0.yaml"),
    )
    parser.add_argument("--axis-config", type=Path, default=None)
    parser.add_argument("--axis-variant-id", default=None)
    parser.add_argument("--native-activation-run-dir", type=Path, required=True)
    parser.add_argument("--endpoint-activation-run-dir", type=Path, required=True)
    parser.add_argument("--final-activation-run-dir", type=Path, required=True)
    parser.add_argument(
        "--response-jsonl",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="pythia-410m-deduped")
    parser.add_argument("--dataset-name", default="pile-deduped-pythia-preshuffled")
    parser.add_argument("--probe-set", default="concept-attribution-256-512-v0")
    parser.add_argument("--output-variant", default="concept-target-bundle-layer12")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = Path(".").resolve()
    experiment_config_path = resolve_path(args.experiment_config, repo_root)
    run_dirs = {
        "native_step256": resolve_path(args.native_activation_run_dir, repo_root),
        "endpoint_step512": resolve_path(args.endpoint_activation_run_dir, repo_root),
        "final_step143000": resolve_path(args.final_activation_run_dir, repo_root),
    }

    run_dir = resolve_run_dir(args)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    results_dir = run_dir / "results"
    vectors_dir = results_dir / "vectors"
    means_dir = results_dir / "means"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    inputs_dir = run_dir / "inputs"
    for directory in [results_dir, vectors_dir, means_dir, meta_dir, checkpoints_dir, logs_dir, inputs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    status_path = meta_dir / "status.json"
    progress_path = checkpoints_dir / "progress.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"
    bundle_path = results_dir / "concept_target_bundle.json"
    evaluation_path = results_dir / "evaluation_records.jsonl"

    if status_path.exists() and not args.force_completed:
        status = load_json(status_path) or {}
        if status.get("state") == "completed" and bundle_path.exists() and evaluation_path.exists():
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    completed_steps: list[str] = []
    counts = {"targets": 0, "evaluation_records": 0}
    write_status(status_path, "running", "concept target bundle build started", counts)
    append_log(log_path, "start", {"run_dir": str(run_dir)})

    try:
        experiment_config = load_yaml(experiment_config_path)
        axis_settings = experiment_config["axis_construction"]
        axis_config_path = resolve_path(
            args.axis_config or Path(axis_settings["axis_config"]), repo_root
        )
        axis_variant_id = args.axis_variant_id or str(axis_settings["axis_variant_id"])
        response_jsonl = resolve_path(
            args.response_jsonl or Path(axis_settings["fixed_response_jsonl"]), repo_root
        )
        axis_config = load_yaml(axis_config_path)
        variant = select_axis_variant(axis_config, axis_variant_id)
        construction_ids, evaluation_ids = validate_question_split(experiment_config)
        model_config = experiment_config["model"]
        expected_layer = int(model_config["layer"])
        expected_revisions = {
            "native_step256": str(model_config["scoring_revision"]),
            "endpoint_step512": str(model_config["endpoint_revision"]),
            "final_step143000": str(model_config["final_revision"]),
        }
        completed_steps.append("loaded_configs")

        responses = load_jsonl(response_jsonl)
        category_coverage = validate_response_categories(responses, construction_ids, evaluation_ids)
        response_by_id = {str(row["rollout_id"]): row for row in responses}
        if len(response_by_id) != len(responses):
            raise ValueError("fixed response JSONL contains duplicate rollout ids")
        completed_steps.append("validated_question_split")

        builds: dict[str, dict[str, Any]] = {}
        activation_inputs: dict[str, dict[str, Any]] = {}
        for axis_name, activation_run_dir in run_dirs.items():
            index_path = activation_run_dir / "results" / "activation_index.jsonl"
            manifest_path_in = activation_run_dir / "meta" / "run_manifest.json"
            rows = load_jsonl(index_path)
            context = activation_context(rows, expected_revisions[axis_name], expected_layer)
            if str(context["model_id"]) != str(model_config["model_id"]):
                raise ValueError(f"unexpected model for {axis_name}: {context['model_id']}")
            builds[axis_name] = build_axis(rows, construction_ids, variant, repo_root)
            activation_inputs[axis_name] = {
                "run_dir": str(activation_run_dir),
                "activation_index": str(index_path),
                "activation_index_sha256": file_sha256(index_path),
                "run_manifest_sha256": file_sha256(manifest_path_in),
                "context": context,
            }
        completed_steps.append("built_construction_split_axes")

        import torch

        innovation, innovation_raw_norm, innovation_orthogonality = build_innovation(
            builds["endpoint_step512"]["axis"], builds["native_step256"]["axis"]
        )
        builds["innovation_256_to_512"] = {
            "axis": innovation,
            "raw_norm": innovation_raw_norm,
            "counts": builds["endpoint_step512"]["counts"],
        }

        vector_paths: dict[str, Path] = {}
        for axis_name, build in builds.items():
            vector_path = vectors_dir / f"{axis_name}.pt"
            torch.save(build["axis"], vector_path)
            vector_paths[axis_name] = vector_path
            if "default_mean" in build:
                torch.save(build["default_mean"], means_dir / f"{axis_name}__default_mean.pt")
                torch.save(build["contrast_mean"], means_dir / f"{axis_name}__contrast_mean.pt")
        reference_mean = (
            builds["native_step256"]["default_mean"] + builds["native_step256"]["contrast_mean"]
        ) / 2
        reference_mean_path = means_dir / "step256_probe_midpoint.pt"
        torch.save(reference_mean, reference_mean_path)
        completed_steps.append("wrote_target_vectors")

        evaluation_defaults, evaluation_contrasts = select_contrast_rows(
            responses, evaluation_ids, variant
        )
        evaluation_records = sorted(
            evaluation_defaults + evaluation_contrasts,
            key=lambda row: str(row["rollout_id"]),
        )
        if any(int(row["question_id"]) in construction_ids for row in evaluation_records):
            raise ValueError("evaluation records contain construction questions")
        write_jsonl(evaluation_path, evaluation_records)
        completed_steps.append("wrote_evaluation_records")

        construction_sorted = sorted(construction_ids)
        evaluation_sorted = sorted(evaluation_ids)
        target_rows = [
            target_metadata(
                "native_step256",
                expected_revisions["native_step256"],
                vector_paths["native_step256"],
                construction_sorted,
                builds["native_step256"]["raw_norm"],
            ),
            target_metadata(
                "endpoint_step512",
                expected_revisions["endpoint_step512"],
                vector_paths["endpoint_step512"],
                construction_sorted,
                builds["endpoint_step512"]["raw_norm"],
            ),
            target_metadata(
                "final_step143000",
                expected_revisions["final_step143000"],
                vector_paths["final_step143000"],
                construction_sorted,
                builds["final_step143000"]["raw_norm"],
            ),
            target_metadata(
                "innovation_256_to_512",
                "step256_to_step512",
                vector_paths["innovation_256_to_512"],
                construction_sorted,
                innovation_raw_norm,
            ),
        ]

        pairwise_cosines = {
            "native_step256__endpoint_step512": float(
                torch.dot(builds["native_step256"]["axis"], builds["endpoint_step512"]["axis"]).item()
            ),
            "native_step256__final_step143000": float(
                torch.dot(builds["native_step256"]["axis"], builds["final_step143000"]["axis"]).item()
            ),
            "endpoint_step512__final_step143000": float(
                torch.dot(builds["endpoint_step512"]["axis"], builds["final_step143000"]["axis"]).item()
            ),
            "innovation__native_step256": innovation_orthogonality,
            "innovation__endpoint_step512": float(
                torch.dot(innovation, builds["endpoint_step512"]["axis"]).item()
            ),
            "innovation__final_step143000": float(
                torch.dot(innovation, builds["final_step143000"]["axis"]).item()
            ),
        }
        if abs(innovation_orthogonality) > 1e-5:
            raise ValueError(f"innovation direction is not orthogonal to native axis: {innovation_orthogonality}")

        bundle = {
            "schema_version": "0.1",
            "bundle_id": f"concept-targets-step256-step512-layer{expected_layer}",
            "model_id": model_config["model_id"],
            "layer": expected_layer,
            "axis_variant_id": axis_variant_id,
            "construction_question_ids": construction_sorted,
            "evaluation_question_ids": evaluation_sorted,
            "question_category_coverage": category_coverage,
            "targets": target_rows,
            "reference_mean": {
                "reference_name": "step256_probe_midpoint",
                "vector_path": str(reference_mean_path),
                "definition": "mean(native_step256_default_mean, native_step256_contrast_mean)",
            },
            "evaluation_records_jsonl": str(evaluation_path),
            "counts": {
                "construction_default_records": builds["endpoint_step512"]["counts"]["default_records"],
                "construction_contrast_records": builds["endpoint_step512"]["counts"]["contrast_records"],
                "evaluation_default_records": len(evaluation_defaults),
                "evaluation_contrast_records": len(evaluation_contrasts),
                "evaluation_records": len(evaluation_records),
            },
            "pairwise_cosines": pairwise_cosines,
            "source": {
                "experiment_config": str(experiment_config_path),
                "axis_config": str(axis_config_path),
                "fixed_response_jsonl": str(response_jsonl),
                "activation_runs": activation_inputs,
            },
        }
        write_json(bundle_path, bundle)
        write_json(results_dir / "results.json", bundle)
        completed_steps.append("validated_bundle")

        counts = {"targets": len(target_rows), "evaluation_records": len(evaluation_records)}
        manifest = {
            "schema_version": "0.1",
            "builder": "ConceptTargetBundleBuilder",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "inputs": {
                "experiment_config": {
                    "path": str(experiment_config_path),
                    "sha256": file_sha256(experiment_config_path),
                },
                "axis_config": {"path": str(axis_config_path), "sha256": file_sha256(axis_config_path)},
                "fixed_response_jsonl": {"path": str(response_jsonl), "sha256": file_sha256(response_jsonl)},
                "activation_runs": activation_inputs,
            },
            "selection": {
                "construction_question_ids": construction_sorted,
                "evaluation_question_ids": evaluation_sorted,
                "default_prompt_ids": variant.get("default_prompt_ids", []),
                "contrast_role_groups": variant.get("contrast_role_groups", []),
            },
            "outputs": {
                "concept_target_bundle": str(bundle_path),
                "evaluation_records_jsonl": str(evaluation_path),
                "vectors": {name: str(path) for name, path in vector_paths.items()},
                "reference_mean": str(reference_mean_path),
            },
            "validation": {
                "passed": True,
                "innovation_native_absolute_dot": abs(innovation_orthogonality),
                "category_coverage": category_coverage,
            },
        }
        write_json(manifest_path, manifest)
        write_progress(progress_path, "completed", completed_steps)
        write_status(status_path, "completed", "concept target bundle build completed", counts)
        append_log(log_path, "completed", counts)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "bundle": str(bundle_path),
                    "evaluation_records": str(evaluation_path),
                    "targets": len(target_rows),
                    "evaluation_record_count": len(evaluation_records),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        write_progress(progress_path, "failed", completed_steps)
        write_status(
            status_path,
            "failed",
            f"concept target bundle build failed: {type(exc).__name__}: {exc}",
            counts,
        )
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
