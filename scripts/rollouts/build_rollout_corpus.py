#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def variant_id(index: int) -> str:
    return f"iv{index:02d}"


def render_template(template: str, **values: str) -> str:
    normalized_values = {key: normalize_text(value) for key, value in values.items()}
    return template.format(**normalized_values).strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def validate_config(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    roles = config.get("roles", {})
    role_groups = config.get("role_groups", {})
    questions = config.get("selected_questions", {}).get("selected", [])
    defaults = config.get("default_prompts", [])
    expected = config.get("validation_expectations", {})

    if len(roles) != int(expected.get("role_count", 48)):
        errors.append(f"expected {expected.get('role_count', 48)} roles, found {len(roles)}")
    if len(questions) != int(expected.get("question_count", 20)):
        errors.append(f"expected {expected.get('question_count', 20)} questions, found {len(questions)}")
    if len(defaults) != int(expected.get("default_prompt_count", 4)):
        errors.append(f"expected {expected.get('default_prompt_count', 4)} default prompts, found {len(defaults)}")

    grouped_roles: list[str] = []
    for group_id, group in role_groups.items():
        group_roles = list(group.get("roles", []))
        grouped_roles.extend(group_roles)
        if len(group_roles) != 16:
            errors.append(f"role group {group_id!r} expected 16 roles, found {len(group_roles)}")
        for role_id in group_roles:
            if role_id not in roles:
                errors.append(f"role group {group_id!r} references missing role {role_id!r}")

    extra_roles = sorted(set(roles) - set(grouped_roles))
    if extra_roles:
        errors.append(f"roles are defined but not assigned to a group: {extra_roles}")

    question_ids = [int(question["id"]) for question in questions]
    if len(question_ids) != len(set(question_ids)):
        errors.append("selected question ids must be unique")

    required_categories = set(expected.get("required_question_categories", []))
    found_categories = {str(question["category"]) for question in questions}
    missing_categories = sorted(required_categories - found_categories)
    if missing_categories:
        errors.append(f"missing required question categories: {missing_categories}")

    for role_id, role in roles.items():
        variants = role.get("instruction_variants", [])
        if not variants:
            errors.append(f"role {role_id!r} has no instruction variants")
        if role.get("source_status") == "planned_upstream_import":
            warnings.append(f"role {role_id!r} uses a planned_upstream_import placeholder instruction")

    return errors, warnings


def build_role_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    questions = config["selected_questions"]["selected"]
    roles = config["roles"]
    role_groups = config["role_groups"]
    template = config["construction"]["role_record_template"]
    readout_policy = config["construction"]["readout_policy"]
    first_n = int(config["construction"]["role_instruction_variants"]["first_n"])

    role_to_group = {
        role_id: group_id
        for group_id, group in role_groups.items()
        for role_id in group["roles"]
    }

    records: list[dict[str, Any]] = []
    for role_id in role_to_group:
        role = roles[role_id]
        selected_variants = role["instruction_variants"][:first_n]
        for variant_index, instruction in enumerate(selected_variants, start=1):
            role_variant_id = variant_id(variant_index)
            role_instruction = normalize_text(instruction)
            for question in questions:
                question_id = int(question["id"])
                rollout_id = f"role__{role_id}__q{question_id:03d}__{role_variant_id}"
                prompt_text = render_template(
                    template,
                    role_instruction=role_instruction,
                    question=str(question["question"]),
                )
                records.append(
                    {
                        "rollout_id": rollout_id,
                        "record_type": "role",
                        "role_id": role_id,
                        "role_group": role_to_group[role_id],
                        "role_instruction_variant_id": role_variant_id,
                        "role_instruction": role_instruction,
                        "role_source_status": role["source_status"],
                        "question_id": question_id,
                        "question_category": str(question["category"]),
                        "question": normalize_text(question["question"]),
                        "prompt_text": prompt_text,
                        "readout_policy": readout_policy,
                        "source": {
                            "rollout_config_id": config["rollout_config_id"],
                            "source_material_config": config["source_material"]["config"],
                            "corpus_type": config["construction"]["corpus_type"],
                            "include_trait_instructions": False,
                            "include_old_instruction_conditions": False,
                            "role_source_path": role.get("source_path"),
                        },
                    }
                )
    return records


def build_default_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    questions = config["selected_questions"]["selected"]
    defaults = config["default_prompts"]
    readout_policy = config["construction"]["readout_policy"]
    default_template = config["construction"]["default_record_template"]
    bare_template = config["construction"]["bare_question_template"]

    records: list[dict[str, Any]] = []
    for default in defaults:
        default_id = str(default["default_prompt_id"])
        default_prompt = normalize_text(default.get("prompt", ""))
        template = bare_template if default_id == "bare_question" else default_template
        for question in questions:
            question_id = int(question["id"])
            rollout_id = f"default__{default_id}__q{question_id:03d}"
            prompt_text = render_template(
                template,
                default_prompt=default_prompt,
                question=str(question["question"]),
            )
            records.append(
                {
                    "rollout_id": rollout_id,
                    "record_type": "default",
                    "default_prompt_id": default_id,
                    "default_prompt": default_prompt,
                    "question_id": question_id,
                    "question_category": str(question["category"]),
                    "question": normalize_text(question["question"]),
                    "prompt_text": prompt_text,
                    "readout_policy": readout_policy,
                    "source": {
                        "rollout_config_id": config["rollout_config_id"],
                        "source_material_config": config["source_material"]["config"],
                        "corpus_type": config["construction"]["corpus_type"],
                        "include_trait_instructions": False,
                        "include_old_instruction_conditions": False,
                    },
                }
            )
    return records


def validate_records(config: dict[str, Any], records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    expected = config["validation_expectations"]

    rollout_ids = [record["rollout_id"] for record in records]
    if len(rollout_ids) != len(set(rollout_ids)):
        errors.append("rollout_id values must be unique")

    role_records = [record for record in records if record["record_type"] == "role"]
    default_records = [record for record in records if record["record_type"] == "default"]

    if len(role_records) != int(expected["expected_role_records"]):
        errors.append(f"expected {expected['expected_role_records']} role records, found {len(role_records)}")
    if len(default_records) != int(expected["expected_default_records"]):
        errors.append(f"expected {expected['expected_default_records']} default records, found {len(default_records)}")
    if len(records) != int(expected["expected_total_records"]):
        errors.append(f"expected {expected['expected_total_records']} total records, found {len(records)}")

    forbidden = {"trait_axis_id", "condition", "polarity", "instruction_positive", "instruction_negative", "instruction_neutral"}
    for record in records:
        overlap = sorted(forbidden.intersection(record))
        if overlap:
            errors.append(f"record {record['rollout_id']} contains forbidden fields {overlap}")

    placeholder_count = sum(
        1 for record in role_records if record.get("role_source_status") == "planned_upstream_import"
    )
    if placeholder_count:
        warnings.append(f"{placeholder_count} role records use planned_upstream_import role instructions")

    return errors, warnings


def build_manifest(
    config_path: Path,
    source_material_path: Path,
    output_jsonl: Path,
    records: list[dict[str, Any]],
    validation_errors: list[str],
    validation_warnings: list[str],
) -> dict[str, Any]:
    role_records = [record for record in records if record["record_type"] == "role"]
    default_records = [record for record in records if record["record_type"] == "default"]
    return {
        "schema_version": "0.1",
        "builder": "RolloutCorpusBuilder",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rollout_config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "source_material_config": {
            "path": str(source_material_path),
            "sha256": file_sha256(source_material_path),
        },
        "output_jsonl": str(output_jsonl),
        "output_jsonl_sha256": file_sha256(output_jsonl),
        "target_counts": {
            "role_records": 960,
            "default_records": 80,
            "records_total": 1040,
        },
        "actual_counts": {
            "role_records": len(role_records),
            "default_records": len(default_records),
            "records_total": len(records),
            "role_groups": count_by(role_records, "role_group"),
            "question_categories": count_by(records, "question_category"),
            "role_source_status": count_by(role_records, "role_source_status"),
            "default_prompt_id": count_by(default_records, "default_prompt_id"),
        },
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
            "warnings": validation_warnings,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed default-vs-role rollout corpus.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rollouts/assistant_axis_roles_v0.yaml"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/rollouts/assistant_axis_rollouts_v0.jsonl"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/rollouts/assistant_axis_rollouts_v0_manifest.json"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    source_material_path = Path(config["source_material"]["config"])
    load_yaml(source_material_path)

    config_errors, config_warnings = validate_config(config)
    records = build_role_records(config) + build_default_records(config)
    record_errors, record_warnings = validate_records(config, records)
    errors = config_errors + record_errors
    warnings = config_warnings + record_warnings

    if errors:
        summary = {"status": "failed", "errors": errors, "warnings": warnings}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    write_jsonl(args.output_jsonl, records)
    manifest = build_manifest(
        config_path=args.config,
        source_material_path=source_material_path,
        output_jsonl=args.output_jsonl,
        records=records,
        validation_errors=errors,
        validation_warnings=warnings,
    )
    write_json(args.output_manifest, manifest)

    summary = {
        "status": "completed",
        "records_total": len(records),
        "role_records": manifest["actual_counts"]["role_records"],
        "default_records": manifest["actual_counts"]["default_records"],
        "warnings": len(warnings),
        "output_jsonl": str(args.output_jsonl),
        "output_manifest": str(args.output_manifest),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
