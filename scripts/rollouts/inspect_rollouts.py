#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
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


def count_by(records: list[dict[str, Any]], key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(key)
        if value is not None:
            counts[str(value)] += 1
    return counts


def print_table(title: str, counts: Counter[str] | dict[str, int]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, count in sorted(counts.items()):
        print(f"{key}: {count}")


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rollout_id",
        "record_type",
        "role_group",
        "role_id",
        "role_source_status",
        "default_prompt_id",
        "question_id",
        "question_category",
        "prompt_text",
    ]
    return {key: record[key] for key in keys if key in record}


def print_records(title: str, records: list[dict[str, Any]], limit: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not records:
        print("(none)")
        return
    for record in records[:limit]:
        print(json.dumps(compact_record(record), indent=2, ensure_ascii=False, sort_keys=True))


def print_manifest_summary(manifest: dict[str, Any]) -> None:
    print("Manifest")
    print("--------")
    print(f"builder: {manifest.get('builder')}")
    print(f"validation_passed: {manifest.get('validation', {}).get('passed')}")
    print(f"output_jsonl: {manifest.get('output_jsonl')}")
    print(f"records_total: {manifest.get('actual_counts', {}).get('records_total')}")
    warnings = manifest.get("validation", {}).get("warnings", [])
    print(f"warnings: {len(warnings)}")
    if warnings:
        print(f"first_warning: {warnings[0]}")


def print_question_slice(records: list[dict[str, Any]], question_id: int, limit: int) -> None:
    selected = [record for record in records if int(record.get("question_id", -1)) == question_id]
    print_records(f"Question Slice q{question_id:03d}", selected, limit)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect fixed rollout corpus counts and samples.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/rollouts/assistant_axis_rollouts_v0.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rollouts/assistant_axis_rollouts_v0_manifest.json"),
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--question-id", type=int, default=0)
    parser.add_argument("--role-group", type=str, default="non_assistant_non_neutral")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = load_jsonl(args.jsonl)
    manifest = load_json(args.manifest)

    role_records = [record for record in records if record.get("record_type") == "role"]
    default_records = [record for record in records if record.get("record_type") == "default"]
    group_records = [record for record in role_records if record.get("role_group") == args.role_group]

    print_manifest_summary(manifest)
    print_table("Record Types", count_by(records, "record_type"))
    print_table("Role Groups", count_by(role_records, "role_group"))
    print_table("Question Categories", count_by(records, "question_category"))
    print_table("Default Prompt IDs", count_by(default_records, "default_prompt_id"))
    print_table("Role Source Status", count_by(role_records, "role_source_status"))

    print_records("Sample Role Records", role_records, args.samples)
    print_records("Sample Default Records", default_records, args.samples)
    print_records(f"Sample Role Group: {args.role_group}", group_records, args.samples)
    print_question_slice(records, args.question_id, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
