#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_rollouts_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rollouts = load_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for record in rollouts:
        rollout_id = str(record["rollout_id"])
        if rollout_id in by_id:
            raise ValueError(f"duplicate rollout_id in rollout corpus: {rollout_id}")
        by_id[rollout_id] = record
    return by_id


def normalize_text(text: Any) -> str:
    return str(text).strip()


def normalize_response_record(record: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    generated_response = normalize_text(record.get("generated_response", ""))
    prompt_text = normalize_text(record.get("prompt_text", rollout["prompt_text"]))
    response_source = record.get("response_source", {})
    if not isinstance(response_source, dict):
        response_source = {"mode": "unknown", "model": str(response_source)}
    response_id = str(record.get("response_id") or f"response__{record['rollout_id']}")
    words = generated_response.split()
    return {
        "response_id": response_id,
        "rollout_id": str(record["rollout_id"]),
        "record_type": rollout["record_type"],
        "role_id": rollout.get("role_id"),
        "role_group": rollout.get("role_group"),
        "default_prompt_id": rollout.get("default_prompt_id"),
        "question_id": rollout["question_id"],
        "question_category": rollout["question_category"],
        "prompt_text": prompt_text,
        "generated_response": generated_response,
        "response_char_count": len(generated_response),
        "response_word_count": len(words),
        "response_source": {
            "mode": str(response_source.get("mode", "unknown")),
            "model": str(response_source.get("model", "unknown")),
        },
        "validation_tags": ["fixed_response_imported"],
    }


def validate_import(
    raw_records: list[dict[str, Any]],
    output_records: list[dict[str, Any]],
    rollouts_by_id: dict[str, dict[str, Any]],
    require_full_corpus: bool,
    allow_prompt_mismatch: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    raw_ids = [str(record.get("rollout_id", "")) for record in raw_records]
    duplicate_ids = sorted({rollout_id for rollout_id in raw_ids if raw_ids.count(rollout_id) > 1})
    if duplicate_ids:
        errors.append(f"duplicate response rollout_ids: {duplicate_ids}")

    missing_ids = sorted(rollout_id for rollout_id in raw_ids if rollout_id not in rollouts_by_id)
    if missing_ids:
        errors.append(f"response records reference unknown rollout_ids: {missing_ids}")

    empty_ids = sorted(
        record["rollout_id"] for record in output_records if not record["generated_response"].strip()
    )
    if empty_ids:
        errors.append(f"empty generated_response for rollout_ids: {empty_ids}")

    unknown_source_ids = sorted(
        record["rollout_id"]
        for record in output_records
        if record["response_source"]["mode"] == "unknown" or record["response_source"]["model"] == "unknown"
    )
    if unknown_source_ids:
        errors.append(f"unknown response_source mode/model for rollout_ids: {unknown_source_ids}")

    if not allow_prompt_mismatch:
        mismatched = []
        for record in output_records:
            rollout_prompt = normalize_text(rollouts_by_id[record["rollout_id"]]["prompt_text"])
            if record["prompt_text"] != rollout_prompt:
                mismatched.append(record["rollout_id"])
        if mismatched:
            errors.append(f"prompt_text mismatch for rollout_ids: {sorted(mismatched)}")

    if require_full_corpus:
        missing_from_response_file = sorted(set(rollouts_by_id) - set(raw_ids))
        if missing_from_response_file:
            errors.append(f"full corpus import missing {len(missing_from_response_file)} rollout_ids")
    else:
        warnings.append("fixture/subset import: not all rollout_ids are covered")

    return errors, warnings


def build_manifest(
    rollout_jsonl: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    rollouts_by_id: dict[str, dict[str, Any]],
    raw_records: list[dict[str, Any]],
    output_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    mode: str,
) -> dict[str, Any]:
    empty_count = sum(1 for record in output_records if not record["generated_response"].strip())
    raw_ids = [str(record.get("rollout_id", "")) for record in raw_records]
    duplicate_count = len({rollout_id for rollout_id in raw_ids if raw_ids.count(rollout_id) > 1})
    missing_count = sum(1 for rollout_id in raw_ids if rollout_id not in rollouts_by_id)
    return {
        "schema_version": "0.1",
        "importer": "FixedResponseImporter",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "rollout_jsonl": {
            "path": str(rollout_jsonl),
            "sha256": file_sha256(rollout_jsonl),
        },
        "input_responses_jsonl": {
            "path": str(input_jsonl),
            "sha256": file_sha256(input_jsonl),
        },
        "output_responses_jsonl": {
            "path": str(output_jsonl),
            "sha256": file_sha256(output_jsonl),
        },
        "counts": {
            "rollout_records": len(rollouts_by_id),
            "input_response_records": len(raw_records),
            "output_response_records": len(output_records),
            "empty_responses": empty_count,
            "missing_rollout_ids": missing_count,
            "duplicate_response_ids": duplicate_count,
        },
        "validation": {
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and normalize fixed generated responses.")
    parser.add_argument("--rollout-jsonl", type=Path, default=Path("data/rollouts/assistant_axis_rollouts_v0.jsonl"))
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["fixture", "full"], default="fixture")
    parser.add_argument("--allow-prompt-mismatch", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rollouts_by_id = load_rollouts_by_id(args.rollout_jsonl)
    raw_records = load_jsonl(args.input_jsonl)

    output_records: list[dict[str, Any]] = []
    pre_errors: list[str] = []
    for record in raw_records:
        rollout_id = str(record.get("rollout_id", ""))
        if rollout_id not in rollouts_by_id:
            continue
        try:
            output_records.append(normalize_response_record(record, rollouts_by_id[rollout_id]))
        except KeyError as exc:
            pre_errors.append(f"missing required field {exc} in response record for rollout_id={rollout_id}")

    errors, warnings = validate_import(
        raw_records=raw_records,
        output_records=output_records,
        rollouts_by_id=rollouts_by_id,
        require_full_corpus=args.mode == "full",
        allow_prompt_mismatch=args.allow_prompt_mismatch,
    )
    errors = pre_errors + errors

    if not errors:
        write_jsonl(args.output_jsonl, output_records)
    else:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("", encoding="utf-8")

    manifest = build_manifest(
        rollout_jsonl=args.rollout_jsonl,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        rollouts_by_id=rollouts_by_id,
        raw_records=raw_records,
        output_records=output_records,
        errors=errors,
        warnings=warnings,
        mode=args.mode,
    )
    write_json(args.output_manifest, manifest)

    summary = {
        "status": "completed" if not errors else "failed",
        "mode": args.mode,
        "input_response_records": len(raw_records),
        "output_response_records": len(output_records) if not errors else 0,
        "errors": len(errors),
        "warnings": len(warnings),
        "output_jsonl": str(args.output_jsonl),
        "output_manifest": str(args.output_manifest),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
