#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


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


def load_existing_result_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record["rollout_id"]) for record in load_jsonl(path) if "rollout_id" in record}


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
        / args.variant
        / run_id
    )


def selection_key(record: dict[str, Any]) -> str:
    if record.get("record_type") == "default":
        return "default"
    return f"role:{record.get('role_group')}"


def select_rollouts(records: list[dict[str, Any]], limit: int | None, sample_mode: str) -> list[dict[str, Any]]:
    if limit is None:
        return records
    if limit < 1:
        raise ValueError("--limit must be positive when provided")
    if sample_mode == "first":
        return records[:limit]
    if sample_mode != "stratified":
        raise ValueError(f"unsupported sample mode: {sample_mode}")
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(selection_key(record), []).append(record)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < limit and keys:
        next_keys: list[str] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def template_fixture_response(rollout: dict[str, Any]) -> str:
    question = str(rollout.get("question", "the question")).strip()
    return (
        "This is a deterministic template fixture response for validating the fixed-response "
        "generation pipeline. It briefly addresses the prompt without serving as a final "
        f"scientific response: {question}"
    )


def clean_generated_response(text: str, strip_lists: bool, trim_to_sentence: bool) -> str:
    cleaned = text.strip()
    if strip_lists:
        cleaned = re.split(r"\n\s*(?:[-*]|\d+[.)])\s+", cleaned, maxsplit=1)[0].strip()
        if cleaned.endswith(":"):
            cleaned = cleaned[:-1].rstrip() + "."
    if trim_to_sentence:
        sentence_end = max(cleaned.rfind("."), cleaned.rfind("?"), cleaned.rfind("!"))
        if sentence_end >= 0:
            cleaned = cleaned[: sentence_end + 1].strip()
    return cleaned


def torch_dtype_from_name(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def build_hf_prompt(
    tokenizer: Any,
    prompt_text: str,
    use_chat_template: bool,
    system_prompt: str,
    user_suffix: str,
) -> str:
    user_text = prompt_text
    if user_suffix.strip():
        user_text = f"{prompt_text.rstrip()}\n\n{user_suffix.strip()}"
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if system_prompt.strip():
        return f"{system_prompt.strip()}\n\n{user_text}"
    return user_text


def load_hf_local_generator(args: argparse.Namespace) -> Callable[[dict[str, Any]], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_id,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_id,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device_map,
    )
    model.eval()

    def generate(rollout: dict[str, Any]) -> str:
        prompt_text = build_hf_prompt(
            tokenizer,
            rollout["prompt_text"],
            args.use_chat_template,
            args.system_prompt,
            args.user_suffix,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)
        completion_ids = output_ids[0, input_width:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    return generate


def build_response_record(
    rollout: dict[str, Any],
    args: argparse.Namespace,
    generator: Callable[[dict[str, Any]], str] | None,
) -> dict[str, Any]:
    if args.provider == "template_fixture":
        generated_response = template_fixture_response(rollout)
        response_source = {
            "mode": "template_fixture",
            "model": "deterministic_template_fixture",
        }
        generation_config = {
            "provider": args.provider,
            "intended_use": "smoke_test_only",
        }
    elif args.provider == "hf_local" and generator is not None:
        raw_generated_response = generator(rollout)
        generated_response = clean_generated_response(
            raw_generated_response,
            strip_lists=args.strip_lists,
            trim_to_sentence=args.trim_to_sentence,
        )
        response_source = {
            "mode": "hf_local",
            "model": args.hf_model_id,
        }
        generation_config = {
            "provider": args.provider,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_chat_template": args.use_chat_template,
            "system_prompt": args.system_prompt,
            "user_suffix": args.user_suffix,
            "strip_lists": args.strip_lists,
            "trim_to_sentence": args.trim_to_sentence,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "local_files_only": args.local_files_only,
        }
    else:
        raise NotImplementedError(f"unsupported provider: {args.provider}")
    if not generated_response.strip():
        raise ValueError(f"empty generated response for rollout_id={rollout['rollout_id']}")
    return {
        "response_id": f"response__{rollout['rollout_id']}",
        "rollout_id": rollout["rollout_id"],
        "record_type": rollout.get("record_type"),
        "role_id": rollout.get("role_id"),
        "role_group": rollout.get("role_group"),
        "default_prompt_id": rollout.get("default_prompt_id"),
        "question_id": rollout.get("question_id"),
        "question_category": rollout.get("question_category"),
        "prompt_text": rollout["prompt_text"],
        "generated_response": generated_response,
        "response_source": response_source,
        "generation_config": generation_config,
    }


def write_progress(path: Path, selected_ids: list[str], completed_ids: set[str], cursor: int) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "updated_at_utc": utc_now(),
            "cursor": cursor,
            "selected_count": len(selected_ids),
            "completed_count": len(completed_ids),
            "remaining_count": len(set(selected_ids) - completed_ids),
            "completed_rollout_ids": sorted(completed_ids),
        },
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


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    rollout_jsonl: Path,
    result_jsonl: Path,
    selected_count: int,
    completed_count: int,
) -> None:
    write_json(
        path,
        {
            "schema_version": "0.1",
            "runner": "FixedResponseGenerator",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "provider": args.provider,
            "provider_note": (
                "template_fixture is for smoke tests only; hf_local is intended for fixed response generation "
                "when the configured local/Hugging Face model is available."
            ),
            "hf_generation": {
                "hf_model_id": args.hf_model_id,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "use_chat_template": args.use_chat_template,
                "system_prompt": args.system_prompt,
                "user_suffix": args.user_suffix,
                "strip_lists": args.strip_lists,
                "trim_to_sentence": args.trim_to_sentence,
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
                "local_files_only": args.local_files_only,
                "hf_cache_dir": str(args.hf_cache_dir) if args.hf_cache_dir else None,
            },
            "rollout_jsonl": {
                "path": str(rollout_jsonl),
                "sha256": file_sha256(rollout_jsonl),
            },
            "results": {
                "raw_generated_responses_jsonl": {
                    "path": str(result_jsonl),
                    "sha256": file_sha256(result_jsonl),
                }
            },
            "selection": {
                "limit": args.limit,
                "sample_mode": args.sample_mode,
                "selected_count": selected_count,
                "completed_count": completed_count,
            },
            "artifact_paths": {
                "inputs": str(run_dir / "inputs"),
                "checkpoints": str(run_dir / "checkpoints"),
                "results": str(run_dir / "results"),
                "logs": str(run_dir / "logs"),
                "meta": str(run_dir / "meta"),
            },
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate raw fixed responses for rollout prompts.")
    parser.add_argument("--rollout-jsonl", type=Path, default=Path("data/rollouts/assistant_axis_rollouts_v0.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--experiment-name", default="assistant_axis_attribution")
    parser.add_argument("--model-name", default="fixed-response-generator")
    parser.add_argument("--dataset-name", default="fixed-aa-rollouts-v0")
    parser.add_argument("--probe-set", default="assistant-axis-rollouts-v0")
    parser.add_argument("--variant", default="template-fixture")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--provider", choices=["template_fixture", "hf_local"], default="template_fixture")
    parser.add_argument("--allow-template-fixture", action="store_true")
    parser.add_argument("--hf-model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--system-prompt",
        default=(
            "Answer the user's question directly in one short paragraph of 2 to 3 complete sentences. "
            "Do not use bullet points or numbered lists."
        ),
    )
    parser.add_argument(
        "--user-suffix",
        default="Answer format: one short paragraph, 2 to 3 complete sentences, no bullet points, no numbered list.",
    )
    parser.add_argument("--strip-lists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trim-to-sentence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["first", "stratified"], default="first")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--force-completed", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.provider == "template_fixture" and not args.allow_template_fixture:
        raise SystemExit("template_fixture requires --allow-template-fixture")
    if args.save_every < 1:
        raise SystemExit("--save-every must be positive")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    run_dir = resolve_run_dir(args)
    inputs_dir = run_dir / "inputs"
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    meta_dir = run_dir / "meta"
    result_jsonl = results_dir / "generated_responses_raw.jsonl"
    progress_path = checkpoints_dir / "progress.json"
    status_path = meta_dir / "status.json"
    manifest_path = meta_dir / "run_manifest.json"
    log_path = logs_dir / "run.log"

    for directory in [inputs_dir, checkpoints_dir, results_dir, logs_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if status_path.exists() and not args.force_completed:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            print(json.dumps({"status": "skipped_completed", "run_dir": str(run_dir)}, indent=2))
            return 0

    rollouts = select_rollouts(load_jsonl(args.rollout_jsonl), args.limit, args.sample_mode)
    selected_ids = [str(record["rollout_id"]) for record in rollouts]
    existing_ids = load_existing_result_ids(result_jsonl)
    completed_ids = existing_ids.intersection(selected_ids)

    write_status(
        status_path,
        "running",
        "fixed response generation started",
        {"selected": len(selected_ids), "completed": len(completed_ids)},
    )
    append_log(log_path, "start", {"run_dir": str(run_dir), "selected": len(selected_ids)})
    write_json(inputs_dir / "selected_rollout_ids.json", {"rollout_ids": selected_ids})

    cursor = 0
    try:
        generator = load_hf_local_generator(args) if args.provider == "hf_local" else None
        for cursor, rollout in enumerate(rollouts, start=1):
            rollout_id = str(rollout["rollout_id"])
            if rollout_id in completed_ids:
                continue
            response_record = build_response_record(rollout, args, generator)
            append_jsonl(result_jsonl, response_record)
            completed_ids.add(rollout_id)
            if len(completed_ids) % args.save_every == 0:
                write_progress(progress_path, selected_ids, completed_ids, cursor)
                append_log(log_path, "progress", {"cursor": cursor, "completed": len(completed_ids)})
        final_state = "completed" if len(completed_ids) == len(selected_ids) else "failed"
        final_message = (
            "fixed response generation completed" if final_state == "completed" else "missing generated responses"
        )
    except Exception as exc:
        final_state = "failed"
        final_message = f"fixed response generation failed: {type(exc).__name__}: {exc}"
        append_log(log_path, "error", {"error_type": type(exc).__name__, "message": str(exc)})

    write_progress(progress_path, selected_ids, completed_ids, cursor)
    write_status(
        status_path,
        final_state,
        final_message,
        {"selected": len(selected_ids), "completed": len(completed_ids)},
    )
    write_manifest(
        manifest_path,
        args,
        run_dir,
        args.rollout_jsonl,
        result_jsonl,
        selected_count=len(selected_ids),
        completed_count=len(completed_ids),
    )
    append_log(log_path, final_state, {"completed": len(completed_ids), "selected": len(selected_ids)})

    print(
        json.dumps(
            {
                "status": final_state,
                "message": final_message,
                "run_dir": str(run_dir),
                "raw_generated_responses_jsonl": str(result_jsonl),
                "selected": len(selected_ids),
                "completed": len(completed_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_state == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
