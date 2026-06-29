#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from score_training_sequence_gradients import score_batch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states, use_cache
        hidden = self.embedding(input_ids)
        logits = self.projection(hidden)
        # Match Transformers semantics: index 0 is the embedding output and
        # index layer+1 is the requested block output.
        return SimpleNamespace(logits=logits, hidden_states=(hidden, hidden))


def fixture_samples(count: int, vocab_size: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        length = 7 + (index % 5)
        token_ids = [1 + ((index * 7 + position * 3) % (vocab_size - 1)) for position in range(length)]
        rows.append(
            {
                "sample_id": f"batch-invariance-{index:03d}",
                "window_id": "synthetic_step256_to_step512",
                "uid": f"fixture-{index:03d}",
                "batch_idx": 256 + index,
                "source_file": "synthetic_fixture",
                "token_count": len(token_ids),
                "token_ids": token_ids,
            }
        )
    return rows


def scorer_args(batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=batch_size,
        layer=0,
        max_input_tokens=64,
        model_id="synthetic-tiny-causal-lm",
        revision="deterministic-fixture-v0",
        save_gradient_vectors=False,
        save_token_axis_scores=False,
        torch_dtype="float32",
    )


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in rows}


def compare_rows(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    ref = by_id(reference)
    cand = by_id(candidate)
    if set(ref) != set(cand):
        raise ValueError("batch-size runs do not contain identical sample IDs")
    metrics: dict[str, list[float]] = {"loss": [], "update_pressure_norm": []}
    axis_names = sorted(reference[0]["axis_dot_scores"])
    for axis_name in axis_names:
        metrics[f"dot:{axis_name}"] = []
        metrics[f"cosine:{axis_name}"] = []
        metrics[f"token_dot_mean:{axis_name}"] = []
    for sample_id in sorted(ref):
        left = ref[sample_id]
        right = cand[sample_id]
        metrics["loss"].append(abs(float(left["loss"]) - float(right["loss"])))
        metrics["update_pressure_norm"].append(
            abs(float(left["update_pressure_norm"]) - float(right["update_pressure_norm"]))
        )
        for axis_name in axis_names:
            metrics[f"dot:{axis_name}"].append(
                abs(float(left["axis_dot_scores"][axis_name]) - float(right["axis_dot_scores"][axis_name]))
            )
            metrics[f"cosine:{axis_name}"].append(
                abs(float(left["axis_scores"][axis_name]) - float(right["axis_scores"][axis_name]))
            )
            metrics[f"token_dot_mean:{axis_name}"].append(
                abs(
                    float(left["token_axis_dot_diagnostics"][axis_name]["mean"])
                    - float(right["token_axis_dot_diagnostics"][axis_name]["mean"])
                )
            )
    summaries = {
        name: {
            "max_absolute_delta": max(values),
            "mean_absolute_delta": sum(values) / len(values),
            "passed": max(values) <= tolerance,
        }
        for name, values in metrics.items()
    }
    return {
        "shared_records": len(ref),
        "tolerance": tolerance,
        "metrics": summaries,
        "passed": all(summary["passed"] for summary in summaries.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate scorer invariance between float32 batch sizes 1 and N.")
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.candidate_batch_size < 2 or args.records < args.candidate_batch_size:
        raise SystemExit("candidate batch size must be >=2 and <= record count")
    if args.tolerance <= 0:
        raise SystemExit("tolerance must be positive")

    run_id = args.run_id or default_run_id()
    run_dir = (
        args.output_root
        / "assistant_axis_attribution"
        / "pythia-410m-deduped"
        / "pile-deduped-pythia-preshuffled"
        / "concept-attribution-256-512-v0"
        / "activation-gradient-batch-invariance-codepath"
        / run_id
    )
    results_dir = run_dir / "results"
    meta_dir = run_dir / "meta"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for directory in (results_dir, meta_dir, checkpoints_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = TinyCausalLM(vocab_size=47, hidden_size=16).float().eval()
    tokenizer = TinyTokenizer()
    samples = fixture_samples(args.records, vocab_size=47)
    axes = {
        "endpoint_step512": torch.randn(16, dtype=torch.float32),
        "final_step143000": torch.randn(16, dtype=torch.float32),
        "innovation_256_to_512": torch.randn(16, dtype=torch.float32),
    }
    axes = {name: vector / torch.linalg.vector_norm(vector) for name, vector in axes.items()}

    reference: list[dict[str, Any]] = []
    for sample in samples:
        reference.extend(
            score_batch(
                [sample], scorer_args(1), model, tokenizer, axes,
                "endpoint_step512", "final_step143000", run_dir, results_dir,
            )
        )
    candidate: list[dict[str, Any]] = []
    for start in range(0, len(samples), args.candidate_batch_size):
        batch = samples[start : start + args.candidate_batch_size]
        candidate.extend(
            score_batch(
                batch, scorer_args(args.candidate_batch_size), model, tokenizer, axes,
                "endpoint_step512", "final_step143000", run_dir, results_dir,
            )
        )

    comparison = compare_rows(reference, candidate, args.tolerance)
    comparison.update(
        {
            "schema_version": "0.1",
            "gate_scope": "deterministic_synthetic_code_path",
            "torch_dtype": "float32",
            "reference_batch_size": 1,
            "candidate_batch_size": args.candidate_batch_size,
            "seed": args.seed,
            "full_pythia_gate_status": "pending_required_inputs",
            "required_inputs": [
                "construction-split concept target bundle",
                "shared real packed-sequence sample",
                "EleutherAI/pythia-410m-deduped@step256 model files",
            ],
        }
    )
    write_jsonl(results_dir / "batch_size_1_scores.jsonl", reference)
    write_jsonl(results_dir / f"batch_size_{args.candidate_batch_size}_scores.jsonl", candidate)
    write_json(results_dir / "results.json", comparison)
    write_json(
        meta_dir / "run_manifest.json",
        {
            "schema_version": "0.1",
            "runner": "ActivationGradientBatchInvarianceValidator",
            "created_at_utc": utc_now(),
            "run_dir": str(run_dir),
            "gate_scope": comparison["gate_scope"],
            "seed": args.seed,
            "records": args.records,
            "reference_batch_size": 1,
            "candidate_batch_size": args.candidate_batch_size,
            "torch_dtype": "float32",
            "tolerance": args.tolerance,
        },
    )
    state = "completed" if comparison["passed"] else "failed"
    write_json(meta_dir / "status.json", {"schema_version": "0.1", "state": state, "updated_at_utc": utc_now()})
    write_json(
        checkpoints_dir / "progress.json",
        {"schema_version": "0.1", "state": state, "completed_records": args.records, "total_records": args.records},
    )
    (logs_dir / "run.log").write_text(
        json.dumps({"time_utc": utc_now(), "event": state, "passed": comparison["passed"]}) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": state, "passed": comparison["passed"], "run_dir": str(run_dir)}, indent=2))
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
