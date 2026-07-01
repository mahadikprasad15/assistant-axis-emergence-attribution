from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/data/finalize_adaptive_fopci_subset.py"
AXES = ["endpoint_step512", "final_step143000", "innovation_256_to_512"]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_adaptive_finalizer_builds_disjoint_exact_subset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subset, vector, activation = root / "subset", root / "vector", root / "activation"
        samples, vector_rows, activation_rows = [], [], []
        for index in range(80):
            sample_id = f"sample-{index:03d}"
            samples.append({"sample_id": sample_id, "window_id": "w", "token_ids": [index, index + 1]})
            vector_rows.append({"sample_id": sample_id, "axis_scores": {axis: {"mean_raw_projection": index + axis_index / 10} for axis_index, axis in enumerate(AXES)}})
            activation_rows.append({"sample_id": sample_id, "axis_metrics": {axis: {"dot": (79 - index) + axis_index / 10} for axis_index, axis in enumerate(AXES)}})
        write_jsonl(subset / "results/activation_gradient_sequences.jsonl", samples)
        write_jsonl(subset / "results/fopci_random_sequences.jsonl", samples[:5])
        write_jsonl(vector / "results/vector_filter_scores.jsonl", vector_rows)
        write_jsonl(activation / "results/attribution_scores.jsonl", activation_rows)
        config = {
            "sampling": {"seed": 1729, "fopci_random_size": 5, "fopci_adaptive_size": 12, "fopci_sample_size": 17,
                         "adaptive_strata": {"concordant_high": 2, "concordant_low": 2,
                            "vector_filter_high_activation_gradient_low": 2, "activation_gradient_high_vector_filter_low": 2,
                            "endpoint_final_target_disagreement": 2, "near_zero_control": 2}},
            "axis_targets": {"primary": AXES},
        }
        config_path = root / "config.yaml"
        import yaml
        config_path.write_text(yaml.safe_dump(config))
        output = root / "artifacts"
        completed = subprocess.run([
            sys.executable, str(SCRIPT), "--subset-run-dir", str(subset), "--vector-run-dir", str(vector),
            "--activation-run-dir", str(activation), "--experiment-config", str(config_path),
            "--output-root", str(output), "--run-id", "adaptive-v0",
        ], check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr + completed.stdout
        run = next(output.rglob("adaptive-v0"))
        adaptive = [json.loads(line) for line in (run / "results/adaptive_sequences.jsonl").read_text().splitlines()]
        combined = [json.loads(line) for line in (run / "results/combined_fopci_sequences.jsonl").read_text().splitlines()]
        summary = json.loads((run / "results/results.json").read_text())
        assert len(adaptive) == 12 and len(combined) == 17
        assert not ({row["sample_id"] for row in adaptive} & {row["sample_id"] for row in samples[:5]})
        assert summary["counts"]["adaptive_strata"] == config["sampling"]["adaptive_strata"]
        assert (run / "meta/status.json").exists()


if __name__ == "__main__":
    test_adaptive_finalizer_builds_disjoint_exact_subset()
    print("Adaptive FOPCI subset finalizer tests passed")
