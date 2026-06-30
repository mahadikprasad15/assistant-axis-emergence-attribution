from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analysis" / "analyze_three_method_attribution.py"
SPEC = importlib.util.spec_from_file_location("three_method_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_runs(root: Path) -> tuple[Path, Path, Path]:
    vector_run, activation_run, fopci_run = root / "vector", root / "activation", root / "fopci"
    vector_rows, activation_rows, fopci_rows = [], [], []
    for index in range(6):
        sample_id = f"sample-{index}"
        score = float(index - 2)
        vector_rows.append({
            "sample_id": sample_id, "mean_hidden_norm": 2.0,
            "axis_scores": {"axis": {"mean_raw_projection": score + 1.0}},
            "centered_axis_scores": {"axis": {"mean_centered_projection": score}},
        })
        activation_rows.append({
            "sample_id": sample_id, "loss": 1.0 + index, "update_pressure_norm": 0.5 + index,
            "axis_metrics": {"axis": {"dot": score * 2.0, "cosine": score / 4.0}},
        })
        fopci_rows.append({
            "sample_id": sample_id, "loss": 1.0 + index, "sequence_gradient_norm": 1.5 + index,
            "axis_scores": {"axis": {"negative_gradient_dot": score * 3.0, "gradient_cosine": score / 4.0}},
        })
    write_jsonl(vector_run / "results/vector_filter_scores.jsonl", vector_rows)
    write_jsonl(activation_run / "results/attribution_scores.jsonl", activation_rows)
    write_jsonl(fopci_run / "results/fopci_scores.jsonl", fopci_rows)
    return vector_run, activation_run, fopci_run


def test_normalized_analysis_writes_durable_results() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        vector, activation, fopci = make_runs(root)
        output = root / "artifacts"
        previous = sys.argv
        try:
            sys.argv = [
                "analyze_three_method_attribution.py",
                "--vector-run-dir", str(vector),
                "--activation-run-dir", str(activation),
                "--fopci-run-dir", str(fopci),
                "--top-k", "2",
                "--bootstrap-samples", "50",
                "--output-root", str(output),
                "--run-id", "normalized-v0",
            ]
            assert ANALYSIS.main() == 0
        finally:
            sys.argv = previous
        summary_path = next(output.rglob("results/results.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = summary["axes"]["axis"]["normalized_comparisons"][
            "activation_cosine__vs__fopci_cosine"
        ]
        assert summary["records"] == 6
        assert result["spearman"] == 1.0
        assert result["top_overlap"]["count"] == 2
        assert (summary_path.parents[1] / "meta/status.json").exists()
        assert (summary_path.parent / "normalized_scores.csv").exists()


def test_rank_ties_use_average_rank() -> None:
    assert ANALYSIS.ranks([10.0, 10.0, 20.0]) == [1.5, 1.5, 3.0]


if __name__ == "__main__":
    test_normalized_analysis_writes_durable_results()
    test_rank_ties_use_average_rank()
    print("Three-method normalized analysis tests passed")
