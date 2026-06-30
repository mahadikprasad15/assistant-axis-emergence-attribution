from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analysis" / "compare_fopci_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_fopci", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def write_scores(run_dir: Path, adjustment: float) -> None:
    results = run_dir / "results"
    results.mkdir(parents=True)
    rows = []
    for index, value in enumerate([-0.5, 0.25, 1.0]):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "axis_scores": {
                    "target": {"negative_gradient_dot": value + adjustment}
                },
            }
        )
    (results / "fopci_scores.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_comparator_writes_passing_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference = root / "reference"
        candidate = root / "candidate"
        output = root / "artifacts"
        write_scores(reference, 0.0)
        write_scores(candidate, 1e-8)
        previous = sys.argv
        try:
            sys.argv = [
                "compare_fopci_runs.py",
                "--reference-run-dir",
                str(reference),
                "--candidate-run-dir",
                str(candidate),
                "--output-root",
                str(output),
                "--run-id",
                "comparison-v0",
            ]
            assert COMPARE.main() == 0
        finally:
            sys.argv = previous
        summary_path = next(output.rglob("fopci_score_agreement_summary.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["passed"] is True
        assert summary["shared_records"] == 3
        assert summary["axes"][0]["sign_agreement"] == 1.0


if __name__ == "__main__":
    test_comparator_writes_passing_artifact()
    print("FOPCI comparison tests passed")
