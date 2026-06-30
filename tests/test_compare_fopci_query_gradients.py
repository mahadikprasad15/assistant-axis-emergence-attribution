from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analysis" / "compare_fopci_query_gradients.py"
SPEC = importlib.util.spec_from_file_location("compare_query", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def test_query_comparator_writes_passing_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference = root / "reference.pt"
        candidate = root / "candidate.pt"
        output = root / "artifacts"
        payload = {
            "parameter_names": ["layer.weight"],
            "scope": {"scope_hash": "scope"},
            "gradients": {"target": [torch.tensor([[1.0, -2.0], [0.5, 3.0]])]},
        }
        torch.save(payload, reference)
        candidate_payload = dict(payload)
        candidate_payload["gradients"] = {
            "target": [payload["gradients"]["target"][0] + 1e-8]
        }
        torch.save(candidate_payload, candidate)
        previous = sys.argv
        try:
            sys.argv = [
                "compare_fopci_query_gradients.py",
                "--reference-bundle",
                str(reference),
                "--candidate-bundle",
                str(candidate),
                "--output-root",
                str(output),
                "--run-id",
                "query-comparison-v0",
            ]
            assert COMPARE.main() == 0
        finally:
            sys.argv = previous
        summary_path = next(output.rglob("query_gradient_agreement.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["passed"] is True
        assert summary["axes"][0]["cosine"] > 0.999999


if __name__ == "__main__":
    test_query_comparator_writes_passing_artifact()
    print("FOPCI query comparison tests passed")
