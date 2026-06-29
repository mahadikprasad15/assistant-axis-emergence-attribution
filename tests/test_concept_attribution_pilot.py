from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import yaml


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analysis" / "run_concept_attribution_pilot.py"
SPEC = importlib.util.spec_from_file_location("pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_preflight_accepts_shared_valid_inputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "config.yaml"
        sample_path = root / "sample.jsonl"
        bundle_path = root / "concept_target_bundle.json"
        evaluation_path = root / "evaluation_records.jsonl"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "training_window": {"window_id": "step256_to_step512"},
                    "axis_targets": {
                        "primary": ["endpoint_step512", "final_step143000", "innovation_256_to_512"]
                    },
                    "model": {"scoring_revision": "step256"},
                }
            ),
            encoding="utf-8",
        )
        rows = [
            {
                "sample_id": f"sample-{index}",
                "window_id": "step256_to_step512",
                "uid": f"uid-{index}",
                "batch_idx": 256 + index,
                "token_ids": [1, 2, 3],
            }
            for index in range(2)
        ]
        sample_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(evaluation_path, {"question_id": 7})
        write_json(
            bundle_path,
            {
                "construction_question_ids": [0],
                "evaluation_question_ids": [7],
                "evaluation_records_jsonl": str(evaluation_path),
                "targets": [
                    {"axis_name": "endpoint_step512"},
                    {"axis_name": "final_step143000"},
                    {"axis_name": "innovation_256_to_512"},
                ],
            },
        )
        result = PILOT.validate_inputs(sample_path, bundle_path, config_path, pilot_size=2)
        assert result["pilot_size"] == 2
        assert result["sample_ids"] == ["sample-0", "sample-1"]
        assert result["checkpoint_revision"] == "step256"


if __name__ == "__main__":
    test_preflight_accepts_shared_valid_inputs()
    print("Concept-attribution pilot preflight tests passed")
