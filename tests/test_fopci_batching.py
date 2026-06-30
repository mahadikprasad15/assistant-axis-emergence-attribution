from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analysis" / "score_first_order_concept_influence.py"
SPEC = importlib.util.spec_from_file_location("fopci", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
FOPCI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FOPCI)


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        values = [1] + [2 + (ord(character) % 11) for character in text]
        return values[:max_length] if max_length is not None else values

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
    ) -> dict[str, object]:
        if isinstance(text, str):
            return {"input_ids": self.encode(text, max_length if truncation else None)}
        rows = [self.encode(value, max_length if truncation else None) for value in text]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ToyConceptModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(4)
        self.embedding = torch.nn.Embedding(16, 5)
        self.projection = torch.nn.Linear(5, 5, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        embedded = self.embedding(input_ids)
        hidden = torch.tanh(self.projection(embedded))
        return SimpleNamespace(hidden_states=(embedded, hidden))


class ToyLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(8)
        self.embedding = torch.nn.Embedding(17, 6)
        self.output = torch.nn.Linear(6, 17, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(logits=self.output(torch.tanh(self.embedding(input_ids))))


def test_batched_query_gradient_matches_single_record_accumulation() -> None:
    model = ToyConceptModel()
    tokenizer = ToyTokenizer()
    records = [
        {
            "rollout_id": f"default-{index}",
            "record_type": "default",
            "prompt_text": f"prompt {index}",
            "generated_response": "assistant response",
        }
        for index in range(2)
    ] + [
        {
            "rollout_id": f"role-{index}",
            "record_type": "role",
            "prompt_text": f"different prompt {index}",
            "generated_response": "role response",
        }
        for index in range(4)
    ]
    targets = {
        "target_a": torch.tensor([1.0, -0.5, 0.2, 0.1, -0.3]),
        "target_b": torch.tensor([-0.2, 0.7, 0.4, -0.6, 0.1]),
    }
    parameters = [model.projection.weight]
    reference, _ = FOPCI.build_query_gradients(
        records,
        model,
        tokenizer,
        parameters,
        0,
        targets,
        "\n",
        128,
        batch_size=1,
    )
    candidate, summary = FOPCI.build_query_gradients(
        records,
        model,
        tokenizer,
        parameters,
        0,
        targets,
        "\n",
        128,
        batch_size=3,
    )
    assert summary["query_batch_size"] == 3
    for target_name in targets:
        torch.testing.assert_close(
            candidate[target_name][0],
            reference[target_name][0],
            atol=2e-6,
            rtol=2e-6,
        )


def test_directional_batch_matches_sequential_raw_fopci_dot() -> None:
    model = ToyLanguageModel()
    parameter_names = ["output.weight"]
    parameters = [model.output.weight]
    torch.manual_seed(12)
    query_gradients = {
        "target_a": [torch.randn_like(model.output.weight)],
        "target_b": [torch.randn_like(model.output.weight)],
    }
    query_norms = {
        name: math.sqrt(sum(float(torch.sum(part.double() ** 2)) for part in parts))
        for name, parts in query_gradients.items()
    }
    samples = [
        {
            "sample_id": f"sample-{index}",
            "window_id": "step256_to_step512",
            "uid": str(index),
            "batch_idx": 256 + index,
            "token_ids": [2 + index, 4, 6, 8, 10, 12][: 4 + index],
            "source_file": "toy.parquet",
        }
        for index in range(2)
    ]
    scope = {
        "parameter_scope_id": "layer12_only",
        "parameter_count": model.output.weight.numel(),
        "parameter_tensor_count": 1,
        "scope_hash": "toy-scope",
    }
    args = argparse.Namespace(
        max_input_tokens=32,
        revision="step256",
        torch_dtype="float32",
    )
    reference = [
        FOPCI.score_sequence(
            sample,
            model,
            parameters,
            query_gradients,
            query_norms,
            scope,
            args,
        )
        for sample in samples
    ]
    candidate = FOPCI.score_sequence_batch_directional(
        samples,
        model,
        parameter_names,
        parameters,
        query_gradients,
        query_norms,
        scope,
        args,
    )
    for expected, actual in zip(reference, candidate):
        assert actual["sample_id"] == expected["sample_id"]
        assert actual["sequence_gradient_norm"] is None
        assert actual["scoring_mode"] == "directional_jvp"
        assert abs(actual["loss"] - expected["loss"]) < 1e-6
        for target_name in query_gradients:
            expected_dot = expected["axis_scores"][target_name]["negative_gradient_dot"]
            actual_dot = actual["axis_scores"][target_name]["negative_gradient_dot"]
            assert abs(actual_dot - expected_dot) < 2e-5
            assert actual["axis_scores"][target_name]["gradient_cosine"] is None


if __name__ == "__main__":
    test_batched_query_gradient_matches_single_record_accumulation()
    test_directional_batch_matches_sequential_raw_fopci_dot()
    print("FOPCI batching tests passed")
