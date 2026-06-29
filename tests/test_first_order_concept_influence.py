from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analysis" / "score_first_order_concept_influence.py"
SPEC = importlib.util.spec_from_file_location("fopci", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
FOPCI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FOPCI)


class TinyLayeredModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = torch.nn.ModuleList(
            [torch.nn.Linear(3, 3, bias=False) for _ in range(4)]
        )
        self.readout = torch.nn.Linear(3, 1, bias=False)


def test_parameter_scope_selects_requested_layer() -> None:
    model = TinyLayeredModel()
    names, parameters, summary = FOPCI.resolve_parameter_scope(
        model, "layer12_only", layer=2, every_nth_layer=2
    )
    assert names == ["gpt_neox.layers.2.weight"]
    assert parameters == [model.gpt_neox.layers[2].weight]
    assert summary["parameter_count"] == 9
    assert len(summary["scope_hash"]) == 64


def test_streamed_negative_dot_matches_flattened_dot() -> None:
    torch.manual_seed(1729)
    model = TinyLayeredModel().float()
    _, parameters, _ = FOPCI.resolve_parameter_scope(
        model, "all_parameters", layer=2, every_nth_layer=2
    )
    x_query = torch.randn(2, 3)
    x_sequence = torch.randn(2, 3)

    def scalar_for(x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in model.gpt_neox.layers:
            hidden = torch.tanh(layer(hidden))
        return model.readout(hidden).mean()

    query = FOPCI.gradients_for_scalar(scalar_for(x_query), parameters, retain_graph=False)
    sequence = FOPCI.gradients_for_scalar(scalar_for(x_sequence), parameters, retain_graph=False)
    streamed = -sum(float(torch.sum(left * right).item()) for left, right in zip(sequence, query))
    flattened = -float(
        torch.dot(
            torch.cat([value.reshape(-1) for value in sequence]),
            torch.cat([value.reshape(-1) for value in query]),
        ).item()
    )
    assert abs(streamed - flattened) < 1e-7


if __name__ == "__main__":
    test_parameter_scope_selects_requested_layer()
    test_streamed_negative_dot_matches_flattened_dot()
    print("FOPCI helper tests passed")
