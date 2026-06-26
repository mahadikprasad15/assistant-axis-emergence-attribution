#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any


def check_import(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "ok": False,
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "ok": True,
        "module": module_name,
        "version": getattr(module, "__version__", None),
    }


def check_torch_tensor() -> dict[str, Any]:
    try:
        import torch

        tensor = torch.tensor([1.0, 2.0])
        return {
            "ok": True,
            "tensor_sum": float(tensor.sum().item()),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local model runtime imports and torch basics.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON only.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    checks = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "imports": {
            "torch": check_import("torch"),
            "transformers": check_import("transformers"),
            "accelerate": check_import("accelerate"),
        },
        "torch_tensor": check_torch_tensor(),
    }
    ok = all(item["ok"] for item in checks["imports"].values()) and checks["torch_tensor"]["ok"]
    payload = {"ok": ok, "checks": checks}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
