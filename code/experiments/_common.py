"""Shared helpers for experiment scripts (makes the repo importable without pip install)."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def environment_info(device: torch.device) -> dict:
    from spsa_nn.device import describe

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": describe(device),
        "platform": platform.platform(),
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"saved {path}")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
