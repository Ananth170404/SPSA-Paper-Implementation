"""Device selection (GPU if available, otherwise CPU) and seeding helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def get_device(name: str = "auto") -> torch.device:
    """Resolve 'auto' | 'cuda' | 'cuda:N' | 'mps' | 'cpu' to a torch.device."""
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. Install a CUDA "
            "build of PyTorch (see README) or run with --device cpu."
        )
    return torch.device(name)


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return str(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    """A torch.Generator living on `device` (CUDA generators avoid host syncs)."""
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    g = torch.Generator(device=gen_device)
    g.manual_seed(seed)
    return g


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
