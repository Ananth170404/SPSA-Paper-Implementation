"""Central place for filesystem locations.

Everything is resolved relative to the repository root so that scripts work no
matter which directory they are launched from. Both locations can be overridden
with environment variables:

    SPSA_DATA_DIR     where datasets (MNIST) are stored / downloaded to
    SPSA_RESULTS_DIR  where experiment outputs (json, png) are written
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    path = Path(os.environ.get("SPSA_DATA_DIR", REPO_ROOT / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    path = Path(os.environ.get("SPSA_RESULTS_DIR", REPO_ROOT / "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def mnist_dir() -> Path:
    path = data_dir() / "mnist"
    path.mkdir(parents=True, exist_ok=True)
    return path
