"""Data sets used in the paper.

Didactic 2-D problems (Fig. 1) -- generated on the fly, deterministic per seed:
    xor      4 Gaussian blobs, 2 classes,  labels y in {-1, +1}
    moons    two interleaving half circles, 2 classes, y in {-1, +1}
    circles  three concentric rings, 3 classes, y in {-1, +1}^3 (+1 for the true class)

MNIST (Fig. 2) -- the original IDX files, downloaded once to ``data/mnist/``
(or ``$SPSA_DATA_DIR/mnist``) and cached as ``mnist.npz``. Images are scaled to
[0, 1] and flattened to 784 values; labels are one-hot in {0, 1}^10.

If you cannot download from this machine, put the four files
    train-images-idx3-ubyte(.gz)  train-labels-idx1-ubyte(.gz)
    t10k-images-idx3-ubyte(.gz)   t10k-labels-idx1-ubyte(.gz)
into data/mnist/ by hand (a torchvision ``MNIST/raw`` folder works too).
"""
from __future__ import annotations

import gzip
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .paths import mnist_dir


@dataclass
class Dataset:
    x: torch.Tensor        # (n, d) float32
    y: torch.Tensor        # (n, m) float32 targets
    labels: torch.Tensor   # (n,) int64 class index
    n_classes: int


def _two_class(x: np.ndarray, labels: np.ndarray) -> Dataset:
    y = np.where(labels == 1, 1.0, -1.0)[:, None]
    return Dataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32),
                   torch.tensor(labels, dtype=torch.long), 2)


def make_xor(n: int = 200, noise: float = 0.25, seed: int = 0) -> Dataset:
    """Four Gaussian blobs at (+-1, +-1); class 1 where x1 * x2 > 0."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]], dtype=float)
    blob = rng.integers(0, 4, size=n) if n % 4 else np.repeat(np.arange(4), n // 4)
    x = centers[blob] + noise * rng.standard_normal((n, 2))
    labels = (blob < 2).astype(int)
    return _two_class(x, labels)


def make_moons(n: int = 200, noise: float = 0.1, seed: int = 0) -> Dataset:
    """Same construction as sklearn.datasets.make_moons, re-centred to ~[-1, 1]^2."""
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    outer = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)
    inner = np.stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5], axis=1)
    x = np.concatenate([outer, inner]) + noise * rng.standard_normal((n, 2))
    x = (x - np.array([0.5, 0.25])) / np.array([1.0, 0.75])
    labels = np.concatenate([np.zeros(n_out, int), np.ones(n_in, int)])
    return _two_class(x, labels)


def make_circles(n: int = 120, noise: float = 0.05, seed: int = 0,
                 radii=(0.3, 0.65, 1.0)) -> Dataset:
    """Three concentric rings; ring k is class k. Targets in {-1, +1}^3."""
    rng = np.random.default_rng(seed)
    k = len(radii)
    labels = np.repeat(np.arange(k), int(np.ceil(n / k)))[:n]
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.asarray(radii)[labels]
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1) + noise * rng.standard_normal((n, 2))
    y = -np.ones((n, k))
    y[np.arange(n), labels] = 1.0
    return Dataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32),
                   torch.tensor(labels, dtype=torch.long), k)


DIDACTIC = {"xor": make_xor, "moons": make_moons, "circles": make_circles}


def load_didactic(name: str, **kwargs) -> Dataset:
    if name not in DIDACTIC:
        raise ValueError(f"unknown data set {name!r}; choose from {list(DIDACTIC)}")
    return DIDACTIC[name](**kwargs)


# MNIST -------------------------------------------------------------------------

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}
MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)


def _find_local(directory: Path, stem: str) -> Optional[Path]:
    for folder in (directory, directory / "raw", directory / "MNIST" / "raw"):
        for name in (stem, stem + ".gz", stem.replace("-idx", ".idx")):
            if (folder / name).exists():
                return folder / name
    return None


def _download(stem: str, directory: Path) -> Path:
    target = directory / (stem + ".gz")
    errors = []
    for mirror in MNIST_MIRRORS:
        url = mirror + stem + ".gz"
        try:
            print(f"  downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
                shutil.copyfileobj(r, f)
            return target
        except Exception as exc:  # try the next mirror
            errors.append(f"{url}: {exc}")
            target.unlink(missing_ok=True)
    raise RuntimeError("could not download MNIST file " + stem + ":\n  " + "\n  ".join(errors) +
                       f"\nPlace the file manually into {directory} (see README).")


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def download_mnist(directory: Optional[Path] = None, force: bool = False) -> Path:
    """Make sure data/mnist/mnist.npz exists; returns its path."""
    directory = Path(directory) if directory else mnist_dir()
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "mnist.npz"
    if cache.exists() and not force:
        return cache
    arrays = {}
    for key, stem in MNIST_FILES.items():
        path = _find_local(directory, stem)
        if path is None or force:
            path = _download(stem, directory)
        arrays[key] = _read_idx(path)
    assert arrays["train_images"].shape == (60000, 28, 28), arrays["train_images"].shape
    assert arrays["test_images"].shape == (10000, 28, 28), arrays["test_images"].shape
    np.savez_compressed(cache, **arrays)
    print(f"  MNIST cached at {cache}")
    return cache


def load_mnist(directory: Optional[Path] = None, n_train: Optional[int] = None,
               n_test: Optional[int] = None) -> tuple[Dataset, Dataset]:
    """Returns (train, test). Optionally use only the first n_train / n_test samples."""
    cache = download_mnist(directory)
    with np.load(cache) as d:
        arrays = {k: d[k] for k in d.files}

    def build(images: np.ndarray, labels: np.ndarray, n: Optional[int]) -> Dataset:
        if n is not None:
            images, labels = images[:n], labels[:n]
        x = torch.tensor(images.reshape(len(images), -1), dtype=torch.float32) / 255.0
        lab = torch.tensor(labels.astype(np.int64))
        y = torch.nn.functional.one_hot(lab, 10).float()
        return Dataset(x, y, lab, 10)

    return (build(arrays["train_images"], arrays["train_labels"], n_train),
            build(arrays["test_images"], arrays["test_labels"], n_test))
