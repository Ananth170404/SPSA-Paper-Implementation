"""Download MNIST into data/mnist/ (or $SPSA_DATA_DIR/mnist) and build mnist.npz.

    python scripts/download_mnist.py            # skips if already present
    python scripts/download_mnist.py --force    # re-download
    python scripts/download_mnist.py --dir D:/datasets/mnist
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spsa_nn.datasets import download_mnist, load_mnist  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dir", type=Path, default=None)
p.add_argument("--force", action="store_true")
args = p.parse_args()

path = download_mnist(args.dir, force=args.force)
train, test = load_mnist(args.dir)
print(f"OK: {path}\n  train {tuple(train.x.shape)}  test {tuple(test.x.shape)}")
