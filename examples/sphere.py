"""Algorithm 1 on its own: minimise a 10-dimensional sphere function.

    python examples/sphere.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spsa_nn import spsa  # noqa: E402


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


x0 = np.full(10, 3.0)
rng = np.random.default_rng(0)
for t, x in enumerate(spsa.iterate(sphere, x0, a=0.2, alpha=0.602, c=0.1, gamma=0.101,
                                   t_max=2000, rng=rng), start=1):
    if t in (1, 10, 100, 500, 1000, 2000):
        print(f"t = {t:5d}   f(x_t) = {sphere(x):.6f}")
