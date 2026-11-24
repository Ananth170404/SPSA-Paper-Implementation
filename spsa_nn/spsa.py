"""Algorithm 1 of the paper: plain SPSA for minimising f : R^m -> R.

A direct, NumPy-only transcription of

    for t = 1..t_max:
        a_t = a / t**alpha
        c_t = c / t**gamma
        delta_i ~ U{-1, +1}
        x+ = x_t + c_t * delta
        x- = x_t - c_t * delta
        g_i = (f(x+) - f(x-)) / (2 c_t delta_i)
        x_{t+1} = x_t - a_t * g

The neural-network trainer in :mod:`spsa_nn.layerwise` uses exactly this update
rule, implemented in PyTorch (GPU capable) and vectorised over parameter blocks
and over many independent trials at once.
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional

import numpy as np


def gains(t: int, a: float, alpha: float, c: float, gamma: float) -> tuple[float, float]:
    """Gain sequences a_t = a / t^alpha and c_t = c / t^gamma (t starts at 1)."""
    return a / t**alpha, c / t**gamma


def rademacher(shape, rng: np.random.Generator) -> np.ndarray:
    """Perturbation vector with i.i.d. entries drawn uniformly from {-1, +1}."""
    return rng.integers(0, 2, size=shape).astype(np.float64) * 2.0 - 1.0


def iterate(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    a: float = 0.1,
    alpha: float = 1.0,
    c: float = 0.1,
    gamma: float = 1.0 / 6.0,
    t_max: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Iterator[np.ndarray]:
    """Yield x_{t+1} after every SPSA iteration t = 1..t_max."""
    rng = np.random.default_rng() if rng is None else rng
    x = np.array(x0, dtype=np.float64, copy=True)
    for t in range(1, t_max + 1):
        a_t, c_t = gains(t, a, alpha, c, gamma)
        delta = rademacher(x.shape, rng)
        f_plus = f(x + c_t * delta)
        f_minus = f(x - c_t * delta)
        g_hat = (f_plus - f_minus) / (2.0 * c_t * delta)
        x = x - a_t * g_hat
        yield x


def minimize(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    a: float = 0.1,
    alpha: float = 1.0,
    c: float = 0.1,
    gamma: float = 1.0 / 6.0,
    t_max: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Run Algorithm 1 and return x_{t_max + 1}."""
    x = np.array(x0, dtype=np.float64, copy=True)
    for x in iterate(f, x0, a, alpha, c, gamma, t_max, rng):
        pass
    return x
