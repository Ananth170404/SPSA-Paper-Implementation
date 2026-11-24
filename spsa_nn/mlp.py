"""Feed-forward tanh network of the paper (Eqs. 5-7), batched over trials.

All weights carry a leading "trial" dimension K, so that K independently
initialised networks (e.g. the 100 trials averaged in Fig. 1) are evaluated in
one batched forward pass on the GPU/CPU. K = 1 is a single ordinary network.

Parameters are stored as a flat list ``[W1, b1, W2, b2, ..., WL, bL]`` with

    W_l : (K, n_out, n_in)      b_l : (K, n_out)

and every layer computes  a^l = tanh(beta * (W_l a^{l-1} + b_l))  (Eq. 6).
The output layer uses the same tanh, as in the paper.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch

Params = List[torch.Tensor]


def init_params(
    sizes: Sequence[int],
    n_trials: int = 1,
    init: str = "glorot",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Params:
    """Randomly initialise weights; biases start at zero.

    init:
      'glorot'  W ~ U(-r, r), r = sqrt(6 / (n_in + n_out))   (TF/Keras default)
      'normal'  W ~ N(0, 1 / n_in)
      'standard' W ~ N(0, 1), b ~ N(0, 1)   (tf.random_normal default)
    """
    params: Params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        shape = (n_trials, n_out, n_in)
        if init == "glorot":
            r = math.sqrt(6.0 / (n_in + n_out))
            W = (torch.rand(shape, generator=generator, device=device, dtype=dtype) * 2 - 1) * r
        elif init == "normal":
            W = torch.randn(shape, generator=generator, device=device, dtype=dtype) / math.sqrt(n_in)
        elif init == "standard":
            W = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown init {init!r}")
        if init == "standard":
            b = torch.randn((n_trials, n_out), generator=generator, device=device, dtype=dtype)
        else:
            b = torch.zeros((n_trials, n_out), device=device, dtype=dtype)
        params += [W, b]
    return params


def n_layers(params: Params) -> int:
    return len(params) // 2


def count_parameters(sizes: Sequence[int]) -> int:
    """Weights + biases of a single network (the paper's parameter count)."""
    return sum(n_in * n_out + n_out for n_in, n_out in zip(sizes[:-1], sizes[1:]))


def layer_forward(a_prev: torch.Tensor, W: torch.Tensor, b: torch.Tensor, beta: float) -> torch.Tensor:
    """a_prev: (N, n_in) shared input or (K, N, n_in). Returns (K, N, n_out)."""
    z = torch.matmul(a_prev, W.transpose(-1, -2)) + b.unsqueeze(-2)
    return torch.tanh(beta * z)


def forward(
    x: torch.Tensor,
    params: Params,
    beta: float,
    start_layer: int = 0,
) -> torch.Tensor:
    """Network output y(x | W) (Eq. 7).

    If ``start_layer = l > 0`` then ``x`` must already be the activation a^{l}
    (0-based layer index), which lets the trainer cache everything below the
    layer that is currently being optimised.
    """
    a = x
    for l in range(start_layer, n_layers(params)):
        a = layer_forward(a, params[2 * l], params[2 * l + 1], beta)
    if a.dim() == 2:  # no layers applied
        a = a.unsqueeze(0)
    return a


def activations_upto(x: torch.Tensor, params: Params, beta: float, layer: int) -> torch.Tensor:
    """Input to 0-based layer ``layer`` (i.e. a^{layer}); returns x itself for layer 0."""
    a = x
    for l in range(layer):
        a = layer_forward(a, params[2 * l], params[2 * l + 1], beta)
    return a


def error(y_hat: torch.Tensor, y: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """Sum of squared residuals E(D, W) of Eq. (9), one value per trial -> (K,).

    reduction='sum'  exactly Eq. (9)  (used for the didactic problems / Fig. 1)
    reduction='mean' Eq. (9) divided by n  (the scale shown for MNIST in Fig. 2)
    reduction='elementwise_mean'  Eq. (9) divided by n*m (TF's reduce_mean of the
                     squared residual matrix; the SPSA objective for MNIST)
    """
    per_sample = (y_hat - y).pow(2).sum(dim=-1)  # (K, N)
    if reduction == "sum":
        return per_sample.sum(dim=-1)
    if reduction == "mean":
        return per_sample.mean(dim=-1)
    if reduction == "elementwise_mean":  # tf.reduce_mean(tf.square(y_hat - y))
        return per_sample.mean(dim=-1) / y.shape[-1]
    raise ValueError(reduction)


@torch.no_grad()
def evaluate_error(
    x: torch.Tensor, y: torch.Tensor, params: Params, beta: float,
    reduction: str = "sum", chunk: int = 20000,
) -> torch.Tensor:
    """E(D, W) per trial, computed in chunks to bound memory."""
    total = None
    n = x.shape[0]
    for i in range(0, n, chunk):
        y_hat = forward(x[i:i + chunk], params, beta)
        e = error(y_hat, y[i:i + chunk], "sum")
        total = e if total is None else total + e
    if reduction == "mean":
        return total / n
    if reduction == "elementwise_mean":
        return total / (n * y.shape[-1])
    return total


@torch.no_grad()
def predict_classes(x: torch.Tensor, params: Params, beta: float, chunk: int = 20000) -> torch.Tensor:
    """Predicted class per trial and sample -> (K, N).

    Single output unit: class = (y > 0). Several outputs: argmax.
    """
    out = []
    for i in range(0, x.shape[0], chunk):
        y_hat = forward(x[i:i + chunk], params, beta)
        if y_hat.shape[-1] == 1:
            out.append((y_hat[..., 0] > 0).long())
        else:
            out.append(y_hat.argmax(dim=-1))
    return torch.cat(out, dim=1)


@torch.no_grad()
def accuracy(x: torch.Tensor, labels: torch.Tensor, params: Params, beta: float) -> torch.Tensor:
    """Classification accuracy per trial -> (K,). ``labels`` are integer classes."""
    pred = predict_classes(x, params, beta)
    return (pred == labels.unsqueeze(0)).float().mean(dim=1)
